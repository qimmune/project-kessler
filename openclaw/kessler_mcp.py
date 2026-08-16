#!/usr/bin/env python
"""MCP server exposing the Project Kessler physics engine to an OpenClaw agent.

This is the tool layer. The skills under openclaw/skills/ are the guidance layer
that teaches the agent when to reach for each of these.

Design rule, and the reason the whole thing is trustworthy: the model never
decides whether a maneuver is safe. `simulate_maneuver` flies the burn through
the real propagator and returns the four checks. `issue_burn` refuses to write a
command the engine has not cleared. The agent reasons about *which* burn to try;
the engine decides whether it is allowed.

State persists to openclaw/state/engagements.json so an agent keeps its history
across sessions.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
try:                                    # SDK >= the mcpserver rename
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                      # older SDKs
    from mcp.server.fastmcp import FastMCP as _Server

from kessler.assurance import Engagement, cross_check
from kessler.catalog import classify, load_demo_catalog
from kessler.mission import (Constraints, MissionState, altitude_shortlist,
                             evaluate_maneuver, find_tca, requires_action,
                             synthesize_threat)
from kessler.monitor import select_fleet, sweep_fleet
from kessler.physics import elements, teme_state, timescale

STATE_DIR = ROOT / "openclaw" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = STATE_DIR / "engagements.json"

mcp = _Server(
    "project-kessler",
    instructions=("Orbital collision avoidance. Screen the catalogue, open an "
                  "engagement on a conjunction, propose a burn, and verify it "
                  "with simulate_maneuver before forming any view on whether it "
                  "is safe. issue_burn refuses anything the engine has not "
                  "cleared."))

_ctx: dict = {}


def _catalog():
    if "cat" not in _ctx:
        _ctx["cat"] = load_demo_catalog(limit=None)
    return _ctx["cat"]


def _load_log() -> list:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text())
        except json.JSONDecodeError:
            return []
    return []


def _append_log(entry: dict) -> None:
    log = _load_log()
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2, default=str))


@mcp.tool()
def screen_fleet(name_filter: str = "STARLINK", assets: int = 10,
                 horizon_hours: float = 6.0, threshold_km: float = 25.0) -> dict:
    """Screen protected assets against the whole tracked catalogue.

    Propagates every catalogued object once and screens each asset against that
    one resident state matrix. Returns per-asset closest approach and whether it
    breaches the separation minimum.
    """
    cat = _catalog()
    ts = timescale()
    t0 = ts.now()
    fleet = select_fleet(cat.objects, name_filter, assets)
    if not fleet:
        return {"error": f"no assets matching {name_filter!r}"}
    res = sweep_fleet(fleet, cat.objects, t0, horizon_s=horizon_hours * 3600,
                      coarse_step_s=60.0, threshold_km=threshold_km)
    _ctx["t0"] = t0
    return {
        "catalog_size": res.catalog_size,
        "states_propagated": res.states,
        "state_matrix_mb": round(res.matrix_mb, 1),
        "sweep_seconds": round(res.elapsed_s, 2),
        "backend": res.backend,
        "host_to_device_transfer": res.transfer,
        "assets": [{
            "asset": s.name, "status": s.severity, "altitude_km": round(s.alt_km, 1),
            "closest_object": s.worst.secondary if s.worst else None,
            "miss_km": round(s.worst.miss_km, 3) if s.worst else None,
            "pc": s.worst.pc if s.worst else None,
            "tca_minutes": round(s.worst.tca_offset_s / 60, 1) if s.worst else None,
            "action_required": s.action_required, "assessment": s.reason,
        } for s in res.statuses],
        "action_required_count": len(res.actionable),
    }


@mcp.tool()
def open_conjunction(asset: str, tca_minutes: float = 92.0,
                     miss_km: float = 0.412) -> dict:
    """Open an engagement on a conjunction and return its measured geometry.

    Seeds a deterministic encounter against `asset` for demonstration. The
    geometry returned is measured by the engine, not asserted -- the detector
    finds it the same way it finds a natural one.
    """
    cat = _catalog()
    ts = timescale()
    t0 = ts.now()
    hero = cat.by_name(asset)
    if hero is None:
        return {"error": f"no asset named {asset!r}"}

    r0, v0 = teme_state(hero, t0)
    el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, tca_minutes * 60, miss_km=miss_km)
    enc = find_tca(r0, v0, tr, tv, horizon_s=tca_minutes * 60 * 1.3)
    cc = cross_check(hero, t0, enc["tca_offset_s"])
    c = Constraints()
    act, why = requires_action(enc, c)

    state = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                         nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2,
                         constraints=c)
    eng = Engagement(primary=hero.name, secondary=tname, seeded=True,
                     geometry={k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in enc.items()},
                     consensus=cc.as_dict(), action_required=act, assessment=why)
    _ctx["state"] = state
    _ctx["engagement"] = eng
    _ctx["shortlist"] = altitude_shortlist(cat.objects, el, 50.0)

    return {
        "engagement_id": eng.engagement_id,
        "primary": hero.name, "secondary": tname,
        "tca_offset_s": round(enc["tca_offset_s"], 1),
        "tca_minutes": round(enc["tca_offset_s"] / 60, 1),
        "miss_km": round(enc["miss_km"], 4),
        "relative_speed_kms": round(enc["rel_speed_kms"], 3),
        "probability_of_collision": enc["pc"],
        "radial_km": round(enc["radial_km"], 3),
        "in_track_km": round(enc["in_track_km"], 3),
        "cross_track_km": round(enc["cross_track_km"], 3),
        "propagator_consensus_m": round(cc.residual_km * 1000, 1),
        "action_required": act, "assessment": why,
        "constraints": {"separation_minimum_km": c.min_miss_km,
                        "delta_v_budget_mps": c.dv_budget_mps,
                        "altitude_box_km": c.altitude_box_km,
                        "pc_action_threshold": c.max_pc},
    }


@mcp.tool()
def simulate_maneuver(direction_ric: list[float], delta_v_mps: float,
                      burn_offset_s: float) -> dict:
    """Fly a proposed burn through the physics engine and report what happens.

    This is the check that decides. It returns the new miss distance and
    probability of collision, whether the delta-v is within budget, whether the
    mission altitude box holds, and whether the new orbit creates a fresh
    conjunction anywhere in the catalogue over the next 12 hours.

    Call this before forming any opinion about whether a burn is safe.
    """
    if "state" not in _ctx:
        return {"error": "no open engagement -- call open_conjunction first"}
    state = _ctx["state"]
    eng = _ctx["engagement"]
    res = evaluate_maneuver(state, _ctx["shortlist"], direction_ric,
                            float(delta_v_mps), float(burn_offset_s),
                            float(eng.geometry["tca_offset_s"]))
    _ctx["last_result"] = res
    _ctx["last_proposal"] = {"direction_ric": direction_ric,
                             "delta_v_mps": delta_v_mps,
                             "burn_offset_s": burn_offset_s}
    eng.agent_rounds += 1
    return res


@mcp.tool()
def issue_burn(rationale: str = "") -> dict:
    """Commit the last simulated maneuver as a signed command.

    Refuses unless `simulate_maneuver` cleared that exact burn. The uplink is
    simulated: the command is validated and written to the engagement log.
    """
    res = _ctx.get("last_result")
    if res is None:
        return {"error": "nothing simulated -- call simulate_maneuver first"}
    if not res.get("approved"):
        return {"issued": False, "refused": True,
                "reason": "the engine did not clear this maneuver",
                "failed_checks": res.get("failed_checks", [])}

    eng = _ctx["engagement"]
    p = _ctx["last_proposal"]
    eng.proposal = p
    eng.engine_verdict = res
    eng.authorize("openclaw agent (nemotron)")
    eng.reasoning = [{"t": 0.0, "source": "agent", "text": rationale}] if rationale else []
    _append_log(eng.as_dict())

    return {"issued": True, "engagement_id": eng.engagement_id,
            "command": {"type": "MANEUVER_EXEC", "vehicle": eng.primary,
                        "frame": "RIC", "direction": p["direction_ric"],
                        "delta_v_mps": p["delta_v_mps"],
                        "burn_offset_s": p["burn_offset_s"]},
            "predicted_miss_km": res["new_miss_km"],
            "predicted_pc": res["new_pc"],
            "uplink": eng.uplink,
            "logged_to": str(LOG_PATH)}


@mcp.tool()
def engagement_history(limit: int = 10) -> dict:
    """Past engagements this agent has handled. Persistent across sessions."""
    log = _load_log()
    return {"count": len(log), "log_path": str(LOG_PATH),
            "engagements": [{
                "id": e.get("engagement_id"), "primary": e.get("primary"),
                "secondary": e.get("secondary"),
                "miss_km": e.get("geometry", {}).get("miss_km"),
                "authorization": e.get("authorization"),
                "predicted_miss_km": (e.get("engine_verdict") or {}).get("new_miss_km"),
                "rounds": e.get("agent_rounds"),
            } for e in log[-limit:]]}


if __name__ == "__main__":
    mcp.run()
