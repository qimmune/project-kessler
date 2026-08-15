"""Independent cross-check, engagement record, and the authorization gate.

Three ideas, in the order they run:

* CONSENSUS -- the encounter geometry is computed twice, by two independent
  dynamical models, and the residual between them is reported. A single
  propagator can be confidently wrong; two that disagree tell you so.
* ENGAGEMENT RECORD -- every conjunction produces one auditable object holding
  the geometry, the confidence, the full agent reasoning trail, and the verdict.
* AUTHORIZATION GATE -- nothing reaches the spacecraft without passing through
  it. In SUPERVISED mode a human authorizes or halts; in AUTONOMOUS mode the
  gate self-authorizes and records that it did. Either way the decision is
  logged, and either way this demo stops at a log file: there is no uplink.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from .physics import propagate, teme_state, timescale


# ----------------------------------------------------------------------
# consensus between two independent propagators
# ----------------------------------------------------------------------
@dataclass
class CrossCheck:
    residual_km: float          # disagreement between the two models at TCA
    horizon_s: float
    sigma_floor_km: float       # what that residual implies as a *floor* on uncertainty
    assumed_sigma_km: float     # what Pc actually uses
    consistent: bool

    def as_dict(self) -> dict:
        return asdict(self)


def cross_check(sat, t0, horizon_s: float, assumed_sigma_km: float = 0.35,
                tolerance_km: float = 1.0) -> CrossCheck:
    """Propagate one object two ways and measure how far apart the answers land.

    SGP4 is the model the TLE is defined against. RK4 two-body + J2 is an
    independent integrator seeded from the same state. They share no code path
    beyond the initial conditions, so their residual is a real consistency
    check -- if it blows up, the geometry downstream should not be trusted.
    """
    ts = timescale()
    t_end = ts.tt_jd(t0.tt + horizon_s / 86400.0)
    r_sgp4, _ = teme_state(sat, t_end)

    r0, v0 = teme_state(sat, t0)
    _, rs, _ = propagate(r0, v0, horizon_s, dt_s=5.0)
    r_rk4 = rs[:, -1]

    residual = float(np.linalg.norm(r_sgp4 - r_rk4))
    return CrossCheck(
        residual_km=residual, horizon_s=float(horizon_s),
        sigma_floor_km=residual,
        assumed_sigma_km=assumed_sigma_km,
        consistent=bool(residual < tolerance_km and np.isfinite(residual)))


# ----------------------------------------------------------------------
# authorization
# ----------------------------------------------------------------------
class Mode(str, Enum):
    AUTONOMOUS = "AUTONOMOUS"
    SUPERVISED = "SUPERVISED"


class Authorization(str, Enum):
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    HALTED = "HALTED"
    NOT_REQUIRED = "NOT_REQUIRED"


@dataclass
class Engagement:
    """One conjunction, start to finish, as an auditable record."""
    engagement_id: str = field(default_factory=lambda: f"KES-{uuid.uuid4().hex[:8].upper()}")
    opened_at: float = field(default_factory=time.time)
    primary: str = ""
    secondary: str = ""
    seeded: bool = False

    geometry: dict = field(default_factory=dict)
    consensus: dict = field(default_factory=dict)
    action_required: bool = False
    assessment: str = ""

    proposal: dict | None = None
    engine_verdict: dict | None = None
    agent_rounds: int = 0
    reasoning: list = field(default_factory=list)

    mode: str = Mode.SUPERVISED.value
    authorization: str = Authorization.PENDING.value
    authorized_by: str = ""
    authorized_at: float | None = None
    halt_reason: str = ""
    uplink: str = "SIMULATED — no live actuation in this build"

    def record_reasoning(self, events) -> None:
        self.reasoning = [
            {"t": round(e.ts - self.opened_at, 2), "source": e.kind, "text": e.text}
            for e in events]

    def authorize(self, by: str) -> None:
        self.authorization = Authorization.AUTHORIZED.value
        self.authorized_by = by
        self.authorized_at = time.time()

    def halt(self, by: str, reason: str) -> None:
        self.authorization = Authorization.HALTED.value
        self.authorized_by = by
        self.authorized_at = time.time()
        self.halt_reason = reason

    @property
    def issued(self) -> bool:
        return self.authorization == Authorization.AUTHORIZED.value

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)


class EngagementLog:
    """Append-only log of every engagement this session opened."""

    def __init__(self):
        self.entries: list[Engagement] = []

    def open(self, **kw) -> Engagement:
        e = Engagement(**kw)
        self.entries.append(e)
        return e

    def to_json(self) -> str:
        return json.dumps([e.as_dict() for e in self.entries], indent=2, default=str)

    def summary_rows(self) -> list[dict]:
        return [{
            "ID": e.engagement_id,
            "Opened": time.strftime("%H:%M:%S", time.localtime(e.opened_at)),
            "Primary": e.primary,
            "Secondary": e.secondary,
            "Miss (km)": round(e.geometry.get("miss_km", float("nan")), 3),
            "Pc": f"{e.geometry.get('pc', 0):.1e}",
            "Consensus (m)": round(e.consensus.get("residual_km", 0) * 1000, 1),
            "Action": "YES" if e.action_required else "no",
            "Rounds": e.agent_rounds,
            "Mode": e.mode,
            "Authorization": e.authorization,
        } for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)
