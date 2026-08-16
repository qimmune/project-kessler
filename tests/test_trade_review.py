"""Prove the demo path is genuinely agentic.

The risk this guards against is real and already happened once: the trade-space
UI can silently degrade into pure physics plus a ranking rule, with no model and
no tool use anywhere. This test asserts the critic reaches the physics engine and
can go beyond the generated option set.
"""
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kessler import agents
from kessler.bus import Bus
from kessler.catalog import load_demo_catalog
from kessler.mission import (Constraints, MissionState, altitude_shortlist,
                             find_tca, synthesize_threat)
from kessler.options import solve_options
from kessler.physics import elements, teme_state, timescale


class Block(types.SimpleNamespace):
    pass


class Critic:
    """Probes a cheaper variant, then commits to it."""

    def __init__(self):
        self.turns = 0

    def create(self, **kw):
        self.turns += 1
        saw_result = any(
            isinstance(m.get("content"), list)
            and any(isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in m["content"])
            for m in kw["messages"])
        assert kw.get("tools"), "critic was not given the simulate tool"
        if not saw_result:
            # go looking for something cheaper than anything generated
            return types.SimpleNamespace(content=[Block(
                type="tool_use", id="t1", name="simulate_maneuver",
                input={"direction_ric": [0, 1, 0], "delta_v_mps": 0.21,
                       "burn_offset_s": 240})])
        return types.SimpleNamespace(content=[Block(type="text", text=json.dumps({
            "recommended": "Agent variant 1: 0.210 m/s at T+4 min",
            "rationale": "Igniting four minutes in clears 2.6 km for 0.21 m/s, "
                         "beating every generated option on fuel.",
            "runner_up": "Minimum fuel", "why_not": "costs more for less clearance.",
            "probed": "an earlier ignition than the generated family covered"}))])


def test_demo_path_is_agentic():
    critic = Critic()
    agents._client = lambda: types.SimpleNamespace(messages=critic)
    agents.resolve_backend = lambda: ("nemotron", "nvidia/nemotron-3-super")

    ts = timescale()
    cat = load_demo_catalog(limit=None)
    hero = cat.by_name("STARLINK-1008")
    t0 = ts.now()
    r0, v0 = teme_state(hero, t0)
    el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
    enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
    state = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                         nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2,
                         constraints=Constraints(dv_budget_mps=0.6))
    alert = {"primary": hero.name, "secondary": tname,
             "tca_offset_s": enc["tca_offset_s"], "miss_km": enc["miss_km"],
             "pc": enc["pc"], "rel_speed_kms": enc["rel_speed_kms"]}

    trade = solve_options(state, cat.objects, enc["tca_offset_s"])
    bus = Bus()
    rec = agents.review_trade_space(state, altitude_shortlist(cat.objects, el, 50.0),
                                    trade["options"], alert, bus)

    kinds = [e.kind for e in bus.events]
    assert "tool" in kinds, "the critic never reached the physics engine"
    assert rec["tool_calls"] >= 1, "no simulation was run by the agent"
    assert rec["variants"], "agent produced no variant beyond the generated set"
    assert critic.turns >= 2, "critic did not complete a tool round trip"

    v = rec["variants"][0]
    print(f"  generated options : {len(trade['options'])} "
          f"({sum(1 for o in trade['options'] if o.feasible)} feasible)")
    print(f"  agent tool calls  : {rec['tool_calls']}")
    print(f"  agent variant     : {v['label']} -> {v['result']['new_miss_km']:.3f} km, "
          f"approved={v['result']['approved']}")
    print(f"  recommended       : {rec['recommended']}")
    print(f"  probed            : {rec['probed']}")
    for e in bus.events:
        if e.kind in ("tool", "agent1", "agent2"):
            print(f"    [{e.kind}] {e.text[:96]}")


def test_falls_back_without_a_backend():
    """No model configured must still produce a decision, not a crash."""
    agents._client = lambda: None
    agents.resolve_backend = lambda: ("offline", "")
    ts = timescale()
    cat = load_demo_catalog(limit=None)
    hero = cat.by_name("STARLINK-1008")
    t0 = ts.now(); r0, v0 = teme_state(hero, t0); el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
    enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
    state = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                         nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2,
                         constraints=Constraints(dv_budget_mps=0.6))
    trade = solve_options(state, cat.objects, enc["tca_offset_s"])
    bus = Bus()
    rec = agents.review_trade_space(state, [], trade["options"],
                                    {"tca_offset_s": enc["tca_offset_s"],
                                     "miss_km": enc["miss_km"], "pc": enc["pc"]}, bus)
    assert rec.get("recommended"), "fallback produced no decision"
    print(f"  no backend -> rule-based fallback chose: {rec['recommended']}")


if __name__ == "__main__":
    print("test_demo_path_is_agentic"); test_demo_path_is_agentic()
    print("\ntest_falls_back_without_a_backend"); test_falls_back_without_a_backend()
    print("\nOK")
