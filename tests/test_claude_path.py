"""Exercise the Claude branch of the agent loop with a stubbed client.

There is no API key in CI, but the tool-calling path is the one that runs in the
demo, so it needs to be executed at least once against a fake transport.
"""
import json
import sys
import types

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from kessler import agents
from kessler.bus import Bus
from kessler.catalog import load_demo_catalog
from kessler.mission import (Constraints, MissionState, altitude_shortlist, find_tca,
                             synthesize_threat)
from kessler.physics import elements, teme_state, timescale


class Block(types.SimpleNamespace):
    pass


class FakeMessages:
    """Replays a scripted FDO proposal, then a critic turn that calls the tool."""

    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        system = kw.get("system", "")
        # NB: MAD_SYSTEM also mentions the Flight Dynamics Officer, so the
        # critic must be matched first or every call routes to the proposer.
        if "Mission Assurance Director" not in system:
            payload = json.dumps({"direction_ric": [0, 1, 0], "delta_v_mps": 0.19,
                                  "burn_offset_s": 600, "rationale": "in-track, cheapest axis"})
            return types.SimpleNamespace(content=[Block(type="text", text=payload)])

        # critic: first turn calls the tool, second turn returns a verdict
        has_result = any(
            isinstance(m.get("content"), list)
            and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in m["content"])
            for m in kw["messages"])
        if not has_result:
            return types.SimpleNamespace(content=[Block(
                type="tool_use", id="tu_1", name="simulate_maneuver",
                input={"direction_ric": [0, 1, 0], "delta_v_mps": 0.19, "burn_offset_s": 600})])
        return types.SimpleNamespace(content=[Block(type="text",
                                                    text=json.dumps({"verdict": "APPROVED"}))])


def test_claude_tool_loop():
    agents._client = lambda: types.SimpleNamespace(messages=FakeMessages())

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
                         constraints=Constraints())
    alert = {"primary": hero.name, "secondary": tname,
             "tca_offset_s": enc["tca_offset_s"], "miss_km": enc["miss_km"], "pc": enc["pc"]}

    bus = Bus()
    out = agents.run_resolution(state, altitude_shortlist(cat.objects, el, 50.0), alert, bus)

    kinds = [e.kind for e in bus.events]
    assert "tool" in kinds, "critic never reached the physics engine"
    assert "error" not in kinds, f"clean run should emit no errors, got {[e.text for e in bus.events if e.kind=='error']}"
    assert out["approved"] is True, "scripted safe burn should have been approved"
    assert out["result"]["new_miss_km"] > 2.0
    print(f"  tool called, verdict APPROVED, miss {out['result']['new_miss_km']:.3f} km")
    print(f"  event kinds: {kinds}")


def test_engine_overrides_a_lying_critic():
    """If the model claims APPROVED but the engine disagrees, the engine wins."""
    class LyingMessages(FakeMessages):
        def create(self, **kw):
            system = kw.get("system", "")
            if "Mission Assurance Director" not in system:
                return types.SimpleNamespace(content=[Block(type="text", text=json.dumps(
                    {"direction_ric": [0, 1, 0], "delta_v_mps": 0.001,
                     "burn_offset_s": 600, "rationale": "nowhere near enough"}))])
            has_result = any(
                isinstance(m.get("content"), list)
                and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in m["content"])
                for m in kw["messages"])
            if not has_result:
                return types.SimpleNamespace(content=[Block(
                    type="tool_use", id="tu_1", name="simulate_maneuver",
                    input={"direction_ric": [0, 1, 0], "delta_v_mps": 0.001, "burn_offset_s": 600})])
            return types.SimpleNamespace(content=[Block(type="text",
                                                        text=json.dumps({"verdict": "APPROVED"}))])

    agents._client = lambda: types.SimpleNamespace(messages=LyingMessages())
    ts = timescale()
    cat = load_demo_catalog(limit=None)
    hero = cat.by_name("STARLINK-1008")
    t0 = ts.now(); r0, v0 = teme_state(hero, t0); el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
    enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
    state = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                         nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2)
    alert = {"primary": hero.name, "secondary": tname,
             "tca_offset_s": enc["tca_offset_s"], "miss_km": enc["miss_km"], "pc": enc["pc"]}
    bus = Bus()
    out = agents.run_resolution(state, altitude_shortlist(cat.objects, el, 50.0), alert, bus, max_rounds=1)
    assert out["approved"] is False, "a 0.001 m/s burn must not be approved"
    assert any(e.kind == "error" and "engine said" in e.text.lower() for e in bus.events)
    print("  model claimed APPROVED on an unsafe burn; engine overrode it")


if __name__ == "__main__":
    print("test_claude_tool_loop"); test_claude_tool_loop()
    print("test_engine_overrides_a_lying_critic"); test_engine_overrides_a_lying_critic()
    print("\nOK")
