"""Exercise the Nemotron adapter against a stubbed OpenAI-compatible endpoint.

There is no NVIDIA endpoint in CI, but the translation between OpenAI tool
calling and the Anthropic-shaped call sites is exactly where this breaks, so it
gets driven at least once.
"""
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KESSLER_BACKEND"] = "nemotron"
os.environ["NVIDIA_API_KEY"] = "stub"

from kessler import agents
from kessler.bus import Bus
from kessler.catalog import load_demo_catalog
from kessler.mission import (Constraints, MissionState, altitude_shortlist, find_tca,
                             synthesize_threat)
from kessler.physics import elements, teme_state, timescale


class StubCompletions:
    """Replays an OpenAI-shaped proposal, a tool call, then a verdict."""

    def __init__(self):
        self.seen_tool_roles = 0

    def create(self, model, max_tokens, messages, tools=None, tool_choice=None):
        system = messages[0]["content"]
        assert messages[0]["role"] == "system"
        if "Mission Assurance Director" not in system:
            body = json.dumps({"direction_ric": [0, 1, 0], "delta_v_mps": 0.19,
                               "burn_offset_s": 600, "rationale": "in-track"})
            return _resp(content=body)

        assert tools and tools[0]["type"] == "function", "tools were not translated"
        assert tools[0]["function"]["name"] == "simulate_maneuver"
        if any(m.get("role") == "tool" for m in messages):
            self.seen_tool_roles += 1
            return _resp(content=json.dumps({"verdict": "APPROVED"}))
        return _resp(tool_calls=[types.SimpleNamespace(
            id="call_1", type="function",
            function=types.SimpleNamespace(
                name="simulate_maneuver",
                arguments=json.dumps({"direction_ric": [0, 1, 0],
                                      "delta_v_mps": 0.19, "burn_offset_s": 600})))])


def _resp(content=None, tool_calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_nemotron_tool_loop():
    stub = StubCompletions()
    client = agents.NemotronClient.__new__(agents.NemotronClient)
    client.model = "nvidia/nemotron-3-super"
    client.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=stub))
    client.messages = client
    agents._client = lambda: client

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
             "tca_offset_s": enc["tca_offset_s"], "miss_km": enc["miss_km"],
             "pc": enc["pc"]}

    bus = Bus()
    out = agents.run_resolution(state, altitude_shortlist(cat.objects, el, 50.0),
                                alert, bus)
    kinds = [e.kind for e in bus.events]
    assert "tool" in kinds, "Nemotron critic never reached the physics engine"
    assert stub.seen_tool_roles >= 1, "tool results were not fed back as role=tool"
    assert out["approved"] is True
    assert "Nemotron" in bus.events[0].text
    print(f"  tool call translated, result fed back, verdict APPROVED "
          f"({out['result']['new_miss_km']:.3f} km)")
    print(f"  banner: {bus.events[0].text}")


if __name__ == "__main__":
    print("test_nemotron_tool_loop")
    test_nemotron_tool_loop()
    print("\nOK")
