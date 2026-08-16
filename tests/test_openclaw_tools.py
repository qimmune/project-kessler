"""Drive the OpenClaw/NemoClaw tool surface end to end over MCP.

This is the deliverable the NemoClaw bounty asks for: a working agent tool layer
with live tool use and persistent memory. The test walks the exact sequence an
agent walks -- screen, open, propose, get rejected, correct, commit -- and
asserts the engine refuses to issue a burn it did not clear.
"""
import asyncio
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "kes_mcp", os.path.join(ROOT, "openclaw", "kessler_mcp.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def unwrap(res):
    """CallToolResult -> dict, whichever shape the SDK returned."""
    sc = getattr(res, "structuredContent", None)
    if isinstance(sc, dict):
        return sc.get("result", sc)
    for block in getattr(res, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
    return {}


async def run():
    names = [t.name for t in await m.mcp.list_tools()]
    for expected in ("screen_fleet", "open_conjunction", "simulate_maneuver",
                     "issue_burn", "engagement_history"):
        assert expected in names, f"{expected} not registered"
    print("  tools:", ", ".join(names))

    d = unwrap(await m.mcp.call_tool("screen_fleet", {"assets": 6, "horizon_hours": 6.0}))
    assert d["catalog_size"] > 10000
    print(f"  screen_fleet      {d['catalog_size']:,} objects · "
          f"{d['states_propagated']:,} states · {d['sweep_seconds']}s · {d['backend']}")
    print(f"                    host->device transfer: {d['host_to_device_transfer']}")

    d = unwrap(await m.mcp.call_tool("open_conjunction", {"asset": "STARLINK-1008"}))
    assert d["action_required"] is True
    eng_id = d["engagement_id"]
    print(f"  open_conjunction  {d['primary']} vs {d['secondary']} · "
          f"{d['miss_km']} km · Pc {d['probability_of_collision']:.2e} · "
          f"consensus {d['propagator_consensus_m']} m")

    # the intuitive wrong answer
    d = unwrap(await m.mcp.call_tool("simulate_maneuver",
                                     {"direction_ric": [1, 0, 0], "delta_v_mps": 0.18,
                                      "burn_offset_s": 600}))
    assert d["approved"] is False, "radial burn should not clear the threat"
    print(f"  simulate radial   approved={d['approved']} · {d['new_miss_km']} km · "
          f"failed {d['failed_checks']}")

    # the engine must refuse to commit it
    d = unwrap(await m.mcp.call_tool("issue_burn", {"rationale": "trying it anyway"}))
    assert d.get("issued") is False and d.get("refused") is True
    print(f"  issue_burn        REFUSED — {d['reason']}")

    # the correction
    d = unwrap(await m.mcp.call_tool("simulate_maneuver",
                                     {"direction_ric": [0, 1, 0], "delta_v_mps": 0.19,
                                      "burn_offset_s": 600}))
    assert d["approved"] is True, "in-track burn should clear"
    print(f"  simulate in-track approved={d['approved']} · {d['new_miss_km']} km · "
          f"Pc {d['new_pc']:.1e}")

    d = unwrap(await m.mcp.call_tool("issue_burn",
                                     {"rationale": "in-track is the cheapest axis"}))
    assert d["issued"] is True and d["engagement_id"] == eng_id
    print(f"  issue_burn        ISSUED {d['engagement_id']} → "
          f"{d['predicted_miss_km']} km · uplink {d['uplink']}")

    d = unwrap(await m.mcp.call_tool("engagement_history", {}))
    assert d["count"] >= 1, "engagement did not persist"
    print(f"  history           {d['count']} engagement(s) persisted → {d['log_path']}")
    print("\nOK")


if __name__ == "__main__":
    asyncio.run(run())
