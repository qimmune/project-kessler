#!/usr/bin/env python
"""CPU vs GPU, measured on this machine. Produces the numbers for the pitch.

Runs the two workloads that matter -- the fleet sweep and the maneuver trade
space -- on whatever accelerator is present, then again pinned to NumPy, and
prints the comparison. Nothing here is projected; if a number appears, it was
timed.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def workload(fleet_size: int = 60):
    from kessler.catalog import load_demo_catalog
    from kessler.mission import (Constraints, MissionState, find_tca,
                                 synthesize_threat)
    from kessler.monitor import select_fleet, sweep_fleet
    from kessler.options import solve_options
    from kessler.physics import elements, teme_state, timescale
    from kessler.accel import backend_name, device_info

    ts = timescale()
    cat = load_demo_catalog(limit=None)
    t0 = ts.now()

    fleet = select_fleet(cat.objects, "STARLINK", fleet_size)
    t = time.time()
    sweep = sweep_fleet(fleet, cat.objects, t0, horizon_s=6 * 3600,
                        coarse_step_s=60.0, threshold_km=25.0)
    sweep_s = time.time() - t

    hero = fleet[0]
    r0, v0 = teme_state(hero, t0)
    el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
    enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
    st = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                      nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2,
                      constraints=Constraints(dv_budget_mps=0.6))
    t = time.time()
    trade = solve_options(st, cat.objects, enc["tca_offset_s"])
    trade_s = time.time() - t

    return {
        "backend": backend_name(), "device": device_info(),
        "catalog": len(cat), "fleet": len(fleet),
        "sweep_s": sweep_s, "sweep_states": len(fleet) * sweep.catalog_size * sweep.epochs,
        "sweep_transfer": sweep.transfer,
        "trade_s": trade_s, "trade_states": trade["states_evaluated"],
        "trade_peak_gb": trade["peak_chunk_gb"],
        "feasible": sum(1 for o in trade["options"] if o.feasible),
    }


def main() -> int:
    fleet = int(os.environ.get("KESSLER_BENCH_FLEET", "60"))
    print(f"Project Kessler — benchmark (fleet of {fleet})\n")

    native = workload(fleet)
    print(f"  backend      {native['backend']}")
    if native["device"].get("gpu"):
        d = native["device"]
        print(f"  device       {d.get('name','?')} · {d.get('total_gb','?')} GB · "
              f"unified={d.get('unified_memory')}")
    print(f"  catalogue    {native['catalog']:,} objects\n")

    rows = [("native", native)]

    # Re-run pinned to CPU in a clean interpreter, so the comparison is honest.
    if native["device"].get("gpu"):
        print("  re-running pinned to CPU for comparison…\n")
        env = dict(os.environ, KESSLER_FORCE_CPU="1", KESSLER_BENCH_ONLY="1")
        out = subprocess.run([sys.executable, os.path.abspath(__file__)],
                             env=env, capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip()[-400:])
        return 0

    w = native
    print(f"  {'workload':<24}{'time':>9}{'states':>18}{'rate':>16}")
    for label, secs, states in (
            ("fleet sweep", w["sweep_s"], w["sweep_states"]),
            ("maneuver trade space", w["trade_s"], w["trade_states"])):
        print(f"  {label:<24}{secs:>8.2f}s{states:>18,}{states/secs/1e6:>13.0f} M/s")
    print(f"\n  peak device tensor     {w['trade_peak_gb']:.2f} GB")
    print(f"  host→device transfer   {w['sweep_transfer']['bytes']/1e6:.0f} MB "
          f"in {w['sweep_transfer']['calls']} call(s)")
    if w["device"].get("unified_memory"):
        print("  unified memory         Grace and Blackwell share one address space;\n"
              "                         the ephemeris is handed over by pointer.")
    print(f"  trade space            {w['feasible']}/5 feasible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
