#!/usr/bin/env python
"""Dress rehearsal: the whole demo, driven through the GPU code path.

This is NOT a benchmark and it does not pretend to be one. It installs the
strict CuPy stand-in from tests/test_gpu_path.py so every accelerated branch
executes -- device transfers, chunked screening, host round-trips -- and then
walks the exact sequence the demo walks. The point is to prove the GN100 path
runs the full flow, not just the unit-tested slice, and to show what tomorrow's
console output will look like.

Every timing printed here is this machine's. Structure carries over to the box;
timings do not.

    ./.venv/bin/python scripts/dry_run_gn100.py
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

Y, G, A, R, D, X = ("\033[38;5;220m", "\033[38;5;79m", "\033[38;5;214m",
                    "\033[38;5;203m", "\033[2m", "\033[0m")


def banner(t):
    print(f"\n{D}{'─'*70}{X}\n  {t}\n{D}{'─'*70}{X}")


def main() -> int:
    print(f"{Y}{'='*70}")
    print("  SIMULATED GN100 RUN — device code path, this machine's silicon")
    print(f"{'='*70}{X}")
    print(f"{D}  Structure is what tomorrow gives you. Timings are not.{X}")

    from test_gpu_path import install
    install()

    from kessler import accel
    from kessler.agents import resolve_backend, review_trade_space
    from kessler.bus import Bus
    from kessler.catalog import load_demo_catalog
    from kessler.mission import (Constraints, MissionState, altitude_shortlist,
                                 find_tca, requires_action, synthesize_threat)
    from kessler.monitor import select_fleet, sweep_fleet
    from kessler.options import solve_options
    from kessler.physics import elements, teme_state, timescale

    banner("PREFLIGHT")
    print(f"  accelerator   {accel.backend_name()}")
    info = accel.device_info()
    print(f"  device        {info.get('name','?')} · {info.get('total_gb','?')} GB · "
          f"unified={info.get('unified_memory')}")
    backend, model = resolve_backend()
    print(f"  agent backend {backend}" + (f" · {model}" if model else ""))

    banner("1 · CATALOGUE")
    t = time.time()
    cat = load_demo_catalog(limit=None)
    ts = timescale(); t0 = ts.now()
    print(f"  {len(cat):,} objects loaded in {time.time()-t:.2f}s · {cat.source}")

    banner("2 · FLEET SWEEP  (the resident state matrix)")
    fleet_n = int(os.environ.get("KESSLER_DRY_FLEET", "60"))
    fleet = select_fleet(cat.objects, "STARLINK", fleet_n)
    t = time.time()
    sweep = sweep_fleet(fleet, cat.objects, t0, horizon_s=6 * 3600,
                        coarse_step_s=60.0, threshold_km=25.0)
    dt = time.time() - t
    states = len(fleet) * sweep.catalog_size * sweep.epochs
    print(f"  {len(fleet)} assets × {sweep.catalog_size:,} objects × {sweep.epochs} epochs")
    print(f"  {states:,} states in {dt:.2f}s   matrix {sweep.matrix_mb:.0f} MB resident")
    print(f"  transfers: {sweep.transfer['calls']} call(s), "
          f"{sweep.transfer['bytes']/1e6:.0f} MB")
    if info.get("unified_memory"):
        print(f"  {G}on GB10 these are pointer handoffs, not PCIe copies{X}")
    print(f"  {len(sweep.watching)} on watch · {len(sweep.actionable)} breaching")

    banner("3 · CONJUNCTION")
    hero = fleet[0]
    r0, v0 = teme_state(hero, t0)
    el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
    enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
    c = Constraints(dv_budget_mps=0.35, min_miss_km=2.0)
    act, why = requires_action(enc, c)
    print(f"  {R}{hero.name} vs {tname}{X}")
    print(f"  miss {enc['miss_km']:.3f} km at T+{enc['tca_offset_s']/60:.0f} min · "
          f"{enc['rel_speed_kms']:.2f} km/s · Pc {enc['pc']:.2e}")
    print(f"  action required: {act} — {why}")

    banner("4 · TRADE SPACE  (the chunked device screen)")
    state = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                         nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2,
                         constraints=c)
    t = time.time()
    trade = solve_options(state, cat.objects, enc["tca_offset_s"])
    dt = time.time() - t
    print(f"  {trade['scenarios']} strategies × {trade['passes']} passes = "
          f"{trade['states_evaluated']:,} states in {dt:.2f}s")
    print(f"  logical tensor {trade['tensor_gb']:.2f} GB · "
          f"peak {trade['peak_chunk_gb']:.2f} GB (chunked)")
    print()
    print(f"  {'OPTION':<16}{'CLEARS':>9}{'FUEL':>10}{'COMMIT':>9}{'NEW CJ':>8}  VERDICT")
    for o in trade["options"]:
        col = G if o.feasible else A
        print(f"  {col}{o.label:<16}{o.miss_km:>8.2f}km{o.delta_v_mps:>9.3f}"
              f"{o.burn_offset_s/60:>8.1f}m{o.secondary_count:>8}  "
              f"{'feasible' if o.feasible else ','.join(o.failed)}{X}")

    banner("5 · AGENT REVIEW")
    bus = Bus(sink=lambda e: print(f"  {D}[{e.kind}]{X} {e.text[:88]}"))
    rec = review_trade_space(state, altitude_shortlist(cat.objects, el, 50.0),
                             trade["options"], enc | {"tca_offset_s": enc["tca_offset_s"],
                                                      "primary": hero.name,
                                                      "secondary": tname}, bus)
    print(f"  recommends: {G}{rec.get('recommended')}{X}")

    banner("6 · DECISION")
    feas = [o for o in trade["options"] if o.feasible]
    pick = next((o for o in feas if o.label == rec.get("recommended")), feas[0])
    print(f"  a human selects and presses Execute — no autonomous path exists")
    print(f"  {G}ISSUED{X}  {pick.label}: {enc['miss_km']:.3f} km → "
          f"{pick.miss_km:.2f} km for {pick.delta_v_mps:.3f} m/s")
    print(f"  uplink SIMULATED — validated and logged, never transmitted")

    print(f"\n{Y}{'='*70}")
    print("  FULL PATH EXECUTED ON DEVICE SEMANTICS — no host/device leaks")
    print(f"{'='*70}{X}")
    print(f"{D}  Tomorrow: ./scripts/remote.sh setup|bench|run user@GN100{X}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
