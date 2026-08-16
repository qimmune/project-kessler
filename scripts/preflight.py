#!/usr/bin/env python
"""Verify the box will run the demo, before anyone is watching.

Checks the numerical stack, the accelerator, the catalogue, the agent backend,
and then actually solves a small trade space end to end and times it.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

G, A, R, D = "\033[38;5;79m", "\033[38;5;214m", "\033[38;5;203m", "\033[2m"
X = "\033[0m"
fails: list[str] = []


def ok(m): print(f"  {G}✓{X} {m}")
def warn(m): print(f"  {A}!{X} {m}")
def bad(m): print(f"  {R}✗{X} {m}"); fails.append(m)


def main() -> int:
    print(f"{D}{'-'*66}{X}")

    # ---- stack ----
    try:
        import numpy, skyfield, plotly, streamlit
        ok(f"numpy {numpy.__version__} · skyfield {skyfield.__version__} · "
           f"streamlit {streamlit.__version__}")
    except ImportError as e:
        bad(f"missing dependency: {e}")
        return 1

    # ---- accelerator ----
    from kessler.accel import GPU, backend_name, device_info, memory_report
    info = device_info()
    if GPU:
        ok(f"accelerator: {backend_name()}")
        if info.get("unified_memory"):
            ok(f"unified memory confirmed — {info.get('total_gb','?')} GB shared, "
               f"host↔device copies are pointer handoffs")
        else:
            warn("CUDA present but not reported as integrated/unified — "
                 "transfers will be real copies over PCIe")
        print(f"    {D}{memory_report()}{X}")
    else:
        warn("no CuPy/CUDA — screening runs on NumPy. Demo works, just slower.")
        warn("  install with: .venv/bin/pip install cupy-cuda12x")

    # ---- catalogue ----
    try:
        from kessler.catalog import classify, load_demo_catalog
        from collections import Counter
        t = time.time()
        cat = load_demo_catalog(limit=None)
        n = Counter(classify(s.name) for s in cat.objects)
        ok(f"catalogue: {len(cat):,} objects in {time.time()-t:.2f}s "
           f"({n['payload']:,} payload, {n['debris']:,} debris) — {cat.source}")
    except Exception as e:
        bad(f"catalogue unavailable: {e}")
        return 1

    # ---- agent backend ----
    from kessler.agents import BASE_URL, resolve_backend
    backend, model = resolve_backend()
    if backend == "nemotron":
        ok(f"agent backend: Nemotron {model} @ {BASE_URL}")
        try:
            import urllib.request
            urllib.request.urlopen(BASE_URL.rstrip("/") + "/models", timeout=4)
            ok("  endpoint reachable")
        except Exception as e:
            warn(f"  endpoint not reachable ({type(e).__name__}) — "
                 f"start the NIM, or the deterministic solver takes over")
    elif backend == "claude":
        ok(f"agent backend: Claude {model}")
    else:
        warn("agent backend: deterministic solver (no model configured)")
        warn("  for Nemotron:  export KESSLER_BACKEND=nemotron "
             "KESSLER_BASE_URL=http://localhost:8000/v1")

    # ---- the real thing ----
    print(f"{D}{'-'*66}{X}")
    print("  solving a live trade space…")
    from kessler.mission import (Constraints, MissionState, find_tca,
                                 synthesize_threat)
    from kessler.options import solve_options
    from kessler.physics import elements, teme_state, timescale

    ts = timescale(); t0 = ts.now()
    hero = cat.by_name("STARLINK-1008") or cat.objects[0]
    r0, v0 = teme_state(hero, t0)
    el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
    enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
    st = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                      nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2,
                      constraints=Constraints(dv_budget_mps=0.6))
    t = time.time()
    res = solve_options(st, cat.objects, enc["tca_offset_s"])
    dt = time.time() - t
    feas = sum(1 for o in res["options"] if o.feasible)

    if feas == 0:
        bad("no feasible option — the demo would dead-end")
    else:
        ok(f"trade space: {feas}/{res['scenarios']} feasible in {dt:.1f}s")
    ok(f"  {res['states_evaluated']:,} states · {res['tensor_gb']:.2f} GB logical · "
       f"{res['peak_chunk_gb']:.2f} GB peak · {res['backend']}")
    ok(f"  transfer: {res['transfer']}")
    if dt > 20:
        warn(f"  {dt:.0f}s is slow for a live demo — on GB10 with CuPy this should "
             f"be low single digits. Check the accelerator line above.")

    print(f"{D}{'-'*66}{X}")
    if fails:
        print(f"  {R}{len(fails)} blocking issue(s){X}")
        return 1
    print(f"  {G}READY{X} — ./scripts/run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
