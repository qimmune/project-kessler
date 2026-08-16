"""Generate and evaluate a family of avoidance maneuvers, not just one.

A single answer needs a laptop. The interesting problem is the trade space: for
one conjunction there are many burns that clear it, and they differ in fuel,
in how long you can wait before committing, in what they do to the mission
orbit, and in what new conjunctions they create over the following twelve hours.
An operator wants to choose. Choosing requires all of the options costed, which
means propagating and re-screening each one against the whole catalogue.

That is what makes this a hardware problem. Each scenario is an independent
re-propagation plus a full catalogue screen, and they are evaluated as one
batched tensor rather than a loop, so the accelerator sees a single wide
operation over a resident state matrix.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .accel import Transfer, backend_name, free_pool, to_device, to_host, xp
from .conjunction import collision_probability
from .mission import Constraints, MissionState, altitude_shortlist, find_tca_window
from .physics import (MU, apply_burn, elements, propagate, teme_positions_many,
                      timescale)


@dataclass
class Option:
    """One costed way out."""
    label: str
    strategy: str
    direction_ric: list
    delta_v_mps: float
    burn_offset_s: float

    miss_km: float = 0.0
    pc: float = 0.0
    altitude_drift_km: float = 0.0
    secondary_count: int = 0
    secondary_worst: dict | None = None
    fuel_pct_of_budget: float = 0.0
    decision_time_s: float = 0.0
    feasible: bool = False
    failed: list = field(default_factory=list)

    pros: list = field(default_factory=list)
    cons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# candidate generation
# ----------------------------------------------------------------------
def _mean_motion(r, v) -> float:
    return float(np.sqrt(MU / elements(r, v)["sma_km"] ** 3))


def generate_candidates(state: MissionState, tca_offset_s: float,
                        target_miss_km: float | None = None) -> list[Option]:
    """Span the trade space with physically distinct strategies.

    Each is sized from the analytic response of its own axis, then measured for
    real. The point is that they fail and succeed for different reasons: the
    cheap one commits early, the late one costs fuel, the plane change preserves
    the mission orbit and costs the most.
    """
    c = state.constraints
    target = target_miss_km or c.min_miss_km * 1.25
    n = _mean_motion(state.hero_r0, state.hero_v0)

    def in_track_dv(lead_s: float) -> float:
        # separation from an in-track burn accumulates as ~3*dv*t
        return target / (3.0 * max(lead_s, 1.0)) * 1000.0

    early = min(300.0, tca_offset_s * 0.08)
    mid = tca_offset_s * 0.15
    late = tca_offset_s * 0.45

    cands = [
        Option(label="Minimum fuel", strategy="prograde in-track, early",
               direction_ric=[0, 1, 0],
               delta_v_mps=in_track_dv(tca_offset_s - early), burn_offset_s=early),
        Option(label="Balanced", strategy="prograde in-track, nominal",
               direction_ric=[0, 1, 0],
               delta_v_mps=in_track_dv(tca_offset_s - mid), burn_offset_s=mid),
        Option(label="Latest commit", strategy="prograde in-track, late",
               direction_ric=[0, 1, 0],
               delta_v_mps=in_track_dv(tca_offset_s - late), burn_offset_s=late),
        Option(label="Drop back", strategy="retrograde in-track",
               direction_ric=[0, -1, 0],
               delta_v_mps=in_track_dv(tca_offset_s - mid), burn_offset_s=mid),
        # Cross-track displacement peaks at dv/n; a plane change holds altitude
        # and along-track timing, which is what a ground-station schedule cares
        # about, and costs an order of magnitude more.
        Option(label="Plane shift", strategy="cross-track",
               direction_ric=[0, 0, 1],
               delta_v_mps=target * n * 1000.0 * 1.2, burn_offset_s=mid),
    ]
    for o in cands:
        o.delta_v_mps = round(float(o.delta_v_mps), 4)
        o.decision_time_s = float(o.burn_offset_s)
    return cands


# ----------------------------------------------------------------------
# batched evaluation
# ----------------------------------------------------------------------
def evaluate_options(state: MissionState, catalog_objects, options: list[Option],
                     tca_offset_s: float, secondary_step_s: float = 30.0) -> dict:
    """Fly every option and screen each resulting orbit against the catalogue.

    The per-option post-burn arcs are stacked into one (n_options, T, 3) tensor
    and differenced against the catalogue in a single broadcast, so the whole
    trade space is screened as one operation instead of a Python loop.
    """
    c = state.constraints
    started = time.time()
    Transfer.reset()

    el0 = elements(state.hero_r0, state.hero_v0)
    shortlist = altitude_shortlist(catalog_objects, el0, pad_km=80.0)
    horizon = c.secondary_horizon_s
    n_steps = int(horizon / secondary_step_s) + 1

    arcs, refs = [], []
    for o in options:
        _, rs, vs = propagate(state.hero_r0, state.hero_v0, o.burn_offset_s, dt_s=2.0)
        r_burn, v_burn = rs[:, -1], vs[:, -1]
        d = np.asarray(o.direction_ric, float)
        d = d / np.linalg.norm(d)
        v_post = apply_burn(r_burn, v_burn, d * o.delta_v_mps)

        # primary threat, bracketed to the conjunction being avoided
        _, tr, tv = propagate(state.threat_r0, state.threat_v0, o.burn_offset_s, dt_s=2.0)
        enc = find_tca_window(r_burn, v_post, tr[:, -1], tv[:, -1],
                              centre_s=tca_offset_s - o.burn_offset_s,
                              half_window_s=300.0)
        o.miss_km = round(float(enc["miss_km"]), 4)
        o.pc = float(enc["pc"])

        el_ref, el_new = elements(r_burn, v_burn), elements(r_burn, v_post)
        o.altitude_drift_km = round(abs(
            (el_new["perigee_alt_km"] + el_new["apogee_alt_km"]) / 2
            - (el_ref["perigee_alt_km"] + el_ref["apogee_alt_km"]) / 2), 3)
        o.fuel_pct_of_budget = round(100.0 * o.delta_v_mps / c.dv_budget_mps, 1)

        _, arc, _ = propagate(r_burn, v_post, horizon, dt_s=secondary_step_s)
        arcs.append(arc[:, :n_steps].T)
        refs.append(o)

    # ---- one batched screen over the whole trade space ----
    ts = timescale()
    t_grid = ts.tt_jd(state.t0.tt + np.arange(n_steps) * (secondary_step_s / 86400.0))
    cat_r = teme_positions_many(shortlist, t_grid)[:, :n_steps, :]      # (N, T, 3)
    opt_r = np.stack(arcs, axis=0)                                      # (K, T, 3)

    g_cat = to_device(cat_r)
    g_opt = to_device(opt_r)

    # The full (K, N, T, 3) difference would be gigabytes materialised at once.
    # Chunking over catalogue objects bounds peak memory regardless of how many
    # scenarios are in flight, so the same code runs on a laptop and on 128 GB
    # of unified memory without a separate path.
    n_obj = cat_r.shape[0]
    chunk = max(1, int(os.environ.get("KESSLER_SCREEN_CHUNK", "1500")))
    mins = []
    for i in range(0, n_obj, chunk):
        sub = g_cat[i:i + chunk]
        diff = g_opt[:, None, :, :] - sub[None, :, :, :]
        d = xp.sqrt((diff * diff).sum(axis=3))
        d = xp.where(xp.isnan(d), xp.inf, d)
        mins.append(d.min(axis=2))
        del diff, d
    per_obj_min = to_host(xp.concatenate(mins, axis=1))                 # (K, N)
    del mins, g_cat, g_opt
    free_pool()
    tensor_gb = (opt_r.shape[0] * cat_r.shape[0] * n_steps * 3 * 8) / 1e9
    peak_chunk_gb = (opt_r.shape[0] * min(chunk, n_obj) * n_steps * 3 * 8) / 1e9

    for k, o in enumerate(refs):
        hits = np.where(per_obj_min[k] < c.secondary_screen_km)[0]
        hits = [h for h in hits if shortlist[h].name != state.hero_name]
        o.secondary_count = len(hits)
        if hits:
            j = min(hits, key=lambda h: per_obj_min[k][h])
            o.secondary_worst = {"object": shortlist[j].name,
                                 "miss_km": round(float(per_obj_min[k][j]), 3)}

        o.failed = []
        if o.miss_km < c.min_miss_km or o.pc > c.max_pc:
            o.failed.append("primary_threat_cleared")
        if o.delta_v_mps > c.dv_budget_mps:
            o.failed.append("delta_v_within_budget")
        if o.altitude_drift_km > c.altitude_box_km:
            o.failed.append("altitude_box_held")
        if o.secondary_count:
            o.failed.append("no_new_conjunctions")
        o.feasible = not o.failed

    _characterize(refs, c)
    refs.sort(key=lambda o: (not o.feasible, o.delta_v_mps))

    return {
        "options": refs,
        "screened_objects": len(shortlist),
        "epochs": n_steps,
        "scenarios": len(refs),
        "states_evaluated": len(refs) * len(shortlist) * n_steps,
        "tensor_gb": round(tensor_gb, 3),
        "peak_chunk_gb": round(peak_chunk_gb, 3),
        "elapsed_s": round(time.time() - started, 2),
        "backend": backend_name(),
        "transfer": Transfer.report(),
    }


def solve_options(state: MissionState, catalog_objects, tca_offset_s: float,
                  target_miss_km: float | None = None,
                  calibrate: bool = True) -> dict:
    """Generate the trade space, then calibrate it against measured response.

    The analytic sizing (3*dv*t in-track, dv/n cross-track) is a first-order
    estimate; the true response depends on the encounter geometry, which differs
    per strategy. So the family is flown once, each option is rescaled by the
    ratio of what it achieved to what it needed, and the whole family is flown
    again. Two full passes over the trade space -- which is precisely the work
    that justifies the hardware.
    """
    c = state.constraints
    # Aim just over the requirement, not well past it. Calibrating to a generous
    # margin spends propellant nobody asked for and -- worse -- can push an
    # option that already satisfied the constraint over the delta-v budget,
    # manufacturing an infeasible trade space out of a solvable one.
    target = target_miss_km or c.min_miss_km * 1.06

    cands = generate_candidates(state, tca_offset_s, target)
    first = evaluate_options(state, catalog_objects, cands, tca_offset_s)
    if not calibrate:
        first["passes"] = 1
        return first

    tuned = []
    for o in first["options"]:
        # An option that already clears is left alone. Re-sizing a working
        # solution can only cost fuel or break it.
        if o.feasible:
            tuned.append(Option(label=o.label, strategy=o.strategy,
                                direction_ric=o.direction_ric,
                                delta_v_mps=o.delta_v_mps,
                                burn_offset_s=o.burn_offset_s,
                                decision_time_s=o.burn_offset_s))
            continue
        achieved = max(o.miss_km, 1e-3)
        scale = float(np.clip(target / achieved, 0.4, 6.0))
        # Never let calibration invent a budget violation; propose at most the
        # budget and let the checks decide whether that is enough.
        dv = min(o.delta_v_mps * scale, c.dv_budget_mps)
        tuned.append(Option(label=o.label, strategy=o.strategy,
                            direction_ric=o.direction_ric,
                            delta_v_mps=round(dv, 4),
                            burn_offset_s=o.burn_offset_s,
                            decision_time_s=o.burn_offset_s))

    second = evaluate_options(state, catalog_objects, tuned, tca_offset_s)
    second["passes"] = 2
    second["states_evaluated"] += first["states_evaluated"]
    second["elapsed_s"] = round(second["elapsed_s"] + first["elapsed_s"], 2)
    second["calibration"] = [
        {"option": o.label, "first_pass_km": o.miss_km,
         "first_pass_dv": o.delta_v_mps} for o in first["options"]]
    return second


def _characterize(options: list[Option], c: Constraints) -> None:
    """Attach the trade-offs a human would actually weigh."""
    feas = [o for o in options if o.feasible]
    cheapest = min(feas, key=lambda o: o.delta_v_mps, default=None)
    safest = max(feas, key=lambda o: o.miss_km, default=None)
    latest = max(feas, key=lambda o: o.burn_offset_s, default=None)
    gentlest = min(feas, key=lambda o: o.altitude_drift_km, default=None)

    for o in options:
        o.pros, o.cons = [], []
        if o is cheapest:
            o.pros.append(f"Cheapest option — {o.delta_v_mps:.3f} m/s, "
                          f"{o.fuel_pct_of_budget:.0f}% of budget")
        if o is safest:
            o.pros.append(f"Largest clearance — {o.miss_km:.2f} km")
        if o is latest:
            o.pros.append(f"Latest commit — {o.burn_offset_s/60:.0f} min of decision time, "
                          f"most room for better tracking data")
        if o is gentlest:
            o.pros.append(f"Least mission impact — {o.altitude_drift_km:.2f} km of drift")
        if o.secondary_count == 0 and o.feasible:
            o.pros.append("Creates no new conjunctions over the next 12 h")
        if o.direction_ric[2]:
            o.pros.append("Holds altitude and along-track timing — ground-station "
                          "schedule unaffected")
        if o.direction_ric[1] and o.direction_ric[1] < 0:
            o.pros.append("Lowers the orbit — faster natural decay if the vehicle is lost")

        if o.burn_offset_s < 400:
            o.cons.append(f"Commits early — decision locked in "
                          f"{o.burn_offset_s/60:.0f} min from now")
        if o.fuel_pct_of_budget > 60:
            o.cons.append(f"Consumes {o.fuel_pct_of_budget:.0f}% of the maneuver budget")
        if o.altitude_drift_km > 0.5:
            o.cons.append(f"Moves the orbit {o.altitude_drift_km:.2f} km — "
                          f"station-keeping will have to correct it")
        if o.secondary_count:
            w = o.secondary_worst or {}
            o.cons.append(f"Creates {o.secondary_count} new close approach(es), "
                          f"worst {w.get('object','?')} at {w.get('miss_km','?')} km")
        for f in o.failed:
            o.cons.append(f"FAILS {f.replace('_', ' ')}")
        if not o.pros:
            o.pros.append("Clears the threat within all limits" if o.feasible
                          else "No advantage over the alternatives")
