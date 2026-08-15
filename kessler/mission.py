"""Mission state, the seeded threat, and the maneuver evaluator the critic calls.

`evaluate_maneuver` is the only place a proposed burn is judged. It is a plain
Python function -- the critic agent reaches it through tool calling, so the
verdict comes from the physics engine, never from the model's opinion.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .physics import (timescale, teme_state, propagate, apply_burn, elements,
                      ric_basis, R_EARTH)
from .conjunction import collision_probability, screen_trajectory

MU_KM3 = 398600.4418


# ----------------------------------------------------------------------
# operating constraints -- the rules the critic enforces
# ----------------------------------------------------------------------
@dataclass
class Constraints:
    dv_budget_mps: float = 0.35          # per-maneuver delta-v ceiling
    min_miss_km: float = 2.0             # required separation after the burn
    altitude_box_km: float = 3.0         # allowed drift from nominal mean altitude
    secondary_screen_km: float = 1.5     # a burn may not create a new close approach
    secondary_horizon_s: float = 12 * 3600
    max_pc: float = 1e-4                 # operator action threshold


@dataclass
class MissionState:
    hero_name: str
    hero_r0: np.ndarray
    hero_v0: np.ndarray
    threat_name: str
    threat_r0: np.ndarray
    threat_v0: np.ndarray
    t0: object
    nominal_alt_km: float
    propellant_pct: float = 11.8
    constraints: Constraints = field(default_factory=Constraints)

    def summary(self) -> dict:
        el = elements(self.hero_r0, self.hero_v0)
        return {
            "vehicle": self.hero_name,
            "altitude_km": round(el["alt_km"], 2),
            "inclination_deg": round(el["inc_deg"], 3),
            "period_min": round(el["period_min"], 2),
            "propellant_pct": self.propellant_pct,
        }


# ----------------------------------------------------------------------
# threat synthesis
# ----------------------------------------------------------------------
def synthesize_threat(hero_r0, hero_v0, tca_s: float, miss_km: float = 0.4,
                      plane_angle_deg: float = 100.0, name: str = "COSMOS-1408 DEB"):
    """Build a debris object that will pass `miss_km` from the hero at `tca_s`.

    Seeded, and labelled as such. The object is a real physical trajectory -- it
    is back-propagated from the encounter through the same integrator that
    propagates everything else, so the detector finds it exactly the way it
    would find a natural one. Nothing about the encounter geometry is hardcoded
    downstream; the engine measures it.
    """
    _, rs, vs = propagate(hero_r0, hero_v0, tca_s, dt_s=2.0)
    r_tca, v_tca = rs[:, -1], vs[:, -1]

    r_hat = r_tca / np.linalg.norm(r_tca)
    theta = np.radians(plane_angle_deg)
    # Rodrigues rotation of the velocity about the radial axis -> plane crossing
    v_threat = (v_tca * np.cos(theta)
                + np.cross(r_hat, v_tca) * np.sin(theta)
                + r_hat * np.dot(r_hat, v_tca) * (1 - np.cos(theta)))

    rel_v = v_threat - v_tca
    # Offset perpendicular to the relative velocity: displacing along it would
    # only shift the encounter in time, not change the miss distance.
    offset_dir = np.cross(rel_v, r_hat)
    offset_dir /= np.linalg.norm(offset_dir)
    r_threat = r_tca + offset_dir * miss_km

    # back-propagate to t0
    _, rb, vb = propagate(r_threat, v_threat, -tca_s, dt_s=2.0)
    return name, rb[:, -1], vb[:, -1]


def find_tca_window(r_a, v_a, r_b, v_b, centre_s: float, half_window_s: float,
                    coarse_dt: float = 2.0, fine_dt: float = 0.05) -> dict:
    """Closest approach inside a bracket around a known encounter time.

    Two objects on intersecting orbits re-approach roughly once per revolution.
    Searching an open-ended horizon will happily return the NEXT crossing and
    report a successful avoidance as a failure, so evaluating a maneuver against
    a specific conjunction has to stay bracketed to that conjunction.
    """
    lead = max(0.0, centre_s - half_window_s)
    _, ra0, va0 = propagate(r_a, v_a, lead, dt_s=coarse_dt)
    _, rb0, vb0 = propagate(r_b, v_b, lead, dt_s=coarse_dt)
    span = 2.0 * half_window_s
    out = find_tca(ra0[:, -1], va0[:, -1], rb0[:, -1], vb0[:, -1],
                   horizon_s=span, coarse_dt=coarse_dt, fine_dt=fine_dt)
    out["tca_offset_s"] += lead
    return out


def find_tca(r_a, v_a, r_b, v_b, horizon_s: float, coarse_dt: float = 5.0,
             fine_dt: float = 0.05) -> dict:
    """Closest approach between two integrated trajectories over a whole horizon."""
    _, ra, va = propagate(r_a, v_a, horizon_s, dt_s=coarse_dt)
    _, rb, vb = propagate(r_b, v_b, horizon_s, dt_s=coarse_dt)
    d = np.linalg.norm(rb - ra, axis=0)
    k = int(np.argmin(d))

    lo = max(0.0, (k - 1) * coarse_dt)
    hi = min(horizon_s, (k + 1) * coarse_dt)
    span = hi - lo
    _, ra0, va0 = propagate(r_a, v_a, lo, dt_s=coarse_dt)
    _, rb0, vb0 = propagate(r_b, v_b, lo, dt_s=coarse_dt)
    _, ra2, va2 = propagate(ra0[:, -1], va0[:, -1], span, dt_s=fine_dt)
    _, rb2, vb2 = propagate(rb0[:, -1], vb0[:, -1], span, dt_s=fine_dt)
    d2 = np.linalg.norm(rb2 - ra2, axis=0)
    j = int(np.argmin(d2))

    rh, vh = ra2[:, j], va2[:, j]
    rel = rb2[:, j] - rh
    basis = ric_basis(rh, vh)
    miss = float(d2[j])
    return {
        "tca_offset_s": float(lo + j * fine_dt),
        "miss_km": miss,
        "rel_speed_kms": float(np.linalg.norm(vb2[:, j] - vh)),
        "radial_km": float(rel @ basis[:, 0]),
        "in_track_km": float(rel @ basis[:, 1]),
        "cross_track_km": float(rel @ basis[:, 2]),
        "pc": collision_probability(miss),
    }


def requires_action(encounter: dict, c: Constraints) -> tuple[bool, str]:
    """Does this conjunction actually warrant burning propellant?

    Detecting a close approach is not the same as needing to dodge it. Operators
    maneuver when the geometry breaches their separation minimum or the
    probability of collision crosses the action threshold -- otherwise they log
    it and keep watching. Without this gate the system spends real delta-v
    "improving" an encounter that was already safe.
    """
    breaches = []
    if encounter["miss_km"] < c.min_miss_km:
        breaches.append(f"miss {encounter['miss_km']:.3f} km below the "
                        f"{c.min_miss_km:.1f} km separation minimum")
    if encounter["pc"] > c.max_pc:
        breaches.append(f"Pc {encounter['pc']:.2e} above the {c.max_pc:.0e} action threshold")
    if breaches:
        return True, " and ".join(breaches)
    return False, (f"miss {encounter['miss_km']:.3f} km and Pc {encounter['pc']:.2e} are both "
                   f"within limits")


def altitude_shortlist(catalog_objects, hero_elements: dict, pad_km: float = 50.0) -> list:
    """Drop objects whose altitude band cannot reach the hero's.

    A radial pre-filter is the standard first cut in operational screening: an
    object with a 900 km perigee can never come within kilometres of a 390 km
    orbit, so propagating it is wasted work. Typically trims the catalog by an
    order of magnitude before the expensive pass.
    """
    lo = hero_elements["perigee_alt_km"] - pad_km
    hi = hero_elements["apogee_alt_km"] + pad_km
    keep = []
    for s in catalog_objects:
        m = s.model
        try:
            a_km = (MU_KM3 / (m.no_kozai * 60.0 / (2 * np.pi) * 2 * np.pi / 60.0) ** 2) ** (1 / 3)
            n_rad_s = m.no_kozai / 60.0
            a_km = (MU_KM3 / (n_rad_s ** 2)) ** (1 / 3)
            e = m.ecco
            per = a_km * (1 - e) - R_EARTH
            apo = a_km * (1 + e) - R_EARTH
        except Exception:
            continue
        if apo >= lo and per <= hi:
            keep.append(s)
    return keep


# ----------------------------------------------------------------------
# the tool the critic agent calls
# ----------------------------------------------------------------------
def evaluate_maneuver(state: MissionState, catalog_objects,
                      direction_ric, delta_v_mps: float, burn_offset_s: float,
                      tca_offset_s: float) -> dict:
    """Fly the proposed burn and report what actually happens.

    Returns a verdict dict with every check the critic needs. This is physics,
    not prose -- if the burn is unsafe the numbers say so.
    """
    c = state.constraints
    checks: list[dict] = []

    d = np.asarray(direction_ric, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0:
        return {"valid": False, "error": "direction_ric is a zero vector"}
    d = d / norm

    if burn_offset_s >= tca_offset_s:
        return {"valid": False,
                "error": f"burn at T+{burn_offset_s:.0f}s is after TCA at T+{tca_offset_s:.0f}s"}

    # --- fly to the burn point, apply the impulse ---
    _, rs, vs = propagate(state.hero_r0, state.hero_v0, burn_offset_s, dt_s=2.0)
    r_burn, v_burn = rs[:, -1], vs[:, -1]
    v_post = apply_burn(r_burn, v_burn, d * delta_v_mps)

    # --- new geometry against the threat ---
    _, tr, tv = propagate(state.threat_r0, state.threat_v0, burn_offset_s, dt_s=2.0)
    remaining = tca_offset_s - burn_offset_s
    # Bracket the search to the conjunction we are actually avoiding.
    enc = find_tca_window(r_burn, v_post, tr[:, -1], tv[:, -1],
                          centre_s=remaining, half_window_s=300.0)

    checks.append({
        "check": "primary_threat_cleared",
        "pass": bool(enc["miss_km"] >= c.min_miss_km and enc["pc"] <= c.max_pc),
        "detail": f"miss {enc['miss_km']:.3f} km (needs >= {c.min_miss_km}), "
                  f"Pc {enc['pc']:.2e} (needs <= {c.max_pc:.0e})",
    })

    # --- delta-v budget ---
    checks.append({
        "check": "delta_v_within_budget",
        "pass": bool(delta_v_mps <= c.dv_budget_mps),
        "detail": f"{delta_v_mps:.4f} m/s of {c.dv_budget_mps} m/s ceiling",
    })

    # --- mission altitude box ---
    # Baseline is the *unburned* orbit evaluated at the burn point. Comparing
    # against elements taken at t0 would fold in J2's osculating-element
    # oscillation and reject perfectly good burns.
    el_ref = elements(r_burn, v_burn)
    el_new = elements(r_burn, v_post)
    mean_ref = (el_ref["perigee_alt_km"] + el_ref["apogee_alt_km"]) / 2.0
    mean_alt = (el_new["perigee_alt_km"] + el_new["apogee_alt_km"]) / 2.0
    drift = abs(mean_alt - mean_ref)
    checks.append({
        "check": "altitude_box_held",
        "pass": bool(drift <= c.altitude_box_km),
        "detail": f"burn moved mean altitude {drift:.3f} km of {c.altitude_box_km} km allowed "
                  f"({mean_ref:.2f} -> {mean_alt:.2f} km)",
    })

    # --- does the new orbit hit anything else? ---
    t_burn = timescale().tt_jd(state.t0.tt + burn_offset_s / 86400.0)
    shortlist = altitude_shortlist(catalog_objects, el_new, pad_km=50.0)
    secondary = screen_trajectory(r_burn, v_post, shortlist, t_burn,
                                  horizon_s=c.secondary_horizon_s, step_s=30.0,
                                  threshold_km=c.secondary_screen_km)
    secondary = [s for s in secondary if s["object"] != state.hero_name]
    checks.append({
        "check": "no_new_conjunctions",
        "pass": len(secondary) == 0,
        "detail": (f"{len(secondary)} new approach(es) under {c.secondary_screen_km} km "
                   f"over {c.secondary_horizon_s/3600:.0f} h"
                   + (f": {secondary[0]['object']} at {secondary[0]['miss_km']:.2f} km"
                      if secondary else "")),
    })

    approved = all(ch["pass"] for ch in checks)
    return {
        "valid": True,
        "approved": approved,
        "checks": checks,
        "new_miss_km": round(enc["miss_km"], 4),
        "new_pc": enc["pc"],
        "rel_speed_kms": round(enc["rel_speed_kms"], 3),
        "delta_v_mps": round(float(delta_v_mps), 4),
        "burn_offset_s": float(burn_offset_s),
        "mean_altitude_km": round(mean_alt, 3),
        "altitude_drift_km": round(drift, 3),
        "secondary_conjunctions": secondary[:5],
        "failed_checks": [ch["check"] for ch in checks if not ch["pass"]],
    }
