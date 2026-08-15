"""Orbital state, propagation, and maneuver application.

Two propagators, used deliberately:

* SGP4 (via skyfield) drives catalog-wide screening. It is the model the TLEs are
  defined against, so it is the only correct choice for real objects.
* An RK4 two-body + J2 integrator drives maneuver evaluation. A burn changes the
  osculating state, and round-tripping that back through SGP4 mean elements
  injects conversion error of the same order as the miss distances we are
  arguing about. Integrating the pre-burn and post-burn arcs identically keeps
  the before/after comparison honest.
"""
from __future__ import annotations

import numpy as np
from skyfield.api import EarthSatellite, load

MU = 398600.4418          # km^3 / s^2
R_EARTH = 6378.137        # km
J2 = 1.08262668e-3

_TS = load.timescale()


def timescale():
    return _TS


# ----------------------------------------------------------------------
# state extraction
# ----------------------------------------------------------------------
def state_at(sat: EarthSatellite, t) -> tuple[np.ndarray, np.ndarray]:
    """Geocentric position (km) and velocity (km/s) of `sat` at time(s) `t`."""
    g = sat.at(t)
    return np.asarray(g.position.km, dtype=float), np.asarray(g.velocity.km_per_s, dtype=float)


# ----------------------------------------------------------------------
# TEME working frame
#
# Everything in the screening path stays in TEME -- the frame SGP4 natively
# outputs. Relative geometry between two objects at the same instant is
# identical in TEME and GCRS (they differ by a shared rotation), so skipping the
# conversion costs nothing and removes a whole class of frame bugs. The RK4
# integrator is seeded from TEME states, so the two propagators agree.
# ----------------------------------------------------------------------
def teme_state(sat: EarthSatellite, t) -> tuple[np.ndarray, np.ndarray]:
    """TEME position (km) and velocity (km/s) at a single skyfield Time.

    SGP4 wants UT1, not TAI or TT -- feeding it TAI puts every object ~133 km
    down-track, which silently corrupts every distance in the pipeline.
    """
    err, r, v = sat.model.sgp4(t.whole, t.ut1_fraction)
    if err != 0:
        raise RuntimeError(f"SGP4 error {err} for {sat.name}")
    return np.array(r, dtype=float), np.array(v, dtype=float)


def teme_positions_many(sats, t_array) -> np.ndarray:
    """(n_sats, n_times, 3) TEME positions in km, propagated in C by SatrecArray.

    Objects whose propagation fails (decayed, malformed TLE) come back as NaN so
    callers can drop them without the loop stopping.
    """
    from sgp4.api import SatrecArray

    arr = SatrecArray([s.model for s in sats])
    jd = np.asarray(t_array.whole, dtype=float)
    fr = np.asarray(t_array.ut1_fraction, dtype=float)
    err, r, v = arr.sgp4(jd, fr)
    r = np.asarray(r, dtype=float)
    r[err != 0] = np.nan
    return r


def teme_states_many(sats, t_array):
    """(n_sats, n_times, 3) TEME positions AND velocities, both km-based."""
    from sgp4.api import SatrecArray

    arr = SatrecArray([s.model for s in sats])
    jd = np.asarray(t_array.whole, dtype=float)
    fr = np.asarray(t_array.ut1_fraction, dtype=float)
    err, r, v = arr.sgp4(jd, fr)
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    bad = err != 0
    r[bad] = np.nan
    v[bad] = np.nan
    return r, v


def teme_track(sat: EarthSatellite, t_array) -> np.ndarray:
    """(3, N) TEME positions in km for a single object."""
    return teme_positions_many([sat], t_array)[0].T


# ----------------------------------------------------------------------
# RIC frame  (radial / in-track / cross-track)
# ----------------------------------------------------------------------
def ric_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Columns are the RIC unit vectors expressed in the inertial frame."""
    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    c_hat = h / np.linalg.norm(h)          # cross-track (orbit normal)
    i_hat = np.cross(c_hat, r_hat)         # in-track (completes right-handed set)
    return np.column_stack([r_hat, i_hat, c_hat])


def ric_to_inertial(r: np.ndarray, v: np.ndarray, dv_ric: np.ndarray) -> np.ndarray:
    return ric_basis(r, v) @ np.asarray(dv_ric, dtype=float)


def inertial_to_ric(r: np.ndarray, v: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return ric_basis(r, v).T @ np.asarray(vec, dtype=float)


# ----------------------------------------------------------------------
# RK4 two-body + J2
# ----------------------------------------------------------------------
def _accel(r: np.ndarray) -> np.ndarray:
    x, y, z = r
    rn = np.sqrt(x * x + y * y + z * z)
    two_body = -MU * r / rn ** 3
    k = -1.5 * J2 * MU * R_EARTH ** 2 / rn ** 5
    zr2 = 5.0 * z * z / (rn * rn)
    j2 = k * np.array([x * (1.0 - zr2), y * (1.0 - zr2), z * (3.0 - zr2)])
    return two_body + j2


def _deriv(y: np.ndarray) -> np.ndarray:
    return np.concatenate([y[3:], _accel(y[:3])])


def propagate(r0: np.ndarray, v0: np.ndarray, duration_s: float, dt_s: float = 5.0):
    """Integrate a single state forward. Returns (times_s, positions (3,N), velocities (3,N))."""
    n = max(1, int(round(abs(duration_s) / dt_s)))
    step = duration_s / n
    y = np.concatenate([np.asarray(r0, float), np.asarray(v0, float)])
    ts_out = np.empty(n + 1)
    rs = np.empty((3, n + 1))
    vs = np.empty((3, n + 1))
    ts_out[0], rs[:, 0], vs[:, 0] = 0.0, y[:3], y[3:]
    for i in range(1, n + 1):
        k1 = _deriv(y)
        k2 = _deriv(y + 0.5 * step * k1)
        k3 = _deriv(y + 0.5 * step * k2)
        k4 = _deriv(y + step * k3)
        y = y + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ts_out[i], rs[:, i], vs[:, i] = i * step, y[:3], y[3:]
    return ts_out, rs, vs


def apply_burn(r: np.ndarray, v: np.ndarray, dv_ric_mps: np.ndarray) -> np.ndarray:
    """Return the post-burn velocity. Delta-v arrives in m/s in the RIC frame."""
    dv_kms = np.asarray(dv_ric_mps, dtype=float) / 1000.0
    return v + ric_to_inertial(r, v, dv_kms)


# ----------------------------------------------------------------------
# classical elements (for reporting and the altitude-box constraint)
# ----------------------------------------------------------------------
def elements(r: np.ndarray, v: np.ndarray) -> dict:
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    energy = vn * vn / 2.0 - MU / rn
    a = -MU / (2.0 * energy)
    e_vec = (np.cross(v, h) / MU) - r / rn
    e = float(np.linalg.norm(e_vec))
    inc = float(np.degrees(np.arccos(np.clip(h[2] / hn, -1.0, 1.0))))
    return {
        "sma_km": float(a),
        "ecc": e,
        "inc_deg": inc,
        "perigee_alt_km": float(a * (1 - e) - R_EARTH),
        "apogee_alt_km": float(a * (1 + e) - R_EARTH),
        "period_min": float(2 * np.pi * np.sqrt(a ** 3 / MU) / 60.0),
        "alt_km": float(rn - R_EARTH),
        "speed_kms": float(vn),
    }
