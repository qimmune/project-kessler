"""Conjunction screening: coarse sweep over the catalog, then refinement to TCA.

The screen is O(N) -- one protected asset against every catalogued object -- not
the O(N^2) all-pairs problem. That is also how real operators do it: you screen
the assets you own, not the whole sky against itself.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .accel import to_device, to_host, xp
from .physics import teme_positions_many, teme_states_many, timescale, propagate, elements


@dataclass
class Conjunction:
    primary: str
    secondary: str
    secondary_index: int
    tca_iso: str
    tca_offset_s: float
    miss_km: float
    rel_speed_kms: float
    radial_km: float
    in_track_km: float
    cross_track_km: float
    pc: float

    def as_dict(self) -> dict:
        return asdict(self)


def _time_grid(t0, duration_s: float, step_s: float):
    ts = timescale()
    n = int(round(duration_s / step_s)) + 1
    return ts.tt_jd(t0.tt + np.arange(n) * (step_s / 86400.0))


def collision_probability(miss_km: float, sigma_km: float = 0.35,
                          combined_radius_m: float = 12.0) -> float:
    """Foster-style circular Pc for a spherical combined object.

    A real operator folds in the full 6x6 covariance from the tracking provider.
    Public TLEs carry no covariance, so `sigma_km` stands in for the position
    uncertainty -- documented as an assumption rather than hidden.
    """
    r_m = combined_radius_m
    s_m = sigma_km * 1000.0
    d_m = miss_km * 1000.0
    return float((r_m ** 2 / (2.0 * s_m ** 2)) * np.exp(-(d_m ** 2) / (2.0 * s_m ** 2)))


def refine_encounter(hero, obj, t0, centre_s: float, half_window_s: float,
                     refine_step_s: float = 0.5, secondary_index: int = -1) -> Conjunction | None:
    """Re-propagate one candidate pair around a coarse minimum to pin the true TCA.

    Returns None for co-orbital pairs -- docked modules, formation partners, or
    duplicate TLEs for one physical body. No closing velocity, no conjunction.
    """
    ts = timescale()
    lo = max(0.0, centre_s - half_window_s)
    hi = centre_s + half_window_s
    fine = ts.tt_jd(t0.tt + np.arange(lo, hi + refine_step_s, refine_step_s) / 86400.0)

    pair_r, pair_v = teme_states_many([hero, obj], fine)
    d = np.linalg.norm(pair_r[1] - pair_r[0], axis=1)
    if np.all(np.isnan(d)):
        return None
    j = int(np.nanargmin(d))
    if float(np.linalg.norm(pair_v[1, j] - pair_v[0, j])) < 0.05:
        return None

    rh, vh = pair_r[0, j], pair_v[0, j]
    rel = pair_r[1, j] - rh
    r_hat = rh / np.linalg.norm(rh)
    h = np.cross(rh, vh)
    c_hat = h / np.linalg.norm(h)
    i_hat = np.cross(c_hat, r_hat)
    miss = float(d[j])
    tca_off = lo + j * refine_step_s

    return Conjunction(
        primary=hero.name, secondary=obj.name, secondary_index=secondary_index,
        tca_iso=ts.tt_jd(t0.tt + tca_off / 86400.0).utc_iso(),
        tca_offset_s=float(tca_off), miss_km=miss,
        rel_speed_kms=float(np.linalg.norm(pair_v[1, j] - pair_v[0, j])),
        radial_km=float(rel @ r_hat), in_track_km=float(rel @ i_hat),
        cross_track_km=float(rel @ c_hat), pc=collision_probability(miss))


def screen(hero, catalog_objects, t0, horizon_s: float = 3 * 3600,
           coarse_step_s: float = 60.0, threshold_km: float = 5.0,
           refine_step_s: float = 0.5, exclude_names: tuple = ()) -> list[Conjunction]:
    """Screen `hero` against every object in `catalog_objects`.

    Two stages: a coarse sweep finds candidates within `threshold_km`, then each
    candidate is re-propagated at `refine_step_s` around its coarse minimum to
    pin the true time of closest approach.
    """
    ts = timescale()
    t_grid = _time_grid(t0, horizon_s, coarse_step_s)

    hero_r = teme_positions_many([hero], t_grid)[0]              # (T, 3)
    all_r = teme_positions_many(catalog_objects, t_grid)          # (N, T, 3)

    # The distance sweep runs on the accelerator when one is present.
    g_all = to_device(all_r)
    g_hero = to_device(hero_r)
    dist = xp.linalg.norm(g_all - g_hero[None, :, :], axis=2)      # (N, T)
    dist = xp.where(xp.isnan(dist), xp.inf, dist)
    min_d = to_host(dist.min(axis=1))
    dist = to_host(dist)

    hits = np.where(min_d < threshold_km)[0]

    out: list[Conjunction] = []
    for idx in hits:
        obj = catalog_objects[idx]
        if obj is hero or obj.name in exclude_names:
            continue
        k = int(np.argmin(dist[idx]))
        centre_s = k * coarse_step_s
        lo = max(0.0, centre_s - coarse_step_s)
        hi = min(horizon_s, centre_s + coarse_step_s)
        fine = ts.tt_jd(t0.tt + np.arange(lo, hi + refine_step_s, refine_step_s) / 86400.0)

        pair_r, pair_v = teme_states_many([hero, obj], fine)
        d = np.linalg.norm(pair_r[1] - pair_r[0], axis=1)
        if np.all(np.isnan(d)):
            continue
        j = int(np.nanargmin(d))
        miss = float(d[j])
        if miss >= threshold_km:
            continue
        # Co-orbital objects -- docked modules, formation partners, duplicate TLEs
        # for one physical body -- are not threats. Real screening drops them the
        # same way: no closing velocity, no conjunction.
        if float(np.linalg.norm(pair_v[1, j] - pair_v[0, j])) < 0.05:
            continue

        rh, vh = pair_r[0, j], pair_v[0, j]
        rs = pair_r[1, j]
        vs = pair_v[1, j]
        rel = rs - rh

        r_hat = rh / np.linalg.norm(rh)
        h = np.cross(rh, vh)
        c_hat = h / np.linalg.norm(h)
        i_hat = np.cross(c_hat, r_hat)

        tca_off = lo + j * refine_step_s
        out.append(Conjunction(
            primary=hero.name,
            secondary=obj.name,
            secondary_index=int(idx),
            tca_iso=ts.tt_jd(t0.tt + tca_off / 86400.0).utc_iso(),
            tca_offset_s=float(tca_off),
            miss_km=miss,
            rel_speed_kms=float(np.linalg.norm(vs - vh)),
            radial_km=float(np.dot(rel, r_hat)),
            in_track_km=float(np.dot(rel, i_hat)),
            cross_track_km=float(np.dot(rel, c_hat)),
            pc=collision_probability(miss),
        ))

    out.sort(key=lambda c: c.miss_km)
    return out


def screen_trajectory(hero_r0, hero_v0, catalog_objects, t0, horizon_s: float,
                      step_s: float = 30.0, threshold_km: float = 5.0) -> list[dict]:
    """Screen an *integrated* trajectory (i.e. a post-burn arc) against the catalog.

    The hero arc comes from the RK4+J2 integrator so a maneuver is represented
    exactly; catalogued objects still come from SGP4.
    """
    _, rs, _ = propagate(hero_r0, hero_v0, horizon_s, dt_s=step_s)
    t_grid = _time_grid(t0, horizon_s, step_s)
    n = min(rs.shape[1], len(t_grid))
    hero_track = rs[:, :n].T                                      # (T, 3)

    all_r = teme_positions_many(catalog_objects, t_grid)[:, :n, :]
    g_all, g_hero = to_device(all_r), to_device(hero_track)
    d = xp.linalg.norm(g_all - g_hero[None, :, :], axis=2)
    d = xp.where(xp.isnan(d), xp.inf, d)
    dist, min_d = to_host(d), to_host(d.min(axis=1))

    found = []
    for idx in np.where(min_d < threshold_km)[0]:
        k = int(np.argmin(dist[idx]))
        found.append({
            "object": catalog_objects[idx].name,
            "miss_km": float(min_d[idx]),
            "t_offset_s": float(k * step_s),
            "pc": collision_probability(float(min_d[idx])),
        })
    found.sort(key=lambda d: d["miss_km"])
    return found
