"""Fleet-wide continuous monitoring.

The whole catalog is propagated once into a single resident array, then every
protected asset is screened against that same array. Adding assets costs one
extra track each, not another catalog sweep -- which is the entire argument for
holding the state matrix in memory instead of streaming it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .conjunction import Conjunction, refine_encounter
from .mission import Constraints, requires_action
from .physics import elements, teme_positions_many, teme_state, timescale


@dataclass
class AssetStatus:
    name: str
    alt_km: float
    worst: Conjunction | None = None
    action_required: bool = False
    reason: str = "clear"
    candidates: int = 0

    @property
    def severity(self) -> str:
        if self.action_required:
            return "ACTION"
        if self.worst is not None:
            return "WATCH"
        return "CLEAR"


@dataclass
class SweepResult:
    statuses: list[AssetStatus]
    t0: object
    catalog_size: int
    epochs: int
    states: int
    matrix_mb: float
    elapsed_s: float
    scanned_at: float = field(default_factory=time.time)

    @property
    def actionable(self) -> list[AssetStatus]:
        return [s for s in self.statuses if s.action_required]

    @property
    def watching(self) -> list[AssetStatus]:
        return [s for s in self.statuses if s.severity == "WATCH"]


def select_fleet(catalog_objects, pattern: str = "STARLINK", limit: int = 12) -> list:
    """Pick the protected assets by name match."""
    pattern = pattern.upper().strip()
    hits = [s for s in catalog_objects if pattern in s.name.upper()] if pattern else list(catalog_objects)
    return hits[:limit]


def sweep_fleet(assets, catalog_objects, t0, horizon_s: float = 6 * 3600,
                coarse_step_s: float = 60.0, threshold_km: float = 25.0,
                constraints: Constraints | None = None,
                progress=None) -> SweepResult:
    """Screen every asset against the whole catalog off one propagation."""
    c = constraints or Constraints()
    ts = timescale()
    started = time.time()

    n_epochs = int(round(horizon_s / coarse_step_s)) + 1
    t_grid = ts.tt_jd(t0.tt + np.arange(n_epochs) * (coarse_step_s / 86400.0))

    # ---- the one propagation everything else reads ----
    all_r = teme_positions_many(catalog_objects, t_grid)            # (N, T, 3)
    matrix_mb = all_r.nbytes / 1e6

    asset_names = {id(a) for a in assets}
    statuses: list[AssetStatus] = []

    for k, asset in enumerate(assets):
        if progress:
            progress(k, len(assets), asset.name)
        try:
            r0, v0 = teme_state(asset, t0)
            alt = elements(r0, v0)["alt_km"]
        except Exception:
            continue

        hero_r = teme_positions_many([asset], t_grid)[0]            # (T, 3)
        dist = np.linalg.norm(all_r - hero_r[None, :, :], axis=2)   # (N, T)
        dist = np.where(np.isnan(dist), np.inf, dist)
        min_d = dist.min(axis=1)

        st = AssetStatus(name=asset.name, alt_km=alt)
        order = np.argsort(min_d)
        for idx in order[:12]:
            if not np.isfinite(min_d[idx]) or min_d[idx] >= threshold_km:
                break
            obj = catalog_objects[int(idx)]
            if obj is asset or id(obj) in asset_names and obj.name == asset.name:
                continue
            cj = refine_encounter(asset, obj, t0,
                                  centre_s=float(np.argmin(dist[idx]) * coarse_step_s),
                                  half_window_s=coarse_step_s,
                                  secondary_index=int(idx))
            if cj is None or cj.miss_km >= threshold_km:
                continue
            st.candidates += 1
            if st.worst is None or cj.miss_km < st.worst.miss_km:
                st.worst = cj

        if st.worst is not None:
            act, why = requires_action(
                {"miss_km": st.worst.miss_km, "pc": st.worst.pc}, c)
            st.action_required, st.reason = act, why
        statuses.append(st)

    statuses.sort(key=lambda s: (not s.action_required,
                                 s.worst.miss_km if s.worst else 1e9))
    return SweepResult(statuses=statuses, t0=t0, catalog_size=len(catalog_objects),
                       epochs=n_epochs, states=all_r.shape[0] * all_r.shape[1],
                       matrix_mb=matrix_mb, elapsed_s=time.time() - started)
