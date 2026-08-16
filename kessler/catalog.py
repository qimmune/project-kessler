"""TLE ingest from CelesTrak, with on-disk caching so the demo never depends on wifi."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from skyfield.api import load, EarthSatellite

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_GP = "https://celestrak.org/NORAD/elements/gp.php?GROUP={}&FORMAT=tle"

GROUPS = {
    "active":            _GP.format("active"),
    "starlink":          _GP.format("starlink"),
    # The four big fragmentation events. "active" contains payloads only, so
    # without these the sky looks deceptively clean -- debris is most of the
    # actual population and all of the actual hazard.
    "cosmos-1408-debris": _GP.format("cosmos-1408-debris"),
    "fengyun-1c-debris":  _GP.format("fengyun-1c-debris"),
    "iridium-33-debris":  _GP.format("iridium-33-debris"),
    "cosmos-2251-debris": _GP.format("cosmos-2251-debris"),
}

DEBRIS_GROUPS = ("cosmos-1408-debris", "fengyun-1c-debris",
                 "iridium-33-debris", "cosmos-2251-debris")

MAX_CACHE_AGE_S = 12 * 3600


@dataclass
class Catalog:
    objects: list[EarthSatellite]
    source: str
    fetched_at: float

    def __len__(self) -> int:
        return len(self.objects)

    def by_name(self, needle: str) -> EarthSatellite | None:
        needle = needle.upper()
        for s in self.objects:
            if needle in s.name.upper():
                return s
        return None


def classify(name: str) -> str:
    """Object class from the catalog name, the way trackers label them.

    CelesTrak encodes it in the name: "DEB" for fragmentation debris, "R/B" for
    spent rocket bodies, everything else is an active or defunct payload.
    """
    n = name.upper()
    if " DEB" in n or n.endswith("DEB") or "DEBRIS" in n:
        return "debris"
    if "R/B" in n or "ROCKET BODY" in n:
        return "rocket_body"
    return "payload"


def _cache_path(group: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{group}.tle")


def _cache_is_fresh(path: str) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < MAX_CACHE_AGE_S


def load_group(group: str = "active", reload: bool = False) -> Catalog:
    """Load one CelesTrak group. Falls back to the cached copy if the network is down."""
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; options: {sorted(GROUPS)}")
    path = _cache_path(group)
    want_download = reload or not _cache_is_fresh(path)

    try:
        sats = load.tle_file(GROUPS[group], filename=path, reload=want_download)
        source = "celestrak" if want_download else "cache"
    except Exception as exc:  # offline, DNS blocked, CelesTrak rate-limited
        if not os.path.exists(path):
            raise RuntimeError(
                f"no network and no cached copy of {group!r} at {path}. "
                f"Run once with connectivity to seed the cache."
            ) from exc
        sats = load.tle_file(path)
        source = f"cache (offline: {type(exc).__name__})"

    return Catalog(objects=sats, source=source, fetched_at=os.path.getmtime(path))


def load_full_catalog(reload: bool = False, include_debris: bool = True) -> Catalog:
    """Active payloads plus the major debris clouds, de-duplicated by NORAD id."""
    cat = load_group("active", reload=reload)
    objs = list(cat.objects)
    seen = {s.model.satnum for s in objs}
    sources = [cat.source]
    if include_debris:
        for g in DEBRIS_GROUPS:
            try:
                extra = load_group(g, reload=reload)
            except Exception:
                continue
            for s in extra.objects:
                if s.model.satnum not in seen:
                    seen.add(s.model.satnum)
                    objs.append(s)
            sources.append(g)
    return Catalog(objects=objs, source=" + ".join(sources[:1] + [f"{len(sources)-1} debris sets"]),
                   fetched_at=cat.fetched_at)


def load_demo_catalog(limit: int | None = 2500, reload: bool = False) -> Catalog:
    """The catalog the demo screens against.

    `limit` trims the object count so a full sweep stays interactive. Screening is
    O(N) against one protected asset, so this is a wall-clock knob, not a
    correctness one -- raise it to the full catalog for the 'we held it all in
    memory' story.
    """
    cat = load_full_catalog(reload=reload)
    objs = cat.objects if limit is None else cat.objects[:limit]
    return Catalog(objects=objs, source=cat.source, fetched_at=cat.fetched_at)


if __name__ == "__main__":
    c = load_group("active")
    print(f"{len(c)} objects  (source: {c.source})")
    for s in c.objects[:5]:
        print(f"  {s.model.satnum:>6}  {s.name}")
