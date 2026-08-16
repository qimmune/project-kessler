"""Modeled background debris environment.

The tracked catalogue is a small fraction of what is actually up there. ESA's
figures: roughly 40,000 objects larger than 10 cm are tracked, about 1.1 million
between 1 and 10 cm are not, and on the order of 130 million fragments larger
than 1 mm exist and are inferred statistically rather than observed.

Nobody has state vectors for that population, so it cannot be propagated -- but
it can be sampled from the orbital distribution the fragments are known to
occupy. That is what ESA MASTER and NASA ORDEM do, and it is what this module
does: a representative sample of the untracked environment, drawn from the
altitude and inclination structure that real breakup events produced.

Anything from here is clearly labelled as modeled. It is context, never an
input to conjunction screening -- the engine only ever screens real objects
with real TLEs.
"""
from __future__ import annotations

import numpy as np

from .physics import R_EARTH

# Altitude shells where the debris environment actually concentrates, in km.
# The ~800 km peak is the Iridium-Cosmos collision plus decades of sun-synchronous
# traffic; ~850-1000 km carries Fengyun-1C; the 500-600 km band is Starlink-era
# and decays fastest.
SHELLS = (
    # (mean_alt, sigma, weight, mean_inclination_deg, inc_sigma)
    (780.0,  55.0, 0.30, 86.4, 6.0),
    (850.0,  45.0, 0.22, 98.8, 2.5),
    (975.0,  60.0, 0.16, 82.5, 3.0),
    (630.0,  50.0, 0.14, 97.8, 2.0),
    (545.0,  40.0, 0.10, 53.2, 3.5),
    (1420.0, 90.0, 0.08, 74.0, 8.0),
)


def sample_environment(n: int = 45000, seed: int = 12) -> np.ndarray:
    """(n, 3) inertial positions sampled from the modeled debris environment."""
    rng = np.random.default_rng(seed)
    weights = np.array([s[2] for s in SHELLS], dtype=float)
    weights /= weights.sum()
    counts = rng.multinomial(n, weights)

    out = []
    for (alt, sig, _, inc_mu, inc_sig), k in zip(SHELLS, counts):
        if k == 0:
            continue
        r = R_EARTH + np.abs(rng.normal(alt, sig, k))
        inc = np.radians(np.abs(rng.normal(inc_mu, inc_sig, k)))
        raan = rng.uniform(0, 2 * np.pi, k)
        nu = rng.uniform(0, 2 * np.pi, k)

        ci, si = np.cos(inc), np.sin(inc)
        cr, sr = np.cos(raan), np.sin(raan)
        cn, sn = np.cos(nu), np.sin(nu)
        out.append(np.column_stack([
            r * (cr * cn - sr * sn * ci),
            r * (sr * cn + cr * sn * ci),
            r * (sn * si),
        ]))
    return np.vstack(out) if out else np.empty((0, 3))


# Population figures, for labelling. Sources: ESA Space Environment Report.
POPULATION = {
    "tracked_10cm": 40_000,
    "untracked_1_10cm": 1_100_000,
    "untracked_1mm_1cm": 130_000_000,
}
TOTAL_OVER_1MM = (POPULATION["tracked_10cm"]
                  + POPULATION["untracked_1_10cm"]
                  + POPULATION["untracked_1mm_1cm"])
