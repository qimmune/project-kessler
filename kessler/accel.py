"""GPU array backend for the screening hot loop.

Conjunction screening is one enormous elementwise distance computation over a
resident state matrix -- the shape of problem a GPU exists for. This module
selects CuPy when CUDA is present and falls back to NumPy otherwise, so the same
code path runs on a laptop and on a DGX Spark.

On GB10 the interesting part is what does NOT happen. Grace and Blackwell share
one physical address space, so moving the ephemeris from the SGP4 propagator to
the screening kernel is a pointer handoff rather than a PCIe copy. `xp_asarray`
is where that transfer would otherwise show up, and it is instrumented so the
demo can report the real number instead of asserting it.
"""
from __future__ import annotations

import os
import time

import numpy as np

_FORCE_CPU = os.environ.get("KESSLER_FORCE_CPU", "").lower() in ("1", "true", "yes")

try:  # pragma: no cover - depends on the machine
    if _FORCE_CPU:
        raise ImportError("KESSLER_FORCE_CPU set")
    import cupy as _cp

    _cp.cuda.runtime.getDeviceCount()
    xp = _cp
    GPU = True
except Exception:
    xp = np
    GPU = False


class Transfer:
    """Counts host->device movement so the unified-memory claim is measurable."""
    bytes_moved = 0
    seconds = 0.0
    calls = 0

    @classmethod
    def reset(cls):
        cls.bytes_moved = cls.seconds = 0.0
        cls.bytes_moved = 0
        cls.calls = 0

    @classmethod
    def report(cls) -> dict:
        return {"backend": backend_name(), "gpu": GPU, "calls": cls.calls,
                "bytes": int(cls.bytes_moved), "seconds": round(cls.seconds, 4)}


def backend_name() -> str:
    if not GPU:
        return "numpy (CPU)"
    try:
        props = xp.cuda.runtime.getDeviceProperties(0)
        return f"cupy on {props['name'].decode()}"
    except Exception:
        return "cupy (CUDA)"


def device_info() -> dict:
    info = {"backend": backend_name(), "gpu": GPU, "unified_memory": False}
    if not GPU:
        return info
    try:
        p = xp.cuda.runtime.getDeviceProperties(0)
        free, total = xp.cuda.runtime.memGetInfo()
        info.update(name=p["name"].decode(), total_gb=round(total / 1e9, 1),
                    free_gb=round(free / 1e9, 1),
                    # Grace Blackwell and other integrated parts report this;
                    # on a discrete card it stays false and the copy is real.
                    unified_memory=bool(p.get("unifiedAddressing", 0))
                    and bool(p.get("integrated", 0)))
    except Exception:
        pass
    return info


def to_device(a):
    """Move an array to the compute device, timing and sizing the transfer."""
    if not GPU:
        return a
    t = time.perf_counter()
    out = xp.asarray(a)
    xp.cuda.runtime.deviceSynchronize()
    Transfer.calls += 1
    Transfer.bytes_moved += int(getattr(a, "nbytes", 0))
    Transfer.seconds += time.perf_counter() - t
    return out


def free_pool() -> None:
    """Release cached device blocks between passes so peak stays predictable."""
    if not GPU:
        return
    try:
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def memory_report() -> dict:
    if not GPU:
        return {"gpu": False}
    try:
        pool = xp.get_default_memory_pool()
        free, total = xp.cuda.runtime.memGetInfo()
        return {"gpu": True, "pool_used_gb": round(pool.used_bytes() / 1e9, 3),
                "device_free_gb": round(free / 1e9, 2),
                "device_total_gb": round(total / 1e9, 2)}
    except Exception:
        return {"gpu": True}


def to_host(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)
