"""Exercise the CuPy code path without a GPU.

The accelerated paths in this repo have never run on CUDA. The bug that shows up
first on a real device is not a missing kernel -- it is silently mixing a device
array with a host array, which NumPy happily broadcasts and CuPy refuses.

This substitutes a strict stand-in for CuPy: arrays are a distinct type, any
operation that mixes them with a plain ndarray raises the way CuPy does, and
getting data back to the host requires an explicit asnumpy(). If the pipeline
runs clean through this, the real device path is very likely correct too.
"""
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class DeviceArray(np.ndarray):
    """Stands in for cupy.ndarray. Refuses to interoperate with host arrays."""

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        for x in inputs:
            if isinstance(x, np.ndarray) and not isinstance(x, DeviceArray):
                raise TypeError(
                    f"Unsupported type <class 'numpy.ndarray'> in {ufunc.__name__} — "
                    f"a host array reached a device operation. In CuPy this is a "
                    f"hard error, not a broadcast.")
        # Drop to plain ndarray to do the work, then re-tag the result as device
        # memory. Calling super() with subclass inputs returns NotImplemented.
        plain = [np.asarray(x).view(np.ndarray) if isinstance(x, DeviceArray) else x
                 for x in inputs]
        if "out" in kw:
            kw["out"] = tuple(np.asarray(o).view(np.ndarray) if isinstance(o, DeviceArray)
                              else o for o in kw["out"])
        out = getattr(ufunc, method)(*plain, **kw)
        if isinstance(out, np.ndarray):
            out = out.view(DeviceArray)
        return out

    def get(self):
        return np.asarray(self).view(np.ndarray)


def _wrap(a):
    return np.asarray(a).view(DeviceArray)


def _make_fake_cupy():
    m = types.ModuleType("fake_cupy")
    m.ndarray = DeviceArray

    def asarray(a, *a_, **kw):
        if isinstance(a, DeviceArray):
            return a
        return _wrap(np.asarray(a, *a_, **kw))

    def asnumpy(a):
        if not isinstance(a, DeviceArray):
            raise TypeError("asnumpy() called on something that is not a device array")
        return np.asarray(a).view(np.ndarray)

    def _lift(fn):
        def inner(*args, **kw):
            args = [np.asarray(x) if isinstance(x, DeviceArray) else x for x in args]
            return _wrap(fn(*args, **kw))
        return inner

    m.asarray, m.asnumpy = asarray, asnumpy
    m.where = _lift(np.where)
    m.isnan = _lift(np.isnan)
    m.sqrt = _lift(np.sqrt)
    m.concatenate = _lift(np.concatenate)
    m.stack = _lift(np.stack)
    m.inf = np.inf
    m.linalg = types.SimpleNamespace(norm=_lift(np.linalg.norm))

    class _Pool:
        def free_all_blocks(self): pass
        def used_bytes(self): return 0

    m.get_default_memory_pool = lambda: _Pool()
    m.get_default_pinned_memory_pool = lambda: _Pool()
    m.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            getDeviceCount=lambda: 1,
            deviceSynchronize=lambda: None,
            memGetInfo=lambda: (100 * 10**9, 128 * 10**9),
            getDeviceProperties=lambda i: {b"name": b"NVIDIA GB10",
                                           "name": b"NVIDIA GB10",
                                           "integrated": 1, "unifiedAddressing": 1}))
    return m


def install():
    from kessler import accel
    fake = _make_fake_cupy()
    accel.xp = fake
    accel.GPU = True
    for mod in ("kessler.conjunction", "kessler.monitor", "kessler.options"):
        __import__(mod)
        sys.modules[mod].xp = fake
    return fake


def test_full_pipeline_on_device_semantics():
    install()
    from kessler import accel
    from kessler.catalog import load_demo_catalog
    from kessler.mission import (Constraints, MissionState, find_tca,
                                 synthesize_threat)
    from kessler.monitor import select_fleet, sweep_fleet
    from kessler.options import solve_options
    from kessler.physics import elements, teme_state, timescale

    print(f"  backend reports : {accel.backend_name()}")
    info = accel.device_info()
    print(f"  device info     : {info}")
    assert info["gpu"] is True
    assert info["unified_memory"] is True, "unified memory should be detected on GB10"

    ts = timescale()
    cat = load_demo_catalog(limit=None)
    t0 = ts.now()

    # 1. fleet sweep -- the big broadcast
    fleet = select_fleet(cat.objects, "STARLINK", 8)
    res = sweep_fleet(fleet, cat.objects, t0, horizon_s=3 * 3600,
                      coarse_step_s=120.0, threshold_km=25.0)
    print(f"  fleet sweep     : {len(res.statuses)} assets, {res.states:,} states, "
          f"{res.elapsed_s:.1f}s")
    assert res.statuses, "sweep returned nothing"
    assert res.transfer["calls"] > 0, "no host->device transfers were recorded"
    print(f"  transfers       : {res.transfer['calls']} calls, "
          f"{res.transfer['bytes']/1e6:.0f} MB")

    # 2. the trade space -- the chunked screen
    hero = fleet[0]
    r0, v0 = teme_state(hero, t0)
    el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
    enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
    state = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                         nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2,
                         constraints=Constraints(dv_budget_mps=0.6))
    trade = solve_options(state, cat.objects, enc["tca_offset_s"])
    feas = sum(1 for o in trade["options"] if o.feasible)
    print(f"  trade space     : {feas}/{trade['scenarios']} feasible, "
          f"{trade['states_evaluated']:,} states, peak {trade['peak_chunk_gb']:.2f} GB")
    assert feas >= 1, "no feasible option on the device path"

    # 3. every returned number must be a host value, not a device handle
    for o in trade["options"]:
        for field in ("miss_km", "delta_v_mps", "altitude_drift_km"):
            v = getattr(o, field)
            assert isinstance(v, float) and not isinstance(v, DeviceArray), \
                f"{o.label}.{field} leaked a device array to the UI"
    assert not isinstance(res.statuses[0].alt_km, DeviceArray)
    print("  host/device     : no device arrays leaked into results")


def test_mixing_is_actually_caught():
    """Confirm the shim is strict, so a pass above means something."""
    install()
    from kessler.accel import to_device
    d = to_device(np.ones((3, 3)))
    try:
        _ = d - np.ones((3, 3))
    except TypeError as e:
        print(f"  shim strictness : mixing raises as expected — {str(e)[:58]}…")
        return
    raise AssertionError("the shim allowed host/device mixing; the test proves nothing")


if __name__ == "__main__":
    print("test_mixing_is_actually_caught"); test_mixing_is_actually_caught()
    print("\ntest_full_pipeline_on_device_semantics"); test_full_pipeline_on_device_semantics()
    print("\nOK")
