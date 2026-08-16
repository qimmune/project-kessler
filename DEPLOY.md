# Deploying on the Acer Veriton GN100 (GB10 Grace Blackwell)

Three commands, cold box to running demo.

```bash
git clone https://github.com/qimmune/project-kessler.git
cd project-kessler
./scripts/bootstrap.sh      # detects CUDA, installs the stack, warms the catalogue
./scripts/run.sh            # opens the console
```

`bootstrap.sh` is safe to re-run and finishes by running preflight, so you know
whether the box is demo-ready before anyone is watching.

---

## What bootstrap does

1. Detects OS, architecture, and CUDA version from `nvidia-smi`.
2. Creates `.venv` and installs the base stack.
3. Installs the **CuPy wheel matching the box's CUDA major version**
   (`cupy-cuda13x`, `cupy-cuda12x`, …). Grace is aarch64; wheels exist for it.
   If none matches, screening falls back to NumPy — the demo still runs, slower.
4. Fetches ~18,700 TLEs from CelesTrak into `data/`. **After this the demo runs
   with no network**, which matters on conference wifi.
5. Runs preflight.

## Preflight

```bash
.venv/bin/python scripts/preflight.py
```

Checks the stack, the accelerator, whether unified memory is actually reported,
the catalogue, the agent backend and whether its endpoint answers — then solves
a real trade space and times it. It will tell you if the run is too slow to demo
live.

Expected on the GN100:

```
✓ accelerator: cupy on NVIDIA GB10
✓ unified memory confirmed — 128 GB shared, host↔device copies are pointer handoffs
✓ trade space: 4/5 feasible in <a few> s
✓   ~99,000,000 states · 1.19 GB logical · 0.26 GB peak · cupy on NVIDIA GB10
```

On this laptop (CPU, for comparison) the same work takes **8.1 s** and reports
`numpy (CPU)` with zero transfer. That contrast is the GB10 argument — run both
and show them side by side:

```bash
KESSLER_FORCE_CPU=1 .venv/bin/python scripts/preflight.py
```

## Nemotron

`run.sh` auto-detects a local NIM on `localhost:8000` and switches to it. To be
explicit:

```bash
export KESSLER_BACKEND=nemotron
export KESSLER_BASE_URL=http://localhost:8000/v1
export KESSLER_MODEL=<the model your NIM serves>
```

With no model configured the deterministic solver drives the same tools and the
same physics, so the demo never hard-fails on a missing endpoint.

## Knobs

| Variable | Default | Purpose |
|---|---|---|
| `KESSLER_PORT` | 8760 | UI port |
| `KESSLER_BACKEND` | auto | `nemotron` · `claude` · `offline` |
| `KESSLER_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `KESSLER_FORCE_CPU` | unset | Pin NumPy to demo the contrast |
| `KESSLER_SCREEN_CHUNK` | 1500 | Objects per screening chunk; raise it on 128 GB to cut kernel launches |

## Memory

The trade space is a logical `(scenarios × objects × epochs × 3)` tensor — about
1.19 GB at demo settings. It is **chunked over catalogue objects**, so peak
device memory stays near 0.26 GB no matter how many scenarios are in flight.
On 128 GB of unified memory you can raise `KESSLER_SCREEN_CHUNK` substantially
and trade memory for fewer, larger kernels.

## If something breaks on the day

| Symptom | Fix |
|---|---|
| `no CuPy/CUDA` in preflight | `\.venv/bin/pip install cupy-cuda12x`. Demo still runs without it. |
| Catalogue fetch fails | `data/*.tle` from any previous run is enough; the loader falls back to cache automatically. |
| Nemotron endpoint down | Nothing to do — the deterministic solver takes over and the physics is identical. |
| Trade space too slow live | Lower `Assets protected` in **Advanced**, or drop the modeled debris layer to `off`. |
| UI looks stale after a code change | Streamlit keeps session state across reloads; hit **Reset**, or restart. |
