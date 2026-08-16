# Project Kessler — context for Claude

Autonomous orbital traffic control, built for the NVIDIA Spark Hack (Seattle).
Read this before changing anything; several decisions here are load-bearing and
look arbitrary from the outside.

## What it does

Ingests the live CelesTrak TLE catalogue, propagates every object, screens a
fleet of protected satellites for collisions, generates a **trade space of five
avoidance maneuvers**, has an agent probe that trade space through the physics
engine, and hands a human the costed options to choose from.

One button: `./scripts/run.sh` → **Engage orbital traffic control**.

## Non-obvious decisions — do not "simplify" these

**Two propagators, deliberately.** SGP4 (skyfield) drives catalogue screening
because it is the model TLEs are *defined against*. An RK4 two-body + J2
integrator drives maneuver arcs, because a burn changes the osculating state and
round-tripping that through SGP4 mean elements injects error the same size as
the miss distances under discussion. They agree to ~66 m over 30 min, measured
every run in `cross_check()`.

**Everything in the screening path stays in TEME.** Relative geometry between
two objects at one instant is identical in TEME and GCRS, so the conversion buys
nothing and costs a class of frame bugs.

**SGP4 is driven by UT1, not TAI or TT.** Feeding it TAI puts every object
~133 km down-track and silently corrupts every distance downstream. This was
found by validating against skyfield's own GCRS output. See `physics.teme_state`.

**TCA search must be bracketed.** Two objects on intersecting orbits re-approach
roughly once per revolution, so an open-ended horizon returns the *next*
crossing and scores a successful avoidance as a failure. `find_tca_window`
exists for this reason.

**The altitude box is measured against the unburned orbit at the burn point,**
not against elements taken at t0. Comparing across different points in the orbit
folds in J2's osculating oscillation and rejects perfectly good burns.

**The batched screen is chunked over catalogue objects.** The full
`(scenarios × objects × epochs × 3)` difference is ~1.19 GB at demo settings and
grows with every scenario. Chunking holds peak near 0.26 GB and, on CPU, is
*faster* than the unchunked version from cache locality.

**Detection is not the same as needing to maneuver.** `requires_action()` gates
every engagement on the separation minimum and the Pc threshold. Without it the
system burns propellant "improving" encounters that were already safe.

**The engine outranks the model.** If an agent claims a maneuver is approved and
the checks contradict it, the engine wins and the disagreement is logged. In the
OpenClaw tool surface, `issue_burn` refuses anything `simulate_maneuver` has not
cleared — enforced in the tool, not the prompt.

## Regression that already happened once

A UI rewrite silently removed the agent from `app.py`: it called `solve_options`
(pure physics) plus a single-shot ranking call with no tools, which degraded to
a rule when no API key was set — so the flagship demo ran with **zero inference**
while the agent loop sat unused in `run_demo.py`.

`tests/test_trade_review.py` guards this. If you touch the demo path, run it.
**The demo must exercise a tool-calling agent, or the whole "Do" track claim and
both bounties are void.**

## Honest scope — keep it honest

Real: the TLE catalogue, SGP4 with the drag term, conjunction geometry, Pc,
Δv against a published thruster spec, the secondary screen against ~6,900 real
objects, agent reasoning and tool calls.

Seeded: the demo conjunction. `synthesize_threat()` back-propagates a debris
object from a chosen encounter so the demo is deterministic. It is a real
physical trajectory; the detector finds it the way it finds organic ones, and
the engine independently measures the geometry it was asked for.

Modeled: the 45,000-fragment cloud in the visualization. Sampled from published
debris shell distributions, **never screened against**, labelled as such in the
legend. ESA's ~130M figure is for fragments >1 mm, essentially none of which are
tracked — say "larger than a millimetre, and 99.97% untracked" or it is wrong.

Mocked: there is no uplink. Approved commands are validated and logged.

## Layout

```
kessler/
  catalog.py      CelesTrak ingest, caching, object classification
  physics.py      SGP4/TEME, RK4+J2, RIC frame, elements
  conjunction.py  screening, refinement to TCA, Pc
  mission.py      constraints, threat synthesis, evaluate_maneuver (the tool)
  options.py      trade-space generation + batched evaluation
  agents.py       FDO/MAD loop, Nemotron + Claude backends, trade-space review
  monitor.py      fleet sweep off one resident propagation
  assurance.py    cross-check, engagement log, authorization gate
  accel.py        CuPy/NumPy backend + transfer instrumentation
  environment.py  modeled debris (visual only)
openclaw/         MCP tool server + SKILL.md files for OpenClaw deployment
scripts/          bootstrap, preflight, run, test
app.py            the one-button console
app_advanced.py   older multi-mode console (fleet monitor, live scan)
run_demo.py       CLI, exercises the original two-agent loop
```

## Conventions

- Backends resolve automatically: Nemotron if an NVIDIA key or base URL is set,
  Claude if an Anthropic key is, otherwise a deterministic solver that drives the
  same tools and the same checks. Never let a missing key hard-fail the demo.
- `st.session_state["done"]` carries `STATE_SCHEMA`; bump it when the shape
  changes or stale state from a hot-reload will crash the next run.
- Every `st.plotly_chart` needs an explicit `key=` — Streamlit derives ids from
  call parameters and identical charts collide.
- Run `./scripts/test.sh` before pushing.

## Where it runs

Built and tested on macOS/CPU. Target is the Acer Veriton GN100 (GB10 Grace
Blackwell). `scripts/preflight.py` verifies the box; `DEPLOY.md` is the runbook.
**The GPU path has never executed** — no CUDA on the development machine — so
`accel.py` is exercised only through its NumPy fallback. Run preflight on the
box first.
