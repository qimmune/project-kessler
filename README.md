# Project Kessler

Autonomous orbital traffic control. Ingests the live CelesTrak catalog, screens a
protected satellite against every object in it, and when a collision is coming,
runs a two-agent loop that designs an evasion burn, verifies it against the
physics engine, and issues it — with no human in the loop.

Built for NVIDIA Spark Hack Seattle (track: Do).

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

./.venv/bin/streamlit run app.py            # dashboard, all three modes
./.venv/bin/python run_demo.py              # console: seeded threat, autonomous
./.venv/bin/python run_demo.py --supervised # hold at the authorization gate
./.venv/bin/python run_demo.py --organic    # no seeding, screen for a real one
./.venv/bin/python run_demo.py --monitor    # fleet sweep
```

Set `ANTHROPIC_API_KEY` to put Claude in both agent seats. Without it the demo
still runs end to end on a deterministic solver that calls the same tool against
the same physics — the loop, the checks, and the verdicts are identical.

---

## What actually runs

| Layer | Module | What it does |
|---|---|---|
| Ingest | `kessler/catalog.py` | CelesTrak TLEs via skyfield, cached to `data/` so the demo survives bad wifi |
| Propagation | `kessler/physics.py` | SGP4 for the catalog; RK4 two-body + J2 for maneuver arcs |
| Screening | `kessler/conjunction.py` | Coarse sweep → candidates → refinement to true TCA |
| Mission | `kessler/mission.py` | Threat synthesis, constraints, and `evaluate_maneuver` — the tool |
| Agents | `kessler/agents.py` | FDO proposer + Mission Assurance critic, tool-calling loop |
| Assurance | `kessler/assurance.py` | Consensus cross-check, engagement log, authorization gate |
| Monitoring | `kessler/monitor.py` | Fleet-wide sweep off one resident propagation |
| UI | `app.py` | 3D globe, alert log, agent console, gate, engagement log |

### Two propagators, on purpose

SGP4 is the model TLEs are *defined against*, so it drives all catalog screening.
But a burn changes the osculating state, and round-tripping that back through
SGP4 mean elements injects conversion error of the same order as the miss
distances under discussion. So maneuver evaluation integrates the pre-burn and
post-burn arcs identically with RK4+J2. Measured agreement between the two over
30 minutes: **66 metres**.

Everything in the screening path stays in TEME, the frame SGP4 natively outputs.
Relative geometry between two objects at one instant is identical in TEME and
GCRS, so skipping the conversion costs nothing and removes a class of frame bugs.

> SGP4 wants **UT1**. Feeding it TAI puts every object ~133 km down-track and
> silently corrupts every distance in the pipeline. `kessler/physics.py` has the
> validation that caught this.

### The screen is O(N), not O(N²)

One protected asset against the whole catalog — which is also how real operators
work. An altitude pre-filter drops objects whose orbit cannot physically reach
the hero's band, typically **16,085 → ~1,330** before the expensive pass.

Measured on an M-series laptop:

| Operation | Scale | Time |
|---|---|---|
| Full-catalog propagation | 16,085 objects × 180 epochs = 2.9M states | **0.70 s** |
| 6-hour screen, 60 s cadence | 16,085 objects | **1.69 s** |
| One maneuver evaluation (the tool) | full re-propagation + 12 h secondary screen | **~0.9 s** |
| Alert → issued burn command | 2 agent rounds | **~1.3 s** |

The full 24-hour sweep at 60 s cadence is 23.2M state vectors — 1.73 GB of
ephemeris resident, before covariance sampling or parallel maneuver scenarios.
That is the GB10 unified-memory argument, and `run_demo.py` prints the real
numbers rather than asserting them.

---

## Three modes

| Mode | What it does |
|---|---|
| **Seeded threat** | Deterministic encounter, guaranteed to fire. Use this on stage. |
| **Live scan** | Screens one asset against the real catalog, no seeding. May legitimately find nothing — that is the honest answer. |
| **Fleet monitor** | Sweeps many assets continuously, auto-dispatching agents only to the ones that breach. Optional auto-rescan. |

The fleet sweep propagates the catalog **once** into a single resident array and
reuses it for every asset. Adding assets costs one extra track each, not another
catalog sweep — measured: **12 assets against 16,085 objects in 2.23 s**, off one
5.8M-state, 139 MB matrix.

## Detection is not the same as needing to maneuver

`requires_action()` gates every engagement. A conjunction escalates to the agents
only if it breaches the separation minimum **or** crosses the Pc action threshold.
Otherwise it is logged and watched.

Without this gate the system spends real propellant "improving" encounters that
were already safe — a 3.0 km approach with a 2.0 km minimum and a Pc of 6.5e-20
would still trigger a burn, buying 0.13 km for 0.18 m/s. A typical fleet sweep
returns several WATCH entries and zero ACTION, which is the correct answer.

## Authority: supervised or autonomous

Two settings, and the demo can show either:

* **SUPERVISED** — the engine-approved burn holds at an authorization gate. A
  human authorizes or halts, and the decision is recorded with their name.
* **AUTONOMOUS** — the gate self-authorizes and records that it did.

Either way **nothing is transmitted**. The uplink is simulated; the approved
command is validated and written to the engagement log.

## Consensus cross-check

Before anyone acts on the geometry, it is computed twice by two independent
dynamical models and the residual reported. SGP4 is the model the TLE is defined
against; RK4 two-body + J2 is an independent integrator seeded from the same
state. They share no code path beyond the initial conditions.

| Horizon | SGP4 vs RK4+J2 residual |
|---|---|
| 10 min | 5.3 m |
| 30 min | 42.1 m |
| 92 min | ~100 m |

A single propagator can be confidently wrong; two that disagree tell you so. If
the residual exceeds tolerance, the run flags the geometry as untrustworthy.

> The residual is a **floor** on position uncertainty, not the real thing. TLE
> uncertainty dominates it by an order of magnitude, so `collision_probability`
> still uses a documented 350 m sigma. Both numbers are surfaced rather than
> conflated.

## Engagement log

Every conjunction produces one auditable record: geometry, consensus residual,
the full agent reasoning trail with timestamps, the engine verdict, the
authorization decision and who made it. Downloadable as JSON from the dashboard.

---

## The agent loop

**Agent 1 — Flight Dynamics Officer.** Receives the conjunction alert, proposes a
burn as JSON: direction in RIC, magnitude, ignition time.

**Agent 2 — Mission Assurance Director.** Never approves from reasoning. It calls
`simulate_maneuver`, which flies the burn and returns four checks:

1. **Primary threat cleared** — new miss ≥ 2 km and Pc ≤ 1e-4
2. **Delta-v within budget** — ≤ 0.35 m/s
3. **Altitude box held** — ≤ 3 km drift, measured against the *unburned* orbit at
   the same instant so J2's osculating oscillation can't cause false rejections
4. **No new conjunctions** — the post-burn arc is re-screened against the catalog
   over 12 hours

Rejections carry the failing numbers back to Agent 1, which recalculates.

**The engine outranks the model.** If Claude claims APPROVED but the checks
failed, the loop logs the disagreement and takes the engine's answer.
`tests/test_claude_path.py` asserts this.

### The debate you'll see

The first proposal is the intuitive one and the wrong one — a **radial** burn,
"climb above the debris." Radial displacement oscillates at roughly `2·Δv/n`, a
few hundred metres here, because radial energy comes straight back half an orbit
later. The engine measures it, the critic rejects it with the number, and the FDO
switches to **in-track**, where separation accumulates as `3·Δv·t`.

That is not scripted. Measured response at 92 minutes of lead time:

| Δv (m/s) | miss (km) | verdict |
|---|---|---|
| 0.02 | 0.683 | rejected — insufficient separation |
| 0.10 | 1.514 | rejected — insufficient separation |
| **0.15** | **2.066** | **approved** |
| 0.30 | 3.707 | approved |
| 0.50 | 3.974 | rejected — over Δv budget |

10.8 km of separation per m/s, against the `3·Δv·t` prediction of ~14.8 km/(m/s)
before the crossing-geometry factor. The physics is doing the work.

---

## LeoLabs

The dashboard can embed LeoLabs' public LEO visualization beside our own screen —
one is what we compute from public TLEs, the other is what a commercial
phased-array radar network actually observes. Toggle it in the sidebar.

Two views are wired: the full LEO catalog (~27,600 tracked objects) and today's
conjunctions.

**Licensing.** LeoLabs' [terms for sharing](https://platform.leolabs.space/visualizations_terms_for_sharing)
permit use for *non-commercial educational, academic, or research purposes*,
which a hackathon build is, provided credit is given with a link to
https://leolabs.space and their marks are not removed. `render_leolabs()` in
`app.py` carries that attribution — **keep it attached to the frame.** Their
terms do not cover commercial or promotional use, so if this becomes a product,
the embed comes out until there is an agreement.

**Access.** Both visualization URLs are public — no login, no API key. They also
send no `X-Frame-Options` and no CSP `frame-ancestors`, so they embed cleanly.
The authenticated LeoLabs *API* (state vectors, conjunction data, tasking) is a
different matter and needs a key from `accounts@leolabs.space`; nothing here
depends on it.

**Caveat.** The 3D view failed to render inside a cross-origin iframe in a
headless test browser while working fine standalone. The sidebar panel therefore
offers a direct link as well as the frame. Check the embed on your own machine
before relying on it in a demo.

---

## Honest scope

**Real:** live TLE catalog; SGP4 including the drag term; conjunction geometry,
miss distance, and TCA; Δv and burn duration; RIC frame mechanics; the secondary
screen against the full catalog; unscripted agent reasoning and tool calls.

**Seeded:** the demo conjunction. `synthesize_threat` builds a debris object by
back-propagating from a chosen encounter, so the demo is deterministic. It is a
real physical trajectory through the same integrator, and nothing downstream is
hardcoded — the engine *measures* the encounter it finds. Ask for 0.412 km and
the engine independently reports 0.412 km.

Run `--organic` to skip the seed and screen for a natural conjunction instead.

**Mocked:** there is no uplink; the approved command is validated and logged.
Vehicle propellant state is a plausible constant — real feeds are operator-private.

**Assumption to flag:** public TLEs carry no covariance, so `collision_probability`
uses a fixed 350 m position sigma in a Foster-style circular Pc. A real operator
folds in the tracking provider's full 6×6. The constant is documented in
`kessler/conjunction.py`, not buried.

---

## Demo choreography

1. **Idle** — globe turning, live catalog, protected asset highlighted.
2. **RUN AVOIDANCE** — screening banner, then the red conjunction alert:
   miss 0.412 km, TCA T+92 min, Pc 2.9e-4, closing at 11.75 km/s.
3. **Agent console** — FDO proposes radial. Engine simulates. **REJECTED**, with
   the number. FDO switches to in-track. **APPROVED.**
4. **Globe updates** — original path greys out, post-burn track bends clear of
   the threat, burn point marked.
5. **Metrics** — 0.412 → 2.48 km, Pc 2.9e-4 → 6.7e-15, 0.189 m/s spent, 2 rounds.

Say the last line out loud: *nobody approved that burn.*

## Tuning

`--tca-min` sets lead time; shorter windows force larger burns and more agent
rounds. Tighten `Constraints.dv_budget_mps` in the sidebar to force a longer
debate. `--limit` caps catalog size for slower machines.
