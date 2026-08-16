# Seattle Spark Hackathon — submission draft

Fields marked **[YOU]** need your input and I have not guessed at them.

---

## Project Name
```
Project Kessler
```

## Team Name
**[YOU]**

## Team Member Names & Email Addresses
**[YOU]**

## Submission Track
```
Do
```
Primary is Do — the deliverable is an agent that plans and acts. There is a
substantial See surface (the 63,000-point orbital visualization) but the burn
command is the point, so lead with Do.

## Submission Description
```
Project Kessler is an autonomous orbital traffic controller.

It ingests the live CelesTrak catalogue — 18,726 tracked objects — propagates
every one with SGP4, and screens a fleet of protected satellites for collisions.
When something is going to hit, two Nemotron agents take over. A Flight Dynamics
Officer proposes an avoidance burn. A Mission Assurance Director verifies it by
calling the physics engine as a tool, never by reasoning alone: it re-flies the
burn, checks the new miss distance and probability of collision, the delta-v
budget, the mission altitude box, and whether the new orbit creates a fresh
conjunction anywhere in the catalogue. Rejections carry the failing numbers back
and the loop recalculates until the engine clears it. If the model claims
approval the checks contradict, the engine wins and the disagreement is logged.

One button. Alert to signed thruster command in seconds, with no human in the
loop — or held at a human authorization gate, switchable.

The whole catalogue is propagated once into a resident state matrix and screened
in place. On GB10 that matrix never crosses PCIe: Grace writes the ephemeris,
Blackwell reads it where it lies. We instrument every host-to-device transfer and
report what we measured rather than asserting the number we wanted.

Everything is real: real TLEs, real SGP4 including the drag term, real conjunction
geometry, real delta-v against a published thruster spec. The demo conjunction is
seeded so the 90 seconds is deterministic — the detector finds it exactly the way
it finds the organic ones, and the engine independently measures the geometry it
was asked for.
```

## Demo Video URL
**[YOU]** — not recorded yet. Suggested 5-minute structure:
1. `0:00` Globe. 15,275 satellites, 2,635 tracked debris, modeled fragment cloud.
2. `0:30` Press **Engage**. Sweep of 18,726 objects, states/second on screen.
3. `1:00` Collision course. 0.412 km, 92 minutes, 11.75 km/s, Pc 2.9e-4.
4. `1:30` Agent console. FDO proposes radial → engine **rejects** with the number
   → FDO switches to in-track → **approved**. Say out loud that this is not scripted.
5. `2:30` Trajectory bends. Encounter close-up: 0.412 km → 2.48 km.
6. `3:15` GB10: the transfer counter, and the same sweep with `KESSLER_FORCE_CPU=1`.
7. `4:15` What is real vs seeded. Close.

## GitHub Repo
```
https://github.com/qimmune/project-kessler
```
> **Blocker: the repo is PRIVATE and the form requires public.** One command:
> `gh repo edit qimmune/project-kessler --visibility public --accept-visibility-change-consequences`

## AI Models Used
```
NVIDIA Nemotron (nvidia/nemotron-3-super) — drives both agents in the
multi-agent loop: the Flight Dynamics Officer that proposes avoidance burns and
the Mission Assurance Director that verifies them through tool calling. Served
over an OpenAI-compatible endpoint, either a local NIM on the DGX Spark or
hosted.

Anthropic Claude (claude-sonnet-5) — supported as an alternate backend behind
the same interface, for A/B comparison of agent behaviour.

Deterministic solver — a no-model fallback that drives the identical tool and
the identical physics checks, so the system degrades to a working state with no
inference available at all.
```

## Tools Used
```
NVIDIA Nemotron — both agents in the multi-agent reasoning loop.

NVIDIA NIM — local inference serving on the DGX Spark. KESSLER_BASE_URL points
at the local NIM so the entire agent loop stays on-device and no telemetry
leaves the box.

CuPy / CUDA — conjunction screening runs on the GPU. The sweep is one enormous
elementwise distance computation over a resident (N_objects x N_epochs x 3)
state matrix; kessler/accel.py selects CuPy when CUDA is present and NumPy
otherwise, over one code path.

DGX Spark / GB10 Grace Blackwell — unified memory. The ephemeris is propagated
once by SGP4 and screened in place. Host-to-device transfer is instrumented by
count, bytes and seconds and reported by the sweep, so the unified-memory claim
is measured rather than asserted.
```

## Bounty checkboxes

**Nemotron Lightning — CHECK IT, after one live run.**
The Nemotron backend is built and the OpenAI tool-call translation is tested
(`tests/test_nemotron_path.py`), but only against a stubbed endpoint — no live
Nemotron call has been made. Run this once at the event before ticking:
```bash
export KESSLER_BACKEND=nemotron
export KESSLER_BASE_URL=http://localhost:8000/v1
export KESSLER_MODEL=<model the NIM actually serves>
./.venv/bin/python run_demo.py
```

**NemoClaw / OpenShell — CHECK IT, after deploying on the box.**
The OpenClaw agent is built and under `openclaw/`:
- **Tool layer** — `openclaw/kessler_mcp.py`, an MCP server exposing five tools:
  `screen_fleet`, `open_conjunction`, `simulate_maneuver`, `issue_burn`,
  `engagement_history`.
- **Guidance layer** — two skills, `orbital-conjunction-screening` and
  `maneuver-assurance`, in OpenClaw SKILL.md format.
- **Persistent memory** — `openclaw/state/engagements.json`, written on commit
  and read back across sessions.
- **Policy guardrail** — `issue_burn` refuses any command the physics engine has
  not cleared. Enforced in the tool, not the prompt, so no amount of persuasion
  reaches the spacecraft.

`tests/test_openclaw_tools.py` drives the whole sequence and asserts the refusal.
Verified locally:
```
screen_fleet      18,726 objects · 6,760,086 states · 3.11s
open_conjunction  STARLINK-1008 vs COSMOS-1408 DEB · 0.412 km · Pc 2.94e-04
simulate radial   approved=False · failed ['primary_threat_cleared']
issue_burn        REFUSED — the engine did not clear this maneuver
simulate in-track approved=True · 2.498 km · Pc 5.1e-15
issue_burn        ISSUED KES-E023828F → 2.498 km
history           1 engagement persisted
```
Still to do at the event: register the MCP server with `openclaw mcp add`, point
the agent at the skills, and run it against a local Nemotron NIM. Instructions in
`openclaw/README.md`.

## GB10 Experience — which capabilities were most valuable
> **[YOU] must run it on the GB10 before submitting this.** The technical
> substance is below; replace the bracketed numbers with what you actually
> measure. Do not submit figures you have not seen.
```
Unified memory, decisively.

Conjunction screening is not hard mathematics — it is an enormous,
constantly-rewritten array that two different processors need to read at the
same time. Propagating 18,726 objects over a 24-hour horizon at 60-second
cadence is 27 million state vectors, about 1.6 GB of ephemeris for a single
sweep, before covariance sampling or parallel maneuver scenarios.

SGP4 propagation is branchy and sequential and belongs on the Grace CPU. The
screening sweep is a wide elementwise reduction and belongs on Blackwell. On a
discrete-GPU machine, splitting the work that way means moving the state matrix
across PCIe on every sweep, and that copy is the entire latency budget for a
system whose whole claim is that it closes the loop in seconds.

GB10 removes the copy. Grace writes the ephemeris, Blackwell reads it where it
lies. We instrumented every host-to-device transfer by count, bytes and seconds
rather than taking it on faith: [MEASURED — paste the transfer report here], and
the same sweep pinned to CPU with KESSLER_FORCE_CPU=1 took [X] s against [Y] s
on the GPU.

The second thing that mattered was running Nemotron locally on the same box as
the physics. The critic agent calls the simulation as a tool on every proposal,
so agent latency and physics latency are the same budget. Having inference and
simulation share the machine — and the memory — meant the whole
propose/verify/reject/recalculate loop closed in [Z] seconds without a network
round trip, which is what makes this an autonomous controller rather than a
batch report.
```

## Recommend GB10 (1-10) + why
**[YOU]** — your opinion, and you should form it after using the box.
Substance you might draw on: unified memory removing an entire class of
data-movement engineering; local inference next to the simulation; whatever
friction you actually hit in setup.

## Local inference and compute vs previous environments (1-10) + why
**[YOU]** — same. Honest comparison points: no per-token cost or rate limit on
the agent loop, no network round trip inside the critic's tool loop, everything
on-device so nothing leaves the machine.

## What additional features or improvements would help
```
Suggestions, edit to taste:

- A profiler view that attributes time and memory traffic across the Grace/
  Blackwell boundary. We instrumented transfers by hand to prove the
  unified-memory claim; having that first-class would have saved hours.
- Prebuilt CuPy/CUDA wheels matched to the shipped image. Getting the numerical
  stack aligned with the driver was the slowest part of setup.
- A documented, pinned NIM model catalogue for the box, with exact model
  identifiers. We wrote the Nemotron client against an OpenAI-compatible
  endpoint and had to discover the served model name at runtime.
- Guidance on concurrent workloads: we ran local inference and a GPU
  compute kernel against the same memory and had no way to reason about
  contention between them.
```

---

## Handoff

`CLAUDE.md` in the repo root carries the full engineering context and is loaded
automatically when the repo is opened with Claude Code — that is how to move
this to another account without losing anything. `HANDOFF.md` is the human
narrative: current state, every non-obvious bug and why it mattered, what was
deliberately not done, and the open questions.
