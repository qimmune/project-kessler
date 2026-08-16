# Handoff

Everything a new person — or a new Claude — needs to continue this work.

`CLAUDE.md` sits next to this file and is loaded automatically by Claude Code
when the repo is opened. **That is the transfer mechanism**: clone the repo,
open it with Claude, and the context comes with it. This document is the
narrative version for humans.

---

## Where the project stands

| Piece | State |
|---|---|
| Orbital physics engine | Working, validated against SGP4 to ~66 m / 30 min |
| Catalogue ingest | Working, 18,726 objects cached, runs offline after first fetch |
| Conjunction screening | Working, O(N) with altitude pre-filter, ~2 s for a 12-asset fleet sweep |
| Trade space (5 options) | Working, ~99M states per run, chunked to 0.26 GB peak |
| Agent loop | Working — tool-calling critic that probes beyond the generated set |
| Nemotron backend | Written and unit-tested against a stub. **Never run against a live endpoint.** |
| CuPy acceleration | Written. **Never executed on a GPU** — no CUDA on the dev machine. |
| OpenClaw deployment | MCP server + skills written and tested. **Never registered with a live OpenClaw agent.** |
| One-button UI | Working, verified end to end in both autonomous and supervised modes |
| Deck | 5 slides, `Project-Kessler.pptx`, on Cameron's Desktop |
| Repo | Public: https://github.com/qimmune/project-kessler |

## The three things that are not done

1. **Nothing has run on the GN100.** Every GPU number in the pitch is a
   projection. `scripts/preflight.py` will tell you the truth in about a minute
   on the box — run it first.
2. **No demo video.** The form requires one. A beat sheet is in `SUBMISSION.md`.
3. **Neither bounty box should be ticked yet.** Nemotron and OpenClaw are both
   built and tested against stubs, but neither has touched a live endpoint.
   Ticking before that would not survive code review.

## Bugs found along the way, and why they mattered

These are the ones that were silently wrong rather than loudly broken. Each cost
real time to find; none would have been caught by "does it run".

**SGP4 fed TAI instead of UT1.** Put every object ~133 km down-track. Distances
still *looked* plausible, which is what made it dangerous. Found by computing a
separation two ways — through skyfield's GCRS and through the raw TEME path —
and noticing they disagreed by 133 km.

**The TCA search found the wrong encounter.** Objects on intersecting orbits
re-approach every revolution, so an open-ended search horizon returned the
*next* crossing. Successful avoidances were being scored as failures, and the
response to Δv was non-monotonic, which is the tell.

**The altitude-box check compared osculating elements at different orbit
points**, folding in J2 oscillation and rejecting good burns.

**Ten minutes of lead time is physically insufficient.** No in-budget burn can
clear 2 km in 600 s; separation accumulates as ~3·Δv·t. The demo default moved
to 92 minutes.

**The batched screen materialised 1.19 GB at once** and grew with every scenario.
Chunking cut peak 5× *and* made it faster.

**A UI rewrite silently removed the agent from the demo.** For two commits the
flagship path ran pure physics plus a ranking rule, with no inference anywhere.
`tests/test_trade_review.py` now guards it. This is the failure mode to watch:
the demo can look completely healthy while being non-agentic.

**Streamlit specifics that bite:** session state survives hot-reloads, so a dict
written by older code crashes newer code (hence `STATE_SCHEMA`); and identical
charts collide on auto-generated ids (hence explicit `key=` on every one).

## Things deliberately not done

- **No synthetic debris in the screening path.** The modeled cloud is visual
  only. Padding the tracked catalogue to match a headline number would be the
  one claim a judge could break.
- **No LeoLabs API integration.** Their public visualisation is embedded under
  their sharing terms with attribution; the authenticated API needs a key nobody
  has. Attribution must stay attached to the frame.
- **No vision/multi-station triangulation.** Was considered; the inputs are TLEs,
  which are already position estimates, so simulating sensors would make the
  demo less honest rather than more.

## If you are picking this up cold

```bash
git clone https://github.com/qimmune/project-kessler.git
cd project-kessler
./scripts/bootstrap.sh     # sets everything up, ends with preflight
./scripts/test.sh          # 4 suites, all should pass
./scripts/run.sh           # the demo
```

Then read `CLAUDE.md`. It is short and every line in it is there because
something went wrong without it.

## Open questions for the team

- **Autonomous or supervised for the pitch?** The deck's punchline is
  `requires_human_ack: false`; the trade-space UI hands a human five costed
  options. Both are built and switchable. They tell different stories — pick one
  before presenting, and re-cut slide 5 to match.
- **How long is the trade space allowed to take live?** 8 s on CPU here. If the
  GPU does not materially beat that on the box, drop `Assets protected` or the
  modeled debris layer rather than standing in silence.
