# Project Kessler on OpenClaw / NemoClaw

Deploys the orbital traffic controller as an OpenClaw agent powered by Nemotron.

- **Tool layer** — `kessler_mcp.py`, an MCP server exposing the physics engine.
- **Guidance layer** — two skills under `skills/`, one per agent role.
- **Persistent memory** — `state/engagements.json`, written by `issue_burn` and
  read back by `engagement_history`.

## The guardrail that matters

The model never decides whether a maneuver is safe. `simulate_maneuver` flies the
burn through the real propagator and returns four hard checks, and `issue_burn`
**refuses to emit a command the engine has not cleared**. That refusal is
enforced in the tool, not in the prompt, so no amount of persuasion reaches the
spacecraft. `tests/test_openclaw_tools.py` asserts it by proposing a burn that
fails and confirming the commit is denied.

## Install

```bash
# 1. register the tool server
openclaw mcp add project-kessler -- \
  /path/to/project-kessler/.venv/bin/python \
  /path/to/project-kessler/openclaw/kessler_mcp.py

# 2. point the agent at the skills
#    openclaw config, or ~/.openclaw/config:
#
#      agents:
#        defaults:
#          skills:
#            - orbital-conjunction-screening
#            - maneuver-assurance
#        skillDirs:
#          - /path/to/project-kessler/openclaw/skills

# 3. run Nemotron locally on the DGX Spark and point the agent at it
export KESSLER_BACKEND=nemotron
export KESSLER_BASE_URL=http://localhost:8000/v1
export KESSLER_MODEL=<model your NIM serves>
```

Verify the tool surface without an agent attached:

```bash
./.venv/bin/python tests/test_openclaw_tools.py
```

## Try it

> Sweep our Starlink fleet for collisions and fix anything that needs fixing.

The agent screens the catalogue, opens an engagement on the breach, proposes a
burn, gets rejected by the physics, corrects, and commits — with the whole
exchange written to the engagement log.
