---
name: maneuver-assurance
description: Design and verify a collision-avoidance burn for a satellite with an open conjunction, then issue the command. Use after a conjunction has been opened and a maneuver is required.
user-invocable: true
metadata:
  {
    "openclaw":
      {
        "requires": { "bins": ["python3"] },
        "os": ["darwin", "linux"],
        "emoji": "🔥",
      },
  }
---

# Maneuver assurance

You design an avoidance burn and then prove it is safe. Both halves matter, and
the second one is not optional.

## The rule

**Never conclude a maneuver is safe from reasoning.** Call `simulate_maneuver`.
It flies the burn through the real propagator and returns four checks: the
primary threat cleared, the delta-v within budget, the mission altitude box held,
and no new conjunction created anywhere in the catalogue over the next twelve
hours. Your opinion about a burn is worth nothing next to what that tool returns.

`issue_burn` will refuse anything `simulate_maneuver` has not cleared, so there is
no path around this. If you find yourself wanting to argue with a rejection,
propose a different burn instead.

## Designing the burn

The frame is RIC — `direction_ric` is `[radial, in_track, cross_track]` as a unit
vector.

In-track is almost always the answer. Separation from an in-track burn grows as
roughly `3 × delta_v × time_from_burn_to_closest_approach`, so it accumulates,
and an earlier burn buys far more distance per metre-per-second than a bigger
one. Radial and cross-track displacements oscillate rather than accumulate — the
energy comes back half an orbit later — so they cost several times as much for
the same miss distance. "Climb above the debris" is the intuitive move and the
wrong one.

Size the first attempt from that relationship rather than guessing, and prefer
igniting earlier over spending more propellant. Fuel is the one thing a satellite
cannot be resupplied with; it sets the vehicle's remaining service life.

## Reading a rejection

Each failed check names itself and gives you the number that failed it.

- `primary_threat_cleared` — the burn was too small or too late. Scale the
  delta-v, or move the ignition earlier, which is cheaper.
- `delta_v_within_budget` — you cannot buy it with fuel. Buy it with lead time.
- `altitude_box_held` — the burn moved the orbit out of its service band. Smaller
  burn, earlier ignition.
- `no_new_conjunctions` — the new orbit runs into something else. Try the
  opposite in-track direction.

## Committing

When `simulate_maneuver` returns `approved: true`, call `issue_burn` with a one
sentence rationale. The uplink is simulated: the command is validated and written
to a persistent engagement log rather than transmitted to a spacecraft. Say so
plainly when you report what you did.

Use `engagement_history` to see what you have handled before — it persists across
sessions.
