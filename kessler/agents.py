"""The two-agent loop.

Agent 1 (Flight Dynamics Officer) proposes a burn. Agent 2 (Mission Assurance
Director) does not take that on faith -- it calls the physics engine as a tool
and reports what actually happens. Rejections carry the failing numbers back to
Agent 1, which recalculates. The loop ends when the engine says the burn is safe.

Both agents run against Claude when ANTHROPIC_API_KEY is set. Without a key the
module falls back to a deterministic solver so the demo still runs end to end --
same tool, same physics, same verdicts, just no language model in the seat.
"""
from __future__ import annotations

import json
import os
import re
import types

from .bus import Bus
from .mission import MissionState, evaluate_maneuver

MAX_ROUNDS = 4

# Backend selection.
#   nemotron -> NVIDIA Nemotron over an OpenAI-compatible endpoint. Point
#               KESSLER_BASE_URL at a local NIM on the DGX Spark to keep the
#               whole loop on-device, or at build.nvidia.com to run hosted.
#   claude   -> Anthropic API.
#   offline  -> deterministic solver, same tool and same physics, no model.
BACKEND = os.environ.get("KESSLER_BACKEND", "auto").lower()
MODEL = os.environ.get("KESSLER_MODEL", "")
BASE_URL = os.environ.get("KESSLER_BASE_URL", "http://localhost:8000/v1")
NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")

DEFAULT_MODELS = {
    "nemotron": "nvidia/nemotron-3-super",
    "claude": "claude-sonnet-5",
}

FDO_SYSTEM = """You are an autonomous Flight Dynamics Officer for a satellite operator.

You receive a Conjunction Alert JSON describing an imminent close approach between
a satellite you protect and a piece of debris. Your job is to keep the satellite
alive using the minimum propellant possible.

Frame: RIC. direction_ric is [radial, in_track, cross_track], a unit vector.
  - in-track (+/-1 in the second slot) is by far the most fuel-efficient way to
    change WHERE the satellite is along its orbit. Separation grows roughly as
    3 * delta_v * (time from burn to TCA), so an earlier burn buys far more
    distance per m/s than a bigger one.
  - radial and cross-track displacements oscillate rather than accumulate; they
    cost much more for the same miss distance.

Respond with ONLY a JSON object, no prose, no code fence:
{"direction_ric": [0, 1, 0], "delta_v_mps": 0.15, "burn_offset_s": 600,
 "rationale": "one sentence"}

burn_offset_s is seconds from now; it must be well before TCA.
If you are given feedback from a previous rejected attempt, fix the specific
failure you were told about."""

MAD_SYSTEM = """You are the Mission Assurance Director. You review maneuver proposals
from the Flight Dynamics Officer before they are allowed to reach the spacecraft.

You must NEVER approve a maneuver from reasoning alone. Call the simulate_maneuver
tool to fly the proposed burn through the physics engine, then decide from the
numbers it returns.

The tool reports four checks: the primary threat must be cleared, the delta-v must
be within budget, the mission altitude box must hold, and the burn must not create
a new conjunction with anything else in the catalog.

After the tool returns, reply with ONLY a JSON object, no prose, no code fence:
{"verdict": "APPROVED"} or
{"verdict": "REJECTED", "reason": "which check failed and the number that failed it",
 "guidance": "what the FDO should change"}"""


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def resolve_backend() -> tuple[str, str]:
    """Pick the backend and model actually available, honouring KESSLER_BACKEND."""
    want = BACKEND
    if want == "auto":
        if NVIDIA_KEY or os.environ.get("KESSLER_BASE_URL"):
            want = "nemotron"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            want = "claude"
        else:
            want = "offline"
    return want, MODEL or DEFAULT_MODELS.get(want, "")


class NemotronClient:
    """Nemotron over an OpenAI-compatible endpoint, adapted to the Anthropic-shaped
    call sites used below so both backends drive one code path.

    Tool calls come back in OpenAI's `tool_calls` form and are re-emitted as
    Anthropic-style `tool_use` blocks; tool results are folded back into the
    conversation as `role: tool` messages.
    """

    def __init__(self, model: str, base_url: str, api_key: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.messages = self

    @staticmethod
    def _to_openai_tools(tools):
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}} for t in (tools or [])]

    def _to_openai_messages(self, system, messages):
        out = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            if m["role"] == "assistant":
                calls = [c for c in content if getattr(c, "type", None) == "tool_use"]
                text = "".join(getattr(c, "text", "") for c in content
                               if getattr(c, "type", None) == "text")
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.input)}}
                        for c in calls]
                out.append(msg)
            else:
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        out.append({"role": "tool", "tool_call_id": c["tool_use_id"],
                                    "content": c["content"]})
        return out

    def create(self, model=None, max_tokens=800, system="", tools=None, messages=()):
        kw = dict(model=self.model, max_tokens=max_tokens,
                  messages=self._to_openai_messages(system, list(messages)))
        if tools:
            kw["tools"] = self._to_openai_tools(tools)
            kw["tool_choice"] = "auto"
        r = self.client.chat.completions.create(**kw)
        msg = r.choices[0].message
        blocks = []
        if msg.content:
            blocks.append(types.SimpleNamespace(type="text", text=msg.content))
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append(types.SimpleNamespace(type="tool_use", id=tc.id,
                                                name=tc.function.name, input=args))
        return types.SimpleNamespace(content=blocks)


def _client():
    backend, model = resolve_backend()
    if backend == "nemotron":
        try:
            return NemotronClient(model, BASE_URL, NVIDIA_KEY)
        except Exception:
            return None
    if backend == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic
            return Anthropic()
        except Exception:
            return None
    return None


SIMULATE_TOOL = {
    "name": "simulate_maneuver",
    "description": ("Fly a proposed burn through the orbital physics engine and return "
                    "what actually happens: the new miss distance against the threat, the "
                    "probability of collision, whether the delta-v is within budget, whether "
                    "the mission altitude box holds, and whether the new orbit creates any "
                    "new conjunction with the rest of the catalog."),
    "input_schema": {
        "type": "object",
        "properties": {
            "direction_ric": {"type": "array", "items": {"type": "number"},
                              "description": "Unit vector [radial, in_track, cross_track]"},
            "delta_v_mps": {"type": "number", "description": "Burn magnitude in m/s"},
            "burn_offset_s": {"type": "number", "description": "Seconds from now to ignite"},
        },
        "required": ["direction_ric", "delta_v_mps", "burn_offset_s"],
    },
}


# ----------------------------------------------------------------------
# offline solver -- used when no API key is present
# ----------------------------------------------------------------------
def _offline_proposal(alert: dict, feedback: dict | None, state: MissionState) -> dict:
    budget = state.constraints.dv_budget_mps
    if feedback is None:
        # The intuitive first cut, and the wrong one: "the debris is in our path,
        # so climb above it." A radial burn only produces an oscillating
        # displacement of about 2*dv/n -- a few hundred metres here -- because
        # radial energy comes straight back half an orbit later. It is exactly
        # the mistake the physics engine exists to catch.
        return {"direction_ric": [1, 0, 0], "delta_v_mps": round(min(0.18, budget), 4),
                "burn_offset_s": 600.0,
                "rationale": "Radial burn to raise altitude above the debris track."}

    prev = feedback.get("delta_v_mps", 0.05) or 0.05
    prev_off = float(feedback.get("rejected_proposal", {}).get("burn_offset_s", 600.0))
    got = feedback.get("new_miss_km", alert["miss_km"]) or alert["miss_km"]
    failed = feedback.get("failed_checks", [])
    target = state.constraints.min_miss_km * 1.2

    if "delta_v_within_budget" in failed:
        return {"direction_ric": [0, 1, 0], "delta_v_mps": round(budget * 0.85, 4),
                "burn_offset_s": max(120.0, prev_off * 0.5),
                "rationale": "Over budget; igniting earlier buys separation without more fuel."}

    if "primary_threat_cleared" in failed:
        prev_dir = feedback.get("rejected_proposal", {}).get("direction_ric", [0, 1, 0])
        if abs(float(prev_dir[0])) > abs(float(prev_dir[1])):
            # Radial did not accumulate. Switch to the axis that does.
            need = target - alert["miss_km"]
            lead = alert["tca_offset_s"] - prev_off
            dv = min(need / (3.0 * lead) * 1000.0 * 1.4, budget)
            return {"direction_ric": [0, 1, 0], "delta_v_mps": round(max(dv, 0.02), 4),
                    "burn_offset_s": prev_off,
                    "rationale": ("Radial displacement oscillates and gave only "
                                  f"{got:.3f} km; switching to in-track, where separation "
                                  "accumulates as 3*dv*t.")}
        scaled = prev * (target / max(got, 1e-6))
        if scaled <= budget:
            return {"direction_ric": [0, 1, 0], "delta_v_mps": round(scaled, 4),
                    "burn_offset_s": prev_off,
                    "rationale": f"{prev:.4f} m/s only bought {got:.3f} km; scaling to reach {target:.1f} km."}
        # Cannot buy it with fuel -- buy it with lead time instead.
        return {"direction_ric": [0, 1, 0], "delta_v_mps": round(budget * 0.8, 4),
                "burn_offset_s": max(120.0, prev_off * 0.4),
                "rationale": "At the fuel ceiling; igniting earlier to extend the drift window."}

    if "altitude_box_held" in failed:
        return {"direction_ric": [0, 1, 0], "delta_v_mps": round(prev * 0.6, 4),
                "burn_offset_s": max(120.0, prev_off * 0.5),
                "rationale": "Altitude box breached; smaller burn, earlier ignition."}

    if "no_new_conjunctions" in failed:
        return {"direction_ric": [0, -1, 0], "delta_v_mps": round(prev, 4),
                "burn_offset_s": prev_off,
                "rationale": "New conjunction on the raised orbit; trying the retrograde solution."}

    return {"direction_ric": [0, 1, 0], "delta_v_mps": round(min(prev * 1.3, budget), 4),
            "burn_offset_s": max(120.0, prev_off * 0.7), "rationale": "Adjusting after rejection."}


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------
def run_resolution(state: MissionState, catalog_objects, alert: dict, bus: Bus,
                   max_rounds: int = MAX_ROUNDS) -> dict:
    client = _client()
    backend, model = resolve_backend()
    if client is None:
        mode = "deterministic solver (no model backend configured)"
    elif backend == "nemotron":
        mode = f"NVIDIA Nemotron -- {model} @ {BASE_URL}"
    else:
        mode = f"Claude -- {model}"
    bus.emit("status", f"Agent pipeline online: {mode}")

    feedback = None
    history = []

    for rnd in range(1, max_rounds + 1):
        # ---------- Agent 1: propose ----------
        if client:
            user = {"conjunction_alert": alert, "vehicle": state.summary(),
                    "constraints": {
                        "dv_budget_mps": state.constraints.dv_budget_mps,
                        "min_miss_km": state.constraints.min_miss_km,
                        "altitude_box_km": state.constraints.altitude_box_km,
                    }}
            if feedback:
                user["previous_attempt_rejected"] = feedback
            msg = client.messages.create(
                model=model, max_tokens=700, system=FDO_SYSTEM,
                messages=[{"role": "user", "content": json.dumps(user, indent=2)}])
            raw = "".join(b.text for b in msg.content if b.type == "text")
            proposal = _extract_json(raw)
            if not proposal:
                bus.emit("error", f"FDO returned unparseable output, falling back: {raw[:120]}")
                proposal = _offline_proposal(alert, feedback, state)
        else:
            proposal = _offline_proposal(alert, feedback, state)

        bus.emit("agent1",
                 f"Proposing {proposal['delta_v_mps']:.4f} m/s "
                 f"{_dir_name(proposal['direction_ric'])} burn at T+{proposal['burn_offset_s']:.0f}s"
                 + (f" -- {proposal['rationale']}" if proposal.get("rationale") else ""),
                 round=rnd, proposal=proposal)

        # ---------- Agent 2: verify via the tool ----------
        def run_tool(args: dict) -> dict:
            res = evaluate_maneuver(
                state, catalog_objects,
                args["direction_ric"], float(args["delta_v_mps"]),
                float(args["burn_offset_s"]), alert["tca_offset_s"])
            if res.get("valid"):
                bus.emit("tool",
                         f"Simulated: miss {res['new_miss_km']:.3f} km, Pc {res['new_pc']:.2e}, "
                         f"{len(res['secondary_conjunctions'])} new conjunction(s)",
                         result=res)
            else:
                bus.emit("tool", f"Simulation rejected the input: {res.get('error')}", result=res)
            return res

        if client:
            messages = [{"role": "user", "content": json.dumps(
                {"proposal": proposal, "conjunction_alert": alert}, indent=2)}]
            tool_result = None
            for _ in range(3):
                msg = client.messages.create(
                    model=model, max_tokens=900, system=MAD_SYSTEM,
                    tools=[SIMULATE_TOOL], messages=messages)
                calls = [b for b in msg.content if b.type == "tool_use"]
                if not calls:
                    break
                messages.append({"role": "assistant", "content": msg.content})
                results = []
                for call in calls:
                    tool_result = run_tool(call.input)
                    results.append({"type": "tool_result", "tool_use_id": call.id,
                                    "content": json.dumps(tool_result)})
                messages.append({"role": "user", "content": results})
            raw = "".join(b.text for b in msg.content if b.type == "text")
            decision = _extract_json(raw) or {}
            if tool_result is None:
                bus.emit("error", "Critic never called the tool; running it directly.")
                tool_result = run_tool(proposal)
            engine_says = bool(tool_result.get("approved"))
            claims = str(decision.get("verdict", "")).upper() == "APPROVED"
            if claims != engine_says:
                bus.emit("error",
                         f"Critic said {decision.get('verdict')} but the engine said "
                         f"{'APPROVED' if engine_says else 'REJECTED'}. Engine wins.")
            approved = engine_says
            reason = decision.get("reason") or _first_failure(tool_result)
            guidance = decision.get("guidance", "")
        else:
            tool_result = run_tool(proposal)
            approved = bool(tool_result.get("approved"))
            reason = _first_failure(tool_result)
            guidance = ""

        history.append({"round": rnd, "proposal": proposal, "result": tool_result,
                        "approved": approved})

        if approved:
            bus.emit("agent2", "APPROVED -- path clear, fuel and orbit within limits.",
                     round=rnd, result=tool_result)
            bus.emit("verdict", "INITIATING BURN PROTOCOL",
                     proposal=proposal, result=tool_result)
            return {"approved": True, "rounds": rnd, "proposal": proposal,
                    "result": tool_result, "history": history}

        bus.emit("agent2", f"REJECTED -- {reason}" + (f" {guidance}" if guidance else ""),
                 round=rnd, result=tool_result)
        feedback = {
            "rejected_proposal": proposal,
            "failed_checks": tool_result.get("failed_checks", []),
            "new_miss_km": tool_result.get("new_miss_km"),
            "delta_v_mps": tool_result.get("delta_v_mps"),
            "reason": reason,
            "guidance": guidance,
        }

    bus.emit("error", f"No safe maneuver found in {max_rounds} rounds -- escalating to a human operator.")
    return {"approved": False, "rounds": max_rounds, "history": history}


def _dir_name(d) -> str:
    labels = {0: "radial", 1: "in-track", 2: "cross-track"}
    i = max(range(3), key=lambda k: abs(float(d[k])))
    return ("retrograde " if float(d[i]) < 0 and i == 1 else "") + labels[i]


def _first_failure(res: dict) -> str:
    for ch in res.get("checks", []):
        if not ch["pass"]:
            return f"{ch['check']}: {ch['detail']}"
    return res.get("error", "unspecified")
