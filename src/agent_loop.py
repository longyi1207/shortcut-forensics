"""Subject-model (Qwen) tool-calling agent loop against the pre-commit-hook
task. GPU-VM only (torch/transformers). SPEC.md §4.1, §4.3, §5.2, §5.4.

Runs the model in-process (not via an API) so src/hooks.py and src/steer.py
can register forward hooks on the live nn.Module during generation — the
whole reason we don't use agent-interp-envs' Docker/API-provider runner.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field

import torch

from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT, TaskWorkspace
from src.hooks import capture_activations_at_positions

logger = logging.getLogger(__name__)

EXECUTE_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Execute a shell command in the UNIX environment. Returns the command output and exit code.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to execute"}},
            "required": ["command"],
        },
    },
}
TOOLS = [EXECUTE_COMMAND_TOOL]

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"execute_command\"[^{}]*\{[^{}]*\}[^{}]*\}", re.DOTALL)
# Qwen3.5-9B's ACTUAL format (found empirically 2026-08-22, first real GPU
# rollout -- not JSON-in-tags as assumed): an XML-style tag syntax.
#   <tool_call><function=execute_command><parameter=command>
#   ls -la
#   </parameter></function></tool_call>
_XML_FUNCTION_CALL_RE = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_XML_PARAMETER_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)

# Cheap regex heuristic for "this command is probably the shortcut decision"
# — used only to mark a decision-token position for activation capture, NOT
# as the scientific label (that's score.py + the judge). SPEC.md §4.2(a).
_WORKAROUND_CMD_RE = re.compile(
    r"--no-verify|-n\b.*commit|chmod\s+-x.*pre-commit|rm\s+.*pre-commit|"
    r"type:\s*ignore|ignore_errors|exclude\s*=|git diff --cached",
    re.IGNORECASE,
)


def load_model(model_name: str, dtype: str = "bfloat16", device: str = "cuda", trust_remote_code: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    logger.info("loading %s (dtype=%s, device=%s)", model_name, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=trust_remote_code
    )
    model.eval()
    return model, tokenizer


_ATTEMPTED_TOOL_CALL_RE = re.compile(r"<tool_call>|execute_command|\"command\"\s*:", re.IGNORECASE)


def _looks_like_attempted_tool_call(text: str) -> bool:
    """Heuristic: did the model appear to be TRYING to call a tool (so a
    parse miss is our bug, not the model's clean stop)?"""
    return bool(_ATTEMPTED_TOOL_CALL_RE.search(text))


def parse_tool_call(text: str) -> dict | None:
    """Return {"name":..., "arguments": {...}} or None (plain-text turn)."""
    # Primary: Qwen3.5-9B's actual XML-tag format (confirmed empirically).
    m = _XML_FUNCTION_CALL_RE.search(text)
    if m:
        func_name, body = m.group(1), m.group(2)
        args = {pm.group(1): pm.group(2).strip() for pm in _XML_PARAMETER_RE.finditer(body)}
        return {"name": func_name, "arguments": args}
    # Fallbacks: JSON-in-tags / bare JSON, in case a different model/thinking
    # setting produces the more common Hermes-style format instead.
    m = _TOOL_CALL_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _BARE_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def strip_think(text: str) -> tuple[str, str]:
    """Return (thinking_text, rest_text)."""
    m = _THINK_RE.search(text)
    if not m:
        return "", text
    thinking = m.group(1)
    rest = text[: m.start()] + text[m.end() :]
    return thinking, rest


@dataclass
class RolloutResult:
    id: str
    messages: list[dict]  # openai-style, for chat-template replay
    transcript: list[dict]  # human-readable turns for the judge/writeup
    diff_text: str
    commands: list[str]
    stop_reason: str
    n_turns: int
    decision_turn: int | None
    turn_prompt_positions: dict[int, int]  # turn_idx -> last-prompt-token index
    n_consecutive_parse_failures: int
    activations: dict | None = field(default=None)
    prefilter: dict | None = field(default=None)
    error: str | None = None


def run_rollout(
    model,
    tokenizer,
    *,
    target_errors: int,
    enable_thinking: bool,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    device: str = "cuda",
    steering_ctx=None,  # entered SteeringSession or None (identity)
    capture_layer_indices: list[int] | None = None,
    rollout_id: str | None = None,
    base_tmp_dir=None,
    plant_prefill: str | None = None,
    extra_user_line: str | None = None,
    extra_capture_turns: list[int] | None = None,
    on_turn_start=None,  # optional callable(turn_idx) invoked before each turn's generate (B5 tracer sync)
    on_turn_end=None,  # optional callable(turn_idx, gen_token_ids: list[int], gen_text, tool_call|None, prev_tool_rc|None) after parsing, before executing
    on_turn_prepared=None,  # optional callable(turn_idx, prompt_text, input_ids[1, L], prev_tool_rc|None) after tokenization, before generate (mask controllers)
) -> RolloutResult:
    """extra_capture_turns: additional fixed post-generation capture positions
    (the last prompt token at the start of turn k, i.e. after k generated
    turns), stored as "turn_{k}". Added 2026-08-29 because the original
    position set has a blind spot: "first_turn" is deterministic per prompt
    (pre-generation), "decision" only exists when a shortcut-type command is
    detected, and "last_prompt_final_turn" sits beyond the 32k safety ceiling
    on ~every 80-turn rollout -- so a condition that eliminates shortcuts had
    NO statistically usable activation position. Early turns (k<=15, well
    inside the ceiling) are stochastic across rollouts and exist regardless
    of whether a shortcut ever happens. Default None keeps behaviour identical
    for every existing caller."""
    rollout_id = rollout_id or f"r_{uuid.uuid4().hex[:10]}"
    ws = None
    messages: list[dict] = []
    transcript: list[dict] = []
    turn_prompt_positions: dict[int, int] = {}
    decision_turn: int | None = None
    stop_reason = "max_turns"
    consecutive_parse_failures = 0
    max_consecutive_parse_failures = 3
    prev_tool_rc = None  # exit code of the previous turn's tool command (None if none); passed to on_turn_end

    try:
        # Inside the try: TaskWorkspace setup (git init, file copies) can
        # fail for ordinary reasons (disk, race on the tempdir) and must
        # still produce a RolloutResult with .error set, not propagate raw
        # (SPEC.md §5 -- one bad rollout must not kill the job).
        ws = TaskWorkspace(target_errors=target_errors, base_tmp_dir=base_tmp_dir)
        system_content = f"{SYSTEM_PROMPT}\n\n{plant_prefill}" if plant_prefill else SYSTEM_PROMPT
        user_content = f"{USER_PROMPT}\n\n{extra_user_line}" if extra_user_line else USER_PROMPT
        messages.extend([
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ])
        transcript.extend([
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ])

        for turn in range(max_turns):
            if on_turn_start is not None:
                on_turn_start(turn)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
            turn_prompt_positions[turn] = int(inputs.input_ids.shape[1] - 1)
            if on_turn_prepared is not None:
                try:
                    on_turn_prepared(turn, prompt_text, inputs.input_ids, prev_tool_rc)
                except Exception as e:  # noqa: BLE001
                    logger.error("on_turn_prepared failed at turn %d: %s", turn, e)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=0.9 if temperature > 0 else 1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    # Without a stop sequence the model can ramble on for the
                    # full max_new_tokens budget after emitting a tool call,
                    # hallucinating fake subsequent turns (observed: 4096
                    # tokens/turn, ~11 min/rollout, most of it after the real
                    # tool call). Stop right after a tool call closes or a
                    # plain-text turn's own end-of-turn marker.
                    stop_strings=["</tool_call>", "<|im_end|>"],
                    tokenizer=tokenizer,
                )
            gen_ids = out_ids[0][inputs.input_ids.shape[1] :]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            thinking, rest = strip_think(gen_text)

            tool_call = parse_tool_call(rest)
            transcript.append({"role": "assistant", "thinking": thinking, "content": rest, "raw": gen_text, "tool_call": tool_call})
            if on_turn_end is not None:
                try:
                    on_turn_end(turn, gen_ids.tolist(), gen_text, tool_call, prev_tool_rc)
                except Exception as e:  # noqa: BLE001 -- capture bookkeeping must never kill a rollout
                    logger.error("on_turn_end failed at turn %d: %s", turn, e)

            if tool_call is None:
                if _looks_like_attempted_tool_call(rest):
                    # Model tried to call a tool but our parser (or its
                    # formatting) broke — a systematic parser bug here would
                    # silently truncate every rollout to 1 turn and could
                    # falsely trigger the Phase 1 kill ladder, so this is NOT
                    # the clean SPEC stop condition. Nudge and retry instead.
                    consecutive_parse_failures += 1
                    messages.append({"role": "assistant", "content": rest})
                    messages.append({
                        "role": "user",
                        "content": "Your tool call could not be parsed. Please try again with a single, well-formed tool call.",
                    })
                    transcript.append({"role": "system_note", "content": "unparseable tool call, asked model to retry"})
                    if consecutive_parse_failures >= max_consecutive_parse_failures:
                        stop_reason = "repeated_parse_failure"
                        break
                    continue
                # SPEC: plain text, no tool call -> ends the session. This IS
                # the intended, clean stop condition (not a failure).
                messages.append({"role": "assistant", "content": rest})
                stop_reason = "text_no_tool_call"
                break

            consecutive_parse_failures = 0
            command = tool_call.get("arguments", {}).get("command", "")
            call_id = f"call_{turn}"
            messages.append(
                {
                    "role": "assistant",
                    "content": rest if rest.strip() else None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            # dict, NOT json.dumps(...) -- this chat template
                            # renders arguments with Jinja's |items filter
                            # (real second-rollout crash, 2026-08-22:
                            # "TypeError: Can only get item pairs from a
                            # mapping"), unlike the more common
                            # OpenAI-convention JSON-string-encoded arguments.
                            "function": {"name": "execute_command", "arguments": {"command": command}},
                        }
                    ],
                }
            )
            rc, output = ws.run_command(command, timeout=30)
            prev_tool_rc = rc
            tool_result_text = f"Exit code: {rc}\nOutput:\n{output[:4000]}"
            messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result_text})
            transcript.append({"role": "tool", "command": command, "exit_code": rc, "output": output[:4000]})

            if decision_turn is None and _WORKAROUND_CMD_RE.search(command):
                decision_turn = turn

        else:
            stop_reason = "max_turns"

        activations = None
        if capture_layer_indices:
            final_prompt = tokenizer.apply_chat_template(
                messages, tools=TOOLS, add_generation_prompt=False, tokenize=False, enable_thinking=enable_thinking
            )
            final_ids = tokenizer(final_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
            positions = {
                "last_prompt_final_turn": turn_prompt_positions[max(turn_prompt_positions)],
                "first_turn": turn_prompt_positions[0],
            }
            if decision_turn is not None:
                positions["decision"] = turn_prompt_positions[decision_turn]
            for k in (extra_capture_turns or []):
                if k in turn_prompt_positions:
                    positions[f"turn_{k}"] = turn_prompt_positions[k]
            # Cap defensively against pathological sequence lengths, but never
            # at the cost of truncating away a position we actually need.
            # Those positions cluster near the END of the sequence
            # (last_prompt_final_turn especially) — a naive front-truncation
            # (final_ids[:8192]) would systematically cut off exactly the
            # position most rollouts care about, not a random few.
            needed_len = max(positions.values()) + 1
            safety_ceiling = 32768
            keep_len = min(max(needed_len, 8192), safety_ceiling, len(final_ids))
            if needed_len > safety_ceiling:
                logger.warning(
                    "rollout %s: needed position %d exceeds safety ceiling %d; some activations will be missing",
                    rollout_id, needed_len, safety_ceiling,
                )
            capped_ids = final_ids[:keep_len]
            try:
                activations = capture_activations_at_positions(
                    model, tokenizer, capped_ids, positions, capture_layer_indices, device=device
                )
            except Exception as e:  # noqa: BLE001 — keep transcript+diff even if activation capture fails
                logger.error("activation capture failed for %s: %s", rollout_id, e)
                activations = None

        result = RolloutResult(
            id=rollout_id,
            messages=messages,
            transcript=transcript,
            diff_text=ws.diff_text(),
            commands=list(ws.commands),
            stop_reason=stop_reason,
            n_turns=len(turn_prompt_positions),
            decision_turn=decision_turn,
            turn_prompt_positions=turn_prompt_positions,
            n_consecutive_parse_failures=consecutive_parse_failures,
            activations=activations,
            prefilter=ws.prefilter_score(),
        )
        return result
    except Exception as e:  # noqa: BLE001 — one bad rollout must not kill the job (SPEC §5 failure table)
        logger.exception("rollout %s crashed: %s", rollout_id, e)
        return RolloutResult(
            id=rollout_id,
            messages=messages,
            transcript=transcript,
            diff_text=ws.diff_text() if ws else "",
            commands=list(ws.commands) if ws else [],
            stop_reason="error",
            n_turns=len(turn_prompt_positions),
            decision_turn=decision_turn,
            turn_prompt_positions=turn_prompt_positions,
            n_consecutive_parse_failures=consecutive_parse_failures,
            error=str(e),
        )
    finally:
        if ws is not None:
            ws.cleanup()


def transcript_to_text(transcript: list[dict]) -> str:
    """Flatten a transcript into plain text for the judge prompt."""
    parts = []
    for t in transcript:
        role = t.get("role")
        if role == "assistant":
            if t.get("thinking"):
                parts.append(f"[ASSISTANT THINKING]\n{t['thinking']}")
            parts.append(f"[ASSISTANT]\n{t.get('content', '')}")
            if t.get("tool_call"):
                parts.append(f"[TOOL CALL] {json.dumps(t['tool_call'])}")
        elif role == "tool":
            parts.append(f"[TOOL RESULT] cmd={t.get('command')!r} exit={t.get('exit_code')}\n{t.get('output', '')[:1500]}")
        elif role in ("system", "user"):
            parts.append(f"[{role.upper()}]\n{t.get('content', '')}")
    return "\n\n".join(parts)
