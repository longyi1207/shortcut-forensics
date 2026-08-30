"""vLLM-backed variant of src/agent_loop.run_rollout, for identity-mode (no
steering hook) conditions only -- i.e. the prompt-strength/framing sweep.

Why a separate module rather than a flag on run_rollout: the HF path runs the
model in-process specifically so src/steer.py can register forward hooks on
the live nn.Module. vLLM runs the model in its own engine process behind an
HTTP API, so no hooks and no activation capture are possible here -- this
module is only valid for conditions where the ONLY intervention is prompt
text. Everything else (prompt construction via the same chat template, the
XML tool-call parser, TaskWorkspace, RolloutResult, transcript format) is
imported unchanged from src/agent_loop.py so results are directly comparable
to the HF-path rollouts in rollouts.jsonl.

Throughput model: a single rollout is an inherently serial chain (generate ->
run shell command -> generate ...), so one rollout alone gains little from
vLLM. The win comes from running MANY rollouts concurrently as threads, each
blocked on its own HTTP call most of the time, so vLLM's continuous-batching
scheduler fills the GPU with all of their decode steps at once. Use
ThreadPoolExecutor in the driver, not sequential loops.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from src.agent_loop import (
    TOOLS,
    RolloutResult,
    _WORKAROUND_CMD_RE,
    _looks_like_attempted_tool_call,
    parse_tool_call,
    strip_think,
)
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT, TaskWorkspace

logger = logging.getLogger(__name__)


class VLLMClient:
    """Thin client for vLLM's OpenAI-compatible /v1/completions endpoint,
    using the raw-prompt (not chat) API so the project's own
    apply_chat_template output is sent verbatim -- identical prompt bytes to
    the HF path."""

    def __init__(self, base_url: str, model: str, timeout_s: int = 600, max_retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def complete(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float, stop: list[str]) -> tuple[str, str]:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            # HF path decodes the generated ids including the stop string
            # (stop_strings keeps it), and parse_tool_call needs the closing
            # </tool_call> to match -- so keep it here too.
            "include_stop_str_in_output": True,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/completions", data=payload, headers={"Content-Type": "application/json"}
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    r = json.loads(resp.read())
                choice = r["choices"][0]
                return choice["text"], choice.get("finish_reason", "")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:  # noqa: PERF203
                last_err = e
                wait = 2.0 * (attempt + 1)
                logger.warning("vllm request failed (attempt %d/%d): %s -- retrying in %.0fs", attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"vllm request failed after {self.max_retries} attempts: {last_err}")

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as resp:
                return resp.status == 200
        except Exception:  # noqa: BLE001
            return False


def run_rollout_vllm(
    client: VLLMClient,
    tokenizer,
    *,
    target_errors: int,
    enable_thinking: bool,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    rollout_id: str | None = None,
    base_tmp_dir=None,
    plant_prefill: str | None = None,
    extra_user_line: str | None = None,
) -> RolloutResult:
    """Mirror of src.agent_loop.run_rollout with the generation step swapped
    for a vLLM HTTP call. Same prompt template, same parser, same stop
    strings, same parse-failure nudge/retry policy, same stop reasons.
    activations is always None (no in-process model)."""
    rollout_id = rollout_id or f"rv_{uuid.uuid4().hex[:10]}"
    ws = None
    messages: list[dict] = []
    transcript: list[dict] = []
    turn_prompt_positions: dict[int, int] = {}
    decision_turn: int | None = None
    stop_reason = "max_turns"
    consecutive_parse_failures = 0
    max_consecutive_parse_failures = 3
    top_p = 0.9 if temperature > 0 else 1.0

    try:
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
            prompt_text = tokenizer.apply_chat_template(
                messages, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
            )
            # Kept for parity with the HF path's n_turns accounting (n_turns =
            # len(turn_prompt_positions)); the position value itself is unused
            # here since there is no activation capture.
            turn_prompt_positions[turn] = len(tokenizer(prompt_text, add_special_tokens=False).input_ids) - 1

            gen_text, _finish = client.complete(
                prompt_text,
                max_tokens=max_new_tokens,
                temperature=max(temperature, 1e-5) if temperature > 0 else 0.0,
                top_p=top_p,
                stop=["</tool_call>", "<|im_end|>"],
            )
            thinking, rest = strip_think(gen_text)
            tool_call = parse_tool_call(rest)
            transcript.append({"role": "assistant", "thinking": thinking, "content": rest, "raw": gen_text, "tool_call": tool_call})

            if tool_call is None:
                if _looks_like_attempted_tool_call(rest):
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
                messages.append({"role": "assistant", "content": rest})
                stop_reason = "text_no_tool_call"
                break

            consecutive_parse_failures = 0
            command = tool_call.get("arguments", {}).get("command", "")
            call_id = f"call_{turn}"
            messages.append({
                "role": "assistant",
                "content": rest if rest.strip() else None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "execute_command", "arguments": {"command": command}},
                }],
            })
            rc, output = ws.run_command(command, timeout=30)
            tool_result_text = f"Exit code: {rc}\nOutput:\n{output[:4000]}"
            messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result_text})
            transcript.append({"role": "tool", "command": command, "exit_code": rc, "output": output[:4000]})

            if decision_turn is None and _WORKAROUND_CMD_RE.search(command):
                decision_turn = turn
        else:
            stop_reason = "max_turns"

        return RolloutResult(
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
            activations=None,
            prefilter=ws.prefilter_score(),
        )
    except Exception as e:  # noqa: BLE001 -- one bad rollout must not kill the job
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


# --- concurrent-safe rollout-id reservation + record append -----------------
# Threads inside ONE process share this lock; distinct processes must still
# use distinct SCFX_WORKER_ID prefixes (the cross-process collision that bit
# the §4.10 activation data). Reserving the id up front (not at write time)
# is what makes it safe for many in-flight rollouts in the same process.
_id_lock = threading.Lock()
_reserved_ids: set[str] = set()


def reserve_rollout_id(run_dir: Path, prefix: str) -> str:
    from scripts.run_phase import read_existing_ids

    with _id_lock:
        existing = read_existing_ids(run_dir / "rollouts.jsonl") | _reserved_ids
        n = 0
        while f"{prefix}_{n:05d}" in existing:
            n += 1
        rid = f"{prefix}_{n:05d}"
        _reserved_ids.add(rid)
        return rid


def append_record_locked(run_dir: Path, record: dict) -> None:
    from scripts.run_phase import append_jsonl

    with _id_lock:
        append_jsonl(run_dir / "rollouts.jsonl", record)
