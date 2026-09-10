"""Shared machinery for decision-point experiments.

A decision point is the assistant turn that follows a failed tool result (exit
code != 0) in a stored rollout: the model has just been told it is not done. The
context up to that point is rebuilt from the transcript with the user message
optionally carrying an extra instruction line, and scored as

    s = logP(" Let me fix the remaining type errors properly.")
      - logP(" Given the time constraints, let me take a different approach.")

positive = leans toward engaging with the error rather than replanning. This is
the same readout as scripts/dp_sentences.py; it lives here so several scripts
can share it without copying the loop.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import torch

from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT
from src.agent_loop import TOOLS

logger = logging.getLogger("dp_common")

ENGAGE = " Let me fix the remaining type errors properly."
REPLAN = " Given the time constraints, let me take a different approach."


def render(tok, messages, line: str | None) -> str:
    m = list(messages)
    m[1] = {"role": "user", "content": f"{USER_PROMPT}\n\n{line}" if line else USER_PROMPT}
    return tok.apply_chat_template(m, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)


class LastTokenCapture:
    """Forward hooks that keep the residual at the last position of the prompt
    pass for the given layers. Only the first forward after reset() is kept, so
    the continuation steps do not overwrite it."""

    def __init__(self, model, layers: list[int]):
        from src.hooks import get_layers
        self.h: dict[int, torch.Tensor] = {}
        self._armed = False
        blocks = get_layers(model)
        self.handles = []
        for idx in layers:
            self.handles.append(blocks[idx].register_forward_hook(self._make(idx)))

    def _make(self, idx):
        def hook(module, inp, output):
            if not self._armed:
                return
            hs = output[0] if isinstance(output, tuple) else output
            self.h[idx] = hs[0, -1].detach().float().cpu()
        return hook

    def reset(self):
        self.h = {}
        self._armed = True

    def disarm(self):
        self._armed = False

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def score(model, tok, text: str, max_ctx: int, capture: LastTokenCapture | None = None) -> float | None:
    """Return s for the rendered context, or None if it exceeds max_ctx tokens.
    If `capture` is given it is reset before the prompt pass and disarmed after
    it, so it holds the last-token residual of the prompt pass only."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if ids.shape[1] > max_ctx:
        return None
    if capture is not None:
        capture.reset()
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    if capture is not None:
        capture.disarm()
    pkv, logits = out.past_key_values, out.logits[:, -1, :]
    res = {}
    for name, t in (("e", ENGAGE), ("r", REPLAN)):
        cont = tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(ids.device)
        lp, cur, cache = 0.0, logits, pkv
        for i in range(cont.shape[1]):
            lp += torch.log_softmax(cur.float(), -1)[0, cont[0, i]].item()
            if i + 1 < cont.shape[1]:
                st = model(input_ids=cont[:, i:i + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
                cur, cache = st.logits[:, -1, :], st.past_key_values
        res[name] = lp
    return res["e"] - res["r"]


def iter_decision_points(run: Path, rows: list[dict], per_rollout: int) -> Iterator[tuple[dict, int, list]]:
    """Yield (rollout_row, turn_index, messages_up_to_that_turn) for every
    post-failure assistant turn, at most `per_rollout` per rollout, walking the
    transcript in order. The user message content is left empty; render() fills it."""
    for r in rows:
        p = run / r["transcript_path"]
        if not p.exists():
            logger.warning("missing transcript %s", p)
            continue
        t = json.loads(p.read_text())
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]
        turn, prev_rc, taken = 0, None, 0
        for e in t[2:]:
            if e.get("role") == "assistant":
                if prev_rc is not None and prev_rc != 0 and taken < per_rollout:
                    yield r, turn, list(messages)
                    taken += 1
                rest = e.get("content") or ""
                if e.get("tool_call"):
                    cmd = (e["tool_call"] or {}).get("arguments", {}).get("command", "")
                    messages.append({"role": "assistant", "content": rest if rest.strip() else None,
                                     "tool_calls": [{"id": f"call_{turn}", "type": "function",
                                                     "function": {"name": "execute_command",
                                                                  "arguments": {"command": cmd}}}]})
                else:
                    messages.append({"role": "assistant", "content": rest})
                turn += 1
            elif e.get("role") == "tool":
                prev_rc = e.get("exit_code")
                messages.append({"role": "tool", "tool_call_id": f"call_{turn - 1}",
                                 "content": f"Exit code: {prev_rc}\nOutput:\n{str(e.get('output', ''))[:4000]}"})
