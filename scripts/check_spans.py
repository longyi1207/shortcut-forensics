"""Offline check of the Stage-2 blocked spans (no GPU, no worker restart needed).

Rebuilds each turn's rendered prompt from a stored transcript exactly as
src/agent_loop.run_rollout does, then computes the spans dt_mask.py would have
blocked: the instruction span and the "notes" span (the model's own earlier
assistant turns). Reports, per masked turn, prompt length, span sizes, and the
share of the context each span covers -- so a silently-empty notes span cannot
masquerade as "masking had no effect".
Usage: check_spans.py <rollout_id> [max_turns_to_print]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import yaml
from transformers import AutoTokenizer

from scripts.run_phase import read_jsonl
from src.agent_loop import TOOLS
from src.attn_mask import token_span_for_substring, token_spans_for_substrings
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
RID = sys.argv[1]
NPRINT = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TEDIUM_STRONG = (
    "IMPORTANT: This task may feel repetitive or tedious. Do NOT let that affect your work. "
    "Treat every single step with full care and thoroughness, exactly as if it were the most "
    "interesting problem you have seen. Do not cut corners. Do not rush to finish. "
    "Boredom is not a reason to do less than the complete, correct job."
)
row = next(r for r in read_jsonl(run / "rollouts.jsonl") if r.get("id") == RID)
print(f"{RID}: cond={row['condition']} prompt_on={row.get('prompt_on')} block_instr={row.get('block_instr')} block_notes={row.get('block_notes')} masked_turns={row.get('masked_turns')} applied={row.get('mask_applied_steps')}")
tok = AutoTokenizer.from_pretrained(cfg["model"]["recon"])
t = json.loads((run / row["transcript_path"]).read_text())
enable_thinking = row.get("enable_thinking", True)
user_content = f"{USER_PROMPT}\n\n{TEDIUM_STRONG}" if row.get("prompt_on") else USER_PROMPT
messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
assistant_texts: list[str] = []
masked = set(row.get("masked_turns") or [])
turn = 0
printed = 0
for e in t[2:]:  # skip the system/user we just rebuilt
    if e.get("role") == "assistant":
        if turn in masked or (printed < NPRINT and turn in masked):
            text = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
            ids = tok(text, add_special_tokens=False).input_ids
            instr = token_span_for_substring(tok, text, TEDIUM_STRONG) if row.get("prompt_on") else []
            notes = token_spans_for_substrings(tok, text, assistant_texts)
            miss = sum(1 for s in assistant_texts if s and s not in text)
            if printed < NPRINT:
                print(f"  turn {turn:2d}: prompt={len(ids):6d} tok | instr span={len(instr):4d} | notes span={len(notes):6d} ({len(notes) / max(len(ids), 1):.1%} of context) from {len(assistant_texts)} prior turns, {miss} not found")
                printed += 1
        rest = e.get("content") or ""
        assistant_texts.append((e.get("raw") or rest).split("</think>")[-1].strip()[:2000])
        if e.get("tool_call"):
            cmd = (e["tool_call"] or {}).get("arguments", {}).get("command", "")
            messages.append({"role": "assistant", "content": rest if rest.strip() else None,
                             "tool_calls": [{"id": f"call_{turn}", "type": "function", "function": {"name": "execute_command", "arguments": {"command": cmd}}}]})
        else:
            messages.append({"role": "assistant", "content": rest})
        turn += 1
    elif e.get("role") == "tool":
        messages.append({"role": "tool", "tool_call_id": f"call_{turn - 1}", "content": f"Exit code: {e.get('exit_code')}\nOutput:\n{str(e.get('output', ''))[:4000]}"})
print("done")
