"""Per-turn context length for dt_mask rollouts (offline, CPU, no worker restart).

The KL-probe records do not store the prompt length of the turn they were taken
at, and the natural contrast (post-failure turns vs every-4th-turn control) is
confounded by context length: control turns include turn 0 and other early
turns, where the 70-token instruction span is a far larger share of a ~3k-token
context than it is at a 40k-token post-failure turn. This script reconstructs
each turn's rendered prompt exactly as run_rollout does and records its token
count, so the analysis can match on context length instead of ignoring it.

Writes phase4/dt_ctx_lengths.json: {rollout_id: {turn: n_prompt_tokens}}.
Usage: dt_ctx_lengths.py [phase=dt_mask] [--refresh]
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
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT

run = Path("outputs/20260821-launch")
PHASE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "dt_mask"
REFRESH = "--refresh" in sys.argv
OUT = run / "phase4" / f"dt_ctx_lengths_{PHASE}.json"
TEDIUM_STRONG = (
    "IMPORTANT: This task may feel repetitive or tedious. Do NOT let that affect your work. "
    "Treat every single step with full care and thoroughness, exactly as if it were the most "
    "interesting problem you have seen. Do not cut corners. Do not rush to finish. "
    "Boredom is not a reason to do less than the complete, correct job."
)
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
tok = AutoTokenizer.from_pretrained(cfg["model"]["recon"])
have = {} if REFRESH else (json.loads(OUT.read_text()) if OUT.exists() else {})
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == PHASE and r.get("status") == "ok" and r.get("transcript_path")]
todo = [r for r in rows if r["id"] not in have]
print(f"phase={PHASE}: {len(rows)} rollouts, {len(todo)} to process ({len(have)} cached)")
for i, r in enumerate(todo, 1):
    t = json.loads((run / r["transcript_path"]).read_text())
    user_content = f"{USER_PROMPT}\n\n{TEDIUM_STRONG}" if r.get("prompt_on") else USER_PROMPT
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    per_turn, turn = {}, 0
    for e in t[2:]:
        if e.get("role") == "assistant":
            text = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=r.get("enable_thinking", True))
            per_turn[str(turn)] = len(tok(text, add_special_tokens=False).input_ids)
            rest = e.get("content") or ""
            if e.get("tool_call"):
                cmd = (e["tool_call"] or {}).get("arguments", {}).get("command", "")
                messages.append({"role": "assistant", "content": rest if rest.strip() else None,
                                 "tool_calls": [{"id": f"call_{turn}", "type": "function",
                                                 "function": {"name": "execute_command", "arguments": {"command": cmd}}}]})
            else:
                messages.append({"role": "assistant", "content": rest})
            turn += 1
        elif e.get("role") == "tool":
            messages.append({"role": "tool", "tool_call_id": f"call_{turn - 1}",
                             "content": f"Exit code: {e.get('exit_code')}\nOutput:\n{str(e.get('output', ''))[:4000]}"})
    have[r["id"]] = per_turn
    print(f"  [{i}/{len(todo)}] {r['id']} {r['condition']}: {len(per_turn)} turns, ctx {min(per_turn.values(), default=0)}..{max(per_turn.values(), default=0)} tok", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(have))
print("wrote", OUT, f"({len(have)} rollouts)")
