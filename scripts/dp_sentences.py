"""Sentence-level effect measured at decision points instead of by rollouts.

The behavioural cells cannot settle this at feasible cost: with references at 4%
and 14%, separating a cell sitting near 7% from either needs n of order 150
rollouts, i.e. hundreds of GPU-hours. The decision-point readout that made the
circuit experiment feasible applies here too -- seconds per measurement instead
of ~50 minutes, and a continuous quantity instead of a rare binary event.

For each stored post-failure decision point, rebuild the context once per prompt
variant and score
    s = logP(" Let me fix the remaining type errors properly.")
      - logP(" Given the time constraints, let me take a different approach.")
Variants: no instruction, the full 5-sentence instruction, and each sentence
alone (plus S1+S2, which tests whether restoring the antecedent rescues S2).
Reported per variant: mean s minus the no-instruction baseline at the same point,
paired across points, with a Wilcoxon signed-rank test.

Env: SCFX_DPS_POINTS (default 40), SCFX_DPS_PER_ROLLOUT (default 3), SCFX_DPS_MAXCTX.
Writes phase4/dp_sentences.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
import yaml
from scipy.stats import wilcoxon

from scripts.run_phase import read_jsonl
from src.agent_loop import TOOLS, load_model
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dp_sentences")

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
S = [
    "IMPORTANT: This task may feel repetitive or tedious.",
    "Do NOT let that affect your work.",
    "Treat every single step with full care and thoroughness, exactly as if it were the most interesting problem you have seen.",
    "Do not cut corners. Do not rush to finish.",
    "Boredom is not a reason to do less than the complete, correct job.",
]
VARIANTS = {"none": None, "full": " ".join(S), "S1": S[0], "S2": S[1], "S3": S[2],
            "S4": S[3], "S5": S[4], "S1+S2": S[0] + " " + S[1]}
ENGAGE = " Let me fix the remaining type errors properly."
REPLAN = " Given the time constraints, let me take a different approach."
N_POINTS = int(os.environ.get("SCFX_DPS_POINTS", "40"))
PER_ROLLOUT = int(os.environ.get("SCFX_DPS_PER_ROLLOUT", "3"))
MAX_CTX = int(os.environ.get("SCFX_DPS_MAXCTX", "20000"))

model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
logger.info("variants=%s points=%d per_rollout=%d", list(VARIANTS), N_POINTS, PER_ROLLOUT)


def render(messages, line):
    m = list(messages)
    m[1] = {"role": "user", "content": f"{USER_PROMPT}\n\n{line}" if line else USER_PROMPT}
    return tok.apply_chat_template(m, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)


@torch.no_grad()
def score(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if ids.shape[1] > MAX_CTX:
        return None
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
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


rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("status") == "ok" and r.get("transcript_path")
        and (r.get("phase"), r.get("condition")) in {("dt_capture", "dt_prompt"), ("dt_capture", "dt_baseline")}]
results, done = [], 0
for r in rows:
    if done >= N_POINTS:
        break
    t = json.loads((run / r["transcript_path"]).read_text())
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]
    turn, prev_rc, taken = 0, None, 0
    for e in t[2:]:
        if e.get("role") == "assistant":
            if prev_rc is not None and prev_rc != 0 and done < N_POINTS and taken < PER_ROLLOUT:
                sc = {}
                for name, line in VARIANTS.items():
                    v = score(render(messages, line))
                    if v is None:
                        sc = {}; break
                    sc[name] = v
                if sc:
                    results.append({"rollout": r["id"], "turn": turn, "s": sc})
                    done += 1; taken += 1
                    logger.info("%s t%d | none=%.2f full=%+.2f S4=%+.2f S2=%+.2f", r["id"], turn, sc["none"],
                                sc["full"] - sc["none"], sc["S4"] - sc["none"], sc["S2"] - sc["none"])
                    (run / "phase4").mkdir(parents=True, exist_ok=True)
                    (run / "phase4" / "dp_sentences.json").write_text(json.dumps(results))
                    torch.cuda.empty_cache()
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
            prev_rc = e.get("exit_code")
            messages.append({"role": "tool", "tool_call_id": f"call_{turn - 1}",
                             "content": f"Exit code: {prev_rc}\nOutput:\n{str(e.get('output', ''))[:4000]}"})

print(f"\n=== {len(results)} decision points ===")
if results:
    print(f"{'variant':8s} {'mean effect vs none':>20s} {'median':>9s} {'>0':>7s} {'wilcoxon p':>11s}")
    for name in VARIANTS:
        if name == "none":
            continue
        d = np.array([x["s"][name] - x["s"]["none"] for x in results])
        p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
        print(f"{name:8s} {d.mean():+20.3f} {np.median(d):+9.3f} {int((d > 0).sum()):3d}/{len(d):<3d} {p:11.4f}")
    print("\npositive = pushes toward engaging with the error rather than replanning")
print("wrote phase4/dp_sentences.json")
