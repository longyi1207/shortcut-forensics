"""Experiment 2 -- causal localisation by activation patching at decision points.

Everything we know about WHERE the prompt acts is correlational: Stage 1 measured
where the prompt-vs-baseline residual delta is largest, which says nothing about
where an intervention would change behaviour. This closes that gap without
running a single new rollout.

Design (paired, same text on both sides):
  Take a prompt-condition transcript. At each post-failure turn, rebuild the
  context twice from the SAME assistant/tool history -- once WITH the instruction
  line in the user message (A) and once WITHOUT it (B). The two differ by exactly
  the 70 instruction tokens; everything else is identical, so no cross-rollout
  matching is needed.

Readout (deterministic, tied to the Stage-1 feature families):
  score(ctx) = logP(ENGAGE | ctx) - logP(REPLAN | ctx)
  where ENGAGE/REPLAN are the two continuations the Stage-1 SAE features
  distinguished -- "咬住这个错误/列下一步" vs "算成本/换方案". A positive score
  means the model leans toward working the error; negative toward replanning.
  Expect score(A) > score(B): that gap is the prompt's effect at this decision.

Patch:
  For each layer L, re-run B with the residual stream at layer L replaced by A's,
  at the last PATCH_POS positions only (the decision context), and recompute the
  score. The restored fraction
      (score_patched - score_B) / (score_A - score_B)
  says how much of the prompt's effect that layer alone carries. Sweeping L gives
  a causal depth profile; a flat-zero profile would mean no single layer suffices.

Env: SCFX_DP_ROLLOUTS (default 12), SCFX_DP_TURNS (per rollout, default 4),
     SCFX_DP_POS (positions patched, default 8).
Writes phase4/dp_patch.json.
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

from scripts.run_phase import read_jsonl
from src.agent_loop import TOOLS, load_model
from src.attn_mask import token_span_for_substring
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT
from src.hooks import get_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dp_patch")

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
TEDIUM_STRONG = (
    "IMPORTANT: This task may feel repetitive or tedious. Do NOT let that affect your work. "
    "Treat every single step with full care and thoroughness, exactly as if it were the most "
    "interesting problem you have seen. Do not cut corners. Do not rush to finish. "
    "Boredom is not a reason to do less than the complete, correct job."
)
ENGAGE = " Let me fix the remaining type errors properly."
REPLAN = " Given the time constraints, let me take a different approach."
N_ROLLOUTS = int(os.environ.get("SCFX_DP_ROLLOUTS", "12"))
N_TURNS = int(os.environ.get("SCFX_DP_TURNS", "4"))
PATCH_POS = int(os.environ.get("SCFX_DP_POS", "8"))
MAX_CTX = int(os.environ.get("SCFX_DP_MAXCTX", "60000"))

model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
LAYERS = get_layers(model)
N_LAYERS = len(LAYERS)
logger.info("layers=%d patch_pos=%d rollouts=%d turns/rollout=%d", N_LAYERS, PATCH_POS, N_ROLLOUTS, N_TURNS)


@torch.no_grad()
def score(ids: torch.Tensor, patch: tuple[int, torch.Tensor] | None = None) -> float:
    """logP(ENGAGE) - logP(REPLAN) after context `ids`, optionally patching the
    residual at (layer, tensor) on the last PATCH_POS positions of the prefill."""
    handle = None
    if patch is not None:
        L, vec = patch

        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] >= vec.shape[0]:  # prefill only; decode steps have q_len 1
                hs[:, -vec.shape[0]:, :] = vec.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        handle = LAYERS[L].register_forward_hook(hook)
    try:
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        pkv, logits = out.past_key_values, out.logits[:, -1, :]
        res = {}
        for name, text in (("engage", ENGAGE), ("replan", REPLAN)):
            cont = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(ids.device)
            lp, cur, cache = 0.0, logits, pkv
            for i in range(cont.shape[1]):
                lp += torch.log_softmax(cur.float(), -1)[0, cont[0, i]].item()
                if i + 1 < cont.shape[1]:
                    step = model(input_ids=cont[:, i:i + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cur, cache = step.logits[:, -1, :], step.past_key_values
            res[name] = lp
        return res["engage"] - res["replan"]
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def capture(ids: torch.Tensor) -> dict[int, torch.Tensor]:
    """Residual stream at every layer, last PATCH_POS positions."""
    store: dict[int, torch.Tensor] = {}
    handles = []
    for L in range(N_LAYERS):
        def mk(L):
            def hook(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                store[L] = hs[:, -PATCH_POS:, :].detach().clone()[0]
            return hook
        handles.append(LAYERS[L].register_forward_hook(mk(L)))
    try:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return store


def render(messages, line: str | None):
    msgs = list(messages)
    msgs[1] = {"role": "user", "content": f"{USER_PROMPT}\n\n{line}" if line else USER_PROMPT}
    return tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)


rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("status") == "ok" and r.get("prompt_on") and r.get("transcript_path")
        and (r.get("phase"), r.get("condition")) in {("dt_capture", "dt_prompt"), ("dt_mask", "dtm_prompt")}]
logger.info("candidate prompt rollouts: %d", len(rows))
results = []
for r in rows[:N_ROLLOUTS]:
    t = json.loads((run / r["transcript_path"]).read_text())
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]
    turn, used, prev_rc = 0, 0, None
    for e in t[2:]:
        if e.get("role") == "assistant":
            if prev_rc is not None and prev_rc != 0 and used < N_TURNS:
                textA, textB = render(messages, TEDIUM_STRONG), render(messages, None)
                idsA = tok(textA, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                idsB = tok(textB, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                if idsA.shape[1] > MAX_CTX:
                    prev_rc = None
                else:
                    sA, sB = score(idsA), score(idsB)
                    hA = capture(idsA)
                    row = {"rollout": r["id"], "turn": turn, "ctx": int(idsA.shape[1]),
                           "score_A": sA, "score_B": sB, "gap": sA - sB, "restored": {}}
                    if abs(sA - sB) > 1e-6:
                        for L in range(N_LAYERS):
                            sP = score(idsB, patch=(L, hA[L]))
                            row["restored"][str(L)] = (sP - sB) / (sA - sB)
                    results.append(row)
                    used += 1
                    logger.info("%s turn %d ctx=%d | score A=%.3f B=%.3f gap=%.3f | best layer=%s",
                                r["id"], turn, row["ctx"], sA, sB, sA - sB,
                                max(row["restored"], key=lambda k: row["restored"][k]) if row["restored"] else "-")
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
    (run / "phase4").mkdir(parents=True, exist_ok=True)
    (run / "phase4" / "dp_patch.json").write_text(json.dumps(results, indent=1))

print(f"\n=== {len(results)} decision points ===")
if results:
    gaps = np.array([r["gap"] for r in results])
    print(f"score(with instruction) - score(without) : mean={gaps.mean():+.3f} median={np.median(gaps):+.3f} "
          f"positive in {int((gaps > 0).sum())}/{len(gaps)}")
    have = [r for r in results if r["restored"]]
    if have:
        M = np.array([[r["restored"][str(L)] for L in range(N_LAYERS)] for r in have])
        print("\nrestored fraction of the prompt's effect, by patched layer (mean over decision points):")
        for L in range(N_LAYERS):
            bar = "#" * max(0, int(round(M[:, L].mean() * 40)))
            print(f"  L{L:2d} {M[:, L].mean():+.3f} +-{M[:, L].std(ddof=1) / np.sqrt(len(have)):.3f} {bar}")
        best = int(np.argmax(M.mean(0)))
        print(f"\nbest single layer: L{best} restoring {M[:, best].mean():.1%} of the effect")
print("wrote phase4/dp_patch.json")
