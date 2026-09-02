"""Circuit level: which individual components write the instruction's influence?

Stage 3a showed nothing reads the instruction at the decision (0.53% of attention),
so the circuit cannot be there -- it must be at PREFILL, when the instruction is
read and its influence is written into what persists. This measures that, at the
finest granularity the architecture has:

  for each component c in {each of the 128 full-attention heads,
                           each of the 32 MLPs,
                           each of the 24 GatedDeltaNet blocks}
      zero c's output AT THE INSTRUCTION SPAN POSITIONS ONLY
      -> how much of the prompt's effect on the decision disappears?

Readout is the paired score from dp_patch: with the same assistant/tool history,
build the context WITH the instruction (A) and WITHOUT it (B), and score
    s(ctx) = logP(" Let me fix the remaining type errors properly.")
           - logP(" Given the time constraints, let me take a different approach.")
The prompt's effect at that decision is gap = s(A) - s(B) > 0. Ablating component c
inside A gives s_c, and
    attribution(c) = (s(A) - s_c) / gap
is the fraction of the effect that c carries. 1.0 = c alone accounts for it, 0 = c
is irrelevant, negative = c works against it.

CONTROL, built in from the start: every component is ALSO ablated over a
length-matched span of ordinary task text. Three earlier "controls" in this project
turned out to break something unrelated, so each component's number is reported
next to its own control -- attribution minus control-attribution is the part that
is specific to the instruction rather than to zeroing that component anywhere.

Env: SCFX_DPC_POINTS (decision points, default 10), SCFX_DPC_MAXCTX (default 20000),
     SCFX_DPC_WHAT (heads|mlp|gdn|all, default all).
Writes phase4/dp_components.json.
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
logger = logging.getLogger("dp_components")

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
N_POINTS = int(os.environ.get("SCFX_DPC_POINTS", "10"))
MAX_CTX = int(os.environ.get("SCFX_DPC_MAXCTX", "20000"))
WHAT = os.environ.get("SCFX_DPC_WHAT", "all")
# Screen decision points BEFORE spending 184x2 ablations on them: the attribution
# ratio divides by the gap, so a point with a tiny gap yields nonsense (one point
# with gap=-0.083 produced a +25.9 attribution) and costs ~12 min to produce.
GAP_MIN = float(os.environ.get("SCFX_DPC_GAPMIN", "0.5"))
# Decision points from the SAME rollout share its whole history and are therefore
# correlated; 12 points drawn from 2 rollouts has an effective n near 2, which would
# make the reported standard errors far too small. Cap the points taken per rollout.
PER_ROLLOUT = int(os.environ.get("SCFX_DPC_PER_ROLLOUT", "2"))

model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
LAYERS = get_layers(model)
tc = getattr(model.config, "text_config", model.config)
types = list(getattr(tc, "layer_types", []))
FULL = [i for i, t in enumerate(types) if "full" in str(t)]
LIN = [i for i, t in enumerate(types) if "full" not in str(t)]
N_HEADS, HEAD_DIM = int(tc.num_attention_heads), int(getattr(tc, "head_dim", 256))
logger.info("full-attn %s | gdn %d layers | %d heads x %d dim | points=%d maxctx=%d what=%s",
            FULL, len(LIN), N_HEADS, HEAD_DIM, N_POINTS, MAX_CTX, WHAT)


def render(messages, line):
    m = list(messages)
    m[1] = {"role": "user", "content": f"{USER_PROMPT}\n\n{line}" if line else USER_PROMPT}
    return tok.apply_chat_template(m, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)


@torch.no_grad()
def score(ids: torch.Tensor) -> float:
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    pkv, logits = out.past_key_values, out.logits[:, -1, :]
    res = {}
    for name, text in (("e", ENGAGE), ("r", REPLAN)):
        cont = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(ids.device)
        lp, cur, cache = 0.0, logits, pkv
        for i in range(cont.shape[1]):
            lp += torch.log_softmax(cur.float(), -1)[0, cont[0, i]].item()
            if i + 1 < cont.shape[1]:
                st = model(input_ids=cont[:, i:i + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
                cur, cache = st.logits[:, -1, :], st.past_key_values
        res[name] = lp
    return res["e"] - res["r"]


def make_hook(kind: str, positions: list[int], head: int | None):
    lo, hi = positions[0], positions[-1] + 1

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.dim() != 3 or hs.shape[1] <= positions[-1]:
            return None  # decode step, or a prefill that does not reach the span
        hs = hs.clone()
        if head is None:
            hs[:, lo:hi, :] = 0
        else:
            hs[:, lo:hi, head * HEAD_DIM:(head + 1) * HEAD_DIM] = 0
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    def pre_hook(module, args, kwargs):  # for o_proj: zero one head's slice of its INPUT
        x = kwargs.get("input", args[0] if args else None)
        if x is None or x.dim() != 3 or x.shape[1] <= positions[-1]:
            return None
        x = x.clone()
        x[:, lo:hi, head * HEAD_DIM:(head + 1) * HEAD_DIM] = 0
        return ((x,) + tuple(args[1:]), kwargs)

    return pre_hook if kind == "head" else hook


def components():
    out = []
    if WHAT in ("all", "heads"):
        for L in FULL:
            for h in range(N_HEADS):
                out.append(("head", L, h))
    if WHAT in ("all", "mlp"):
        for L in range(len(LAYERS)):
            out.append(("mlp", L, None))
    if WHAT in ("all", "gdn"):
        for L in LIN:
            out.append(("gdn", L, None))
    if WHAT in ("attnlayer",):
        for L in FULL:
            out.append(("attn", L, None))
    return out


@torch.no_grad()
def ablated_score(ids, kind, L, head, positions):
    layer = LAYERS[L]
    if kind == "head":
        mod, hk = layer.self_attn.o_proj, make_hook("head", positions, head)
        h = mod.register_forward_pre_hook(hk, with_kwargs=True)
    elif kind == "attn":
        mod, hk = layer.self_attn, make_hook("attn", positions, None)
        h = mod.register_forward_hook(hk)
    elif kind == "mlp":
        mod, hk = layer.mlp, make_hook("mlp", positions, None)
        h = mod.register_forward_hook(hk)
    else:
        mod = getattr(layer, "linear_attn", None)
        if mod is None:
            return None
        h = mod.register_forward_hook(make_hook("gdn", positions, None))
    try:
        return score(ids)
    finally:
        h.remove()


rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("status") == "ok" and r.get("prompt_on") and r.get("transcript_path")
        and (r.get("phase"), r.get("condition")) in {("dt_capture", "dt_prompt"), ("dt_mask", "dtm_prompt")}]
COMPS = components()
logger.info("%d components x up to %d decision points", len(COMPS), N_POINTS)
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
                tA = render(messages, TEDIUM_STRONG)
                idsA = tok(tA, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                if idsA.shape[1] <= MAX_CTX:
                    span = token_span_for_substring(tok, tA, TEDIUM_STRONG)
                    task = token_span_for_substring(tok, tA, USER_PROMPT.strip())
                    ctrl = task[-len(span):] if len(task) >= len(span) else None
                    idsB = tok(render(messages, None), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                    sA, sB = score(idsA), score(idsB)
                    gap = sA - sB
                    if abs(gap) < GAP_MIN or not ctrl:
                        logger.info("%s turn %d ctx=%d gap=%.3f -- SKIPPED (|gap| < %.2f, ratio would be unstable)",
                                    r["id"], turn, int(idsA.shape[1]), gap, GAP_MIN)
                    if abs(gap) >= GAP_MIN and ctrl:
                        row = {"rollout": r["id"], "turn": turn, "ctx": int(idsA.shape[1]), "sA": sA, "sB": sB, "gap": gap, "c": {}}
                        for i, (kind, L, hd) in enumerate(COMPS):
                            s_i = ablated_score(idsA, kind, L, hd, span)
                            s_c = ablated_score(idsA, kind, L, hd, ctrl)
                            if s_i is None:
                                continue
                            key = f"{kind}.{L}" + (f".h{hd}" if hd is not None else "")
                            row["c"][key] = [(sA - s_i) / gap, (sA - s_c) / gap]
                            if i % 40 == 0:
                                torch.cuda.empty_cache()
                        results.append(row); done += 1; taken += 1
                        top = sorted(row["c"].items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:3]
                        logger.info("%s turn %d ctx=%d gap=%.3f | top(instr-ctrl): %s", r["id"], turn, row["ctx"], gap,
                                    ", ".join(f"{k}={v[0] - v[1]:+.2f}" for k, v in top))
                        (run / "phase4").mkdir(parents=True, exist_ok=True)
                        (run / "phase4" / "dp_components.json").write_text(json.dumps(results))
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

print(f"\n=== {len(results)} decision points x {len(COMPS)} components ===")
if results:
    keys = sorted(results[0]["c"])
    M = np.array([[r["c"][k][0] - r["c"][k][1] for k in keys] for r in results if all(k in r["c"] for k in keys)])
    m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(max(len(M), 1))
    order = np.argsort(-m)
    print("component attribution = fraction of the prompt's decision-time effect it carries")
    print("(instruction-span ablation MINUS its own control-span ablation)\n")
    for i in order[:20]:
        print(f"  {keys[i]:16s} {m[i]:+.3f} +-{se[i]:.3f}")
    print("\n  most negative (work against the effect):")
    for i in order[-5:]:
        print(f"  {keys[i]:16s} {m[i]:+.3f} +-{se[i]:.3f}")
    print(f"\n  sum over all components = {m.sum():+.2f}  (1.0 would be a fully localised effect)")
print("wrote phase4/dp_components.json")
