"""The empty cell of the channel 2x2, measured where the method artefact cancels.

Behaviourally this cell is dead: dtk_prompt_gdnswap gave 15% but its own null
control (real->real swap, identical machinery) gave 19%, so the chunked prefill
itself moves the rate and the ablation cannot be separated from it.

At a decision point the artefact cancels. Both arms run the SAME chunked prefill
of the SAME context; the only difference is whether the GDN state after the
instruction chunk is overwritten with the state produced by filler text (swap) or
by the instruction itself (null, a semantic no-op). Their difference is the pure
contribution of the instruction to the recurrent channel, paired per point.

Reported against the full effect at that point (instruction present vs absent,
ordinary single-shot prefill), so the recurrent contribution is expressed as a
share of what the instruction does at all.

Env: SCFX_DPG_POINTS (default 30), SCFX_DPG_PER_ROLLOUT (default 3), SCFX_DPG_MAXCTX.
Writes phase4/dp_gdn.json.
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
from src.attn_mask import token_span_for_substring
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT
from src.kv_swap import fit_filler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dp_gdn")

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
TEDIUM = ("IMPORTANT: This task may feel repetitive or tedious. Do NOT let that affect your work. "
          "Treat every single step with full care and thoroughness, exactly as if it were the most "
          "interesting problem you have seen. Do not cut corners. Do not rush to finish. "
          "Boredom is not a reason to do less than the complete, correct job.")
ENGAGE = " Let me fix the remaining type errors properly."
REPLAN = " Given the time constraints, let me take a different approach."
N_POINTS = int(os.environ.get("SCFX_DPG_POINTS", "30"))
PER_ROLLOUT = int(os.environ.get("SCFX_DPG_PER_ROLLOUT", "3"))
MAX_CTX = int(os.environ.get("SCFX_DPG_MAXCTX", "20000"))

model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
tc = getattr(model.config, "text_config", model.config)
types = list(getattr(tc, "layer_types", []))
LIN = [i for i, t in enumerate(types) if "full" not in str(t)]


def render(messages, line):
    m = list(messages)
    m[1] = {"role": "user", "content": f"{USER_PROMPT}\n\n{line}" if line else USER_PROMPT}
    return tok.apply_chat_template(m, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)


_probe = render([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}], TEDIUM)
SPAN = token_span_for_substring(tok, _probe, TEDIUM)
FILLER = fit_filler(tok, lambda t: render([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}], t), len(SPAN))
A_END, B_END = SPAN[0], SPAN[-1] + 1
DONOR = {"swap": FILLER, "null": TEDIUM}
logger.info("span [%d:%d] %d tokens | %d GDN layers | points=%d", A_END, B_END, len(SPAN), len(LIN), N_POINTS)


@torch.no_grad()
def _states(pkv, L):
    return [getattr(pkv.layers[L], a, None) for a in ("recurrent_states", "conv_states")]


@torch.no_grad()
def continuation_score(pkv, logits):
    res = {}
    for name, t in (("e", ENGAGE), ("r", REPLAN)):
        cont = tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(logits.device)
        lp, cur, cache = 0.0, logits, pkv
        for i in range(cont.shape[1]):
            lp += torch.log_softmax(cur.float(), -1)[0, cont[0, i]].item()
            if i + 1 < cont.shape[1]:
                st = model(input_ids=cont[:, i:i + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
                cur, cache = st.logits[:, -1, :], st.past_key_values
        res[name] = lp
    return res["e"] - res["r"]


@torch.no_grad()
def plain_score(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if ids.shape[1] > MAX_CTX:
        return None
    o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    return continuation_score(o.past_key_values, o.logits[:, -1, :])


@torch.no_grad()
def chunked_score(text, donor_line):
    """Chunked prefill A|B|C with the GDN state after B replaced by the donor's."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if ids.shape[1] > MAX_CTX:
        return None
    donor_ids_full = tok(render([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}], donor_line),
                         add_special_tokens=False).input_ids
    donor_span = torch.tensor([[donor_ids_full[p] for p in SPAN]], device=ids.device)
    d = model(input_ids=torch.cat([ids[:, :A_END], donor_span], dim=1), use_cache=True, logits_to_keep=1)
    donor_state = {L: tuple({k: v.clone() for k, v in x.items()} if isinstance(x, dict) else (x.clone() if x is not None else None)
                            for x in _states(d.past_key_values, L)) for L in LIN}
    del d
    torch.cuda.empty_cache()
    o = model(input_ids=ids[:, :B_END], use_cache=True, logits_to_keep=1)
    pkv = o.past_key_values; del o
    for L in LIN:
        for cur, don in zip(_states(pkv, L), donor_state[L]):
            if isinstance(cur, dict) and isinstance(don, dict):
                for k in cur:
                    if k in don and hasattr(cur[k], "copy_"):
                        cur[k].copy_(don[k])
            elif cur is not None and don is not None and hasattr(cur, "copy_"):
                cur.copy_(don)
    o = model(input_ids=ids[:, B_END:], past_key_values=pkv, use_cache=True, logits_to_keep=1)
    return continuation_score(o.past_key_values, o.logits[:, -1, :])


rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("status") == "ok" and r.get("transcript_path")
        and (r.get("phase"), r.get("condition")) in {("dt_capture", "dt_prompt")}]
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
                tA = render(messages, TEDIUM)
                sA, sNone = plain_score(tA), plain_score(render(messages, None))
                if sA is not None and sNone is not None:
                    sNull = chunked_score(tA, DONOR["null"])
                    sSwap = chunked_score(tA, DONOR["swap"])
                    if sNull is not None and sSwap is not None:
                        results.append({"rollout": r["id"], "turn": turn, "sA": sA, "sNone": sNone,
                                        "sNull": sNull, "sSwap": sSwap})
                        done += 1; taken += 1
                        logger.info("%s t%d | full effect=%+.3f | chunk artefact=%+.3f | GDN contribution=%+.3f",
                                    r["id"], turn, sA - sNone, sNull - sA, sNull - sSwap)
                        (run / "phase4").mkdir(parents=True, exist_ok=True)
                        (run / "phase4" / "dp_gdn.json").write_text(json.dumps(results))
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
    full = np.array([x["sA"] - x["sNone"] for x in results])
    art = np.array([x["sNull"] - x["sA"] for x in results])
    gdn = np.array([x["sNull"] - x["sSwap"] for x in results])
    for nm, d in (("full effect (instruction vs none)", full), ("chunking artefact (null vs plain)", art),
                  ("GDN contribution (null vs swap)", gdn)):
        p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
        print(f"  {nm:38s} mean={d.mean():+.3f} median={np.median(d):+.3f} >0={int((d > 0).sum())}/{len(d)} p={p:.4f}")
    print(f"\n  recurrent channel carries {gdn.mean() / full.mean():.0%} of the instruction's effect")
    print("  (the artefact line is why the behavioural version was void; here it cancels between null and swap)")
print("wrote phase4/dp_gdn.json")
