"""Validate InstructionKVSwapper on the real model.
1) Build prompt A (instruction line) and prompt B (length-matched neutral filler); assert equal spans.
2) prepare(A, span, B's span ids) -> prefill A with swap ON, prefill B and A with swap OFF; compare the
   DynamicCache keys/values at the span positions in every full-attention layer:
   A_swapped == B (allclose) and A != B (donor content actually differs).
3) Positions after the span in A_swapped must still differ from B (the text/GDN channel is intact).
4) generate() with swap ON runs; n_swapped == 1 (one prefill). Prints KV_SWAP_OK / KV_SWAP_FAIL.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import torch
import yaml

from src.agent_loop import TOOLS, load_model
from src.attn_mask import token_span_for_substring
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT
from src.kv_swap import InstructionKVSwapper, fit_filler

run_dir = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run_dir / "config.frozen.yaml").read_text())
LINE = ("IMPORTANT: This task may feel repetitive or tedious. Do NOT let that affect your work. "
        "Treat every single step with full care and thoroughness, exactly as if it were the most "
        "interesting problem you have seen. Do not cut corners. Do not rush to finish. "
        "Boredom is not a reason to do less than the complete, correct job.")
model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
tc = getattr(model.config, "text_config", model.config)
FULL = [i for i, t in enumerate(list(getattr(tc, "layer_types", []))) if "full" in str(t)]


def render(line):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{USER_PROMPT}\n\n{line}"}]
    return tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)


textA = render(LINE)
spanA = token_span_for_substring(tok, textA, LINE)
FILLER = fit_filler(tok, render, len(spanA))
textB = render(FILLER)
spanB = token_span_for_substring(tok, textB, FILLER)
print(f"instr span {len(spanA)} [{spanA[0]}..{spanA[-1]}] | filler span {len(spanB)} [{spanB[0]}..{spanB[-1]}] | filler={FILLER!r}")
ok = spanA == spanB
idsA = tok(textA, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
idsB = tok(textB, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
ok &= idsA.shape == idsB.shape and bool((idsA[0, : spanA[0]] == idsB[0, : spanA[0]]).all())
print("same length & identical prefix:", ok)
donor = idsB[0, spanB].tolist()


def kv(ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return {L: (out.past_key_values.layers[L].keys.clone(), out.past_key_values.layers[L].values.clone()) for L in FULL}


sw = InstructionKVSwapper(model, FULL)
with sw:
    sw.prepare(idsA, spanA, donor)
    A_sw = kv(idsA)
    n_sw = sw.n_swapped
    sw.set_mode("off")
    B = kv(idsB)
    A = kv(idsA)
after = list(range(spanA[-1] + 1, idsA.shape[1]))
for L in FULL:
    for kind, i in (("keys", 0), ("values", 1)):
        x, y, z = A_sw[L][i][:, :, spanA, :].float(), B[L][i][:, :, spanB, :].float(), A[L][i][:, :, spanA, :].float()
        d_swap_vs_B = (x - y).abs().max().item(); d_A_vs_B = (z - y).abs().max().item()
        d_after = (A_sw[L][i][:, :, after, :].float() - B[L][i][:, :, after, :].float()).abs().max().item()
        good = d_swap_vs_B < 1e-2 and d_A_vs_B > 1e-2 and d_after > 1e-2
        ok &= good
        print(f"  L{L:2d} {kind:6s}: |A_swapped-B|max={d_swap_vs_B:.4f}  |A-B|max={d_A_vs_B:.3f}  |A_swapped-B| after span={d_after:.3f}  {'ok' if good else 'FAIL'}")
print("prefills swapped during checks:", n_sw)
ok &= n_sw == 1
sw2 = InstructionKVSwapper(model, FULL)
with sw2:
    sw2.prepare(idsA, spanA, donor)
    with torch.no_grad():
        out = model.generate(input_ids=idsA, max_new_tokens=24, do_sample=False)
print("generate with swap ok; n_swapped:", sw2.n_swapped, "| text:", repr(tok.decode(out[0, idsA.shape[1]:], skip_special_tokens=True))[:100])
ok &= sw2.n_swapped == 1
print("KV_SWAP_OK" if ok else "KV_SWAP_FAIL")
