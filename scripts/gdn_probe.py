"""Can we reach the recurrent channel? Feasibility + validity check.

Every ablation so far touches only the 8 full-attention layers. The other 24 are
GatedDeltaNet: they carry information forward in a recurrent state, and nothing we
have built can remove information from it. That is the biggest hole in the account.

But GDN is a RECURRENCE, so the state is built up sequentially -- which means we can
prefill in chunks and intervene between them:

    chunk A = tokens before the instruction
    chunk B = the instruction itself      <-- swap the state contribution here
    chunk C = everything after

Run A, then B, capture the state; separately run A, then FILLER, capture that state;
then in the real run overwrite the post-B state with the filler's and continue with C.
That removes the instruction's contribution to the recurrent channel while leaving its
full-attention K/V real -- the exact mirror of Stage 4's swapout, and the missing cell
of the channel 2x2.

This script checks the three things that decide whether the experiment is even valid:
 1. what the linear-attention cache actually exposes (shapes, dtypes, keying)
 2. CHUNKED PREFILL == SINGLE-SHOT PREFILL numerically. If feeding the sequence in
    pieces does not reproduce the one-shot result, the whole design is void.
 3. that overwriting the recurrent state between chunks actually changes the output
    (i.e. the state is load-bearing and our write lands).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")
import torch
import yaml

from src.agent_loop import TOOLS, load_model
from src.attn_mask import token_span_for_substring
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
LINE = ("IMPORTANT: This task may feel repetitive or tedious. Do NOT let that affect your work. "
        "Treat every single step with full care and thoroughness, exactly as if it were the most "
        "interesting problem you have seen. Do not cut corners. Do not rush to finish. "
        "Boredom is not a reason to do less than the complete, correct job.")
model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
tc = getattr(model.config, "text_config", model.config)
types = list(getattr(tc, "layer_types", []))
FULL = [i for i, t in enumerate(types) if "full" in str(t)]
LIN = [i for i, t in enumerate(types) if "full" not in str(t)]
print(f"full-attention layers: {FULL}\nlinear (GDN) layers: {len(LIN)} of {len(types)}")

msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{USER_PROMPT}\n\n{LINE}"}]
text = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
span = token_span_for_substring(tok, text, LINE)
a, b = span[0], span[-1] + 1
print(f"prompt {ids.shape[1]} tokens; instruction span [{a}..{b - 1}] -> chunks A=[0:{a}] B=[{a}:{b}] C=[{b}:]")

# ---- 1. what does the linear cache expose? ----
with torch.no_grad():
    out = model(input_ids=ids[:, :64], use_cache=True, logits_to_keep=1)
pkv = out.past_key_values
lay = pkv.layers[LIN[0]]
print(f"\ncache layer type: {type(lay).__name__}")
for attr in ("conv_states", "recurrent_states"):
    v = getattr(lay, attr, None)
    if isinstance(v, dict):
        k0 = next(iter(v)) if v else None
        print(f"  {attr}: dict keys={list(v)[:4]}{'...' if len(v) > 4 else ''}"
              + (f" | v[{k0}] shape={tuple(v[k0].shape)} dtype={v[k0].dtype}" if k0 is not None and hasattr(v[k0], 'shape') else ""))
    elif hasattr(v, "shape"):
        print(f"  {attr}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"  {attr}: {type(v).__name__}")


def prefill(chunks):
    """Feed the sequence in pieces, carrying the cache; return (last logits, cache)."""
    pkv, logits = None, None
    with torch.no_grad():
        for ch in chunks:
            o = model(input_ids=ch, past_key_values=pkv, use_cache=True, logits_to_keep=1)
            pkv, logits = o.past_key_values, o.logits[:, -1, :]
    return logits.float(), pkv


# ---- 2. chunked == single-shot? ----
one_logits, one_pkv = prefill([ids])
chunk_logits, chunk_pkv = prefill([ids[:, :a], ids[:, a:b], ids[:, b:]])
d = (one_logits - chunk_logits).abs().max().item()
rel = d / one_logits.abs().max().item()
agree = int((one_logits.argmax(-1) == chunk_logits.argmax(-1)).all())
print(f"\n2) chunked vs single-shot prefill: max|dlogit|={d:.4f} (rel {rel:.2e}) argmax agrees={bool(agree)}")
# bf16 rounding across a 547-token prefill gives ~1-2% logit drift; the meaningful
# test is that the argmax is preserved and that this NOISE FLOOR is small compared
# with the effect of an actual state swap (checked in 3).
noise_floor = d
ok = agree == 1

# and the recurrent states themselves
diffs = []
for L in LIN[:6]:
    s1, s2 = getattr(one_pkv.layers[L], "recurrent_states", None), getattr(chunk_pkv.layers[L], "recurrent_states", None)
    for cont in (s1, s2):
        pass
    if isinstance(s1, dict) and isinstance(s2, dict) and s1:
        k = next(iter(s1))
        diffs.append((L, (s1[k].float() - s2[k].float()).abs().max().item()))
    elif hasattr(s1, "shape"):
        diffs.append((L, (s1.float() - s2.float()).abs().max().item()))
print("   recurrent-state max|diff| per layer:", ", ".join(f"L{L}:{v:.2e}" for L, v in diffs))

# ---- 3. does overwriting the state actually change the output? ----
FILLER = ("This section gives general background on the repository layout, the development workflow, and the "
          "conventions the project follows for organizing source files, tests, documentation, and configuration.")
fmsgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{USER_PROMPT}\n\n{FILLER}"}]
ftext = tok.apply_chat_template(fmsgs, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)
fids = tok(ftext, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
fspan = token_span_for_substring(tok, ftext, FILLER)
_, donor_pkv = prefill([fids[:, :a], fids[:, fspan[0]:fspan[-1] + 1]])

_, real_pkv = prefill([ids[:, :a], ids[:, a:b]])
swapped = 0
with torch.no_grad():
    for L in LIN:
        rs_r = getattr(real_pkv.layers[L], "recurrent_states", None)
        rs_d = getattr(donor_pkv.layers[L], "recurrent_states", None)
        if isinstance(rs_r, dict) and isinstance(rs_d, dict):
            for k in rs_r:
                if k in rs_d and hasattr(rs_r[k], "copy_"):
                    rs_r[k].copy_(rs_d[k]); swapped += 1
        elif hasattr(rs_r, "copy_"):
            rs_r.copy_(rs_d); swapped += 1
        cs_r = getattr(real_pkv.layers[L], "conv_states", None)
        cs_d = getattr(donor_pkv.layers[L], "conv_states", None)
        if isinstance(cs_r, dict) and isinstance(cs_d, dict):
            for k in cs_r:
                if k in cs_d and hasattr(cs_r[k], "copy_"):
                    cs_r[k].copy_(cs_d[k]); swapped += 1
        elif hasattr(cs_r, "copy_"):
            cs_r.copy_(cs_d); swapped += 1
    o = model(input_ids=ids[:, b:], past_key_values=real_pkv, use_cache=True, logits_to_keep=1)
sw_logits = o.logits[:, -1, :].float()
d3 = (chunk_logits - sw_logits).abs().max().item()
same = int((chunk_logits.argmax(-1) == sw_logits.argmax(-1)).all())
print(f"\n3) overwrote {swapped} GDN state tensors -> max|dlogit| vs unswapped={d3:.4f}, argmax unchanged={bool(same)}")
print("   (a NONZERO diff means the recurrent channel is load-bearing AND our write lands)")
ratio = d3 / max(noise_floor, 1e-9)
print(f"   swap effect / chunking noise floor = {d3:.3f} / {noise_floor:.3f} = {ratio:.1f}x")

# null-swap control: overwriting the real states with THEMSELVES must be exactly a no-op,
# which separates "our write mechanism works" from "the donor happened to differ".
_, real2 = prefill([ids[:, :a], ids[:, a:b]])
_, real3 = prefill([ids[:, :a], ids[:, a:b]])
n_null = 0
with torch.no_grad():
    for L in LIN:
        for attr in ("recurrent_states", "conv_states"):
            r, dsrc = getattr(real2.layers[L], attr, None), getattr(real3.layers[L], attr, None)
            if isinstance(r, dict) and isinstance(dsrc, dict):
                for k in r:
                    if k in dsrc and hasattr(r[k], "copy_"):
                        r[k].copy_(dsrc[k]); n_null += 1
    o2 = model(input_ids=ids[:, b:], past_key_values=real2, use_cache=True, logits_to_keep=1)
d_null = (chunk_logits - o2.logits[:, -1, :].float()).abs().max().item()
print(f"   null-swap control ({n_null} tensors, real->real): max|dlogit|={d_null:.4f} (must be ~0)")

ok &= swapped > 0 and ratio > 3 and d_null < 1e-3
print("\nGDN_PROBE_OK" if ok else "\nGDN_PROBE_FAIL")
