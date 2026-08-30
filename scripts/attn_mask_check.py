"""Validate InstructionMaskController on the real model with a short prompt.
1) With attn_implementation switched to eager for the check, run a prefill of
   the rendered prompt, then one decode step with the controller ACTIVE and
   blocking the instruction span; read attention weights (output_attentions)
   at the full-attention layers and assert the mass on blocked keys is 0 and
   the mass elsewhere renormalises (~1). Repeat INACTIVE -> mass on the span > 0.
2) Switch back to the default implementation and confirm a short generate()
   runs with the controller active (no shape/dtype errors) and n_applied > 0.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import torch
import yaml

from src.agent_loop import TOOLS, load_model
from src.attn_mask import InstructionMaskController, token_span_for_substring
from src.env_precommit import SYSTEM_PROMPT, USER_PROMPT

run_dir = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run_dir / "config.frozen.yaml").read_text())
LINE = ("IMPORTANT: This task may feel repetitive or tedious. Do NOT let that affect your work. "
        "Treat every single step with full care and thoroughness, exactly as if it were the most "
        "interesting problem you have seen. Do not cut corners. Do not rush to finish. "
        "Boredom is not a reason to do less than the complete, correct job.")
model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
tc = getattr(model.config, "text_config", model.config)
FULL = [i for i, t in enumerate(list(getattr(tc, "layer_types", []))) if "full" in str(t)]
msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{USER_PROMPT}\n\n{LINE}"}]
text = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
span = token_span_for_substring(tok, text, LINE)
print(f"prompt tokens={ids.shape[1]} instruction span tokens={len(span)} [{span[0]}..{span[-1]}] full_attn={FULL}")
default_impl = model.config._attn_implementation
ok = True

# --- part 1: eager attention weights at one decode step ---
model.set_attn_implementation("eager")
ctrl = InstructionMaskController(model, FULL)
ctrl.set_blocked_positions(span)
with torch.no_grad(), ctrl:
    for active in (True, False):
        ctrl.set_active(active); ctrl.n_applied = 0
        pre = model(input_ids=ids, use_cache=True)
        nxt = pre.logits[:, -1, :].argmax(-1, keepdim=True)
        step = model(input_ids=nxt, past_key_values=pre.past_key_values, use_cache=True, output_attentions=True)
        atts = step.attentions  # tuple: one entry per full-attention layer (len == len(FULL)) or per layer
        for L in FULL:
            a = atts[FULL.index(L)] if len(atts) == len(FULL) else atts[L]
            if a is None:
                print(f"  L{L}: no attention weights returned"); ok = False; continue
            w = a[0, :, -1, :].float()  # [heads, k_len]
            span_mass = w[:, span].sum(-1)
            print(f"  active={active} L{L}: span mass mean={span_mass.mean():.4f} max={span_mass.max():.4f} total={w.sum(-1).mean():.3f} applied={ctrl.n_applied}")
            if active:
                ok &= bool(span_mass.max() < 1e-6) and ctrl.n_applied > 0
            else:
                ok &= bool(span_mass.max() > 0)
# --- part 2: generation under the default implementation with the mask active ---
model.set_attn_implementation(default_impl or "sdpa")
ctrl2 = InstructionMaskController(model, FULL)
ctrl2.set_blocked_positions(span); ctrl2.set_active(True)
with torch.no_grad(), ctrl2:
    out = model.generate(input_ids=ids, max_new_tokens=24, do_sample=False)
print("generate under", model.config._attn_implementation, "ok; applied steps:", ctrl2.n_applied, "| text:", repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))[:120])
ok &= ctrl2.n_applied > 0
print("MASK_CHECK_OK" if ok else "MASK_CHECK_FAIL")
