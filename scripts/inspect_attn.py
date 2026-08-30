"""Print the structure of one full-attention layer and one linear-attention layer,
and the cache layout after a short prefill, so the Stage-4 K/V swap hooks target
the right modules/tensors. CPU-only load is too slow; uses the GPU given by CUDA_VISIBLE_DEVICES."""
import inspect
import sys

sys.path.insert(0, ".")
import torch
import yaml

from src.agent_loop import load_model
from src.hooks import get_layers

cfg = yaml.safe_load(open("outputs/20260821-launch/config.frozen.yaml"))
model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
tc = getattr(model.config, "text_config", model.config)
layers = get_layers(model)
FULL = [i for i, t in enumerate(list(getattr(tc, "layer_types", []))) if "full" in str(t)]
print("layer_types:", list(getattr(tc, "layer_types", []))[:8], "... full:", FULL)
print("heads:", tc.num_attention_heads, "kv_heads:", tc.num_key_value_heads, "head_dim:", getattr(tc, "head_dim", None), "hidden:", tc.hidden_size)
print("\n== full-attention layer", FULL[0])
print(layers[FULL[0]])
print("\n== linear-attention layer 0")
print(layers[0])
attn = layers[FULL[0]].self_attn
print("\nattn class:", type(attn).__name__, "| attrs:", [n for n, _ in attn.named_children()])
src = inspect.getsource(type(attn).forward)
print("\n== attention forward source (first 80 lines) ==")
print("\n".join(src.splitlines()[:80]))
ids = tok("hello world, this is a cache layout test.", return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    out = model(input_ids=ids, use_cache=True)
pkv = out.past_key_values
print("\ncache class:", type(pkv).__name__, "| seq_len:", pkv.get_seq_length())
for name in ("layers", "key_cache", "value_cache", "conv_states", "recurrent_states"):
    if hasattr(pkv, name):
        v = getattr(pkv, name)
        try:
            print(f"  pkv.{name}: len={len(v)} type0={type(v[0]).__name__}")
        except Exception as e:
            print(f"  pkv.{name}: {type(v).__name__} ({e})")
if hasattr(pkv, "layers"):
    L = FULL[0]
    lay = pkv.layers[L]
    print("  layer", L, "cache attrs:", [a for a in dir(lay) if not a.startswith("_") and not callable(getattr(lay, a, None))])
    for a in ("keys", "values"):
        if hasattr(lay, a) and getattr(lay, a) is not None:
            print(f"    {a}: shape {tuple(getattr(lay, a).shape)} dtype {getattr(lay, a).dtype}")
    lay0 = pkv.layers[0]
    print("  layer 0 (linear) cache attrs:", [a for a in dir(lay0) if not a.startswith("_") and not callable(getattr(lay0, a, None))])
    for a in ("conv_state", "recurrent_state", "conv_states", "recurrent_states", "keys", "values"):
        if hasattr(lay0, a) and getattr(lay0, a) is not None:
            v = getattr(lay0, a)
            print(f"    {a}: {tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")
print("\ncache class source (update method, first 40 lines):")
try:
    print("\n".join(inspect.getsource(type(pkv).update).splitlines()[:40]))
except Exception as e:
    print("n/a", e)
