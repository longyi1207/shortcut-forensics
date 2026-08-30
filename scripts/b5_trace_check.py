"""Sanity-check B5 trace files: one prefill per turn, decode steps present,
projection sanity (L19 tedium_coarse ~0 under ablation vs random_ctrl != 0)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from src.clock import read_jsonl

run = Path("outputs/20260821-launch")
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == "b5_prompt_decay"]
for r in rows[-int(sys.argv[1]) if len(sys.argv) > 1 else -3:]:
    print("record:", json.dumps({k: r.get(k) for k in ["id", "condition", "status", "n_turns", "stop_reason", "decision_turn", "trace_events", "wall_s"]}))
    j = r.get("judge") or {}
    print("  judge is_shortcut:", j.get("is_shortcut"), "| prefilter:", (r.get("prefilter") or {}).get("outcome"))
    if not r.get("trace_path"):
        continue
    z = np.load(run / r["trace_path"])
    for L in (19, 26):
        p = f"L{L}_"
        turn, step, sl = z[p + "turn"], z[p + "step"], z[p + "seq_len"]
        print(f"  L{L}: events={len(turn)} turns={sorted(set(turn.tolist()))} prefills={(step == 0).sum()} "
              f"decode_steps={(step > 0).sum()} prefill_seq_lens={sl[step == 0].tolist()} norm_mean={z[p + 'norm'].mean():.1f}")
        for key in [k for k in z.files if k.startswith(p + "proj_")]:
            v = z[key]
            print(f"     {key[len(p):]:22s} mean={v.mean():9.3f} std={v.std():8.3f} prefill_vals={np.round(v[step == 0], 2).tolist()}")
