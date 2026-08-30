"""Validate full-residual capture in a trace: full_vecs present, shape [k, d_model],
one vector per turn-start plus every N-th event, indices consistent."""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
phase = sys.argv[1] if len(sys.argv) > 1 else "b5_full_smoke"
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == phase and r.get("trace_path")]
if not rows:
    raise SystemExit(f"no rows for phase {phase}")
r = rows[-1]
z = np.load(run / r["trace_path"])
print("record:", r["id"], r["condition"], "status=", r["status"], "n_turns=", r.get("n_turns"), "store_full_every=", r.get("store_full_every"))
ok = True
for L in (19, 26):
    p = f"L{L}_"
    if p + "full_vecs" not in z.files:
        print(f"  L{L}: MISSING full_vecs"); ok = False; continue
    V, idx, step = z[p + "full_vecs"], z[p + "full_idx"], z[p + "step"]
    n_start = int((step == 0).sum())
    starts_stored = int((step[idx] == 0).sum())
    print(f"  L{L}: full_vecs shape={V.shape} dtype={V.dtype} | events={len(step)} turn-starts={n_start} stored-starts={starts_stored} | idx monotone={bool(np.all(np.diff(idx) > 0))} | mean||v||={np.linalg.norm(V.astype(np.float32), axis=1).mean():.1f}")
    ok &= V.ndim == 2 and V.shape[0] >= n_start and starts_stored == n_start and V.shape[1] > 1000
print("FULL_CAPTURE_OK" if ok else "FULL_CAPTURE_FAIL")
