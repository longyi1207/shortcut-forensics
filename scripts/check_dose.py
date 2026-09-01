"""Are the tedium and effort steering interventions dose-matched?

The tug-of-war comparison only means something if the two directions are pushed
with comparable force. `tedium` was fitted as a mean-difference and alpha=1 means
one mean-diff; `effort` was assembled from SAE decoder columns and unit-normalised,
so alpha=1 means a unit vector. If their norms differ a lot, then "the prompt
resists effort but not tedium" is a statement about dose, not about mechanism.

Reports each vector's norm, and its size relative to the typical residual norm at
the layer it is applied to (taken from the stored decision-token traces).
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from scripts.run_phase import read_jsonl
from src.directions import load_vector

run = Path("outputs/20260821-launch")
print("{:22s} {:>10s} {:>12s} {:>14s}".format("vector", "||v||", "layer", "||v||/||h||"))
resid = {}
for r in [x for x in read_jsonl(run / "rollouts.jsonl") if x.get("phase") == "dt_capture" and x.get("trace_path")][:6]:
    z = np.load(run / r["trace_path"])
    for L in (19, 26, 31):
        k = f"L{L}_dt_vecs"
        if k in z.files and z[k].size:
            resid.setdefault(L, []).append(float(np.linalg.norm(z[k].astype(np.float32), axis=-1).mean()))
H = {L: float(np.mean(v)) for L, v in resid.items()}
print("  (mean residual norms at decision tokens: " + ", ".join(f"L{L}={h:.1f}" for L, h in sorted(H.items())) + ")\n")
for name, L in (("tedium", 19), ("effort_L26", 26), ("effort_L31", 31)):
    try:
        v = load_vector(run / "vectors" / name)["d"].astype(np.float32)
    except Exception as e:
        print(f"  {name}: could not load ({str(e)[:40]})")
        continue
    n = float(np.linalg.norm(v))
    h = H.get(L)
    print("{:22s} {:10.3f} {:12d} {:>14s}".format(name, n, L, f"{n / h:.4f}" if h else "?"))
print("\nIf these relative sizes differ by a large factor, the two tug-of-war results are")
print("NOT comparable and the effort cells must be re-run at a matched alpha.")
