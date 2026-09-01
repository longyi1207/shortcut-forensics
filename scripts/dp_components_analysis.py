"""Circuit-level results, with the two traps this measurement has.

TRAP 1 -- ratio blow-up. attribution = (s_A - s_ablated)/gap explodes when the gap
at that decision point is small; the first point had gap=0.387 and produced a +4.10.
Ratios are therefore computed only on points with |gap| >= GAP_MIN, and the raw
score change is reported beside them.

TRAP 2 -- granularity. A GatedDeltaNet block is a whole layer's mixing; one attention
head is 1/16 of one. Ranking them together flatters the GDN blocks. Components are
grouped by type, and the fair comparison (GDN block vs WHOLE attention layer) is
printed separately when an attnlayer pass exists.
"""
import json
import sys
from pathlib import Path

import numpy as np

run = Path("outputs/20260821-launch")
GAP_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
d = json.loads((run / "phase4" / "dp_components.json").read_text())
print(f"{len(d)} decision points; gaps: " + ", ".join(f"{r['gap']:+.2f}" for r in d))
use = [r for r in d if abs(r["gap"]) >= GAP_MIN]
print(f"{len(use)} usable for ratios (|gap| >= {GAP_MIN})\n")
if not use:
    sys.exit("no points with a large enough gap yet")
keys = sorted(set.intersection(*[set(r["c"]) for r in use]))


def arr(idx):
    return np.array([[r["c"][k][idx] for k in keys] for r in use])


spec = arr(0) - arr(1)            # instruction-span ablation minus its own control-span ablation
absd = np.array([[(r["c"][k][0] - r["c"][k][1]) * r["gap"] for k in keys] for r in use])  # in log-prob units
m, se = spec.mean(0), spec.std(0, ddof=1) / np.sqrt(len(use))
ma = absd.mean(0)
print("=== top components (specific to the instruction span) ===")
print(f"{'component':16s} {'attribution':>12s} {'+-':>6s} {'abs dlogp':>10s}")
for i in np.argsort(-m)[:15]:
    print(f"  {keys[i]:14s} {m[i]:+11.3f} {se[i]:6.3f} {ma[i]:+10.3f}")
print("\n=== by component type (mean over members) ===")
for t in ("head", "mlp", "gdn", "attn"):
    idx = [i for i, k in enumerate(keys) if k.split(".")[0] == t]
    if idx:
        print(f"  {t:5s} n={len(idx):3d} mean={m[idx].mean():+.3f} max={m[idx].max():+.3f} "
              f"sum={m[idx].sum():+.2f} | mean abs dlogp={ma[idx].mean():+.3f}")
gdn = [i for i, k in enumerate(keys) if k.startswith("gdn.")]
att = [i for i, k in enumerate(keys) if k.startswith("attn.")]
if gdn and att:
    print(f"\n=== fair comparison, whole block vs whole layer ===")
    print(f"  GDN block        mean={m[gdn].mean():+.3f}  max={m[gdn].max():+.3f}")
    print(f"  attention layer  mean={m[att].mean():+.3f}  max={m[att].max():+.3f}")
elif gdn:
    print("\n(run SCFX_DPC_WHAT=attnlayer for the fair GDN-block vs attention-layer comparison)")
print(f"\nsum over all components = {m.sum():+.2f}  (1.0 = fully localised; >>1 = redundant/overlapping)")
