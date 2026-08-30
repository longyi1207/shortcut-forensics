"""Sanity-check a dt_heads trace: shapes, mass ranges, and per-layer instruction/control mass."""
import sys
from pathlib import Path

import numpy as np

p = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/20260821-launch/traces/rsmoke_dth_00000_heads.npz")
z = np.load(p)
m = z["mass"]
print("file", p, "| mass shape", m.shape, "| tags", z["tag"].tolist(), "| turns", z["turn"].tolist(), "| steps", z["step"].tolist(), "| prev_rc", z["prev_rc"].tolist())
print("full_layers", z["full_layers"].tolist(), "| span_names", z["span_names"].tolist(), "| tag_names", z["tag_names"].tolist())
print("mass range", float(m.min()), float(m.max()), "| any nan", bool(np.isnan(m).any()))
for i in range(m.shape[0]):
    print(f"row {i} tag={z['tag'][i]} step={z['step'][i]}: per-layer mean-over-heads " + " | ".join(f"L{L}: " + " ".join(f"{n}={m[i, li, :, si].mean():.3f}" for si, n in enumerate(z['span_names'])) for li, L in enumerate(z["full_layers"])))
print("max single-head instr mass per layer:", {int(L): round(float(m[:, li, :, 0].max()), 3) for li, L in enumerate(z["full_layers"])})
