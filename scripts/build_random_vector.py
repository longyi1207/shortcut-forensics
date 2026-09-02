"""A random direction at the same magnitude, as the control the tug-of-war needs.

Dose-matched result: pushing `effort` at alpha=26 takes the prompt condition from
4% to 19% (p=0.05) while leaving the baseline unchanged (14% -> 15%). That reads
as "this is the axis the prompt's protection lives on" -- but only if pushing an
ARBITRARY direction just as hard does NOT also abolish the protection. Without
this control the result could just mean "the protection is fragile to any large
perturbation at L26".

Writes vectors/randdir_L26 (seeded, unit norm, so alpha=26 matches the effort cell).
"""
import json
import sys
from pathlib import Path

import numpy as np

run = Path("outputs/20260821-launch")
rng = np.random.default_rng(20260902)
d = 4096
v = rng.standard_normal(d).astype(np.float32)
v /= np.linalg.norm(v)
eff = np.load(run / "vectors" / "effort_L26.npz")["d"].astype(np.float32)
cos = float(v @ eff / (np.linalg.norm(v) * np.linalg.norm(eff)))
p = run / "vectors" / "randdir_L26.npz"
np.savez(p, d=v, layer=26, val_acc=float("nan"), pair_ids=np.array([], dtype=object),
         meta=json.dumps({"kind": "random_control", "seed": 20260902, "cos_with_effort": cos}))
print(f"wrote {p} | ||v||={np.linalg.norm(v):.3f} | cos(random, effort) = {cos:+.4f}")
