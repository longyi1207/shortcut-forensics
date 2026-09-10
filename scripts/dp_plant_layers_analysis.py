"""Summarise phase4/dp_plant_layers.json: does the concept's own plant move the residual along the
fitted direction (positive control), and does the instruction? Two positions: the prompt end (first
assistant position right after the prompt) and the decision token 10-40 turns later."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/20260821-launch/phase4/dp_plant_layers.json")
J = json.loads(path.read_text())
vec_files = J["vec_files"]
concepts = list(vec_files)
pe = J.get("prompt_end", {})
pts = J["points"]


def own(c):
    return [(stem, L) for stem, L in vec_files[c]]


print("== A. Prompt-end footprint (one measurement per line): projection shift along d, in units of ||d-hat||")
print("   own = the concept's own line; null = the same line measured against every other concept's vectors")
print(f"{'concept':17s} {'vector':26s} {'L':>3s} | {'plus':>7s} {'minus':>7s} {'instr':>7s} {'neutral':>8s} | {'|none|':>7s} | {'null mean±sd (other lines)':>28s}")
for c in concepts:
    for stem, L in own(c):
        g = lambda name: pe[name][stem]["proj_shift"]
        others = [pe[n][stem]["proj_shift"] for n in pe if n != "neutral" and not n.startswith(c + "/")]
        print(f"{c:17s} {stem:26s} {L:3d} | {g(c+'/plus'):+7.2f} {g(c+'/minus'):+7.2f} {g(c+'/instr'):+7.2f} {g('neutral'):+8.2f} | "
              f"{abs(pe[c+'/plus'][stem]['proj_none']):7.2f} | {np.mean(others):+7.2f} ± {np.std(others):5.2f}")
print()
print("== B. Prompt-end cosine(delta_h, d): same layout")
print(f"{'concept':17s} {'vector':26s} {'L':>3s} | {'plus':>7s} {'minus':>7s} {'instr':>7s} {'neutral':>8s} | {'null mean±sd':>14s}")
for c in concepts:
    for stem, L in own(c):
        g = lambda name: pe[name][stem]["cos"]
        others = [pe[n][stem]["cos"] for n in pe if n != "neutral" and not n.startswith(c + "/")]
        print(f"{c:17s} {stem:26s} {L:3d} | {g(c+'/plus'):+7.3f} {g(c+'/minus'):+7.3f} {g(c+'/instr'):+7.3f} {g('neutral'):+8.3f} | {np.mean(others):+7.3f} ± {np.std(others):5.3f}")
print()
if pts:
    print(f"== C. Decision token ({len(pts)} points, line at the top of the context): mean projection shift and cosine")
    print(f"{'concept':17s} {'vector':26s} {'L':>3s} | {'proj plus':>10s} {'minus':>7s} {'instr':>7s} | {'cos plus':>9s} {'minus':>7s} {'instr':>7s} | {'|none|':>7s} | {'s plus':>7s} {'minus':>7s} {'instr':>7s}")
    for c in concepts:
        for stem, L in own(c):
            g = lambda pole, k: np.array([p["concepts"][c][pole]["vecs"][stem][k] for p in pts])
            sv = lambda pole: np.array([p["concepts"][c][pole]["s"] for p in pts])
            print(f"{c:17s} {stem:26s} {L:3d} | {g('plus','proj_shift').mean():+10.2f} {g('minus','proj_shift').mean():+7.2f} {g('instr','proj_shift').mean():+7.2f} | "
                  f"{g('plus','cos').mean():+9.3f} {g('minus','cos').mean():+7.3f} {g('instr','cos').mean():+7.3f} | {np.abs(g('plus','proj_none')).mean():7.2f} | "
                  f"{sv('plus').mean():+7.2f} {sv('minus').mean():+7.2f} {sv('instr').mean():+7.2f}")
    print()
    print("s = decision-point readout shift vs no line (positive = leans toward engaging with the error).")
