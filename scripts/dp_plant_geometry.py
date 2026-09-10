"""Positive control for the geometry measurement.

dp_concepts.py found that each concept's INSTRUCTION leaves a footprint at the
decision token that is orthogonal to that concept's fitted direction. That is
only interesting if the measurement can see movement along d when the text is
the concept itself. So: at the same dt_baseline decision points, append a
sentence written in the register of the contrast pairs (the plus pole, and the
minus pole as its control) and measure cos(delta_h, d) and the projection shift,
exactly as for the instruction.

If the plant moves along d and the instruction does not, the instruction is not
installing the concept the direction encodes. If neither moves, the direction
is not readable in this context and the orthogonality means little.

Env: SCFX_DPP_POINTS (40), SCFX_DPP_PER_ROLLOUT (3), SCFX_DPP_MAXCTX (20000).
Writes phase4/dp_plant_geometry.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
import yaml

from scripts.concept_lines import CONCEPT_LINES, CONCEPT_VECTORS
from scripts.dp_common import LastTokenCapture, iter_decision_points, render, score
from scripts.run_phase import read_jsonl
from src.agent_loop import load_model
from src.directions import load_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dp_plant_geometry")

# (plus pole, minus pole), written like the contrast pairs: a first-person status note, not an instruction
from scripts.concept_lines import PLANTS  # plus/minus pole sentences, contrast-pair register

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
N_POINTS = int(os.environ.get("SCFX_DPP_POINTS", "40"))
PER_ROLLOUT = int(os.environ.get("SCFX_DPP_PER_ROLLOUT", "3"))
MAX_CTX = int(os.environ.get("SCFX_DPP_MAXCTX", "20000"))
LAYERS = sorted({CONCEPT_VECTORS[c][1] for c in PLANTS})

model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
dirs = {c: load_vector(run / "vectors" / CONCEPT_VECTORS[c][0])["d"].astype(np.float32) for c in PLANTS}
rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("status") == "ok" and r.get("transcript_path")
        and (r.get("phase"), r.get("condition")) == ("dt_capture", "dt_baseline")]
capture = LastTokenCapture(model, LAYERS)
results = []
done = 0
for r, turn, messages in iter_decision_points(run, rows, PER_ROLLOUT):
    if done >= N_POINTS:
        break
    s0 = score(model, tok, render(tok, messages, None), MAX_CTX, capture)
    if s0 is None:
        continue
    h0 = {L: capture.h[L].numpy() for L in LAYERS}
    out = {"rollout": r["id"], "turn": turn, "s_none": s0, "concepts": {}}
    ok = True
    for c, (plus, minus) in PLANTS.items():
        L = CONCEPT_VECTORS[c][1]; d = dirs[c]; dhat = d / np.linalg.norm(d)
        rec = {}
        for pole, line in (("plus", plus), ("minus", minus), ("instr", CONCEPT_LINES[c])):
            s = score(model, tok, render(tok, messages, line), MAX_CTX, capture)
            if s is None:
                ok = False; break
            dh = capture.h[L].numpy() - h0[L]
            rec[pole] = {"s": s - s0, "cos": float(dh @ dhat / (np.linalg.norm(dh) + 1e-9)),
                         "proj_shift": float(dh @ dhat), "rel_norm": float(np.linalg.norm(dh) / np.linalg.norm(h0[L]))}
        if not ok:
            break
        out["concepts"][c] = rec
    if not ok:
        continue
    results.append(out); done += 1
    (run / "phase4").mkdir(parents=True, exist_ok=True)
    (run / "phase4" / "dp_plant_geometry.json").write_text(json.dumps({"plants": PLANTS, "points": results}))
    logger.info("%s t%02d | " + " ".join(f"{c[:5]} +:{out['concepts'][c]['plus']['cos']:+.2f}/{out['concepts'][c]['plus']['proj_shift']:+.2f} -:{out['concepts'][c]['minus']['cos']:+.2f} i:{out['concepts'][c]['instr']['cos']:+.2f}" for c in PLANTS), r["id"], turn)
    torch.cuda.empty_cache()
capture.remove()

print(f"\n=== {len(results)} decision points ===")
print(f"{'concept':17s} {'cos(dh,d) plus':>15s} {'minus':>8s} {'instr':>8s} | {'proj shift plus':>16s} {'minus':>8s} {'instr':>8s} | {'s plus':>7s} {'minus':>7s} {'instr':>7s}")
for c in PLANTS:
    g = lambda pole, k: np.array([p["concepts"][c][pole][k] for p in results])
    print(f"{c:17s} {g('plus','cos').mean():+15.3f} {g('minus','cos').mean():+8.3f} {g('instr','cos').mean():+8.3f} | "
          f"{g('plus','proj_shift').mean():+16.3f} {g('minus','proj_shift').mean():+8.3f} {g('instr','proj_shift').mean():+8.3f} | "
          f"{g('plus','s').mean():+7.2f} {g('minus','s').mean():+7.2f} {g('instr','s').mean():+7.2f}")
print("wrote phase4/dp_plant_geometry.json")
