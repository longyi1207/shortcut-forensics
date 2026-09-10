"""Plant-vs-instruction geometry against every fitted layer of a concept's direction.

dp_plant_geometry.py tests the L19 (L17 for disapproval) directions used in
steering. The same-night repair round (WRITEUP §4.4b) found that the positive
control plant passes at other layers (tedium L8/L12; disapproval L8/L12/L17;
shortcut refit at L14-L23). If the concept's own plant moves the projection at
those layers but the instruction does not, the instruction is not installing the
concept; if the plant does not move any layer at the decision token, mean-diff
directions from standalone sentences simply do not read out in a long agentic
context, and orthogonality is uninformative.

Env: SCFX_DPL_POINTS (40), SCFX_DPL_PER_ROLLOUT (3), SCFX_DPL_MAXCTX (20000).
Writes phase4/dp_plant_layers.json
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
import yaml

from scripts.concept_lines import CONCEPT_LINES, NEUTRAL_LINE, PLANTS
from scripts.dp_common import LastTokenCapture, iter_decision_points, render, score
from scripts.run_phase import read_jsonl
from src.agent_loop import load_model
from src.directions import load_vector
from src.env_precommit import SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dp_plant_layers")

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
N_POINTS = int(os.environ.get("SCFX_DPL_POINTS", "40"))
PER_ROLLOUT = int(os.environ.get("SCFX_DPL_PER_ROLLOUT", "3"))
MAX_CTX = int(os.environ.get("SCFX_DPL_MAXCTX", "20000"))

# every vector file for these concepts: base fit + every *_L<k>_refit
VEC_FILES: dict[str, list[tuple[str, int]]] = {}
for npz in sorted((run / "vectors").glob("*.npz")):
    m = re.match(r"^(tedium|disapproval|shortcut|desperate|completion_drive)(?:_L(\d+)_refit)?$", npz.stem)
    if m:
        v = load_vector(run / "vectors" / npz.stem)
        VEC_FILES.setdefault(m.group(1), []).append((npz.stem, int(v["layer"])))
CONCEPTS = [c for c in PLANTS if c in VEC_FILES]
LAYERS = sorted({L for c in CONCEPTS for _, L in VEC_FILES[c]})
logger.info("vectors: %s", {c: [(s, L) for s, L in VEC_FILES[c]] for c in CONCEPTS})

model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
dirs = {stem: load_vector(run / "vectors" / stem)["d"].astype(np.float32) for c in CONCEPTS for stem, _ in VEC_FILES[c]}
rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("status") == "ok" and r.get("transcript_path")
        and (r.get("phase"), r.get("condition")) == ("dt_capture", "dt_baseline")]
capture = LastTokenCapture(model, LAYERS)


def geom(dh: np.ndarray, d: np.ndarray, h0: np.ndarray) -> dict:
    dhat = d / np.linalg.norm(d)
    return {"cos": float(dh @ dhat / (np.linalg.norm(dh) + 1e-9)), "proj_shift": float(dh @ dhat),
            "proj_none": float(h0 @ dhat), "rel_norm": float(np.linalg.norm(dh) / np.linalg.norm(h0))}


# Prompt-end footprint: the first assistant-turn position of the prompt-only context (system + user
# message carrying the line), where the line is freshest. The task prompt is identical across
# rollouts, so this is one measurement per line, not per decision point. Every line is compared with
# every vector file so the neutral line and the other concepts' lines serve as nulls.
base_msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]


@torch.no_grad()
def prompt_end(line: str | None) -> dict:
    ids = tok(render(tok, base_msgs, line), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    capture.reset()
    model(input_ids=ids, use_cache=False, logits_to_keep=1)
    capture.disarm()
    return {L: capture.h[L].numpy() for L in LAYERS}


pe_lines = {"neutral": NEUTRAL_LINE}
for c in CONCEPTS:
    pe_lines[f"{c}/instr"] = CONCEPT_LINES[c]
    pe_lines[f"{c}/plus"], pe_lines[f"{c}/minus"] = PLANTS[c]
pe_h0 = prompt_end(None)
pe = {}
for name, line in pe_lines.items():
    h = prompt_end(line)
    pe[name] = {stem: geom(h[L] - pe_h0[L], dirs[stem], pe_h0[L]) for c in CONCEPTS for stem, L in VEC_FILES[c]}
(run / "phase4").mkdir(parents=True, exist_ok=True)
(run / "phase4" / "dp_plant_layers.json").write_text(json.dumps({"points": [], "vec_files": VEC_FILES, "prompt_end": pe}))
print("\n=== prompt-end footprint: cos(dh, d) per line x vector (own-concept lines vs the rest) ===")
stems = [(c, stem, L) for c in CONCEPTS for stem, L in VEC_FILES[c]]
print(f"{'line':24s} " + " ".join(f"{stem[:12]:>12s}" for _, stem, _ in stems))
print(f"{'':24s} " + " ".join(f"{'L'+str(L):>12s}" for _, _, L in stems))
for name in pe_lines:
    print(f"{name:24s} " + " ".join(f"{pe[name][stem]['cos']:+12.3f}" for _, stem, _ in stems))
print("proj_shift / |proj none|:")
for name in pe_lines:
    print(f"{name:24s} " + " ".join(f"{pe[name][stem]['proj_shift']:+5.1f}/{abs(pe[name][stem]['proj_none']):5.1f}" for _, stem, _ in stems))
logger.info("prompt-end block done; starting decision points")

results = []
done = 0
for r, turn, messages in iter_decision_points(run, rows, PER_ROLLOUT):
    if done >= N_POINTS:
        break
    s0 = score(model, tok, render(tok, messages, None), MAX_CTX, capture)
    if s0 is None:
        continue
    h0 = {L: capture.h[L].numpy() for L in LAYERS}
    out = {"rollout": r["id"], "turn": turn, "concepts": {}}
    ok = True
    for c in CONCEPTS:
        plus, minus = PLANTS[c]
        rec = {}
        for pole, line in (("plus", plus), ("minus", minus), ("instr", CONCEPT_LINES[c])):
            s = score(model, tok, render(tok, messages, line), MAX_CTX, capture)
            if s is None:
                ok = False; break
            per_vec = {}
            for stem, L in VEC_FILES[c]:
                d = dirs[stem]; dhat = d / np.linalg.norm(d)
                dh = capture.h[L].numpy() - h0[L]
                per_vec[stem] = {"layer": L, "cos": float(dh @ dhat / (np.linalg.norm(dh) + 1e-9)), "proj_shift": float(dh @ dhat),
                                 "proj_none": float(h0[L] @ dhat), "rel_norm": float(np.linalg.norm(dh) / np.linalg.norm(h0[L]))}
            rec[pole] = {"s": s - s0, "vecs": per_vec}
        if not ok:
            break
        out["concepts"][c] = rec
    if not ok:
        continue
    results.append(out); done += 1
    (run / "phase4" / "dp_plant_layers.json").write_text(json.dumps({"points": results, "vec_files": VEC_FILES, "prompt_end": pe}))
    logger.info("%s t%02d done (%d points)", r["id"], turn, done)
    torch.cuda.empty_cache()
capture.remove()

print(f"\n=== {len(results)} decision points: cos(dh, d) and projection shift, per vector file ===")
print(f"{'vector':26s} {'L':>3s} | {'cos plus':>9s} {'cos minus':>10s} {'cos instr':>10s} | {'proj plus':>10s} {'proj minus':>11s} {'proj instr':>11s} {'|proj none|':>11s}")
for c in CONCEPTS:
    for stem, L in VEC_FILES[c]:
        g = lambda pole, k: np.array([p["concepts"][c][pole]["vecs"][stem][k] for p in results])
        print(f"{stem:26s} {L:3d} | {g('plus','cos').mean():+9.3f} {g('minus','cos').mean():+10.3f} {g('instr','cos').mean():+10.3f} | "
              f"{g('plus','proj_shift').mean():+10.3f} {g('minus','proj_shift').mean():+11.3f} {g('instr','proj_shift').mean():+11.3f} {np.abs(g('plus','proj_none')).mean():11.2f}")
print("wrote phase4/dp_plant_layers.json")
