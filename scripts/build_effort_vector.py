"""Experiment 1a -- build the "cost-of-effort / replanning" direction from the
Stage-1 decision-token features, so it can be steered with the existing machinery.

Stage 1 named, at the decision tokens, two families that the prompt moves in
opposite directions (labels from top-activating events, scripts/dt_feature_context.py):

  SUPPRESSED by the prompt  (cost-of-effort framing -> change of plan, and
  re-framing the requirement):
    "Given the time constraints, let me take a different approach. Instead of..."
    "This is a lot of files to fix" / "This is taking a while"
    "mypy is set to strict mode which..." / "However, the user said..."

  ENHANCED by the prompt (engagement with the specific error, next-fix list):
    "I see the issue - I used `any` instead of `Any`"
    "There are still errors. Let me fix them: 1. transform.py ..."

The `tedium` direction is NOT this: no feature here has |cos| > 0.14 with it,
and steering `tedium` decoupled completely from prompting (tug-of-war 15/30 vs
17/32, p=1.0; and at half dose 10/21 vs 10/21, p=1.0). If THIS direction behaves
differently under the same tests, we have found the axis the prompt actually uses.

Writes vectors/effort_L26 (and _L31): sum of SAE decoder columns of the
suppressed features minus the enhanced ones, unit-normalised, so that ADDING it
should push toward replanning and ABLATING it toward engagement.
Usage: build_effort_vector.py [layer ...]   (default 26 31)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from huggingface_hub import hf_hub_download

run = Path("outputs/20260821-launch")
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
LAYERS = [int(a) for a in sys.argv[1:]] or [26, 31]

# feature ids from scripts/dt_analysis.py + dt_feature_context.py (Stage 1, n=20x2)
SUPPRESSED = {  # baseline-enriched at decision tokens = cost-of-effort / replanning / requirement re-framing
    15: [28052, 48296, 42935, 5391, 17036, 13073],
    19: [25652, 22744, 57909, 43090, 37929, 58933],
    23: [50243, 4211, 23813, 5750, 41012],
    26: [39386, 27604, 24639, 36115, 38644, 14070, 62067, 10984],
    27: [19271, 9327, 63567, 47052],
    31: [2340, 32059, 34878, 16430, 17437, 54464, 53952],
}
ENHANCED = {  # prompt-enriched = engagement with the specific error / next-fix enumeration
    19: [8689, 15970, 34564, 39135],
    23: [28165, 27934, 37880, 41963, 42980],
    26: [44539, 27586, 32756],
    27: [20346, 1850, 37525, 27300, 57885, 65019],
    31: [16707, 9424, 9613, 61607, 38665, 30656],
}
out = {}
for L in LAYERS:
    sp = hf_hub_download(SAE_REPO, f"layer{L}.sae.pt")
    W_dec = torch.load(sp, map_location="cpu")["W_dec"].float()  # [d_model, n_features]
    sup, enh = SUPPRESSED.get(L, []), ENHANCED.get(L, [])
    if not sup:
        print(f"L{L}: no suppressed features listed; skipping")
        continue
    v = torch.zeros(W_dec.shape[0])
    for f in sup:
        d = W_dec[:, f]
        v += d / (d.norm() + 1e-8)
    for f in enh:
        d = W_dec[:, f]
        v -= d / (d.norm() + 1e-8)
    v = (v / (v.norm() + 1e-8)).numpy().astype(np.float32)
    p = run / "vectors" / f"effort_L{L}"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p.with_suffix(".npz"), d=v, layer=L, suppressed=np.array(sup), enhanced=np.array(enh))
    # cosine with tedium, for the record
    try:
        ted = np.load(run / "vectors" / "tedium.npz")["d"].astype(np.float32)
        cos = float(v @ ted / (np.linalg.norm(v) * np.linalg.norm(ted) + 1e-8))
    except Exception:
        cos = float("nan")
    out[L] = {"n_suppressed": len(sup), "n_enhanced": len(enh), "cos_with_tedium": cos}
    print(f"L{L}: {len(sup)} suppressed - {len(enh)} enhanced -> vectors/effort_L{L}.npz | cos(effort, tedium) = {cos:+.3f}")
print("\n" + json.dumps(out, indent=1))
print("\nNext: steer it with the existing prompt_channel.py machinery --")
print("  baseline + ABLATE effort  -> does removing the replanning axis reproduce the prompt's protection?")
print("  prompt   + ADD    effort  -> does amplifying it break the prompt? (the tug-of-war, on the right axis)")
