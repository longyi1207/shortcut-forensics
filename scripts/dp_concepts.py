"""Prompt vs direction, per concept, measured at decision points.

For each concept c with a fitted direction d_c (vectors/) and a natural-language
line P_c that targets the same disposition (scripts/concept_lines.py), score
the paired decision-point readout under:

    none          the stored context, no extra line
    neutral       a matched-length line with no concept content (control)
    prompt        P_c appended to the user message
    honest_steer  the sign-appropriate pro-honest steer (ablate d for pro-cheat
                  concepts; +add d for the pro-honest one)
    cheat_steer   the pro-cheat push (+add d for pro-cheat; ablate for pro-honest)
    prompt_cheat  P_c plus the pro-cheat push          (tug-of-war)
    prompt_rand   P_c plus a random direction at the same layer and norm
    rand          the random push alone

and, on the three unsteered passes (none / neutral / prompt), capture the
last-token residual at layers {17, 19, L_c} so the prompt's footprint
delta_h = h(prompt) - h(none) can be compared with d_c (cosine, projection) and
across concepts.

Decision points come from the dt_baseline rollouts only (generated with no
instruction), so the stored history is not already shaped by any prompt.

Env: SCFX_DPC_CONCEPTS (comma list; default all five), SCFX_DPC_POINTS (40),
     SCFX_DPC_PER_ROLLOUT (3), SCFX_DPC_MAXCTX (20000), SCFX_DPC_SEED_TAG.
Writes phase4/dp_concepts_<concept>.json and phase4/dp_concepts_<concept>.npz.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
import yaml

from scripts.concept_lines import CONCEPT_LINES, CONCEPT_VECTORS, NEUTRAL_LINE
from scripts.dp_common import LastTokenCapture, iter_decision_points, render, score
from scripts.run_phase import read_jsonl
from src.agent_loop import load_model
from src.directions import load_vector
from src.steer import SteeringSession, random_unit_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dp_concepts")

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
CONCEPTS = [c.strip() for c in os.environ.get("SCFX_DPC_CONCEPTS", ",".join(CONCEPT_LINES)).split(",") if c.strip()]
N_POINTS = int(os.environ.get("SCFX_DPC_POINTS", "40"))
PER_ROLLOUT = int(os.environ.get("SCFX_DPC_PER_ROLLOUT", "3"))
MAX_CTX = int(os.environ.get("SCFX_DPC_MAXCTX", "20000"))
CAPTURE_LAYERS = sorted({17, 19} | {CONCEPT_VECTORS[c][1] for c in CONCEPTS})

model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
hidden = model.config.hidden_size

for name, line in list(CONCEPT_LINES.items()) + [("neutral", NEUTRAL_LINE)]:
    logger.info("line %-16s %3d tokens", name, len(tok(line, add_special_tokens=False).input_ids))


def build_specs(concept: str) -> dict[str, tuple[str | None, list[tuple[np.ndarray, int, str, float]]]]:
    stem, layer, sign = CONCEPT_VECTORS[concept]
    d = load_vector(run / "vectors" / stem)["d"].astype(np.float32)
    seed = int(hashlib.sha256(f"dp_concepts:{concept}".encode()).hexdigest(), 16) % (2**31)
    r = random_unit_vector(hidden, seed=seed).astype(np.float32)
    if sign == "pro_cheat":
        honest = (d, layer, "ablate", 1.0)
        cheat = (d, layer, "add", 1.0)
        rand = (r * float(np.linalg.norm(d)), layer, "add", 1.0)
    else:
        honest = (d, layer, "add", 1.0)
        cheat = (d, layer, "ablate", 1.0)
        rand = (r, layer, "ablate", 1.0)
    P = CONCEPT_LINES[concept]
    return {
        "none": (None, []),
        "neutral": (NEUTRAL_LINE, []),
        "prompt": (P, []),
        "honest_steer": (None, [honest]),
        "cheat_steer": (None, [cheat]),
        "prompt_cheat": (P, [cheat]),
        "prompt_rand": (P, [rand]),
        "rand": (None, [rand]),
    }, d, layer, sign


rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("status") == "ok" and r.get("transcript_path")
        and (r.get("phase"), r.get("condition")) == ("dt_capture", "dt_baseline")]
logger.info("concepts=%s points=%d per_rollout=%d max_ctx=%d capture=%s from %d dt_baseline rollouts",
            CONCEPTS, N_POINTS, PER_ROLLOUT, MAX_CTX, CAPTURE_LAYERS, len(rows))

capture = LastTokenCapture(model, CAPTURE_LAYERS)
specs = {c: build_specs(c) for c in CONCEPTS}
results = {c: [] for c in CONCEPTS}
deltas = {c: {"h_none": {L: [] for L in CAPTURE_LAYERS}, "d_prompt": {L: [] for L in CAPTURE_LAYERS},
              "d_neutral": {L: [] for L in CAPTURE_LAYERS}} for c in CONCEPTS}
(run / "phase4").mkdir(parents=True, exist_ok=True)


def save(concept: str):
    (run / "phase4" / f"dp_concepts_{concept}.json").write_text(json.dumps({
        "concept": concept, "layer": specs[concept][2], "sign": specs[concept][3],
        "line": CONCEPT_LINES[concept], "neutral_line": NEUTRAL_LINE, "max_ctx": MAX_CTX,
        "capture_layers": CAPTURE_LAYERS, "points": results[concept]}))
    arrs = {}
    for k, per_layer in deltas[concept].items():
        for L, vs in per_layer.items():
            if vs:
                arrs[f"{k}_L{L}"] = np.stack(vs).astype(np.float16)
    np.savez_compressed(run / "phase4" / f"dp_concepts_{concept}.npz", **arrs)


done = 0
for r, turn, messages in iter_decision_points(run, rows, PER_ROLLOUT):
    if done >= N_POINTS:
        break
    # the three unsteered passes are shared across concepts at this point
    shared: dict[str, float | None] = {}
    h: dict[str, dict[int, torch.Tensor]] = {}
    for name, line in (("none", None), ("neutral", NEUTRAL_LINE)):
        shared[name] = score(model, tok, render(tok, messages, line), MAX_CTX, capture)
        h[name] = dict(capture.h)
    if shared["none"] is None or shared["neutral"] is None:
        continue  # context too long for this point; skip it entirely
    ok_point = True
    point_out: dict[str, dict] = {}
    for c in CONCEPTS:
        spec, d, layer, sign = specs[c]
        sc: dict[str, float] = {"none": shared["none"], "neutral": shared["neutral"]}
        # prompt pass with capture
        sc["prompt"] = score(model, tok, render(tok, messages, spec["prompt"][0]), MAX_CTX, capture)
        hp = dict(capture.h)
        for name in ("honest_steer", "cheat_steer", "prompt_cheat", "prompt_rand", "rand"):
            line, steers = spec[name]
            with ExitStack() as stack:
                for (vec, L, mode, alpha) in steers:
                    stack.enter_context(SteeringSession(model, [L], vec, mode=mode, alpha=alpha))
                sc[name] = score(model, tok, render(tok, messages, line), MAX_CTX, None)
        if any(v is None for v in sc.values()):
            ok_point = False
            break
        dhat = d / np.linalg.norm(d)
        hn, hne = h["none"][layer].numpy(), h["neutral"][layer].numpy()
        hpv = hp[layer].numpy()
        dp, dn = hpv - hn, hne - hn
        proj = {"none": float(hn @ dhat), "neutral": float(hne @ dhat), "prompt": float(hpv @ dhat)}
        cos = {"prompt_vs_d": float(dp @ dhat / (np.linalg.norm(dp) + 1e-9)),
               "neutral_vs_d": float(dn @ dhat / (np.linalg.norm(dn) + 1e-9)),
               "prompt_vs_neutral": float(dp @ dn / (np.linalg.norm(dp) * np.linalg.norm(dn) + 1e-9))}
        norms = {"h_none": float(np.linalg.norm(hn)), "d_prompt": float(np.linalg.norm(dp)),
                 "d_neutral": float(np.linalg.norm(dn)), "d": float(np.linalg.norm(d))}
        point_out[c] = {"rollout": r["id"], "turn": turn, "s": sc, "proj": proj, "cos": cos, "norms": norms}
        for L in CAPTURE_LAYERS:
            deltas[c]["h_none"][L].append(h["none"][L].numpy())
            deltas[c]["d_prompt"][L].append(hp[L].numpy() - h["none"][L].numpy())
            deltas[c]["d_neutral"][L].append(h["neutral"][L].numpy() - h["none"][L].numpy())
    if not ok_point:
        continue
    for c in CONCEPTS:
        results[c].append(point_out[c])
        save(c)
        s = point_out[c]["s"]
        logger.info("%s t%02d %-16s none=%+.2f prompt=%+.2f neutral=%+.2f honest=%+.2f cheat=%+.2f p+cheat=%+.2f p+rand=%+.2f rand=%+.2f | cos(dh,d)=%+.3f proj none/prompt=%.1f/%.1f",
                    r["id"], turn, c, s["none"], s["prompt"] - s["none"], s["neutral"] - s["none"],
                    s["honest_steer"] - s["none"], s["cheat_steer"] - s["none"], s["prompt_cheat"] - s["none"],
                    s["prompt_rand"] - s["none"], s["rand"] - s["none"], point_out[c]["cos"]["prompt_vs_d"],
                    point_out[c]["proj"]["none"], point_out[c]["proj"]["prompt"])
    done += 1
    torch.cuda.empty_cache()

capture.remove()
logger.info("finished: %d points for %s", done, CONCEPTS)
