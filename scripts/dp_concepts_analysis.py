"""Analyse phase4/dp_concepts_<concept>.json|npz (from scripts/dp_concepts.py).

Per concept, over paired decision points:
  behaviour proxy  mean effect of each variant vs none, Wilcoxon p
  tug-of-war       prompt+cheat vs cheat alone (does the prompt resist the push?)
                   prompt+rand  vs rand alone  (does it resist a random push of the same size?)
  geometry         cos(delta_h_prompt, d) and cos(delta_h_neutral, d) per point and of the mean delta;
                   projection h.d_hat under none / neutral / prompt (does the prompt move the readout?)
Then a cross-concept matrix: mean prompt delta at L19 for every concept against every direction.

Usage: python scripts/dp_concepts_analysis.py [run_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

run = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/20260821-launch")
VARIANTS = ["prompt", "neutral", "honest_steer", "cheat_steer", "prompt_cheat", "prompt_rand", "rand"]


def wp(d):
    d = np.asarray(d, dtype=float)
    return wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")


def load_vec(stem):
    z = np.load(run / "vectors" / f"{stem}.npz", allow_pickle=True)
    return z["d"].astype(np.float32), int(z["layer"])


concepts = sorted(p.stem.replace("dp_concepts_", "") for p in (run / "phase4").glob("dp_concepts_*.json") if p.stem != "dp_concepts_summary")
summary = {}
for c in concepts:
    J = json.loads((run / "phase4" / f"dp_concepts_{c}.json").read_text())
    pts = J["points"]
    if not pts:
        continue
    n = len(pts)
    s = {v: np.array([p["s"][v] - p["s"]["none"] for p in pts]) for v in VARIANTS}
    print(f"\n=== {c}  (L{J['layer']}, {J['sign']}, n={n} decision points from {len({p['rollout'] for p in pts})} rollouts) ===")
    print(f"{'variant':14s} {'mean vs none':>13s} {'median':>8s} {'>0':>7s} {'p':>8s}")
    for v in VARIANTS:
        print(f"{v:14s} {s[v].mean():+13.3f} {np.median(s[v]):+8.3f} {int((s[v] > 0).sum()):3d}/{n:<3d} {wp(s[v]):8.4f}")
    tug = s["prompt_cheat"] - s["cheat_steer"]          # prompt's contribution under the push
    tug_r = s["prompt_rand"] - s["rand"]                 # prompt's contribution under a random push
    prompt_alone = s["prompt"]
    print(f"\n  prompt alone                 {prompt_alone.mean():+.3f}  p={wp(prompt_alone):.4f}")
    print(f"  prompt's contribution under +cheat push   {tug.mean():+.3f}  p={wp(tug):.4f}   (== prompt alone? diff {(tug - prompt_alone).mean():+.3f}, p={wp(tug - prompt_alone):.4f})")
    print(f"  prompt's contribution under random push   {tug_r.mean():+.3f}  p={wp(tug_r):.4f}   (diff vs alone {(tug_r - prompt_alone).mean():+.3f}, p={wp(tug_r - prompt_alone):.4f})")
    print(f"  push alone: concept {s['cheat_steer'].mean():+.3f}  random {s['rand'].mean():+.3f}")
    cos_p = np.array([p["cos"]["prompt_vs_d"] for p in pts]); cos_n = np.array([p["cos"]["neutral_vs_d"] for p in pts])
    cos_pn = np.array([p["cos"]["prompt_vs_neutral"] for p in pts])
    proj = {k: np.array([p["proj"][k] for p in pts]) for k in ("none", "neutral", "prompt")}
    nrm = {k: np.array([p["norms"][k] for p in pts]) for k in ("h_none", "d_prompt", "d_neutral")}
    print(f"\n  geometry at L{J['layer']}: cos(dh_prompt, d) = {cos_p.mean():+.3f} ± {cos_p.std():.3f}   cos(dh_neutral, d) = {cos_n.mean():+.3f} ± {cos_n.std():.3f}   cos(dh_prompt, dh_neutral) = {cos_pn.mean():+.3f}")
    print(f"  |dh_prompt|/|h| = {(nrm['d_prompt'] / nrm['h_none']).mean():.3f}   |dh_neutral|/|h| = {(nrm['d_neutral'] / nrm['h_none']).mean():.3f}")
    dproj = proj["prompt"] - proj["none"]; dproj_n = proj["neutral"] - proj["none"]
    print(f"  projection on d_hat: none {proj['none'].mean():+.2f}  prompt {proj['prompt'].mean():+.2f} (shift {dproj.mean():+.3f}, p={wp(dproj):.4f})  neutral shift {dproj_n.mean():+.3f} (p={wp(dproj_n):.4f})")
    Z = np.load(run / "phase4" / f"dp_concepts_{c}.npz")
    d, L = load_vec({"tedium": "tedium", "desperate": "desperate", "shortcut": "shortcut", "completion_drive": "completion_drive", "disapproval": "disapproval_L17_refit"}[c])
    mean_dp = Z[f"d_prompt_L{L}"].astype(np.float32).mean(0)
    mean_dn = Z[f"d_neutral_L{L}"].astype(np.float32).mean(0)
    cm = float(mean_dp @ d / (np.linalg.norm(mean_dp) * np.linalg.norm(d)))
    cmn = float(mean_dn @ d / (np.linalg.norm(mean_dn) * np.linalg.norm(d)))
    print(f"  cos(MEAN dh_prompt, d) = {cm:+.3f}   cos(MEAN dh_neutral, d) = {cmn:+.3f}   (null scale in {len(d)} dims: ~{1/np.sqrt(len(d)):.3f})")
    summary[c] = {"n": n, "prompt": prompt_alone.mean(), "p_prompt": wp(prompt_alone), "cheat": s["cheat_steer"].mean(),
                  "honest": s["honest_steer"].mean(), "tug_contrib": tug.mean(), "tug_r_contrib": tug_r.mean(),
                  "cos_point": cos_p.mean(), "cos_mean": cm, "proj_shift": dproj.mean(), "p_proj": wp(dproj)}

# cross-concept: each concept's mean prompt delta at L19 against every L19 direction
print("\n=== cross-concept: cos(mean dh_prompt @L19, d_x @L19) ===")
dirs = {x: load_vec(x)[0] for x in ("tedium", "desperate", "shortcut", "completion_drive", "disapproval")}
print(f"{'prompt \\ direction':20s}" + "".join(f"{x[:12]:>13s}" for x in dirs))
for c in concepts:
    Z = np.load(run / "phase4" / f"dp_concepts_{c}.npz")
    if "d_prompt_L19" not in Z:
        continue
    m = Z["d_prompt_L19"].astype(np.float32).mean(0)
    print(f"{c:20s}" + "".join(f"{float(m @ v / (np.linalg.norm(m) * np.linalg.norm(v))):>+13.3f}" for v in dirs.values()))
# and between prompts
print("\n=== cos between mean prompt deltas @L19 ===")
M = {c: np.load(run / "phase4" / f"dp_concepts_{c}.npz")["d_prompt_L19"].astype(np.float32).mean(0) for c in concepts
     if "d_prompt_L19" in np.load(run / "phase4" / f"dp_concepts_{c}.npz")}
print(f"{'':20s}" + "".join(f"{x[:12]:>13s}" for x in M))
for a, va in M.items():
    print(f"{a:20s}" + "".join(f"{float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))):>+13.3f}" for vb in M.values()))

print("\n=== summary ===")
print(f"{'concept':18s} {'n':>3s} {'prompt':>8s} {'p':>7s} {'+cheat':>8s} {'honest':>8s} {'prompt|push':>12s} {'prompt|rand':>12s} {'cos pt':>8s} {'cos mean':>9s} {'proj shift':>11s}")
for c, v in summary.items():
    print(f"{c:18s} {v['n']:3d} {v['prompt']:+8.3f} {v['p_prompt']:7.4f} {v['cheat']:+8.3f} {v['honest']:+8.3f} {v['tug_contrib']:+12.3f} {v['tug_r_contrib']:+12.3f} {v['cos_point']:+8.3f} {v['cos_mean']:+9.3f} {v['proj_shift']:+11.3f}")
(run / "phase4" / "dp_concepts_summary.json").write_text(json.dumps(summary, indent=1))
