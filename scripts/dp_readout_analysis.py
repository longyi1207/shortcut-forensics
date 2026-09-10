"""Does s = logP(engage) - logP(replan) separate shortcut from honest rollouts?

Reads phase4/dp_readout_check_*.json (scripts/dp_readout_check.py) and reports
AUROC of several per-rollout summaries of s against the judge's is_shortcut,
plus the same against the continuous shortcut_score. Bootstrap CI over rollouts.

Usage: python scripts/dp_readout_analysis.py [run_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

run = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/20260821-launch")
rows = []
for p in sorted((run / "phase4").glob("dp_readout_check_*.json")):
    rows += json.loads(p.read_text())
rows = [r for r in rows if r["points"]]
y = np.array([r["is_shortcut"] for r in rows], dtype=int)
print(f"{len(rows)} labelled identity rollouts with a scored decision point: {y.sum()} shortcut, {(1 - y).sum()} honest")


def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def boot_ci(score, y, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() and (1 - y[i]).sum():
            vals.append(auroc(score[i], y[i]))
    return np.percentile(vals, [2.5, 97.5])


summaries = {
    "first point": lambda r: r["points"][0]["s"],
    "mean of points": lambda r: float(np.mean([p["s"] for p in r["points"]])),
    "min of points": lambda r: float(np.min([p["s"] for p in r["points"]])),
    "last point": lambda r: r["points"][-1]["s"],
}
print(f"\n{'summary':16s} {'AUROC (shortcut higher = >0.5 means engaged MORE)':>50s} {'95% CI':>16s} {'MWU p':>8s} {'rho vs score':>13s}")
for name, f in summaries.items():
    sc = np.array([f(r) for r in rows], dtype=float)
    a = auroc(sc, y); lo, hi = boot_ci(sc, y)
    p = mannwhitneyu(sc[y == 1], sc[y == 0]).pvalue
    score10 = np.array([r["shortcut_score"] if r["shortcut_score"] is not None else np.nan for r in rows], dtype=float)
    ok = ~np.isnan(score10)
    rho = spearmanr(sc[ok], score10[ok]).correlation if ok.sum() > 5 else float("nan")
    print(f"{name:16s} {a:>50.3f} [{lo:.3f}, {hi:.3f}] {p:8.4f} {rho:+13.3f}")
print("\nThe readout is used with the sign 'higher = more honest', so for it to track behaviour AUROC should sit well BELOW 0.5 here.")
print("Values near 0.5 mean the proxy does not see who cheats; the sentence/component results then rank prompt variants on a quantity that is not the behaviour.")

# by phase, since the two identity pools were generated weeks apart
for ph in sorted({r["phase"] for r in rows}):
    sub = [r for r in rows if r["phase"] == ph]; yy = np.array([r["is_shortcut"] for r in sub], dtype=int)
    sc = np.array([float(np.mean([p["s"] for p in r["points"]])) for r in sub])
    print(f"  {ph:12s} n={len(sub):3d} shortcuts={yy.sum():2d}  AUROC(mean s)={auroc(sc, yy):.3f}")
