"""B5 follow-up: token-SPARSE readouts + random-control sanity.

The main b5_analysis averages the projection over all ~370 generated tokens
per turn. If a prompt shifts only a sparse subset of "trait-bearing" tokens
(the 2026 prompting-vs-steering literature's claim), a per-turn mean can hide
it. Per rollout x layer x direction, over generated tokens (step>0) in each
of three turn windows (early t<10, mid 10<=t<40, late t>=40), compute:
  * mean            (what b5_analysis already uses)
  * p90 / max       (tail of the distribution -- sparse large shifts)
  * frac |proj|>tau (spec's C9 sparsity metric; tau = 0.5 x median |proj| of
                     the BASELINE arm's early window, per layer x direction)
and compare prompt vs baseline (and steer vs baseline) with Mann-Whitney.
Also prints the random_ctrl comparisons so a spurious condition effect on a
meaningless direction would show up.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import yaml
from scipy.stats import mannwhitneyu

from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
MAX_TURNS = yaml.safe_load((run / "config.frozen.yaml").read_text())["env"]["max_turns"]
CONDS = ["b5_baseline", "b5_prompt", "b5_steer", "b5_prompt_steer"]
WINDOWS = {"early(t<10)": (0, 10), "mid(10-40)": (10, 40), "late(t>=40)": (40, MAX_TURNS)}
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == "b5_prompt_decay" and r.get("status") == "ok" and r.get("max_turns") == MAX_TURNS and r.get("trace_path")]
by = {c: [r for r in rows if r["condition"] == c] for c in CONDS}
print("n per arm:", {c: len(v) for c, v in by.items()})

# collect per-rollout token arrays per window
tok = {c: {} for c in CONDS}  # tok[c][(L,dir,win)] -> list of np.array (per rollout)
for c in CONDS:
    for r in by[c]:
        z = np.load(run / r["trace_path"])
        for L in (19, 26):
            p = f"L{L}_"
            turn, step = z[p + "turn"], z[p + "step"]
            for key in [k for k in z.files if k.startswith(p + "proj_")]:
                d = key[len(p) + 5:]
                v = z[key]
                for wn, (lo, hi) in WINDOWS.items():
                    m = (step > 0) & (turn >= lo) & (turn < hi)
                    if m.any():
                        tok[c].setdefault((L, d, wn), []).append(v[m])

def stat(arrs, fn):
    return np.array([fn(a) for a in arrs if len(a)])

for L in (19, 26):
    dirs = sorted({k[1] for k in tok["b5_baseline"] if k[0] == L})
    for d in dirs:
        base_early = np.concatenate(tok["b5_baseline"].get((L, d, "early(t<10)"), [np.array([])]))
        tau = 0.5 * np.median(np.abs(base_early)) if len(base_early) else 0.0
        print(f"\n=== L{L} {d}  (tau={tau:.2f}) ===")
        for wn in WINDOWS:
            base = tok["b5_baseline"].get((L, d, wn), [])
            if len(base) < 3:
                continue
            line = f"  {wn:12s}"
            for c in CONDS:
                arrs = tok[c].get((L, d, wn), [])
                if len(arrs) < 3:
                    continue
                mean = stat(arrs, np.mean); p90 = stat(arrs, lambda a: np.percentile(a, 90)); mx = stat(arrs, np.max)
                frac = stat(arrs, lambda a: np.mean(np.abs(a) > tau))
                if c == "b5_baseline":
                    bm, bp, bx, bf = mean, p90, mx, frac
                    line += f" | {c[3:]:13s} mean={mean.mean():6.2f} p90={p90.mean():6.2f} max={mx.mean():6.2f} frac>tau={frac.mean():.2f} (n{len(arrs)})"
                else:
                    pm = mannwhitneyu(mean, bm).pvalue; pp = mannwhitneyu(p90, bp).pvalue; px = mannwhitneyu(mx, bx).pvalue; pf = mannwhitneyu(frac, bf).pvalue
                    line += f" | {c[3:]:13s} mean={mean.mean():6.2f}(p{pm:.2f}) p90={p90.mean():6.2f}(p{pp:.2f}) max={mx.mean():6.2f}(p{px:.2f}) frac>tau={frac.mean():.2f}(p{pf:.2f}) (n{len(arrs)})"
            print(line)
