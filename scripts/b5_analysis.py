"""B5 "Prompt Decay" analysis (WRITEUP §4.12; app/prompt_decay spec).

Inputs: rollouts.jsonl rows with phase="b5_prompt_decay" (production rows only:
max_turns == config max_turns) and their traces/<rid>.npz.

Per rollout and layer we build two per-turn series from the trace:
  prefill(t)   projection at the turn-start position (step==0) for turn t
  decode(t)    mean projection over the generated tokens of turn t (step>0)
Then, per condition, the across-rollout mean +/- 95% CI at each turn t, and
the pre-registered readouts:
  * prompt-vs-baseline gap over t  (does the prompt's shift DECAY?)
    -> gap(t) = mean_prompt(t) - mean_baseline(t); fit gap(t) ~ a*exp(-t/tau)
       by least squares on turns with >= MIN_N rollouts in both arms;
       report tau (half-life = tau*ln2) and the early (t<10) vs late (t>=40) gap.
  * steer-vs-baseline gap over t   (does steering PERSIST?) -- same fit.
  * random_ctrl series             (sanity: should not move with condition)
  * behaviour vs t: for each condition, cumulative fraction of rollouts whose
    decision_turn (first shortcut-type command) is <= t, plus final shortcut
    rate (judge) and prefilter good/partial/bad.
Also a Mann-Whitney of decode(t) prompt vs baseline in early/late windows.

Usage: python scripts/b5_analysis.py [--min-n 3] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import yaml
from scipy.optimize import curve_fit
from scipy.stats import fisher_exact, mannwhitneyu

from scripts.run_phase import is_shortcut, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--min-n", type=int, default=3)
ap.add_argument("--json", type=str, default=None)
args = ap.parse_args()

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
MAX_TURNS = cfg["env"]["max_turns"]
CONDS = ["b5_baseline", "b5_prompt", "b5_steer", "b5_prompt_steer"]
LAYERS = (19, 26)

rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("phase") == "b5_prompt_decay" and r.get("status") == "ok" and r.get("max_turns") == MAX_TURNS and r.get("trace_path")]
by_cond = {c: [r for r in rows if r["condition"] == c] for c in CONDS}
print("=== B5 rollouts per condition ===", {c: len(v) for c, v in by_cond.items()})

# series[cond][layer][dir]["prefill"|"decode"] -> list over rollouts of np.array[MAX_TURNS] (nan where turn absent)
series: dict = {c: {L: collections.defaultdict(lambda: {"prefill": [], "decode": []}) for L in LAYERS} for c in CONDS}
for c in CONDS:
    for r in by_cond[c]:
        z = np.load(run / r["trace_path"])
        for L in LAYERS:
            p = f"L{L}_"
            if p + "turn" not in z.files:
                continue
            turn, step = z[p + "turn"], z[p + "step"]
            for key in [k for k in z.files if k.startswith(p + "proj_")]:
                name = key[len(p) + len("proj_"):]
                v = z[key]
                pre = np.full(MAX_TURNS, np.nan)
                dec = np.full(MAX_TURNS, np.nan)
                for t in range(MAX_TURNS):
                    m0 = (turn == t) & (step == 0)
                    m1 = (turn == t) & (step > 0)
                    if m0.any():
                        pre[t] = v[m0][0]
                    if m1.any():
                        dec[t] = v[m1].mean()
                series[c][L][name]["prefill"].append(pre)
                series[c][L][name]["decode"].append(dec)


def mean_ci(arrs: list[np.ndarray]):
    if not arrs:
        return None, None, None
    a = np.vstack(arrs)
    n = np.sum(~np.isnan(a), axis=0)
    m = np.nanmean(a, axis=0)
    s = np.nanstd(a, axis=0, ddof=1)
    ci = 1.96 * s / np.sqrt(np.maximum(n, 1))
    return m, ci, n


def exp_decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def fit_gap(gap: np.ndarray, n_ok: np.ndarray):
    t = np.arange(len(gap))
    m = (~np.isnan(gap)) & (n_ok >= args.min_n)
    if m.sum() < 6:
        return None
    try:
        popt, _ = curve_fit(exp_decay, t[m], gap[m], p0=(gap[m][:3].mean() - gap[m][-3:].mean(), 20.0, gap[m][-3:].mean()), maxfev=20000)
        a, tau, c0 = popt
        return {"a": float(a), "tau_turns": float(tau), "half_life_turns": float(tau * np.log(2)), "asymptote": float(c0),
                "early_gap(t<10)": float(np.nanmean(gap[m][t[m] < 10])) if (t[m] < 10).any() else None,
                "late_gap(t>=40)": float(np.nanmean(gap[m][t[m] >= 40])) if (t[m] >= 40).any() else None,
                "n_turns_fit": int(m.sum())}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


out = {"n": {c: len(by_cond[c]) for c in CONDS}, "layers": {}}
for L in LAYERS:
    out["layers"][L] = {}
    dirs = sorted({d for c in CONDS for d in series[c][L].keys()})
    for d in dirs:
        for kind in ("prefill", "decode"):
            stats = {}
            base_m, base_ci, base_n = mean_ci(series["b5_baseline"][L][d][kind])
            print(f"\n=== L{L} {d} [{kind}] ===")
            for c in CONDS:
                m, ci, n = mean_ci(series[c][L][d][kind])
                if m is None:
                    print(f"  {c:16s} (no data)")
                    continue
                pick = [0, 2, 5, 10, 20, 40, 60, 79]
                line = "  ".join(f"t{t}:{m[t]:7.2f}±{ci[t]:5.2f}(n{n[t]})" for t in pick if t < MAX_TURNS and n[t] > 0)
                print(f"  {c:16s} {line}")
                stats[c] = {"mean": m.tolist(), "ci95": ci.tolist(), "n": n.tolist()}
            if base_m is not None:
                for c in ("b5_prompt", "b5_steer", "b5_prompt_steer"):
                    m, ci, n = mean_ci(series[c][L][d][kind])
                    if m is None:
                        continue
                    gap = m - base_m
                    n_ok = np.minimum(n, base_n)
                    f = fit_gap(gap, n_ok)
                    stats[f"gap_{c}_vs_baseline"] = {"gap": gap.tolist(), "fit": f}
                    if f and "tau_turns" in f:
                        print(f"  gap {c}-baseline: early(t<10)={f['early_gap(t<10)']}, late(t>=40)={f['late_gap(t>=40)']}, tau={f['tau_turns']:.1f} turns (half-life {f['half_life_turns']:.1f})")
                    # window tests on decode means
                    if kind == "decode":
                        A = np.vstack(series[c][L][d][kind]); B = np.vstack(series["b5_baseline"][L][d][kind])
                        for lo, hi, tag in ((0, 10, "early"), (40, MAX_TURNS, "late")):
                            a = np.nanmean(A[:, lo:hi], axis=1); b = np.nanmean(B[:, lo:hi], axis=1)
                            a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
                            if len(a) >= 3 and len(b) >= 3:
                                pv = mannwhitneyu(a, b, alternative="two-sided").pvalue
                                print(f"     {tag} window t[{lo},{hi}): {c} mean={a.mean():.2f} vs baseline {b.mean():.2f}  MW p={pv:.3f} (n={len(a)},{len(b)})")
                                stats[f"mw_{c}_{tag}"] = {"p": float(pv), "n": [int(len(a)), int(len(b))]}
            out["layers"][L][f"{d}/{kind}"] = stats

print("\n=== behaviour vs t ===")
beh = {}
base = by_cond["b5_baseline"]; bs = sum(is_shortcut(r) for r in base)
for c in CONDS:
    g = by_cond[c]
    if not g:
        continue
    s = sum(is_shortcut(r) for r in g)
    dts = sorted(r["decision_turn"] for r in g if r.get("decision_turn") is not None)
    cum = {t: sum(1 for d in dts if d <= t) / len(g) for t in (10, 20, 40, 60, 79)}
    pf = collections.Counter((r.get("prefilter") or {}).get("outcome") for r in g)
    p = fisher_exact([[s, len(g) - s], [bs, len(base) - bs]])[1] if c != "b5_baseline" and base else float("nan")
    print(f"  {c:16s} n={len(g):2d} shortcut={s}/{len(g)} ({s / len(g):.2f}) p_vs_base={p:.3f} | decision_turn median={np.median(dts) if dts else None} | cum frac with decision<=t: {cum} | prefilter g/p/b={pf.get('good', 0)}/{pf.get('partial', 0)}/{pf.get('bad', 0)}")
    beh[c] = {"n": len(g), "shortcuts": s, "p_vs_base": None if np.isnan(p) else float(p), "decision_turns": dts, "cum": cum, "prefilter": dict(pf)}
out["behaviour"] = beh
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2, default=float))
    print("wrote", args.json)
