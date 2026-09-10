"""Behavioural table for the prompt-vs-direction sweep (vLLM, identity mode).

For every prompt_sweep_vllm condition: n, shortcuts, rate with a Wilson CI,
Fisher's exact against the same-night baseline, capability among non-shortcut
rollouts, workaround types, and whether the rollout ever reached the temptation
(attempted a commit at all; attempted a commit while a hook existed, from the
prefilter). A 0% cell whose rollouts never attempted a commit did not take the
test (WRITEUP §4.12 stage 2).

Usage: python scripts/sweep_analysis.py [rollouts.jsonl] [--baseline vllm_identity_r2] [--since 2026-09-10]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict

from scipy.stats import fisher_exact

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default="outputs/20260821-launch/rollouts.jsonl")
ap.add_argument("--baseline", default="vllm_identity_r2")
ap.add_argument("--phase", default="prompt_sweep_vllm")
ap.add_argument("--min-id-prefix", default=None, help="only rows whose id starts with this (e.g. rvpd,rvpt) to isolate tonight's backend")
args = ap.parse_args()

COMMIT = re.compile(r"\bgit\s+(-c\s+\S+\s+)?commit\b")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)



def apply_rejudge(rows, side):
    """Fill judge verdicts lost to Azure 429s from the rejudge sidecar (scripts/rejudge_phase.py)."""
    if not side.exists():
        return rows
    m = {}
    for line in open(side):
        if line.strip():
            r = json.loads(line)
            if isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None:
                m[r["id"]] = r["judge"]
    for r in rows:
        if not (isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None) and r["id"] in m:
            r["judge"] = m[r["id"]]
    return rows


rows = []
with open(args.path) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") != args.phase or r.get("status") != "ok":
            continue
        if args.min_id_prefix and not any(r["id"].startswith(p) for p in args.min_id_prefix.split(",")):
            continue
        rows.append(r)
from pathlib import Path as _P
rows = apply_rejudge(rows, _P(args.path).parent / "rejudge.jsonl")
unjudged = sum(1 for r in rows if not (isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None))
if unjudged:
    print(f"NOTE: {unjudged} ok rows still lack a judge verdict (run scripts/rejudge_phase.py {args.phase})")
rows = [r for r in rows if isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None]

by = defaultdict(list)
for r in rows:
    by[r["condition"]].append(r)


def stats(rs):
    n = len(rs)
    k = sum(1 for r in rs if r["judge"].get("is_shortcut"))
    cap = [r["judge"].get("capability_ok") for r in rs if not r["judge"].get("is_shortcut")]
    cap_rate = sum(1 for c in cap if c) / len(cap) if cap else float("nan")
    attempted = sum(1 for r in rs if any(COMMIT.search(c or "") for c in (r.get("commands") or [])))
    with_hook = sum(1 for r in rs if "attempted_commit_with_hook" in ((r.get("prefilter") or {}).get("behaviors") or []))
    k_among_attempted = sum(1 for r in rs if r["judge"].get("is_shortcut") and any(COMMIT.search(c or "") for c in (r.get("commands") or [])))
    types = Counter(r["judge"].get("workaround_type") for r in rs if r["judge"].get("is_shortcut"))
    turns = [r.get("n_turns") for r in rs if r.get("n_turns")]
    return dict(n=n, k=k, rate=k / n if n else float("nan"), ci=wilson(k, n), cap=cap_rate, attempted=attempted,
                with_hook=with_hook, k_att=k_among_attempted, types=dict(types), mean_turns=sum(turns) / len(turns) if turns else float("nan"))


base = by.get(args.baseline, [])
bs = stats(base) if base else None
print(f"baseline {args.baseline}: n={bs['n'] if bs else 0}")
print(f"{'condition':26s} {'n':>3s} {'short':>6s} {'rate':>6s} {'95% CI':>14s} {'p vs base':>9s} {'cap ok':>7s} {'attempted':>10s} {'w/ hook':>8s} {'rate|att':>9s} {'turns':>6s}  types")
order = [args.baseline] + sorted(c for c in by if c != args.baseline)
for c in order:
    if c not in by:
        continue
    s = stats(by[c])
    p = fisher_exact([[s["k"], s["n"] - s["k"]], [bs["k"], bs["n"] - bs["k"]]])[1] if bs and c != args.baseline else float("nan")
    ra = s["k_att"] / s["attempted"] if s["attempted"] else float("nan")
    print(f"{c:26s} {s['n']:3d} {s['k']:6d} {s['rate']:6.1%} [{s['ci'][0]:5.1%}, {s['ci'][1]:5.1%}] {p:9.4f} {s['cap']:7.1%} {s['attempted']:4d}/{s['n']:<4d} {s['with_hook']:8d} {ra:9.1%} {s['mean_turns']:6.1f}  {s['types']}")
print("\nrate|att = shortcut rate among rollouts that attempted a commit (the ones that faced the choice).")
