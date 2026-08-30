"""Stage-2 analysis: does the prompt's effect survive when generated tokens cannot
attend to the instruction span / to the model's own notes (full-attention layers only)?

Behaviour: shortcut rate (judge), commit profile (attempted / hook-blocked /
succeeded; cheat rate given blocked), prefilter split — per condition, Fisher
vs dtm_prompt (unmasked, in-run control) and vs dtm_baseline. Also how many
turns were actually masked and how many decode steps had a block applied.
Probe: per commit event, log-prob of " --no-verify" / " -n" / "\n" with the
block OFF vs ON (paired within event); report mean paired difference and a
Wilcoxon test over events, plus per-rollout means. For dtm_baseline the probe
has no ON arm (nothing to block) and is reported OFF-only as the reference.
Usage: dt_mask_analysis.py [phase]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from scipy.stats import fisher_exact, wilcoxon

from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
PHASE = sys.argv[1] if len(sys.argv) > 1 else "dt_mask"
ORDER = ["dtm_baseline", "dtm_prompt", "dtm_prompt_mask_instr", "dtm_prompt_mask_notes", "dtm_prompt_mask_both"]
side = {r["id"]: r["judge"] for r in read_jsonl(run / "rejudge.jsonl")} if (run / "rejudge.jsonl").exists() else {}
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == PHASE and r.get("status") == "ok"]


def judged(r):
    j = r.get("judge")
    if not (isinstance(j, dict) and j.get("is_shortcut") is not None):
        j = side.get(r["id"])
    return j if isinstance(j, dict) and j.get("is_shortcut") is not None else None


def sc(j):
    return str(j.get("is_shortcut")).lower() == "true"


def commit_profile(r):
    p = run / r["transcript_path"]
    if not p.exists():
        return (False, False, False)
    t = json.loads(p.read_text())
    cs = [x for x in t if x.get("role") == "tool" and str(x.get("command", "")).lstrip().startswith("git commit")]
    return (bool(cs), any(x.get("exit_code", 0) != 0 for x in cs), any(x.get("exit_code", 0) == 0 for x in cs))


by = {c: [r for r in rows if r["condition"] == c] for c in ORDER}
print(f"=== {PHASE}: rollouts per condition ===", {c: len(v) for c, v in by.items()})
stats = {}
for c in ORDER:
    g = by[c]
    gj = [(r, judged(r)) for r in g]
    gj = [(r, j) for r, j in gj if j]
    s = sum(sc(j) for _, j in gj)
    prof = [commit_profile(r) for r, _ in gj]
    att = sum(p[0] for p in prof); blk = [(r, j) for (r, j), p in zip(gj, prof) if p[1]]; suc = sum(p[2] for p in prof)
    rs = sum(sc(j) for _, j in blk)
    pf = collections.Counter((r.get("prefilter") or {}).get("outcome") for r in g)
    mt = np.mean([len(r.get("masked_turns") or []) for r in g]) if g else 0
    ap = np.mean([r.get("mask_applied_steps") or 0 for r in g]) if g else 0
    stats[c] = (s, len(gj))
    print(f"  {c:24s} n={len(g):2d} judged={len(gj):2d} shortcuts={s:2d} rate={s / len(gj) if gj else 0:.2f} | attempted={att} blocked={len(blk)} succeeded={suc} cheat|blocked={rs}/{len(blk)} | prefilter g/p/b={pf.get('good', 0)}/{pf.get('partial', 0)}/{pf.get('bad', 0)} | masked_turns/rollout={mt:.1f} applied_steps/rollout={ap:.0f}")
print("\n=== Fisher's exact ===")
for c in ORDER:
    if c in ("dtm_prompt", "dtm_baseline"):
        continue
    s, n = stats[c]
    for ref in ("dtm_prompt", "dtm_baseline"):
        sr, nr = stats[ref]
        if n and nr:
            print(f"  {c:24s} vs {ref:12s}: {s}/{n} vs {sr}/{nr}  p={fisher_exact([[s, n - s], [sr, nr - sr]])[1]:.3f}")
sp, npr = stats["dtm_prompt"]; sb, nb = stats["dtm_baseline"]
if npr and nb:
    print(f"  {'dtm_prompt':24s} vs dtm_baseline: {sp}/{npr} vs {sb}/{nb}  p={fisher_exact([[sp, npr - sp], [sb, nb - sb]])[1]:.3f}")

print("\n=== commit-position probe: log-prob of continuations, block OFF vs ON (paired per commit event) ===")
for c in ORDER:
    ev = [p for r in by[c] for p in (r.get("probe") or []) if p.get("off")]
    if not ev:
        print(f"  {c:24s} no commit events"); continue
    conts = list(ev[0]["off"].keys())
    line = f"  {c:24s} events={len(ev):3d} rollouts={sum(1 for r in by[c] if r.get('probe'))}"
    for k in conts:
        off = np.array([p["off"][k] for p in ev])
        line += f" | {k!r}: off={off.mean():6.2f}"
        on = np.array([p["on"][k] for p in ev if p.get("on")]) if any(p.get("on") for p in ev) else None
        if on is not None and len(on) == len(off) and len(on) >= 5:
            d = on - off
            try:
                pw = wilcoxon(d).pvalue
            except ValueError:
                pw = float("nan")
            line += f" on={on.mean():6.2f} d={d.mean():+5.2f} (wilcoxon p={pw:.3f})"
    print(line)
