"""Stage-2 / Stage-3b analysis: does the prompt's effect survive when generated tokens cannot
attend to the instruction span / to the model's own notes (full-attention layers only)?

Behaviour: shortcut rate (judge), commit profile (attempted / hook-blocked /
succeeded; cheat rate given blocked), prefilter split — per condition, Fisher
vs dtm_prompt (unmasked, in-run control) and vs dtm_baseline. Also how many
turns were actually masked and how many decode steps had a block applied.
Commit probe: per commit event, log-prob of " --no-verify" / " -n" / "\n" with the
block OFF vs ON (paired within event); Wilcoxon over events.
KL probe (added 2026-08-30): per post-failure turn (decision event) and every 4th
turn (control), the first 20 generated tokens replayed with the block OFF vs ON:
KL(off||on) mean/first-step, log-prob of the generated tokens under ON minus OFF,
argmax changes. Unit = rollout (per-rollout means), Mann-Whitney post-fail vs
control within condition, and across conditions vs dtm_prompt_mask_instr.
Conditions: the five Stage-2 conditions plus any `dtm_prompt_mask_instr@<headset>`
rows (Stage 3b) found in the phase.
Usage: dt_mask_analysis.py [phase]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon

from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
PHASE = sys.argv[1] if len(sys.argv) > 1 else "dt_mask"
BASE_ORDER = ["dtm_baseline", "dtm_prompt", "dtm_prompt_mask_instr", "dtm_prompt_mask_notes", "dtm_prompt_mask_both"]
side = {r["id"]: r["judge"] for r in read_jsonl(run / "rejudge.jsonl")} if (run / "rejudge.jsonl").exists() else {}
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == PHASE and r.get("status") == "ok"]
ORDER = BASE_ORDER + sorted({r["condition"] for r in rows} - set(BASE_ORDER))


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
    if g:
        print(f"  {c:32s} n={len(g):2d} judged={len(gj):2d} shortcuts={s:2d} rate={s / len(gj) if gj else 0:.2f} | attempted={att} blocked={len(blk)} succeeded={suc} cheat|blocked={rs}/{len(blk)} | prefilter g/p/b={pf.get('good', 0)}/{pf.get('partial', 0)}/{pf.get('bad', 0)} | masked_turns/rollout={mt:.1f} applied_steps/rollout={ap:.0f}")
print("\n=== Fisher's exact ===")
for c in ORDER:
    if c in ("dtm_prompt", "dtm_baseline"):
        continue
    s, n = stats[c]
    for ref in ("dtm_prompt", "dtm_baseline", "dtm_prompt_mask_instr"):
        if ref == c:
            continue
        sr, nr = stats[ref]
        if n and nr and (ref != "dtm_prompt_mask_instr" or "@" in c):
            print(f"  {c:32s} vs {ref:22s}: {s}/{n} vs {sr}/{nr}  p={fisher_exact([[s, n - s], [sr, nr - sr]])[1]:.3f}")
sp, npr = stats["dtm_prompt"]; sb, nb = stats["dtm_baseline"]
if npr and nb:
    print(f"  {'dtm_prompt':32s} vs dtm_baseline         : {sp}/{npr} vs {sb}/{nb}  p={fisher_exact([[sp, npr - sp], [sb, nb - sb]])[1]:.3f}")

print("\n=== commit-position probe: log-prob of continuations, block OFF vs ON (paired per commit event) ===")
for c in ORDER:
    ev = [p for r in by[c] for p in (r.get("probe") or []) if p.get("off")]
    if not ev:
        if by[c]:
            print(f"  {c:32s} no commit events")
        continue
    conts = list(ev[0]["off"].keys())
    line = f"  {c:32s} events={len(ev):3d} rollouts={sum(1 for r in by[c] if r.get('probe'))}"
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


def kl_rollout_means(r):
    """Per-rollout means of the KL probe, split post-fail vs control."""
    out = {}
    for name, sel in (("post_fail", True), ("control", False)):
        ev = [e for e in (r.get("kl_probe") or []) if bool(e.get("post_fail")) == sel]
        if ev:
            out[name] = {"n": len(ev),
                         "kl_mean": float(np.mean([e["kl_mean"] for e in ev])),
                         "kl_first": float(np.mean([e["kl_first"] for e in ev])),
                         "kl_max": float(np.mean([e["kl_max"] for e in ev])),
                         "lp_drop": float(np.mean([e["lp_gen_on"] - e["lp_gen_off"] for e in ev])),
                         "argmax_changed": float(np.mean([e["argmax_changed"] for e in ev]))}
    return out


print("\n=== KL probe: first-20-token distribution with the block OFF vs ON (unit = rollout; post-failure turns vs every-4th-turn control) ===")
kl_by_cond = {}
for c in ORDER:
    per = [kl_rollout_means(r) for r in by[c]]
    per = [p for p in per if p]
    if not per:
        continue
    kl_by_cond[c] = per
    for name in ("post_fail", "control"):
        vals = [p[name] for p in per if name in p]
        if not vals:
            continue
        R = len(vals)
        def ms(k):
            a = np.array([v[k] for v in vals]); return f"{a.mean():.4f}±{a.std(ddof=1) / np.sqrt(R) if R > 1 else 0:.4f}"
        print(f"  {c:32s} {name:9s} R={R:2d} events/rollout={np.mean([v['n'] for v in vals]):5.1f} | KL mean={ms('kl_mean')} first={ms('kl_first')} max={ms('kl_max')} | lp(gen) on-off={ms('lp_drop')} | argmax changed/20={ms('argmax_changed')}")
    pfv = [p["post_fail"]["kl_mean"] for p in per if "post_fail" in p]
    ctv = [p["control"]["kl_mean"] for p in per if "control" in p]
    if len(pfv) >= 3 and len(ctv) >= 3:
        print(f"  {'':32s} post_fail vs control KL mean: MW p={mannwhitneyu(pfv, ctv, alternative='two-sided').pvalue:.3f}")
# --- context-length-matched contrast -------------------------------------------------
# post-failure turns happen late (long contexts); the every-4th-turn control set
# includes turn 0 and other early turns, where a 70-token span is a much larger
# share of the context. Compare within context-length buckets instead.
CTX = {}
ctxp = run / "phase4" / f"dt_ctx_lengths_{PHASE}.json"
if ctxp.exists():
    CTX = json.loads(ctxp.read_text())
if CTX:
    BUCKETS = [(0, 8000), (8000, 20000), (20000, 50000), (50000, 10**9)]
    print("\n=== KL probe matched on context length (per EVENT, buckets by prompt tokens) ===")
    print("  (events within a rollout are not independent: the per-event p is anti-conservative and is")
    print("   shown only alongside a rollout-clustered test on per-rollout means -- the latter is the honest one)")
    for c in ORDER:
        evs = []
        for r in by[c]:
            per = CTX.get(r["id"], {})
            for e in (r.get("kl_probe") or []):
                n = per.get(str(e.get("turn")))
                if n:
                    evs.append((n, bool(e.get("post_fail")), e["kl_mean"], e["argmax_changed"], r["id"]))
        if not evs:
            continue
        print(f"  {c}")
        for lo, hi in BUCKETS:
            sel = [e for e in evs if lo <= e[0] < hi]
            if not sel:
                continue
            pf = [e for e in sel if e[1]]
            ct = [e for e in sel if not e[1]]

            def f(rows_, k):
                return f"{np.mean([r[k] for r in rows_]):.4f}" if rows_ else "  -   "

            def per_rollout(rows_):
                d = collections.defaultdict(list)
                for r in rows_:
                    d[r[4]].append(r[2])
                return [float(np.mean(v)) for v in d.values()]

            line = f"    ctx {lo // 1000:3d}-{'inf' if hi > 10**8 else hi // 1000:>3}k: n={len(sel):4d} | post_fail n={len(pf):4d} KL={f(pf, 2)} argmax={f(pf, 3)} | control n={len(ct):4d} KL={f(ct, 2)} argmax={f(ct, 3)}"
            if len(pf) >= 5 and len(ct) >= 5:
                line += f" | per-event p={mannwhitneyu([r[2] for r in pf], [r[2] for r in ct], alternative='two-sided').pvalue:.3f}"
            rp, rc = per_rollout(pf), per_rollout(ct)
            if len(rp) >= 3 and len(rc) >= 3:
                line += f" | clustered p={mannwhitneyu(rp, rc, alternative='two-sided').pvalue:.3f} (R={len(rp)}/{len(rc)})"
            elif rp and rc:
                line += f" | R={len(rp)}/{len(rc)} too few for clustered test"
            print(line)
else:
    print(f"\n(no {ctxp.name}: run scripts/dt_ctx_lengths.py for the context-length-matched contrast)")

ref = "dtm_prompt_mask_instr"
if ref in kl_by_cond:
    print("\n  across conditions (post-fail KL mean per rollout) vs", ref)
    rv = [p["post_fail"]["kl_mean"] for p in kl_by_cond[ref] if "post_fail" in p]
    for c, per in kl_by_cond.items():
        if c == ref:
            continue
        cv = [p["post_fail"]["kl_mean"] for p in per if "post_fail" in p]
        if len(cv) >= 3 and len(rv) >= 3:
            print(f"    {c:32s} {np.mean(cv):.4f} vs {np.mean(rv):.4f}  MW p={mannwhitneyu(cv, rv, alternative='two-sided').pvalue:.3f}")
