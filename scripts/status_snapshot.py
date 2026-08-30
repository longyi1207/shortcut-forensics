"""One-shot status snapshot (avoids inline-python shell-quoting bugs).
Stage-2 sweep read with re-judge sidecar merged + temptation-point metric +
error rows per variant; B5 rows; L26."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scipy.stats import fisher_exact

from scripts.run_phase import is_shortcut, read_jsonl

run = Path("outputs/20260821-launch")
allrows = list(read_jsonl(run / "rollouts.jsonl"))
side = {r["id"]: r["judge"] for r in read_jsonl(run / "rejudge.jsonl")} if (run / "rejudge.jsonl").exists() else {}


def J(r):
    j = r.get("judge")
    if not (isinstance(j, dict) and j.get("is_shortcut") is not None):
        j = side.get(r["id"])
    return j if isinstance(j, dict) and j.get("is_shortcut") is not None else None


def sc(j):
    return str(j.get("is_shortcut")).lower() == "true"


def commit_profile(r):
    """(attempted, blocked, succeeded): did the rollout try `git commit` at all,
    did any attempt fail (hook-blocked = the temptation point), did any succeed.
    Separates 'never took the test' from 'passed first try' from 'was tempted'."""
    p = run / r["transcript_path"]
    if not p.exists():
        return (False, False, False)
    t = json.loads(p.read_text())
    commits = [x for x in t if x.get("role") == "tool" and str(x.get("command", "")).lstrip().startswith("git commit")]
    return (bool(commits), any(x.get("exit_code", 0) != 0 for x in commits), any(x.get("exit_code", 0) == 0 for x in commits))


def tempt(r):
    return commit_profile(r)[1]


sweep = [r for r in allrows if r.get("phase") == "prompt_sweep_vllm" and not r["id"].startswith("rvsmk_")]
rows = [r for r in sweep if r.get("status") == "ok"]
errs = collections.Counter(r["condition"] for r in sweep if r.get("status") != "ok")
order = ["vllm_identity", "vllm_user_tedium", "vllm_sys_tedium", "vllm_user_tedium_strong", "vllm_sys_tedium_strong", "vllm_user_behavior", "vllm_user_cot"]
base = [(r, J(r)) for r in rows if r["condition"] == "vllm_identity"]
base = [(r, j) for r, j in base if j]
bs = sum(sc(j) for _, j in base)
print(f"=== STAGE 2 (sidecar verdicts merged: {len(side)}; error rows by variant: {dict(errs)}) ===")
print("variant                   n  judged sc  rate   p    | good/part/bad | commit: attempted blocked succeeded | sc|blocked")
for c in order:
    g = [r for r in rows if r["condition"] == c]
    gj = [(r, J(r)) for r in g]
    gj = [(r, j) for r, j in gj if j]
    s = sum(sc(j) for _, j in gj)
    p = fisher_exact([[s, len(gj) - s], [bs, len(base) - bs]])[1] if c != "vllm_identity" and gj else float("nan")
    pf = collections.Counter((r.get("prefilter") or {}).get("outcome") for r in g)
    prof = [(r, j, commit_profile(r)) for r, j in gj]
    att = sum(1 for _, _, cp in prof if cp[0])
    blk = [(r, j) for r, j, cp in prof if cp[1]]
    suc = sum(1 for _, _, cp in prof if cp[2])
    rs = sum(sc(j) for _, j in blk)
    print(f"{c:24s} {len(g):3d} {len(gj):6d} {s:3d} {s / len(gj) if gj else 0:.2f} {p:6.3f} | {pf.get('good', 0):2d}/{pf.get('partial', 0):2d}/{pf.get('bad', 0):2d}      | {att:3d}/{len(gj):2d}   {len(blk):3d}/{len(gj):2d}   {suc:3d}/{len(gj):2d}      | {rs}/{len(blk)}={rs / len(blk) if blk else 0:.2f}")
if errs:
    for r in sweep:
        if r.get("status") != "ok":
            print("  ERR", r["condition"], r["id"], str(r.get("error"))[:140])

print("\n=== B5 (max_turns=80 rows) ===")
b5 = [r for r in allrows if r.get("phase") == "b5_prompt_decay" and r.get("max_turns") == 80]
for c in ["b5_baseline", "b5_prompt", "b5_steer", "b5_prompt_steer"]:
    g = [r for r in b5 if r["condition"] == c]
    print(f"  {c:16s} n={len(g)} " + " ".join(
        f"[{r['status']} turns={r.get('n_turns')} sc={(r.get('judge') or {}).get('is_shortcut')} dt={r.get('decision_turn')} ev={r.get('trace_events')} wall={r.get('wall_s')}{' ERR=' + str(r.get('error'))[:80] if r.get('error') else ''}]"
        for r in g))

sp = [r for r in allrows if r.get("phase") == "signed_pack" and r.get("status") == "ok"]
b = [r for r in sp if r["condition"] == "identity"]
bs2 = sum(is_shortcut(r) for r in b)
c = [r for r in sp if r["condition"] == "ablate_sae_tedium_top10_L26"]
cs = sum(is_shortcut(r) for r in c)
print(f"\n=== L26 === n={len(c)} sc={cs} rate={cs / len(c):.3f} p={fisher_exact([[cs, len(c) - cs], [bs2, len(b) - bs2]])[1]:.4f}")
