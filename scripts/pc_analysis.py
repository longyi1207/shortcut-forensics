"""prompt_channel causal cells: behaviour + commit profile + trace projections,
with Fisher vs the relevant references (HF backend):
  pc_base_add_delta26     vs b5_baseline (4/30)           -> does adding the prompt's mean delta reproduce the prompt effect?
  pc_prompt_ablate_delta26 vs b5_prompt (0/21)             -> does projecting it out remove the prompt effect?
  pc_prompt_add_tedium19   vs b5_prompt (0/21) and vs add_pos_tedium (16/30, signed_pack) -> tug-of-war
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from scipy.stats import fisher_exact

from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
side = {r["id"]: r["judge"] for r in read_jsonl(run / "rejudge.jsonl")} if (run / "rejudge.jsonl").exists() else {}
rows = list(read_jsonl(run / "rollouts.jsonl"))


def J(r):
    j = r.get("judge")
    if not (isinstance(j, dict) and j.get("is_shortcut") is not None):
        j = side.get(r["id"])
    return j if isinstance(j, dict) and j.get("is_shortcut") is not None else None


def cnt(phase, cond, max_turns=80):
    g = [r for r in rows if r.get("phase") == phase and r.get("condition") == cond and r.get("status") == "ok" and (max_turns is None or r.get("max_turns", max_turns) == max_turns)]
    gj = [(r, J(r)) for r in g]; gj = [(r, j) for r, j in gj if j]
    s = sum(str(j.get("is_shortcut")).lower() == "true" for _, j in gj)
    return g, gj, s


def profile(g):
    att = blk = suc = 0
    for r in g:
        p = run / r["transcript_path"]
        if not p.exists():
            continue
        t = json.loads(p.read_text())
        cs = [x for x in t if x.get("role") == "tool" and str(x.get("command", "")).lstrip().startswith("git commit")]
        att += bool(cs); blk += any(x.get("exit_code", 0) != 0 for x in cs); suc += any(x.get("exit_code", 0) == 0 for x in cs)
    return att, blk, suc


refs = {"b5_baseline": cnt("b5_prompt_decay", "b5_baseline"), "b5_prompt": cnt("b5_prompt_decay", "b5_prompt"), "add_pos_tedium": cnt("signed_pack", "add_pos_tedium", None), "identity(HF)": cnt("signed_pack", "identity", None)}
cells = {"pc_base_add_delta26": ["b5_baseline"], "pc_prompt_ablate_delta26": ["b5_prompt", "b5_baseline"], "pc_prompt_add_tedium19": ["b5_prompt", "add_pos_tedium", "b5_baseline"]}
for name, (g, gj, s) in refs.items():
    print(f"ref {name:22s} n={len(gj):3d} shortcuts={s:2d} rate={s / len(gj) if gj else 0:.2f}")
for cell, rl in cells.items():
    g, gj, s = cnt("prompt_channel", cell)
    if not gj:
        print(f"\n{cell}: no rows yet"); continue
    att, blk, suc = profile([r for r, _ in gj])
    pf = collections.Counter((r.get("prefilter") or {}).get("outcome") for r, _ in gj)
    mp = collections.defaultdict(list)
    for r, _ in gj:
        for L, d in (r.get("trace_mean_proj") or {}).items():
            for k, v in (d or {}).items():
                if v is not None:
                    mp[f"L{L}:{k}"].append(v)
    print(f"\n{cell}: n={len(gj)} shortcuts={s} rate={s / len(gj):.2f} | commit attempted/blocked/succeeded={att}/{blk}/{suc} | prefilter g/p/b={pf.get('good', 0)}/{pf.get('partial', 0)}/{pf.get('bad', 0)}")
    print("   mean trace projections: " + ", ".join(f"{k}={np.mean(v):.2f}" for k, v in sorted(mp.items())))
    for ref in rl:
        rg, rgj, rs = refs[ref]
        if rgj:
            print(f"   vs {ref:15s}: {s}/{len(gj)} vs {rs}/{len(rgj)}  Fisher p={fisher_exact([[s, len(gj) - s], [rs, len(rgj) - rs]])[1]:.3f}")
