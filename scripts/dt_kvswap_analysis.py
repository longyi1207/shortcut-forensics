"""Stage 4 analysis — 2x2 of text x full-attention content: shortcut rates, commit
profiles, Fisher tests against the relevant references (pooled instruction-text
references dt_prompt + dtm_prompt; pooled no-line references dt_baseline + dtm_baseline).
  swapout vs instruction refs : does removing the instruction's attention CONTENT (text kept) remove the effect?
  swapin  vs filler           : does injecting the instruction's attention content (no text) produce the effect?
  filler  vs no-line refs     : is the neutral filler itself inert?
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
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


def grp(pairs, max_turns=80):
    g = [r for r in rows if (r.get("phase"), r.get("condition")) in pairs and r.get("status") == "ok" and r.get("max_turns", max_turns) == max_turns]
    gj = [(r, J(r)) for r in g]
    gj = [(r, j) for r, j in gj if j]
    return gj, sum(str(j.get("is_shortcut")).lower() == "true" for _, j in gj)


def profile(gj):
    att = blk = suc = 0
    for r, _ in gj:
        p = run / r["transcript_path"]
        if not p.exists():
            continue
        t = json.loads(p.read_text())
        cs = [x for x in t if x.get("role") == "tool" and str(x.get("command", "")).lstrip().startswith("git commit")]
        att += bool(cs); blk += any(x.get("exit_code", 0) != 0 for x in cs); suc += any(x.get("exit_code", 0) == 0 for x in cs)
    return att, blk, suc


cells = {
    "instr refs (dt_prompt+dtm_prompt)": {("dt_capture", "dt_prompt"), ("dt_mask", "dtm_prompt")},
    "no-line refs (dt_baseline+dtm_baseline)": {("dt_capture", "dt_baseline"), ("dt_mask", "dtm_baseline")},
    "dtk_prompt_swapout": {("dt_kvswap", "dtk_prompt_swapout")},
    "dtk_filler": {("dt_kvswap", "dtk_filler")},
    "dtk_filler_swapin": {("dt_kvswap", "dtk_filler_swapin")},
    # instruction fully readable; a length-matched span of ordinary task text
    # gets the filler K/V instead. Tests whether a K/V mismatch is disruptive
    # per se -- without it, an elevated swapin rate cannot be told apart from
    # "inconsistent K/V degrades the model".
    "dtk_prompt_swapctrl": {("dt_kvswap", "dtk_prompt_swapctrl")},
}
res = {}
for name, pairs in cells.items():
    gj, s = grp(pairs)
    res[name] = (gj, s)
    if gj:
        att, blk, suc = profile(gj)
        sw = collections.Counter(r.get("n_swapped_prefills") is not None and r.get("n_swapped_prefills") >= r.get("n_turns", 0) for r, _ in gj) if name.startswith("dtk") else None
        print(f"{name:40s} n={len(gj):2d} shortcuts={s:2d} rate={s / len(gj):.2f} | commit attempted/blocked/succeeded={att}/{blk}/{suc}" + (f" | swap-every-turn={dict(sw)}" if sw else ""))
    else:
        print(f"{name:40s} no rows")


def fisher(a, b):
    (ga, sa), (gb, sb) = res[a], res[b]
    if not ga or not gb:
        return
    p = fisher_exact([[sa, len(ga) - sa], [sb, len(gb) - sb]])[1]
    print(f"  {a} {sa}/{len(ga)}  vs  {b} {sb}/{len(gb)}  Fisher p={p:.3f}")


print("\ncontrasts — is the instruction's full-attention CONTENT necessary?")
fisher("dtk_prompt_swapout", "instr refs (dt_prompt+dtm_prompt)")
fisher("dtk_prompt_swapout", "no-line refs (dt_baseline+dtm_baseline)")
print("\ncontrasts — is it sufficient?")
fisher("dtk_filler_swapin", "dtk_filler")
fisher("dtk_filler_swapin", "no-line refs (dt_baseline+dtm_baseline)")
fisher("dtk_filler_swapin", "instr refs (dt_prompt+dtm_prompt)")
print("\ncontrasts — controls (is the filler inert? is a K/V mismatch disruptive per se?)")
fisher("dtk_filler", "no-line refs (dt_baseline+dtm_baseline)")
fisher("dtk_prompt_swapctrl", "instr refs (dt_prompt+dtm_prompt)")
fisher("dtk_prompt_swapctrl", "no-line refs (dt_baseline+dtm_baseline)")
fisher("dtk_prompt_swapctrl", "dtk_prompt_swapout")
fisher("dtk_prompt_swapctrl", "dtk_filler_swapin")
