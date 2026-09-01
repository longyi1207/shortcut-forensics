"""Print the recurrent-channel (GDN swap) rows and their running rate, next to the
reference cells that complete the channel 2x2."""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
side = {r["id"]: r["judge"] for r in read_jsonl(run / "rejudge.jsonl")} if (run / "rejudge.jsonl").exists() else {}
rows = list(read_jsonl(run / "rollouts.jsonl"))


def sc(r):
    j = r.get("judge")
    if not (isinstance(j, dict) and j.get("is_shortcut") is not None):
        j = side.get(r["id"])
    return isinstance(j, dict) and str(j.get("is_shortcut")).lower() == "true"


def complete(r):
    """Was EVERY turn actually ablated? run_rollout catches custom_prefill errors and
    falls back to a normal prefill, so an OOM on a late turn silently leaves that turn
    UNABLATED. Rows written before the flag existed are judged by prefill_turns."""
    if r.get("ablation_complete") is not None:
        return bool(r["ablation_complete"])
    pt, nt = r.get("prefill_turns"), r.get("n_turns")
    return None if (pt is None or not nt) else pt >= nt


for cond in ("dtk_prompt_gdnswap", "dtk_prompt_gdnnull"):
    allg = [r for r in rows if r.get("condition") == cond and r.get("status") == "ok"]
    g = [r for r in allg if complete(r)]
    bad = [r for r in allg if not complete(r)]
    if bad:
        print("!! {} rows EXCLUDED (ablation incomplete: {}): {}".format(
            len(bad), ", ".join("{}={}/{}".format(r["id"], r.get("prefill_turns"), r.get("n_turns")) for r in bad[:4]), cond))
    print("=== {} : {} usable rows ===".format(cond, len(g)))
    for r in g:
        print("  {} turns={} swapped={} over {} prefills shortcut={} wall={}s".format(
            r["id"], r.get("n_turns"), r.get("swapped_tensors"), r.get("prefill_turns"), sc(r), r.get("wall_s")))
    if g:
        print("  --> {}/{} = {:.2f}".format(sum(sc(r) for r in g), len(g), sum(sc(r) for r in g) / len(g)))

print("\n=== the channel 2x2 (attention content x recurrent state) ===")
REF = {
    "prompt intact          (real, real)": {("dt_capture", "dt_prompt"), ("dt_mask", "dtm_prompt")},
    "swapout                (filler, real)": {("dt_kvswap", "dtk_prompt_swapout")},
    "gdnswap                (real, filler)": {("dt_kvswap", "dtk_prompt_gdnswap")},
    "filler text            (filler, filler)": {("dt_kvswap", "dtk_filler")},
    "no instruction at all": {("dt_capture", "dt_baseline"), ("dt_mask", "dtm_baseline")},
}
for name, pairs in REF.items():
    g = [r for r in rows if (r.get("phase"), r.get("condition")) in pairs and r.get("status") == "ok"]
    g = [r for r in g if r.get("condition") not in ("dtk_prompt_gdnswap", "dtk_prompt_gdnnull") or complete(r)]
    if g:
        print("  {:42s} {:2d}/{:2d} = {:.2f}".format(name, sum(sc(r) for r in g), len(g), sum(sc(r) for r in g) / len(g)))
    else:
        print("  {:42s} (no rows yet)".format(name))
