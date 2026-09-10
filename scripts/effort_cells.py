"""Experiment 1 readout: does the cost-of-effort / replanning axis behave the way
the tedium axis did not?

  pc_base_ablate_effort26 : no prompt, ABLATE the effort direction
      -> if removing the replanning axis reproduces the prompt's protection,
         the prompt's job is (at least partly) to suppress that axis.
  pc_prompt_add_effort26  : prompt on, ADD the effort direction
      -> the tug-of-war on the right axis. On the tedium axis the prompt gave
         ZERO resistance (15/30 vs 17/32, p=1.0; at half dose 10/21 vs 10/21,
         p=1.0). If the prompt resists here, we have found the axis it uses.
References are pooled across phases, as elsewhere.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scipy.stats import fisher_exact

from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
side = {r["id"]: r["judge"] for r in read_jsonl(run / "rejudge.jsonl")} if (run / "rejudge.jsonl").exists() else {}
rows = list(read_jsonl(run / "rollouts.jsonl"))


def sc(r):
    j = r.get("judge")
    if not (isinstance(j, dict) and j.get("is_shortcut") is not None):
        j = side.get(r["id"])
    return isinstance(j, dict) and str(j.get("is_shortcut")).lower() == "true"


def cell(pairs):
    g = [r for r in rows if (r.get("phase"), r.get("condition")) in pairs and r.get("status") == "ok"]
    g = [r for r in g if (r.get("judge") or side.get(r["id"]))]
    return sum(sc(r) for r in g), len(g)


CELLS = {
    "prompt intact (pooled)": {("b5_prompt_decay", "b5_prompt"), ("dt_capture", "dt_prompt"), ("dt_mask", "dtm_prompt")},
    "no instruction (pooled)": {("b5_prompt_decay", "b5_baseline"), ("dt_capture", "dt_baseline"), ("dt_mask", "dtm_baseline")},
    "baseline + ABLATE effort": {("prompt_channel", "pc_base_ablate_effort26")},
    "prompt + ADD effort (a=26, matched)": {("prompt_channel", "pc_prompt_add_effort26_a26")},
    "no prompt + ADD effort (a=26)": {("prompt_channel", "pc_base_add_effort26_a26")},
    "[underdosed a=1, retired] prompt+add": {("prompt_channel", "pc_prompt_add_effort26")},
    "[tedium axis] prompt + ADD tedium a=1": {("prompt_channel", "pc_prompt_add_tedium19")},
    "[tedium axis] ADD tedium a=1, no prompt": {("signed_pack", "add_pos_tedium")},
}
res = {}
for name, pairs in CELLS.items():
    s, n = cell(pairs)
    res[name] = (s, n)
    print("{:40s} {:2d}/{:2d} = {}".format(name, s, n, "{:.2f}".format(s / n) if n else "-"))


def f(a, b):
    (sa, na), (sb, nb) = res[a], res[b]
    if na and nb:
        print("  {:38s} vs {:30s} {}/{} vs {}/{}  p={:.4f}".format(
            a, b, sa, na, sb, nb, fisher_exact([[sa, na - sa], [sb, nb - sb]])[1]))


print("\n--- does ablating the replanning axis reproduce the prompt's protection? ---")
f("baseline + ABLATE effort", "no instruction (pooled)")
f("baseline + ABLATE effort", "prompt intact (pooled)")
print("\n--- the tug-of-war, on the right axis ---")
f("prompt + ADD effort (a=26, matched)", "prompt intact (pooled)")
f("prompt + ADD effort (a=26, matched)", "no prompt + ADD effort (a=26)")
f("no prompt + ADD effort (a=26)", "no instruction (pooled)")
print("\n(for contrast, the same test on the tedium axis gave prompt+add = 15/30 vs prompt 0/21, p<0.001,")
print(" i.e. the prompt offered no resistance at all)")
