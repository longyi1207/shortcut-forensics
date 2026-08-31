"""Stage-2 behavioural test against POOLED references.

The in-run controls are only n=12 each, and dtm_baseline happened to land 0/12
(historically the no-instruction arms run ~13-17%), so per-cell Fisher tests
against them are badly underpowered. This pools every rollout of the same
condition across phases:
  prompt, no mask : b5_prompt + dt_prompt + dtm_prompt
  no instruction  : b5_baseline + dt_baseline + dtm_baseline
Caveat printed with the result: the phases differ in instrumentation
(split_prefill and the probes were added in dt_mask), which does not change the
sampling distribution but is not bit-identical, so the pooling is defensible
rather than exact. Both pooled and in-run numbers are reported.
"""
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


def cell(pairs):
    g = [r for r in rows if (r.get("phase"), r.get("condition")) in pairs and r.get("status") == "ok"]
    gj = [(r, J(r)) for r in g]
    gj = [(r, j) for r, j in gj if j]
    return sum(str(j.get("is_shortcut")).lower() == "true" for _, j in gj), len(gj)


POOLED_PROMPT = {("b5_prompt_decay", "b5_prompt"), ("dt_capture", "dt_prompt"), ("dt_mask", "dtm_prompt")}
POOLED_BASE = {("b5_prompt_decay", "b5_baseline"), ("dt_capture", "dt_baseline"), ("dt_mask", "dtm_baseline")}
MASKS = {
    "mask_instr": {("dt_mask", "dtm_prompt_mask_instr")},
    "mask_notes": {("dt_mask", "dtm_prompt_mask_notes")},
    "mask_both": {("dt_mask", "dtm_prompt_mask_both")},
}
sp, np_ = cell(POOLED_PROMPT)
sb, nb = cell(POOLED_BASE)
print(f"pooled prompt (no mask): {sp}/{np_} = {sp / np_:.3f}")
print(f"pooled no-instruction  : {sb}/{nb} = {sb / nb:.3f}")
print(f"  prompt vs no-instruction: Fisher p={fisher_exact([[sp, np_ - sp], [sb, nb - sb]])[1]:.4f}  <- the effect being masked away")
print()
for name, pairs in MASKS.items():
    s, n = cell(pairs)
    p_prompt = fisher_exact([[s, n - s], [sp, np_ - sp]])[1]
    p_base = fisher_exact([[s, n - s], [sb, nb - sb]])[1]
    print(f"{name:12s} {s}/{n} = {s / n:.2f} | vs pooled prompt p={p_prompt:.4f} | vs pooled no-instruction p={p_base:.4f}")
print()
print("Caveat: phases differ in instrumentation (split_prefill + probes added in dt_mask);")
print("same sampling distribution, not bit-identical. In-run-only numbers are in dt_mask_analysis.py.")
