"""One table per factor: prompting vs direction steering, behaviour and activation.

Columns
  prompt (vLLM)     shortcut rate under the concept line vs the same-night vLLM baseline
  steer (HF)        shortcut rate under the sign-appropriate steer vs the HF identity baseline
  dp prompt         decision-point readout shift under the line (proxy; see dp_readout_check)
  dp steer          decision-point readout shift under the pro-honest steer
  cos(dh, d)        cosine between the prompt's mean activation footprint and the fitted direction
  proj shift        change in the projection h.d_hat when the line is present

Usage: python scripts/factor_compare.py [rollouts.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

run = Path("outputs/20260821-launch")
path = sys.argv[1] if len(sys.argv) > 1 else str(run / "rollouts.jsonl")

PROMPT = {"tedium": "vllm_user_tedium_strong_r2", "desperate": "vllm_user_desperate", "shortcut": "vllm_user_shortcut",
          "completion_drive": "vllm_user_completion", "disapproval": "vllm_user_disapproval"}
STEER = {"tedium": "ablate_tedium", "desperate": "ablate_desperate", "shortcut": "ablate_shortcut",
         "completion_drive": "ablate_completion_drive", "disapproval": "add_pos_disapproval_L17"}
STEER_CHEAT = {"tedium": "add_pos_tedium"}


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


cells = defaultdict(lambda: [0, 0])  # (phase, condition, id-prefix-class) -> [k, n]
_rows = [json.loads(l) for l in open(path) if l.strip()]
_rows = apply_rejudge([r for r in _rows if r.get("status") == "ok"], Path(path).parent / "rejudge.jsonl")
for r in _rows:
    if not r.get("judge") or r["judge"].get("is_shortcut") is None:
        continue
    key = (r.get("phase"), r.get("condition"))
    cells[key][1] += 1
    cells[key][0] += int(bool(r["judge"]["is_shortcut"]))
    if r.get("phase") == "prompt_sweep_vllm" and r["id"].startswith(("rvpd", "rvpt")):
        k2 = (r.get("phase"), r.get("condition") + "@tonight")
        cells[k2][1] += 1
        cells[k2][0] += int(bool(r["judge"]["is_shortcut"]))


def cell(phase, cond):
    k, n = cells.get((phase, cond), [0, 0])
    return k, n


def fmt(k, n):
    return f"{k:3d}/{n:<3d} {k / n:6.1%}" if n else "      -      "


def pval(a, b):
    (k1, n1), (k2, n2) = a, b
    if not n1 or not n2:
        return float("nan")
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


vb = cell("prompt_sweep_vllm", "vllm_identity_r2@tonight")
vn = cell("prompt_sweep_vllm", "vllm_user_neutral@tonight")
hb = cell("signed_pack", "identity")
summary = json.loads((run / "phase4" / "dp_concepts_summary.json").read_text()) if (run / "phase4" / "dp_concepts_summary.json").exists() else {}

print(f"baselines: vLLM tonight identity {fmt(*vb)}   neutral line {fmt(*vn)} (p vs identity {pval(vn, vb):.3f})   HF identity {fmt(*hb)}")
print(f"\n{'factor':17s} | {'prompt (vLLM)':>16s} {'p':>7s} | {'steer (HF)':>16s} {'p':>7s} | {'dp prompt':>9s} {'dp steer':>8s} | {'cos(dh,d)':>9s} {'proj shift':>10s}")
for c in PROMPT:
    pc = cell("prompt_sweep_vllm", PROMPT[c] + "@tonight")
    sc = cell("signed_pack", STEER[c])
    s = summary.get(c, {})
    print(f"{c:17s} | {fmt(*pc)} {pval(pc, vb):7.4f} | {fmt(*sc)} {pval(sc, hb):7.4f} | {s.get('prompt', float('nan')):+9.2f} {s.get('honest', float('nan')):+8.2f} | {s.get('cos_mean', float('nan')):+9.3f} {s.get('proj_shift', float('nan')):+10.3f}")
if cells.get(("signed_pack", "add_pos_tedium")):
    print(f"\n(tedium pro-cheat steer +add: {fmt(*cell('signed_pack', 'add_pos_tedium'))}, p vs HF identity {pval(cell('signed_pack', 'add_pos_tedium'), hb):.4f})")
for cond in ("pc_prompt_add_tedium19", "pc_prompt_add_rand19", "pc_add_rand19"):
    k, n = cell("prompt_channel", cond)
    if n:
        print(f"{cond:26s} {fmt(k, n)}")
print("\nprompt and steer columns sit on different backends (vLLM vs HF); compare each to its own baseline, not to each other in absolute terms.")
