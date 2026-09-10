"""Figures for the prompt-vs-direction study.

  writeup/figs/prompt_vs_steer.png  per factor: shortcut-rate change under the concept's prompt line
                                    (vLLM, vs the same-night vLLM baseline) next to the change under the
                                    sign-appropriate steering intervention (HF, vs the HF signed-pack
                                    baseline). Wilson 95% intervals on each cell; the neutral line and the
                                    random-direction control are drawn as reference bars.
  writeup/figs/geometry.png         per concept: projection shift along the fitted direction at the prompt
                                    end for the plus-pole plant, the minus-pole plant, the instruction and
                                    the neutral line (positive control vs treatment), and the decision-token
                                    cosine for the same four.

Usage: python scripts/fig_prompt_vs_steer.py [rollouts.jsonl]
Every panel is skipped, not faked, when its input is missing.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = Path("outputs/20260821-launch")
path = Path(sys.argv[1]) if len(sys.argv) > 1 else run / "rollouts.jsonl"
figs = Path("writeup/figs")
figs.mkdir(parents=True, exist_ok=True)

FACTORS = ["tedium", "desperate", "shortcut", "completion_drive", "disapproval"]
PROMPT = {"tedium": "vllm_user_tedium_strong_r2", "desperate": "vllm_user_desperate", "shortcut": "vllm_user_shortcut",
          "completion_drive": "vllm_user_completion", "disapproval": "vllm_user_disapproval"}
STEER = {"tedium": "ablate_tedium", "desperate": "ablate_desperate", "shortcut": "ablate_shortcut",
         "completion_drive": "ablate_completion_drive", "disapproval": "add_pos_disapproval_L17"}
VLLM_BASE, HF_BASE = ("prompt_sweep_vllm", "vllm_identity_r2"), ("signed_pack", "identity")
NEUTRAL = ("prompt_sweep_vllm", "vllm_user_neutral")
RAND = ("prompt_channel", "pc_add_rand19")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def judged(r):
    return isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None


rows = [json.loads(l) for l in open(path) if l.strip()]
rows = [r for r in rows if r.get("status") == "ok"]
side = path.parent / "rejudge.jsonl"
if side.exists():
    m = {r["id"]: r["judge"] for r in (json.loads(l) for l in open(side) if l.strip()) if isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None}
    for r in rows:
        if not judged(r) and r["id"] in m:
            r["judge"] = m[r["id"]]
cells = defaultdict(lambda: [0, 0])
for r in rows:
    if not judged(r):
        continue
    key = (r["phase"], r["condition"])
    if r["phase"] == "prompt_sweep_vllm" and not r["id"].startswith(("rvpd", "rvpt")):
        continue  # only tonight's vLLM cells: the old vLLM build had a different baseline
    cells[key][1] += 1
    cells[key][0] += int(bool(r["judge"]["is_shortcut"]))


def rate(key):
    k, n = cells.get(key, [0, 0])
    lo, hi = wilson(k, n)
    return (k / n if n else float("nan"), lo, hi, n)


# ---------- figure 1: behaviour ----------
vb, hb = rate(VLLM_BASE), rate(HF_BASE)
lows, highs = [], []
fig, ax = plt.subplots(figsize=(9.5, 4.6))
x = np.arange(len(FACTORS))
w = 0.36
for off, src, base, color, label in ((-w / 2, PROMPT, vb, "#1f77b4", "prompt line (vLLM) − same-night vLLM baseline"),
                                     (w / 2, STEER, hb, "#d62728", "direction steering (HF) − HF baseline")):
    phase = "prompt_sweep_vllm" if src is PROMPT else "signed_pack"
    for i, f in enumerate(FACTORS):
        p, lo, hi, n = rate((phase, src[f]))
        if not n or math.isnan(base[0]):
            ax.text(i + off, 0.005, "n=0", ha="center", va="bottom", fontsize=7, color=color)
            continue
        d = p - base[0]
        ax.bar(i + off, d, w, color=color, alpha=0.85, label=label if i == 0 else None)
        ax.errorbar(i + off, d, yerr=[[d - (lo - base[0])], [(hi - base[0]) - d]], fmt="none", ecolor="k", capsize=3, lw=1)
        top = hi - base[0] if d >= 0 else lo - base[0]
        ax.text(i + off, top + (0.01 if d >= 0 else -0.01), f"{p:.0%}\nn={n}", ha="center", va="bottom" if d >= 0 else "top", fontsize=7)
        lows.append(lo - base[0]); highs.append(hi - base[0])
nb = rate(NEUTRAL)
if nb[3]:
    ax.axhline(nb[0] - vb[0], color="#1f77b4", ls=":", lw=1, label=f"neutral line − vLLM baseline ({nb[0]:.0%}, n={nb[3]})")
rb = rate(RAND)
if rb[3] and not math.isnan(hb[0]):
    ax.axhline(rb[0] - hb[0], color="#d62728", ls=":", lw=1, label=f"random direction, matched norm − HF baseline ({rb[0]:.0%}, n={rb[3]})")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f.replace("_", "\n") for f in FACTORS])
ax.set_ylabel("Δ shortcut rate vs own-backend baseline")
ax.set_title(f"Prompting vs contrast-direction steering, per factor   "
             f"(vLLM baseline {vb[0]:.0%} n={vb[3]};  HF baseline {hb[0]:.0%} n={hb[3]})", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")
if lows:
    ax.set_ylim(min(lows) - 0.09, max(highs + [0.05]) + 0.16)
fig.tight_layout()
fig.savefig(figs / "prompt_vs_steer.png", dpi=160)
print("wrote", figs / "prompt_vs_steer.png")

# ---------- figure 2: geometry ----------
pl = run / "phase4" / "dp_plant_layers.json"
if pl.exists():
    J = json.loads(pl.read_text())
    pe, pts, vec_files = J.get("prompt_end", {}), J.get("points", []), J["vec_files"]
    # the vector actually used for steering: the base file (concept.npz) except disapproval_L17_refit
    used = {"tedium": "tedium", "desperate": "desperate", "shortcut": "shortcut", "completion_drive": "completion_drive",
            "disapproval": "disapproval_L17_refit"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    poles = [("plus", "plant: plus pole", "#d62728"), ("minus", "plant: minus pole", "#1f77b4"),
             ("instr", "instruction (treatment)", "#2ca02c"), ("neutral", "neutral line", "#7f7f7f")]
    w = 0.2
    for i, c in enumerate(FACTORS):
        stem = used[c]
        if c not in vec_files:
            continue
        for j, (pole, lab, col) in enumerate(poles):
            name = "neutral" if pole == "neutral" else f"{c}/{pole}"
            if name in pe and stem in pe[name]:
                axes[0].bar(i + (j - 1.5) * w, pe[name][stem]["proj_shift"], w, color=col, label=lab if i == 0 else None)
            if pts and pole != "neutral":
                cs = [p["concepts"][c][pole]["vecs"][stem]["cos"] for p in pts if c in p["concepts"]]
                if cs:
                    axes[1].bar(i + (j - 1.5) * w, np.mean(cs), w, color=col, label=lab if i == 0 else None,
                                yerr=np.std(cs) / math.sqrt(len(cs)), capsize=2, error_kw={"lw": 0.8})
    for ax, ttl, yl in ((axes[0], "Prompt end: shift along the fitted direction", "Δh · d̂  (residual units)"),
                        (axes[1], f"Decision token, {len(pts)} points: cos(Δh, d)", "cosine")):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(FACTORS)))
        ax.set_xticklabels([f.replace("_", "\n") for f in FACTORS], fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Does the line move the residual along the direction it is compared with? (own vector per concept)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "geometry.png", dpi=160)
    print("wrote", figs / "geometry.png")
else:
    print("skip geometry.png: no", pl)
