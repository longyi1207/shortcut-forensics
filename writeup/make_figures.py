"""Figures for the MATS write-up. Direct-labelled (the palette's contrast check
obligates visible labels), recessive axes, one scale per panel."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#d5d4d0", "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def barh_rates(ax, labels, ks, ns, colors, title, note=None):
    ys = np.arange(len(labels))[::-1]
    for y, lab, k, n, c in zip(ys, labels, ks, ns, colors):
        p, lo, hi = wilson(k, n)
        ax.barh(y, p, height=0.55, color=c, zorder=3)
        ax.plot([lo, hi], [y, y], color=INK2, lw=1.4, zorder=4)
        ax.text(hi + 0.015, y, f"{p:.0%}  ({k}/{n})", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, 0.85); ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("shortcut rate  (95% Wilson CI)")
    ax.grid(axis="x", color="#eceae6", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, weight="bold", loc="left", pad=10)
    if note:
        ax.text(0, -0.30, note, transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")


# ---- Fig 1: prompt and steering decouple -------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 3.5))
barh_rates(ax,
    ["no instruction", "prompt", "steer tedium (no prompt)", "prompt + steer tedium"],
    [8, 2, 17, 15], [59, 53, 32, 30],
    [INK2, BLUE, ORANGE, ORANGE],
    "The prompt eliminates the shortcut — yet offers zero resistance\nto steering the same concept it is about",
    "Prompt + steer is indistinguishable from steer alone (p = 1.0) and far from\nprompt alone (p < 0.001). At half dose: 10/21 vs 10/21, p = 1.0 — not saturation.")
plt.tight_layout(); plt.savefig("fig1_decoupling.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 2: sentence level ----------------------------------------------------
d = json.load(open("data/dp_sentences.json"))
order = ["full", "S4", "S5", "S2", "S3", "S1+S2", "S1"]
names = {"full": "full instruction (5 sentences, 70 tok)",
         "S4": 'S4  "Do not cut corners. Do not rush to finish."  (11 tok)',
         "S5": 'S5  "Boredom is not a reason to do less…"',
         "S2": 'S2  "Do NOT let that affect your work."',
         "S3": 'S3  "Treat every step with full care…"',
         "S1+S2": "S1 + S2  (antecedent restored)",
         "S1": 'S1  "This task may feel repetitive or tedious."'}
fig, ax = plt.subplots(figsize=(8.4, 3.9))
ys = np.arange(len(order))[::-1]
for y, k in zip(ys, order):
    v = np.array([x["s"][k] - x["s"]["none"] for x in d])
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    c = BLUE if k == "full" else (AQUA if k in ("S4", "S5") else (INK2 if m < 0.2 else YELLOW))
    ax.barh(y, m, height=0.55, color=c, zorder=3)
    ax.plot([m - 1.96 * se, m + 1.96 * se], [y, y], color=INK2, lw=1.4, zorder=4)
    ax.text(m + 1.96 * se + 0.02, y, f"{m:+.2f}", va="center", fontsize=9, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([names[k] for k in order], fontsize=9)
ax.axvline(0, color="#d5d4d0", lw=0.9)
ax.set_xlabel("shift toward engaging with the error rather than replanning\n(log-prob difference vs. no instruction, 40 paired decision points)")
ax.grid(axis="x", color="#eceae6", lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.set_title("One clause of ~11 tokens carries 78% of the effect,\nand naming the feeling cancels the sentence after it",
             fontsize=11.5, weight="bold", loc="left", pad=10)
ax.text(0, -0.46, "S2 alone +0.53 vs S1+S2 +0.14: restoring the antecedent does not\nrescue S2, it destroys it. Bars are mean +- 95% CI over paired points.",
        transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
plt.tight_layout(); plt.savefig("fig2_sentences.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 3: where the instruction is written ----------------------------------
dc = json.load(open("data/dp_components.json"))
keys = sorted(set.intersection(*[set(x["c"]) for x in dc]))
spec = np.array([[x["c"][k][0] - x["c"][k][1] for k in keys] for x in dc])
m = spec.mean(0); se = spec.std(0, ddof=1) / np.sqrt(len(spec))
tot = {t: m[[i for i, k in enumerate(keys) if k.split(".")[0] == t]].sum() for t in ("head", "mlp", "gdn")}
top = np.argsort(-m)[:8]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1, 1.35]})
tl = ["128 attention heads", "32 MLPs", "24 GatedDeltaNet blocks"]
tv = [tot["head"], tot["mlp"], tot["gdn"]]
ys = np.arange(3)[::-1]
for y, lab, v, c in zip(ys, tl, tv, [ORANGE, YELLOW, BLUE]):
    a1.barh(y, v, height=0.5, color=c, zorder=3)
    a1.text(v + 0.06 if v > 0 else v - 0.06, y, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", fontsize=9, color=INK)
a1.set_yticks(ys); a1.set_yticklabels(tl, fontsize=9)
a1.axvline(0, color="#d5d4d0", lw=0.9); a1.set_xlim(-1.6, 2.0)
a1.grid(axis="x", color="#eceae6", lw=0.8, zorder=0); a1.set_axisbelow(True)
a1.set_xlabel("summed attribution")
a1.set_title("By component type", fontsize=10.5, weight="bold", loc="left", pad=8)
ys2 = np.arange(len(top))[::-1]
for y, i in zip(ys2, top):
    c = BLUE if keys[i].startswith("gdn") else (ORANGE if keys[i].startswith("head") else YELLOW)
    a2.barh(y, m[i], height=0.55, color=c, zorder=3)
    a2.plot([m[i] - 1.96 * se[i], m[i] + 1.96 * se[i]], [y, y], color=INK2, lw=1.4, zorder=4)
    a2.text(m[i] + 1.96 * se[i] + 0.03, y, f"{m[i]:+.2f}", va="center", fontsize=9, color=INK)
a2.set_yticks(ys2); a2.set_yticklabels([keys[i] for i in top], fontsize=9)
a2.grid(axis="x", color="#eceae6", lw=0.8, zorder=0); a2.set_axisbelow(True)
a2.set_xlim(0, 1.75); a2.set_xlabel("attribution (share of the instruction's decision-time effect)")
a2.set_title("Top individual components — all in layers 0–15", fontsize=10.5, weight="bold", loc="left", pad=8)
fig.suptitle("Ablating each of 184 components at the instruction's tokens:\nthe write happens at the bottom of the network",
             fontsize=11.5, weight="bold", x=0.005, ha="left", y=1.06)
plt.tight_layout(); plt.savefig("fig3_components.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 4: the channel question ---------------------------------------------
g = json.load(open("data/dp_gdn.json"))
full = np.array([x["sA"] - x["sNone"] for x in g])
art = np.array([x["sNull"] - x["sA"] for x in g])
gdn = np.array([x["sNull"] - x["sSwap"] for x in g])
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11.0, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
fig.suptitle("Neither channel carries the instruction on its own",
             fontsize=11.5, weight="bold", x=0.005, ha="left", y=1.04)
barh_rates(b1, ["prompt (both channels real)", "attention content → filler", "filler text + attention content injected", "filler text (neither)"],
           [2, 4, 9, 5], [32, 43, 43, 24], [BLUE, AQUA, ORANGE, INK2],
           "Attention channel")
b1.text(0, -0.36, "Injecting the instruction's entire attention content (21%) is\nindistinguishable from its text-only control (21%), p = 1.000.",
        transform=b1.transAxes, fontsize=8.5, color=INK2, va="top")
lbl = ["instruction's full effect", "chunked-prefill artefact\n(null swap — the control)", "recurrent channel\n(state → filler)"]
vals = [full, art, gdn]
ys = np.arange(3)[::-1]
for y, lab, v, c in zip(ys, lbl, vals, [BLUE, INK2, AQUA]):
    mm, ss = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    b2.barh(y, mm, height=0.5, color=c, zorder=3)
    b2.plot([mm - 1.96 * ss, mm + 1.96 * ss], [y, y], color=INK2, lw=1.4, zorder=4)
    b2.text(mm + 1.96 * ss + 0.03, y, f"{mm:+.3f}", va="center", fontsize=9, color=INK)
b2.set_yticks(ys); b2.set_yticklabels(lbl, fontsize=9)
b2.axvline(0, color="#d5d4d0", lw=0.9); b2.set_xlim(-0.15, 1.35)
b2.grid(axis="x", color="#eceae6", lw=0.8, zorder=0); b2.set_axisbelow(True)
b2.set_xlabel("log-prob shift (30 paired decision points)")
b2.set_title("Recurrent channel", fontsize=10.5, weight="bold", loc="left", pad=8)
b2.text(0, -0.36, "Removing the instruction's contribution to the recurrent state costs\n~10%. The artefact that voided the behavioural version: p = 0.19.",
        transform=b2.transAxes, fontsize=8.5, color=INK2, va="top")
plt.tight_layout(); plt.savefig("fig4_channels.png", dpi=200, bbox_inches="tight"); plt.close()
print("wrote fig1_decoupling.png fig2_sentences.png fig3_components.png fig4_channels.png")
