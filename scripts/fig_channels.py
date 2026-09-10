"""Figure: how the instruction's protection is carried through the context (judge-labelled cells from
WRITEUP §4.12 Stage 2, attention masking after every failed tool result)."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cells = [("no instruction\nin the prompt", 8, 59, "#7f7f7f"), ("instruction present,\nnothing blocked", 2, 53, "#1f77b4"),
         ("agent cannot read\nthe instruction", 2, 20, "#2ca02c"), ("agent cannot read\nits own earlier notes", 3, 20, "#2ca02c"),
         ("agent cannot read\neither of them", 5, 20, "#d62728"), ("control: cannot read a\nsame-size chunk of\ntool output", 2, 21, "#9467bd")]


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


fig, ax = plt.subplots(figsize=(10.5, 4.4))
for i, (name, k, n, col) in enumerate(cells):
    p = k / n; lo, hi = wilson(k, n)
    ax.bar(i, p, 0.62, color=col, alpha=0.9)
    ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="k", capsize=3, lw=1)
    ax.text(i, hi + 0.012, f"{k}/{n}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(cells))); ax.set_xticklabels([c[0] for c in cells], fontsize=8)
ax.set_ylabel("shortcut rate (judge-labelled)")
ax.set_ylim(0, 0.62)
ax.set_title("Tedium instruction in the prompt. In every turn after a failed check, the agent's attention to one part of the context is blocked.", fontsize=9.5)
ax.axhline(8 / 59, color="#7f7f7f", ls=":", lw=1); ax.axhline(2 / 53, color="#1f77b4", ls=":", lw=1)
ax.text(-0.42, 8 / 59 + 0.008, "no-instruction rate", fontsize=7, color="#7f7f7f", ha="left")
ax.text(1.62, 2 / 53 + 0.008, "instruction-intact rate", fontsize=7, color="#1f77b4", ha="left")
fig.tight_layout(); fig.savefig("writeup/figs/channels.png", dpi=160)
print("wrote writeup/figs/channels.png")
