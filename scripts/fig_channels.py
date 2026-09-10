"""Figure: how the instruction's protection is carried through the context (judge-labelled cells from
WRITEUP §4.12 Stage 2, attention masking after every failed tool result)."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cells = [("no instruction", 8, 59, "#7f7f7f"), ("instruction intact", 2, 53, "#1f77b4"),
         ("mask attention to\ninstruction", 2, 20, "#2ca02c"), ("mask attention to\nown earlier notes", 3, 20, "#2ca02c"),
         ("mask both", 5, 20, "#d62728"), ("mask same-size span\nof tool output (control)", 2, 21, "#9467bd")]


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


fig, ax = plt.subplots(figsize=(9, 3.9))
for i, (name, k, n, col) in enumerate(cells):
    p = k / n; lo, hi = wilson(k, n)
    ax.bar(i, p, 0.62, color=col, alpha=0.9)
    ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="k", capsize=3, lw=1)
    ax.text(i, hi + 0.012, f"{k}/{n}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(cells))); ax.set_xticklabels([c[0] for c in cells], fontsize=8)
ax.set_ylabel("shortcut rate (judge-labelled)")
ax.set_ylim(0, 0.62)
ax.set_title("After a failed check, block the model's attention to ... (Qwen3.5-9B, HF, tedium instruction)", fontsize=10)
ax.axhline(8 / 59, color="#7f7f7f", ls=":", lw=1); ax.axhline(2 / 53, color="#1f77b4", ls=":", lw=1)
fig.tight_layout(); fig.savefig("writeup/figs/channels.png", dpi=160)
print("wrote writeup/figs/channels.png")
