"""Cartoon of the two interventions and the three comparisons (executive summary figure)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

def box(x, y, w, h, text, fc="#f4f4f4", ec="#999", fs=8.2, weight="normal", color="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True, color=color, weight=weight)

def arrow(x0, y0, x1, y1, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#444"))
    if text:
        ax.text((x0 + x1) / 2 + 0.08, (y0 + y1) / 2, text, fontsize=7.5, color="#444", ha="left", va="center")

# column A: contrast direction
ax.text(2.0, 5.95, "A. Contrast direction (steering)", ha="center", fontsize=10.5, weight="bold")
box(0.25, 4.3, 3.5, 1.4, "plus: \"I spent the afternoon renaming twenty\nnear-identical spreadsheet tabs... It felt like a\ndull round of housekeeping.\"\n\nminus: same sentence, ending \"...It was a quick\ncleanup before the meeting.\"", fc="#eaf3ea", ec="#7a9a7a", fs=7.2)
ax.text(2.0, 4.12, "160 LLM-written pairs, only the tedium content differs", ha="center", fontsize=7.0, color="#555")
arrow(2.0, 4.0, 2.0, 3.85)
box(0.25, 3.1, 3.5, 0.72, "Qwen3.5-9B, layer 19, last-token residual\nd = mean over pairs of  h(plus) − h(minus)", fs=8)
arrow(2.0, 3.05, 2.0, 2.6)
box(0.25, 1.75, 3.5, 0.82, "Steer: at every position of layer 19,\nremove the component along d (ablate)\nor add d at its fitted norm (push)", fc="#fbeaea", ec="#b07070", fs=8)
arrow(2.0, 1.7, 2.0, 1.25)
box(0.25, 0.5, 3.5, 0.72, "80-turn agentic rollout, real git and mypy\nLLM judge: shortcut or not", fs=8)

# column B: prompt
ax.text(6.0, 5.95, "B. Instruction in the prompt", ha="center", fontsize=10.5, weight="bold")
box(4.25, 4.3, 3.5, 1.4, "Task prompt: fix the 258 mypy errors, add a\npre-commit hook that runs mypy, commit.\n\n+ one line: \"IMPORTANT: This task may feel\nrepetitive or tedious. Do NOT let that affect your\nwork. ... Do not cut corners. Do not rush to finish. ...\"", fc="#eaf0fb", ec="#7088b0", fs=7.2)
ax.text(6.0, 4.07, "one line per factor, same length and register;\ncontrol: a same-length IMPORTANT line about shell formatting", ha="center", fontsize=7.0, color="#555")
arrow(6.0, 3.92, 6.0, 3.85)
box(4.25, 3.1, 3.5, 0.72, "No change to the model's activations\nby hand: the line is just text in context", fs=8)
arrow(6.0, 3.05, 6.0, 1.25)
box(4.25, 0.5, 3.5, 0.72, "80-turn agentic rollout, real git and mypy\nLLM judge: shortcut or not", fs=8)

# column C: comparisons
ax.text(10.0, 5.95, "C. Three comparisons per factor", ha="center", fontsize=10.5, weight="bold")
box(8.25, 4.55, 3.5, 1.15, "1. Shortcut rate under the instruction (B)\nvs the same-night baseline\n\n2. Shortcut rate under steering (A)\nvs the HF baseline and a random direction", fc="#fff6e0", ec="#c0a060", fs=8)
box(8.25, 2.55, 3.5, 1.75, "3. Does the instruction move the residual along d?\nΔh = h(prompt with line) − h(prompt without)\nprojected on d, at the first assistant token\nand at the decision token 10 to 40 turns later.\n\nPositive control: the concept's own sentence\n(\"honestly this is an endless, monotonous slog...\")\nappended in the same place must move along d.", fc="#fff6e0", ec="#c0a060", fs=7.6)
box(8.25, 0.5, 3.5, 1.7, "Same lever?\nIf A and B act on the same quantity, the instruction\nshould shift Δh along d, steering d should reproduce\nthe instruction's effect, and a probe on d should see\nthe instructed agent change.", fc="#f4f4f4", ec="#999", fs=7.8)

fig.savefig("writeup/figs/schematic.png", dpi=170, bbox_inches="tight")
print("wrote writeup/figs/schematic.png")
