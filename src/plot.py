"""Plots for Phase 7 (type-split + figures) and Phase 4/5 (cosine, AUC).
Local/torch-free — matplotlib + numpy only, per SPEC.md phase table
("Phase 7 | local or GPU").
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_cosine_heatmap(names: list[str], mat: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 1.1 * len(names) + 2))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            weight = "bold" if abs(v) > 0.7 and i != j else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontweight=weight, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    ax.set_title("Direction cosine similarity (|cos| > 0.7 flagged bold = \"same horse\")")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_signed_auc(rows: list[dict], out_path: Path) -> None:
    """rows: [{"concept":..., "hypothesis_auc":..., "raw_auc":..., "sign":...}]"""
    names = [r["concept"] for r in rows]
    hyp = [r["hypothesis_auc"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4))
    colors = ["#2b6cb0" if r["sign"] == "pro_cheat" else ("#c53030" if r["sign"] == "pro_honest" else "#718096") for r in rows]
    ax.bar(names, hyp, color=colors)
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("hypothesis-signed AUC (shortcut vs honest)")
    ax.set_title("Phase 5 readout: does each direction predict shortcuts as pre-registered?")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rate_by_condition(rows: list[dict], out_path: Path) -> None:
    """rows: [{"condition":..., "shortcut_rate":..., "n":..., "capability_ok_rate":...}]"""
    names = [r["condition"] for r in rows]
    rates = [r["shortcut_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 4.5))
    bars = ax.bar(names, rates, color="#c53030")
    for bar, r in zip(bars, rows):
        ax.annotate(f"n={r['n']}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("shortcut rate")
    ax.set_title("Phase 6: shortcut rate by intervention condition")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_workaround_type_split(rows: list[dict], out_path: Path) -> None:
    """rows: [{"condition":..., "counts": {workaround_type: n, ...}}]"""
    all_types = sorted({t for r in rows for t in r["counts"]})
    conditions = [r["condition"] for r in rows]
    bottoms = np.zeros(len(conditions))
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(conditions)), 5))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(all_types):
        vals = np.array([r["counts"].get(t, 0) for r in rows], dtype=float)
        ax.bar(conditions, vals, bottom=bottoms, label=t, color=cmap(i % 10))
        bottoms += vals
    ax.set_ylabel("count")
    ax.set_title("Phase 7: workaround type by condition")
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
