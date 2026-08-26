"""Direction fitting + validation math. Pure numpy — no torch import at
module level, so this is testable on the laptop without a GPU/torch install.
save/load lazily import torch only to also emit the SPEC-mandated .pt file;
the .npz companion is what local (torch-free) code (Phase 7 plotting,
tests) reads back.

SPEC.md §5.1: mean-diff d = E[h+] - E[h-]. Validate before steer:
1. held-out pair accuracy >= 90% or drop the concept
2. cosine matrix, flag |cos| > 0.7 as "same horse"
3. lexical control (scrambled pairs should collapse accuracy)
4. compliance vs disapproval cosine check
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def mean_diff(plus_vecs: np.ndarray, minus_vecs: np.ndarray) -> np.ndarray:
    """d = E[h+] - E[h-]. plus_vecs/minus_vecs: [n, hidden_dim]."""
    plus_vecs = np.asarray(plus_vecs, dtype=np.float64)
    minus_vecs = np.asarray(minus_vecs, dtype=np.float64)
    return plus_vecs.mean(axis=0) - minus_vecs.mean(axis=0)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a ~zero vector")
    return v / n


def pair_accuracy(d: np.ndarray, plus_vecs: np.ndarray, minus_vecs: np.ndarray) -> float:
    """Fraction of held-out pairs where h+ · d > h- · d (ranks + above -)."""
    plus_vecs = np.asarray(plus_vecs, dtype=np.float64)
    minus_vecs = np.asarray(minus_vecs, dtype=np.float64)
    plus_scores = plus_vecs @ d
    minus_scores = minus_vecs @ d
    return float(np.mean(plus_scores > minus_scores))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = sorted(vectors.keys())
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = cosine(vectors[a], vectors[b])
    return names, mat


def fit_direction_sweep(
    train_plus_by_layer: dict[int, np.ndarray],
    train_minus_by_layer: dict[int, np.ndarray],
    val_plus_by_layer: dict[int, np.ndarray],
    val_minus_by_layer: dict[int, np.ndarray],
) -> dict:
    """Fit mean-diff at each candidate layer, pick the layer with the best
    held-out val accuracy. Returns {layer, d, val_acc, per_layer_val_acc}."""
    best = None
    per_layer_acc = {}
    for layer, tp in train_plus_by_layer.items():
        tm = train_minus_by_layer[layer]
        d = mean_diff(tp, tm)
        acc = pair_accuracy(d, val_plus_by_layer[layer], val_minus_by_layer[layer])
        per_layer_acc[layer] = acc
        if best is None or acc > best["val_acc"]:
            best = {"layer": layer, "d": d, "val_acc": acc}
    best["per_layer_val_acc"] = per_layer_acc
    return best


def signed_auc(scores_shortcut: np.ndarray, scores_honest: np.ndarray, sign: str) -> dict:
    """AUC of shortcut vs honest projections. SPEC.md §5.3: report with the
    pre-registered sign, but keep the raw number too — a pro_honest concept
    scoring LOW on shortcuts (raw AUC < 0.5) is still informative (feature
    was "off" during natural cheats), not a failure to be hidden by flipping.

    raw_auc: standard AUC, shortcut = positive class.
    hypothesis_auc: raw_auc for pro_cheat/unknown; 1-raw_auc for pro_honest
        (i.e. "does this vector score honest > shortcut as hypothesized").
    """
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(len(scores_shortcut)), np.zeros(len(scores_honest))])
    scores = np.concatenate([scores_shortcut, scores_honest])
    raw_auc = float(roc_auc_score(y, scores))
    hyp_auc = (1.0 - raw_auc) if sign == "pro_honest" else raw_auc
    return {"raw_auc": raw_auc, "hypothesis_auc": hyp_auc, "sign": sign}


def save_vector(
    path: Path,
    d: np.ndarray,
    layer: int,
    pair_ids: list[str],
    val_acc: float,
    meta: dict | None = None,
) -> None:
    """Write vectors/{concept}.npz (always) and .pt (if torch available)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta or {}
    np.savez(
        path.with_suffix(".npz"),
        d=d,
        layer=layer,
        val_acc=val_acc,
        pair_ids=np.array(pair_ids, dtype=object),
        meta=np.array(json.dumps(meta)),
    )
    try:
        import torch

        torch.save(
            {"d": torch.tensor(d, dtype=torch.float32), "layer": layer, "pair_ids": pair_ids, "val_acc": val_acc, "meta": meta},
            path.with_suffix(".pt"),
        )
    except ImportError:
        logger.info("torch not available locally; wrote .npz only for %s (write .pt on the VM)", path.name)


def load_vector(path: Path) -> dict:
    """Prefer .npz (torch-free); fall back to .pt if only that exists."""
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        return {
            "d": data["d"],
            "layer": int(data["layer"]),
            "val_acc": float(data["val_acc"]),
            "pair_ids": list(data["pair_ids"]),
            "meta": json.loads(str(data["meta"])),
        }
    pt_path = path.with_suffix(".pt")
    if pt_path.exists():
        import torch

        obj = torch.load(pt_path, map_location="cpu", weights_only=False)  # our own file, not a 3rd-party checkpoint
        return {
            "d": obj["d"].numpy(),
            "layer": obj["layer"],
            "val_acc": obj["val_acc"],
            "pair_ids": obj["pair_ids"],
            "meta": obj.get("meta", {}),
        }
    raise FileNotFoundError(f"no vector at {path} (.npz or .pt)")


def load_all_vectors(vectors_dir: Path) -> dict[str, dict]:
    vectors_dir = Path(vectors_dir)
    out = {}
    for npz in sorted(vectors_dir.glob("*.npz")):
        concept = npz.stem
        out[concept] = load_vector(vectors_dir / concept)
    return out
