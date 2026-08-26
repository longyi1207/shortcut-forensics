"""Residual-stream capture via PyTorch forward hooks. GPU-VM only (torch
required). Pattern lifted from code/emotion_vectors/src/activation_extraction.py
but capturing the LAST token (not a mean over a window) per SPEC.md §5.1.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_layers(model):
    """Return the list of decoder layers for any causal LM (Qwen included:
    Qwen*ForCausalLM exposes model.model.layers like Llama)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot find decoder layers on {type(model).__name__}")


def resolve_layer_band(model, frac_lo: float, frac_hi: float) -> list[int]:
    """Integer layer indices whose depth fraction falls in [frac_lo, frac_hi]."""
    n = len(get_layers(model))
    lo = max(0, int(round(n * frac_lo)))
    hi = min(n - 1, int(round(n * frac_hi)))
    if hi < lo:
        hi = lo
    return list(range(lo, hi + 1))


def extract_hidden_last_token(
    model,
    tokenizer,
    text: str,
    layer_indices: list[int],
    device: str = "cuda",
    max_length: int = 512,
) -> dict[int, np.ndarray]:
    """Single forward pass; return {layer_idx: hidden_dim ndarray} at the
    last non-pad token. Used for contrast-pair fitting (Phase 4)."""
    layers = get_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook_fn(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach()

        return hook_fn

    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for idx in layer_indices:
        out[idx] = captured[idx][0, -1].float().cpu().numpy()
    return out


def extract_hidden_last_token_batch(
    model,
    tokenizer,
    texts: list[str],
    layer_indices: list[int],
    device: str = "cuda",
    max_length: int = 512,
    show_progress: bool = True,
) -> dict[int, np.ndarray]:
    """Sequential last-token extraction over many short texts (contrast
    pairs). Returns {layer_idx: [n_texts, hidden_dim] ndarray}."""
    from tqdm import tqdm

    per_layer: dict[int, list[np.ndarray]] = {idx: [] for idx in layer_indices}
    it = tqdm(texts, desc="  extracting activations", leave=False) if show_progress else texts
    for text in it:
        single = extract_hidden_last_token(model, tokenizer, text, layer_indices, device=device, max_length=max_length)
        for idx in layer_indices:
            per_layer[idx].append(single[idx])
    return {idx: np.stack(vecs) for idx, vecs in per_layer.items()}


def capture_activations_at_positions(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    positions: dict[str, int],
    layer_indices: list[int],
    device: str = "cuda",
) -> dict[int, dict[str, np.ndarray]]:
    """Single teacher-forced forward pass over a full (already-generated)
    conversation's token ids. Returns {layer_idx: {position_name: vector}}.

    Used for natural-rollout activation dumps (Phase 2): we already have the
    full transcript's token ids from generation, so one forward pass with
    hooks is simpler and more robust than instrumenting incremental decode.
    """
    layers = get_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook_fn(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach()

        return hook_fn

    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    try:
        ids = input_ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    seq_len = ids.shape[1]
    out: dict[int, dict[str, np.ndarray]] = {idx: {} for idx in layer_indices}
    for name, pos in positions.items():
        if pos < 0 or pos >= seq_len:
            logger.warning("position %s=%d out of range (seq_len=%d), skipping", name, pos, seq_len)
            continue
        for idx in layer_indices:
            out[idx][name] = captured[idx][0, pos].float().cpu().numpy()
    return out
