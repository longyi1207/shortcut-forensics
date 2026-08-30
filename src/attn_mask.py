"""Instruction-readability control for the decision-token circuit program
(WRITEUP §4.12 Stage 2). Blocks attention edges from NEWLY GENERATED tokens to
chosen key positions (the instruction span, and/or the model's own earlier
assistant-turn "notes") in the FULL-ATTENTION layers only, via forward
pre-hooks on each Qwen3_5Attention module that rewrite the `attention_mask`
kwarg. Linear-attention (GDN) layers are untouched, so anything that survives
masking is carried by the recurrent channel (or by prefill-time notes).

Scope, deliberately: decode steps only (q_len == 1). Turn prefills re-encode
the whole ~100k-token context and a dense 4D mask there would be tens of GB;
leaving prefill unmasked also keeps the model's prefill-time "notes" (Li,
arXiv:2606.17107) intact, so this isolates DIRECT reading of the span at
generation time. Stage 4 (K/V substitution at prefill) handles the
"nobody can read it" case.

Mask semantics follow transformers' 4D float convention: 0 = attend,
finfo(dtype).min = blocked; shape [batch, 1, q_len, k_len]. If the incoming
mask is None (sdpa causal fast path) we build one; if it is a 4D tensor we
clone and add the block. k_len is taken from `cache_position` when present,
else from the cache's seen length + q_len.
"""
from __future__ import annotations

import torch

from src.hooks import get_layers


class InstructionMaskController:
    def __init__(self, model, full_attention_layers: list[int], heads_by_layer: dict[int, list[int]] | None = None, n_heads: int | None = None):
        """heads_by_layer: optional {layer: [query-head indices]} restricting the
        block to specific heads (Stage 3 edge ablation). None = all heads.
        n_heads must be given when heads_by_layer is used (query heads, e.g. 16)."""
        self.model = model
        self.layers = list(full_attention_layers)
        self.blocked: set[int] = set()
        self.active = False
        self._handles = []
        self.n_applied = 0  # decode steps on which a block was applied (diagnostic)
        self.heads_by_layer = {int(k): sorted(set(int(h) for h in v)) for k, v in (heads_by_layer or {}).items()}
        self.n_heads = n_heads
        if self.heads_by_layer and not self.n_heads:
            raise ValueError("n_heads required with heads_by_layer")

    # ---- span management ----------------------------------------------------
    def set_blocked_positions(self, positions) -> None:
        self.blocked = set(int(p) for p in positions)

    def set_active(self, flag: bool) -> None:
        self.active = bool(flag)

    # ---- hook -----------------------------------------------------------------
    def _pre_hook(self, module, args, kwargs):
        if not self.active or not self.blocked:
            return None
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None or hs.dim() != 3 or hs.shape[1] != 1:
            return None  # prefill or unexpected shape: leave untouched
        cache_position = kwargs.get("cache_position")
        pkv = kwargs.get("past_key_values")
        if cache_position is not None:
            k_len = int(cache_position[-1].item()) + 1
        elif pkv is not None:
            k_len = int(pkv.get_seq_length(getattr(module, "layer_idx", 0))) + 1
        else:
            return None
        idx = [p for p in self.blocked if 0 <= p < k_len]
        if not idx:
            return None
        heads = None
        if self.heads_by_layer:
            heads = self.heads_by_layer.get(int(getattr(module, "layer_idx", -1)))
            if not heads:
                return None  # this layer has no selected heads -> leave untouched
        neg = torch.finfo(hs.dtype).min
        mask = kwargs.get("attention_mask")
        H = self.n_heads if heads is not None else 1
        if mask is None:
            new = torch.zeros((hs.shape[0], H, 1, k_len), dtype=hs.dtype, device=hs.device)
        else:
            new = mask.clone().to(hs.dtype)
            if new.dim() != 4 or new.shape[-1] < k_len:
                return None
            if heads is not None and new.shape[1] == 1:
                new = new.expand(-1, H, -1, -1).clone()
        if heads is None:
            new[..., idx] = neg
        else:
            for h in heads:  # per-head edge block: only these query heads lose the span
                new[:, h, :, idx] = neg
        kwargs["attention_mask"] = new
        self.n_applied += 1
        return (args, kwargs)

    def __enter__(self):
        layers = get_layers(self.model)
        for L in self.layers:
            attn = getattr(layers[L], "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer {L} has no self_attn (not a full-attention layer?)")
            self._handles.append(attn.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def token_span_for_substring(tokenizer, text: str, sub: str, occurrence: str = "first") -> list[int]:
    """Token indices (in tokenizer(text, add_special_tokens=False)) covering `sub` in `text`."""
    start = text.find(sub) if occurrence == "first" else text.rfind(sub)
    if start < 0 or not sub:
        return []
    end = start + len(sub)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end]


def token_spans_for_substrings(tokenizer, text: str, subs: list[str]) -> list[int]:
    """Union of token indices covering each of `subs` (each matched at its first occurrence after the previous match)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    out: set[int] = set()
    cursor = 0
    for sub in subs:
        if not sub:
            continue
        start = text.find(sub, cursor)
        if start < 0:
            continue
        end = start + len(sub)
        cursor = end
        out.update(i for i, (a, b) in enumerate(offs) if b > start and a < end)
    return sorted(out)
