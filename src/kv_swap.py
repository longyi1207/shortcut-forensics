"""Stage 4 — content ablation of the instruction span in the FULL-ATTENTION
channel (WRITEUP §4.12; PIPELINE.md).

The text stays in the prompt (so the linear-attention/GDN recurrent channel and
the residual stream at the span positions still see it), but the K/V that the
full-attention layers store for the span positions are replaced by those of a
length-matched DONOR text (a neutral filler, or -- in the mirror condition --
the real instruction injected under filler text). Implemented as forward hooks
on each full-attention layer's k_proj / v_proj: Qwen3_5Attention applies
k_norm (per-head RMSNorm) and RoPE to the k_proj output *position-wise*, and
v_proj feeds the value directly, so overwriting the projection outputs at the
span positions is exactly equivalent to overwriting the cached keys/values
there (verified by scripts/kv_swap_check.py against DynamicCache tensors).

Mode "capture": run a prefill of prompt[: span_end+1] with the span token ids
replaced by the donor's -> store k_proj/v_proj outputs at the span positions.
Mode "swap": at every prefill that covers the span (each agent turn re-encodes
the whole context; decode steps have q_len 1 and are skipped), overwrite the
span positions with the stored donor outputs. Positions are identical across
turns because the span sits in the fixed first user turn.
"""
from __future__ import annotations

import torch

from src.attn_mask import token_span_for_substring
from src.hooks import get_layers

NEUTRAL_SEED = (
    "This section gives general background on the repository layout, the development workflow, and the "
    "conventions the project follows for organizing source files, tests, documentation, and configuration, "
    "together with notes about the tooling that is commonly used and where to find further details. "
)


class InstructionKVSwapper:
    def __init__(self, model, full_attention_layers: list[int]):
        self.model = model
        self.layers = list(full_attention_layers)
        self.span: list[int] = []
        self.mode = "off"  # off | capture | swap
        self.store: dict[tuple[int, str], torch.Tensor] = {}
        self._handles = []
        self.n_swapped = 0  # prefills on which the swap was applied (counted on k_proj of the first layer)

    def _make_hook(self, L: int, kind: str):
        def hook(module, args, out):
            if self.mode == "off" or not self.span:
                return None
            if out.dim() != 3 or out.shape[1] <= self.span[-1]:
                return None  # decode step / prefill that does not reach the span
            if self.mode == "capture":
                self.store[(L, kind)] = out[:, self.span, :].detach().clone()
                return None
            src = self.store.get((L, kind))
            if src is None:
                return None
            new = out.clone()
            new[:, self.span, :] = src.to(out.dtype)
            if kind == "k" and L == self.layers[0]:
                self.n_swapped += 1
            return new
        return hook

    def __enter__(self):
        layers = get_layers(self.model)
        for L in self.layers:
            attn = getattr(layers[L], "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer {L} has no self_attn")
            self._handles.append(attn.k_proj.register_forward_hook(self._make_hook(L, "k")))
            self._handles.append(attn.v_proj.register_forward_hook(self._make_hook(L, "v")))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    @torch.no_grad()
    def prepare(self, prompt_ids: torch.Tensor, span: list[int], donor_ids: list[int]) -> None:
        """Capture the donor's k/v at `span` (prefill of prompt[:span_end+1] with donor ids in the span), then arm swap mode."""
        if len(donor_ids) != len(span):
            raise ValueError(f"donor length {len(donor_ids)} != span length {len(span)}")
        self.span = [int(p) for p in span]
        ids = prompt_ids[:, : self.span[-1] + 1].clone()
        ids[0, self.span] = torch.tensor(donor_ids, device=ids.device, dtype=ids.dtype)
        self.store = {}
        self.mode = "capture"
        try:
            self.model(input_ids=ids, use_cache=True)
        finally:
            self.mode = "swap"
        missing = [(L, k) for L in self.layers for k in ("k", "v") if (L, k) not in self.store]
        if missing:
            raise RuntimeError(f"capture failed for {missing}")

    def set_mode(self, mode: str) -> None:
        self.mode = mode


def fit_filler(tokenizer, render, target_len: int, seed_text: str = NEUTRAL_SEED, max_tries: int = 400) -> str:
    """Find neutral filler text whose IN-CONTEXT token span (render(text) -> prompt) has exactly target_len tokens."""
    words = (seed_text * 6).split()
    best, best_n = None, -1
    for k in range(1, min(len(words), max_tries)):
        cand = " ".join(words[:k]).rstrip(".,") + "."
        n = len(token_span_for_substring(tokenizer, render(cand), cand))
        if n == target_len:
            return cand
        if n > target_len:
            break
        best, best_n = cand, n
    cand = best or "Note."
    for _ in range(40):  # pad one short token at a time
        cand = cand[:-1] + " and so on."
        n = len(token_span_for_substring(tokenizer, render(cand), cand))
        if n == target_len:
            return cand
        if n > target_len:
            cand = cand[:-1]  # drop the final '.' and retry with a different tail
            n = len(token_span_for_substring(tokenizer, render(cand), cand))
            if n == target_len:
                return cand
    raise RuntimeError(f"could not fit a filler of exactly {target_len} tokens (best {best_n})")
