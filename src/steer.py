"""Ablation / addition steering hooks. GPU-VM only (torch). SPEC.md §5.4.

ablate: project the component along d_hat out of the residual stream at
every token, every layer in the target set — h' = h - (h . d_hat) d_hat.
add (+alpha): h' = h + alpha * d, using d's own mean-diff magnitude (Turner
ActAdd convention: alpha=1.0 means "one mean-diff's worth"), so the SPEC's
"no strength sweep, alpha=1.0" default has a concrete, non-arbitrary scale.

Hooks are persistent across an entire multi-turn agent rollout (register
once before the tool loop, remove once after) per SPEC.md §5.4.
"""

from __future__ import annotations

import numpy as np
import torch

from src.hooks import get_layers


class SteeringSession:
    """Context manager: register ablate/add hooks on `layer_indices` for the
    duration of a rollout. mode: "ablate" | "add" | "identity"."""

    def __init__(
        self,
        model,
        layer_indices: list[int],
        d: np.ndarray | None,
        mode: str,
        alpha: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.layer_indices = layer_indices
        self.mode = mode
        self.alpha = alpha
        self.handles = []
        if mode == "identity" or d is None:
            self.d_hat = None
            self.d_raw = None
        else:
            d = np.asarray(d, dtype=np.float32)
            norm = np.linalg.norm(d)
            if norm < 1e-12:
                raise ValueError("cannot steer with a ~zero direction vector")
            self.d_hat = torch.tensor(d / norm, dtype=torch.float32, device=device)
            self.d_raw = torch.tensor(d, dtype=torch.float32, device=device)

    def _make_hook(self):
        mode, alpha, d_hat, d_raw = self.mode, self.alpha, self.d_hat, self.d_raw

        def hook_fn(module, inp, output):
            is_tuple = isinstance(output, tuple)
            hs = output[0] if is_tuple else output
            dh = d_hat.to(hs.device, dtype=hs.dtype)
            if mode == "ablate":
                proj = (hs @ dh).unsqueeze(-1) * dh  # [..., seq, 1] * [dim] -> broadcast
                hs = hs - proj
            elif mode == "add":
                dr = d_raw.to(hs.device, dtype=hs.dtype)
                hs = hs + alpha * dr
            else:
                return output
            return (hs,) + output[1:] if is_tuple else hs

        return hook_fn

    def __enter__(self):
        if self.mode != "identity":
            layers = get_layers(self.model)
            hook_fn = self._make_hook()
            for idx in self.layer_indices:
                self.handles.append(layers[idx].register_forward_hook(hook_fn))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def random_unit_vector(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)
