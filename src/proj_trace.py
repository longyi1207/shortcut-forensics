"""Generation-time projection tracer for B5 "Prompt Decay" (app/prompt_decay
spec; WRITEUP §4.12). Records, at EVERY forward call during a rollout, the
residual-stream projection of the last sequence position onto a set of named
direction vectors, at one or more layers -- i.e. a per-token time series of
"how much of concept v is in the stream right now", for the whole 80-turn
rollout, with no post-hoc forward pass and no 32k-token capture ceiling.

Why this instead of capture_activations_at_positions: that path re-runs one
teacher-forced pass over the finished ~100k-token conversation and stores
full vectors at a handful of positions, capped at 32k tokens for memory. Here
we piggy-back on generation itself: the prefill forward of each turn exposes
the turn-start (prompt-final) position, and every decode step exposes the
newly generated token (seq_len == 1 with KV cache). We store only scalar
projections, so an 80-turn rollout costs ~ (tokens x directions x layers)
floats, not gigabytes of activations.

Hook ordering: PyTorch runs forward hooks in registration order and chains
their return values, so if a SteeringSession is entered BEFORE this tracer on
the same layer, the tracer sees the steered residual (the thing we want for
the steer / prompt+steer arms). Enter the tracer LAST.

Prefill detection: hs.shape[1] > 1. Decode: hs.shape[1] == 1. Multi-chunk
prefill is not used by HF generate here (single prompt), so one prefill call
per turn is assumed; the turn index must be set by the caller via set_turn()
right before each generate (run_rollout's on_turn_start hook does this).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.hooks import get_layers


@dataclass
class TraceBuffer:
    layer: int
    names: list[str]
    turn: list[int] = field(default_factory=list)
    step: list[int] = field(default_factory=list)  # 0 = prefill (turn-start position), k>=1 = k-th generated token
    seq_len: list[int] = field(default_factory=list)
    proj: dict[str, list[float]] = field(default_factory=dict)
    norm: list[float] = field(default_factory=list)  # ||h|| at that position, for scale-normalised views
    full_idx: list[int] = field(default_factory=list)  # event indices at which a full residual vector was stored
    full_vecs: list[np.ndarray] = field(default_factory=list)  # float16 [d_model] residuals at those events

    def __post_init__(self):
        self.proj = {n: [] for n in self.names}


class ProjectionTracer:
    """Context manager. directions: {layer: {name: np.ndarray[d_model]}}.
    Vectors are used as given (NOT re-normalised) so callers can pass unit
    vectors for cosine-like readings or raw mean-diff vectors to match the
    project's existing `h @ vec` readout convention.

    store_full_every: if > 0, ALSO store the full residual vector (float16) at
    every prefill/turn-start event and at every store_full_every-th event
    overall, so a later whole-residual / SAE-wide comparison between
    conditions is possible without re-running rollouts (WRITEUP §4.12,
    "what does the prompt move, if not tedium"). ~5 MB per layer per
    80-turn rollout at store_full_every=50."""

    def __init__(self, model, directions: dict[int, dict[str, np.ndarray]], device: str = "cuda", store_full_every: int = 0):
        self.model = model
        self.device = device
        self.store_full_every = int(store_full_every)
        self.layers = sorted(directions)
        self._dirs = {
            layer: {name: torch.tensor(np.asarray(v, dtype=np.float32), device=device) for name, v in dmap.items()}
            for layer, dmap in directions.items()
        }
        self.buffers = {layer: TraceBuffer(layer, list(dmap.keys())) for layer, dmap in directions.items()}
        self._turn = -1
        self._step_in_turn: dict[int, int] = {}
        self._handles = []
        self.paused = False

    def set_turn(self, turn: int) -> None:
        self._turn = turn
        for layer in self.layers:
            self._step_in_turn[layer] = 0

    def _make_hook(self, layer: int):
        buf = self.buffers[layer]
        dirs = self._dirs[layer]

        def hook_fn(module, inp, output):
            if self.paused:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() != 3:
                return None
            seq_len = int(hs.shape[1])
            h = hs[0, -1].float()
            step = 0 if seq_len > 1 else self._step_in_turn.get(layer, 0) + 1
            self._step_in_turn[layer] = step
            event_idx = len(buf.turn)
            if self.store_full_every > 0 and (step == 0 or event_idx % self.store_full_every == 0):
                buf.full_idx.append(event_idx)
                buf.full_vecs.append(h.to(torch.float16).cpu().numpy())
            buf.turn.append(self._turn)
            buf.step.append(step)
            buf.seq_len.append(seq_len)
            buf.norm.append(float(torch.linalg.vector_norm(h).item()))
            for name, d in dirs.items():
                buf.proj[name].append(float((h @ d).item()))
            return None  # never modify the output

        return hook_fn

    def __enter__(self):
        layers = get_layers(self.model)
        for layer in self.layers:
            self._handles.append(layers[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for layer, buf in self.buffers.items():
            p = f"L{layer}_"
            out[p + "turn"] = np.asarray(buf.turn, dtype=np.int32)
            out[p + "step"] = np.asarray(buf.step, dtype=np.int32)
            out[p + "seq_len"] = np.asarray(buf.seq_len, dtype=np.int32)
            out[p + "norm"] = np.asarray(buf.norm, dtype=np.float32)
            for name, vals in buf.proj.items():
                out[p + "proj_" + name] = np.asarray(vals, dtype=np.float32)
            if self.store_full_every > 0:
                out[p + "full_idx"] = np.asarray(buf.full_idx, dtype=np.int32)
                out[p + "full_vecs"] = (np.stack(buf.full_vecs).astype(np.float16) if buf.full_vecs
                                        else np.zeros((0, 0), dtype=np.float16))
        return out

    def summary(self) -> dict:
        s = {}
        for layer, buf in self.buffers.items():
            s[layer] = {"n_events": len(buf.turn), "n_turns": len(set(buf.turn)),
                        "mean_proj": {n: (float(np.mean(v)) if v else None) for n, v in buf.proj.items()}}
        return s


# Tag codes for decision-token capture (stored as int8 in the npz).
DT_TAGS = {"turn_start": 0, "commit_cmd": 1, "post_fail_first20": 2, "other_cmd": 3, "every50": 4, "post_fail_start": 5}


class DecisionTracer(ProjectionTracer):
    """ProjectionTracer plus event-locked FULL residual capture at DECISION
    tokens (WRITEUP §4.12, Stage 1 of the decision-token circuit program).

    Every decode step's last-position residual at `decision_layers` is
    buffered on CPU (float16) for the current turn. When the runner has seen
    the turn's generated text it calls commit_turn(keep) with
    {tag: [step indices]} (step 0 = the turn-start prefill position, step k =
    k-th generated token); only those vectors are kept, tagged. Buffer cost is
    ~ n_layers x tokens_this_turn x d_model x 2 bytes on CPU (tens of MB), so
    it is reset every turn. Unknown tags raise. Vectors that are selected under
    several tags are stored once per tag (simplest for analysis).
    """

    def __init__(self, model, directions, device: str = "cuda", store_full_every: int = 0, decision_layers=None):
        self.decision_layers = sorted(set(decision_layers or []))
        # make sure hooks exist on every decision layer even if it has no projection directions
        dirs = {int(k): dict(v) for k, v in directions.items()}
        for L in self.decision_layers:
            dirs.setdefault(L, {})
        super().__init__(model, dirs, device=device, store_full_every=store_full_every)
        self._turn_buf: dict[int, list[np.ndarray]] = {L: [] for L in self.decision_layers}
        self.kept: dict[int, dict[str, list]] = {L: {"turn": [], "step": [], "tag": [], "vecs": []} for L in self.decision_layers}

    def set_turn(self, turn: int) -> None:
        super().set_turn(turn)
        for L in self.decision_layers:
            self._turn_buf[L] = []

    def _make_hook(self, layer: int):
        base_hook = super()._make_hook(layer)
        if layer not in self.decision_layers:
            return base_hook
        buf = self._turn_buf

        def hook_fn(module, inp, output):
            base_hook(module, inp, output)
            if self.paused:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() == 3:
                buf[layer].append(hs[0, -1].to(torch.float16).cpu().numpy())
            return None

        return hook_fn

    def commit_turn(self, keep: dict[str, list[int]]) -> dict[str, int]:
        """Keep tagged steps of the current turn. Returns {tag: n_kept}."""
        counts: dict[str, int] = {}
        for tag, steps in keep.items():
            if tag not in DT_TAGS:
                raise ValueError(f"unknown tag {tag}")
            n = 0
            for L in self.decision_layers:
                b = self._turn_buf[L]
                for s in steps:
                    if 0 <= s < len(b):
                        self.kept[L]["turn"].append(self._turn)
                        self.kept[L]["step"].append(int(s))
                        self.kept[L]["tag"].append(DT_TAGS[tag])
                        self.kept[L]["vecs"].append(b[s])
                        n += 1
            counts[tag] = n // max(len(self.decision_layers), 1)
        return counts

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        out = super().to_npz_dict()
        for L in self.decision_layers:
            k = self.kept[L]
            p = f"L{L}_dt_"
            out[p + "turn"] = np.asarray(k["turn"], dtype=np.int32)
            out[p + "step"] = np.asarray(k["step"], dtype=np.int32)
            out[p + "tag"] = np.asarray(k["tag"], dtype=np.int8)
            out[p + "vecs"] = np.stack(k["vecs"]).astype(np.float16) if k["vecs"] else np.zeros((0, 0), dtype=np.float16)
        return out
