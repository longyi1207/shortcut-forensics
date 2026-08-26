"""src/hooks.py + src/steer.py against a tiny real nn.Module stack (not the
real Qwen weights, but real torch forward hooks) -- these can only otherwise
be exercised on the GPU VM. Catches hook registration/removal and
ablate/add math bugs before they cost GPU-hours.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.hooks import get_layers, resolve_layer_band
from src.steer import SteeringSession, random_unit_vector


class FakeDecoderLayer(nn.Module):
    """Returns (hidden_states,) like a real HF decoder layer's forward()."""

    def forward(self, hidden_states):
        return (hidden_states,)


class FakeCausalLM(nn.Module):
    def __init__(self, n_layers=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(n_layers)])

    def forward(self, hidden_states):
        for layer in self.model.layers:
            (hidden_states,) = layer(hidden_states)
        return hidden_states


def test_get_layers_finds_decoder_stack():
    model = FakeCausalLM(n_layers=6)
    layers = get_layers(model)
    assert len(layers) == 6


def test_resolve_layer_band_respects_fractions():
    model = FakeCausalLM(n_layers=40)
    band = resolve_layer_band(model, 0.60, 0.75)
    assert band[0] == round(40 * 0.60)
    assert band[-1] == round(40 * 0.75)
    assert all(0 <= i < 40 for i in band)


def test_steering_session_ablate_zeroes_projection():
    model = FakeCausalLM(n_layers=4)
    dim = 16
    d = np.zeros(dim)
    d[0] = 1.0  # unit vector along dim 0

    hs = torch.zeros(1, 3, dim)
    hs[0, :, 0] = 5.0  # component entirely along d
    hs[0, :, 1] = 2.0  # orthogonal component, should survive

    with SteeringSession(model, [1], d, "ablate", device="cpu"):
        out = model(hs)

    assert torch.allclose(out[0, :, 0], torch.zeros(3), atol=1e-5)
    assert torch.allclose(out[0, :, 1], torch.full((3,), 2.0), atol=1e-5)


def test_steering_session_add_shifts_by_alpha_times_d():
    model = FakeCausalLM(n_layers=4)
    dim = 8
    d = np.zeros(dim)
    d[2] = 3.0  # raw (non-unit) direction

    hs = torch.zeros(1, 2, dim)
    with SteeringSession(model, [0], d, "add", alpha=2.0, device="cpu"):
        out = model(hs)

    expected = torch.zeros(1, 2, dim)
    expected[..., 2] = 6.0  # alpha * d[2]
    assert torch.allclose(out, expected, atol=1e-5)


def test_steering_session_identity_is_noop():
    model = FakeCausalLM(n_layers=4)
    hs = torch.randn(1, 3, 8)
    with SteeringSession(model, [1], None, "identity", device="cpu"):
        out = model(hs)
    assert torch.equal(out, hs)


def test_steering_session_removes_hooks_on_exit():
    model = FakeCausalLM(n_layers=4)
    dim = 8
    d = random_unit_vector(dim, seed=0)
    hs = torch.randn(1, 2, dim)

    with SteeringSession(model, [0, 1], d, "ablate", device="cpu"):
        steered = model(hs.clone())
    unsteered_after = model(hs.clone())

    assert not torch.equal(steered, hs)  # hook actually changed output while active
    assert torch.equal(unsteered_after, hs)  # and had zero effect once removed


def test_steering_session_only_hooks_specified_layers():
    model = FakeCausalLM(n_layers=4)
    dim = 8
    d = np.ones(dim) / np.sqrt(dim)
    hs = torch.randn(1, 2, dim)

    # ablate at layer 3 only should differ from ablating at layer 0 only,
    # given the same input (both zero-effect here since layers are identity
    # passthroughs and hs is unchanged between them -- so instead check the
    # handle count directly).
    sess = SteeringSession(model, [0, 3], d, "ablate", device="cpu")
    with sess:
        assert len(sess.handles) == 2
    assert sess.handles == []


def test_random_unit_vector_is_unit_norm_and_deterministic():
    v1 = random_unit_vector(32, seed=42)
    v2 = random_unit_vector(32, seed=42)
    v3 = random_unit_vector(32, seed=43)
    assert np.isclose(np.linalg.norm(v1), 1.0)
    assert np.array_equal(v1, v2)
    assert not np.array_equal(v1, v3)
