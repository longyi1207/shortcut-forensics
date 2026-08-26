"""Mean-diff recovers a planted axis on toy tensors. SPEC.md §9."""

from __future__ import annotations

import numpy as np

from src.directions import cosine, fit_direction_sweep, mean_diff, pair_accuracy, signed_auc


def _planted_data(dim=64, n=200, seed=0, noise=1.0, signal=3.0, true_dir=None):
    rng = np.random.default_rng(seed)
    if true_dir is None:
        true_dir = rng.normal(size=dim)
        true_dir /= np.linalg.norm(true_dir)
    plus = rng.normal(size=(n, dim)) * noise + signal * true_dir
    minus = rng.normal(size=(n, dim)) * noise - signal * true_dir
    return plus, minus, true_dir


def test_mean_diff_recovers_planted_axis():
    plus, minus, true_dir = _planted_data()
    d = mean_diff(plus, minus)
    assert cosine(d, true_dir) > 0.9


def test_mean_diff_high_val_accuracy_on_held_out():
    train_plus, train_minus, true_dir = _planted_data(seed=1, signal=5.0)
    val_plus, val_minus, _ = _planted_data(seed=2, signal=5.0, true_dir=true_dir)
    d = mean_diff(train_plus, train_minus)
    acc = pair_accuracy(d, val_plus, val_minus)
    assert acc >= 0.9


def test_mean_diff_fails_on_pure_noise():
    dim, n = 64, 200
    rng = np.random.default_rng(3)
    plus = rng.normal(size=(n, dim))
    minus = rng.normal(size=(n, dim))
    d = mean_diff(plus, minus)
    val_plus = rng.normal(size=(50, dim))
    val_minus = rng.normal(size=(50, dim))
    acc = pair_accuracy(d, val_plus, val_minus)
    assert 0.3 < acc < 0.7  # should be near chance, not spuriously high


def test_fit_direction_sweep_picks_best_layer():
    dim, n = 32, 150
    rng = np.random.default_rng(4)
    good_dir = rng.normal(size=dim)
    good_dir /= np.linalg.norm(good_dir)

    def make(seed, signal):
        r = np.random.default_rng(seed)
        base_p = r.normal(size=(n, dim))
        base_m = r.normal(size=(n, dim))
        return base_p + signal * good_dir, base_m - signal * good_dir

    train_plus_by_layer = {}
    train_minus_by_layer = {}
    val_plus_by_layer = {}
    val_minus_by_layer = {}
    for layer, signal in [(5, 0.05), (10, 3.0), (15, 0.1)]:  # layer 10 is the "good" layer
        tp, tm = make(layer, signal)
        vp, vm = make(layer + 100, signal)
        train_plus_by_layer[layer] = tp
        train_minus_by_layer[layer] = tm
        val_plus_by_layer[layer] = vp
        val_minus_by_layer[layer] = vm

    best = fit_direction_sweep(train_plus_by_layer, train_minus_by_layer, val_plus_by_layer, val_minus_by_layer)
    assert best["layer"] == 10
    assert best["val_acc"] > 0.85


def test_signed_auc_pro_cheat_vs_pro_honest_interpretation():
    rng = np.random.default_rng(5)
    scores_shortcut = rng.normal(loc=1.0, scale=0.5, size=100)
    scores_honest = rng.normal(loc=-1.0, scale=0.5, size=100)

    pro_cheat = signed_auc(scores_shortcut, scores_honest, sign="pro_cheat")
    assert pro_cheat["raw_auc"] > 0.8
    assert pro_cheat["hypothesis_auc"] == pro_cheat["raw_auc"]

    pro_honest = signed_auc(scores_shortcut, scores_honest, sign="pro_honest")
    assert pro_honest["raw_auc"] == pro_cheat["raw_auc"]  # same data, same raw number
    assert pro_honest["hypothesis_auc"] < 0.2  # honest doesn't score higher here -> low hypothesis support
