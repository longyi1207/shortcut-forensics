"""Fix attempt for disapproval's failed positive control: check whether a
WIDER layer range (not just the 19-24 band the standard 60-75%-depth
heuristic picked) has a layer that both classifies contrast pairs well AND
survives the lexical scramble control. Reuses the existing accepted
train/val pairs (no new pair generation/judging needed) and the exact same
fit/validate code path as phase4, just over more layers.
"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np

from scripts.run_phase import read_jsonl, _scramble, VAL_ACC_GATE, LEXICAL_CONTROL_MAX_ACC
from src.directions import fit_direction_sweep, pair_accuracy
from src.agent_loop import load_model
from src.hooks import extract_hidden_last_token_batch
import yaml

run_dir = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run_dir / "config.frozen.yaml").read_text())

concept = "tedium"
pdir = run_dir / "pairs" / concept
train = read_jsonl(pdir / "train.jsonl")
val = read_jsonl(pdir / "val.jsonl")
print(f"train={len(train)} val={len(val)}")

model, tokenizer = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])

LAYERS = list(range(2, 27))  # wide sweep, well beyond the original 19-24 band

train_plus = extract_hidden_last_token_batch(model, tokenizer, [p["plus"] for p in train], LAYERS)
train_minus = extract_hidden_last_token_batch(model, tokenizer, [p["minus"] for p in train], LAYERS)
val_plus = extract_hidden_last_token_batch(model, tokenizer, [p["plus"] for p in val], LAYERS)
val_minus = extract_hidden_last_token_batch(model, tokenizer, [p["minus"] for p in val], LAYERS)

best = fit_direction_sweep(train_plus, train_minus, val_plus, val_minus)
print("overall best layer (by val_acc):", best["layer"], best["val_acc"])
print("per_layer_val_acc:", json.dumps({str(k): v for k, v in sorted(best["per_layer_val_acc"].items())}, indent=2))

# Run lexical scramble control at EVERY layer with val_acc >= gate, not just the single best,
# so we can find a layer that passes BOTH gates, not just whichever has highest raw accuracy.
candidates = [l for l, acc in best["per_layer_val_acc"].items() if acc >= VAL_ACC_GATE]
print(f"\ncandidates clearing VAL_ACC_GATE={VAL_ACC_GATE}: {sorted(candidates)}")

results = {}
for layer in sorted(candidates):
    d = fit_direction_sweep(
        {layer: train_plus[layer]}, {layer: train_minus[layer]},
        {layer: val_plus[layer]}, {layer: val_minus[layer]},
    )["d"]
    scram_plus_txt = [_scramble(p["plus"], i) for i, p in enumerate(val)]
    scram_minus_txt = [_scramble(p["minus"], i + 10000) for i, p in enumerate(val)]
    scram_plus = extract_hidden_last_token_batch(model, tokenizer, scram_plus_txt, [layer], show_progress=False)
    scram_minus = extract_hidden_last_token_batch(model, tokenizer, scram_minus_txt, [layer], show_progress=False)
    scrambled_acc = pair_accuracy(d, scram_plus[layer], scram_minus[layer])
    val_acc = best["per_layer_val_acc"][layer]
    passed = val_acc >= VAL_ACC_GATE and scrambled_acc <= LEXICAL_CONTROL_MAX_ACC
    results[layer] = {"val_acc": val_acc, "scrambled_val_acc": scrambled_acc, "passed_gate": passed}
    print(f"layer {layer:2d}: val_acc={val_acc:.3f} scrambled={scrambled_acc:.3f} passed={passed}")

(run_dir / "phase4" / "tedium_wide_sweep.json").write_text(json.dumps(results, indent=2))
print("\nwritten to phase4/tedium_wide_sweep.json")
print("\nlayers that pass BOTH gates:", sorted([l for l, r in results.items() if r["passed_gate"]]))
