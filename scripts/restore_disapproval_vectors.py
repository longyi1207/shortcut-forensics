"""Restore vectors/disapproval_L{8,12,17,19}_refit.npz -- overwritten by a
buggy tedium fix script that had "disapproval" hardcoded in its save path.
Refit is deterministic given the same pairs, so this exactly reproduces what
fix_disapproval_plant.py originally saved.
"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path

from scripts.run_phase import read_jsonl
from src.directions import fit_direction_sweep, save_vector
from src.agent_loop import load_model
from src.hooks import extract_hidden_last_token_batch
import yaml

run_dir = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run_dir / "config.frozen.yaml").read_text())
concept = "disapproval"
pdir = run_dir / "pairs" / concept
train = read_jsonl(pdir / "train.jsonl")
val = read_jsonl(pdir / "val.jsonl")

CANDIDATE_LAYERS = [8, 12, 17, 19]
model, tokenizer = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])

train_plus = extract_hidden_last_token_batch(model, tokenizer, [p["plus"] for p in train], CANDIDATE_LAYERS)
train_minus = extract_hidden_last_token_batch(model, tokenizer, [p["minus"] for p in train], CANDIDATE_LAYERS)
val_plus = extract_hidden_last_token_batch(model, tokenizer, [p["plus"] for p in val], CANDIDATE_LAYERS)
val_minus = extract_hidden_last_token_batch(model, tokenizer, [p["minus"] for p in val], CANDIDATE_LAYERS)

pair_ids = [p["id"] for p in train]
for layer in CANDIDATE_LAYERS:
    d = fit_direction_sweep(
        {layer: train_plus[layer]}, {layer: train_minus[layer]},
        {layer: val_plus[layer]}, {layer: val_minus[layer]},
    )["d"]
    save_vector(run_dir / "vectors" / f"disapproval_L{layer}_refit", d, layer, pair_ids, 0.0,
                meta={"concept": "disapproval", "note": "RESTORED after accidental overwrite by tedium fix script"})
    print(f"restored L{layer}, norm={float((d**2).sum()**0.5):.3f}")
print("done")
