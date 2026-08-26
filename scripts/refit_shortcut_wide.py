"""shortcut's problem is different from disapproval/tedium: it already validates
cleanly (95% held-out acc) and passes its positive control at L19 -- it's just
causally inert there. Tonight's sweep showed L22/L26/L29 (reusing the L19-fit
vector, just applied deeper) suppress shortcuts hard. This script properly,
independently re-fits + validates NEW directions at a wide layer range so we
stop reusing the L19 vector and can causally test a direction that was actually
fit and validated at its own layer.
"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np

from scripts.run_phase import read_jsonl, _scramble, VAL_ACC_GATE, LEXICAL_CONTROL_MAX_ACC
from src.directions import fit_direction_sweep, pair_accuracy, save_vector
from src.agent_loop import load_model
from src.hooks import extract_hidden_last_token_batch
import yaml

run_dir = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run_dir / "config.frozen.yaml").read_text())

concept = "shortcut"
pdir = run_dir / "pairs" / concept
train = read_jsonl(pdir / "train.jsonl")
val = read_jsonl(pdir / "val.jsonl")
print(f"train={len(train)} val={len(val)}")

model, tokenizer = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])

LAYERS = list(range(14, 32))  # below and above the original 19-24 band, up to the model's max layer

train_plus = extract_hidden_last_token_batch(model, tokenizer, [p["plus"] for p in train], LAYERS)
train_minus = extract_hidden_last_token_batch(model, tokenizer, [p["minus"] for p in train], LAYERS)
val_plus = extract_hidden_last_token_batch(model, tokenizer, [p["plus"] for p in val], LAYERS)
val_minus = extract_hidden_last_token_batch(model, tokenizer, [p["minus"] for p in val], LAYERS)

best = fit_direction_sweep(train_plus, train_minus, val_plus, val_minus)
print("overall best layer (by val_acc):", best["layer"], best["val_acc"])
print("per_layer_val_acc:", json.dumps({str(k): v for k, v in sorted(best["per_layer_val_acc"].items())}, indent=2))

results = {}
pair_ids = [p["id"] for p in train]
for layer in LAYERS:
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
    if passed:
        # save each passing layer's independently-fit vector under its own name,
        # so phase6 can load e.g. vectors/shortcut_L26_refit.npz directly
        save_vector(run_dir / "vectors" / f"shortcut_L{layer}_refit", d, layer, pair_ids, val_acc,
                    meta={"concept": "shortcut", "scrambled_val_acc": scrambled_acc, "note": "independently refit, not reused from L19"})

(run_dir / "phase4" / "shortcut_wide_sweep.json").write_text(json.dumps(results, indent=2))
print("\nwritten to phase4/shortcut_wide_sweep.json")
print("\nlayers that pass BOTH gates (saved as vectors/shortcut_L{layer}_refit.npz):", sorted([l for l, r in results.items() if r["passed_gate"]]))
