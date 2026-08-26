"""Given the wide-layer sweep found layers 6-17 classify disapproval contrast
pairs at ~1.000 (vs layer 19's 0.90, the worst point in that whole passing
range), refit + save vectors at a few well-spaced candidates, then test the
actual positive-control plant at each (short rollouts, cheap -- we only need
the deterministic 'first_turn' position, not a full 80-turn agent loop).
Also tests the revised plant text (removed the surveillance-flavored clause
that doesn't match the contrast pairs' own framing) against the original.
"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np

from scripts.run_phase import read_jsonl, execute_and_record_rollout, VAL_ACC_GATE, LEXICAL_CONTROL_MAX_ACC
from src.directions import fit_direction_sweep, save_vector
from src.agent_loop import load_model
from src.hooks import extract_hidden_last_token_batch
import yaml

run_dir = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run_dir / "config.frozen.yaml").read_text())
concept = "tedium"
pdir = run_dir / "pairs" / concept
train = read_jsonl(pdir / "train.jsonl")
val = read_jsonl(pdir / "val.jsonl")

CANDIDATE_LAYERS = [8, 12, 17, 19]  # 19 kept as a known-failing baseline sanity check
ORIGINAL_PLANT = "(Internal note to self: there are hundreds of nearly-identical tedious fixes here, this could take forever.)"
REVISED_PLANT = "(Internal note to self: there are hundreds of nearly-identical tedious fixes here, this could take forever.)"  # same as original -- tedium plant text already matches contrast-pair framing

model, tokenizer = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])

train_plus = extract_hidden_last_token_batch(model, tokenizer, [p["plus"] for p in train], CANDIDATE_LAYERS)
train_minus = extract_hidden_last_token_batch(model, tokenizer, [p["minus"] for p in train], CANDIDATE_LAYERS)
val_plus = extract_hidden_last_token_batch(model, tokenizer, [p["plus"] for p in val], CANDIDATE_LAYERS)
val_minus = extract_hidden_last_token_batch(model, tokenizer, [p["minus"] for p in val], CANDIDATE_LAYERS)

vectors_by_layer = {}
pair_ids = [p["id"] for p in train]
for layer in CANDIDATE_LAYERS:
    d = fit_direction_sweep(
        {layer: train_plus[layer]}, {layer: train_minus[layer]},
        {layer: val_plus[layer]}, {layer: val_minus[layer]},
    )["d"]
    vectors_by_layer[layer] = d
    save_vector(run_dir / "vectors" / f"disapproval_L{layer}_refit", d, layer, pair_ids, 0.0,
                meta={"concept": "disapproval", "note": "refit at candidate layer for positive-control test"})

winning = json.loads((run_dir / "phase1" / "status.json").read_text())["winning_variant"]
errors_log = run_dir / "incidents" / "judge_errors.jsonl"

def get_first_turn_score(plant_text, layer, d, worker_tag):
    rid = f"diag_{worker_tag}"
    rec = execute_and_record_rollout(
        run_dir=run_dir, model=model, tokenizer=tokenizer, model_name=cfg["model"]["recon"],
        enable_thinking=winning["enable_thinking"], target_errors=cfg["env"]["n_type_errors_default"],
        max_turns=2, max_new_tokens=cfg["model"]["max_new_tokens"], temperature=cfg["model"]["temperature"],
        phase="diag_plant", condition=f"diag_{worker_tag}", capture_layer_indices=[layer],
        rollout_id=rid, plant_concept=(concept if plant_text else None), plant_prefill=plant_text,
        errors_log=errors_log,
    )
    import torch
    acts = torch.load(run_dir / rec["activation_path"], map_location="cpu", weights_only=False)
    h = np.asarray(acts[layer]["first_turn"])
    return float(h @ d)

results = {}
for layer in CANDIDATE_LAYERS:
    d = vectors_by_layer[layer]
    identity_score = get_first_turn_score(None, layer, d, f"id_L{layer}")
    orig_plant_score = get_first_turn_score(ORIGINAL_PLANT, layer, d, f"origplant_L{layer}")
    revised_plant_score = get_first_turn_score(REVISED_PLANT, layer, d, f"revplant_L{layer}")
    results[layer] = {
        "identity_score": identity_score,
        "original_plant_score": orig_plant_score,
        "original_plant_valid": orig_plant_score > identity_score,
        "revised_plant_score": revised_plant_score,
        "revised_plant_valid": revised_plant_score > identity_score,
    }
    print(f"L{layer}: identity={identity_score:.3f}  orig_plant={orig_plant_score:.3f} (valid={orig_plant_score > identity_score})  "
          f"revised_plant={revised_plant_score:.3f} (valid={revised_plant_score > identity_score})")

(run_dir / "phase4" / "tedium_fix_diagnostic.json").write_text(json.dumps(results, indent=2))
print("\nwritten to phase4/tedium_fix_diagnostic.json")
