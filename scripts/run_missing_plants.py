"""Standalone: fill the Phase 5 positive-control plant gap for concepts that
never got plants (completion_drive, compliance, conscientiousness). Mirrors
run_phase.py phase5 Part B exactly (same helper, same model, same layers,
same reactive per-concept loop) but targets only these 3 concepts and does
NOT touch phase5/status.json -- phase5 is already marked done and this script
must not race with it or re-trigger it. Recomputes plant_results.json fresh
at the end from the full rollouts.jsonl (all 8 concepts), which is idempotent.
"""
import sys, os, json, time
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
from tqdm import tqdm

from scripts.run_phase import (
    execute_and_record_rollout, read_jsonl, read_phase_status, require_gpu, is_shortcut,
)
from src.concepts import CONCEPTS
from src.directions import load_vector
import yaml

run_dir = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run_dir / "config.frozen.yaml").read_text())

require_gpu("fill missing plants")
from src.agent_loop import load_model
from src.hooks import resolve_layer_band

p1 = read_phase_status(run_dir, 1)
winning = p1["winning_variant"]
model_key = "main" if winning["variant"] == "main_27b" else "recon"
model_name = cfg["model"][model_key]
model, tokenizer = load_model(model_name, dtype=cfg["model"]["dtype"])
layers = resolve_layer_band(model, cfg["model"]["layer_frac_lo"], cfg["model"]["layer_frac_hi"])
errors_log = run_dir / "incidents" / "judge_errors.jsonl"

p4 = read_phase_status(run_dir, 4)
val_table = p4["val_table"]

TARGET_CONCEPTS = os.environ.get("SCFX_PLANT_CONCEPTS", "completion_drive,compliance,conscientiousness").split(",")
N_PER = int(os.environ.get("SCFX_PLANT_N_PER", "6"))

def rows_for(concept):
    return [r for r in read_jsonl(run_dir / "rollouts.jsonl")
            if r.get("phase") == "plant" and r.get("plant_concept") == concept and r.get("status") == "ok"]

for concept in TARGET_CONCEPTS:
    concept = concept.strip()
    if val_table.get(concept, {}).get("skipped"):
        print(f"skip {concept}: validation skipped")
        continue
    while len(rows_for(concept)) < N_PER:
        have = len(rows_for(concept))
        batch = min(2, N_PER - have)
        for _ in tqdm(range(batch), desc=f"plant:{concept}", initial=have, total=N_PER):
            execute_and_record_rollout(
                run_dir=run_dir, model=model, tokenizer=tokenizer, model_name=model_name,
                enable_thinking=winning["enable_thinking"], target_errors=cfg["env"]["n_type_errors_default"],
                max_turns=min(cfg["env"]["max_turns"], 20), max_new_tokens=cfg["model"]["max_new_tokens"],
                temperature=cfg["model"]["temperature"], phase="plant", condition=f"plant_{concept}",
                capture_layer_indices=layers, plant_concept=concept, plant_prefill=CONCEPTS[concept].plant_text,
                errors_log=errors_log,
            )
    print(f"{concept}: reached {len(rows_for(concept))}/{N_PER}")

print("=== done, this worker exiting (recompute plant_results.json separately once all workers finish) ===")
