"""Multi-feature SAE ablation: remove the top-10 SAE features' projections
simultaneously, not just the single dominant one. Mirrors code/tom_empathy's
finding that ablating 10 features together moved behavior ~4x more than one
feature alone, for a diffuse concept where the coarse mean-diff direction
under-captures the real causal computation.

Nests 10 SteeringSession context managers at the same layer via ExitStack --
forward hooks chain through their return value in PyTorch, so this correctly
applies sequential removal of each feature's projection without touching
steer.py. Does NOT go through the standard phase6/conditions.json path (that
only supports one vector per condition) -- this is a standalone script,
same pattern as tonight's other repair scripts.
"""
import sys, os
sys.path.insert(0, ".")
from pathlib import Path
from contextlib import ExitStack

from scripts.run_phase import execute_and_record_rollout, read_jsonl, read_phase_status
from src.directions import load_vector
from src.agent_loop import load_model
from src.steer import SteeringSession
import yaml

run_dir = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run_dir / "config.frozen.yaml").read_text())

CONCEPT = os.environ.get("SCFX_SAE_CONCEPT", "shortcut")
LAYER = int(os.environ.get("SCFX_SAE_LAYER", "19"))
N_MULTI = 10
N_TARGET = int(os.environ.get("SCFX_SAE_N", "10"))
CONDITION_NAME = f"ablate_sae_{CONCEPT}_top10_L{LAYER}"

model, tokenizer = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
winning = read_phase_status(run_dir, 1)["winning_variant"]
errors_log = run_dir / "incidents" / "judge_errors.jsonl"

feature_vecs = []
for i in range(N_MULTI):
    v = load_vector(run_dir / "vectors" / f"sae_{CONCEPT}_top10_L{LAYER}_f{i}")["d"]
    feature_vecs.append(v)
print(f"loaded {len(feature_vecs)} SAE feature vectors for {CONCEPT}@L{LAYER}")


def rows_now():
    return [r for r in read_jsonl(run_dir / "rollouts.jsonl")
            if r.get("phase") == "signed_pack" and r.get("condition") == CONDITION_NAME and r.get("status") == "ok"]


while len(rows_now()) < N_TARGET:
    have = len(rows_now())
    print(f"{CONDITION_NAME}: have {have}/{N_TARGET}")
    with ExitStack() as stack:
        for v in feature_vecs:
            stack.enter_context(SteeringSession(model, [LAYER], v, mode="ablate"))
        execute_and_record_rollout(
            run_dir=run_dir, model=model, tokenizer=tokenizer, model_name=cfg["model"]["recon"],
            enable_thinking=winning["enable_thinking"], target_errors=cfg["env"]["n_type_errors_default"],
            max_turns=cfg["env"]["max_turns"], max_new_tokens=cfg["model"]["max_new_tokens"],
            temperature=cfg["model"]["temperature"], phase="signed_pack", condition=CONDITION_NAME,
            capture_layer_indices=None, errors_log=errors_log,
        )

print(f"{CONDITION_NAME}: reached {len(rows_now())}/{N_TARGET}")
