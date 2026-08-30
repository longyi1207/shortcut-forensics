"""Multi-feature SAE steering, add or ablate mode. Sufficiency-test variant of
sae_multi_ablate.py: mode="add" amplifies the top-10 SAE features' projections
simultaneously instead of removing them, to test whether strengthening a
concept (e.g. tedium) increases the target behavior (shortcut rate) -- the
natural complement to the ablation (necessity) tests already run.

Nests 10 SteeringSession context managers at the same layer via ExitStack --
forward hooks chain through their return value in PyTorch, so this correctly
applies sequential steering of each feature's direction without touching
steer.py. Standalone script (bypasses phase6/conditions.json, which only
supports one vector per condition), same pattern as sae_multi_ablate.py.
"""
import sys, os, json
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

CONCEPT = os.environ.get("SCFX_SAE_CONCEPT", "tedium")
LAYER = int(os.environ.get("SCFX_SAE_LAYER", "19"))
MODE = os.environ.get("SCFX_SAE_MODE", "add")
ALPHA = float(os.environ.get("SCFX_SAE_ALPHA", "1.0"))
N_MULTI = 10
N_TARGET = int(os.environ.get("SCFX_SAE_N", "30"))
PREFIX = "add_pos_sae" if MODE == "add" else "ablate_sae"
CONDITION_NAME = f"{PREFIX}_{CONCEPT}_top10_L{LAYER}"

model, tokenizer = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
winning = read_phase_status(run_dir, 1)["winning_variant"]
errors_log = run_dir / "incidents" / "judge_errors.jsonl"

probe = json.loads((run_dir / "phase4" / f"sae_probe_{CONCEPT}_L{LAYER}.json").read_text())
mean_diffs = {f["rank"]: f["mean_diff"] for f in probe["top_features"]}

feature_vecs = []
for i in range(N_MULTI):
    v = load_vector(run_dir / "vectors" / f"sae_{CONCEPT}_top10_L{LAYER}_f{i}")["d"]  # unit-norm decoder direction
    if MODE == "add":
        v = v * mean_diffs[i]  # scale to "one empirical mean-diff's worth" for this feature, Turner-ActAdd convention
    feature_vecs.append(v)
print(f"loaded {len(feature_vecs)} SAE feature vectors for {CONCEPT}@L{LAYER}, mode={MODE}, alpha={ALPHA}")
if MODE == "add":
    print("per-feature mean_diff scales:", [round(mean_diffs[i], 3) for i in range(N_MULTI)])


def rows_now():
    return [r for r in read_jsonl(run_dir / "rollouts.jsonl")
            if r.get("phase") == "signed_pack" and r.get("condition") == CONDITION_NAME and r.get("status") == "ok"]


while len(rows_now()) < N_TARGET:
    have = len(rows_now())
    print(f"{CONDITION_NAME}: have {have}/{N_TARGET}")
    with ExitStack() as stack:
        for v in feature_vecs:
            stack.enter_context(SteeringSession(model, [LAYER], v, mode=MODE, alpha=ALPHA))
        execute_and_record_rollout(
            run_dir=run_dir, model=model, tokenizer=tokenizer, model_name=cfg["model"]["recon"],
            enable_thinking=winning["enable_thinking"], target_errors=cfg["env"]["n_type_errors_default"],
            max_turns=cfg["env"]["max_turns"], max_new_tokens=cfg["model"]["max_new_tokens"],
            temperature=cfg["model"]["temperature"], phase="signed_pack", condition=CONDITION_NAME,
            capture_layer_indices=None, errors_log=errors_log,
        )

print(f"{CONDITION_NAME}: reached {len(rows_now())}/{N_TARGET}")
