"""Combined ablation: remove two concepts' directions simultaneously, at their
own layers, in the same rollout -- tests whether two independently-significant
single-mechanism effects are the same underlying computation (result plateaus
near either alone) or genuinely separate contributing pathways (result drops
further, roughly additively).

Targets the project's two most solid single-mechanism results: `tedium`
ablated at layer 19 (coarse mean-diff direction; SAE §4.7 found the stronger
effect but this uses the coarse vector for a clean, simple combination) and
`shortcut` ablated at a deep layer (the L19-fitted vector, hook inserted at
L22/26/29 -- Finding 01, p=0.027 pooled). Two SteeringSessions at different
layers don't need hook-chaining tricks (register_forward_hook on different
modules is independent) -- ExitStack is used only for symmetry with the SAE
multi-feature scripts.
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

TEDIUM_LAYER = int(os.environ.get("SCFX_COMBO_TEDIUM_LAYER", "19"))
SHORTCUT_LAYER = int(os.environ.get("SCFX_COMBO_SHORTCUT_LAYER", "26"))
N_TARGET = int(os.environ.get("SCFX_COMBO_N", "30"))
CONDITION_NAME = f"combined_ablate_tedium_L{TEDIUM_LAYER}_shortcut_L{SHORTCUT_LAYER}"

model, tokenizer = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
winning = read_phase_status(run_dir, 1)["winning_variant"]
errors_log = run_dir / "incidents" / "judge_errors.jsonl"

tedium_vec = load_vector(run_dir / "vectors" / "tedium")["d"]
shortcut_vec = load_vector(run_dir / "vectors" / "shortcut")["d"]  # L19-fitted, applied at SHORTCUT_LAYER
print(f"combined ablation: tedium@L{TEDIUM_LAYER} + shortcut(L19-fitted)@L{SHORTCUT_LAYER}, target n={N_TARGET}")


def rows_now():
    return [r for r in read_jsonl(run_dir / "rollouts.jsonl")
            if r.get("phase") == "signed_pack" and r.get("condition") == CONDITION_NAME and r.get("status") == "ok"]


while len(rows_now()) < N_TARGET:
    have = len(rows_now())
    print(f"{CONDITION_NAME}: have {have}/{N_TARGET}")
    with ExitStack() as stack:
        stack.enter_context(SteeringSession(model, [TEDIUM_LAYER], tedium_vec, mode="ablate"))
        stack.enter_context(SteeringSession(model, [SHORTCUT_LAYER], shortcut_vec, mode="ablate"))
        execute_and_record_rollout(
            run_dir=run_dir, model=model, tokenizer=tokenizer, model_name=cfg["model"]["recon"],
            enable_thinking=winning["enable_thinking"], target_errors=cfg["env"]["n_type_errors_default"],
            max_turns=cfg["env"]["max_turns"], max_new_tokens=cfg["model"]["max_new_tokens"],
            temperature=cfg["model"]["temperature"], phase="signed_pack", condition=CONDITION_NAME,
            capture_layer_indices=None, errors_log=errors_log,
        )

print(f"{CONDITION_NAME}: reached {len(rows_now())}/{N_TARGET}")
