#!/usr/bin/env bash
# Sequential overnight run; completion gates are NOT policy-quality gates.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if (( $# != 0 )); then
    echo "Usage: bash $0 (no arguments)" >&2
    exit 2
fi
source train.env
set -u
export NUM_ENVS="${NUM_ENVS:-512}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
dry_run="${HOLOMOTION_TRAIN_DRY_RUN:-0}"

if [[ "$dry_run" == 1 ]]; then
    run_dir="$REPO_ROOT/logs/HoloMotionK1SequentialCurriculum/DRY_RUN"
else
    mkdir -p logs/HoloMotionK1SequentialCurriculum
    # Exclusive directory creation prevents checkpoint reuse from other runs.
    run_dir="$(mktemp -d "$REPO_ROOT/logs/HoloMotionK1SequentialCurriculum/overnight_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
fi
echo "Curriculum output: $run_dir"
echo "3000 -> 5000 -> 12000 PPO iterations; automatic completion-based transitions."
echo "No tracking-quality approval is inferred at stage transitions."

previous_checkpoint="$REPO_ROOT/logs/HoloMotionBoosterK1LaFANWalkDanceJointRewardSmall/20261016_092811-motion_tracking_booster_k1_lafan_walk_dance_joint_reward_small/model_19999.pt"

validate_checkpoint() {
    "$Train_CONDA_PREFIX/bin/python" - "$1" "$2" <<'PY'
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file

path = Path(sys.argv[1])
expected_iteration = int(sys.argv[2])
state = torch.load(path, map_location="cpu", weights_only=False)
if int(state.get("iter", -1)) != expected_iteration:
    raise RuntimeError(f"Unexpected checkpoint iteration: {path}")
for role in ("actor", "critic"):
    folder = path.with_suffix("") / role
    safe = folder / "model.safetensors"
    weights = (
        load_file(str(safe), device="cpu")
        if safe.is_file()
        else torch.load(folder / "pytorch_model.bin", map_location="cpu",
                        weights_only=True)
    )
    if not weights:
        raise RuntimeError(f"Empty {role} weights: {folder}")
    for name, tensor in weights.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite {role} weight: {name}")
print(f"Checkpoint integrity passed: {path}", flush=True)
PY
}

if [[ "$dry_run" != 1 ]]; then
    validate_checkpoint "$previous_checkpoint" 19999
fi
labels=(simple dynamic complex)
iterations=(3000 5000 12000)
for index in 0 1 2; do
    stage=$((index + 1))
    config="motion_tracking_booster_k1_curriculum_s${stage}_${labels[$index]}"
    stage_dir="$run_dir/stage_${stage}_${labels[$index]}"
    echo "Starting stage $stage from $previous_checkpoint"
    # Run synchronously. A nonzero exit terminates the entire chain.
    CONFIG_NAME="$config" bash "$SCRIPT_DIR/train_motion_tracking.sh" \
        "checkpoint='$previous_checkpoint'" \
        "experiment_dir='$stage_dir'" \
        "algo.config.num_learning_iterations=${iterations[$index]}" \
        "algo.config.checkpoint_load_mode=finetune" \
        "algo.config.finetune_continue_iteration=false"
    previous_checkpoint="$stage_dir/model_$((iterations[$index] - 1)).pt"
    if [[ "$dry_run" != 1 ]]; then
        validate_checkpoint "$previous_checkpoint" "$((iterations[$index] - 1))"
    fi
done
if [[ "$dry_run" == 1 ]]; then
    echo "Dry run finished; no training or output directories created."
else
    echo "All three training stages completed. Final checkpoint: $previous_checkpoint"
    echo "Evaluate tracking, stability and forgetting before calling this policy successful."
fi
