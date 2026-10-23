#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export POLICY_NPZ_DIR="logs/HoloMotionK1SequentialCurriculum/overnight_20261020_153041_Bl3s1P/stage_3_complex/isaaclab_eval_output_model_11999_k1_curriculum_comparison_20261021"
export TRAIN_DATASET_ROOT="data/holomotion_k1_v1/k1_curriculum_comparison_20261021"
if [[ ! -d "$POLICY_NPZ_DIR" ]]; then
    echo "Comparison rollouts are not ready: $POLICY_NPZ_DIR" >&2
    exit 1
fi
cp "$TRAIN_DATASET_ROOT/comparison_selection.json" "$POLICY_NPZ_DIR/comparison_selection.json"
exec bash "$SCRIPT_DIR/view_booster_k1_training_comparison.sh" "$@"
