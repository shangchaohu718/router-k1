#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export POLICY_NPZ_DIR="logs/HoloMotionK1SequentialCurriculum/overnight_20261020_153041_Bl3s1P/stage_3_complex/isaaclab_eval_output_model_11999_moonwalk_v14"
export TRAIN_DATASET_ROOT="data/holomotion_k1_v1/filtered_robot/moonwalk_v14"
exec bash "$SCRIPT_DIR/view_booster_k1_training_comparison.sh" \
    --title "K1 v14 太空步：参考 / 未专项训练的策略" "$@"
