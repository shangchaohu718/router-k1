#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

source train.env

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

POLICY_NPZ_DIR="${POLICY_NPZ_DIR:-logs/HoloMotionBoosterK1LaFAN/20261009_092900-motion_tracking_booster_k1_lafan/isaaclab_eval_output_model_9999_lafan1_booster_k1_gmr_h5_train_subject1_4}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-data/lafan1_booster_k1_gmr_h5_train_subject1_4}"

"${Train_CONDA_PREFIX}/bin/python" \
    tools/booster_k1_retarget/k1_policy_comparison_viewer.py \
    --policy-npz-dir "${POLICY_NPZ_DIR}" \
    --dataset-root "${TRAIN_DATASET_ROOT}" \
    "$@"
