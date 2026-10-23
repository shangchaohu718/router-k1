#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

source train.env

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

CKPT_PATH="${CKPT_PATH:-logs/HoloMotionBoosterK1LaFAN/20261009_092900-motion_tracking_booster_k1_lafan/model_9999.pt}"
TRAIN_H5_DATASET_PATH="${TRAIN_H5_DATASET_PATH:-['data/lafan1_booster_k1_gmr_h5_train_subject1_4']}"
NUM_ENVS="${NUM_ENVS:-2}"
MOTION_NAMES="${MOTION_NAMES:-['aiming1_subject1','aiming1_subject4']}"
MAX_FRAME_LENGTH="${MAX_FRAME_LENGTH:-20000}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-5000}"
DATA_LOADER_NUM_WORKERS="${DATA_LOADER_NUM_WORKERS:-2}"
DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-600}"
EXCLUDED_MOTION_NAMES="${EXCLUDED_MOTION_NAMES:-['obstacles1_subject1','obstacles1_subject2','obstacles2_subject1','obstacles2_subject2','obstacles3_subject3','obstacles3_subject4','obstacles4_subject2','obstacles4_subject3','obstacles4_subject4','obstacles5_subject2','obstacles5_subject3','obstacles5_subject4','obstacles6_subject1','obstacles6_subject4']}"

if [[ "${DATA_LOADER_NUM_WORKERS}" == "0" ]]; then
    DATA_LOADER_TIMEOUT=0
fi

extra_args=()
if [[ -n "${MOTION_NAMES}" ]]; then
    extra_args+=("robot.motion.handpicked_motion_names=${MOTION_NAMES}")
fi

"${Train_CONDA_PREFIX}/bin/accelerate" launch \
    holomotion/src/evaluation/eval_motion_tracking_single.py \
    --config-name=evaluation/eval_isaaclab \
    robot=booster/K1/22dof/22dof_training_isaaclab \
    env/terminations=termination_motion_tracking \
    headless=true \
    num_envs="${NUM_ENVS}" \
    export_policy=false \
    dump_npzs=true \
    calc_per_clip_metrics=false \
    generate_report=false \
    motion_h5_path="${TRAIN_H5_DATASET_PATH}" \
    +use_kv_cache=true \
    export_only=false \
    checkpoint="${CKPT_PATH}" \
    project_name=HoloMotionBoosterK1LaFANTrainReplay \
    robot.motion.max_frame_length="${MAX_FRAME_LENGTH}" \
    +algo.config.offline_eval_max_steps="${MAX_EVAL_FRAMES}" \
    robot.motion.excluded_motion_names="${EXCLUDED_MOTION_NAMES}" \
    robot.motion.dataloader.num_workers="${DATA_LOADER_NUM_WORKERS}" \
    robot.motion.dataloader.timeout="${DATA_LOADER_TIMEOUT}" \
    "${extra_args[@]}"
