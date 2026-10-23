# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

source train.env

# The shared conda environment may have an editable HoloMotion install that
# points at a different checkout. Always import this workspace first so K1
# training uses the code and configs from holomtion_boosterK1 without changing
# the environment used by the original repository.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ $(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l) -eq 1 ]]; then
    USE_MULTI_GPU=false
else
    USE_MULTI_GPU=true
fi


config_name="${CONFIG_NAME:-motion_tracking_v1_4_0}"

# These historical K1 configs point at the pre-geometry-review snapshot.
# Keep their saved experiment configs reproducible, but do not silently train
# again on candidates that have since been placed in contact/collision review.
case "$config_name" in
    motion_tracking_booster_k1_curriculum_s1_simple|motion_tracking_booster_k1_curriculum_s2_dynamic|motion_tracking_booster_k1_curriculum_s3_complex|motion_tracking_booster_k1_release_v1_clean_small)
        review_report="data/holomotion_k1_v1/active_review_snapshot.json"
        if [[ -f "$review_report" ]]; then
            echo "This historical config uses the pre-review K1 dataset. Regenerate it against the active reviewed snapshot before general training. Report: $review_report" >&2
            exit 2
        fi
        ;;
esac

num_envs="${NUM_ENVS:-4096}"

COMMON_ARGS=(
    "holomotion/src/training/train.py"
    "--config-name=training/motion_tracking/${config_name}"
    "num_envs=${num_envs}"
    "headless=true"
    "experiment_name=${config_name}"
    "$@"
)

LAUNCH_ARGS=()
if [[ "${USE_MULTI_GPU}" == "true" ]]; then
    LAUNCH_ARGS+=(--multi_gpu)
fi

if [[ "${HOLOMOTION_TRAIN_DRY_RUN:-0}" == "1" ]]; then
    printf 'Training command:'
    printf ' %q' "${Train_CONDA_PREFIX}/bin/accelerate" launch
    printf ' %q' "${LAUNCH_ARGS[@]}" "${COMMON_ARGS[@]}"
    printf '\n'
    exit 0
fi

exec "${Train_CONDA_PREFIX}/bin/accelerate" launch \
    "${LAUNCH_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
