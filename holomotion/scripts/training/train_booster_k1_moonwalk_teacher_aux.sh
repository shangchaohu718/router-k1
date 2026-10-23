#!/usr/bin/env bash
# Separate 500-iteration auxiliary-teacher pilot, not a resumed baseline job.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
test -f data/holomotion_k1_v1/moonwalk_lower_body_teacher_v56_v1/teacher.npz
test -f logs/HoloMotionK1MoonwalkV14Finetune/20261021_092605-motion_tracking_booster_k1_moonwalk_v14_finetune/model_1000.pt
export CONFIG_NAME=motion_tracking_booster_k1_moonwalk_teacher_aux
export NUM_ENVS="${NUM_ENVS:-64}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
exec bash "$SCRIPT_DIR/train_motion_tracking.sh" "$@"
