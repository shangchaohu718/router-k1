#!/usr/bin/env bash
# Each invocation runs ONE stage. Inspect its tracking before advancing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

stage="${1:-1}"
previous_checkpoint="${2:-}"
case "$stage" in
    1) config=motion_tracking_booster_k1_curriculum_s1_simple; iterations=3000 ;;
    2) config=motion_tracking_booster_k1_curriculum_s2_dynamic; iterations=5000 ;;
    3) config=motion_tracking_booster_k1_curriculum_s3_complex; iterations=12000 ;;
    *) echo "Usage: bash $0 {1|2|3} [previous_stage_checkpoint.pt]" >&2; exit 2 ;;
esac
if (( $# > 2 )); then
    echo "Unexpected arguments; only stage and previous checkpoint are supported." >&2
    exit 2
fi

args=()
if [[ "$stage" != 1 && -z "$previous_checkpoint" ]]; then
    echo "Stage $stage requires the previous stage's checkpoint; no automatic fallback." >&2
    exit 2
fi
if [[ -n "$previous_checkpoint" ]]; then
    if [[ ! -f "$previous_checkpoint" ]]; then
        echo "Checkpoint not found: $previous_checkpoint" >&2
        exit 2
    fi
    model_dir="${previous_checkpoint%.pt}"
    for role in actor critic; do
        if [[ ! -f "$model_dir/$role/model.safetensors" && ! -f "$model_dir/$role/pytorch_model.bin" ]]; then
            echo "Missing $role weights under $model_dir" >&2
            exit 2
        fi
    done
    # Quote Hydra's string value so paths containing spaces remain valid.
    args+=("checkpoint='$previous_checkpoint'")
fi

export CONFIG_NAME="$config"
export NUM_ENVS="${NUM_ENVS:-512}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Sequential curriculum stage $stage: $iterations additional PPO iterations."
echo "This command will NOT automatically advance to the next stage."
bash "$SCRIPT_DIR/train_motion_tracking.sh" "${args[@]}"
