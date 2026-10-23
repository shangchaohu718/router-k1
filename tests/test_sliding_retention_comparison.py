"""Matched prefixes prevent early-failure averages from hiding loss."""

import numpy as np
import pytest

from tools.booster_k1_retarget.compare_sliding_router_retention import (
    compare_arrays,
)


def trajectory(n):
    return {
        "robot_dof_pos": np.zeros((n, 2)),
        "ref_dof_pos": np.zeros((n, 2)),
        "robot_global_translation": np.zeros((n, 3, 3)),
        "ref_global_translation": np.zeros((n, 3, 3)),
    }


def test_unmatched_bad_suffix_is_excluded_and_world_translation_cancels():
    base, candidate = trajectory(5), trajectory(3)
    base["robot_dof_pos"][3:] = 100
    candidate["robot_dof_pos"] += 0.25
    candidate["robot_global_translation"] += 50
    candidate["ref_global_translation"] += 50
    result = compare_arrays(base, candidate)
    assert result["matched_frames"] == 3
    assert result["matched_joint_mae_increase_rad"] == 0.25
    assert result["candidate_matched_body_error_mm"] == 0


def test_misaligned_reference_is_not_treated_as_policy_regression():
    base, candidate = trajectory(3), trajectory(3)
    candidate["ref_dof_pos"][1] = 0.01
    with pytest.raises(ValueError, match="Reference mismatch"):
        compare_arrays(base, candidate)


def test_empty_prefix_cannot_be_compared():
    with pytest.raises(ValueError, match="No common"):
        compare_arrays(trajectory(0), trajectory(3))
