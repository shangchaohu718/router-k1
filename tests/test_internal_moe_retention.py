"""Retention evidence excludes padding and distinguishes incomplete clips."""

import json

import numpy as np

from tools.booster_k1_retarget.compare_internal_moe_retention import compare


def test_retention_uses_valid_prefix_for_joints_and_route_selections(tmp_path):
    for name in ("base", "candidate"):
        folder = tmp_path / name / "checkpoint/isaaclab_eval_output_test"
        folder.mkdir(parents=True)
        q = np.zeros((5, 2))
        if name == "candidate":
            q[3:] = 999
        meta = {
            "motion_key": "old_clip",
            "clip_length": 15,
            "valid_prefix_len": 3,
        }
        np.savez(
            folder / "old.npz",
            metadata=np.asarray(json.dumps(meta)),
            robot_dof_pos=q,
            ref_dof_pos=np.zeros((5, 2)),
            robot_global_translation=np.zeros((5, 2, 3)),
            ref_global_translation=np.zeros((5, 2, 3)),
        )
    np.savez(
        tmp_path / "candidate/ordered_observations.npz",
        old_fine_experts=np.asarray(8),
        selected_experts=np.asarray([[0, 1], [0, 1], [0, 1], [8, 9], [8, 9]]),
        motion_keys=np.asarray(["old_clip"] * 5),
        reference_frames=np.arange(5),
        batch_ids=np.zeros(5),
    )
    report = compare(tmp_path / "base", tmp_path / "candidate")
    assert report["all_robot_joint_prefixes_bitwise_equal"]
    assert report["valid_recorded_steps"] == 3
    assert report["steps_selecting_added_experts"] == 0
    assert report["base_completed"] == report["candidate_completed"] == 0
