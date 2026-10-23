import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holomotion.src.motion_tracking.moonwalk_teacher import (
    TeacherBank,
    heading_xy,
    leg_score,
    mechanism_score,
    windowed_bonus,
)


def mechanism_inputs():
    z = torch.tensor([[[0.0, 0.0], [0.0, 0.05]]])
    v = torch.tensor([[[[-0.6, 0.0], [-0.6, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    return dict(
        points_z=z.clone(),
        velocity_xy=v.clone(),
        contact_force=torch.full((1, 2), 100.0),
        target_z=z,
        target_velocity=v,
        roles=torch.tensor([[2, 1]]),
        quality=torch.ones(1, 2),
    )


def test_paired_slide_beats_pure_backwards_translation():
    args = mechanism_inputs()
    correct = mechanism_score(**args)
    args["velocity_xy"][:, 1, :, 0] = -0.6
    backwards = mechanism_score(**args)
    assert correct.item() > 0.95 and backwards.item() < 0.1


def test_flat_support_foot_cannot_replace_heel_lift():
    args = mechanism_inputs()
    args["points_z"][:, 1, 1] = 0
    assert mechanism_score(**args).item() == 0


def test_airborne_sliding_foot_cannot_use_other_foot_contact():
    args = mechanism_inputs()
    args["points_z"][:, 0] += 0.1
    args["contact_force"][:, 0] = 0
    assert mechanism_score(**args).item() < 0.001


def test_unqualified_teacher_or_unknown_role_pays_no_complete_bonus():
    args = mechanism_inputs()
    args["quality"][:, 1] = 0
    assert mechanism_score(**args).item() == 0
    args["quality"][:] = 1
    args["roles"][:] = 0
    assert mechanism_score(**args).item() == 0


def test_joint_teacher_has_dense_bounded_signal_without_clipping():
    target = torch.zeros(2, 2, 6)
    actual = target.clone().requires_grad_()
    quality = torch.ones(2, 2)
    score = leg_score(actual, target, quality, 0.2)
    torch.testing.assert_close(score, torch.ones(2))
    worse = leg_score(actual + 0.4, target, quality, 0.2)
    assert torch.all(worse < score)
    worse.sum().backward()
    assert torch.isfinite(actual.grad).all()
    with pytest.raises(ValueError):
        leg_score(actual, target, quality, 0)


def test_heading_projection_is_yaw_equivariant():
    v = torch.tensor([[[[1.0, 2.0, 0.0]]]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    rotated = torch.tensor([[[[-2.0, 1.0, 0.0]]]])
    torch.testing.assert_close(
        heading_xy(v, identity), heading_xy(rotated, q), atol=1e-6, rtol=0
    )


def test_window_resets_on_role_switch_or_random_reference_jump():
    state = None
    roles = torch.tensor([[2, 1]])
    for i in range(8):
        value, state = windowed_bonus(
            torch.ones(1), torch.tensor([100 + i]), roles, state
        )
        assert value.item() == float(i == 7)
    value, state = windowed_bonus(
        torch.ones(1), torch.tensor([108]), roles.flip(-1), state
    )
    assert value.item() == 0
    value, state = windowed_bonus(
        torch.ones(1), torch.tensor([300]), roles.flip(-1), state
    )
    assert value.item() == 0 and state["count"].item() == 1


def make_bank(tmp_path):
    legs = [
        s + "_" + j + "_joint"
        for s in ("left", "right")
        for j in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee_pitch",
            "ankle_pitch",
            "ankle_roll",
        )
    ]
    names = ["head_joint"] + legs
    metadata = {
        "schema": "k1_lower_body_teacher_v1",
        "fps": 50,
        "frames": 500,
        "reference_motion_key": "v14",
        "expected_window_key": "v14__start_0_len_500",
        "reference_dof_names": names,
    }
    path = tmp_path / "teacher.npz"
    q = np.broadcast_to(
        np.arange(500, dtype=np.float32)[:, None, None], (500, 2, 6)
    ).copy()
    np.savez(
        path,
        metadata=np.array(json.dumps(metadata)),
        joint_names=np.array(legs),
        joint_pos=q,
        sole_relative_xy=np.zeros((500, 2, 2, 2), np.float32),
        sole_ground_z=np.zeros((500, 2, 2), np.float32),
        sole_velocity_heading_xy=np.zeros((500, 2, 2, 2), np.float32),
        role_code=np.zeros((500, 2), np.int8),
        quality=np.ones((500, 2), np.float32),
        reference_dof_pos=np.zeros((500, len(names)), np.float32),
        foot_offsets=np.zeros((2, 2, 3), np.float32),
    )
    robot = SimpleNamespace(
        joint_names=list(reversed(names)),
        body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        data=SimpleNamespace(joint_pos=torch.zeros(3, len(names))),
    )
    batch = SimpleNamespace(
        raw_motion_keys=["v14"],
        motion_keys=["v14__start_0_len_500"],
        lengths=torch.tensor([500]),
        tensors={"ref_dof_pos": torch.zeros(1, 500, len(names))},
    )
    command = SimpleNamespace(
        _motion_cache=SimpleNamespace(current_batch=batch),
        _frame_indices=torch.tensor([0, 155, 499]),
        urdf_dof_names=names,
        cfg=SimpleNamespace(target_fps=50, n_fut_frames=10),
    )
    return TeacherBank(path, robot), command, robot


def test_teacher_uses_reference_next_frame_not_elapsed_episode_time(tmp_path):
    bank, command, robot = make_bank(tmp_path)
    target = bank.select(command)
    torch.testing.assert_close(
        target["joint_pos"][:, 0, 0], torch.tensor([1.0, 156.0, 499.0])
    )
    assert len(bank.joint_ids) == 12
    assert robot.joint_names.index("head_joint") not in bank.joint_ids


def test_teacher_rejects_other_motion_or_cropped_reference(tmp_path):
    bank, command, _ = make_bank(tmp_path)
    command._motion_cache.current_batch.motion_keys = ["v14__start_50_len_450"]
    with pytest.raises(ValueError, match="window mismatch"):
        bank.select(command)


def test_teacher_rejects_changed_reference_poses(tmp_path):
    bank, command, _ = make_bank(tmp_path)
    command._motion_cache.current_batch.tensors["ref_dof_pos"][0, 40, 2] = 0.01
    with pytest.raises(ValueError, match="reference changed"):
        bank.select(command)
