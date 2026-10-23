from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget.holosmpl_h5_to_k1_h5 import K1_DOF_NAMES
from tools.booster_k1_retarget.probe_native_position_retarget import (
    canonical_start,
    proportion_lower_body,
)
from tools.booster_k1_retarget.salvage_repaired_reference import (
    classify_frames,
)


def signals(n=100):
    return dict(
        q=np.zeros((n, 22)),
        limits=np.tile([-1.0, 1.0], (22, 1)),
        sole_z=np.zeros((n, 2)),
        collision=np.zeros(n),
        source_z=np.zeros(n),
        direction_errors=np.zeros((n, 4)),
        quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
    )


def test_genuine_source_flight_is_not_rejected_as_false_air():
    args = signals()
    args["sole_z"][20:60] = 0.15
    args["source_z"][20:60] = 0.16
    assert not any(mask.any() for mask in classify_frames(**args).values())


def test_source_grounded_robot_floating_is_localized():
    args = signals()
    args["sole_z"][20:60] = 0.15
    mask = classify_frames(**args)["false_air_source_grounded_robot_over_8cm"]
    assert mask.sum() == 40
    assert mask[20] and not mask[19] and not mask[60]


def test_speed_outlier_marks_both_sides_of_transition():
    args = signals()
    args["q"][50:, 0] = 0.8
    mask = classify_frames(**args)["joint_speed_over_35_rad_s"]
    np.testing.assert_array_equal(np.flatnonzero(mask), [49, 50])


def test_prolonged_ankle_limit_not_extended_knee_or_brief_ankle_touch():
    args = signals()
    args["q"][:, K1_DOF_NAMES.index("left_knee_pitch_joint")] = 1.0
    args["q"][10:40, K1_DOF_NAMES.index("right_ankle_roll_joint")] = 1.0
    args["q"][70:80, K1_DOF_NAMES.index("left_ankle_roll_joint")] = 1.0
    mask = classify_frames(**args)["ankle_limit_saturation_for_0p5s"]
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(10, 40))


def test_explicit_canonical_start_survives_nested_grounding_and_salvage():
    assert (
        canonical_start(
            {"source_start_frame_trim": 10, "source_frame_start": 200}
        )
        == 210
    )
    assert (
        canonical_start(
            {
                "canonical_frame_start": 210,
                "source_start_frame_trim": 0,
                "source_frame_start": 200,
            }
        )
        == 210
    )


def test_root_teleport_is_rejected_but_quaternion_sign_change_is_not():
    args = signals()
    args["root_pos"] = np.zeros((100, 3))
    args["root_pos"][50:, 0] = 1
    args["quat"][60:] *= -1
    mask = classify_frames(**args)["root_speed_over_8m_s_or_20rad_s"]
    np.testing.assert_array_equal(np.flatnonzero(mask), [49, 50])


def test_large_one_sided_arm_mapping_error_is_localized():
    args = signals()
    args["arm_errors"] = np.zeros((100, 2))
    args["arm_errors"][60:70, 1] = 70
    mask = classify_frames(**args)["upper_arm_direction_over_60deg"]
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(60, 70))


def test_robot_proportions_preserve_source_leg_directions_and_hip_center():
    model = mujoco.MjModel.from_xml_path(
        str(
            Path(__file__).resolve().parents[1]
            / "assets/robots/booster_k1/K1_22dof.xml"
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = SimpleNamespace(
        model=model, rot_offsets1={"pelvis": Rotation.identity()}
    )
    human = {
        "pelvis": [np.array([1.0, 2.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])]
    }
    for side, y in (("left", 0.08), ("right", -0.08)):
        for joint, xyz in (
            ("hip", [1, 2 + y, 1]),
            ("knee", [1.1, 2 + y, 0.6]),
            ("ankle", [0.95, 2 + y, 0.2]),
        ):
            human[side + "_" + joint] = [
                np.array(xyz, dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0]),
            ]
    scale = 0.45
    source_center = (
        np.mean([human[s + "_hip"][0] for s in ("left", "right")], axis=0)
        * scale
    )
    result = proportion_lower_body(human, r, scale, data)
    actual_center = (
        np.mean([result[s + "_hip"][0] for s in ("left", "right")], axis=0)
        * scale
    )
    np.testing.assert_allclose(actual_center, source_center)
    actual_width = (
        np.linalg.norm(result["left_hip"][0] - result["right_hip"][0]) * scale
    )
    robot_width = np.linalg.norm(
        data.xpos[model.body("left_hip_yaw_link").id]
        - data.xpos[model.body("right_hip_yaw_link").id]
    )
    np.testing.assert_allclose(actual_width, robot_width)
    for side in ("left", "right"):
        points = [
            result[side + "_" + j][0] * scale for j in ("hip", "knee", "ankle")
        ]
        rest = [
            data.xpos[model.body(side + "_" + j + "_link").id]
            for j in ("hip_yaw", "knee_pitch", "ankle_pitch")
        ]
        for i, direction in enumerate(([0.1, 0, -0.4], [-0.15, 0, -0.4])):
            vector = points[i + 1] - points[i]
            np.testing.assert_allclose(
                vector / np.linalg.norm(vector),
                np.array(direction) / np.linalg.norm(direction),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                np.linalg.norm(vector), np.linalg.norm(rest[i + 1] - rest[i])
            )
    assert (
        canonical_start(
            {
                "canonical_frame_start": 210,
                "source_start_frame_trim": 0,
                "source_frame_start": 200,
            }
        )
        == 210
    )
