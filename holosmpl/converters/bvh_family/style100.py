"""100STYLE native T-pose frames to approximate SMPL; no LaFAN offsets.

The source uses X-left/Y-up/Z-forward with a common T-pose joint basis.
Extra spine joints are composed in global space, not silently dropped.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from holosmpl.converters.bvh_family.lafan1 import (
    _forward_kinematics,
    _local_rotations_and_positions,
)
from holosmpl.converters.bvh_family.noitom import load_bvh_motion
from holosmpl.core.processing.resample import resample_motion_to_fps

PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
]
NAMES = [
    "Hips",
    "LeftHip",
    "RightHip",
    "Chest",
    "LeftKnee",
    "RightKnee",
    "Chest2",
    "LeftAnkle",
    "RightAnkle",
    "Chest4",
    "LeftToe",
    "RightToe",
    "Neck",
    "LeftCollar",
    "RightCollar",
    "Head",
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftWrist",
    "RightWrist",
]
YUP_TO_ZUP = Rotation.from_euler("x", 90, degrees=True)


def canonical_from_global(
    rotations, root_position, source_fps, target_fps, metadata
):
    count = len(root_position)
    root = (YUP_TO_ZUP * rotations[0]).as_rotvec()
    pose = np.zeros((count, 23, 3), dtype=np.float32)
    for j in range(1, 22):
        pose[:, j - 1] = (
            rotations[PARENTS[j]].inv() * rotations[j]
        ).as_rotvec()
    trans = YUP_TO_ZUP.apply(root_position)
    root, pose, trans, resample = resample_motion_to_fps(
        root_orient=root,
        pose_body=pose.reshape(count, 69),
        trans=trans,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    if not all(np.isfinite(a).all() for a in (root, pose, trans)):
        raise ValueError("Nonfinite canonical arrays")
    return dict(
        root_orient=root,
        pose_body=pose,
        trans=trans,
        betas=np.zeros(16, dtype=np.float32),
        gender="neutral",
        source_fps=source_fps,
        target_fps=target_fps,
        metadata={
            **metadata,
            **resample,
            "schema_version": "canonical_smpl_v1",
            "pose_body_dim": 69,
            "pose_body_layout": "smpl_23_body",
            "canonical_coordinate_system": "canonical_z_up_xyz_m",
            "source_fps": source_fps,
            "target_fps": target_fps,
            "unit": "m",
            "up_axis": "z",
            "world_gravity": [0, 0, -1],
            "is_strict_smpl": False,
            "root_frame_certified": False,
            "betas_policy": "neutral_zero_not_fitted",
            "approximation_warning": "Skeleton rotation transfer, not fitted SMPL shape; requires visual and robot QA.",
        },
    )


def trim_range(path):
    path = Path(path)
    style, movement = path.stem.rsplit("_", 1)
    table = next(
        (
            p / "Frame_Cuts.csv"
            for p in path.parents
            if (p / "Frame_Cuts.csv").is_file()
        ),
        None,
    )
    if table is None:
        raise ValueError(
            "Official Frame_Cuts.csv must exist in an ancestor folder"
        )
    with table.open(encoding="utf-8-sig", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["STYLE_NAME"] == style)
    return (
        int(row[movement + "_START"]),
        int(row[movement + "_STOP"]),
        style,
        movement,
    )


def classify_style100_source(source_path):
    try:
        trim_range(source_path)
        return {"status": "convertible_bvh", "source_path": str(source_path)}
    except (ValueError, StopIteration, KeyError) as exc:
        return {"status": "unsupported_missing_trim", "reason": str(exc)}


def convert_style100_sample(source_path, *, input_root, target_fps):
    source_path = Path(source_path)
    motion = load_bvh_motion(source_path)
    start, stop, style, movement = trim_range(source_path)
    if not 0 <= start < stop <= len(motion.values):
        raise ValueError("Invalid official trim range")
    local_r, local_p = _local_rotations_and_positions(motion)
    global_r, global_p = _forward_kinematics(motion, local_r, local_p)
    return canonical_from_global(
        [global_r[n][start:stop] for n in NAMES],
        global_p["Hips"][start:stop] * 0.01,
        1 / motion.frame_time,
        target_fps,
        {
            "dataset": "100STYLE",
            "source_format": "bvh",
            "source_path": str(source_path),
            "source_relative_path": source_path.relative_to(
                input_root
            ).as_posix(),
            "style": style,
            "movement": movement,
            "trim_start": start,
            "trim_stop": stop,
            "trim_policy": "official_indices_zero_based_half_open",
            "source_coordinate_system": "x_left_y_up_z_forward_cm",
            "body_pose_policy": "native_Tpose_global_rotations_to_SMPL_parents",
        },
    )
