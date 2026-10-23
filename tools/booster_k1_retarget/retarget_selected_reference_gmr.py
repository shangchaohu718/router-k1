#!/usr/bin/env python3
"""Retarget the selected human loop through the existing GMR booster_k1 solver.

The output is deliberately a provisional GMR-asset motion, not a Booster Train
motion-loader NPZ.  A final official-asset FK/name/axis audit is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import smplx
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import params as gmr_params
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast


DOF_NAMES = [
    "AAHead_yaw", "Head_pitch", "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll",
    "Left_Elbow_Pitch", "Left_Elbow_Yaw", "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Hip_Pitch",
    "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-xml", type=Path)
    parser.add_argument("--ik-config", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if bool(args.official_xml) != bool(args.ik_config):
        raise ValueError("--official-xml and --ik-config must be supplied together")
    if args.official_xml:
        gmr_params.ROBOT_XML_DICT["booster_k1"] = args.official_xml
        gmr_params.IK_CONFIG_DICT["smplx"]["booster_k1"] = args.ik_config

    with np.load(args.source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("betas", "root_orient", "pose_body", "trans", "target_fps")}
    if float(data["target_fps"]) != 50.0:
        raise ValueError("selected reference must be 50 Hz")
    count = len(data["trans"])
    if count != 500:
        raise ValueError(f"selected 10-second reference must contain 500 frames, got {count}")

    # This explicit setting is required by the available local SMPL-X asset;
    # GMR's stock loader defaults to 16 shape coefficients and fails here.
    body_model = smplx.create(
        args.body_model_root, "smplx", gender="neutral", use_pca=False,
        num_betas=10, num_expression_coeffs=10,
    )
    output = body_model(
        betas=torch.tensor(data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(data["root_orient"]).float(),
        body_pose=torch.tensor(data["pose_body"]).float(),
        transl=torch.tensor(data["trans"]).float(),
        left_hand_pose=torch.zeros(count, 45), right_hand_pose=torch.zeros(count, 45),
        jaw_pose=torch.zeros(count, 3), leye_pose=torch.zeros(count, 3),
        reye_pose=torch.zeros(count, 3), return_full_pose=True,
    )
    frames, fps = get_smplx_data_offline_fast(
        {"mocap_frame_rate": torch.tensor(50), "pose_body": data["pose_body"]},
        body_model, output, tgt_fps=50,
    )
    retargeter = GMR(
        actual_human_height=float(1.66 + 0.1 * data["betas"][0]),
        src_human="smplx", tgt_robot="booster_k1",
    )
    qpos = np.asarray([retargeter.retarget(frame) for frame in frames], dtype=np.float32)
    if qpos.shape != (count, 29) or not np.isfinite(qpos).all():
        raise RuntimeError(f"invalid GMR output {qpos.shape}; finite={np.isfinite(qpos).all()}")
    metadata = {
        "schema_version": "k1-moonwalk-gmr-provisional-v1",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "retargeter": "existing GMR booster_k1 / smplx_to_k1.json",
        "fps": float(fps),
        "frames": count,
        "root_quaternion_order": "wxyz from GMR qpos",
        "dof_names": DOF_NAMES,
        "asset_status": str(args.official_xml.resolve()) if args.official_xml else "GMR K1_serial.xml only; not validated against final official Booster Train asset",
        "not_ready_for_ppo": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_joint_names = DOF_NAMES
    if args.official_xml:
        output_joint_names = [
            "aahead_yaw_joint", "aahead_pitch_joint", "aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_elbow_pitch_joint", "left_elbow_yaw_joint", "aaright_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_elbow_pitch_joint", "right_elbow_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
            "left_hip_yaw_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_pitch_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
    np.savez_compressed(
        args.output,
        root_pos=qpos[:, :3], root_quat_wxyz=qpos[:, 3:7], joint_pos=qpos[:, 7:],
        joint_names=np.asarray(output_joint_names), metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({"output": str(args.output), "frames": count, "fps": fps, "joint_shape": list(qpos[:, 7:].shape)}, indent=2))


if __name__ == "__main__":
    main()
