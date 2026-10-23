#!/usr/bin/env python3
"""Retarget LaFAN1 BVH files to Booster K1 robot HDF5 v2."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GMR_VENDOR_ROOT = HERE / "vendor" / "general_motion_retargeting_k1"
K1_XML_PATH = REPO_ROOT / "assets" / "robots" / "booster_k1" / "K1_22dof.xml"
K1_BVH_CONFIG_PATH = HERE / "configs" / "bvh_lafan1_to_official_k1.json"

for package_root in (REPO_ROOT, GMR_VENDOR_ROOT):
    package_path = str(package_root)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from general_motion_retargeting import (  # noqa: E402
    GeneralMotionRetargeting,
    params as gmr_params,
)
from general_motion_retargeting.utils.lafan1 import (  # noqa: E402
    load_bvh_file,
)

from holomotion.src.training.data_production.robot_h5 import (  # noqa: E402
    RobotH5ShardWriter,
)

K1_DOF_NAMES = (
    "aahead_yaw_joint",
    "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
K1_DOF_DIM = len(K1_DOF_NAMES)
K1_QPOS_DIM = 7 + K1_DOF_DIM
ROBOT_ARRAY_NAMES = ("ref_root_pos", "ref_root_rot", "ref_dof_pos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-folder",
        "--src_folder",
        dest="src_folder",
        type=Path,
        required=True,
        help="Folder containing LaFAN1 BVH files.",
    )
    parser.add_argument(
        "--output-root",
        "--tgt_folder",
        dest="output_root",
        type=Path,
        required=True,
        help="New output root for the K1 robot HDF5 v2 dataset.",
    )
    parser.add_argument(
        "--ik-config",
        type=Path,
        default=K1_BVH_CONFIG_PATH,
        help=(
            "GMR BVH-to-K1 IK configuration. Defaults to the original "
            "mapping configuration."
        ),
    )
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument(
        "--target-fps", "--target_fps", type=float, default=50.0
    )
    parser.add_argument(
        "--pattern",
        default="*.bvh",
        help="BVH filename glob relative to the source root.",
    )
    parser.add_argument(
        "--subject-split",
        choices=("all", "train", "val"),
        default="all",
        help=(
            "LaFAN1 subject split: train keeps subjects 1-4, val keeps "
            "subject 5, and all keeps every subject."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most this many BVH files; 0 means all files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Use at most this many source frames per clip; 0 means all frames."
        ),
    )
    return parser.parse_args()


def configure_k1_gmr(ik_config_path: Path) -> None:
    if not K1_XML_PATH.is_file():
        raise FileNotFoundError(f"K1 XML not found: {K1_XML_PATH}")
    if not ik_config_path.is_file():
        raise FileNotFoundError(
            f"K1 BVH config not found: {ik_config_path}"
        )
    gmr_params.ROBOT_XML_DICT["booster_k1"] = K1_XML_PATH
    gmr_params.IK_CONFIG_DICT["bvh_lafan1"]["booster_k1"] = ik_config_path


def retarget_bvh(path: Path, *, max_frames: int) -> np.ndarray:
    human_frames, actual_human_height = load_bvh_file(
        str(path),
        format="lafan1",
    )
    if max_frames > 0:
        human_frames = human_frames[:max_frames]
    if not human_frames:
        raise ValueError(f"BVH contains no motion frames: {path}")

    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="booster_k1",
        actual_human_height=actual_human_height,
        solver="daqp",
        verbose=False,
    )
    qpos = np.asarray(
        [retargeter.retarget(frame) for frame in human_frames],
        dtype=np.float32,
    )
    if qpos.shape != (len(human_frames), K1_QPOS_DIM):
        raise ValueError(f"unexpected K1 qpos shape for {path}: {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"K1 qpos contains NaN or Inf: {path}")
    return qpos


def resample_qpos(
    qpos: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("source_fps and target_fps must be positive")
    if qpos.shape[0] <= 1 or np.isclose(source_fps, target_fps):
        return qpos.astype(np.float32, copy=True)

    duration = (qpos.shape[0] - 1) / source_fps
    target_frames = int(round(duration * target_fps)) + 1
    source_time = np.arange(qpos.shape[0], dtype=np.float64) / source_fps
    target_time = np.arange(target_frames, dtype=np.float64) / target_fps
    target_time[-1] = min(target_time[-1], source_time[-1])

    output = np.empty((target_frames, qpos.shape[1]), dtype=np.float32)
    for axis in range(3):
        output[:, axis] = np.interp(
            target_time,
            source_time,
            qpos[:, axis],
        )

    root_xyzw = qpos[:, [4, 5, 6, 3]]
    root_rotation = Rotation.from_quat(root_xyzw)
    output_xyzw = Slerp(source_time, root_rotation)(target_time).as_quat()
    output[:, 3:7] = output_xyzw[:, [3, 0, 1, 2]]

    for axis in range(7, qpos.shape[1]):
        output[:, axis] = np.interp(
            target_time,
            source_time,
            qpos[:, axis],
        )
    return output


def qpos_to_robot_arrays(qpos: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ref_root_pos": qpos[:, :3].astype(np.float32, copy=False),
        "ref_root_rot": qpos[:, [4, 5, 6, 3]].astype(
            np.float32,
            copy=False,
        ),
        "ref_dof_pos": qpos[:, 7:].astype(np.float32, copy=False),
    }


def motion_key(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    return "__".join(relative.parts)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_root = args.src_folder.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"BVH source root not found: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if args.max_files < 0 or args.max_frames < 0:
        raise ValueError("max-files and max-frames must not be negative")

    bvh_paths = sorted(source_root.rglob(args.pattern))
    if args.subject_split == "train":
        bvh_paths = [
            path for path in bvh_paths if not path.stem.endswith("_subject5")
        ]
    elif args.subject_split == "val":
        bvh_paths = [
            path for path in bvh_paths if path.stem.endswith("_subject5")
        ]
    if args.max_files > 0:
        bvh_paths = bvh_paths[: args.max_files]
    if not bvh_paths:
        raise ValueError(f"no BVH files found under {source_root}")

    ik_config_path = args.ik_config.resolve()
    configure_k1_gmr(ik_config_path)
    writer = RobotH5ShardWriter(
        output_root / "shards" / "gmr_k1_000000.h5",
        compression="lzf",
        dof_pos_dim=K1_DOF_DIM,
    )
    clips: dict[str, dict[str, object]] = {}
    motion_keys: list[str] = []

    for path in tqdm(bvh_paths, desc="Retargeting LaFAN1 to K1"):
        key = motion_key(path, source_root)
        source_qpos = retarget_bvh(path, max_frames=args.max_frames)
        qpos = resample_qpos(
            source_qpos,
            source_fps=args.source_fps,
            target_fps=args.target_fps,
        )
        arrays = qpos_to_robot_arrays(qpos)
        metadata = {
            "schema_version": "gmr_k1_robot_h5_clip_v1",
            "dataset": "LaFAN1",
            "subject_split": args.subject_split,
            "source_format": "bvh",
            "source_path": str(path),
            "source_relative_path": path.relative_to(source_root).as_posix(),
            "source_fps": float(args.source_fps),
            "motion_fps": float(args.target_fps),
            "source_num_frames": int(source_qpos.shape[0]),
            "num_frames": int(qpos.shape[0]),
            "retarget_backend": "GMR",
            "target_robot": "booster_k1",
        }
        start, length = writer.append_motion(
            motion_id=len(motion_keys),
            arrays=arrays,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        motion_keys.append(key)
        clips[key] = {
            "motion_key": key,
            "shard": 0,
            "clip_idx": writer.clip_count - 1,
            "start": int(start),
            "length": int(length),
            "available_arrays": list(ROBOT_ARRAY_NAMES),
            "metadata": metadata,
        }

    shard_summary = writer.finalize()
    manifest = {
        "version": 2,
        "root": str(output_root),
        "hdf5_shards": [
            {
                "file": "shards/gmr_k1_000000.h5",
                "num_clips": int(shard_summary["num_clips"]),
                "num_frames": int(shard_summary["num_frames"]),
            }
        ],
        "clips": clips,
        "motion_keys": motion_keys,
        "dof_names": list(K1_DOF_NAMES),
        "array_names": list(ROBOT_ARRAY_NAMES),
        "chunks_t": 1024,
        "compression": "lzf",
        "source_schema": "lafan1_bvh",
        "subject_split": args.subject_split,
        "source_root": str(source_root),
        "retarget_backend": "GMR",
        "target_robot": "booster_k1",
        "target_fps": float(args.target_fps),
        "retarget_config": {
            "robot_xml": str(K1_XML_PATH),
            "ik_config": str(ik_config_path),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "nan_npz_paths.json", [])
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "num_clips": len(motion_keys),
                "num_frames": int(shard_summary["num_frames"]),
                "target_fps": float(args.target_fps),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
