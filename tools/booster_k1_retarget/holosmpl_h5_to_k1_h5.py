#!/usr/bin/env python3

# 输出k1版本的HDF5供给Holomotion使用
"""Convert formal HoloSMPL HDF5 clips to Booster K1 robot HDF5 v2."""

from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]

K1_XML_PATH = REPO_ROOT / "assets" / "robots" / "booster_k1" / "K1_22dof.xml"

K1_IK_CONFIG_PATH = (
    REPO_ROOT
    / "tools"
    / "booster_k1_retarget"
    / "configs"
    / "gmr_smplx_to_official_k1_v2.json"
)

DEFAULT_SMPL_MODEL_PATH = (
    REPO_ROOT
    / "thirdparties"
    / "smpl_models"
    / "SMPL_python_v.1.1.0"
    / "smpl"
    / "models"
    / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
)

GMR_VENDOR_ROOT = (
    REPO_ROOT
    / "tools"
    / "booster_k1_retarget"
    / "vendor"
    / "general_motion_retargeting_k1"
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


class HoloSmplToGmrConverter:
    """Convert HoloSMPL frames into the body dictionary expected by GMR."""

    def __init__(self, smpl_model_path: Path) -> None:
        if not smpl_model_path.is_file():
            raise FileNotFoundError(f"SMPL model not found: {smpl_model_path}")

        repo_path = str(REPO_ROOT)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from holoretarget._engine_impl import (
            SMPLX_BODY_NAMES,
            SmplSkeletonAdapter,
        )

        adapter = SmplSkeletonAdapter(smpl_model_path=smpl_model_path)

        self.body_names = tuple(SMPLX_BODY_NAMES)
        # 记录关节名字
        self.parents = np.asarray(
            adapter.smpl_parents,
            dtype=np.int32,
        )
        # 记住谁连着谁
        self.parent_offsets = np.asarray(
            adapter.smpl_rest_parent_offsets,
            dtype=np.float64,
        )
        # 关节之间相隔多远

        if len(self.body_names) != 24:
            raise ValueError(
                f"expected 24 SMPL body names, got {len(self.body_names)}"
            )

        if self.parents.shape != (24,):
            raise ValueError(
                f"expected SMPL parents shape (24,), got {self.parents.shape}"
            )

        if self.parent_offsets.shape != (24, 3):
            raise ValueError(
                f"expected SMPL offsets shape (24, 3), "
                f"got {self.parent_offsets.shape}"
            )

    def __call__(
        self,
        *,
        pose_aa: np.ndarray,
        root_trans: np.ndarray,
    ) -> dict[str, list[np.ndarray]]:
        """Convert one HoloSMPL frame to GMR human body data."""

        pose_aa = np.asarray(
            pose_aa,
            dtype=np.float64,
        ).reshape(-1)

        root_trans = np.asarray(
            root_trans,
            dtype=np.float64,
        ).reshape(-1)

        if pose_aa.shape != (72,):
            raise ValueError(
                f"pose_aa must have shape (72,), got {pose_aa.shape}"
            )

        if root_trans.shape != (3,):
            raise ValueError(
                f"root_trans must have shape (3,), got {root_trans.shape}"
            )

        if not np.isfinite(pose_aa).all():
            raise ValueError("pose_aa contains NaN or Inf")

        if not np.isfinite(root_trans).all():
            raise ValueError("root_trans contains NaN or Inf")

        local_rotations = Rotation.from_rotvec(pose_aa.reshape(24, 3))

        global_positions = np.zeros(
            (24, 3),
            dtype=np.float64,
        )
        global_rotations: list[Rotation] = []

        for joint_id in range(24):
            if joint_id == 0:
                global_positions[joint_id] = root_trans
                global_rotations.append(local_rotations[joint_id])
                continue

            parent_id = int(self.parents[joint_id])

            if parent_id < 0 or parent_id >= joint_id:
                raise ValueError(
                    f"invalid parent {parent_id} for SMPL joint {joint_id}"
                )

            parent_rotation = global_rotations[parent_id]

            global_positions[joint_id] = global_positions[
                parent_id
            ] + parent_rotation.apply(self.parent_offsets[joint_id])

            global_rotations.append(
                parent_rotation * local_rotations[joint_id]
            )

        human_body_data = {}

        for joint_id, body_name in enumerate(self.body_names):
            position = global_positions[joint_id].copy()

            rotation_wxyz = global_rotations[joint_id].as_quat(
                scalar_first=True
            )

            human_body_data[body_name] = [
                position,
                rotation_wxyz.copy(),
            ]

        return human_body_data


def load_gmr_modules():
    """Load the vendored General Motion Retargeting package."""

    vendor_path = str(GMR_VENDOR_ROOT)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

    from general_motion_retargeting import (  # type: ignore[import-not-found]  # type: ignore[import-not-found]
        GeneralMotionRetargeting,
        params,
    )

    return GeneralMotionRetargeting, params


def create_k1_retargeter(*, actual_human_height: float):
    """Create GMR using the official Booster K1 model and IK configuration."""

    if actual_human_height <= 0.0:
        raise ValueError("actual_human_height must be positive")

    if not K1_XML_PATH.is_file():
        raise FileNotFoundError(f"K1 XML not found: {K1_XML_PATH}")

    if not K1_IK_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"K1 IK configuration not found: {K1_IK_CONFIG_PATH}"
        )

    retargeter_cls, gmr_params = load_gmr_modules()

    gmr_params.ROBOT_XML_DICT["booster_k1"] = K1_XML_PATH
    gmr_params.IK_CONFIG_DICT["smplx"]["booster_k1"] = K1_IK_CONFIG_PATH

    return retargeter_cls(
        src_human="smplx",
        tgt_robot="booster_k1",
        actual_human_height=actual_human_height,
        solver="daqp",
        verbose=False,
    )


def validate_k1_retargeter(retargeter) -> tuple[str, ...]:
    """Validate the MuJoCo model dimensions and joint ordering."""

    model = retargeter.model

    if model.nq != K1_QPOS_DIM:
        raise ValueError(
            f"K1 qpos dimension mismatch: "
            f"expected {K1_QPOS_DIM}, got {model.nq}"
        )

    actual_dof_names = []

    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if joint_name is None:
            raise ValueError(f"K1 joint {joint_id} does not have a name")

        actual_dof_names.append(joint_name)

    actual_dof_names = tuple(actual_dof_names)

    if actual_dof_names != K1_DOF_NAMES:
        raise ValueError(
            "K1 joint order mismatch:\n"
            f"expected: {K1_DOF_NAMES}\n"
            f"actual:   {actual_dof_names}"
        )

    return actual_dof_names


def load_holosmpl_clip(
    *,
    input_root: Path,
    clip_index: int,
):
    """Load one clip from a formal HoloSMPL HDF5 dataset."""

    if clip_index < 0:
        raise ValueError("clip_index must not be negative")

    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"HoloSMPL manifest not found: {manifest_path}"
        )

    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from holomotion.src.training.data_production.robot_h5 import (
        iter_holosmpl_h5_clips,
    )

    for current_index, clip in enumerate(iter_holosmpl_h5_clips(input_root)):
        if current_index == clip_index:
            return clip

    raise IndexError(
        f"clip_index {clip_index} is outside the HoloSMPL dataset"
    )


def load_all_holosmpl_clips(*, input_root: Path) -> list:
    """Load all clips from a formal HoloSMPL HDF5 dataset."""

    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"HoloSMPL manifest not found: {manifest_path}"
        )

    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from holomotion.src.training.data_production.robot_h5 import (
        iter_holosmpl_h5_clips,
    )

    clips = list(iter_holosmpl_h5_clips(input_root))
    if not clips:
        raise ValueError(f"no HoloSMPL clips found under {input_root}")
    return clips


def initialize_k1_reference_root(retargeter, human_body_data):
    """Seed the free joint at the first target, without changing hinge joints.

    GMR's zero-yaw reset can converge to a wrong bounded arm branch while
    rotating the whole body into place. Use the already calibrated root target
    (including its quaternion offset), not the uncalibrated SMPL orientation.
    update_targets mutates its input, so never pass the caller's original dict.
    """
    copied = {
        name: [np.array(pos, copy=True), np.array(rot, copy=True)]
        for name, (pos, rot) in human_body_data.items()
    }
    retargeter.update_targets(copied)
    pos, quat = retargeter.scaled_human_data[retargeter.human_root_name]
    seed = retargeter.configuration.data.qpos.copy()
    seed[:3] = pos
    seed[3:7] = quat
    if not np.isfinite(seed).all() or not np.isclose(
        np.linalg.norm(quat), 1.0, atol=1e-5
    ):
        raise ValueError("Invalid calibrated K1 root seed")
    retargeter.configuration.update(seed)


def retarget_holosmpl_clip_to_k1(
    *,
    clip,
    retargeter,
    human_converter: HoloSmplToGmrConverter,
    max_frames: int,
) -> dict[str, np.ndarray]:
    """Retarget one HoloSMPL clip to K1 reference arrays."""

    if max_frames < 0:
        raise ValueError("max_frames must not be negative")

    frame_count = clip.frame_count

    if max_frames > 0:
        frame_count = min(frame_count, max_frames)

    if frame_count <= 0:
        raise ValueError("the selected clip does not contain frames")

    reference_qpos = np.empty(
        (frame_count, K1_QPOS_DIM),
        dtype=np.float32,
    )

    for frame_index in range(frame_count):
        human_body_data = human_converter(
            pose_aa=clip.pose_aa[frame_index],
            root_trans=clip.root_trans[frame_index],
        )
        if frame_index == 0:
            initialize_k1_reference_root(retargeter, human_body_data)
            clip.metadata["ik_initialization"] = (
                "calibrated_first_reference_root_v1"
            )

        qpos = np.asarray(
            retargeter.retarget(human_body_data),
            dtype=np.float32,
        )

        if qpos.shape != (K1_QPOS_DIM,):
            raise ValueError(
                f"GMR returned qpos shape {qpos.shape} "
                f"at frame {frame_index}; "
                f"expected ({K1_QPOS_DIM},)"
            )

        if not np.isfinite(qpos).all():
            raise ValueError(f"GMR returned NaN or Inf at frame {frame_index}")

        reference_qpos[frame_index] = qpos

    ref_root_pos = reference_qpos[:, :3]

    ref_root_rot_wxyz = reference_qpos[:, 3:7]

    ref_root_rot_xyzw = ref_root_rot_wxyz[
        :,
        [1, 2, 3, 0],
    ]

    ref_dof_pos = reference_qpos[:, 7:]

    return {
        "ref_root_pos": ref_root_pos.astype(
            np.float32,
            copy=False,
        ),
        "ref_root_rot": ref_root_rot_xyzw.astype(
            np.float32,
            copy=False,
        ),
        "ref_dof_pos": ref_dof_pos.astype(
            np.float32,
            copy=False,
        ),
    }


def _k1_clip_metadata(clip, *, frame_count: int) -> dict:
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "schema_version": "gmr_k1_robot_h5_clip_v1",
            "num_frames": int(frame_count),
            "motion_fps": float(metadata.get("target_fps", 50.0)),
            "retarget_backend": "GMR",
            "target_robot": "booster_k1",
            "source_holosmpl_shard": str(clip.shard_path),
            "source_holosmpl_clip_index": int(clip.clip_index),
        }
    )
    return metadata


def write_k1_robot_h5(
    *,
    output_root: Path,
    clip,
    arrays: dict[str, np.ndarray],
) -> Path:
    """Write one retargeted K1 clip as robot HDF5 v2."""

    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")

    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from holomotion.src.training.data_production.robot_h5 import (
        RobotH5ShardWriter,
    )

    shard_path = output_root / "shards" / "gmr_k1_000000.h5"

    writer = RobotH5ShardWriter(
        shard_path,
        compression="lzf",
        dof_pos_dim=K1_DOF_DIM,
    )

    frame_count = int(arrays["ref_dof_pos"].shape[0])

    metadata = _k1_clip_metadata(clip, frame_count=frame_count)

    start, length = writer.append_motion(
        motion_id=0,
        arrays=arrays,
        metadata_json=json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
        ),
    )

    shard_summary = writer.finalize()

    relative_shard_path = shard_path.relative_to(output_root).as_posix()

    array_names = [
        "ref_root_pos",
        "ref_root_rot",
        "ref_dof_pos",
    ]

    manifest = {
        "version": 2,
        "root": str(output_root),
        "hdf5_shards": [
            {
                "file": relative_shard_path,
                "num_clips": 1,
                "num_frames": int(shard_summary["num_frames"]),
            }
        ],
        "clips": {
            clip.motion_key: {
                "motion_key": clip.motion_key,
                "shard": 0,
                "clip_idx": 0,
                "start": int(start),
                "length": int(length),
                "available_arrays": array_names,
                "metadata": metadata,
            }
        },
        "motion_keys": [clip.motion_key],
        "dof_names": list(K1_DOF_NAMES),
        "array_names": array_names,
        "chunks_t": 1024,
        "compression": "lzf",
        "source_schema": "formal_h5_manifest_v1",
        "source_holosmpl_h5": str(clip.shard_path),
        "retarget_backend": "GMR",
        "target_robot": "booster_k1",
        "retarget_config": {
            "robot_xml": str(K1_XML_PATH),
            "ik_config": str(K1_IK_CONFIG_PATH),
        },
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    nan_paths = output_root / "nan_npz_paths.json"
    nan_paths.write_text(
        "[]\n",
        encoding="utf-8",
    )

    return manifest_path


def write_k1_robot_h5_dataset(
    *,
    output_root: Path,
    input_root: Path,
    clips: list,
    retargeter,
    human_converter: HoloSmplToGmrConverter,
    max_frames: int,
    overwrite: bool,
    shard_target_frames: int,
) -> Path:
    """Retarget and write multiple HoloSMPL clips as one K1 dataset."""

    if shard_target_frames <= 0:
        raise ValueError("shard_target_frames must be positive")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"output root already exists: {output_root}; "
                "pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    from holomotion.src.training.data_production.robot_h5 import (
        RobotH5ShardWriter,
    )

    shard_root = output_root / "shards"
    hdf5_shards: list[dict] = []
    clips_manifest: dict[str, dict] = {}
    motion_keys: list[str] = []
    writer = None
    shard_index = 0
    total_frames = 0
    started_at = time.monotonic()

    def finalize_writer(current_writer) -> None:
        summary = current_writer.finalize()
        hdf5_shards.append(
            {
                "file": current_writer.path.relative_to(
                    output_root
                ).as_posix(),
                "num_clips": int(summary["num_clips"]),
                "num_frames": int(summary["num_frames"]),
            }
        )

    for clip_number, clip in enumerate(clips, start=1):
        requested_frames = clip.frame_count
        if max_frames > 0:
            requested_frames = min(requested_frames, max_frames)

        if writer is None or (
            writer.clip_count > 0
            and writer.frame_count + requested_frames > shard_target_frames
        ):
            if writer is not None:
                finalize_writer(writer)
                shard_index += 1
            writer = RobotH5ShardWriter(
                shard_root / f"gmr_k1_{shard_index:06d}.h5",
                compression="lzf",
                dof_pos_dim=K1_DOF_DIM,
            )

        # Each dance is an independent sequence. Reset the IK state so the
        # final pose of one clip cannot leak into the next clip.
        retargeter.setup_retarget_configuration()
        retargeter.ground_offset = 0.0
        arrays = retarget_holosmpl_clip_to_k1(
            clip=clip,
            retargeter=retargeter,
            human_converter=human_converter,
            max_frames=max_frames,
        )
        frame_count = int(arrays["ref_dof_pos"].shape[0])
        metadata = _k1_clip_metadata(
            clip,
            frame_count=frame_count,
        )
        motion_id = len(motion_keys)
        start, length = writer.append_motion(
            motion_id=motion_id,
            arrays=arrays,
            metadata_json=json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        if clip.motion_key in clips_manifest:
            raise ValueError(f"duplicate motion key: {clip.motion_key}")
        motion_keys.append(clip.motion_key)
        clips_manifest[clip.motion_key] = {
            "motion_key": clip.motion_key,
            "shard": shard_index,
            "clip_idx": writer.clip_count - 1,
            "start": int(start),
            "length": int(length),
            "available_arrays": [
                "ref_root_pos",
                "ref_root_rot",
                "ref_dof_pos",
            ],
            "metadata": metadata,
        }
        total_frames += length
        elapsed = max(time.monotonic() - started_at, 1e-6)
        print(
            f"[gmr-k1] {clip_number}/{len(clips)} clips "
            f"| frames={total_frames} "
            f"| elapsed={elapsed:.1f}s",
            flush=True,
        )

    if writer is not None:
        finalize_writer(writer)

    array_names = ["ref_root_pos", "ref_root_rot", "ref_dof_pos"]
    manifest = {
        "version": 2,
        "root": str(output_root),
        "hdf5_shards": hdf5_shards,
        "clips": clips_manifest,
        "motion_keys": motion_keys,
        "dof_names": list(K1_DOF_NAMES),
        "array_names": array_names,
        "chunks_t": 1024,
        "compression": "lzf",
        "source_schema": "formal_h5_manifest_v1",
        "source_holosmpl_h5_root": str(input_root),
        "retarget_backend": "GMR",
        "target_robot": "booster_k1",
        "retarget_config": {
            "robot_xml": str(K1_XML_PATH),
            "ik_config": str(K1_IK_CONFIG_PATH),
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "nan_npz_paths.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert formal HoloSMPL HDF5 clips to Booster K1 robot HDF5 v2."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory of the formal HoloSMPL HDF5 dataset.",
    )

    parser.add_argument(
        "--clip-index",
        type=int,
        default=0,
        help="Zero-based clip index used when --all-clips is not set.",
    )

    parser.add_argument(
        "--all-clips",
        action="store_true",
        help="Retarget every clip into one sharded K1 HDF5 dataset.",
    )

    parser.add_argument(
        "--smpl-model",
        type=Path,
        default=DEFAULT_SMPL_MODEL_PATH,
        help="Path to the licensed neutral SMPL model PKL.",
    )

    parser.add_argument(
        "--human-height",
        type=float,
        default=1.8,
        help="Assumed human height in meters.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Maximum frames to retarget; use 0 to process the complete clip."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for the K1 robot HDF5 v2 dataset.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-root when it already exists.",
    )

    parser.add_argument(
        "--shard-target-frames",
        type=int,
        default=250_000,
        help="Approximate maximum frame count per output HDF5 shard.",
    )

    return parser.parse_args()


def main() -> None:
    """Initialize and validate the Booster K1 retargeter."""

    args = parse_args()

    # Fail fast on the licensed external asset before initializing MuJoCo/GMR.
    human_converter = HoloSmplToGmrConverter(
        smpl_model_path=args.smpl_model,
    )

    retargeter = create_k1_retargeter(
        actual_human_height=args.human_height,
    )

    dof_names = validate_k1_retargeter(retargeter)

    if args.all_clips:
        clips = load_all_holosmpl_clips(input_root=args.input_root)
        manifest_path = write_k1_robot_h5_dataset(
            output_root=args.output_root,
            input_root=args.input_root,
            clips=clips,
            retargeter=retargeter,
            human_converter=human_converter,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
            shard_target_frames=args.shard_target_frames,
        )
        print(f"K1 HDF5 manifest: {manifest_path}")
        print(f"Converted clips: {len(clips)}")
    else:
        clip = load_holosmpl_clip(
            input_root=args.input_root,
            clip_index=args.clip_index,
        )
        k1_arrays = retarget_holosmpl_clip_to_k1(
            clip=clip,
            retargeter=retargeter,
            human_converter=human_converter,
            max_frames=args.max_frames,
        )
        if args.output_root.exists() and args.overwrite:
            shutil.rmtree(args.output_root)
        manifest_path = write_k1_robot_h5(
            output_root=args.output_root,
            clip=clip,
            arrays=k1_arrays,
        )
        print(f"K1 HDF5 manifest: {manifest_path}")
        print(f"K1 root position shape: {k1_arrays['ref_root_pos'].shape}")
        print(f"K1 root rotation shape: {k1_arrays['ref_root_rot'].shape}")
        print(f"K1 joint position shape: {k1_arrays['ref_dof_pos'].shape}")
        print(f"Clip index: {args.clip_index}")
        print(f"Motion key: {clip.motion_key}")
        print(f"Motion frames: {clip.frame_count}")
        print(f"Pose shape: {clip.pose_aa.shape}")
        print(f"Root translation shape: {clip.root_trans.shape}")
        print(f"Shape beta shape: {clip.shape_beta.shape}")

    print(f"HoloSMPL input: {args.input_root}")
    print(f"K1 XML: {K1_XML_PATH}")
    print(f"K1 IK config: {K1_IK_CONFIG_PATH}")
    print(f"K1 qpos dimension: {K1_QPOS_DIM}")
    print(f"K1 controlled joints: {len(dof_names)}")

    for index, name in enumerate(dof_names):
        print(f"{index:02d}: {name}")


if __name__ == "__main__":
    main()
