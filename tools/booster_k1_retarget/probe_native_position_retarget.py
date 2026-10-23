"""Isolated native-position / cross-leg collision IK experiment, no source writes."""

import argparse
from collections import OrderedDict
import copy
import json
from pathlib import Path
import sys

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from holosmpl_h5_to_k1_h5 import (
    HoloSmplToGmrConverter,
    create_k1_retargeter,
    initialize_k1_reference_root,
    K1_DOF_NAMES,
)
from holosmpl.converters.bvh_family.cmu_asf import load_cmu_fk
from holosmpl.converters.bvh_family.style100 import YUP_TO_ZUP
from holosmpl.converters.bvh_family.noitom import load_bvh_motion
from holosmpl.converters.bvh_family.lafan1 import (
    _local_rotations_and_positions,
    _forward_kinematics,
    convert_lafan1_sample,
)
from holomotion.src.training.data_production.robot_h5 import RobotH5ShardWriter


def cross_leg_depth(data, left_geoms, right_geoms):
    return max(
        (
            max(0.0, -float(c.dist))
            for c in data.contact[: data.ncon]
            if (c.geom1 in left_geoms and c.geom2 in right_geoms)
            or (c.geom2 in left_geoms and c.geom1 in right_geoms)
        ),
        default=0.0,
    )


def collision_safe_step(
    model, data, previous, proposed, left_geoms, right_geoms, tolerance=0.001
):
    """Backtrack nonlinear IK overshoot; emitted poses keep the original geometry.

    Large amounts of backtracking must undergo a subsequent fidelity review.
    """
    data.qpos[:] = previous
    mujoco.mj_forward(model, data)
    if cross_leg_depth(data, left_geoms, right_geoms) > tolerance:
        raise ValueError(
            "IK backtracking requires a collision-free previous pose"
        )
    tangent = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, tangent, 1.0, previous, proposed)
    for attempt in range(15):
        alpha = 0.5**attempt
        candidate = previous.copy()
        mujoco.mj_integratePos(model, candidate, tangent, alpha)
        data.qpos[:] = candidate
        mujoco.mj_forward(model, data)
        if cross_leg_depth(data, left_geoms, right_geoms) <= tolerance:
            return candidate, alpha
    return previous.copy(), 0.0


def native_positions(metadata, canonical_metadata):
    path = Path(metadata["source_path"])
    if path.suffix == ".amc":
        _, start, _, _ = load_cmu_fk(path)
        mapping = {"pelvis": "root", "head": "head"}
        for side, short in (("left", "l"), ("right", "r")):
            mapping.update(
                {
                    f"{side}_{human}": short + bone
                    for human, bone in [
                        ("hip", "femur"),
                        ("knee", "tibia"),
                        ("ankle", "foot"),
                        ("foot", "toes"),
                        ("shoulder", "humerus"),
                        ("elbow", "radius"),
                    ]
                }
            )
        positions = {
            key: YUP_TO_ZUP.apply(start[value])
            for key, value in mapping.items()
        }
        offset = 0
    elif path.suffix.lower() == ".bvh":
        motion = load_bvh_motion(path)
        lr, lp = _local_rotations_and_positions(motion)
        _, pos = _forward_kinematics(motion, lr, lp)
        mapping = {"pelvis": "Hips", "head": "Head"}
        lafan = "LeftUpLeg" in pos
        for side, prefix in (("left", "Left"), ("right", "Right")):
            mapping.update(
                {
                    f"{side}_{human}": prefix + bone
                    for human, bone in [
                        ("hip", "UpLeg" if lafan else "Hip"),
                        ("knee", "Leg" if lafan else "Knee"),
                        ("ankle", "Foot" if lafan else "Ankle"),
                        ("foot", "Toe"),
                        ("shoulder", "Arm" if lafan else "Shoulder"),
                        ("elbow", "ForeArm" if lafan else "Elbow"),
                    ]
                }
            )
        positions = {
            key: YUP_TO_ZUP.apply(pos[value]) * 0.01
            for key, value in mapping.items()
        }
        offset = canonical_metadata.get("trim_start", 0)
    else:
        raise ValueError(f"No native source adapter for {path}")
    return positions, offset


def canonical_arrays(metadata):
    """Load or regenerate human arrays without touching an existing source."""
    if "canonical_path" in metadata:
        with np.load(
            metadata["canonical_path"], allow_pickle=False
        ) as canonical:
            root, body, trans = (
                canonical[name]
                for name in ("root_orient", "pose_body", "trans")
            )
            cm = json.loads(str(canonical["metadata"]))
    elif metadata.get("dataset") == "LaFAN1":
        path = Path(metadata["source_path"])
        canonical = convert_lafan1_sample(
            path, input_root=path.parent, target_fps=50
        )
        root, body, trans = (
            canonical[name] for name in ("root_orient", "pose_body", "trans")
        )
        cm = canonical["metadata"]
    else:
        raise ValueError("Missing canonical source")
    if body.shape[1] == 63:
        body = np.pad(body, ((0, 0), (0, 6)))
    if body.shape[1] != 69:
        raise ValueError(f"Unexpected canonical pose layout: {body.shape}")
    return root, body, trans, cm


def canonical_start(metadata):
    if "canonical_frame_start" in metadata:
        return int(metadata["canonical_frame_start"])
    return int(metadata.get("source_start_frame_trim", 0)) + int(
        metadata.get("source_frame_start", 0)
    )


def proportion_lower_body(human, retargeter, scale, rest_data):
    """Robot-sized hip spacing/bone lengths, preserving human limb directions.

    Root translation follows scaled native hip center. The trunk is not the
    human pelvis joint: account for the robot trunk-to-hip joint offset.
    No floor pinning, collision-geometry changes, or per-frame joint clipping.
    """
    model = retargeter.model
    trunk = rest_data.xpos[model.body("trunk").id]
    offsets = {
        s: rest_data.xpos[model.body(s + "_hip_yaw_link").id] - trunk
        for s in ("left", "right")
    }
    root_rotation = (
        Rotation.from_quat(human["pelvis"][1], scalar_first=True)
        * retargeter.rot_offsets1["pelvis"]
    )
    old_root = human["pelvis"][0].copy()
    hip_center = np.mean(
        [human[s + "_hip"][0] for s in ("left", "right")], axis=0
    )
    new_root = hip_center * scale - root_rotation.apply(
        np.mean(list(offsets.values()), axis=0)
    )
    directions = {
        s: [
            human[s + "_knee"][0] - human[s + "_hip"][0],
            human[s + "_ankle"][0] - human[s + "_knee"][0],
        ]
        for s in ("left", "right")
    }
    for value in human.values():
        value[0] += new_root / scale - old_root
    for side in ("left", "right"):
        hip = new_root + root_rotation.apply(offsets[side])
        p = [
            rest_data.xpos[model.body(side + "_" + name + "_link").id]
            for name in ("hip_yaw", "knee_pitch", "ankle_pitch")
        ]
        d = directions[side]
        if any(np.linalg.norm(v) < 1e-6 for v in d):
            raise ValueError("Degenerate human leg bone")
        knee = hip + d[0] / np.linalg.norm(d[0]) * np.linalg.norm(p[1] - p[0])
        ankle = knee + d[1] / np.linalg.norm(d[1]) * np.linalg.norm(
            p[2] - p[1]
        )
        for name, value in (("hip", hip), ("knee", knee), ("ankle", ankle)):
            human[side + "_" + name][0] = value / scale
    return human


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.input / "manifest.json").read_text())
    converter = HoloSmplToGmrConverter(args.smpl_model)
    writer = RobotH5ShardWriter(
        args.output / "shards/k1_000000.h5", dof_pos_dim=22
    )
    clips = {}
    reports = []
    cache = OrderedDict()
    for key, original in source["clips"].items():
        if not any(key.startswith(prefix) for prefix in args.keys):
            continue
        metadata = copy.deepcopy(original["metadata"])
        cache_key = (metadata.get("canonical_path"), metadata["source_path"])
        if cache_key not in cache:
            root, body, trans, cm = canonical_arrays(metadata)
            native, offset = (
                native_positions(metadata, cm) if args.native else (None, 0)
            )
            cache[cache_key] = (root, body, trans, cm, native, offset)
            if len(cache) > 8:
                cache.popitem(last=False)
        root, body, trans, cm, native, offset = cache[cache_key]
        cache.move_to_end(cache_key)
        retargeter = create_k1_retargeter(actual_human_height=1.8)
        rest_data = mujoco.MjData(retargeter.model)
        mujoco.mj_forward(retargeter.model, rest_data)
        scale = None
        if args.anatomical_scale:
            # True SMPL sources (e.g. AIST++) have no BVH/ASF skeleton.
            # Measure their canonical FK bones, explicitly recording this origin.
            if native is None:
                first = converter(
                    pose_aa=np.r_[root[0], body[0]], root_trans=trans[0]
                )
                scale_positions = {
                    name: np.asarray(value[0])[None]
                    for name, value in first.items()
                }
            else:
                scale_positions = native
            robot_lengths = []
            human_lengths = []
            for side in ("left", "right"):
                points = [
                    retargeter.configuration.data.xpos[
                        retargeter.model.body(side + "_" + name).id
                    ].copy()
                    for name in (
                        "hip_yaw_link"
                        if args.proportional_skeleton
                        else "hip_roll_link",
                        "knee_pitch_link",
                        "ankle_pitch_link",
                    )
                ]
                robot_lengths.append(
                    sum(
                        np.linalg.norm(points[j + 1] - points[j])
                        for j in (0, 1)
                    )
                )
                human_lengths.append(
                    np.median(
                        np.linalg.norm(
                            scale_positions[side + "_knee"]
                            - scale_positions[side + "_hip"],
                            axis=1,
                        )
                        + np.linalg.norm(
                            scale_positions[side + "_ankle"]
                            - scale_positions[side + "_knee"],
                            axis=1,
                        )
                    )
                )
            scale = float(np.mean(robot_lengths) / np.mean(human_lengths))
            if not 0.2 < scale < 1.0:
                raise ValueError(f"Implausible native/K1 leg scale: {scale}")
            retargeter.human_scale_table = {
                key: scale for key in retargeter.human_scale_table
            }
        check_data = mujoco.MjData(retargeter.model)
        if args.collision:
            groups = []
            for side in ("left", "right"):
                groups.append(
                    [
                        g
                        for g in range(retargeter.model.ngeom)
                        if retargeter.model.geom_contype[g]
                        and retargeter.model.body(
                            int(retargeter.model.geom_bodyid[g])
                        ).name.startswith(side + "_")
                        and any(
                            part
                            in retargeter.model.body(
                                int(retargeter.model.geom_bodyid[g])
                            ).name
                            for part in ("hip_", "knee_", "ankle_")
                        )
                    ]
                )
            retargeter.ik_limits.append(
                mink.CollisionAvoidanceLimit(
                    retargeter.model,
                    geom_pairs=[tuple(groups)],
                    minimum_distance_from_collisions=0.005,
                    collision_detection_distance=0.05,
                )
            )
        for table in (
            retargeter.human_body_to_task1,
            retargeter.human_body_to_task2,
        ):
            for side in ("left", "right"):
                table[f"{side}_foot"].set_orientation_cost(
                    args.foot_orientation_cost
                )
        trim = canonical_start(metadata)
        n = min(int(original["length"]), args.max_frames)
        if trim < 0 or trim + n > len(root):
            raise ValueError(
                f"Source frame span out of bounds: {key}: {trim}:{trim + n}/{len(root)}"
            )
        poses = []
        step_scales = []
        for frame in range(n):
            i = frame + trim
            human = converter(
                pose_aa=np.r_[root[i], body[i]], root_trans=trans[i]
            )
            if args.native:
                sample = i * float(metadata["source_fps"]) / 50 + offset
                lower = min(int(np.floor(sample)), len(native["pelvis"]) - 1)
                upper = min(lower + 1, len(native["pelvis"]) - 1)
                alpha = sample - np.floor(sample)
                for name, values in native.items():
                    human[name][0] = (
                        values[lower] * (1 - alpha) + values[upper] * alpha
                    )
            if args.proportional_skeleton:
                if scale is None:
                    raise ValueError(
                        "Proportional skeleton requires anatomical scale"
                    )
                human = proportion_lower_body(
                    human, retargeter, scale, rest_data
                )
            # Match robot ankle-link origins to anatomical ankles, not toe joints.
            for side in ("left", "right"):
                human[f"{side}_foot"] = [
                    np.array(v, copy=True) for v in human[f"{side}_ankle"]
                ]
            if frame == 0:
                initialize_k1_reference_root(retargeter, human)
            previous = retargeter.configuration.data.qpos.copy()
            proposed = retargeter.retarget(human)
            if args.line_search:
                if not args.collision:
                    raise ValueError("line-search requires collision groups")
                proposed, alpha = collision_safe_step(
                    retargeter.model,
                    check_data,
                    previous,
                    proposed,
                    set(groups[0]),
                    set(groups[1]),
                )
                retargeter.configuration.update(proposed)
                step_scales.append(alpha)
            poses.append(proposed)
        qpos = np.asarray(poses, dtype=np.float32)
        if not np.isfinite(qpos).all():
            raise ValueError(f"Non-finite IK: {key}")
        metadata.update(
            {
                "mapping_probe": "robot_proportioned_source_ik_v3"
                if args.proportional_skeleton
                else "native_positions_ankle_targets_crossleg_collision_v1",
                "quality_status": "unreviewed_mapping_candidate",
                "num_frames": n,
                "motion_fps": 50.0,
                "native_positions": args.native,
                "anatomical_leg_scale": scale,
                "proportional_lower_body": args.proportional_skeleton,
                "canonical_frame_start": trim,
                "canonical_frame_count": n,
                "anatomical_scale_source": "native_source_fk"
                if args.native
                else "canonical_neutral_smpl_fk",
                "crossleg_collision": args.collision,
                "foot_orientation_cost": args.foot_orientation_cost,
                "nonlinear_collision_line_search": args.line_search,
                "backtracked_frame_fraction": float(
                    np.mean(np.asarray(step_scales) < 1.0)
                )
                if step_scales
                else 0.0,
                "mapping_probe_source_root": str(args.input.resolve()),
                "ik_initialization": "calibrated_first_reference_root_v1",
            }
        )
        start, length = writer.append_motion(
            motion_id=len(clips),
            arrays={
                "ref_root_pos": qpos[:, :3],
                "ref_root_rot": qpos[:, [4, 5, 6, 3]],
                "ref_dof_pos": qpos[:, 7:],
            },
            metadata_json=json.dumps(metadata),
        )
        entry = copy.deepcopy(original)
        entry.update(
            shard=0,
            clip_idx=len(clips),
            start=start,
            length=length,
            metadata=metadata,
        )
        clips[key] = entry
        reports.append({"motion_key": key, "frames": length})
        print(key, length, "done", flush=True)
    writer.finalize()
    manifest = {
        "version": 2,
        "dataset_id": args.output.name,
        "root": str(args.output.resolve()),
        "hdf5_shards": [
            {
                "file": "shards/k1_000000.h5",
                "num_clips": len(clips),
                "num_frames": sum(c["length"] for c in clips.values()),
            }
        ],
        "clips": clips,
        "motion_keys": list(clips),
        "dof_names": list(K1_DOF_NAMES),
        "array_names": ["ref_root_pos", "ref_root_rot", "ref_dof_pos"],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output / "probe_report.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--keys", nargs="+", required=True)
    p.add_argument("--max-frames", type=int, default=1000)
    p.add_argument("--native", action="store_true")
    p.add_argument("--collision", action="store_true")
    p.add_argument("--line-search", action="store_true")
    p.add_argument("--anatomical-scale", action="store_true")
    p.add_argument("--proportional-skeleton", action="store_true")
    p.add_argument("--foot-orientation-cost", type=float, default=0.5)
    run(p.parse_args())
