"""Recover contiguous reference candidates, retaining every rejected interval.

This is a source-aware kinematic review, NOT simulation or training approval.
Only a constant clip-level floor shift is applied. No per-frame foot pinning,
joint clipping, mirroring, temporal stretching or stitching across bad frames.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from tools.booster_k1_retarget.holosmpl_h5_to_k1_h5 import (
    HoloSmplToGmrConverter,
    K1_DOF_NAMES,
)
from tools.booster_k1_retarget.probe_native_position_retarget import (
    canonical_arrays,
    canonical_start,
    native_positions,
)
from tools.booster_k1_retarget.localize_and_salvage import intervals, padded
from holomotion.src.training.data_production.robot_h5 import RobotH5ShardWriter

REPO = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def source_bundle(source_path, canonical_path, dataset, converter):
    metadata = {"source_path": source_path, "dataset": dataset}
    if canonical_path is not None:
        metadata["canonical_path"] = canonical_path
    root, body, trans, cm = canonical_arrays(metadata)
    native = Path(source_path).suffix.lower() in {".bvh", ".amc"}
    if native:
        positions, offset = native_positions(metadata, cm)
        origin = "native_skeleton_fk"
    else:
        full = [
            converter(pose_aa=np.r_[r, b], root_trans=p)
            for r, b, p in zip(root, body, trans)
        ]
        positions = {
            name: np.asarray([f[name][0] for f in full]) for name in full[0]
        }
        offset = 0
        origin = "canonical_neutral_smpl_fk_not_native_capture_contacts"
    ground = float(
        np.percentile(
            np.min(
                np.stack(
                    [
                        positions[s + "_" + j][:, 2]
                        for s in ("left", "right")
                        for j in ("ankle", "foot")
                    ]
                ),
                axis=0,
            ),
            1,
        )
    )
    return positions, ground, origin, len(root), offset, native


def source_positions(metadata, length, converter):
    positions, ground, origin, canonical_length, offset, native = (
        source_bundle(
            metadata["source_path"],
            metadata.get("canonical_path"),
            metadata.get("dataset"),
            converter,
        )
    )
    start = canonical_start(metadata)
    if start < 0 or start + length > canonical_length:
        raise ValueError("Human/robot frame correspondence out of bounds")
    if native:
        ratio = float(metadata["source_fps"]) / 50
        t = (start + np.arange(length)) * ratio + offset
        # Canonical resampling clips its rounded final sample to the source
        # endpoint. Mirror that documented behavior, not an arbitrary time warp.
        if t[0] < 0 or t[-1] > len(positions["pelvis"]) - 1 + max(1, ratio):
            raise ValueError(
                "Native source frame correspondence out of bounds"
            )
        samples = {
            name: np.stack(
                [
                    np.interp(t, np.arange(len(v)), v[:, axis])
                    for axis in range(3)
                ],
                axis=1,
            )
            for name, v in positions.items()
        }
    else:
        samples = {
            name: v[start : start + length] for name, v in positions.items()
        }
    return samples, ground, origin


def angular_error(a, b):
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if np.any(an < 1e-6) or np.any(bn < 1e-6):
        raise ValueError("Degenerate limb vector")
    return np.degrees(
        np.arccos(np.clip(np.sum(a * b, axis=-1) / (an * bn), -1, 1))
    )


def classify_frames(
    q,
    limits,
    sole_z,
    collision,
    source_z,
    direction_errors,
    quat,
    root_pos=None,
    arm_errors=None,
):
    speed_bad = np.any(np.abs(np.diff(q, axis=0)) * 50 > 35, axis=1)
    fast = np.zeros(len(q), dtype=bool)
    fast[:-1] |= speed_bad
    fast[1:] |= speed_bad
    ankle_saturation = np.zeros(len(q), dtype=bool)
    for j, name in enumerate(K1_DOF_NAMES):
        if "ankle" not in name:
            continue
        near = (
            np.minimum(q[:, j] - limits[j, 0], limits[j, 1] - q[:, j]) < 0.005
        )
        for s, e in intervals(near):
            if e - s >= 25:
                ankle_saturation[s:e] = True
    root_jump = np.zeros(len(q), dtype=bool)
    if root_pos is not None:
        linear = np.linalg.norm(np.diff(root_pos, axis=0), axis=1) * 50
        angular = (
            2
            * np.arccos(
                np.clip(np.abs(np.sum(quat[1:] * quat[:-1], axis=1)), 0, 1)
            )
            * 50
        )
        root_jump[:-1] |= (linear > 8) | (angular > 20)
        root_jump[1:] |= (linear > 8) | (angular > 20)
    return {
        "root_speed_over_8m_s_or_20rad_s": root_jump,
        "upper_arm_direction_over_60deg": np.max(arm_errors, axis=1) > 60
        if arm_errors is not None
        else np.zeros(len(q), dtype=bool),
        "joint_speed_over_35_rad_s": fast,
        "joint_limit_violation_over_0p01_rad": np.any(
            (q < limits[:, 0] - 0.01) | (q > limits[:, 1] + 0.01), axis=1
        ),
        "opposing_leg_collision_over_5mm": collision > 0.005,
        "ankle_limit_saturation_for_0p5s": ankle_saturation,
        "floor_penetration_over_2cm": sole_z.min(axis=1) < -0.02,
        # Genuine source flights are explicitly not rejected here.
        "false_air_source_grounded_robot_over_8cm": (source_z < 0.025)
        & (sole_z.min(axis=1) > 0.08),
        "leg_direction_p75_over_30deg": np.percentile(
            direction_errors, 75, axis=1
        )
        > 30,
        "root_tilt_over_55deg_complex_review": (
            1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
        )
        < np.cos(np.deg2rad(55)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--min-frames", type=int, default=100)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((a.input / "manifest.json").read_text())
    if source["dof_names"] != list(K1_DOF_NAMES):
        raise ValueError("DOF order mismatch")
    converter = HoloSmplToGmrConverter(a.smpl_model)
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "assets/robots/booster_k1/K1_22dof.xml")
    )
    data = mujoco.MjData(model)
    ids = [
        i
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    limits = model.jnt_range[ids]
    bodies = {
        s + "_" + j: model.body(s + "_" + j + "_link").id
        for s in ("left", "right")
        for j in (
            "hip_roll",
            "knee_pitch",
            "ankle_pitch",
            "shoulder_roll",
            "elbow_yaw",
        )
    }
    groups = [
        {
            g
            for g in range(model.ngeom)
            if model.geom_contype[g]
            and model.body(int(model.geom_bodyid[g])).name.startswith(s + "_")
            and any(
                t in model.body(int(model.geom_bodyid[g])).name
                for t in ("hip_", "knee_", "ankle_")
            )
        }
        for s in ("left", "right")
    ]
    foot_geoms = [
        next(
            g
            for g in range(model.ngeom)
            if model.geom_contype[g]
            and model.geom_bodyid[g] == model.body(s + "_ankle_roll_link").id
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        )
        for s in ("left", "right")
    ]
    categories = (
        "01_kinematic_candidates",
        "80_short_or_source_review",
        "90_rejected_intervals",
    )
    writers = {
        name: RobotH5ShardWriter(
            a.output / name / "shards/k1_000000.h5", dof_pos_dim=22
        )
        for name in categories
    }
    manifests = {
        name: {
            "version": 2,
            "dataset_id": a.output.name + "_" + name,
            "dof_names": list(K1_DOF_NAMES),
            "array_names": source["array_names"],
            "clips": {},
            "motion_keys": [],
            "training_approved": False,
        }
        for name in categories
    }
    reports = []
    for key, clip in source["clips"].items():
        n, start = int(clip["length"]), int(clip["start"])
        with h5py.File(
            a.input / source["hdf5_shards"][clip["shard"]]["file"], "r"
        ) as f:
            arrays = {
                name: f[name][start : start + n]
                for name in source["array_names"]
            }
        pos, quat, q = (
            arrays[k] for k in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
        )
        if (
            n < 2
            or not all(np.isfinite(v).all() for v in arrays.values())
            or np.any(np.abs(np.linalg.norm(quat, axis=1) - 1) > 1e-3)
        ):
            raise ValueError(f"Invalid input arrays: {key}")
        if not np.isclose(clip["metadata"].get("motion_fps", 50), 50):
            raise ValueError("Only 50 Hz input is supported")
        sole_z, collision = np.empty((n, 2)), np.zeros(n)
        robot = {name: np.empty((n, 3)) for name in bodies}
        for frame in range(n):
            data.qpos[:] = np.r_[
                pos[frame], quat[frame, [3, 0, 1, 2]], q[frame]
            ]
            mujoco.mj_forward(model, data)
            for side, g in enumerate(foot_geoms):
                sole_z[frame, side] = (
                    data.geom_xpos[g, 2]
                    - np.abs(data.geom_xmat[g].reshape(3, 3)[2])
                    @ model.geom_size[g]
                )
            for name, bid in bodies.items():
                robot[name][frame] = data.xpos[bid]
            collision[frame] = max(
                [
                    max(0, -float(c.dist))
                    for c in data.contact[: data.ncon]
                    if (c.geom1 in groups[0] and c.geom2 in groups[1])
                    or (c.geom2 in groups[0] and c.geom1 in groups[1])
                ],
                default=0,
            )
        shift = float(np.percentile(sole_z.min(axis=1), 1))
        pos[:, 2] -= shift
        sole_z -= shift
        error = None
        try:
            human, ground, origin = source_positions(
                clip["metadata"], n, converter
            )
            scale = clip["metadata"].get("anatomical_leg_scale")
            if scale is None:
                lengths = [
                    np.linalg.norm(
                        human[s + "_knee"] - human[s + "_hip"], axis=1
                    )
                    + np.linalg.norm(
                        human[s + "_ankle"] - human[s + "_knee"], axis=1
                    )
                    for s in ("left", "right")
                ]
                scale = 0.410702 / float(np.median(lengths))
            hz = (
                np.min(
                    np.stack(
                        [
                            human[s + "_" + j][:, 2]
                            for s in ("left", "right")
                            for j in ("ankle", "foot")
                        ]
                    ),
                    axis=0,
                )
                - ground
            ) * scale
            angles = np.stack(
                [
                    angular_error(
                        robot[s + "_" + r2] - robot[s + "_" + r1],
                        human[s + "_" + h2] - human[s + "_" + h1],
                    )
                    for s in ("left", "right")
                    for r1, r2, h1, h2 in (
                        ("hip_roll", "knee_pitch", "hip", "knee"),
                        ("knee_pitch", "ankle_pitch", "knee", "ankle"),
                    )
                ],
                axis=1,
            )
            arm_angles = np.stack(
                [
                    angular_error(
                        robot[s + "_elbow_yaw"] - robot[s + "_shoulder_roll"],
                        human[s + "_elbow"] - human[s + "_shoulder"],
                    )
                    for s in ("left", "right")
                ],
                axis=1,
            )
            reasons = classify_frames(
                q,
                limits,
                sole_z,
                collision,
                hz,
                angles,
                quat,
                root_pos=pos,
                arm_errors=arm_angles,
            )
            bad = np.logical_or.reduce(list(reasons.values()))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            error = str(exc)
            origin = "unavailable_hold_all"
            angles = np.full((n, 4), np.nan)
            reasons = {}
            bad = np.zeros(n, dtype=bool)
        rejected = padded(bad, 5)
        good = np.zeros(n, dtype=bool)
        if error is None:
            for s, e in intervals(~rejected):
                if e - s >= a.min_frames:
                    good[s:e] = True
        masks = {
            categories[0]: good,
            categories[2]: rejected,
            categories[1]: ~(good | rejected),
        }
        assert np.all(sum(v.astype(int) for v in masks.values()) == 1)
        for category, mask in masks.items():
            for s, e in intervals(mask):
                new_key = f"{key}__repair_frames_{s}_{e}"
                meta = dict(clip["metadata"])
                meta.update(
                    parent_motion_key=key,
                    parent_robot_root=str(a.input.resolve()),
                    parent_frame_start=s,
                    parent_frame_end=e,
                    canonical_frame_start=canonical_start(clip["metadata"])
                    + s,
                    canonical_frame_count=e - s,
                    floor_alignment="constant_clip_p01_collision_sole_z",
                    floor_z_subtracted_m=shift,
                    quality_status=category,
                    source_fidelity_origin=origin,
                    training_approved=False,
                    num_frames=e - s,
                )
                index = len(manifests[category]["clips"])
                out_start, length = writers[category].append_motion(
                    motion_id=index,
                    arrays={
                        name: value[s:e] for name, value in arrays.items()
                    },
                    metadata_json=json.dumps(meta),
                )
                manifests[category]["clips"][new_key] = {
                    "shard": 0,
                    "clip_idx": index,
                    "start": out_start,
                    "length": length,
                    "metadata": meta,
                }
                manifests[category]["motion_keys"].append(new_key)
        row = {
            "motion_key": key,
            "frames": n,
            "candidate_frames": int(good.sum()),
            "rejected_frames": int(rejected.sum()),
            "short_or_source_review_frames": int(masks[categories[1]].sum()),
            "source_fidelity_origin": origin,
            "source_error": error,
            "native_leg_direction_mean_deg": float(np.mean(angles))
            if error is None
            else None,
            "native_leg_direction_p95_deg": float(np.percentile(angles, 95))
            if error is None
            else None,
            "source_upper_arm_direction_mean_deg": float(np.mean(arm_angles))
            if error is None
            else None,
            "source_upper_arm_direction_p95_deg": float(
                np.percentile(arm_angles, 95)
            )
            if error is None
            else None,
            "max_leg_collision_m": float(collision.max()),
            "reason_intervals": {
                name: intervals(mask)
                for name, mask in reasons.items()
                if mask.any()
            },
        }
        reports.append(row)
        print(
            key,
            "candidate",
            row["candidate_frames"],
            "/",
            n,
            "source_error",
            error,
            flush=True,
        )
    for name, writer in writers.items():
        writer.finalize()
        manifests[name]["hdf5_shards"] = [
            {
                "file": "shards/k1_000000.h5",
                "num_clips": len(manifests[name]["clips"]),
                "num_frames": sum(
                    c["length"] for c in manifests[name]["clips"].values()
                ),
            }
        ]
        (a.output / name / "manifest.json").write_text(
            json.dumps(manifests[name], indent=2) + "\n"
        )
    report = {
        "input": str(a.input.resolve()),
        "training_approved": False,
        "checks": "every-frame FK/collision, numerical limits, canonical/native leg direction and contact mismatch; not dynamics",
        "source_frames": sum(r["frames"] for r in reports),
        "candidate_frames": sum(r["candidate_frames"] for r in reports),
        "rejected_frames": sum(r["rejected_frames"] for r in reports),
        "short_or_source_review_frames": sum(
            r["short_or_source_review_frames"] for r in reports
        ),
        "clips": reports,
    }
    (a.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
