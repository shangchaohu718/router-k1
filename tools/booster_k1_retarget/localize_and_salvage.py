"""Localize quarantined frames; recover conservative contiguous candidates.

Never overwrites sources or promotes candidates to physics-approved data.
Frame intervals are zero-based and end-exclusive in the source robot clip.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import h5py
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from holomotion.src.training.data_production.robot_h5 import RobotH5ShardWriter


def intervals(mask):
    edges = np.diff(
        np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    )
    return list(
        zip(
            np.where(edges == 1)[0].tolist(), np.where(edges == -1)[0].tolist()
        )
    )


def padded(mask, radius):
    out = np.array(mask, dtype=bool, copy=True)
    for start, end in intervals(mask):
        out[max(0, start - radius) : min(len(out), end + radius)] = True
    return out


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction-audit", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = json.loads((args.snapshot / "all_motions.json").read_text())
    direction_map = {}
    if args.direction_audit:
        audit = json.loads(args.direction_audit.read_text())
        if audit["frame_stride"] != 1:
            raise ValueError("Directional salvage requires every-frame audit")
        direction_map = {
            (str(Path(r["robot_root"]).resolve()), r["robot_motion_key"]): r
            for r in audit["clips"]
        }
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "assets/robots/booster_k1/K1_22dof.xml")
    )
    data = mujoco.MjData(model)
    joints = [
        i
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    names = [model.joint(i).name for i in joints]
    limits = model.jnt_range[joints]
    feet = {
        model.body(side + "_ankle_roll_link").id for side in ("left", "right")
    }
    geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in feet
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        and model.geom_contype[i]
    ]
    assert len(geoms) == 2
    reports, segments = [], []
    for index, row in enumerate(
        r for r in records if r["quality"] == "quarantine"
    ):
        root = Path(row["robot_root"])
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["dof_names"] == names
        key = row.get("robot_motion_key", row["motion"])
        clip = manifest["clips"][key]
        start, n = clip["start"], clip["length"]
        fps = float(clip["metadata"]["motion_fps"])
        assert fps == 50
        with h5py.File(
            root / manifest["hdf5_shards"][clip["shard"]]["file"], "r"
        ) as f:
            arrays = {
                k: f[k][start : start + n] for k in manifest["array_names"]
            }
        pos, rot, q = (
            arrays[k] for k in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
        )
        finite = (
            np.isfinite(pos).all(1)
            & np.isfinite(rot).all(1)
            & np.isfinite(q).all(1)
        )
        norm = np.linalg.norm(rot, axis=1)
        valid = finite & (np.abs(norm - 1) < 1e-3)
        speed = np.abs(np.diff(q, axis=0)) * fps
        fast_edges = np.any(speed > 35, axis=1)
        fast = np.zeros(n, dtype=bool)
        fast[:-1] |= fast_edges
        fast[1:] |= fast_edges
        limit_bad = np.any(
            (q < limits[:, 0] - 0.01) | (q > limits[:, 1] + 0.01), axis=1
        )
        # Quaternion z-axis projection; xyzw layout.
        tilt = np.degrees(
            np.arccos(
                np.clip(1 - 2 * (rot[:, 0] ** 2 + rot[:, 1] ** 2), -1, 1)
            )
        )
        heights = np.full(n, np.nan)
        for frame in np.flatnonzero(valid):
            data.qpos[:] = np.r_[
                pos[frame], rot[frame, [3, 0, 1, 2]], q[frame]
            ]
            mujoco.mj_forward(model, data)
            heights[frame] = min(
                data.geom_xpos[g, 2]
                - np.abs(data.geom_xmat[g].reshape(3, 3)[2])
                @ model.geom_size[g]
                for g in geoms
            )
        shift = (
            float(np.nanpercentile(heights, 1))
            if np.isfinite(heights).any()
            else 0.0
        )
        heights -= shift
        masks = {
            "invalid_pose": ~valid,
            "joint_speed_over_35_rad_s": fast,
            "joint_limit": limit_bad,
            "tilt_over_55deg": tilt > 55,
            "sole_below_minus_2cm": heights < -0.02,
            # Diagnostic only: airborne frames may be legitimate jumps.
            "both_feet_above_12cm_review": heights > 0.12,
        }
        reasons = row.get("reasons", [])
        holds = list(row.get("review_reasons", []))
        direction = direction_map.get((str(root.resolve()), key))
        directional_bad = np.zeros(n, dtype=bool)
        if direction:
            if direction["samples"] != n:
                raise ValueError("Incomplete frame direction audit")
            for event in direction["sampled_error_frames"]:
                directional_bad[event["segment_frame"]] = True
            # All spatial concerns now become explicit frame exclusions;
            # every retained segment still receives its own aggregate audit.
            holds = [
                r
                for r in holds
                if not any(
                    token in r
                    for token in (
                        "limb_direction",
                        "root_orientation",
                        "lower_body_regression",
                    )
                )
            ]
            masks["direction_error"] = directional_bad
        holds += [
            r
            for r in reasons
            if r
            not in {
                "joint_speed_over_35_rad_s",
                "large_tilt_defer_to_complex_pool",
            }
        ]
        if row.get("difficulty") == "L5" or row.get("split") not in {
            "train",
            "val",
            "test",
        }:
            holds.append("scene_or_split_review")
        if re.search(
            r"calibrat|wrestl|motorcycle|chair|ladder|stair|climb",
            row["motion"],
            re.I,
        ):
            holds.append("semantic_scene_review")
        bad = (
            ~valid
            | fast
            | limit_bad
            | (tilt > 55)
            | (heights < -0.02)
            | directional_bad
        )
        # Also remove an initial 0.2 s and 0.2 s on either side of bad spans.
        bad[: min(10, n)] = True
        safe = ~padded(bad, 10)
        spans = [(s, e) for s, e in intervals(safe) if e - s >= 150]
        report = {
            "source": row["source"],
            "motion": row["motion"],
            "robot_root": str(root),
            "robot_motion_key": key,
            "split": row["split"],
            "frames": n,
            "fps": fps,
            "frame_convention": "zero_based_end_exclusive_relative_to_robot_clip",
            "constant_floor_shift_m": shift,
            "intervals": {
                k: [
                    {
                        "start": s,
                        "end": e,
                        "start_seconds": s / fps,
                        "end_seconds": e / fps,
                    }
                    for s, e in intervals(mask)
                ]
                for k, mask in masks.items()
            },
            "speed_events": [
                {
                    "from_frame": int(i),
                    "to_frame": int(i + 1),
                    "joint": names[int(j)],
                    "speed_rad_s": float(speed[i, j]),
                }
                for i, j in np.argwhere(speed > 35)
            ],
            "whole_clip_holds": holds,
            "retained_segments": [],
        }
        if not holds:
            for s, e in spans:
                h = heights[s:e]
                if np.percentile(h, 50) > 0.12 or np.percentile(h, 5) > 0.05:
                    continue
                token = f"{index:04d}_{s:07d}_{e:07d}"
                level = row.get("difficulty", "L4")
                if level not in {"L1", "L2", "L3", "L4"}:
                    level = "L4"
                dest = (
                    args.output
                    / "recovered_candidates"
                    / level
                    / row["split"]
                    / token
                )
                out = {k: v[s:e].copy() for k, v in arrays.items()}
                out["ref_root_pos"][:, 2] -= shift
                metadata = dict(clip["metadata"])
                metadata.update(
                    {
                        "num_frames": e - s,
                        "source_robot_root": str(root),
                        "source_motion_key": key,
                        "source_frame_start": s,
                        "source_frame_end": e,
                        "floor_z_subtracted_m": shift,
                        "quality_status": "segmented_numerical_pass_visual_physics_pending",
                        "split": row["split"],
                    }
                )
                writer = RobotH5ShardWriter(
                    dest / "shards/k1_000000.h5", dof_pos_dim=22
                )
                out_start, length = writer.append_motion(
                    motion_id=0, arrays=out, metadata_json=json.dumps(metadata)
                )
                writer.finalize()
                out_key = f"{key}__frames_{s}_{e}"
                out_clip = {
                    **clip,
                    "motion_key": out_key,
                    "shard": 0,
                    "clip_idx": 0,
                    "start": out_start,
                    "length": length,
                    "metadata": metadata,
                }
                dump(
                    dest / "manifest.json",
                    {
                        **manifest,
                        "dataset_id": token,
                        "root": str(dest.resolve()),
                        "motion_keys": [out_key],
                        "clips": {out_key: out_clip},
                        "hdf5_shards": [
                            {
                                "file": "shards/k1_000000.h5",
                                "num_clips": 1,
                                "num_frames": length,
                            }
                        ],
                        "quality_status": metadata["quality_status"],
                    },
                )
                dump(dest / "nan_npz_paths.json", [])
                entry = {
                    "source": row["source"],
                    "parent_motion": row["motion"],
                    "motion": out_key,
                    "robot_root": str(dest.resolve()),
                    "split": row["split"],
                    "difficulty": level,
                    "difficulty_is_provisional": True,
                    "source_frame_start": s,
                    "source_frame_end": e,
                    "frames": e - s,
                    "seconds": (e - s) / fps,
                    "quality": metadata["quality_status"],
                }
                segments.append(entry)
                report["retained_segments"].append(entry)
        dump(args.output / "frame_reports" / f"{index:04d}.json", report)
        reports.append(
            {
                k: v
                for k, v in report.items()
                if k not in {"intervals", "speed_events", "retained_segments"}
            }
            | {
                "segment_count": len(report["retained_segments"]),
                "report": f"frame_reports/{index:04d}.json",
            }
        )
        if index % 20 == 0:
            print(f"audited={index + 1} segments={len(segments)}", flush=True)
    dump(args.output / "segments.json", segments)
    dump(args.output / "audit_index.json", reports)
    dump(
        args.output / "summary.json",
        {
            "audited_quarantine_clips": len(reports),
            "recovered_candidate_segments": len(segments),
            "recovered_parent_clips": len(
                {(r["source"], r["parent_motion"]) for r in segments}
            ),
            "recovered_hours": sum(r["seconds"] for r in segments) / 3600,
            "source_snapshot": str(args.snapshot.resolve()),
            "training_started": False,
            "training_approved": False,
            "originals_preserved": True,
            "policy": "No interpolation; >=3 s contiguous spans; 0.2 s margins; whole-clip scene holds retained; mapping holds become frame masks only with a complete every-frame direction audit",
        },
    )
    print(
        json.dumps(
            json.loads((args.output / "summary.json").read_text()), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
