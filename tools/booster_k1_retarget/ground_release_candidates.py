"""Create a new floor-aligned candidate set with transparent rejection reasons.

One constant vertical shift per clip preserves jumps; never pins feet per frame.
The numerical gate is deliberately separate from visual/dynamic approval.
"""

import argparse
import json
from pathlib import Path
import sys
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from holomotion.src.training.data_production.robot_h5 import RobotH5ShardWriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trim-start", type=int, default=10)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads((args.input / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "assets/robots/booster_k1/K1_22dof.xml")
    )
    data = mujoco.MjData(model)
    joints = [
        i
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    assert source["dof_names"] == [model.joint(i).name for i in joints]
    feet = {model.body(s + "_ankle_roll_link").id for s in ("left", "right")}
    geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in feet
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        and model.geom_contype[i]
    ]
    assert len(geoms) == 2
    writer = RobotH5ShardWriter(
        args.output / "shards/k1_000000.h5", dof_pos_dim=22
    )
    clips, reports = {}, []
    for key in source["motion_keys"]:
        clip = source["clips"][key]
        start = clip["start"]
        length = clip["length"]
        with h5py.File(
            args.input / source["hdf5_shards"][clip["shard"]]["file"], "r"
        ) as f:
            arrays = {
                k: f[k][start + args.trim_start : start + length]
                for k in source["array_names"]
            }
        pos, q, rot = (
            arrays[k] for k in ("ref_root_pos", "ref_dof_pos", "ref_root_rot")
        )
        if len(q) < 2 or not all(
            np.isfinite(a).all() for a in arrays.values()
        ):
            reports.append({"motion": key, "reasons": ["invalid_arrays"]})
            continue
        heights = []
        for i in range(len(q)):
            data.qpos[:] = np.r_[pos[i], rot[i, [3, 0, 1, 2]], q[i]]
            mujoco.mj_forward(model, data)
            heights.append(
                min(
                    data.geom_xpos[g, 2]
                    - np.abs(data.geom_xmat[g].reshape(3, 3)[2])
                    @ model.geom_size[g]
                    for g in geoms
                )
            )
        heights = np.asarray(heights)
        shift = float(np.percentile(heights, 1))
        pos[:, 2] -= shift
        heights -= shift
        limits = model.jnt_range[joints]
        violation = float(
            np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0).max()
        )
        fps = float(clip["metadata"]["motion_fps"])
        if not np.isclose(fps, 50):
            raise ValueError(f"Expected 50 Hz training reference, got {fps}")
        speed = float(np.abs(np.diff(q, axis=0) * fps).max())
        penetration = float(np.mean(heights < -0.02))
        tilt = np.degrees(
            np.arccos(
                np.clip(Rotation.from_quat(rot).as_matrix()[:, 2, 2], -1, 1)
            )
        )
        reasons = []
        if violation > 0.01:
            reasons.append("joint_limit")
        if speed > 35:
            reasons.append("joint_speed_over_35_rad_s")
        if penetration > 0.05:
            reasons.append("frequent_floor_penetration")
        if np.percentile(heights, 50) > 0.12:
            reasons.append("floor_alignment_ambiguous")
        if np.percentile(heights, 5) > 0.05:
            reasons.append("insufficient_ground_contact_review")
        if np.percentile(tilt, 95) > 55:
            reasons.append("large_tilt_defer_to_complex_pool")
        row = {
            "motion": key,
            "source_frames": length,
            "trimmed_start_frames": args.trim_start,
            "z_subtracted_m": shift,
            "joint_speed_max": speed,
            "penetration_fraction": penetration,
            "tilt_p95": float(np.percentile(tilt, 95)),
            "accepted_numerical_gate": not reasons,
            "reasons": reasons,
        }
        reports.append(row)
        if reasons:
            continue
        metadata = dict(clip["metadata"])
        metadata.update(
            {
                "floor_alignment": "constant_clip_p01_collision_sole_z",
                "floor_z_subtracted_m": shift,
                "source_dataset": str(args.input.resolve()),
                "source_start_frame_trim": args.trim_start,
                "num_frames": len(q),
                "quality_status": "numerical_pass_visual_and_physics_pending",
            }
        )
        index = len(clips)
        out_start, n = writer.append_motion(
            motion_id=index, arrays=arrays, metadata_json=json.dumps(metadata)
        )
        clips[key] = {
            "motion_key": key,
            "shard": 0,
            "clip_idx": index,
            "start": out_start,
            "length": n,
            "metadata": metadata,
            "available_arrays": source["array_names"],
        }
    summary = writer.finalize()
    manifest = {
        **source,
        "dataset_id": args.output.name,
        "root": str(args.output.resolve()),
        "motion_keys": list(clips),
        "clips": clips,
        "hdf5_shards": [
            {
                "file": "shards/k1_000000.h5",
                "num_clips": len(clips),
                "num_frames": summary["num_frames"],
            }
        ],
        "quality_status": "numerical_gate_only",
    }
    for name, value in [
        ("manifest.json", manifest),
        ("nan_npz_paths.json", []),
        ("preparation_report.json", reports),
    ]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        f"{args.output}: accepted={len(clips)} deferred={len(reports) - len(clips)} frames={summary['num_frames']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
