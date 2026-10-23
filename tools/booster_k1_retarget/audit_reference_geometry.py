"""Audit every reference frame for floor gaps, leg limits and leg intersections.

Airborne and crossed-leg poses are review signals, not automatic rejection:
running, jumping and deliberate cross-steps require source-motion comparison.
"""

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from tools.booster_k1_retarget.localize_and_salvage import intervals

REPO = Path(__file__).resolve().parents[2]


def audit(root: Path, output: Path, stride: int = 1):
    if stride < 1:
        raise ValueError("stride must be positive")
    manifest = json.loads((root / "manifest.json").read_text())
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
    if "dof_names" in manifest and manifest["dof_names"] != names:
        raise ValueError("Reference/model DOF ordering differs")
    limits = model.jnt_range[joints]
    feet = [
        model.body(side + "_ankle_roll_link").id for side in ("left", "right")
    ]
    foot_geoms = [
        [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == body
            and model.geom_contype[g]
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        ]
        for body in feet
    ]
    assert all(len(gs) == 1 for gs in foot_geoms)
    leg_bodies = [
        {
            i
            for i in range(model.nbody)
            if model.body(i).name.startswith(side + "_")
            and any(
                p in model.body(i).name for p in ("hip_", "knee_", "ankle_")
            )
        }
        for side in ("left", "right")
    ]
    leg_joints = [
        i
        for i, name in enumerate(names)
        if any(x in name for x in ("hip_", "knee_", "ankle_"))
    ]
    rows = []
    for key, clip in manifest["clips"].items():
        s, n = int(clip["start"]), int(clip["length"])
        with h5py.File(
            root / manifest["hdf5_shards"][clip["shard"]]["file"], "r"
        ) as f:
            pos, quat, q = [
                f[name][s : s + n]
                for name in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
            ]
        indices = np.unique(np.r_[np.arange(0, n, stride), n - 1])
        z = np.empty((len(indices), 2))
        depth = np.zeros(len(indices))
        lateral = np.empty(len(indices))
        collision_pairs = set()
        for sample, frame in enumerate(indices):
            data.qpos[:] = np.r_[
                pos[frame], quat[frame, [3, 0, 1, 2]], q[frame]
            ]
            mujoco.mj_forward(model, data)
            for side, gs in enumerate(foot_geoms):
                z[sample, side] = min(
                    data.geom_xpos[g, 2]
                    - np.abs(data.geom_xmat[g].reshape(3, 3)[2])
                    @ model.geom_size[g]
                    for g in gs
                )
            left_y = data.xmat[model.body("trunk").id].reshape(3, 3)[:, 1]
            lateral[sample] = (
                data.xpos[feet[0]] - data.xpos[feet[1]]
            ) @ left_y
            for contact in data.contact[: data.ncon]:
                b1, b2 = (
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                )
                if (b1 in leg_bodies[0] and b2 in leg_bodies[1]) or (
                    b2 in leg_bodies[0] and b1 in leg_bodies[1]
                ):
                    depth[sample] = max(depth[sample], -float(contact.dist))
                    if contact.dist < -0.005:
                        collision_pairs.add(
                            (model.body(b1).name, model.body(b2).name)
                        )
        fps = float(clip.get("metadata", {}).get("motion_fps", 50))
        air = z.min(axis=1) > 0.05
        spans = intervals(air)
        frame_spans = [
            [int(indices[s]), min(n, int(indices[e - 1]) + stride)]
            for s, e in spans
        ]
        penetration = z.min(axis=1) < -0.02
        near = np.minimum(q - limits[:, 0], limits[:, 1] - q) < 0.005
        limit_margin = np.maximum(
            np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0.0
        )
        rows.append(
            {
                "motion_key": key,
                "frames": n,
                "fps": fps,
                "both_feet_above_5cm_fraction": float(air.mean()),
                "longest_both_feet_above_5cm_seconds": max(
                    [e - s for s, e in frame_spans], default=0
                )
                / fps,
                "both_feet_air_intervals": frame_spans,
                "floor_penetration_over_2cm_fraction": float(
                    penetration.mean()
                ),
                "sole_min_z_m": float(z.min()),
                "sole_min_p50_z_m": float(np.median(z.min(axis=1))),
                "sole_min_max_z_m": float(z.min(axis=1).max()),
                "worst_air_frame": int(indices[np.argmax(z.min(axis=1))]),
                "leg_intersection_max_depth_m": float(depth.max()),
                "leg_intersection_over_5mm_fraction": float(
                    np.mean(depth > 0.005)
                ),
                "leg_intersection_pairs": sorted(collision_pairs),
                "worst_intersection_frame": int(indices[np.argmax(depth)]),
                "crossed_feet_fraction": float(np.mean(lateral < -0.02)),
                "max_joint_limit_violation_rad": float(limit_margin.max()),
                "leg_joint_limit_saturation_fraction": {
                    names[j]: float(near[:, j].mean())
                    for j in leg_joints
                    if near[:, j].mean() > 0.05
                },
                "metadata": clip.get("metadata", {}),
            }
        )
        print(
            key,
            "air",
            round(air.mean(), 3),
            "air_duration",
            round(rows[-1]["longest_both_feet_above_5cm_seconds"], 2),
            "intersection",
            round(depth.max(), 3),
            flush=True,
        )
    result = {
        "root": str(root.resolve()),
        "method": "MuJoCo FK and collision geometry, no physics integration; stride>1 intervals are screening estimates",
        "frame_stride": stride,
        "clips": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    audit(args.root, args.output, args.stride)
