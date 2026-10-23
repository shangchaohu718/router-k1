"""Report numerical and sampled FK concerns without declaring physics validity."""

import argparse
import json
from pathlib import Path
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]


def audit(root, stride=5):
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
    if names != manifest["dof_names"]:
        raise ValueError("K1 joint order mismatch")
    limits = model.jnt_range[joints]
    foot_bodies = {
        model.body(side + "_ankle_roll_link").id for side in ("left", "right")
    }
    geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in foot_bodies
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        and model.geom_contype[i]
    ]
    if len(geoms) != 2:
        raise ValueError("Expected two foot collision boxes")
    rows = []
    for key in manifest["motion_keys"]:
        clip = manifest["clips"][key]
        s, n = clip["start"], clip["length"]
        with h5py.File(
            root / manifest["hdf5_shards"][clip["shard"]]["file"], "r"
        ) as f:
            pos, rot, q = [
                f[k][s : s + n]
                for k in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
            ]
        finite = all(np.isfinite(x).all() for x in (pos, rot, q))
        if not finite:
            rows.append({"motion": key, "hard_error": "nonfinite"})
            continue
        fps = clip["metadata"].get("motion_fps", 50)
        violation = np.maximum(
            np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0
        )
        foot_min = []
        for i in range(0, n, stride):
            data.qpos[:] = np.r_[pos[i], rot[i, [3, 0, 1, 2]], q[i]]
            mujoco.mj_forward(model, data)
            foot_min.append(
                min(
                    data.geom_xpos[g, 2]
                    - np.dot(
                        np.abs(data.geom_xmat[g].reshape(3, 3)[2]),
                        model.geom_size[g],
                    )
                    for g in geoms
                )
            )
        foot_min = np.array(foot_min)
        tilt = np.degrees(
            np.arccos(
                np.clip(Rotation.from_quat(rot).as_matrix()[:, 2, 2], -1, 1)
            )
        )
        rows.append(
            {
                "motion": key,
                "frames": n,
                "seconds": n / fps,
                "max_joint_limit_violation_rad": float(violation.max()),
                "max_quaternion_norm_error": float(
                    np.abs(np.linalg.norm(rot, axis=1) - 1).max()
                ),
                "max_joint_speed_rad_s": float(
                    np.abs(np.diff(q, axis=0) * fps).max()
                ),
                "root_height_range_m": [
                    float(pos[:, 2].min()),
                    float(pos[:, 2].max()),
                ],
                "root_tilt_p95_deg": float(np.percentile(tilt, 95)),
                "sampled_foot_min_z_m": float(foot_min.min()),
                "sampled_foot_z_p05_m": float(np.percentile(foot_min, 5)),
                "sampled_penetration_over_2cm_fraction": float(
                    np.mean(foot_min < -0.02)
                ),
                "sampled_both_feet_above_5cm_fraction": float(
                    np.mean(foot_min > 0.05)
                ),
                "review_required": bool(
                    violation.max() > 0.01
                    or np.mean(foot_min < -0.02) > 0.05
                    or np.percentile(foot_min, 5) > 0.05
                ),
            }
        )
    return {
        "root": str(root.resolve()),
        "status": "NUMERICAL_FK_ONLY_NOT_PHYSICS_VALIDATION",
        "fk_stride": stride,
        "clips": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
