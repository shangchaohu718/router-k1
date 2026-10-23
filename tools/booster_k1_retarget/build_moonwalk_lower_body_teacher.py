"""Extract a quality-masked K1 trajectory teacher; never overwrite source data.

This custom auxiliary NPZ is not a human source, a replacement reference H5,
an action-cloning dataset, or an independent validation set.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
OFFSETS = np.array(
    [
        [
            [
                0.08672072738409042,
                0.0010306332260370255,
                -0.038234006613492966,
            ],
            [
                -0.06461569666862488,
                0.0010306332260370255,
                -0.038234006613492966,
            ],
        ],
        [
            [
                0.08672072738409042,
                -0.0010306332260370255,
                -0.038234006613492966,
            ],
            [
                -0.06461569666862488,
                -0.0010306332260370255,
                -0.038234006613492966,
            ],
        ],
    ]
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heading_xy(vectors, quat):
    forward = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).apply(
        np.tile([1.0, 0.0, 0.0], (len(quat), 1))
    )
    yaw = np.arctan2(forward[:, 1], forward[:, 0])
    while yaw.ndim < vectors.ndim - 1:
        yaw = yaw[:, None]
    return np.stack(
        [
            np.cos(yaw) * vectors[..., 0] + np.sin(yaw) * vectors[..., 1],
            -np.sin(yaw) * vectors[..., 0] + np.cos(yaw) * vectors[..., 1],
        ],
        axis=-1,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--joint-manifest", type=Path, required=True)
    p.add_argument("--teacher-motion", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    meta = json.loads(a.trajectory.with_suffix(".json").read_text())
    manifest = json.loads((a.reference_root / "manifest.json").read_text())
    if len(manifest["clips"]) != 1:
        raise ValueError(
            "Teacher extraction requires the isolated v14 reference"
        )
    key, clip = next(iter(manifest["clips"].items()))
    names = manifest["dof_names"]
    runtime_names = json.loads(a.joint_manifest.read_text())[
        "runtime_joint_names"
    ]
    if len(set(runtime_names)) != 22 or set(runtime_names) != set(names):
        raise ValueError("Teacher/student joint names differ")
    debug_names = (
        meta.get("action_chain_debug", {})
        .get("evaluation_target_bound", {})
        .get("joint_names")
    )
    if debug_names is not None and debug_names != runtime_names:
        raise ValueError("Teacher recording joint order differs from gate")
    if digest(a.teacher_motion) != meta["motion_sha256"]:
        raise ValueError("Teacher source motion hash differs")
    with h5py.File(
        a.reference_root / manifest["hdf5_shards"][clip["shard"]]["file"], "r"
    ) as f:
        s, n = int(clip["start"]), int(clip["length"])
        ref = {
            k: f[k][s : s + n]
            for k in ("ref_dof_pos", "ref_root_pos", "ref_root_rot")
        }
    if n != 500 or clip["metadata"]["motion_fps"] != 50:
        raise ValueError(
            "This teacher requires the exact 500-frame, 50 Hz v14 clip"
        )
    with np.load(a.teacher_motion, allow_pickle=False) as motion:
        order = [motion["joint_names"].tolist().index(name) for name in names]
        if motion["joint_pos"].shape != (n, 22) or not np.allclose(
            motion["joint_pos"][:, order],
            ref["ref_dof_pos"],
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(
                "Teacher motion and student reference have different poses/phases"
            )
        if float(motion["fps"]) != 50:
            raise ValueError("Teacher source motion rate differs")
        source_body_names = motion["body_names"].tolist()
        reference_quat = motion["body_quat_w"][
            :, source_body_names.index("trunk")
        ].copy()
    with np.load(a.trajectory, allow_pickle=False) as f:
        data = {
            k: f[k].copy()
            for k in (
                "reference_frame",
                "root_actual",
                "root_quat_actual",
                "root_reference",
                "joint_pos_actual",
                "joint_vel_actual",
                "foot_point_actual",
                "foot_point_reference",
                "foot_point_velocity_actual",
                "role_code",
                "foot_site_min_height_m",
                "foot_ground_compatible_contact",
                "foot_contact_force_norm_n",
                "cross_leg_contact_force_matrix_n",
            )
        }
    if not np.array_equal(
        data["reference_frame"], np.arange(n)
    ) or not np.isclose(n / meta["duration_s"], 50):
        raise ValueError("Teacher frames are missing/reordered or not 50 Hz")
    if any(not np.isfinite(v).all() for v in data.values()):
        raise ValueError("Nonfinite teacher data")
    if any(
        bool(v)
        for k, v in meta["termination_reasons"].items()
        if k not in {"motion_end", "time_out"}
    ):
        raise ValueError("Teacher run terminated abnormally")
    q = data["joint_pos_actual"][
        :, [runtime_names.index(name) for name in names]
    ]
    qv = data["joint_vel_actual"][
        :, [runtime_names.index(name) for name in names]
    ]
    model_path = REPO / "assets/robots/booster_k1/K1_22dof.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(model)
    ids = [
        i
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    if [model.joint(i).name for i in ids] != names:
        raise ValueError("FK model joint order differs")
    foot_ids = [
        model.body(side + "_ankle_roll_link").id for side in ("left", "right")
    ]
    actual_fk = []
    ref_fk = []
    for i in range(n):
        for out, root, quat, joints in (
            (actual_fk, data["root_actual"], data["root_quat_actual"], q),
            (
                ref_fk,
                ref["ref_root_pos"],
                ref["ref_root_rot"][:, [3, 0, 1, 2]],
                ref["ref_dof_pos"],
            ),
        ):
            d.qpos[:] = np.r_[root[i], quat[i], joints[i]]
            mujoco.mj_forward(model, d)
            out.append(
                [
                    d.xpos[b] + OFFSETS[j] @ d.xmat[b].reshape(3, 3).T
                    for j, b in enumerate(foot_ids)
                ]
            )
    fk_error = float(
        np.max(
            np.linalg.norm(
                np.array(actual_fk) - data["foot_point_actual"], axis=-1
            )
        )
    )
    if fk_error > 0.001:
        raise ValueError(
            f"Teacher geometry differs from student model: {fk_error} m"
        )
    teacher_relative_ref = heading_xy(
        data["foot_point_reference"]
        - data["root_reference"][:, None, None, :],
        reference_quat,
    )
    own_relative_ref = heading_xy(
        np.array(ref_fk) - ref["ref_root_pos"][:, None, None, :],
        ref["ref_root_rot"][:, [3, 0, 1, 2]],
    )
    alignment = float(
        np.max(
            np.linalg.norm(teacher_relative_ref - own_relative_ref, axis=-1)
        )
    )
    if alignment > 0.001:
        raise ValueError(
            f"Teacher reference frame/heading mismatch: {alignment} m"
        )
    points = data["foot_point_actual"]
    xy = heading_xy(
        points - data["root_actual"][:, None, None, :],
        data["root_quat_actual"],
    )
    velocity = heading_xy(
        data["foot_point_velocity_actual"], data["root_quat_actual"]
    )
    leg_names = [
        side + "_" + part + "_joint"
        for side in ("left", "right")
        for part in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee_pitch",
            "ankle_pitch",
            "ankle_roll",
        )
    ]
    leg_ids = [names.index(name) for name in leg_names]
    gap = points[:, :, 1, 2] - points[:, :, 0, 2]
    roles = data["role_code"]
    if not np.isin(roles, [0, 1, 2]).all():
        raise ValueError("Unknown teacher phase code")
    paired = ((roles[:, 0] == 1) & (roles[:, 1] == 2)) | (
        (roles[:, 0] == 2) & (roles[:, 1] == 1)
    )
    tilt = np.arccos(
        np.clip(
            Rotation.from_quat(
                data["root_quat_actual"][:, [1, 2, 3, 0]]
            ).as_matrix()[:, 2, 2],
            -1,
            1,
        )
    )
    cross = np.linalg.norm(
        data["cross_leg_contact_force_matrix_n"], axis=-1
    ).max(axis=(1, 2))
    valid_global = (
        (tilt < np.deg2rad(15)) & (cross < 1.0) & (np.arange(n) >= 10)
    )
    limit = model.jnt_range[ids][leg_ids].reshape(2, 6, 2)
    legq = q[:, leg_ids].reshape(n, 2, 6)
    legv = qv[:, leg_ids].reshape(n, 2, 6)
    valid_leg = (
        (legq >= limit[:, :, 0] - 0.01) & (legq <= limit[:, :, 1] + 0.01)
    ).all(-1) & (np.abs(legv) < 35).all(-1)
    grounded = (
        data["foot_ground_compatible_contact"]
        & (data["foot_contact_force_norm_n"] > 5)
        & (data["foot_site_min_height_m"] >= -0.015)
        & (points[:, :, 0, 2] < 0.025)
    )
    semantic = np.where(
        roles == 1,
        gap > 0.01,
        np.where(
            roles == 2,
            (np.abs(gap) < 0.035) & (velocity[..., 0].mean(-1) < -0.03),
            False,
        ),
    )
    quality = (
        valid_global[:, None]
        & valid_leg
        & grounded
        & semantic
        & paired[:, None]
    ).astype(np.float32)
    if (quality.prod(-1) > 0).sum() < 20:
        raise ValueError("Too little paired high-quality teacher coverage")
    report = {
        "schema": "k1_lower_body_teacher_v1",
        "fps": 50,
        "frames": n,
        "reference_motion_key": key,
        "expected_window_key": f"{key}__start_{s}_len_{n}",
        "reference_dof_names": names,
        "trajectory": str(a.trajectory.resolve()),
        "trajectory_sha256": digest(a.trajectory),
        "teacher_metadata_sha256": digest(a.trajectory.with_suffix(".json")),
        "teacher_motion_sha256": digest(a.teacher_motion),
        "reference_manifest_sha256": digest(
            a.reference_root / "manifest.json"
        ),
        "teacher_checkpoint_sha256": meta["checkpoint_sha256"],
        "teacher_phase_friction": meta.get("phase_foot_friction"),
        "student_friction_changed": False,
        "teacher_fk_max_error_m": fk_error,
        "reference_alignment_max_xy_error_m": alignment,
        "quality_foot_frames": quality.sum(0).astype(int).tolist(),
        "quality_paired_frames": int(quality.prod(-1).sum()),
        "not_copied": [
            "teacher actions",
            "upper-body joints",
            "root trajectory",
            "phase-varying friction",
        ],
        "limitations": "Auxiliary lower-body targets from one friction-assisted teacher rollout; not independent validation or a guaranteed transferable policy.",
    }
    a.output.mkdir(parents=True)
    np.savez_compressed(
        a.output / "teacher.npz",
        joint_names=np.array(leg_names),
        joint_pos=legq.astype(np.float32),
        sole_relative_xy=xy.astype(np.float32),
        sole_ground_z=points[..., 2].astype(np.float32),
        sole_velocity_heading_xy=velocity.astype(np.float32),
        role_code=roles,
        quality=quality,
        reference_dof_pos=ref["ref_dof_pos"].astype(np.float32),
        foot_offsets=OFFSETS.astype(np.float32),
        metadata=np.array(json.dumps(report)),
    )
    (a.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
