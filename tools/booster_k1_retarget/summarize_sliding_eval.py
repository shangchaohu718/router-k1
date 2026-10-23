"""Summarize native NPZ tracking and phase-aware foot kinematics on CPU.

Grounding below is a geometry proxy. Native dumps do not contain contact
forces, so this tool never calls a near-ground point a verified contact.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget.k1_policy_comparison_viewer import (
    DEFAULT_ROBOT_XML,
)


def masked_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def foot_points(model, data, pos, rot, q, offsets):
    feet = [
        model.body(side + "_ankle_roll_link").id for side in ("left", "right")
    ]
    points = []
    for frame in range(len(q)):
        data.qpos[:] = np.r_[pos[frame], rot[frame, [3, 0, 1, 2]], q[frame]]
        mujoco.mj_forward(model, data)
        points.append(
            [
                data.xpos[body] + offsets[j] @ data.xmat[body].reshape(3, 3).T
                for j, body in enumerate(feet)
            ]
        )
    return np.array(points)


def summarize(path, future_frames=10):
    with np.load(path, allow_pickle=False) as source:
        meta = json.loads(str(source["metadata"]))
        count = min(
            len(source["robot_dof_pos"]),
            int(meta.get("valid_prefix_len", len(source["robot_dof_pos"]))),
        )
        arrays = {
            key: source[key][:count].copy()
            for key in (
                "robot_dof_pos",
                "ref_dof_pos",
                "robot_global_translation",
                "ref_global_translation",
                "robot_global_rotation_quat",
            )
        }
    if count < 3:
        raise ValueError(f"Too few valid evaluation frames: {path}")
    expected = max(1, int(meta["clip_length"]) - future_frames)
    q = arrays["robot_dof_pos"]
    root = arrays["robot_global_translation"][:, 0]
    ref_root = arrays["ref_global_translation"][:, 0]
    result = {
        "file": str(path),
        "motion_key": meta["motion_key"],
        "valid_frames": count,
        "expected_frames": expected,
        "coverage": min(1.0, count / expected),
        "completed": count >= expected,
        "duration_s": count / float(meta.get("motion_fps", 50)),
        "joint_mae_rad": float(np.abs(q - arrays["ref_dof_pos"]).mean()),
        "global_body_error_mm": float(
            np.linalg.norm(
                arrays["robot_global_translation"]
                - arrays["ref_global_translation"],
                axis=-1,
            ).mean()
            * 1000
        ),
        "root_height_within_25cm": bool(
            (np.abs(root[:, 2] - ref_root[:, 2]) < 0.25).all()
        ),
    }
    if "moonwalk" not in meta["motion_key"].lower():
        return result
    teacher = (
        Path(__file__).resolve().parents[2]
        / "data/holomotion_k1_v1"
        / "moonwalk_lower_body_teacher_v56_v1/teacher.npz"
    )
    with np.load(teacher, allow_pickle=False) as labels:
        reference = labels["reference_dof_pos"]
        offsets = labels["foot_offsets"]
        roles = labels["role_code"]
    matching = np.abs(arrays["ref_dof_pos"][:, None] - reference[None]).max(-1)
    phase = matching.argmin(-1)
    if matching[np.arange(count), phase].max() > 1e-5:
        raise ValueError(
            "Reference does not match the audited phase label bank"
        )
    roles = roles[phase]
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_ROBOT_XML))
    if model.nq != q.shape[1] + 7:
        raise ValueError(
            "Kinematic model and evaluation joint dimensions disagree"
        )
    data = mujoco.MjData(model)
    rot = arrays["robot_global_rotation_quat"][:, 0]
    points = foot_points(model, data, root, rot, q, offsets)
    dt = 1 / float(meta.get("motion_fps", 50))
    velocity = np.gradient(points, dt, axis=0)
    heading = Rotation.from_quat(rot).apply(
        np.tile([1.0, 0.0, 0.0], (count, 1))
    )
    heading[:, -1] = 0
    heading /= np.linalg.norm(heading, axis=-1, keepdims=True).clip(1e-8)
    foot_vx = (velocity.mean(2) * heading[:, None]).sum(-1)
    toe_speed = np.linalg.norm(velocity[:, :, 0, :2], axis=-1)
    heel_gap = points[:, :, 1, 2] - points[:, :, 0, 2]
    anchor, slide = roles == 1, roles == 2
    grounded = (points[:, :, 0, 2] < 0.025) & (
        points[:, :, :, 2].min(-1) > -0.015
    )
    slide_ok = grounded & (foot_vx < -0.03) & (np.abs(heel_gap) < 0.035)
    anchor_ok = grounded & (heel_gap > 0.01) & (toe_speed < 0.1)
    paired = ((roles[:, 0] == 1) & (roles[:, 1] == 2)) | (
        (roles[:, 0] == 2) & (roles[:, 1] == 1)
    )
    success = np.where(
        slide, slide_ok, np.where(anchor, anchor_ok, False)
    ).all(-1)
    result.update(
        {
            "support_toe_speed_mean_m_s": masked_mean(toe_speed, anchor),
            "support_slip_over_0p1_m_s_fraction": masked_mean(
                toe_speed > 0.1, anchor
            ),
            "slide_backward_speed_mean_m_s": masked_mean(-foot_vx, slide),
            "slide_grounded_kinematic_proxy_fraction": masked_mean(
                grounded, slide
            ),
            "anchor_heel_gap_mean_mm": masked_mean(heel_gap * 1000, anchor),
            "paired_mechanism_kinematic_proxy_fraction": masked_mean(
                success, paired
            ),
            "backward_root_distance_m": float(
                -(root[-1] - root[0]) @ heading[0]
            ),
            "notes": (
                "Grounding/support are geometry/velocity proxies, "
                "not measured contact forces."
            ),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    reports, failures = [], []
    for path in sorted(args.npz_dir.glob("*.npz")):
        try:
            reports.append(summarize(path))
        except (ValueError, KeyError) as error:
            failures.append({"file": str(path), "error": str(error)})
    if not reports:
        raise ValueError("No valid evaluation NPZs were found")
    result = {
        "clips": reports,
        "failures": failures,
        "completion_rate": float(np.mean([r["completed"] for r in reports])),
        "note": (
            "Per-prefix errors have different horizons after early ending; "
            "compare matched prefixes for direct error claims."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
