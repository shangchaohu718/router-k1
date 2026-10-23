"""Read-only directional reachability probe; never emits repaired poses.

Fits shoulder direction within real K1 limits, not a replacement retargeter.
Two seeded bounded fits are diagnostic, not proof of global unreachability.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget import bvh_to_robot_dataset as old
from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import (
    REPO,
    load,
)
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def direction(angles, sign):
    pitch, roll = angles
    return sign * np.array(
        [
            np.sin(pitch) * np.sin(roll),
            np.cos(roll),
            np.cos(pitch) * np.sin(roll),
        ]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--key", default="dance2_subject4")
    p.add_argument("--variant", default="old_task_weights")
    a = p.parse_args()
    m = json.loads((a.root / "input/manifest.json").read_text())
    meta = m["clips"][a.key]["metadata"]
    arrays = load(a.root / a.variant / "raw", a.key)
    frames, _ = old.load_bvh_file(meta["source_path"], format="lafan1")
    t = (
        meta["canonical_frame_start"] + np.arange(len(arrays["ref_dof_pos"]))
    ) * 0.6
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "assets/robots/booster_k1/K1_22dof.xml")
    )
    data = mujoco.MjData(model)
    root_rot = Rotation.from_quat(arrays["ref_root_rot"])
    result = {
        "training_started": False,
        "source_data_overwritten": False,
        "motion": a.key,
        "variant": a.variant,
        "canonical_frame_start": meta["canonical_frame_start"],
        "diagnostic_only": True,
        "limitations": "Fixed-root upper-arm direction only, no forearm/hand/collision/speed feasibility. A successful fit demonstrates directional reachability; failed fits do not prove global unreachability. No fitted joint trajectories saved.",
        "sides": {},
    }
    names = m["dof_names"]
    for side, prefix, sign in [("left", "Left", 1), ("right", "Right", -1)]:
        joints = [
            ("aa" + side + "_shoulder_pitch_joint"),
            side + "_shoulder_roll_joint",
        ]
        ids = [names.index(j) for j in joints]
        bounds = np.array([model.jnt_range[model.joint(j).id] for j in joints])
        points = []
        for bone in [prefix + "Arm", prefix + "ForeArm", prefix + "Hand"]:
            v = np.array([f[bone][0] for f in frames])
            points.append(
                np.stack(
                    [
                        np.interp(t, np.arange(len(v)), v[:, i])
                        for i in range(3)
                    ],
                    axis=1,
                )
            )
        desired = root_rot.inv().apply(points[1] - points[0])
        desired /= np.linalg.norm(desired, axis=1)[:, None]
        observed = []
        best = []
        formula_error = []
        forearm_errors = []
        for frame in range(0, len(desired), 2):
            angles = arrays["ref_dof_pos"][frame, ids]
            actual = direction(angles, sign)
            data.qpos[:] = np.r_[
                arrays["ref_root_pos"][frame],
                arrays["ref_root_rot"][frame, [3, 0, 1, 2]],
                arrays["ref_dof_pos"][frame],
            ]
            mujoco.mj_forward(model, data)
            source_forearm = points[2][frame] - points[1][frame]
            source_forearm /= np.linalg.norm(source_forearm)
            robot_forearm = data.xmat[
                model.body(side + "_elbow_yaw_link").id
            ].reshape(3, 3) @ np.array([0, sign, 0])
            forearm_errors.append(
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(source_forearm, robot_forearm), -1, 1
                            )
                        )
                    )
                )
            )
            fk = (
                data.xpos[model.body(side + "_elbow_yaw_link").id]
                - data.xpos[model.body(side + "_shoulder_roll_link").id]
            )
            fk = root_rot[frame].inv().apply(fk / np.linalg.norm(fk))
            formula_error.append(float(np.linalg.norm(fk - actual)))
            target = desired[frame]
            error = lambda v: float(
                np.degrees(np.arccos(np.clip(np.dot(v, target), -1, 1)))
            )
            observed.append(error(actual))
            scores = [observed[-1]]
            for seed in [angles, np.mean(bounds, axis=1)]:
                fit = least_squares(
                    lambda x: direction(x, sign) - target,
                    np.clip(seed, bounds[:, 0], bounds[:, 1]),
                    bounds=(bounds[:, 0], bounds[:, 1]),
                    max_nfev=80,
                )
                scores.append(error(direction(fit.x, sign)))
            best.append(min(scores))
        if max(formula_error) > 1e-5:
            raise ValueError("Analytic direction disagrees with actual K1 FK")
        observed, best = np.array(observed), np.array(best)
        result["sides"][side] = {
            "sampled_frames": len(best),
            "fk_formula_max_error": max(formula_error),
            "actual_p95_deg": float(np.percentile(observed, 95)),
            "forearm_axis_p95_deg": float(np.percentile(forearm_errors, 95)),
            "best_found_p95_deg": float(np.percentile(best, 95)),
            "actual_over60deg_but_fit_under10deg_samples": int(
                np.sum((observed > 60) & (best < 10))
            ),
            "best_found_over30deg_samples": int(np.sum(best > 30)),
        }
    dump(a.root / (a.key + "_" + a.variant + "_arm_reachability.json"), result)


if __name__ == "__main__":
    main()
