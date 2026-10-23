from pathlib import Path

import mujoco
import numpy as np

from tools.booster_k1_retarget.probe_native_position_retarget import (
    collision_safe_step,
    cross_leg_depth,
)


def model_and_groups():
    model = mujoco.MjModel.from_xml_path(
        str(
            Path(__file__).resolve().parents[1]
            / "assets/robots/booster_k1/K1_22dof.xml"
        )
    )
    groups = []
    for side in ("left", "right"):
        groups.append(
            {
                g
                for g in range(model.ngeom)
                if model.geom_contype[g]
                and model.body(int(model.geom_bodyid[g])).name.startswith(
                    side + "_"
                )
                and any(
                    k in model.body(int(model.geom_bodyid[g])).name
                    for k in ("hip_", "knee_", "ankle_")
                )
            }
        )
    return model, groups


def test_nonlinear_crossed_leg_step_is_backtracked_using_real_collision_geometry():
    model, groups = model_and_groups()
    data = mujoco.MjData(model)
    previous = data.qpos.copy()
    bad = previous.copy()
    bad[model.joint("left_hip_roll_joint").qposadr[0]] = -0.25
    bad[model.joint("right_hip_roll_joint").qposadr[0]] = 0.25
    data.qpos[:] = bad
    mujoco.mj_forward(model, data)
    assert cross_leg_depth(data, *groups) > 0.005
    result, alpha = collision_safe_step(model, data, previous, bad, *groups)
    assert 0 <= alpha < 1
    data.qpos[:] = result
    mujoco.mj_forward(model, data)
    assert cross_leg_depth(data, *groups) <= 0.001
    assert np.isfinite(result).all()


def test_valid_free_root_motion_is_not_clamped_or_mirrored():
    model, groups = model_and_groups()
    data = mujoco.MjData(model)
    previous = data.qpos.copy()
    target = previous.copy()
    target[0] += 0.1
    target[1] -= 0.2
    result, alpha = collision_safe_step(model, data, previous, target, *groups)
    assert alpha == 1.0
    np.testing.assert_allclose(result, target)
