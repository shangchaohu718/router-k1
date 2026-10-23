from pathlib import Path

import mujoco
import numpy as np
import pytest

from tools.booster_k1_retarget.retry_guarded_reference import bounded_step


def model():
    return mujoco.MjModel.from_xml_path(
        str(
            Path(__file__).resolve().parents[1]
            / "assets/robots/booster_k1/K1_22dof.xml"
        )
    )


def test_ankle_limit_overshoot_backtracks_entire_step_without_clipping():
    m = model()
    q = m.qpos0.copy()
    v = np.zeros(m.nv)
    joint = m.joint("right_ankle_roll_joint")
    v[joint.dofadr[0]] = 1.0
    v[0] = 0.1
    result, alpha, diagnostic = bounded_step(m, q, v, 1.0)
    assert 0 < alpha < 1
    assert result[joint.qposadr[0]] <= joint.range[1]
    np.testing.assert_allclose(result[0] - q[0], 0.1 * alpha)
    assert diagnostic["max_proposed_violation_rad"] > 0


def test_valid_motion_unmodified_and_invalid_previous_not_silently_clipped():
    m = model()
    q = m.qpos0.copy()
    v = np.zeros(m.nv)
    v[0] = 0.1
    result, alpha, _ = bounded_step(m, q, v, 0.02)
    assert alpha == 1
    np.testing.assert_allclose(result[0] - q[0], 0.002)
    joint = m.joint("right_ankle_roll_joint")
    q[joint.qposadr[0]] = 1.0
    with pytest.raises(ValueError, match="valid previous"):
        bounded_step(m, q, v, 0.02)
