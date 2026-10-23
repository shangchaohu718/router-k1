import numpy as np
from scipy.spatial.transform import Rotation
from tools.booster_k1_retarget.diagnose_lafan_arm_reachability import direction


def test_arm_direction_matches_serial_axes():
    for sign in [-1, 1]:
        for pitch, roll in [(0.5, 0.8), (-2, -1), (1, 0)]:
            expected = (
                Rotation.from_euler("y", pitch)
                * Rotation.from_euler("x", roll)
            ).apply([0, sign, 0])
            np.testing.assert_allclose(
                direction([pitch, roll], sign), expected, atol=1e-12
            )
