import numpy as np
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget.k1_policy_comparison_viewer import (
    align_initial_heading,
)


def test_initial_heading_matches_without_reflection_or_removing_turns():
    base = Rotation.from_euler(
        "zyx", [[0, 5, 3], [20, 8, 4], [60, 10, 5]], degrees=True
    )
    positions = np.array([[0.0, 0.0, 0.7], [1.0, 0.2, 0.8], [2.0, 0.5, 0.75]])
    outputs = []
    for yaw in (90, -170):
        offset = Rotation.from_euler("z", yaw, degrees=True)
        rot = offset * base
        pos = offset.apply(positions) + [15.0, -20.0, 0.0]
        aligned_pos, aligned_rot = align_initial_heading(pos, rot.as_quat())
        outputs.append((aligned_pos, aligned_rot))
        np.testing.assert_allclose(aligned_pos[:, 2], positions[:, 2])
        # A rigid rotation preserves pairwise distances and chirality.
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(aligned_pos, axis=0), axis=1),
            np.linalg.norm(np.diff(pos, axis=0), axis=1),
        )
        recovered = Rotation.from_quat(aligned_rot)
        np.testing.assert_allclose(np.linalg.det(recovered.as_matrix()), 1.0)
        np.testing.assert_allclose(
            (recovered[0].inv() * recovered[1:]).as_matrix(),
            (rot[0].inv() * rot[1:]).as_matrix(),
            atol=1e-12,
        )
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_quat(outputs[0][1]).as_matrix(),
        Rotation.from_quat(outputs[1][1]).as_matrix(),
        atol=1e-12,
    )
