import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget.retry_cmu_native_foot_frame import foot_frame


def test_neutral_plane_and_rigid_transform_covariance_not_world_up_pinning():
    points = np.array([[-0.1, 0, 0.05], [0, 0, 0], [0.05, 0, 0]])
    base = foot_frame(*points)
    np.testing.assert_allclose(base.as_matrix(), np.eye(3), atol=1e-12)
    rotation = Rotation.from_euler("xyz", [180, 25, 60], degrees=True)
    moved = rotation.apply(points) + [1, 2, 3]
    actual = foot_frame(*moved)
    np.testing.assert_allclose(
        actual.as_matrix(), (rotation * base).as_matrix(), atol=1e-12
    )
    assert actual.as_matrix()[2, 2] < 0


def test_degenerate_source_geometry_is_rejected_not_invented():
    with pytest.raises(ValueError, match="Collinear"):
        foot_frame([-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0])
