"""Switch construction preserves source data and valid rotations."""

import numpy as np
import pytest

from tools.booster_k1_retarget.prepare_sliding_router_fixtures import (
    transition_arrays,
)


def test_switch_fixture_preserves_central_source_and_has_stationary_ends():
    n = 20
    slide = {
        "ref_dof_pos": np.linspace(0, 1, n * 2).reshape(n, 2),
        "ref_root_pos": np.c_[
            np.linspace(0, -1, n), np.zeros(n), np.full(n, 0.6)
        ],
        "ref_root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
    }
    before = {k: v.copy() for k, v in slide.items()}
    neutral = {
        "ref_dof_pos": np.array([[0.2, 0.3]]),
        "ref_root_pos": np.array([[3.0, 4.0, 0.7]]),
    }
    actual, interval = transition_arrays(
        slide, neutral, hold_before=3, hold_after=4, blend=4
    )
    assert interval == [7, 19]
    for key, value in slide.items():
        np.testing.assert_allclose(actual[key][7:19], value[4:16])
        np.testing.assert_array_equal(slide[key], before[key])
        np.testing.assert_allclose(actual[key][3], actual[key][2])
        np.testing.assert_allclose(actual[key][-5], actual[key][-4])
    np.testing.assert_allclose(
        np.linalg.norm(actual["ref_root_rot"], axis=-1), 1
    )
    np.testing.assert_allclose(actual["ref_root_pos"][:3, :2], 0)
    np.testing.assert_allclose(actual["ref_root_pos"][-4:, 0], -1)


def test_switch_fixture_rejects_missing_central_motion():
    with pytest.raises(ValueError, match="nonempty"):
        transition_arrays({"ref_dof_pos": np.zeros((10, 2))}, {}, blend=5)
