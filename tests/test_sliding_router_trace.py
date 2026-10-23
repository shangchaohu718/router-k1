"""Trace analysis must exclude reassigned and reset reference suffixes."""

import numpy as np

from tools.booster_k1_retarget.summarize_sliding_router_trace import (
    first_episode,
)


def test_trace_stops_at_reset_or_assignment_change_and_respects_env_indices():
    ids = np.array([0, 2, 4, 6])
    frames = np.array([0, 0, 1, 1, 0, 2, 1, 3])
    keys = np.array(["walk"] * 8)
    np.testing.assert_array_equal(first_episode(ids, keys, frames), [0, 2])
    frames[4] = 2
    frames[6] = 3
    keys[6] = "dance"
    np.testing.assert_array_equal(first_episode(ids, keys, frames), [0, 2, 4])
    np.testing.assert_array_equal(
        first_episode(np.array([], dtype=int), keys, frames), []
    )
