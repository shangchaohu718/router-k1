from types import SimpleNamespace

import numpy as np

from tools.booster_k1_retarget.audit_foot_direction_contact import foot_directions


def test_direction_samples_do_not_alias_mutable_mujoco_state():
    state = SimpleNamespace(xmat=np.tile(np.eye(3).reshape(1,9),(2,1)))
    captured = foot_directions(state,[0,1])
    state.xmat[:] = 0
    np.testing.assert_array_equal(captured,[[1,0,0],[1,0,0]])
