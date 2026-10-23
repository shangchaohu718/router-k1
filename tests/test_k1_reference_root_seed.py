import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools/booster_k1_retarget/holosmpl_h5_to_k1_h5.py"
    )
    spec = importlib.util.spec_from_file_location("k1_seed_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_uses_calibrated_root_preserves_hinges_and_input():
    q = np.r_[np.zeros(3), [1, 0, 0, 0], np.arange(22) * 0.01]
    data = SimpleNamespace(qpos=q.copy())
    configuration = SimpleNamespace(
        data=data, update=lambda x: setattr(data, "qpos", x)
    )
    calibrated = [np.array([1.0, 2.0, 0.6]), np.array([0.0, 0.0, 0.0, 1.0])]
    r = SimpleNamespace(configuration=configuration, human_root_name="pelvis")

    def update_targets(human):
        human["pelvis"][0][:] = 123  # mirror vendor's in-place transformations
        r.scaled_human_data = {"pelvis": calibrated}

    r.update_targets = update_targets
    human = {"pelvis": [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]}
    load_module().initialize_k1_reference_root(r, human)
    np.testing.assert_array_equal(data.qpos[:3], calibrated[0])
    np.testing.assert_array_equal(data.qpos[3:7], calibrated[1])
    np.testing.assert_array_equal(data.qpos[7:], q[7:])
    np.testing.assert_array_equal(human["pelvis"][0], np.zeros(3))


def test_bad_seed_is_rejected_before_configuration_update():
    r = SimpleNamespace(
        human_root_name="pelvis",
        update_targets=lambda _: None,
        scaled_human_data={"pelvis": [np.zeros(3), np.zeros(4)]},
        configuration=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(29))),
    )
    with pytest.raises(ValueError, match="root seed"):
        load_module().initialize_k1_reference_root(r, {})
