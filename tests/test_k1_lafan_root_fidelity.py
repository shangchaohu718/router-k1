import numpy as np
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget.review_lafan_root_fidelity import (
    orientation_errors,
    retained_spans,
)


def test_root_drift_detected_and_quaternion_sign_does_not_matter():
    desired = Rotation.from_euler("z", [0, 0, 0], degrees=True)
    actual = Rotation.from_euler("z", [0, 15, 40], degrees=True)
    np.testing.assert_allclose(
        orientation_errors(actual.as_quat(), -desired.as_quat()),
        [0, 15, 40],
        atol=1e-10,
    )
    common = Rotation.from_euler("xyz", [15, 20, 80], degrees=True)
    np.testing.assert_allclose(
        orientation_errors(
            (common * actual).as_quat(), (common * desired).as_quat()
        ),
        [0, 15, 40],
        atol=1e-10,
    )


def test_gate_keeps_other_bad_frames_and_padding_without_stitching():
    row = {
        "frames": 300,
        "source_error": None,
        "reason_intervals": {"old_bad": [[200, 201]]},
    }
    errors = np.zeros(300)
    errors[100] = 20
    assert retained_spans(row, errors) == [(0, 95), (106, 195), (206, 300)]
    assert retained_spans({**row, "source_error": "missing"}, errors) == []
