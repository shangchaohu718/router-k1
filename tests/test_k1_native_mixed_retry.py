from tools.booster_k1_retarget.retry_native_cmu_mixed_references import (
    eligible_row,
)


def test_only_mixed_ankle_cases_with_enough_frames_and_source_are_targeted():
    row = {
        "frames": 100,
        "source_error": None,
        "reason_intervals": {
            "ankle_limit_saturation_for_0p5s": [[0, 30]],
            "upper_arm_direction_over_60deg": [[50, 55]],
        },
    }
    assert eligible_row(row)
    assert not eligible_row({**row, "frames": 49})
    assert not eligible_row({**row, "source_error": "missing source"})
    assert not eligible_row(
        {**row, "reason_intervals": {"ankle_limit_saturation_for_0p5s": []}}
    )
    assert not eligible_row(
        {**row, "reason_intervals": {"upper_arm_direction_over_60deg": []}}
    )
