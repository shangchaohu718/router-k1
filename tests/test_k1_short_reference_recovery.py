from tools.booster_k1_retarget.recover_short_reference_candidates import (
    eligible,
)


def clip(s, e):
    return {
        "length": e - s,
        "metadata": {"parent_frame_start": s, "parent_frame_end": e},
    }


def test_only_loader_compatible_short_runs_are_eligible():
    r = {"frames": 200, "source_error": None, "reason_intervals": {}}
    assert eligible(clip(0, 50), r)
    assert eligible(clip(0, 99), r)
    assert not eligible(clip(0, 49), r)
    assert not eligible(clip(0, 100), r)


def test_missing_source_or_bad_boundary_is_never_recovered():
    r = {
        "frames": 200,
        "source_error": None,
        "reason_intervals": {"collision": [[80, 90]]},
    }
    assert eligible(clip(0, 75), r)
    assert not eligible(clip(0, 76), r)
    assert eligible(clip(95, 150), r)
    assert not eligible(clip(94, 150), r)
    r["source_error"] = "missing human source"
    assert not eligible(clip(0, 75), r)
