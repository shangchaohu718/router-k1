from pathlib import Path
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/booster_k1_retarget")
)
from audit_reserve_leakage import group_keys
from finalize_targeted_repair import best_nonoverlap


def test_cmu_subject_and_lafan_subject():
    assert group_keys({"source": "CMU", "motion": "09_01"}) & group_keys(
        {"source": "CMU", "motion": "9_02"}
    )
    assert group_keys(
        {"source": "LaFAN1", "motion": "walk1_subject5"}
    ) & group_keys({"source": "LaFAN1", "motion": "dance2_subject5"})


def test_aist_dancer_and_filename_family():
    a = group_keys(
        {"source": "AIST++", "motion": "gBR_sBM_cAll_d01_mBR0_ch01"}
    )
    b = group_keys(
        {"source": "AIST++", "motion": "gBR_sBM_cAll_d02_mBR0_ch01"}
    )
    c = group_keys(
        {"source": "AIST++", "motion": "gPO_sBM_cAll_d01_mPO1_ch02"}
    )
    assert a & b
    assert a & c


def test_style_group_and_segment_parent():
    assert group_keys(
        {"source": "100STYLE", "motion": "Neutral_FW"}
    ) & group_keys(
        {
            "source": "100STYLE",
            "motion": "arbitrary_segment",
            "parent_motion": "Neutral_FR",
        }
    )
    assert group_keys({"source": "CMU", "motion": "unidentified"}) is None


def item(name, start, end, new=False):
    return {
        "source": "CMU",
        "motion": name,
        "canonical_start": start,
        "canonical_end": end,
        "new_direction_repair": new,
    }


def test_maximum_coverage_not_greedy_longest():
    rows = [
        item("old", 0, 100),
        item("new1", 0, 40, True),
        item("new2", 40, 120, True),
    ]
    assert best_nonoverlap(rows) == {("CMU", "new1"), ("CMU", "new2")}


def test_preserve_old_on_tie_and_keep_adjacent():
    assert best_nonoverlap(
        [item("old", 0, 100), item("new", 0, 100, True)]
    ) == {("CMU", "old")}
    assert best_nonoverlap([item("a", 0, 100), item("b", 100, 200)]) == {
        ("CMU", "a"),
        ("CMU", "b"),
    }
    assert best_nonoverlap([]) == set()
