from tools.booster_k1_retarget.consolidate_review_candidates import subtract


def test_source_interval_overlap_is_removed_without_joining():
    assert subtract(0, 300, [(100, 200)]) == [(0, 100), (200, 300)]
    assert subtract(0, 300, [(0, 400)]) == []
    assert subtract(0, 300, [(300, 400)]) == [(0, 300)]
    assert subtract(0, 300, [(50, 200), (100, 250)]) == [(0, 50), (250, 300)]
