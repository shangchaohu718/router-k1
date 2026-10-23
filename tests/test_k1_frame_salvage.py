import importlib.util
from pathlib import Path

import numpy as np

PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/booster_k1_retarget/localize_and_salvage.py"
)
SPEC = importlib.util.spec_from_file_location("salvage", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_intervals_end_exclusive():
    assert MODULE.intervals([1, 1, 0, 1]) == [(0, 2), (3, 4)]
    assert MODULE.intervals([]) == []
    assert MODULE.intervals([0, 0]) == []


def test_padding_keeps_boundaries_and_input():
    x = np.array([1, 0, 0, 0, 1], dtype=bool)
    assert MODULE.padded(x, 1).tolist() == [True, True, False, True, True]
    assert x.tolist() == [True, False, False, False, True]
    assert MODULE.padded(x, 10).all()


def test_speed_edge_marks_both_endpoints():
    q = np.zeros((8, 22))
    q[4:, 0] = 1
    edges = np.any(np.abs(np.diff(q, axis=0)) * 50 > 35, axis=1)
    bad = np.zeros(8, bool)
    bad[:-1] |= edges
    bad[1:] |= edges
    assert MODULE.intervals(bad) == [(3, 5)]
