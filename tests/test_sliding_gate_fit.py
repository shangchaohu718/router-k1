"""Gate splits separate motions and purge overlapping future windows."""

import numpy as np
import pytest
import torch
from torch import nn

from tools.booster_k1_retarget.fit_sliding_gate import (
    fold_gate_standardization,
    group_name,
    load_samples,
    split_samples,
)


def test_gate_fit_rejects_legacy_assignment_labels(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        features=np.zeros((2, 4)),
        labels=np.zeros(2),
        motion_keys=np.array(["walk", "walk"]),
    )
    with pytest.raises(ValueError, match="actual per-environment"):
        load_samples([path])


def test_folded_gate_preprocessing_preserves_logits_without_runtime_stats():
    torch.manual_seed(8)
    gate = nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 1))
    x, mean, scale = torch.randn(10, 4), torch.randn(4), torch.rand(4) + 0.1
    expected = gate((x - mean) / scale).detach()
    fold_gate_standardization(gate, mean, scale)
    torch.testing.assert_close(gate(x), expected, rtol=2e-5, atol=2e-6)
    assert set(gate.state_dict()) == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
    }


def test_group_name_keeps_source_motion_windows_together():
    assert (
        group_name("walk3_subject1__frames_10_200__start_0_len_190")
        == "walk3_subject1"
    )
    assert (
        group_name("walk3_subject1__frames_300_500__start_0_len_200")
        == "walk3_subject1"
    )


def test_grouped_negatives_and_purged_temporal_positive_holdout():
    labels = np.r_[np.ones(500), np.zeros(9 * 20)]
    keys = np.array(
        ["moonwalk"] * 500
        + [f"ordinary_{i}" for i in range(9) for _ in range(20)]
    )
    sources = np.r_[np.zeros(500), np.ones(9 * 20)]
    masks, groups = split_samples(labels, keys, sources)
    parts = ["train", "validation", "test"]
    for i, first in enumerate(parts):
        assert len(np.unique(labels[masks[first]])) == 2
        for second in parts[i + 1 :]:
            assert not (masks[first] & masks[second]).any()
            assert not groups[first].intersection(groups[second])
    train = np.flatnonzero(masks["train"] & (labels == 1))
    validation = np.flatnonzero(masks["validation"] & (labels == 1))
    test = np.flatnonzero(masks["test"] & (labels == 1))
    assert validation.min() - train.max() > 32
    assert test.min() - validation.max() > 32
