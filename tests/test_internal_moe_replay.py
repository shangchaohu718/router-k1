"""Ordered teacher replay and both-sided Stage B parameter isolation."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_internal_moe import make_pair

from tools.booster_k1_retarget import fit_internal_moe_router as fitting
from tools.booster_k1_retarget.fit_internal_moe_router import (
    pack_windows,
    preservation_loss,
    read_ordered_recording,
    targets,
)


def recording(path, *, positions=None, steps=None):
    np.savez(
        path,
        schema_version=np.asarray(1),
        observations=np.arange(48, dtype=np.float32).reshape(6, 8),
        motion_keys=np.asarray(["old_clip"] * 6),
        batch_ids=np.zeros(6, dtype=np.int64),
        record_steps=np.arange(6) if steps is None else np.asarray(steps),
        reference_frames=np.arange(6),
        episode_positions=np.asarray(
            [0, 1, 2, 0, 1, 2] if positions is None else positions
        ),
    )


def test_replay_preserves_context_and_does_not_cross_episode_reset(tmp_path):
    path = tmp_path / "record.npz"
    recording(path)
    windows = read_ordered_recording(path, "old", context=2)
    assert [len(sequence) for sequence, _ in windows] == [1, 2, 2, 1, 2, 2]
    assert windows[3][0][0, 0] == 24
    obs, mask, labels = pack_windows(windows, 8, 2)
    assert obs.shape == (6, 2, 8)
    assert mask[3].tolist() == [[False, False], [False, True]]
    assert labels.eq(0).all()


def test_replay_rejects_missing_history_or_unspecified_transition_labels(
    tmp_path,
):
    path = tmp_path / "record.npz"
    recording(path, steps=[0, 2, 1, 3, 4, 5])
    with pytest.raises(ValueError, match="shuffled"):
        read_ordered_recording(path, "old", 4)
    recording(path, positions=[10, 11, 12, 13, 14, 15])
    with pytest.raises(ValueError, match="episode prefix"):
        read_ordered_recording(path, "new", 4)
    recording(path)
    with pytest.raises(ValueError, match="explicit new_frame_intervals"):
        read_ordered_recording(path, "transition", 4)
    windows = read_ordered_recording(
        path, "transition", 4, new_frame_intervals=[[1, 4]]
    )
    assert [label for _, label in windows] == [0, 1, 1, 1, 0, 0]


def test_stage_b_two_teachers_and_only_added_scores_update():
    _, new_teacher = make_pair()
    layer = new_teacher.actor_module.layers[0]
    with torch.no_grad():
        layer.added_router.weight.zero_()
        layer.added_router.bias.fill_(0.5)
        layer.added_down_proj.add_(0.5)
    new_teacher.actor_module.set_expansion("cooperative", stage="eval")
    old_teacher = copy.deepcopy(new_teacher)
    old_teacher.actor_module.set_expansion("legacy", stage="eval")
    student = copy.deepcopy(new_teacher)
    student.actor_module.set_expansion("cooperative", stage="B")
    obs, mask, labels = pack_windows(
        [(torch.randn(4, 8), 0), (torch.randn(2, 8), 1)], 8, 4
    )
    teacher = targets(old_teacher, new_teacher, obs, mask)
    before = {
        key: value.clone() for key, value in student.state_dict().items()
    }
    loss, metrics = preservation_loss(student, obs, mask, labels, teacher)
    assert metrics["new_behavior_kl"] == 0
    assert metrics["old_behavior_kl"] > 0
    assert metrics["old_route_kl"] > 0
    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=1e-2
    )
    loss.backward()
    optimizer.step()
    changed = [
        key
        for key, value in student.state_dict().items()
        if not torch.equal(value, before[key])
    ]
    assert changed
    assert all(
        key.startswith("actor_module.layers.0.added_router.")
        for key in changed
    )
    assert all(p.grad is None for p in old_teacher.parameters())
    assert all(p.grad is None for p in new_teacher.parameters())


def test_calibration_saves_strictly_restorable_actor_without_changing_experts(
    tmp_path, monkeypatch
):
    old, teacher = make_pair()
    teacher.actor_module.set_expansion("cooperative", stage="eval")
    baseline_state = {
        key: value.clone() for key, value in old.state_dict().items()
    }
    teacher.actor_module.layers[0].added_router.bias.data.fill_(0.5)
    checkpoint = tmp_path / "model_new.pt"
    torch.save({}, checkpoint)
    critic = checkpoint.with_suffix("") / "critic"
    critic.mkdir(parents=True)
    (critic / "unmodified.txt").write_text("critic")
    cfg = OmegaConf.create(
        {
            "modules": {"actor": {"internal_expansion": {}}},
            "algo": {"config": {"checkpoint_load_mode": "finetune"}},
        }
    )
    monkeypatch.setattr(fitting, "load_source", lambda _: (teacher, cfg))
    monkeypatch.setattr(fitting, "actor_state", lambda _: baseline_state)
    old_path, new_path = tmp_path / "old.npz", tmp_path / "new.npz"
    recording(old_path)
    recording(new_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {"path": str(old_path), "role": "old"},
                    {"path": str(new_path), "role": "new"},
                ]
            }
        )
    )
    args = SimpleNamespace(
        checkpoint=checkpoint,
        baseline=checkpoint,
        manifest=manifest,
        output_dir=tmp_path / "fit",
        steps=3,
        batch_size=2,
        stride=1,
        learning_rate=1e-3,
        route_weight=1.0,
        seed=7,
    )
    report = fitting.fit(args)
    assert report["all_experts_and_old_state_preserved"]
    assert not report["accepted"]
    from safetensors.torch import load_file

    state = load_file(
        str(args.output_dir / "model_route/actor/model.safetensors")
    )
    restored = copy.deepcopy(teacher)
    restored.allow_legacy_bootstrap = False
    restored.load_state_dict(state, strict=True)
    for key, value in teacher.state_dict().items():
        if ".added_router." not in key:
            assert torch.equal(value, state[key]), key
