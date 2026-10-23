"""Versioned resume progress and adaptive rates, without an environment."""

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tensordict import TensorDict

from holomotion.src.algo.ppo import PPO


def make_algo(path, *, actor_rate=0.001, critic_rate=0.002, mode="resume"):
    algo = PPO.__new__(PPO)
    algo.actor = torch.nn.Linear(2, 2)
    algo.critic = torch.nn.Linear(2, 1)
    algo.actor_learning_rate = actor_rate
    algo.critic_learning_rate = critic_rate
    algo.actor_optimizer = torch.optim.AdamW(
        [
            {
                "params": [algo.actor.weight],
                "lr_multiplier": 3,
                "lr": 3 * actor_rate,
            },
            {
                "params": [algo.actor.bias],
                "lr_multiplier": 1,
                "lr": actor_rate,
            },
        ]
    )
    algo.critic_optimizer = torch.optim.AdamW(
        [
            {
                "params": list(algo.critic.parameters()),
                "lr_multiplier": 2,
                "lr": 2 * critic_rate,
            }
        ]
    )
    algo.config = {"checkpoint_load_mode": mode}
    algo.device = torch.device("cpu")
    algo.is_main_process, algo.is_offline_eval = True, False
    algo.current_learning_iteration = 0
    algo._last_completed_iteration, algo._next_learning_iteration = -1, 0
    algo.log_dir = str(path)

    def save_model(model, directory):
        target = Path(directory)
        target.mkdir()
        torch.save(model.state_dict(), target / "model.pt")

    def load_model(model, target, strict):
        model.load_state_dict(
            torch.load(target, weights_only=True), strict=strict
        )

    algo.accelerator = SimpleNamespace(
        unwrap_model=lambda model: model,
        save_model=mock.Mock(side_effect=save_model),
        wait_for_everyone=lambda: None,
        end_training=lambda: None,
    )
    algo._resolve_model_file_path = lambda checkpoint, name: str(
        Path(checkpoint).with_suffix("") / name / "model.pt"
    )
    algo._load_accelerate_model = mock.Mock(side_effect=load_model)
    algo._release_cuda_cache = lambda: None
    algo.command_name = "synthetic_test"
    return algo


def save_completed(algo, last=4):
    # A tiny optimizer-state fixture; this is not an RL training invocation.
    for model, optimizer in (
        (algo.actor, algo.actor_optimizer),
        (algo.critic, algo.critic_optimizer),
    ):
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    algo.current_learning_iteration = last
    algo._post_update_hook({})
    path = Path(algo.log_dir) / f"model_{last}.pt"
    algo.save(path)
    return path


def fake_learning_loop(algo, iterations):
    """Exercise the real runner boundaries with no simulation or RL update."""
    obs = TensorDict({"x": torch.zeros(1, 2)}, batch_size=[1])
    algo.env = SimpleNamespace(reset_all=lambda: (obs, {}))
    algo._wrap_obs_dict = lambda value: value
    algo._ensure_storage = lambda _: None
    algo.rollout_policy = lambda value: value
    algo.num_learning_iterations = iterations
    algo.log_interval = algo.save_interval = 1
    algo._log_iteration = lambda **_: None
    algo.ep_infos = []
    observed = []

    def update():
        observed.append(algo.current_learning_iteration)
        algo._post_update_hook({})
        return {}

    algo.update = update
    algo.learn()
    return observed


def test_resume_learns_next_iterations_without_overwriting_prior(tmp_path):
    source = make_algo(tmp_path)
    path = save_completed(source, 4)
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = make_algo(tmp_path, actor_rate=0.02, critic_rate=0.03)
    restored.load(path)
    assert restored.current_learning_iteration == 5
    assert restored._next_learning_iteration == 5
    assert fake_learning_loop(restored, 2) == [5, 6]
    assert restored.current_learning_iteration == 7
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
        "model_4.pt",
        "model_5.pt",
        "model_6.pt",
    ]
    # Two real saves (two sidecars each), with the final duplicate skipped.
    assert restored.accelerator.save_model.call_count == 4
    assert fake_learning_loop(restored, 1) == [7]
    saved = torch.load(tmp_path / "model_7.pt", weights_only=False)
    assert saved["iter"] == 7
    assert saved["training_state"]["next_iteration"] == 8


def test_resume_restores_adaptive_rates_and_preserves_group_multipliers(
    tmp_path,
):
    source = make_algo(tmp_path)
    path = save_completed(source)
    restored = make_algo(tmp_path, actor_rate=0.03, critic_rate=0.04)
    restored.load(path)
    assert restored.actor_learning_rate == 0.001
    assert restored.critic_learning_rate == 0.002
    assert restored.actor_optimizer.state
    assert [
        group["lr"] for group in restored.actor_optimizer.param_groups
    ] == [0.003, 0.001]
    restored.desired_kl, restored.schedule = 0.01, "adaptive"
    restored.adaptive_lr_factor = 2
    restored.adaptive_lr_min, restored.adaptive_lr_max = 1e-6, 1
    restored.adaptive_lr_kl_high_factor = 2
    restored.adaptive_lr_kl_low_factor = 0.5
    restored.adaptive_lr_adapt_critic = True
    restored._apply_adaptive_lr(0.1)
    assert restored.actor_learning_rate == 0.0005
    assert restored.critic_learning_rate == 0.001
    assert [
        group["lr"] for group in restored.actor_optimizer.param_groups
    ] == [0.0015, 0.0005]
    assert restored.critic_optimizer.param_groups[0]["lr"] == 0.002


def test_zero_update_metadata_and_legacy_marker_start_at_iteration_zero(
    tmp_path,
):
    source = make_algo(tmp_path)
    path = tmp_path / "model_zero.pt"
    source.save(path, infos={"untrained_zero_update": True})
    metadata = torch.load(path, weights_only=False)
    assert metadata["iter"] == -1
    assert metadata["training_state"]["next_iteration"] == 0
    for legacy in (False, True):
        saved = copy.deepcopy(metadata)
        if legacy:
            del saved["training_state"]
            saved["iter"] = 0
        torch.save(saved, path)
        restored = make_algo(tmp_path)
        restored.load(path)
        assert restored._last_completed_iteration == -1
        assert restored.current_learning_iteration == 0


def test_legacy_resume_infers_completed_iteration_and_effective_base_lr(
    tmp_path,
):
    source = make_algo(tmp_path)
    path = save_completed(source, 12)
    saved = torch.load(path, weights_only=False)
    del saved["training_state"]
    torch.save(saved, path)
    restored = make_algo(tmp_path, actor_rate=0.02, critic_rate=0.03)
    restored.load(path)
    assert restored.current_learning_iteration == 13
    assert restored.actor_learning_rate == 0.001
    assert restored.critic_learning_rate == 0.002
    assert (
        restored._checkpoint_iteration_interpretation
        == "legacy_last_completed"
    )


@pytest.mark.parametrize("continue_iteration", [False, True])
def test_finetune_keeps_fresh_optimizers_and_config_rates(
    tmp_path, continue_iteration
):
    path = save_completed(make_algo(tmp_path), 7)
    restored = make_algo(
        tmp_path, actor_rate=0.03, critic_rate=0.04, mode="finetune"
    )
    restored.config["finetune_continue_iteration"] = continue_iteration
    restored.load(path)
    assert restored.current_learning_iteration == (
        8 if continue_iteration else 0
    )
    assert restored.actor_learning_rate == 0.03
    assert restored.critic_learning_rate == 0.04
    assert not restored.actor_optimizer.state
    assert not restored.critic_optimizer.state


def test_offline_uses_completed_iteration_and_cannot_enter_learning(tmp_path):
    path = save_completed(make_algo(tmp_path), 7)
    restored = make_algo(tmp_path, actor_rate=0.03, critic_rate=0.04)
    restored.is_offline_eval = True
    restored.load(path)
    assert restored.current_learning_iteration == 7
    assert restored._next_learning_iteration == 8
    assert restored.actor_learning_rate == 0.03
    assert not restored.actor_optimizer.state
    with pytest.raises(ValueError, match="Offline evaluation"):
        restored.learn()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("next_iteration", 77),
        ("last_completed_iteration", True),
        ("actor_learning_rate", float("nan")),
        ("critic_learning_rate", -1),
    ],
)
def test_malformed_versioned_state_fails_before_loading_weights(
    tmp_path, field, value
):
    path = save_completed(make_algo(tmp_path))
    saved = torch.load(path, weights_only=False)
    saved["training_state"][field] = value
    torch.save(saved, path)
    restored = make_algo(tmp_path)
    with pytest.raises(ValueError):
        restored.load(path)
    restored._load_accelerate_model.assert_not_called()


@pytest.mark.parametrize(
    "problem", ["ambiguous", "multiplier", "missing", "groups"]
)
def test_legacy_optimizer_ambiguity_is_rejected_before_weights(
    tmp_path, problem
):
    path = save_completed(make_algo(tmp_path))
    saved = torch.load(path, weights_only=False)
    del saved["training_state"]
    groups = saved["actor_optimizer_state_dict"]["param_groups"]
    if problem == "ambiguous":
        groups[1]["lr"] *= 2
    elif problem == "multiplier":
        groups[0]["lr_multiplier"] = 5
    elif problem == "missing":
        del groups[0]["lr"]
    else:
        groups.pop()
    torch.save(saved, path)
    restored = make_algo(tmp_path)
    with pytest.raises(ValueError):
        restored.load(path)
    restored._load_accelerate_model.assert_not_called()


def test_repeated_or_colliding_save_never_overwrites(tmp_path):
    algo = make_algo(tmp_path)
    path = save_completed(algo)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        algo.save(path)
    assert path.read_bytes() == before
    collision = tmp_path / "unrelated.pt"
    collision.with_suffix("").mkdir()
    with pytest.raises(FileExistsError):
        algo.save(collision)
    assert not collision.exists()


def test_zero_iteration_learn_rejects_before_environment_or_save(tmp_path):
    algo = make_algo(tmp_path)
    algo.num_learning_iterations = 0
    with pytest.raises(ValueError, match="num_learning_iterations > 0"):
        algo.learn()
    assert not list(tmp_path.iterdir())
