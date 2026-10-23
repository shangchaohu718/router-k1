"""PPO preserves the sampled environment and chunk layout in cached replay."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf
from test_incremental_g0_g2 import spec
from test_incremental_task_isolation import pair
from test_internal_moe import make_pair
from test_sliding_actor import make_obs

from holomotion.src.algo.algo_utils import PpoTransition, RolloutStorage
from holomotion.src.algo.ppo import PPO
from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.motion_tracking.incremental_task import (
    TASK_KEY,
    tensor_state_hash,
)
from holomotion.src.motion_tracking.task_contract import (
    critic_cycle_stamp,
    policy_stamp,
)


def rollout(*, replay="cached", chunk=2):
    _, actor = make_pair(
        warmup=True, sequence_replay=replay, replay_chunk_size=chunk
    )
    _, critic = pair()
    algo = PPOTF.__new__(PPOTF)
    algo.actor, algo.critic = actor, critic
    algo.device, algo.num_envs, algo.num_actions = torch.device("cpu"), 3, 2
    algo.incremental_contract = spec()
    algo.incremental_task_spec = {"task": "sliding"}
    algo._incremental_reward_terms = ["is_alive"]
    algo._incremental_policy_version = 0
    algo._incremental_sequence_replay = replay
    algo._incremental_replay_chunk_size = chunk
    algo.rollout_forward_chunk_size = chunk
    algo.configured_num_mini_batches = 1
    algo.requested_num_mini_batches = algo.num_mini_batches = 1
    algo.use_aux_state_pred = algo.use_velocity_transition = False
    algo.transition_cls = PpoTransition
    algo.accelerator = SimpleNamespace(
        unwrap_model=lambda model: model, autocast=nullcontext
    )
    algo.global_advantage_norm, algo.is_distributed = True, False
    algo._post_env_step_hook = lambda *args: None
    algo._kv_reset_pending = torch.zeros(3, dtype=torch.bool)
    algo.gamma, algo.lam = 0.9, 0.8
    actor.reset_kv_cache(3, "cpu")
    critic.freeze_cycle_statistics = True
    algo._incremental_rollout_stamp = policy_stamp(actor)
    algo._incremental_critic_stamp = critic_cycle_stamp(critic)
    algo._incremental_rollout_layout = (3, chunk)

    def observation():
        td = make_obs()
        td.set("x", torch.randn(3, 3))
        td.set("record_env", torch.arange(3).reshape(3, 1))
        td.set(TASK_KEY, torch.ones(3, 1, dtype=torch.int64))
        return algo._tag_incremental_observation(td)

    td = observation()
    algo.storage = RolloutStorage(3, 11, td, (2,))
    for step in range(11):
        if algo._kv_reset_pending.any():
            actor.clear_env_cache(algo._kv_reset_pending.nonzero().flatten())
            algo._kv_reset_pending.zero_()
        with torch.no_grad():
            actions = actor(
                td,
                mode="sampling",
                update_obs_norm=False,
                forward_chunk_size=chunk,
            )
            values = critic(td, update_obs_norm=True)
        algo.transition_td = algo._build_transition(td, actions, values)
        done = torch.tensor([step == 3, False, step == 7])
        truncated = torch.zeros(3, dtype=torch.bool)
        info = {
            "incremental_transition": {
                "terminated": done,
                "truncated": truncated,
                "executed_action": actions["actions"].detach().clone(),
                "joint_target": actions["actions"].detach().clone() * 0.5,
                "reward_components": {"is_alive": torch.ones(3, 3)},
            }
        }
        algo.process_env_step(torch.ones(3, 1), done, truncated, info)
        td = observation()
    algo._compute_returns(td)
    return algo


def test_real_rollout_replays_exactly_and_never_shuffles(monkeypatch):
    algo = rollout()

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached incremental replay must not randperm")

    monkeypatch.setattr(torch, "randperm", forbidden)
    algo._verify_incremental_recomputation(torch.arange(3))
    metrics = algo._incremental_recompute_metrics
    assert metrics["layouts_checked"] == 1
    assert metrics["mu"] == metrics["sigma"] == 0
    # Native sampling uses Normal.log_prob while sequence_logp uses the
    # existing analytic expression; retain its established 1e-4 tolerance.
    assert metrics["actions_log_prob"] < 1e-4
    batches = list(algo._sequence_batches(1, 3))
    assert len(batches) == 3
    for batch in batches:
        assert tuple(batch[0].batch_size) == (3, 11)
        assert torch.equal(batch[0]["record_env"][:, 0, 0], torch.arange(3))


@pytest.mark.parametrize("indices", [[2, 0, 1], [0, 1], [0, 0, 2]])
def test_cached_replay_rejects_permutation_subset_and_duplicate(indices):
    algo = rollout()
    indices = torch.tensor(indices)
    with pytest.raises(ValueError, match="original environment order"):
        algo._verify_incremental_recomputation(indices)
    with pytest.raises(ValueError, match="original environment order"):
        list(algo._sequence_batches(1, 1, env_indices=indices))


def test_cached_replay_rejects_partition_and_chunk_changes():
    algo = rollout()
    with pytest.raises(ValueError, match="num_mini_batches=1"):
        list(algo._sequence_batches(2, 1))
    algo.num_mini_batches = 2
    with pytest.raises(ValueError, match="num_mini_batches=1"):
        algo._verify_incremental_recomputation()
    algo.num_mini_batches = 1
    algo.rollout_forward_chunk_size = 1
    with pytest.raises(ValueError, match="replay_chunk_size"):
        algo._check_incremental_policy_lease()
    algo.rollout_forward_chunk_size = 2
    algo.actor.actor_module.replay_chunk_size = 1
    with pytest.raises(ValueError, match="replay_chunk_size"):
        algo._verify_incremental_recomputation()


def test_update_uses_canonical_indices_before_any_optimizer(monkeypatch):
    algo = rollout()
    captured = []

    def stop(indices):
        captured.append(indices.clone())
        raise RuntimeError("stop after pre-update layout validation")

    monkeypatch.setattr(algo, "_verify_incremental_recomputation", stop)
    with mock.patch.object(torch, "randperm", side_effect=AssertionError):
        with pytest.raises(RuntimeError, match="stop after pre-update"):
            algo.update()
    assert torch.equal(captured[0], torch.arange(3))


def setup_stub(**expansion):
    algo = PPOTF.__new__(PPOTF)
    algo.config = {
        "module_dict": {
            "actor": {
                "type": "InternalExpandedMoEPolicy",
                "internal_expansion": expansion,
            }
        }
    }
    algo.command_name = "ref_motion"
    algo.device, algo.num_envs = torch.device("cpu"), 3
    algo.configured_num_mini_batches = 1
    algo.requested_num_mini_batches = algo.num_mini_batches = 1
    algo.rollout_forward_chunk_size = 2
    return algo


@pytest.mark.parametrize("counter", ["configured", "requested", "effective"])
def test_setup_rejects_multiple_batches_including_ddp_expansion(counter):
    algo = setup_stub(sequence_replay="cached", replay_chunk_size=2)
    key = {
        "configured": "configured_num_mini_batches",
        "requested": "requested_num_mini_batches",
        "effective": "num_mini_batches",
    }[counter]
    setattr(algo, key, 2)
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(ValueError, match="num_mini_batches=1"):
            algo._setup_configs()


def test_setup_rejects_sampling_replay_chunk_mismatch():
    algo = setup_stub(sequence_replay="cached", replay_chunk_size=1)
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(ValueError, match="replay_chunk_size"):
            algo._setup_configs()


@pytest.mark.parametrize("source", ["flag", "accelerator", "world_size"])
def test_cached_replay_rejects_distributed_setup(source):
    algo = setup_stub(sequence_replay="cached", replay_chunk_size=2)
    if source == "flag":
        algo.is_distributed = True
    elif source == "accelerator":
        algo.accelerator = SimpleNamespace(num_processes=2)
    else:
        algo.gpu_world_size = 2
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(ValueError, match="single rank"):
            algo._setup_configs()


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("source", ["config", "accelerator"])
def test_cached_replay_rejects_amp_in_config_or_runtime(source, precision):
    algo = setup_stub(sequence_replay="cached", replay_chunk_size=2)
    if source == "config":
        algo.config["mixed_precision"] = precision
    else:
        algo.accelerator = SimpleNamespace(mixed_precision=precision)
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(ValueError, match="full precision"):
            algo._setup_configs()


@pytest.mark.parametrize("mode", ["closed", "learned"])
def test_stage_a_requires_forced_open_admission(mode):
    algo = setup_stub(admission={"mode": mode})
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(ValueError, match="admission.mode='open'"):
            algo._setup_configs()


def test_setup_accepts_open_admission_and_legacy_vectorized_config():
    for expansion in (
        {},
        {"admission": {"mode": "open"}},
        {
            "sequence_replay": "cached",
            "replay_chunk_size": 2,
            "admission": {"mode": "open"},
        },
    ):
        algo = setup_stub(**expansion)
        with mock.patch.object(PPO, "_setup_configs", return_value=None):
            algo._setup_configs()
        if expansion.get("sequence_replay") == "cached":
            assert torch.equal(
                algo._cached_incremental_indices(3), torch.arange(3)
            )
        else:
            assert algo._cached_incremental_indices(3) is None


def test_offline_load_only_ignores_admission_mode(tmp_path):
    old, actor = make_pair()
    algo = PPOTF.__new__(PPOTF)
    algo.actor = actor
    algo.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    algo.is_offline_eval = True
    algo.incremental_task_spec = {
        "task": "sliding",
        "bootstrap_from_legacy": False,
        "legacy_actor_state_hash": tensor_state_hash(old.state_dict()),
    }
    route = {
        "stage": "A",
        "mode": "cooperative",
        "warmup": True,
        "admission": {
            "mode": "open",
            "threshold": 0.99,
            "initial_bias": -8,
            "version": 1,
        },
    }
    algo.config = OmegaConf.create(
        {"module_dict": {"actor": {"internal_expansion": route}}}
    )
    metadata = algo._extra_checkpoint_state()
    path = tmp_path / "checkpoint.pt"
    torch.save(metadata, path)
    algo.config.module_dict.actor.internal_expansion.admission.mode = "closed"
    with mock.patch.object(PPO, "load", return_value="loaded"):
        assert algo.load(str(path)) == "loaded"
    for name, changed in (
        ("threshold", 0.9),
        ("initial_bias", -7),
        ("version", 2),
    ):
        bad = copy.deepcopy(metadata)
        bad["incremental_actor_contract"]["admission"][name] = changed
        torch.save(bad, path)
        with mock.patch.object(PPO, "load") as parent:
            with pytest.raises(ValueError, match="routing checkpoint"):
                algo.load(str(path))
            parent.assert_not_called()
    torch.save(metadata, path)
    algo.is_offline_eval = False
    with mock.patch.object(PPO, "load") as parent:
        with pytest.raises(ValueError, match="routing checkpoint"):
            algo.load(str(path))
        parent.assert_not_called()
