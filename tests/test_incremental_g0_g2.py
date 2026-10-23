"""G0-G2 contracts and counterexamples to false retention claims."""

import copy
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.motion_tracking.incremental_task import (
    TASK_KEY,
    tensor_state_hash,
)
from holomotion.src.motion_tracking.task_contract import (
    TaskSpec,
    check_episode_tasks,
    critic_cycle_stamp,
    policy_stamp,
    tag_rollout,
    validate_episode_ledger,
    validate_rollout,
)
from holomotion.src.motion_tracking.task_gae import normalize_by_task, task_gae


def spec(task="sliding"):
    hashes = {
        k: "a" * 64
        for k in TaskSpec.__dataclass_fields__
        if k.endswith("_hash")
    }
    return TaskSpec(
        task,
        f"{task}_reward_v1",
        critic_id=f"{task}_critic_v1",
        gamma=0.9,
        gae_lambda=0.8,
        timeout_semantics="finite_horizon_terminal",
        **hashes,
    )


def test_immutable_task_spec_pins_gamma_reference_and_termination():
    a = spec()
    with pytest.raises(FrozenInstanceError):
        a.gamma = 0.2
    for name, value in (
        ("gamma", 0.8),
        ("reference_hash", "b" * 64),
        ("termination_hash", "c" * 64),
    ):
        assert replace(a, **{name: value}).digest != a.digest
    assert TaskSpec(**asdict(a)).digest == a.digest
    with pytest.raises(ValueError, match="discount"):
        replace(a, gamma=float("nan"))


def test_task_reward_critic_policy_and_route_ids_are_checked_before_replay():
    td = TensorDict(
        {TASK_KEY: torch.ones(2, 1, dtype=torch.int64)}, batch_size=[2]
    )
    tag_rollout(td, spec(), policy_version=3, route_version=7)
    validate_rollout(
        td, spec(), policy_version=3, route_version=7, route_profile=1
    )
    for name in (
        "task_id",
        "reward_id",
        "critic_id",
        "policy_version",
        "route_version",
        "route_profile",
    ):
        bad = td.clone()
        bad["incremental_" + name][0] += 1
        with pytest.raises(ValueError, match="mismatch"):
            validate_rollout(
                bad, spec(), policy_version=3, route_version=7, route_profile=1
            )
    bad = td.clone()
    bad["incremental_valid_mask"][0] = False
    with pytest.raises(ValueError, match="unpadded"):
        validate_rollout(bad, spec())
    with pytest.raises(ValueError, match="TaskSpec"):
        validate_rollout(td, replace(spec(), gamma=0.8))


def test_gae_and_task_moments_immune_to_other_task_reward_scale():
    rewards = torch.tensor([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]).unsqueeze(-1)
    values = torch.zeros_like(rewards)
    tasks = torch.tensor([[0, 1], [0, 1], [0, 1]]).unsqueeze(-1)
    terminal = torch.zeros_like(tasks, dtype=torch.bool)
    terminal[-1] = True
    kwargs = dict(discounts={0: (0.9, 0.8), 1: (0.95, 0.7)})
    a, r = task_gae(
        rewards, values, values, terminal, terminal & False, tasks, **kwargs
    )
    changed = rewards.clone()
    changed[:, 1] *= 1000
    b, rb = task_gae(
        changed, values, values, terminal, terminal & False, tasks, **kwargs
    )
    na, ma = normalize_by_task(a, tasks)
    nb, mb = normalize_by_task(b, tasks)
    assert torch.equal(a[:, 0], b[:, 0]) and torch.equal(r[:, 0], rb[:, 0])
    assert torch.equal(na[:, 0], nb[:, 0]) and ma[0] == mb[0]
    assert not torch.equal(r[:, 1], rb[:, 1])
    assert not a.requires_grad and not r.requires_grad


def test_terminal_truncation_and_final_observation_have_distinct_bootstrap():
    rewards = torch.ones(1, 3, 1)
    values = torch.full_like(rewards, 2)
    reset_values = torch.full_like(rewards, 999)
    terminated = torch.tensor([True, False, False]).reshape(1, 3, 1)
    truncated = torch.tensor([False, True, False]).reshape(1, 3, 1)
    tasks = torch.ones(1, 3, 1, dtype=torch.int64)
    with pytest.raises(ValueError, match="pre-reset final"):
        task_gae(
            rewards,
            values,
            reset_values,
            terminated,
            truncated,
            tasks,
            discounts={1: (0.9, 0.8)},
            finite_horizon=False,
        )
    a, _ = task_gae(
        rewards,
        values,
        reset_values,
        terminated,
        truncated,
        tasks,
        final_values=torch.full_like(values, 10),
        discounts={1: (0.9, 0.8)},
        finite_horizon=False,
    )
    torch.testing.assert_close(a.flatten(), torch.tensor([-1.0, 8.0, 898.1]))
    finite, _ = task_gae(
        rewards,
        values,
        reset_values,
        terminated,
        truncated,
        tasks,
        discounts={1: (0.9, 0.8)},
        finite_horizon=True,
    )
    assert finite[0, 1, 0] == -1


def test_task_change_without_reset_rejected_and_valid_reset_cuts_trace():
    tasks = torch.tensor([0, 1]).reshape(2, 1, 1)
    dones = torch.zeros_like(tasks, dtype=torch.bool)
    with pytest.raises(ValueError, match="without an episode reset"):
        check_episode_tasks(tasks, dones)
    dones[0] = True
    check_episode_tasks(tasks, dones)
    a, _ = task_gae(
        torch.tensor([1.0, 1000.0]).reshape(2, 1, 1),
        torch.zeros(2, 1, 1),
        torch.zeros(2, 1, 1),
        dones,
        dones & False,
        tasks,
        discounts={0: (0.9, 0.8), 1: (0.9, 0.8)},
    )
    assert a[0].item() == 1


def test_episode_ledger_rejects_missing_reset_or_wrong_timestep():
    td = TensorDict(
        {
            "incremental_episode_id": torch.tensor([0, 0, 1]).reshape(3, 1, 1),
            "incremental_timestep": torch.tensor([3, 4, 0]).reshape(3, 1, 1),
        },
        batch_size=[3, 1],
    )
    dones = torch.tensor([False, True, False]).reshape(3, 1, 1)
    validate_episode_ledger(td, dones)
    td["incremental_timestep"][2] = 5
    with pytest.raises(ValueError, match="reset boundary"):
        validate_episode_ledger(td, dones)


def test_masked_normalization_ignores_padding_and_has_fixed_ddp_collectives():
    a = torch.tensor([1.0, 3.0, 10000.0]).reshape(3, 1, 1)
    tasks = torch.zeros_like(a, dtype=torch.int64)
    valid = torch.tensor([True, True, False]).reshape_as(tasks)
    calls = []

    def reduce(value, reduction):
        calls.append(value.clone())
        return value * 2

    n, moments = normalize_by_task(
        a, tasks, valid, accelerator=SimpleNamespace(reduce=reduce)
    )
    assert n.flatten().tolist() == [-1.0, 1.0, 0.0]
    assert moments[0]["count"] == 4
    assert len(calls) == 2 and calls[1][-1] == 0


def test_critic_registry_rows_and_statistics_only_update_sliding():
    from test_incremental_task_isolation import obs, pair

    old, critic = pair()
    td = obs()
    td[TASK_KEY][:2] = 0
    old_hash = tensor_state_hash(critic.legacy.state_dict())
    expected = old(td[:2], update_obs_norm=False)["values"]
    count = int(critic.sliding.obs_normalizer.count)
    values = critic.evaluate_registered_tasks(td, update_obs_norm=True)[
        "values"
    ]
    assert torch.equal(values[:2], expected)
    assert int(critic.sliding.obs_normalizer.count) == count + 2
    values[2:].square().sum().backward()
    assert all(p.grad is None for p in critic.legacy.parameters())
    assert tensor_state_hash(critic.legacy.state_dict()) == old_hash
    assert set(critic.registry) == {"legacy_critic_v1", "sliding_critic_v1"}


def test_critic_statistics_frozen_during_cycle_then_globally_updated():
    from test_incremental_task_isolation import obs, pair

    _, critic = pair()
    td = obs()
    critic.freeze_cycle_statistics = True
    old_hash = tensor_state_hash(critic.legacy.state_dict())
    before = tensor_state_hash(critic.sliding.obs_normalizer.state_dict())
    first = critic(td, update_obs_norm=True)["values"].detach()
    second = critic(td, update_obs_norm=True)["values"].detach()
    assert torch.equal(first, second)
    assert (
        tensor_state_hash(critic.sliding.obs_normalizer.state_dict()) == before
    )
    critic.finish_statistics_cycle(td)
    assert (
        tensor_state_hash(critic.sliding.obs_normalizer.state_dict()) != before
    )
    assert tensor_state_hash(critic.legacy.state_dict()) == old_hash


def test_frozen_legacy_can_still_drift_when_new_scores_win():
    from test_internal_moe import make_pair
    from test_sliding_actor import make_obs

    old, actor = make_pair(warmup=False)
    raw = make_obs((3, 4))
    old_state = copy.deepcopy(old.state_dict())
    actor.actor_module.set_expansion("legacy", stage="eval")
    expected = actor(raw, actions=torch.zeros(3, 4, 2), mode="sequence_logp")[
        "mu"
    ]
    with torch.no_grad():
        actor.actor_module.layers[0].added_router.bias.fill_(1000)
        actor.actor_module.layers[0].added_down_proj.mul_(4)
    actor.actor_module.set_expansion("cooperative", stage="eval")
    actual = actor(raw, actions=torch.zeros(3, 4, 2), mode="sequence_logp")[
        "mu"
    ]
    assert (actor.actor_module.layers[0].last_topk_indices >= 3).all()
    assert (actual - expected).abs().max() > 1e-3
    assert all(
        torch.equal(v, actor.state_dict()[k]) for k, v in old_state.items()
    )


def test_real_actor_ten_adamw_steps_keep_all_old_parameters_and_buffers():
    from test_internal_moe import make_pair
    from test_sliding_actor import make_obs

    old, actor = make_pair(warmup=True)
    old_state = copy.deepcopy(old.state_dict())
    old_buffers = copy.deepcopy(dict(old.named_buffers()))
    optimizer = torch.optim.AdamW(
        actor.optimizer_parameter_groups(adamw=True),
        lr=0.001,
        weight_decay=0.2,
    )
    for _ in range(10):
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        result = actor(
            make_obs((3, 4)),
            actions=torch.randn(3, 4, 2),
            mode="sequence_logp",
            update_obs_norm=True,
        )
        (-result["actions_log_prob"].mean()).backward()
        optimizer.step()
        assert all(
            torch.equal(v, actor.state_dict()[k]) for k, v in old_state.items()
        )
        assert all(
            torch.equal(v, dict(actor.named_buffers())[k])
            for k, v in old_buffers.items()
        )


def test_policy_lease_detects_mutation_and_profile_cache_is_invalidated():
    from test_incremental_task_isolation import pair
    from test_internal_moe import make_pair
    from test_sliding_actor import make_obs

    _, actor = make_pair()
    actor.reset_kv_cache(3, "cpu")
    actor(make_obs(), mode="inference")
    algo = PPOTF.__new__(PPOTF)
    algo.actor = actor
    _, critic = pair()
    algo.critic = critic
    algo.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    algo._incremental_rollout_stamp = policy_stamp(actor)
    algo._incremental_critic_stamp = critic_cycle_stamp(critic)
    algo._check_incremental_policy_lease()
    with torch.no_grad():
        actor.actor_module.layers[0].added_router.bias.add_(0.01)
    with pytest.raises(ValueError, match="changed between rollout"):
        algo._check_incremental_policy_lease()
    actor.actor_module.set_expansion("legacy", stage="eval")
    assert actor.actor_module._kv_cache_len.eq(0).all()
    assert actor.actor_module.route_revision >= 1


@torch.no_grad()
def test_sequence_causality_and_checkpoint_roundtrip_outputs_routes():
    from test_internal_moe import make_pair
    from test_sliding_actor import make_obs

    _, actor = make_pair(warmup=False)
    td = make_obs((3, 6))
    other = td.clone()
    for _, value in other.items(include_nested=True, leaves_only=True):
        value[:, 3:] += 20
    actions = torch.zeros(3, 6, 2)
    original = actor(td, actions=actions, mode="sequence_logp")["mu"]
    changed = actor(other, actions=actions, mode="sequence_logp")["mu"]
    # Existing grouped-FP32 sequence/cache budget: changing future tokens
    # changes grouped dispatch sizes and may change last-bit GEMM rounding.
    torch.testing.assert_close(
        original[:, :3], changed[:, :3], atol=3e-6, rtol=2e-5
    )
    restored = copy.deepcopy(actor)
    restored.load_state_dict(actor.state_dict(), strict=True)
    expected = actor(td, actions=actions, mode="sequence_logp")
    selections = actor.actor_module.layers[0].last_topk_indices.clone()
    actual = restored(td, actions=actions, mode="sequence_logp")
    assert torch.equal(expected["mu"], actual["mu"])
    assert torch.equal(expected["sigma"], actual["sigma"])
    assert torch.equal(
        selections, restored.actor_module.layers[0].last_topk_indices
    )


def test_zero_slice_gradient_does_not_freeze_packed_adamw_parameter():
    packed = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([packed], lr=0.1, weight_decay=0.1)
    packed.grad = torch.tensor([0.0, 1.0])
    optimizer.step()
    assert packed[0] != 1


def test_fixed_task_rollout_storage_gae_and_recompute_guard():
    from test_incremental_task_isolation import pair
    from test_internal_moe import make_pair
    from test_sliding_actor import make_obs

    from holomotion.src.algo.algo_utils import PpoTransition, RolloutStorage

    _, actor = make_pair(warmup=True)
    _, critic = pair()
    algo = PPOTF.__new__(PPOTF)
    algo.actor, algo.critic = actor, critic
    algo.device, algo.num_envs, algo.num_actions = torch.device("cpu"), 3, 2
    algo.incremental_contract = spec()
    algo.incremental_task_spec = {"task": "sliding"}
    algo._incremental_reward_terms = ["is_alive"]
    algo._incremental_policy_version = 0
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

    def observation():
        td = make_obs()
        td.set("x", torch.randn(3, 3))
        td.set(TASK_KEY, torch.ones(3, 1, dtype=torch.int64))
        return algo._tag_incremental_observation(td)

    td = observation()
    algo.storage = RolloutStorage(3, 6, td, (2,))
    for step in range(6):
        if algo._kv_reset_pending.any():
            actor.clear_env_cache(
                torch.nonzero(algo._kv_reset_pending).flatten()
            )
            algo._kv_reset_pending.zero_()
        with torch.no_grad():
            actions = actor(td, mode="sampling", update_obs_norm=False)
            values = critic(td, update_obs_norm=True)
        algo.transition_td = algo._build_transition(td, actions, values)
        done = torch.tensor([step == 2, False, False])
        truncated = torch.zeros(3, dtype=torch.bool)
        components = torch.ones(3, 3)
        info = {
            "incremental_transition": {
                "terminated": done,
                "truncated": truncated,
                "executed_action": actions["actions"].detach().clone(),
                "joint_target": actions["actions"].detach().clone() * 0.5,
                "reward_components": {"is_alive": components},
            }
        }
        algo.process_env_step(torch.ones(3, 1), done, truncated, info)
        td = observation()
    algo._compute_returns(td)
    assert torch.isfinite(algo.storage.data["advantages"]).all()
    algo._verify_incremental_recomputation()
    assert algo._incremental_recompute_metrics["mu"] < 5e-5
    algo.num_mini_batches = 2
    algo._verify_incremental_recomputation(torch.tensor([2, 0, 1]))
    assert algo._incremental_recompute_metrics["layouts_checked"] == 3
    # A batch-sensitive forward is a counterexample: full rollout passes,
    # but the exact minibatch layout must fail before any optimizer step.
    original_forward = actor.forward

    def batch_sensitive_forward(td, **kwargs):
        result = original_forward(td, **kwargs)
        if td.batch_size[0] < 3:
            result.set("mu", result["mu"] + 0.1)
        return result

    actor.forward = batch_sensitive_forward
    algo._verify_incremental_recomputation()
    with pytest.raises(ValueError, match="no optimizer update allowed"):
        algo._verify_incremental_recomputation(torch.tensor([2, 0, 1]))
    actor.forward = original_forward
    assert torch.equal(
        algo.storage.data["actions"],
        algo.storage.data["obs"]["incremental_executed_action"],
    )
    assert not torch.equal(
        algo.storage.data["actions"],
        algo.storage.data["obs"]["incremental_joint_target"],
    )
    algo.storage.data["mu"][0, 0] += 0.1
    with pytest.raises(ValueError, match="no optimizer update allowed"):
        algo._verify_incremental_recomputation()


def test_complete_config_contract_checks_registry_gamma_and_legacy_route():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from holomotion.src.motion_tracking.task_contract import build_task_spec

    with initialize_config_dir(
        config_dir=str(Path("holomotion/config").resolve()), version_base=None
    ):
        cfg = compose(
            config_name="training/motion_tracking/motion_tracking_booster_k1_moonwalk_task_isolated",
            overrides=[
                "experiment_name=cpu_contract",
                "headless=true",
                "num_processes=1",
                "main_process=true",
                "process_id=0",
            ],
        )
    actor = OmegaConf.to_container(cfg.modules.actor, resolve=True)
    critic = OmegaConf.to_container(cfg.modules.critic, resolve=True)
    env = OmegaConf.to_container(cfg.env.config, resolve=True)
    a = build_task_spec(actor, critic, env, {"gamma": 0.9, "lam": 0.8})
    b = build_task_spec(actor, critic, env, {"gamma": 0.8, "lam": 0.8})
    assert a.digest != b.digest
    altered_actor = copy.deepcopy(actor)
    altered_actor["n_heads"] = 8
    with pytest.raises(ValueError, match="Legacy Actor configuration"):
        build_task_spec(altered_actor, critic, env, {"gamma": 0.9, "lam": 0.8})
    wrong = copy.deepcopy(env)
    wrong["rewards"]["_config"]["incremental_task"]["sliding_motion_keys"] = [
        "wrong"
    ]
    with pytest.raises(ValueError, match="Complete Reward/Critic"):
        build_task_spec(actor, critic, wrong, {"gamma": 0.9, "lam": 0.8})
    critic["task_isolation"]["task"] = "legacy"
    env["rewards"]["_config"]["incremental_task"]["task"] = "legacy"
    with pytest.raises(
        ValueError, match="Legacy task requires legacy routing"
    ):
        build_task_spec(actor, critic, env, {"gamma": 0.9, "lam": 0.8})


def _ddp_moments_worker(rank, rendezvous, output):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2
    )

    def reduce(value, reduction):
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    # No samples of task 1 on rank 0; both ranks must still enter task 1's
    # collective. Changing rank 1's reward scale cannot move task 0 moments.
    values = (
        torch.tensor([1.0, 3.0])
        if rank == 0
        else torch.tensor([1000.0, 3000.0])
    )
    tasks = torch.full((2, 1, 1), rank, dtype=torch.int64)
    normalized, moments = normalize_by_task(
        values.reshape(2, 1, 1),
        tasks,
        accelerator=SimpleNamespace(reduce=reduce),
    )
    from test_incremental_task_isolation import obs, pair

    torch.manual_seed(123)
    _, critic = pair()
    old_hash = tensor_state_hash(critic.legacy.state_dict())
    batch = obs()
    batch["x"].fill_(10 * rank)
    critic.finish_statistics_cycle(batch, SimpleNamespace(reduce=reduce))
    assert tensor_state_hash(critic.legacy.state_dict()) == old_hash
    torch.save(
        critic.sliding.obs_normalizer.state_dict(), f"{output}/norm_{rank}.pt"
    )
    torch.save(
        {"normalized": normalized, "moments": moments}, f"{output}/{rank}.pt"
    )
    dist.destroy_process_group()


def test_two_process_cpu_ddp_task_statistics_with_disjoint_rank_populations(
    tmp_path,
):
    import torch.multiprocessing as mp

    rendezvous = "file://" + str(tmp_path / "gloo_init")
    mp.start_processes(
        _ddp_moments_worker,
        args=(rendezvous, str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    for rank in (0, 1):
        result = torch.load(tmp_path / f"{rank}.pt", weights_only=False)
        assert result["normalized"].flatten().tolist() == [-1.0, 1.0]
        assert result["moments"][0] == {"count": 2, "mean": 2.0, "std": 1.0}
        assert result["moments"][1] == {
            "count": 2,
            "mean": 2000.0,
            "std": 1000.0,
        }
    a = torch.load(tmp_path / "norm_0.pt", weights_only=True)
    b = torch.load(tmp_path / "norm_1.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_reference_contract_hashes_contents_and_rejects_unbounded_population(
    tmp_path,
):
    from holomotion.src.motion_tracking.incremental_task import config_hash
    from holomotion.src.motion_tracking.task_contract import reference_manifest

    command = {
        "ref_motion": {
            "params": {
                "motion_lib_cfg": {
                    "train_hdf5_roots": [str(tmp_path)],
                    "val_hdf5_roots": [],
                }
            }
        }
    }
    path = tmp_path / "clip.h5"
    path.write_bytes(b"original")
    before = config_hash(reference_manifest(command))
    path.write_bytes(b"changed")
    assert config_hash(reference_manifest(command)) != before
    for i in range(128):
        (tmp_path / f"extra_{i}.h5").touch()
    with pytest.raises(ValueError, match="bounded-manifest budget"):
        reference_manifest(command)


def test_reference_manifest_pins_external_shards_and_clip_slices(tmp_path):
    import json

    from holomotion.src.motion_tracking.incremental_task import config_hash
    from holomotion.src.motion_tracking.task_contract import reference_manifest

    root = tmp_path / "subset"
    root.mkdir()
    shard = tmp_path / "external.h5"
    shard.write_bytes(b"clip contents")
    manifest = {
        "version": 2,
        "hdf5_shards": [{"file": str(shard)}, {"file": "/unused.h5"}],
        "clips": {"old": {"shard": 0, "start": 2, "length": 4}},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    command = {
        "ref_motion": {
            "params": {
                "motion_lib_cfg": {
                    "train_hdf5_roots": [str(root)],
                    "val_hdf5_roots": [],
                }
            }
        }
    }
    before = config_hash(reference_manifest(command))
    assert len(reference_manifest(command)) == 2
    shard.write_bytes(b"changed contents")
    changed = config_hash(reference_manifest(command))
    assert changed != before
    manifest["clips"]["old"]["start"] = 3
    path.write_text(json.dumps(manifest))
    assert config_hash(reference_manifest(command)) != changed
