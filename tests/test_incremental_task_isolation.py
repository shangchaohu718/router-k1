"""CPU checks for rewards, value networks, optimizers and stored task IDs."""

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tensordict import TensorDict

from holomotion.src.algo.algo_utils import PpoAuxTransition, RolloutStorage
from holomotion.src.algo.ppo import PPO
from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.agent_modules import PPOCritic
from holomotion.src.modules.incremental_critic import FrozenLegacySlidingCritic
from holomotion.src.motion_tracking.incremental_task import (
    SLIDING_TERMS,
    TASK_KEY,
    FixedTaskReward,
    config_hash,
    select_reward_profile,
    tensor_state_hash,
)

SCHEMA = {"current": {"seq_len": 1, "terms": ["x"]}}


def obs(task=1, batch=(4,)):
    return TensorDict(
        {
            "x": torch.randn(*batch, 3),
            TASK_KEY: torch.full((*batch, 1), task, dtype=torch.int64),
        },
        batch_size=batch,
    )


def pair(task="sliding"):
    cfg = {
        "type": "MLP",
        "output_dim": 1,
        "layer_config": {"hidden_dims": [8], "activation": "SiLU"},
        "obs_norm": {
            "enabled": True,
            "epsilon": 1e-8,
            "update_method": "ema",
            "ema_momentum": 0.1,
        },
    }
    old = PPOCritic(SCHEMA, cfg, obs_example=obs())
    old.obs_normalizer.update(torch.randn(10, 3) + 5)
    cfg["task_isolation"] = {
        "task": task,
        "bootstrap_from_legacy": True,
        "legacy_critic_config_hash": config_hash(cfg),
        "legacy_critic_state_hash": tensor_state_hash(old.state_dict()),
        "legacy_reward_hash": "old",
        "sliding_reward_hash": "new",
    }
    new = FrozenLegacySlidingCritic(SCHEMA, cfg, obs_example=obs())
    new.load_state_dict(old.state_dict(), strict=True)
    return old, new


def test_legacy_critic_and_statistics_unchanged_after_ten_adamw_updates():
    old, new = pair()
    baseline = copy.deepcopy(old.state_dict())
    optim = torch.optim.AdamW(
        [p for p in new.parameters() if p.requires_grad],
        lr=0.03,
        weight_decay=0.5,
    )
    assert {id(p) for g in optim.param_groups for p in g["params"]} == {
        id(p) for p in new.sliding.parameters()
    }
    old_storage = {
        p.untyped_storage().data_ptr() for p in new.legacy.parameters()
    }
    assert not old_storage.intersection(
        p.untyped_storage().data_ptr() for p in new.sliding.parameters()
    )
    before = tensor_state_hash(new.sliding.state_dict())
    for _ in range(10):
        new.train()
        assert not new.legacy.training
        optim.zero_grad(set_to_none=True)
        new(obs(), update_obs_norm=True, forward_chunk_size=2)[
            "values"
        ].square().mean().backward()
        assert all(p.grad is None for p in new.legacy.parameters())
        optim.step()
    assert tensor_state_hash(new.sliding.state_dict()) != before
    for key, value in baseline.items():
        assert torch.equal(new.legacy.state_dict()[key], value), key
    new.task = "legacy"
    sample = obs(0)
    actual = new(sample, update_obs_norm=True)["values"]
    assert not actual.requires_grad
    assert torch.equal(actual, old(sample, update_obs_norm=False)["values"])
    assert tensor_state_hash(new.legacy.state_dict()) == tensor_state_hash(
        baseline
    )


def test_strict_migration_full_restore_and_source_identity():
    old, new = pair()
    state = copy.deepcopy(new.state_dict())
    new.allow_legacy_bootstrap = False
    new.load_state_dict(state)
    for key in (
        "contract_digest",
        "isolation_version",
        "sliding.critic_module.output_head.bias",
    ):
        bad = dict(state)
        del bad[key]
        with pytest.raises(RuntimeError):
            new.load_state_dict(bad)
    for key in ("contract_digest", "isolation_version"):
        bad = copy.deepcopy(state)
        bad[key].add_(1)
        with pytest.raises(RuntimeError, match="mismatch"):
            new.load_state_dict(bad)
    with pytest.raises(RuntimeError, match="bootstrap"):
        new.load_state_dict(old.state_dict())
    new.allow_legacy_bootstrap = True
    bad = copy.deepcopy(old.state_dict())
    bad["critic_module.output_head.bias"].add_(1)
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        new.load_state_dict(bad)


def test_critic_rejects_unloaded_baseline_and_changed_configuration():
    _, loaded = pair()
    loaded._baseline_verified = False
    with pytest.raises(
        ValueError, match="baseline has not been strictly loaded"
    ):
        loaded(obs())
    cfg = {
        "type": "FrozenLegacySlidingCritic",
        "output_dim": 1,
        "task_isolation": {"legacy_critic_config_hash": "wrong"},
    }
    with pytest.raises(ValueError, match="configuration mismatch"):
        FrozenLegacySlidingCritic(SCHEMA, cfg, obs_example=obs())


@pytest.mark.parametrize("bad", [None, 0, 2, 0.5])
def test_critic_uses_stored_task_and_rejects_mismatches(bad):
    _, new = pair()
    td = obs()
    if bad is None:
        td.del_(TASK_KEY)
    else:
        td[TASK_KEY] = torch.full((4, 1), bad)
    with pytest.raises(ValueError, match="task ID"):
        new(td)
    seq = obs(batch=(2, 3)).flatten(0, 1)
    assert new(seq)["values"].shape == (6, 1)


def test_reward_profiles_independent_with_new_coefficient_changes():
    base = {
        "_config": {"reward_prefix": "ref_"},
        "is_alive": {"weight": 0.5, "params": {}},
    }
    recipe = copy.deepcopy(base)
    recipe.update(
        {
            k: {"weight": float(i + 1), "params": {}}
            for i, k in enumerate(SLIDING_TERMS)
        }
    )
    recipe["_config"]["incremental_task"] = {
        "task": "legacy",
        "legacy_reward_hash": config_hash(base),
    }
    original = copy.deepcopy(recipe)
    a = select_reward_profile(recipe)
    recipe[SLIDING_TERMS[0]]["weight"] *= 100
    b = select_reward_profile(recipe)
    assert a == b == base
    assert original[SLIDING_TERMS[0]]["weight"] == 1
    recipe["_config"]["incremental_task"]["task"] = "sliding"
    assert select_reward_profile(recipe)[SLIDING_TERMS[0]]["weight"] == 100
    recipe["is_alive"]["weight"] = 9
    with pytest.raises(ValueError, match="Legacy reward contract"):
        select_reward_profile(recipe)


def test_reward_rejects_wrong_reference_population_and_missing_identity():
    batch = SimpleNamespace(raw_motion_keys=["slide"])
    command = SimpleNamespace(
        _motion_cache=SimpleNamespace(current_batch=batch)
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command)
    )
    func = mock.Mock(return_value=torch.tensor([7.0]))
    func.__name__ = "score"
    reward = FixedTaskReward(
        func, {"task": "sliding", "sliding_motion_keys": ["slide"]}
    )
    assert reward(env).item() == 7
    for keys in (["stand"], ["slide", "stand"], None, []):
        batch.raw_motion_keys = keys
        with pytest.raises(ValueError):
            reward(env)
    assert func.call_count == 1


def test_actual_reward_builder_preserves_tensor_values_and_native_signature():
    # Execute the repository builder with CPU stubs for Isaac config types;
    # no SimulationApp or installed simulator modules are imported.
    from omegaconf import DictConfig, ListConfig, OmegaConf

    source = Path("holomotion/src/env/isaaclab_components/isaaclab_rewards.py")
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "build_rewards_config"
    )

    def is_alive(env, scale=1.0):
        return env.fixed * scale

    def score(env, scale=1.0):
        return env.fixed.square() * scale

    namespace = {
        "DictConfig": DictConfig,
        "ListConfig": ListConfig,
        "OmegaConf": OmegaConf,
        "RewardsCfg": SimpleNamespace,
        "RewardTermCfg": lambda **kw: SimpleNamespace(**kw),
        "resolve_holo_config": copy.deepcopy,
        "isaaclab_mdp": SimpleNamespace(),
        "is_alive": is_alive,
        **{key: score for key in SLIDING_TERMS},
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    build = namespace["build_rewards_config"]
    base = {
        "_config": {},
        "is_alive": {"weight": 2.0, "params": {"scale": 3.0}},
    }
    recipe = copy.deepcopy(base)
    recipe.update(
        {key: {"weight": 1.0, "params": {}} for key in SLIDING_TERMS}
    )
    recipe["_config"]["incremental_task"] = {
        "task": "legacy",
        "legacy_reward_hash": config_hash(base),
        "sliding_motion_keys": ["slide"],
    }
    env = SimpleNamespace(
        step_dt=0.02,
        fixed=torch.tensor([1.0, 2.0]),
        command_manager=SimpleNamespace(
            get_term=lambda _: SimpleNamespace(
                _motion_cache=SimpleNamespace(
                    current_batch=SimpleNamespace(raw_motion_keys=["stand"])
                )
            )
        ),
    )

    def total(config):
        terms = vars(build(config)).values()
        # Native RewardManager applies the control timestep once.
        return sum(
            term.weight * term.func(env, **term.params) * 0.02
            for term in terms
        )

    expected = torch.tensor([0.12, 0.24])
    a = total(recipe)
    recipe[SLIDING_TERMS[0]]["weight"] = 1000
    assert torch.equal(total(recipe), a)
    torch.testing.assert_close(a, expected)
    wrapped = build(recipe).is_alive.func
    assert inspect.signature(wrapped) == inspect.signature(is_alive)


def test_native_storage_gae_respects_terminal_reset_and_retains_task_labels():
    storage = RolloutStorage(1, 3, obs(batch=(1,)), (2,))
    storage.data["obs"][TASK_KEY].fill_(1)
    storage.data["rewards"][:] = torch.tensor([1.0, 2.0, 100.0]).reshape(
        3, 1, 1
    )
    storage.data["values"].zero_()
    storage.data["dones"][:] = torch.tensor([False, True, True]).reshape(
        3, 1, 1
    )
    storage.compute_returns(torch.tensor([[999.0]]), gamma=0.9, lam=0.8)
    torch.testing.assert_close(
        storage.data["advantages"].flatten(), torch.tensor([2.44, 2.0, 100.0])
    )
    # No value/trace from the reset episode enters the previous terminal.
    storage.data["rewards"][2] *= 100
    storage.compute_returns(torch.tensor([[999.0]]), gamma=0.9, lam=0.8)
    torch.testing.assert_close(
        storage.data["advantages"][:2].flatten(), torch.tensor([2.44, 2.0])
    )
    assert storage.data["obs"][TASK_KEY].dtype == torch.int64
    assert (
        storage.data["obs"].transpose(0, 1).flatten(0, 1)[TASK_KEY].eq(1).all()
    )


def test_returns_reject_mixed_task_storage_before_any_bootstrap():
    algo = PPOTF.__new__(PPOTF)
    algo.incremental_task_spec = {"task": "sliding"}
    storage = RolloutStorage(1, 3, obs(batch=(1,)), (2,))
    storage.data["obs"][TASK_KEY].fill_(1)
    storage.data["obs"][TASK_KEY][1].zero_()
    algo.storage = storage
    with pytest.raises(ValueError, match="Mixed or mismatched"):
        algo._compute_returns(obs(batch=(1,)))


def test_strict_optimizer_resume_rejects_bad_groups_and_moment_shapes():
    algo = PPOTF.__new__(PPOTF)
    algo.incremental_task_spec = {"task": "sliding"}
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    assert algo._restore_optimizer_state(
        optimizer, state, optimizer_name="critic"
    )
    bad = copy.deepcopy(state)
    bad["param_groups"][0]["params"] = []
    with pytest.raises(ValueError, match="group size mismatch"):
        algo._restore_optimizer_state(optimizer, bad, optimizer_name="critic")
    bad = copy.deepcopy(state)
    bad["state"][0]["exp_avg"] = torch.ones(3)
    with pytest.raises(ValueError, match="state shape mismatch"):
        algo._restore_optimizer_state(optimizer, bad, optimizer_name="critic")


def test_ppo_checkpoint_contract_checked_before_loading_models(tmp_path):
    from omegaconf import OmegaConf
    from test_internal_moe import make_pair

    old, actor = make_pair()
    algo = PPOTF.__new__(PPOTF)
    algo.actor = actor
    algo.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    algo.is_offline_eval = False
    algo.incremental_task_spec = {
        "task": "sliding",
        "bootstrap_from_legacy": False,
        "legacy_actor_state_hash": tensor_state_hash(old.state_dict()),
    }
    route = {"stage": "A", "mode": "cooperative", "warmup": True}
    algo.config = OmegaConf.create(
        {"module_dict": {"actor": {"internal_expansion": route}}}
    )
    metadata = algo._extra_checkpoint_state()
    path = tmp_path / "checkpoint.pt"
    torch.save(metadata, path)
    with mock.patch.object(PPO, "load", return_value="loaded") as parent:
        assert algo.load(str(path)) == "loaded"
        parent.assert_called_once()
    bad = copy.deepcopy(metadata)
    bad["incremental_task_contract"]["task"] = "legacy"
    torch.save(bad, path)
    with mock.patch.object(PPO, "load") as parent:
        with pytest.raises(
            ValueError, match="task checkpoint contract mismatch"
        ):
            algo.load(str(path))
        parent.assert_not_called()
    bad = copy.deepcopy(metadata)
    bad["incremental_actor_contract"]["warmup"] = False
    torch.save(bad, path)
    with mock.patch.object(PPO, "load") as parent:
        with pytest.raises(
            ValueError, match="routing checkpoint contract mismatch"
        ):
            algo.load(str(path))
        parent.assert_not_called()
    torch.save({"iter": 1}, path)
    with pytest.raises(
        ValueError, match="Missing incremental task checkpoint contract"
    ):
        algo.load(str(path))
    bad = copy.deepcopy(metadata)
    bad["incremental_implementation_contract"] = {}
    torch.save(bad, path)
    with pytest.raises(
        ValueError, match="implementation checkpoint contract mismatch"
    ):
        algo.load(str(path))


def test_value_loss_isolated_but_advantage_changes_actor_gradient():
    from test_internal_moe import make_pair
    from test_sliding_actor import make_obs

    _, actor = make_pair(warmup=True)
    _, critic = pair()
    assert not {
        p.untyped_storage().data_ptr() for p in actor.parameters()
    }.intersection(p.untyped_storage().data_ptr() for p in critic.parameters())
    actor.zero_grad(set_to_none=True)
    critic(obs())["values"].square().mean().backward()
    assert all(p.grad is None for p in actor.parameters())
    td, actions = make_obs((3, 4)), torch.randn(3, 4, 2)
    grads = []
    for advantage in (1.0, -1.0):
        actor.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        out = actor(
            td, actions=actions, mode="sequence_logp", update_obs_norm=False
        )
        (
            -out["actions_log_prob"].mean() * torch.tensor(advantage).detach()
        ).backward()
        grads.append(actor.actor_module.layers[0].added_down_proj.grad.clone())
        assert all(p.grad is None for p in critic.parameters())
    assert grads[0].abs().sum() > 0
    assert torch.equal(grads[0], -grads[1])


def test_finite_horizon_timeout_does_not_use_current_value_as_bootstrap():
    algo = PPOTF.__new__(PPOTF)
    algo.incremental_task_spec = {"task": "sliding"}
    algo.gamma = 0.9
    fields = {key: None for key in PpoAuxTransition.__expected_keys__}
    fields.update(obs=obs(), values=torch.full((4, 1), 100.0))
    algo.transition_td = PpoAuxTransition(**fields, batch_size=[4])
    algo.storage = SimpleNamespace(add=mock.Mock())
    algo._post_env_step_hook = mock.Mock()
    algo._kv_reset_pending = torch.zeros(4, dtype=torch.bool)
    algo.device = "cpu"
    algo.process_env_step(
        torch.ones(4, 1), torch.ones(4, 1), torch.ones(4, dtype=torch.bool), {}
    )
    saved = algo.storage.add.call_args.args[0]
    assert torch.equal(saved.rewards, torch.ones(4, 1))
    assert saved.dones.all()
    assert algo._kv_reset_pending.all()
