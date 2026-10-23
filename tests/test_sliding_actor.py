"""Controller isolation, checkpoint migration, and native PPO replay."""

import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from holomotion.src.algo.ppo import PPO
from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.agent_modules import PPOTFRefRouterActor
from holomotion.src.modules.sliding_actor import PPOTFFrozenBaseSlidingActor


def make_obs(batch=(3,)):
    return TensorDict(
        {
            "unified": TensorDict(
                {
                    "actor_ref_dof_pos_cur": torch.randn(*batch, 2),
                    "actor_dof_pos": torch.randn(*batch, 2),
                    "actor_ref_dof_pos_fut": torch.randn(*batch, 2, 2),
                },
                batch_size=batch,
            )
        },
        batch_size=batch,
    )


def make_config(**router):
    return {
        "type": "FrozenBaseSlidingPolicy",
        "num_fine_experts": 3,
        "num_shared_experts": 1,
        "top_k": 2,
        "use_dynamic_bias": False,
        "obs_embed_mlp_hidden": 16,
        "router_embed_mlp_hidden": 16,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 1,
        "max_ctx_len": 16,
        "ff_mult": 1.0,
        "ff_mult_dense": 2,
        "attn_dropout": 0.0,
        "mlp_dropout": 0.0,
        "noise_std_type": "log",
        "obs_norm": {
            "enabled": True,
            "epsilon": 1e-8,
            "update_method": "ema",
            "ema_momentum": 0.1,
        },
        "output_dim": 2,
        "sliding_router": {
            "mode": "specialist",
            "bootstrap_from_general": True,
            **router,
        },
    }


SCHEMA = {
    "current": {
        "seq_len": 1,
        "terms": ["unified/actor_ref_dof_pos_cur", "unified/actor_dof_pos"],
    },
    "future": {"seq_len": 2, "terms": ["unified/actor_ref_dof_pos_fut"]},
}


def make_actor(config=None):
    return PPOTFFrozenBaseSlidingActor(
        SCHEMA, config or make_config(), 2, 0.2, obs_example=make_obs()
    )


def make_pair(**router):
    config = make_config(**router)
    old_config = copy.deepcopy(config)
    old_config["type"] = "ReferenceRoutedGroupedMoETransformerPolicy"
    original = PPOTFRefRouterActor(
        SCHEMA, old_config, 2, 0.2, obs_example=make_obs()
    )
    original.obs_normalizer.update(torch.randn(15, 8) + 2)
    actor = make_actor(config)
    actor.load_state_dict(original.state_dict(), strict=True)
    return original, actor


def test_factory_and_strict_bootstrap():
    assert (
        PPOTF._select_actor_wrapper_cls(make_config())
        is PPOTFFrozenBaseSlidingActor
    )
    original, actor = make_pair()
    for name, value in original.actor_module.state_dict().items():
        torch.testing.assert_close(
            actor.actor_module.general.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actor.actor_module.specialist.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )
    bad = dict(original.state_dict())
    del bad["actor_module.obs_embed.0.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        make_actor().load_state_dict(bad)
    with pytest.raises(
        RuntimeError, match="requires sliding_router.bootstrap"
    ):
        make_actor(make_config(bootstrap_from_general=False)).load_state_dict(
            original.state_dict()
        )


def test_base_path_exact_after_specialist_optimizer_and_normalizer_freeze():
    original, actor = make_pair()
    actor.train()
    general_before = {
        k: v.clone()
        for k, v in actor.actor_module.general.state_dict().items()
    }
    norm_before = {
        k: v.clone() for k, v in actor.obs_normalizer.state_dict().items()
    }
    sigma_before = actor.log_std.clone()
    seq = make_obs((3, 5))
    optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad], lr=0.01
    )
    before = actor.actor_module.specialist.action_mu_head[-1].weight.clone()
    out = actor(
        seq,
        actions=torch.randn(3, 5, 2),
        mode="sequence_logp",
        update_obs_norm=True,
    )
    (-out["actions_log_prob"].mean()).backward()
    optimizer.step()
    assert not torch.equal(
        before, actor.actor_module.specialist.action_mu_head[-1].weight
    )
    for k, v in general_before.items():
        torch.testing.assert_close(
            actor.actor_module.general.state_dict()[k], v, rtol=0, atol=0
        )
    for k, v in norm_before.items():
        torch.testing.assert_close(
            actor.obs_normalizer.state_dict()[k], v, rtol=0, atol=0
        )
    torch.testing.assert_close(actor.log_std, sigma_before, rtol=0, atol=0)
    assert all(p.grad is None for p in actor.actor_module.general.parameters())
    actor.actor_module.set_routing_mode("base")
    obs = make_obs()
    expected = original(obs, mode="inference", update_obs_norm=False)
    actual = actor(obs, mode="inference", update_obs_norm=True)
    torch.testing.assert_close(actual["mu"], expected["mu"], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["sigma"], expected["sigma"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "mode", ["base", "specialist", "soft", "hard", "temporal_hard"]
)
def test_sequence_replays_rollout_with_both_caches_and_episode_reset(mode):
    _, actor = make_pair(mode=mode, train_gate=mode == "soft", threshold=0.95)
    # Make selection observable by giving the branches different outputs.
    with torch.no_grad():
        actor.actor_module.specialist.action_mu_head[-1].bias.add_(0.25)
    actor.eval()
    sequence = make_obs((3, 7))
    actor.reset_kv_cache(3, "cpu")
    means = []
    for t in range(7):
        if t == 3:
            actor.clear_env_cache(torch.tensor([1]))
        means.append(actor(sequence[:, t], mode="inference")["mu"])
    means = torch.stack(means, dim=1)
    mask = torch.ones(3, 7, 7, dtype=torch.bool).tril()
    mask[1, 3:, :3] = False
    replay = actor(
        sequence, actions=means, mode="sequence_logp", attn_mask=mask
    )
    torch.testing.assert_close(replay["mu"], means, rtol=2e-5, atol=2e-6)
    for branch in (actor.actor_module.general, actor.actor_module.specialist):
        assert branch._kv_cache_len.tolist() == [7, 4, 7]


def test_temporal_gate_dwell_hysteresis_reset_and_sequence_state_isolation():
    _, actor = make_pair(
        mode="temporal_hard",
        threshold=0.95,
        exit_threshold=0.7,
        enter_steps=2,
        exit_steps=2,
    )
    core = actor.actor_module
    # One-frame spikes do not enter; intermediate confidence does not exit.
    probabilities = torch.tensor(
        [
            [0.99, 0.8, 0.99, 0.99, 0.8, 0.6, 0.8, 0.6, 0.6],
            [0.99, 0.99, 0.8, 0.99, 0.99, 0.99, 0.8, 0.8, 0.8],
        ]
    ).unsqueeze(-1)
    mask = torch.ones(2, 9, 9, dtype=torch.bool).tril()
    mask[1, 3:, :3] = False
    expected = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1],
        ]
    )
    core.reset_kv_cache(2, "cpu")
    actual = []
    for t in range(9):
        if t == 3:
            core.clear_env_cache(torch.tensor([1]))
        actual.append(core._temporal_weights(probabilities[:, t]))
    torch.testing.assert_close(
        torch.stack(actual, 1).squeeze(-1), expected.float()
    )
    live_states = [
        getattr(core, name).clone()
        for name in ("_slide_active", "_enter_count", "_exit_count")
    ]
    replay = core._temporal_weights(probabilities, mask)
    torch.testing.assert_close(replay.squeeze(-1), expected.float())
    for name, before in zip(
        ("_slide_active", "_enter_count", "_exit_count"),
        live_states,
        strict=True,
    ):
        torch.testing.assert_close(getattr(core, name), before)
    assert not any(
        "_slide_active" in key or "_enter_count" in key
        for key in actor.state_dict()
    )
    actor.load_state_dict(copy.deepcopy(actor.state_dict()))
    assert core._slide_active.tolist() == [0, 0]
    assert core._enter_count.tolist() == [0, 0]
    assert core._exit_count.tolist() == [0, 0]


@pytest.mark.parametrize("chunk_size", [0, 2])
def test_temporal_gate_actual_branch_selection_and_chunked_cache(chunk_size):
    original, actor = make_pair(
        mode="temporal_hard",
        threshold=0.95,
        exit_threshold=0.7,
        enter_steps=2,
        exit_steps=2,
    )
    expert = copy.deepcopy(original)
    with torch.no_grad():
        actor.actor_module.specialist.action_mu_head[-1].bias.add_(1)
        expert.actor_module.load_state_dict(
            actor.actor_module.specialist.state_dict()
        )
    for controller in (original, expert, actor):
        controller.eval()
        controller.reset_kv_cache(3, "cpu")
    probabilities = [0.99, 0.99, 0.8, 0.6, 0.6, 0.99, 0.99]
    expected_weights = [0, 1, 1, 1, 0, 0, 1]
    for p, selected in zip(probabilities, expected_weights, strict=True):
        obs = make_obs()
        with (
            torch.no_grad(),
            mock.patch.object(
                actor.actor_module,
                "gate_logits",
                return_value=torch.full((3, 1), p).logit(),
            ),
        ):
            base = original(obs, mode="inference", update_obs_norm=False)["mu"]
            slide = expert(obs, mode="inference", update_obs_norm=False)["mu"]
            result = actor(
                obs, mode="inference", forward_chunk_size=chunk_size
            )
        assert result["slide_gate_weight"].flatten().tolist() == [selected] * 3
        torch.testing.assert_close(
            result["mu"], slide if selected else base, rtol=1e-5, atol=1e-6
        )
    for branch in (actor.actor_module.general, actor.actor_module.specialist):
        assert branch._kv_cache_abs_pos.tolist() == [7, 7, 7]


@pytest.mark.parametrize(
    "config",
    [
        dict(threshold=0.5, exit_threshold=0.7),
        dict(threshold=0.95, enter_steps=0),
        dict(threshold=0.95, exit_steps=0),
    ],
)
def test_temporal_gate_rejects_invalid_configuration(config):
    with pytest.raises(ValueError, match="Temporal gate"):
        make_actor(make_config(mode="temporal_hard", **config))


def test_chunking_matches_full_batch_and_keeps_unselected_branch_history():
    _, full = make_pair(mode="hard")
    chunked = copy.deepcopy(full)
    for actor in (full, chunked):
        actor.reset_kv_cache(3, "cpu")
    for _ in range(4):
        obs = make_obs()
        expected = full(obs, mode="inference")
        actual = chunked(obs, mode="inference", forward_chunk_size=2)
        torch.testing.assert_close(actual["mu"], expected["mu"])
    assert chunked.actor_module.specialist._kv_cache_abs_pos.tolist() == [
        4,
        4,
        4,
    ]


def test_gate_supervision_uses_reference_only_and_routes_old_samples_exactly():
    original, actor = make_pair(
        mode="hard", train_specialist=False, train_gate=True
    )
    obs = make_obs()
    # Zero initialization would make feature-isolation testing vacuous.
    with torch.no_grad():
        actor.actor_module.gate[0].weight.fill_(0.05)
        actor.actor_module.gate[-1].weight.fill_(0.02)
    logits = actor.actor_module.gate_logits(
        actor._normalize_actor_obs(actor.assembler(obs), False)
    )
    changed = obs.clone()
    changed["unified", "actor_dof_pos"].fill_(100)
    torch.testing.assert_close(
        logits,
        actor.actor_module.gate_logits(
            actor._normalize_actor_obs(actor.assembler(changed), False)
        ),
    )
    changed_ref = obs.clone()
    changed_ref["unified", "actor_ref_dof_pos_cur"].add_(20)
    ref_logits = actor.actor_module.gate_logits(
        actor._normalize_actor_obs(actor.assembler(changed_ref), False)
    )
    assert not torch.allclose(logits, ref_logits)
    loss = actor.gate_supervision_loss(obs, torch.tensor([0.0, 1.0, 0.0]))
    loss.backward()
    assert actor.actor_module.gate[-1].bias.grad is not None
    assert all(
        p.grad is None for p in actor.actor_module.specialist.parameters()
    )
    with torch.no_grad():
        actor.actor_module.specialist.action_mu_head[-1].bias.add_(10)
        actor.actor_module.gate[-1].weight.zero_()
        actor.actor_module.gate[-1].bias.fill_(-4)
    torch.testing.assert_close(
        actor(obs, mode="inference")["mu"],
        original(obs, mode="inference", update_obs_norm=False)["mu"],
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="finite values"):
        actor.gate_supervision_loss(
            obs, torch.tensor([0.0, float("nan"), 1.0])
        )


@pytest.mark.parametrize("chunk_size", [0, 2])
def test_actual_hard_switches_keep_both_histories_through_wrap_and_resets(
    chunk_size,
):
    original, actor = make_pair(mode="hard")
    expert = copy.deepcopy(original)
    with torch.no_grad():
        actor.actor_module.specialist.action_mu_head[-1].bias.add_(1)
        expert.actor_module.load_state_dict(
            actor.actor_module.specialist.state_dict()
        )
        first, last = actor.actor_module.gate[0], actor.actor_module.gate[-1]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1
        last.weight.zero_()
        last.weight[0, 0] = 8
        last.bias.fill_(-1)
    # Ring-buffer wrap occurs several times; independent reference policies
    # consume the same actual observations regardless of selected controller.
    for module in (
        original.actor_module,
        expert.actor_module,
        actor.actor_module.general,
        actor.actor_module.specialist,
    ):
        module.max_ctx_len = 4
    for controller in (original, expert, actor):
        controller.eval()
        controller.reset_kv_cache(3, "cpu")
    general_before = {
        k: v.clone()
        for k, v in actor.actor_module.general.state_dict().items()
    }
    for step in range(13):
        if step in {3, 9}:
            ids = torch.tensor([1])
            for controller in (original, expert, actor):
                controller.clear_env_cache(ids)
        if step == 11:
            for controller in (original, expert, actor):
                controller.clear_env_cache(None)
        obs = make_obs()
        desired = torch.tensor([(step + i) % 2 for i in range(3)]).bool()
        obs["unified", "actor_ref_dof_pos_cur"][:, 0] = torch.where(
            desired, 30.0, -30.0
        )
        with torch.no_grad():
            base = original(obs, mode="inference", update_obs_norm=False)
            slide = expert(obs, mode="inference", update_obs_norm=False)
            output = actor(
                obs, mode="inference", forward_chunk_size=chunk_size
            )
        assert (
            output["slide_gate_weight"].squeeze(-1).bool().tolist()
            == desired.tolist()
        )
        expected = torch.where(desired[:, None], slide["mu"], base["mu"])
        torch.testing.assert_close(
            output["mu"], expected, rtol=1e-5, atol=1e-6
        )
        for branch, reference in (
            (actor.actor_module.general, original.actor_module),
            (actor.actor_module.specialist, expert.actor_module),
        ):
            for name in (
                "_k_cache",
                "_v_cache",
                "_kv_cache_len",
                "_kv_cache_write_idx",
                "_kv_cache_abs_pos",
            ):
                torch.testing.assert_close(
                    getattr(branch, name), getattr(reference, name)
                )
        torch.testing.assert_close(
            actor.actor_module.general._kv_cache_abs_pos,
            actor.actor_module.specialist._kv_cache_abs_pos,
        )
    for key, value in general_before.items():
        torch.testing.assert_close(
            actor.actor_module.general.state_dict()[key], value, rtol=0, atol=0
        )


def test_soft_gate_intentionally_changes_old_output_but_base_mode_does_not():
    original, actor = make_pair(mode="soft")
    obs = make_obs()
    with torch.no_grad():
        actor.actor_module.specialist.action_mu_head[-1].bias.add_(2)
        actor.actor_module.gate[-1].weight.zero_()
        actor.actor_module.gate[-1].bias.zero_()
        base = original(obs, mode="inference", update_obs_norm=False)["mu"]
        soft = actor(obs, mode="inference")["mu"]
    torch.testing.assert_close(soft, base + 1)
    actor.actor_module.set_routing_mode("base")
    torch.testing.assert_close(
        actor(obs, mode="inference")["mu"], base, rtol=0, atol=0
    )


def test_frozen_base_survives_mode_changes_and_has_no_storage_aliases():
    _, actor = make_pair()
    general = actor.actor_module.general
    specialist = actor.actor_module.specialist
    for (name, p), (other_name, q) in zip(
        general.named_parameters(), specialist.named_parameters(), strict=True
    ):
        assert name == other_name and p.data_ptr() != q.data_ptr()
    for training in (False, True, False, True):
        actor.train(training)
        actor.actor_module.set_collect_routing_stats(True)
        assert not general.training and not actor.obs_normalizer.training
        assert not any(p.requires_grad for p in general.parameters())
        assert not actor.log_std.requires_grad
        assert all(
            not layer.collect_routing_stats
            for layer in general.layers
            if hasattr(layer, "collect_routing_stats")
        )


def test_upper_gate_masks_state_coupled_reference_columns_only():
    actor = make_actor(make_config(zero_feature_columns=[0, 1]))
    branch = actor.actor_module
    with torch.no_grad():
        branch.gate[0].weight.fill_(0.1)
        branch.gate[-1].weight.fill_(0.1)
    x = torch.randn(3, 8)
    changed = x.clone()
    changed[:, :2] += 50
    torch.testing.assert_close(
        branch.gate_logits(x), branch.gate_logits(changed)
    )
    assert not torch.allclose(
        branch.general.sequence_mu(x[:, None]),
        branch.general.sequence_mu(changed[:, None]),
    )
    assert all("_gate_feature_mask" not in key for key in actor.state_dict())


def test_composite_roundtrip_is_strict_and_preserves_freeze_state():
    _, actor = make_pair(mode="soft", train_gate=True)
    saved = copy.deepcopy(actor.state_dict())
    restored = make_actor(
        make_config(mode="soft", train_gate=True, bootstrap_from_general=False)
    )
    restored.load_state_dict(saved)
    restored.train()
    assert not restored.actor_module.general.training
    assert not restored.obs_normalizer.training
    assert not any(
        p.requires_grad for p in restored.actor_module.general.parameters()
    )
    obs = make_obs()
    torch.testing.assert_close(
        restored(obs, mode="inference")["mu"],
        actor(obs, mode="inference")["mu"],
    )
    del saved["actor_module.gate.2.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(saved)
    with pytest.raises(NotImplementedError, match="two-cache"):
        restored.export_onnx("unused.onnx")


def test_soft_ppo_gate_gradient_with_frozen_controllers():
    _, actor = make_pair(mode="soft", train_gate=True, train_specialist=False)
    with torch.no_grad():
        actor.actor_module.specialist.action_mu_head[-1].bias.add_(0.5)
    out = actor(
        make_obs((3, 4)), actions=torch.randn(3, 4, 2), mode="sequence_logp"
    )
    (-out["actions_log_prob"].mean()).backward()
    assert actor.actor_module.gate[-1].bias.grad.abs().sum() > 0
    assert all(p.grad is None for p in actor.actor_module.general.parameters())
    assert all(
        p.grad is None for p in actor.actor_module.specialist.parameters()
    )


def test_loading_checkpoint_invalidates_both_warm_caches():
    _, actor = make_pair()
    actor.reset_kv_cache(3, "cpu")
    with torch.no_grad():
        for _ in range(2):
            actor(make_obs(), mode="inference")
    for branch in (actor.actor_module.general, actor.actor_module.specialist):
        assert branch._kv_cache_abs_pos.tolist() == [2, 2, 2]
    actor.load_state_dict(copy.deepcopy(actor.state_dict()))
    for branch in (actor.actor_module.general, actor.actor_module.specialist):
        for name in (
            "_k_cache",
            "_v_cache",
            "_kv_cache_len",
            "_kv_cache_write_idx",
            "_kv_cache_abs_pos",
        ):
            assert torch.count_nonzero(getattr(branch, name)) == 0
    assert actor.actor_module.last_gate_probability is None
    assert actor.actor_module.last_gate_weight is None


def test_evaluation_gate_recorder_captures_only_reference_features(tmp_path):
    import atexit

    import numpy as np

    from tools.booster_k1_retarget.evaluate_sliding_router import (
        attach_gate_recorder,
    )

    _, actor = make_pair()
    batch = SimpleNamespace(
        motion_keys=["moonwalk_clip", "walk_clip", "dance_clip"]
    )
    command = SimpleNamespace(
        _motion_cache=SimpleNamespace(
            current_batch=batch,
            motion_keys_for_indices=lambda ids: [
                batch.motion_keys[i] for i in ids.tolist()
            ],
        ),
        _clip_indices=torch.tensor([1, 0, 2]),
        _frame_indices=torch.tensor([40, 8, 16]),
    )
    algo = SimpleNamespace(
        actor=actor,
        accelerator=SimpleNamespace(unwrap_model=lambda value: value),
        env=SimpleNamespace(
            _env=SimpleNamespace(
                command_manager=SimpleNamespace(
                    get_term=lambda _: command,
                )
            )
        ),
    )
    destination = tmp_path / "gate.npz"
    save = attach_gate_recorder(algo, destination)
    try:
        obs = make_obs()
        with torch.no_grad():
            actor(obs, mode="inference")
        save()
        with np.load(destination, allow_pickle=False) as saved:
            assert saved["features"].shape == (
                3,
                actor.actor_module.general.router_input_dim,
            )
            assert saved["labels"].tolist() == [0, 1, 0]
            assert saved["motion_keys"].tolist() == [
                "walk_clip",
                "moonwalk_clip",
                "dance_clip",
            ]
            assert saved["reference_frames"].tolist() == [40, 8, 16]
            assert saved["env_ids"].tolist() == [0, 1, 2]
            assert saved["schema_version"] == 2
            np.testing.assert_allclose(saved["gate_weight"], np.ones(3))
            expected = actor._normalize_actor_obs(actor.assembler(obs), False)
            expected = expected.index_select(
                -1, actor.actor_module.general._router_feature_indices
            )
            np.testing.assert_allclose(saved["features"], expected.numpy())
    finally:
        atexit.unregister(save)


@pytest.mark.parametrize("mode", ["specialist", "soft"])
@pytest.mark.parametrize("optimizer_type", ["AdamW", "Adam"])
def test_native_model_setup_optimizes_only_requested_branch(
    mode,
    optimizer_type,
):
    actor_config = make_config(
        mode=mode,
        train_specialist=mode == "specialist",
        train_gate=mode == "soft",
    )
    actor_config["obs_schema"] = SCHEMA
    algo = PPOTF.__new__(PPOTF)
    algo.config = OmegaConf.create(
        {
            "module_dict": {
                "actor": actor_config,
                "critic": {
                    "type": "MLP",
                    "obs_schema": {
                        "flat": {"seq_len": 1, "terms": ["critic_obs"]}
                    },
                    "output_dim": 1,
                    "layer_config": {"hidden_dims": [8], "activation": "SiLU"},
                },
            },
            "init_noise_std": 0.2,
            "min_sigma": 0.01,
            "max_sigma": 1.2,
        }
    )
    obs = make_obs().to_dict()
    obs["critic_obs"] = torch.randn(3, 4)
    algo.env = SimpleNamespace(num_envs=3, reset_all=lambda: (obs, {}))
    algo.device = torch.device("cpu")
    algo.num_actions = 2
    algo.is_main_process = False
    algo.use_aux_router_command_recon = False
    algo.use_aux_router_switch_penalty = False
    algo.optimizer_type = optimizer_type
    algo.actor_learning_rate = algo.critic_learning_rate = 1e-4
    algo.actor_beta1 = algo.critic_beta1 = 0.9
    algo.actor_beta2 = algo.critic_beta2 = 0.999
    algo.accelerator = SimpleNamespace(
        prepare=lambda *args: args, unwrap_model=lambda model: model
    )
    algo._setup_models_and_optimizer()
    optimized = {
        id(p)
        for group in algo.actor_optimizer.param_groups
        for p in group["params"]
    }
    branch = algo.actor.actor_module
    expected = branch.specialist if mode == "specialist" else branch.gate
    assert optimized == {
        id(p) for p in expected.parameters() if p.requires_grad
    }
    assert not optimized.intersection(
        id(p) for p in branch.general.parameters()
    )


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"checkpoint_load_mode": "resume"}, "fresh optimizer"),
        ({"override_sigma": True}, "override_sigma=false"),
        ({"export_policy": True}, "ONNX export"),
        (
            {
                "module_dict": {
                    "actor": make_config(mode="temporal_hard", threshold=0.95)
                }
            },
            "evaluation modes",
        ),
    ],
)
def test_native_setup_rejects_incompatible_bootstrap_options(changes, error):
    algo = PPOTF.__new__(PPOTF)
    algo.config = OmegaConf.create(
        {
            "module_dict": {"actor": make_config()},
            "checkpoint_load_mode": "finetune",
            **changes,
        }
    )
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(ValueError, match=error):
            algo._setup_configs()
