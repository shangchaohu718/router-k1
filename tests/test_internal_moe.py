"""Internal expansion has one trunk/cache and immutable original storage."""

import copy

import pytest
import torch
from test_sliding_actor import SCHEMA, make_config, make_obs

from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.agent_modules import PPOTFRefRouterActor
from holomotion.src.modules.internal_moe import PPOTFInternalMoEActor


def config(**expansion):
    cfg = make_config()
    cfg.pop("sliding_router")
    cfg["type"] = "InternalExpandedMoEPolicy"
    cfg["max_ctx_len"] = 4
    cfg["dense_layer_at_first"] = False
    cfg["internal_expansion"] = {
        "num_added_experts": 2,
        "bootstrap_from_legacy": True,
        "stage": "A",
        "mode": "cooperative",
        **expansion,
    }
    cfg["moe_load_balance"] = {"enabled": True}
    cfg["inactive_expert_margin_to_topk"] = {
        "enabled": True,
        "ratio_floor": 0.1,
    }
    return cfg


def make_pair(**expansion):
    cfg = config(**expansion)
    original_cfg = copy.deepcopy(cfg)
    original_cfg["type"] = "ReferenceRoutedGroupedMoETransformerPolicy"
    old = PPOTFRefRouterActor(
        SCHEMA, original_cfg, 2, 0.2, obs_example=make_obs()
    )
    old.obs_normalizer.update(torch.randn(10, 8) + 2)
    new = PPOTFInternalMoEActor(SCHEMA, cfg, 2, 0.2, obs_example=make_obs())
    new.load_state_dict(old.state_dict(), strict=True)
    return old.eval(), new


def assert_old_state_equal(old, new):
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key


def test_single_actor_and_strict_checkpoint_migration():
    old, new = make_pair()
    assert PPOTF._select_actor_wrapper_cls(config()) is PPOTFInternalMoEActor
    assert not hasattr(new.actor_module, "general")
    assert not hasattr(new.actor_module, "specialist")
    assert len(new.actor_module.layers) == 1
    assert_old_state_equal(old, new)
    bank = new.actor_module.layers[0]
    assert bank.gate_up_proj.data_ptr() != bank.added_gate_up_proj.data_ptr()
    assert torch.equal(bank.added_gate_up_proj, bank.gate_up_proj[:2])
    assert torch.equal(bank.added_down_proj, bank.down_proj[:2])
    bad = dict(old.state_dict())
    del bad["actor_module.obs_embed.0.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        new.load_state_dict(bad)
    full = dict(new.state_dict())
    del full["actor_module.layers.0.added_router.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        new.load_state_dict(full)
    full = dict(new.state_dict())
    del full["actor_module.layers.0.expansion_version"]
    with pytest.raises(RuntimeError, match="Missing key"):
        new.load_state_dict(full)
    new.allow_legacy_bootstrap = False
    with pytest.raises(RuntimeError, match="bootstrap_from_legacy"):
        new.load_state_dict(old.state_dict())


@pytest.mark.parametrize("score", ["softmax", "sigmoid"])
def test_legacy_exact_even_after_training_and_arbitrarily_high_new_scores(
    score,
):
    old, new = make_pair(warmup=True)
    layer = new.actor_module.layers[0]
    old.actor_module.layers[0].routing_score_fn = layer.routing_score_fn = (
        score
    )
    new.train()
    before = layer.added_down_proj.detach().clone()
    optimizer = torch.optim.AdamW(
        [p for p in new.parameters() if p.requires_grad],
        lr=0.01,
        weight_decay=0.2,
    )
    out = new(
        make_obs((3, 7)),
        actions=torch.randn(3, 7, 2),
        mode="sequence_logp",
        update_obs_norm=True,
    )
    (-out["actions_log_prob"].mean()).backward()
    assert layer.added_down_proj.grad.abs().sum() > 0
    assert layer.added_router.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, layer.added_down_proj)
    assert_old_state_equal(old, new)
    assert all(
        p.grad is None
        for name, p in new.named_parameters()
        if name in dict(old.named_parameters())
    )
    with torch.no_grad():
        layer.added_router.bias.fill_(1e6)
    new.actor_module.set_expansion("legacy")
    old.reset_kv_cache(3, "cpu")
    new.reset_kv_cache(3, "cpu")
    old.actor_module.set_collect_routing_stats(True)
    new.actor_module.set_collect_routing_stats(True)
    old_route_buffers = {
        name: tensor.clone()
        for name, tensor in layer.named_buffers()
        if name != "added_assignment_counts"
    }
    for t in range(13):
        if t == 6:
            old.clear_env_cache(torch.tensor([1]))
            new.clear_env_cache(torch.tensor([1]))
        obs = make_obs()
        expected = old(obs, mode="inference", update_obs_norm=False)
        actual = new(obs, mode="inference", update_obs_norm=True)
        torch.testing.assert_close(
            actual["mu"], expected["mu"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual["sigma"], expected["sigma"], rtol=0, atol=0
        )
    new.actor_module.apply_dynamic_bias_update_from_stats()
    for name, tensor in old_route_buffers.items():
        assert torch.equal(tensor, dict(layer.named_buffers())[name]), name
    assert_old_state_equal(old, new)


def test_native_top2_supports_all_three_pairings_and_warmup_gradients():
    _, actor = make_pair()
    layer = actor.actor_module.layers[0]
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.added_router.weight.zero_()
    for bias, new_count in (([-10, -11], 0), ([10, -10], 1), ([10, 11], 2)):
        with torch.no_grad():
            layer.added_router.bias.copy_(torch.tensor(bias))
        _, selected, _ = layer.compute_moe_ffn(x, x, return_routing_debug=True)
        assert ((selected >= 3).sum(-1) == new_count).all()
    layer.set_expansion("cooperative", "A", True)
    with torch.no_grad():
        layer.added_router.bias.fill_(-1e3)
    output, indices, _ = layer.compute_moe_ffn(x, x, return_routing_debug=True)
    assert (indices >= 3).any(-1).all()
    output.square().sum().backward()
    assert layer.added_gate_up_proj.grad.abs().sum() > 0
    assert layer.added_router.bias.grad.abs().sum() > 0


def test_stage_b_freezes_all_experts_and_disables_utilization_pressure():
    _, actor = make_pair()
    layer = actor.actor_module.layers[0]
    actor.actor_module.set_expansion("cooperative", stage="B")
    actor.train()
    trainable = [
        name
        for name, parameter in actor.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable == [
        "actor_module.layers.0.added_router.weight",
        "actor_module.layers.0.added_router.bias",
    ]
    layer.compute_moe_ffn(torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    assert layer.last_moe_load_balance_loss == 0
    assert layer.last_inactive_expert_margin_to_topk_loss == 0


def test_added_router_optimizer_rate_is_isolated_and_survives_lr_updates():
    _, actor = make_pair(router_lr_multiplier=300)
    groups = actor.optimizer_parameter_groups(adamw=True)
    optimizer = torch.optim.AdamW(groups, lr=3e-6)
    critic = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-5)
    algo = PPOTF.__new__(PPOTF)
    algo.actor_optimizer, algo.critic_optimizer = optimizer, critic
    algo.actor_learning_rate, algo.critic_learning_rate = 3e-6, 1e-5
    PPOTF._set_optimizer_learning_rates(algo)
    router_ids = {
        id(p) for p in actor.actor_module.layers[0].added_router.parameters()
    }
    grouped_ids = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            grouped_ids.append(id(parameter))
            assert group["lr"] == pytest.approx(
                9e-4 if id(parameter) in router_ids else 3e-6
            )
    assert len(grouped_ids) == len(set(grouped_ids)) == 4
    assert set(grouped_ids) == {
        id(p) for p in actor.parameters() if p.requires_grad
    }
    algo.actor_learning_rate = 1e-6
    PPOTF._set_optimizer_learning_rates(algo)
    assert max(g["lr"] for g in optimizer.param_groups) == pytest.approx(3e-4)


@pytest.mark.parametrize("warmup", [False, True])
def test_fully_frozen_eval_optimizer_has_empty_groups(warmup):
    _, actor = make_pair(stage="eval", warmup=False)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    optimizer = torch.optim.AdamW(
        actor.optimizer_parameter_groups(adamw=warmup)
    )
    assert all(not group["params"] for group in optimizer.param_groups)
    optimizer.step()
    for key, value in before.items():
        assert torch.equal(actor.state_dict()[key], value)


@pytest.mark.parametrize("warmup", [False, True])
def test_single_cache_chunking_wrap_reset_and_sequence_replay(warmup):
    _, actor = make_pair(warmup=warmup)
    actor.eval()
    chunked = copy.deepcopy(actor)
    actor.reset_kv_cache(3, "cpu")
    chunked.reset_kv_cache(3, "cpu")
    actor.actor_module.set_collect_routing_stats(True)
    chunked.actor_module.set_collect_routing_stats(True)
    seq = make_obs((3, 13))
    steps = []
    sampled_actions, sampled_logp = [], []
    mask = torch.ones(3, 13, 13, dtype=torch.bool).tril()
    # Actual native cached path retains max_ctx_len most recent tokens.
    positions = torch.arange(13)
    mask &= positions[None, :, None] - positions[None, None, :] < 4
    for t in range(13):
        if t == 6:
            actor.clear_env_cache(torch.tensor([1]))
            chunked.clear_env_cache(torch.tensor([1]))
            mask[1, 6:, :6] = False
        with torch.no_grad():
            a = actor(seq[:, t], mode="sampling", update_obs_norm=False)
            b = chunked(seq[:, t], mode="inference", forward_chunk_size=2)
        steps.append(a["mu"])
        sampled_actions.append(a["actions"])
        sampled_logp.append(a["actions_log_prob"])
        torch.testing.assert_close(a["mu"], b["mu"], atol=2e-6, rtol=2e-5)
    for name in (
        "_k_cache",
        "_v_cache",
        "_kv_cache_len",
        "_kv_cache_write_idx",
        "_kv_cache_abs_pos",
    ):
        torch.testing.assert_close(
            getattr(actor.actor_module, name),
            getattr(chunked.actor_module, name),
        )
    torch.testing.assert_close(
        actor.actor_module.layers[0].added_assignment_counts,
        chunked.actor_module.layers[0].added_assignment_counts,
    )
    replay = actor(
        seq,
        actions=torch.stack(sampled_actions, 1),
        mode="sequence_logp",
        attn_mask=mask,
    )
    torch.testing.assert_close(
        replay["mu"], torch.stack(steps, 1), atol=3e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        replay["actions_log_prob"],
        torch.stack(sampled_logp, 1).unsqueeze(-1),
        atol=3e-5,
        rtol=2e-5,
    )
    actor.load_state_dict(actor.state_dict())
    assert actor.actor_module._kv_cache_len.eq(0).all()


def test_mode_switch_invalidates_attention_cache_under_new_profile_contract():
    old, actor = make_pair()
    old.reset_kv_cache(3, "cpu")
    actor.reset_kv_cache(3, "cpu")
    previous = None
    for t in range(12):
        mode = "cooperative" if 3 <= t < 8 else "legacy"
        actor.actor_module.set_expansion(mode)
        if mode != previous:
            old.clear_env_cache(None)
            assert actor.actor_module._kv_cache_len.eq(0).all()
        previous = mode
        obs = make_obs()
        with torch.no_grad():
            expected = old(obs, mode="inference", update_obs_norm=False)
            actual = actor(obs, mode="inference", update_obs_norm=False)
        torch.testing.assert_close(
            actor.actor_module._k_cache,
            old.actor_module._k_cache,
            rtol=0,
            atol=0,
        )
        if mode == "legacy":
            torch.testing.assert_close(
                actual["mu"], expected["mu"], rtol=0, atol=0
            )


@torch.no_grad()
def test_legacy_rejects_long_rolling_mask_without_changing_native_paths():
    torch.manual_seed(1907)
    old, actor = make_pair(mode="legacy", stage="eval")
    seq = make_obs((3, 13))
    actions = torch.zeros(3, 13, 2)
    positions = torch.arange(13)
    causal = torch.ones(3, 13, 13, dtype=torch.bool).tril()
    causal[1, 6:, :6] = False
    rolling = causal & (
        positions[None, :, None] - positions[None, None, :] < 4
    )
    with pytest.raises(ValueError, match="Legacy long rolling-mask sequence"):
        actor(seq, actions=actions, mode="sequence_logp", attn_mask=rolling)

    # Full native sequences, including real episode resets, retain exactly
    # the old operation. Cached inference remains covered by the tests above.
    expected = old(
        seq,
        actions=actions,
        mode="sequence_logp",
        attn_mask=causal,
        update_obs_norm=False,
    )
    actual = actor(
        seq,
        actions=actions,
        mode="sequence_logp",
        attn_mask=causal,
        update_obs_norm=False,
    )
    assert torch.equal(actual["mu"], expected["mu"])
    assert torch.equal(actual["sigma"], expected["sigma"])

    # Context-sized packed windows are a supported diagnostic path: no
    # token's left edge advances within the sequence supplied to the model.
    packed = seq[:, -4:]
    packed_mask = torch.ones(3, 4, 4, dtype=torch.bool).tril()
    expected = old(
        packed,
        actions=actions[:, -4:],
        mode="sequence_logp",
        attn_mask=packed_mask,
        update_obs_norm=False,
    )
    actual = actor(
        packed,
        actions=actions[:, -4:],
        mode="sequence_logp",
        attn_mask=packed_mask,
        update_obs_norm=False,
    )
    assert torch.equal(actual["mu"], expected["mu"])
