"""Hard admission preserves native closed rows and strict state lineage."""

import copy

import pytest
import torch
from test_cached_replay import (
    assert_snapshot,
    episode_mask,
    sample_reference,
    snapshot,
)
from test_internal_moe import assert_old_state_equal, make_pair
from test_sliding_actor import make_obs
from torch.nn import functional as F


def admission_pair(mode="closed", **kwargs):
    return make_pair(
        admission={"mode": mode, "threshold": 0.99, "initial_bias": -8.0},
        sequence_replay="cached",
        replay_chunk_size=2,
        **kwargs,
    )


@torch.no_grad()
@pytest.mark.parametrize("score", ["softmax", "sigmoid"])
def test_closed_gate_is_native_bitwise_with_unbounded_new_scores(score):
    old, actor = admission_pair()
    layer = actor.actor_module.layers[0]
    old.actor_module.layers[0].routing_score_fn = score
    layer.routing_score_fn = score
    layer.added_router.bias.fill_(1e6)
    layer.added_down_proj.add_(100)
    actor.actor_module.set_expansion(
        "cooperative", stage="eval", admission_mode="closed"
    )
    old.reset_kv_cache(3, "cpu")
    actor.reset_kv_cache(3, "cpu")
    for step in range(11):
        if step == 5:
            old.clear_env_cache(torch.tensor([1]))
            actor.clear_env_cache(torch.tensor([1]))
        td = make_obs()
        expected = old(td, mode="inference", update_obs_norm=False)
        actual = actor(td, mode="inference", update_obs_norm=True)
        assert torch.equal(actual["mu"], expected["mu"])
        assert torch.equal(actual["sigma"], expected["sigma"])
        assert (layer.last_topk_indices < layer.num_fine_experts).all()
        assert not layer.last_admission_mask.any()
    assert_old_state_equal(old, actor)


@torch.no_grad()
@pytest.mark.parametrize("score", ["softmax", "sigmoid"])
@pytest.mark.parametrize("route", ["cooperative", "new_only"])
def test_mixed_gate_keeps_closed_rows_native_and_all_routes_finite(
    score, route
):
    torch.manual_seed(809)
    old, actor = admission_pair("learned")
    original = old.actor_module.layers[0]
    layer = actor.actor_module.layers[0]
    original.routing_score_fn = layer.routing_score_fn = score
    actor.actor_module.set_expansion(
        route, stage="eval", admission_mode="learned"
    )
    layer.added_admission.weight.zero_()
    layer.added_admission.weight[0, 0] = 1
    layer.added_admission.bias.zero_()
    layer.added_router.bias.fill_(1e6)
    layer.added_down_proj.add_(10)
    x = torch.randn(3, 1, layer.d_model)
    reference = torch.randn_like(x)
    reference[:, 0, 0] = torch.tensor([-10.0, 10.0, -10.0])
    expected, _, _ = original.compute_moe_ffn(
        x, reference, return_routing_debug=True
    )
    actual, indices, _ = layer.compute_moe_ffn(
        x, reference, return_routing_debug=True
    )
    closed = torch.tensor([True, False, True])
    assert torch.equal(actual[closed], expected[closed])
    assert not torch.equal(actual[~closed], expected[~closed])
    assert torch.isfinite(actual).all()
    assert torch.isfinite(layer.last_topk_weights).all()
    assert torch.isfinite(layer.last_moe_load_balance_loss).all()
    assert torch.isfinite(layer.last_inactive_expert_margin_to_topk_loss).all()
    assert (indices[closed] < layer.num_fine_experts).all()
    assert (indices[~closed] >= layer.num_fine_experts).all()
    assert layer.last_admission_mask[:, 0].tolist() == [False, True, False]


@pytest.mark.parametrize("score", ["softmax", "sigmoid"])
def test_open_stage_a_gradients_only_reach_added_experts_and_router(score):
    old, actor = admission_pair("open", warmup=True)
    layer = actor.actor_module.layers[0]
    layer.routing_score_fn = score
    before = copy.deepcopy(actor.state_dict())
    actor.train()
    td = make_obs((3, 7))
    result = actor(
        td,
        actions=torch.randn(3, 7, 2),
        mode="sequence_logp",
        update_obs_norm=True,
    )
    (-result["actions_log_prob"].mean()).backward()
    for parameter in (
        layer.added_gate_up_proj,
        layer.added_down_proj,
        layer.added_router.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(p.grad is None for p in layer.added_admission.parameters())
    assert all(
        p.grad is None
        for name, p in actor.named_parameters()
        if name in dict(old.named_parameters())
    )
    # Backward alone must not mutate original buffers or any parameter.
    assert all(
        torch.equal(v, actor.state_dict()[k]) for k, v in before.items()
    )
    assert_old_state_equal(old, actor)


def test_stage_b_gate_has_supervised_gradient_without_old_expert_gradients():
    old, actor = admission_pair("learned")
    layer = actor.actor_module.layers[0]
    actor.actor_module.set_expansion(
        "cooperative", stage="B", admission_mode="learned"
    )
    with torch.no_grad():
        layer.added_admission.bias.fill_(10)
        layer.added_router.bias.fill_(1)
    x = torch.randn(3, 1, layer.d_model)
    reference = torch.randn_like(x)
    value = layer.compute_moe_ffn(x, reference)
    gate_loss = F.binary_cross_entropy_with_logits(
        layer.last_admission_logits,
        torch.tensor([[0.0], [1.0], [0.0]]),
    )
    (value.square().mean() + gate_loss).backward()
    assert layer.added_admission.weight.grad.abs().sum() > 0
    assert layer.added_router.weight.grad.abs().sum() > 0
    assert layer.added_gate_up_proj.grad is None
    assert layer.added_down_proj.grad is None
    assert all(
        p.grad is None
        for name, p in actor.named_parameters()
        if name in dict(old.named_parameters())
    )
    assert_old_state_equal(old, actor)


def test_native_bootstrap_builds_gate_but_v1_cannot_silently_become_v2():
    old, actor = admission_pair()
    layer = actor.actor_module.layers[0]
    assert int(layer.expansion_version) == 2
    assert torch.equal(
        layer.added_admission.weight,
        torch.zeros_like(layer.added_admission.weight),
    )
    assert layer.added_admission.bias.item() == -8
    assert_old_state_equal(old, actor)
    _, v1 = make_pair()
    with pytest.raises(RuntimeError, match="version|admission|Missing"):
        actor.load_state_dict(v1.state_dict(), strict=True)


def test_repeated_native_bootstrap_discards_all_previous_added_state():
    old, actor = admission_pair("learned")
    fresh = copy.deepcopy(actor.state_dict())
    layer = actor.actor_module.layers[0]
    with torch.no_grad():
        layer.added_admission.weight.fill_(7)
        layer.added_admission.bias.fill_(15)
        layer.added_admission.metadata_version.add_(9)
        layer.added_admission.config_digest.zero_()
        layer.added_router.weight.fill_(23)
        layer.added_router.bias.fill_(21)
        layer.added_gate_up_proj.fill_(17)
        layer.added_down_proj.fill_(19)
    actor.load_state_dict(old.state_dict(), strict=True)
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in fresh.items())
    reference = torch.full((3, layer.d_model), 10.0)
    assert not layer.added_admission.hard_mask(reference).any()
    assert_old_state_equal(old, actor)


@torch.no_grad()
def test_v2_roundtrip_requires_gate_metadata_and_preserves_output():
    _, actor = admission_pair("learned")
    actor.actor_module.set_expansion(
        "cooperative", stage="eval", admission_mode="learned"
    )
    restored = copy.deepcopy(actor)
    restored.load_state_dict(actor.state_dict(), strict=True)
    actor.reset_kv_cache(3, "cpu")
    restored.reset_kv_cache(3, "cpu")
    td = make_obs()
    expected = actor(td, mode="inference")
    actual = restored(td, mode="inference")
    assert torch.equal(actual["mu"], expected["mu"])
    broken = dict(actor.state_dict())
    del broken["actor_module.layers.0.added_admission.config_digest"]
    with pytest.raises(RuntimeError, match="admission"):
        restored.load_state_dict(broken, strict=True)


@torch.no_grad()
def test_admission_profile_change_invalidates_cache_and_route_revision():
    _, actor = admission_pair()
    policy = actor.actor_module
    policy.set_expansion("cooperative", stage="eval", admission_mode="closed")
    actor.reset_kv_cache(3, "cpu")
    actor(make_obs(), mode="inference")
    assert policy._kv_cache_len.gt(0).all()
    version = policy.route_revision
    policy.set_expansion("cooperative", stage="eval", admission_mode="open")
    assert policy._kv_cache_len.eq(0).all()
    assert policy.route_revision == version + 1
    actor(make_obs(), mode="inference")
    lengths = policy._kv_cache_len.clone()
    policy.set_expansion("cooperative", stage="eval", admission_mode="open")
    assert torch.equal(policy._kv_cache_len, lengths)
    assert policy.route_revision == version + 1


@torch.no_grad()
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("mode", ["closed", "open"])
def test_cached_admission_long_replay_matches_real_step_and_keeps_cache(
    bounded, mode
):
    torch.manual_seed(891)
    old, actor = admission_pair(mode, warmup=mode == "open")
    policy = actor.actor_module
    x = torch.randn(3, 15, policy.input_dim)
    mask, valid, starts = episode_mask(
        3, 15, policy.max_ctx_len if bounded else None
    )
    # Closed admission is compared to the actual unexpanded native policy;
    # open admission is compared to the same trainable policy's sampler.
    reference = old.actor_module if mode == "closed" else policy
    expected = sample_reference(reference, x, valid, starts, chunk=2)
    policy.reset_kv_cache(3, "cpu")
    policy.single_step_mu(torch.randn(3, policy.input_dim))
    saved = snapshot(policy)
    actual = policy.sequence_mu(x, attn_mask=mask)
    assert torch.equal(actual, expected)
    assert_snapshot(policy, saved)
    assert policy.layers[0].last_topk_indices.shape == (3, 15, 2)
    assert policy.layers[0].last_admission_mask.shape == (3, 15)
    assert torch.count_nonzero(actual[~valid]) == 0
    assert_old_state_equal(old, actor)


def test_cached_admission_losses_use_all_valid_steps_and_chunks(monkeypatch):
    torch.manual_seed(919)
    _, actor = admission_pair("open", warmup=True)
    policy = actor.actor_module
    layer = policy.layers[0]
    captured = []
    finalize = layer.finalize_cached_replay

    def capture(record):
        captured.append(record)
        return finalize(record)

    monkeypatch.setattr(layer, "finalize_cached_replay", capture)
    x = torch.randn(3, 11, policy.input_dim)
    mask, valid, _ = episode_mask(3, 11)
    policy.sequence_mu(x, attn_mask=mask)
    assert len(captured) == 1
    record = captured[0]
    assert torch.equal(record["valid_mask"], valid)

    def expected_balance(selected, logits):
        counts = torch.bincount(
            selected.reshape(-1), minlength=layer.total_experts
        )[layer.num_fine_experts :].float()
        frequency = counts / counts.sum().clamp_min(1)
        probability = logits.softmax(-1).mean(0)
        return layer.num_added_experts * (frequency * probability).sum()

    expected = expected_balance(
        record["indices"][valid], record["new_logits"][valid]
    )
    last_chunk = expected_balance(
        record["indices"][-1:, -1:].reshape(-1, 2),
        record["new_logits"][-1:, -1:].reshape(-1, 2),
    )
    torch.testing.assert_close(layer.last_moe_load_balance_loss, expected)
    assert not torch.isclose(expected, last_chunk, atol=1e-4, rtol=1e-4)
    # Padding must not enter either auxiliary loss or its gradients.
    assert torch.isfinite(layer.last_inactive_expert_margin_to_topk_loss)
    layer.last_moe_load_balance_loss.backward()
    assert layer.added_router.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("entry", ["step", "chunk", "sequence", "actor"])
def test_cached_replay_rejects_autocast_before_changing_live_cache(entry):
    _, actor = admission_pair("closed")
    policy = actor.actor_module
    policy.reset_kv_cache(3, "cpu")
    with torch.no_grad():
        policy.single_step_mu(torch.randn(3, policy.input_dim))
    saved = snapshot(policy)
    x = torch.randn(3, policy.input_dim)
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="autocast|AMP|[Ff][Pp]32|precision"),
    ):
        if entry == "step":
            policy.single_step_mu(x)
        elif entry == "chunk":
            policy.single_step_mu_chunked(x, 2)
        elif entry == "sequence":
            policy.sequence_mu(x[:, None, :])
        else:
            actor(make_obs(), mode="inference", forward_chunk_size=2)
    assert_snapshot(policy, saved)
