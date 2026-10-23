"""The incremental replay uses the actual cached sampling computations."""

from types import MethodType

import pytest
import torch
from test_internal_moe import make_pair

from holomotion.src.modules.cached_replay import (
    _CACHE_STATE,
    cached_sequence_replay,
)


def episode_mask(batch, length, context=None):
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[-1, :2] = False
    starts = torch.zeros(batch, length, dtype=torch.long)
    starts[-1, 2:] = 2
    if length > 4:
        starts[1, 4:] = 4
    if length > 7:
        starts[0, 7:] = 7
    time = torch.arange(length)
    mask = (
        (time[:, None] >= time[None, :])[None]
        & valid[:, :, None]
        & valid[:, None, :]
        & (time[None, None, :] >= starts[:, :, None])
    )
    if context is not None:
        mask &= (time[:, None] - time[None, :] < context)[None]
    return mask, valid, starts


def sample_reference(policy, x, valid, starts, chunk):
    policy.reset_kv_cache(x.shape[0], x.device)
    output = []
    for time in range(x.shape[1]):
        clear = ~valid[:, time] | starts[:, time].eq(time)
        if clear.any():
            policy.clear_env_cache(clear.nonzero().flatten())
        obs = x[:, time].clone()
        obs[~valid[:, time]] = 0
        value = (
            policy.single_step_mu_chunked(obs, chunk)
            if chunk
            else policy.single_step_mu(obs)
        )
        value = value.masked_fill(~valid[:, time, None], 0)
        output.append(value)
    return torch.stack(output, dim=1)


def snapshot(policy):
    return {
        name: (getattr(policy, name), getattr(policy, name).clone())
        for name in _CACHE_STATE
        if getattr(policy, name, None) is not None
    }


def assert_snapshot(policy, saved):
    for name, (reference, value) in saved.items():
        assert getattr(policy, name) is reference, name
        assert torch.equal(getattr(policy, name), value), name


@pytest.mark.parametrize("length", [3, 13])
@pytest.mark.parametrize("chunk", [0, 2])
@pytest.mark.parametrize("bounded", [False, True])
@torch.no_grad()
def test_native_sampling_replay_and_live_cache(length, chunk, bounded):
    torch.manual_seed(182)
    old, _ = make_pair()
    policy = old.actor_module
    x = torch.randn(3, length, 8)
    mask, valid, starts = episode_mask(
        3, length, policy.max_ctx_len if bounded else None
    )
    expected = sample_reference(policy, x, valid, starts, chunk)
    policy.set_collect_routing_stats(True)
    policy.single_step_mu(torch.randn(3, 8))
    saved = snapshot(policy)
    result = cached_sequence_replay(policy, x, mask, chunk)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert_snapshot(policy, saved)
    assert policy.layers[0].collect_routing_stats
    assert torch.count_nonzero(result[~valid]) == 0


@torch.no_grad()
def test_no_mask_starts_at_zero_and_restores_absent_cache():
    old, _ = make_pair()
    policy = old.actor_module
    x = torch.randn(3, 12, 8)
    expected = sample_reference(
        policy,
        x,
        torch.ones(3, 12, dtype=torch.bool),
        torch.zeros(3, 12, dtype=torch.long),
        0,
    )
    for name in _CACHE_STATE:
        setattr(policy, name, None)
    result = cached_sequence_replay(policy, x)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert all(getattr(policy, name) is None for name in _CACHE_STATE)


@pytest.mark.parametrize(
    "corruption", ["future", "sparse", "reset", "padding", "dtype"]
)
@torch.no_grad()
def test_invalid_mask_rejected_without_mutating_cache(corruption):
    old, _ = make_pair()
    policy = old.actor_module
    policy.reset_kv_cache(3, "cpu")
    policy.single_step_mu(torch.randn(3, 8))
    saved = snapshot(policy)
    mask, _, _ = episode_mask(3, 9)
    if corruption == "future":
        mask[0, 1, 2] = True
    elif corruption == "sparse":
        mask[0, 3, 1] = False
    elif corruption == "reset":
        mask[1, 4, 1] = True
    elif corruption == "padding":
        mask[2, 0, 1] = True
    else:
        mask = mask.float()
    with pytest.raises(ValueError, match="mask|links"):
        cached_sequence_replay(policy, torch.randn(3, 9, 8), mask)
    assert_snapshot(policy, saved)


@torch.no_grad()
def test_exception_restores_cache_and_collection_flags(monkeypatch):
    old, _ = make_pair()
    policy = old.actor_module
    policy.reset_kv_cache(3, "cpu")
    policy.set_collect_routing_stats(True)
    policy.single_step_mu(torch.randn(3, 8))
    saved = snapshot(policy)
    hooks = dict(policy.norm_f._forward_pre_hooks)
    original = policy.single_step_mu
    calls = 0

    def fail_second(x):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected replay failure")
        return original(x)

    monkeypatch.setattr(policy, "single_step_mu", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        cached_sequence_replay(
            policy, torch.randn(3, 4, 8), return_pre_moe_hidden=True
        )
    assert_snapshot(policy, saved)
    assert policy.layers[0].collect_routing_stats
    assert dict(policy.norm_f._forward_pre_hooks) == hooks


@torch.no_grad()
def test_chunk_records_reassemble_all_steps_and_restore_record_sink(
    monkeypatch,
):
    old, _ = make_pair()
    policy = old.actor_module
    layer = policy.layers[0]
    previous = []
    layer._replay_records = previous
    original = layer.compute_moe_ffn
    reference = []

    def record(self, x, router_x=None, **kwargs):
        result = original(x, router_x, **kwargs)
        entry = {"hidden": x, "router": router_x}
        reference.append(entry)
        if self._replay_records is not None:
            self._replay_records.append(entry)
        return result

    merged_records = []
    monkeypatch.setattr(layer, "compute_moe_ffn", MethodType(record, layer))
    monkeypatch.setattr(
        layer, "finalize_cached_replay", merged_records.append, raising=False
    )
    mask, valid, _ = episode_mask(3, 6)
    cached_sequence_replay(policy, torch.randn(3, 6, 8), mask, chunk_size=2)
    assert layer._replay_records is previous
    assert previous == []
    assert len(merged_records) == 1
    merged = merged_records[0]
    assert torch.equal(merged["valid_mask"], valid)
    for name in ("hidden", "router"):
        expected = torch.cat(
            [
                torch.cat([entry[name] for entry in reference[t : t + 2]])
                for t in range(0, len(reference), 2)
            ],
            dim=1,
        )
        torch.testing.assert_close(merged[name], expected, rtol=0, atol=0)


@torch.no_grad()
def test_finalize_failure_restores_temporary_state(monkeypatch):
    old, _ = make_pair()
    policy = old.actor_module
    layer = policy.layers[0]
    policy.reset_kv_cache(3, "cpu")
    policy.set_collect_routing_stats(True)
    policy.single_step_mu(torch.randn(3, 8))
    saved = snapshot(policy)
    previous = []
    layer._replay_records = previous
    original = layer.compute_moe_ffn

    def record(self, x, router_x=None, **kwargs):
        self._replay_records.append({"hidden": x})
        return original(x, router_x, **kwargs)

    def fail(_):
        raise RuntimeError("injected finalizer failure")

    monkeypatch.setattr(layer, "compute_moe_ffn", MethodType(record, layer))
    monkeypatch.setattr(layer, "finalize_cached_replay", fail, raising=False)
    with pytest.raises(RuntimeError, match="finalizer"):
        cached_sequence_replay(policy, torch.randn(3, 6, 8), chunk_size=2)
    assert_snapshot(policy, saved)
    assert layer._replay_records is previous
    assert previous == []
    assert layer.collect_routing_stats


def test_real_new_expert_and_router_gradients_with_frozen_old_front():
    torch.manual_seed(782)
    old, actor = make_pair(warmup=True)
    actor.train()
    policy = actor.actor_module
    layer = policy.layers[0]
    old_names = dict(old.named_parameters())
    old_state = {key: value.clone() for key, value in old.state_dict().items()}
    mask, valid, _ = episode_mask(3, 11)
    x = torch.randn(3, 11, 8)
    with torch.no_grad():
        expected = cached_sequence_replay(policy, x, mask, chunk_size=2)
    # No optimizer or environment step: this is a differentiability contract.
    result = cached_sequence_replay(policy, x, mask, chunk_size=2)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result[valid].square().mean().backward()
    assert layer.added_gate_up_proj.grad.abs().sum() > 0
    assert layer.added_down_proj.grad.abs().sum() > 0
    assert layer.added_router.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, parameter in actor.named_parameters()
        if name in old_names
    )
    for key, value in old_state.items():
        assert torch.equal(actor.state_dict()[key], value), key


@torch.no_grad()
def test_replay_does_not_reuse_old_routes_after_parameter_change():
    _, actor = make_pair()
    policy = actor.actor_module
    layer = policy.layers[0]
    x = torch.randn(3, 7, 8)
    with torch.no_grad():
        layer.added_router.weight.zero_()
        layer.added_router.bias.fill_(-1e4)
    before = cached_sequence_replay(policy, x)
    with torch.no_grad():
        layer.added_router.bias.fill_(1e4)
        layer.added_down_proj.add_(0.2)
    after = cached_sequence_replay(policy, x)
    assert not torch.allclose(before, after, atol=1e-3, rtol=0)
    assert (layer.last_topk_indices >= layer.num_fine_experts).all()


@torch.no_grad()
def test_auxiliary_hidden_keeps_native_post_layer_convention():
    torch.manual_seed(901)
    old, _ = make_pair()
    policy = old.actor_module
    x = torch.randn(3, 3, 8)
    expected_mu, expected_hidden = policy.sequence_mu(
        x, return_pre_moe_hidden=True
    )
    hooks = dict(policy.norm_f._forward_pre_hooks)
    mu, hidden = cached_sequence_replay(
        policy, x, chunk_size=2, return_pre_moe_hidden=True
    )
    # Sequence versus single-token attention retains the existing FP32
    # tolerance; the capture point matches the actual native return value.
    torch.testing.assert_close(mu, expected_mu, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(hidden, expected_hidden, atol=3e-6, rtol=2e-5)
    assert dict(policy.norm_f._forward_pre_hooks) == hooks


def test_auxiliary_hidden_padding_and_new_expert_gradients():
    _, actor = make_pair(warmup=True)
    policy = actor.actor_module
    mask, valid, _ = episode_mask(3, 11)
    _, hidden = cached_sequence_replay(
        policy,
        torch.randn(3, 11, 8),
        mask,
        chunk_size=2,
        return_pre_moe_hidden=True,
    )
    assert hidden.shape == (3, 11, policy.d_model)
    assert torch.count_nonzero(hidden[~valid]) == 0
    hidden[valid].square().mean().backward()
    layer = policy.layers[0]
    assert layer.added_down_proj.grad.abs().sum() > 0
    assert layer.added_router.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None for parameter in policy.obs_embed.parameters()
    )
    assert all(parameter.grad is None for parameter in layer.attn.parameters())
