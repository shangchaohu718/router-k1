"""Actual DDP broadcasts preserve leases; observed buffer drift does not."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from test_incremental_task_isolation import obs, pair
from test_internal_moe_admission import admission_pair
from test_sliding_actor import make_obs
from torch.nn.parallel import DistributedDataParallel

from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.motion_tracking.task_contract import (
    critic_cycle_stamp,
    policy_stamp,
)


def models():
    torch.manual_seed(734)
    _, actor = admission_pair("open")
    actor.reset_kv_cache(3, "cpu")
    _, critic = pair()
    critic.freeze_cycle_statistics = True
    return actor, critic


def test_real_gloo_single_rank_broadcast_and_mutation(tmp_path):
    """One-rank DDP still performs its real coalesced buffer broadcast."""
    if dist.is_initialized():
        pytest.skip("This test requires ownership of its temporary group")
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(tmp_path / "rendezvous"),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=20),
    )
    try:
        actor, critic = models()
        actor_ddp = DistributedDataParallel(actor)
        critic_ddp = DistributedDataParallel(critic)
        assert actor_ddp.broadcast_buffers and critic_ddp.broadcast_buffers
        actor_stamp = policy_stamp(actor)
        critic_stamp = critic_cycle_stamp(critic)
        version_before = actor.obs_normalizer._mean._version
        gate_before = actor.actor_module.layers[
            0
        ].added_admission.config_digest._version
        algo = PPOTF.__new__(PPOTF)
        algo.actor, algo.critic = actor_ddp, critic_ddp
        algo.accelerator = SimpleNamespace(
            unwrap_model=lambda model: model.module
        )
        algo._incremental_rollout_stamp = actor_stamp
        algo._incremental_critic_stamp = critic_stamp
        # Only exercise lease validation here. Cached multi-rank training is
        # separately gated off; this test does not change that gate.
        algo._cached_incremental_indices = lambda *args: None
        for _ in range(3):
            with torch.no_grad():
                actor_ddp(make_obs(), mode="sampling", update_obs_norm=False)
                critic_ddp(obs(), update_obs_norm=False)
            algo._check_incremental_policy_lease()
        assert actor.obs_normalizer._mean._version > version_before
        assert (
            actor.actor_module.layers[0].added_admission.config_digest._version
            > gate_before
        )
        actor.obs_normalizer._mean.add_(0.1)
        with pytest.raises(ValueError, match="Policy/route changed"):
            algo._check_incremental_policy_lease()
        algo._incremental_rollout_stamp = policy_stamp(actor)
        gate = actor.actor_module.layers[0].added_admission
        for metadata in (gate.metadata_version, gate.config_digest):
            metadata.add_(1)
            with pytest.raises(ValueError, match="Policy/route changed"):
                algo._check_incremental_policy_lease()
            algo._incremental_rollout_stamp = policy_stamp(actor)
        critic.legacy.obs_normalizer._std.add_(0.1)
        with pytest.raises(ValueError, match="Critic state changed"):
            algo._check_incremental_policy_lease()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "name",
    [
        "obs_normalizer._mean",
        "actor_module.layers.0.expert_bias",
        "actor_module.layers.0.ema_routed_expert_usage",
        "actor_module.layers.0.added_admission.config_digest",
        "actor_module.layers.0.added_admission.metadata_version",
        "actor_module._router_feature_indices",
        "actor_module.cos_cached",
    ],
)
@pytest.mark.parametrize("replace_buffer", [False, True])
def test_every_protected_actor_state_rejects_drift_or_replacement(
    name, replace_buffer
):
    actor, _ = models()
    stamp = policy_stamp(actor)
    parent_name, attribute = name.rsplit(".", 1)
    parent = actor.get_submodule(parent_name)
    tensor = getattr(parent, attribute)
    if replace_buffer:
        setattr(parent, attribute, tensor.clone())
    else:
        tensor.add_(1)
    assert policy_stamp(actor) != stamp


def test_unchanged_versions_use_no_device_comparison_or_snapshot_copy(
    monkeypatch,
):
    actor, critic = models()
    actor_stamp, critic_stamp = policy_stamp(actor), critic_cycle_stamp(critic)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unchanged versions must use the lease fast path")

    monkeypatch.setattr(torch, "stack", unexpected)
    monkeypatch.setattr(torch.Tensor, "clone", unexpected)
    assert policy_stamp(actor) == actor_stamp
    assert critic_cycle_stamp(critic) == critic_stamp


def test_value_preserving_writes_reuse_snapshot_and_no_state_dict_entries():
    actor, critic = models()
    original_actor_keys = set(actor.state_dict())
    original_critic_keys = set(critic.state_dict())
    actor_stamp, critic_stamp = policy_stamp(actor), critic_cycle_stamp(critic)
    snapshots = {
        name: value.snapshot
        for name, value in actor._incremental_buffer_lease_cache.items()
    }
    for value in actor.buffers():
        value.copy_(value)
    for value in critic.buffers():
        value.copy_(value)
    assert policy_stamp(actor) == actor_stamp
    assert critic_cycle_stamp(critic) == critic_stamp
    assert all(
        actor._incremental_buffer_lease_cache[name].snapshot is snapshot
        for name, snapshot in snapshots.items()
    )
    assert set(actor.state_dict()) == original_actor_keys
    assert set(critic.state_dict()) == original_critic_keys
    assert not any("lease" in name for name, _ in actor.named_buffers())
    assert not any("lease" in name for name, _ in critic.named_buffers())


def test_observed_drift_cannot_recover_the_original_lease_token():
    actor, _ = models()
    buffer = actor.obs_normalizer._mean
    original = buffer.clone()
    stamp = policy_stamp(actor)
    buffer.add_(1)
    assert policy_stamp(actor) != stamp
    buffer.copy_(original)
    assert policy_stamp(actor) != stamp


def test_exact_restore_before_observation_has_value_freeze_semantics():
    actor, _ = models()
    buffer = actor.obs_normalizer._mean
    original = buffer.clone()
    stamp = policy_stamp(actor)
    buffer.add_(1)
    buffer.copy_(original)
    assert policy_stamp(actor) == stamp


def test_parameter_version_changes_remain_strict_even_with_same_value():
    actor, critic = models()
    actor_stamp, critic_stamp = policy_stamp(actor), critic_cycle_stamp(critic)
    with torch.no_grad():
        next(actor.parameters()).add_(0)
        next(critic.parameters()).add_(0)
    assert policy_stamp(actor) != actor_stamp
    assert critic_cycle_stamp(critic) != critic_stamp


def test_signed_zero_is_a_bitwise_change_and_mutable_added_counts_are_not():
    actor, _ = models()
    buffer = actor.actor_module.layers[0].expert_bias
    stamp = policy_stamp(actor)
    actor.actor_module.layers[0].added_assignment_counts.add_(7)
    assert policy_stamp(actor) == stamp
    buffer[0] = -0.0
    assert policy_stamp(actor) != stamp


def test_deepcopy_has_independent_lease_snapshots():
    actor, _ = models()
    stamp = policy_stamp(actor)
    copied = copy.deepcopy(actor)
    copied_stamp = policy_stamp(copied)
    copied.obs_normalizer._mean.add_(1)
    assert policy_stamp(copied) != copied_stamp
    assert policy_stamp(actor) == stamp
