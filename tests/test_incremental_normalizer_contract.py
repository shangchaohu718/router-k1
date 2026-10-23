"""Reject changed normalization state between sampling and PPO replay."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from test_incremental_g0_g2 import spec
from test_incremental_task_isolation import obs, pair
from test_internal_moe import make_pair
from test_sliding_actor import make_obs

from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.motion_tracking.incremental_task import TASK_KEY
from holomotion.src.motion_tracking.task_contract import (
    critic_cycle_stamp,
    policy_stamp,
    tag_rollout,
    validate_rollout,
)


@pytest.mark.parametrize("replace_buffer", [False, True])
def test_actor_lease_rejects_normalizer_mutation_or_replacement(
    replace_buffer,
):
    _, actor = make_pair()
    _, critic = pair()
    algo = PPOTF.__new__(PPOTF)
    algo.actor, algo.critic = actor, critic
    algo.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    algo._incremental_rollout_stamp = policy_stamp(actor)
    algo._incremental_critic_stamp = critic_cycle_stamp(critic)
    algo._check_incremental_policy_lease()
    if replace_buffer:
        actor.obs_normalizer._mean = actor.obs_normalizer._mean.clone()
    else:
        actor.obs_normalizer._mean.add_(0.1)
    with pytest.raises(ValueError, match="changed between rollout"):
        algo._check_incremental_policy_lease()


@pytest.mark.parametrize("branch", ["legacy", "sliding"])
@pytest.mark.parametrize("kind", ["buffer", "replacement", "parameter"])
def test_critic_lease_covers_all_branch_state(branch, kind):
    _, critic = pair()
    before = critic_cycle_stamp(critic)
    model = getattr(critic, branch)
    with torch.no_grad():
        if kind == "buffer":
            model.obs_normalizer._mean.add_(0.1)
        elif kind == "replacement":
            model.obs_normalizer._mean = model.obs_normalizer._mean.clone()
        else:
            next(model.parameters()).add_(0.1)
    assert critic_cycle_stamp(critic) != before


@torch.no_grad()
def test_frozen_statistics_forwards_leave_both_leases_unchanged():
    _, actor = make_pair()
    actor.reset_kv_cache(3, "cpu")
    actor_before = policy_stamp(actor)
    for _ in range(3):
        actor(make_obs(), mode="sampling", update_obs_norm=False)
        assert policy_stamp(actor) == actor_before

    _, critic = pair()
    critic_before = critic_cycle_stamp(critic)
    for _ in range(3):
        critic(obs(), update_obs_norm=False)
        assert critic_cycle_stamp(critic) == critic_before
    critic.freeze_cycle_statistics = True
    critic(obs(), update_obs_norm=True)
    assert critic_cycle_stamp(critic) == critic_before
    critic.finish_statistics_cycle(obs())
    assert critic_cycle_stamp(critic) != critic_before


@pytest.mark.parametrize("change", ["value", "dtype", "shape", "missing"])
def test_rollout_normalizer_version_checks_complete_stored_field(change):
    td = TensorDict(
        {TASK_KEY: torch.ones(2, 1, dtype=torch.int64)}, batch_size=[2]
    )
    contract = spec()
    tag_rollout(td, contract, normalizer_version=17)
    validate_rollout(td, contract, normalizer_version=17)
    key = "incremental_normalizer_version"
    if change == "value":
        td[key][0] += 1
    elif change == "dtype":
        td.set(key, td[key].float())
    elif change == "shape":
        td.set(key, td[key].expand(2, 2))
    else:
        td.del_(key)
    with pytest.raises(ValueError, match="normalizer_version"):
        validate_rollout(td, contract, normalizer_version=17)
