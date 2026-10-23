"""Fixed-task Stage A contracts. No simulator imports and no mixed-task PPO."""

from __future__ import annotations
import copy
import hashlib
import inspect
import json
from pathlib import Path

import torch

TASK_KEY = "incremental_task_id"
TASK_IDS = {"legacy": 0, "sliding": 1}
SLIDING_TERMS = (
    "moonwalk_sole_pose_tracking",
    "moonwalk_sole_velocity_tracking",
    "moonwalk_heel_contact_tracking",
)


def implementation_contract():
    root = Path(__file__).resolve().parents[3]
    files = (
        "holomotion/src/motion_tracking/incremental_task.py",
        "holomotion/src/motion_tracking/task_contract.py",
        "holomotion/src/motion_tracking/asset_contract.py",
        "holomotion/src/utils/g1_variants.py",
        "holomotion/src/utils/config.py",
        "holomotion/src/motion_tracking/incremental_evaluation.py",
        "holomotion/src/evaluation/eval_motion_tracking_single.py",
        "holomotion/src/motion_tracking/task_gae.py",
        "holomotion/src/motion_tracking/moonwalk_rewards.py",
        "holomotion/src/env/isaaclab_components/isaaclab_rewards.py",
        "holomotion/src/env/isaaclab_components/isaaclab_observation.py",
        "holomotion/src/motion_tracking/actor_observation.py",
        "holomotion/src/motion_tracking/reference_observation.py",
        "holomotion/src/env/motion_tracking.py",
        "holomotion/src/algo/ppo.py",
        "holomotion/src/algo/ppo_tf.py",
        "holomotion/src/algo/algo_base.py",
        "holomotion/src/algo/algo_utils.py",
        "holomotion/src/modules/internal_moe.py",
        "holomotion/src/modules/expert_admission.py",
        "holomotion/src/modules/cached_replay.py",
        "holomotion/src/modules/incremental_critic.py",
        "holomotion/src/modules/agent_modules.py",
        "holomotion/src/modules/network_modules.py",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in files
    }


def config_hash(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def select_reward_profile(config):
    """Copy a full recipe; the native manager applies weight and dt once."""
    config = copy.deepcopy(config)
    spec = config.get("_config", {}).pop("incremental_task", None)
    if spec is None:
        return config
    task = spec["task"]
    if task not in TASK_IDS:
        raise ValueError(f"Unknown incremental task: {task}")
    if any(name not in config for name in SLIDING_TERMS):
        raise ValueError("Incomplete sliding reward recipe")
    legacy = {k: v for k, v in config.items() if k not in SLIDING_TERMS}
    if config_hash(legacy) != spec["legacy_reward_hash"]:
        raise ValueError("Legacy reward contract mismatch")
    return legacy if task == "legacy" else config


def validate_task_tensor(td, task):
    ids = td.get(TASK_KEY, None)
    if ids is None or ids.shape != (*td.batch_size, 1):
        raise ValueError("Missing or malformed stored incremental task ID")
    if ids.dtype != torch.int64 or not torch.all(ids == TASK_IDS[task]):
        raise ValueError("Mixed or mismatched incremental task IDs")


class FixedTaskReward:
    """Fail closed when a fixed-task run loads a different motion population.

    Inspect source identities, never foot velocity or router predictions.
    The entire cache must belong to the run's profile (conservative by design).
    """

    def __init__(self, func, spec, reward_name=None, weight=1.0):
        self.func = func
        self.spec = copy.deepcopy(spec)
        self.reward_name, self.weight = reward_name, float(weight)
        self.__name__ = f"FixedTaskReward[{func.__name__}]"
        # IsaacLab validates callable signatures against top-level params and
        # resolves SceneEntityCfg there. Preserve the original signature.
        self.__signature__ = inspect.signature(func)

    def __call__(self, env, *args, **kwargs):
        command = env.command_manager.get_term("ref_motion")
        cache = getattr(command, "_motion_cache", None)
        batch = getattr(cache, "current_batch", None)
        keys = getattr(batch, "raw_motion_keys", None)
        if keys is None or len(keys) == 0:
            raise ValueError(
                "Incremental rewards require raw motion identities"
            )
        registry = frozenset(self.spec["sliding_motion_keys"])
        if not registry:
            raise ValueError("Empty sliding motion registry")
        expected = self.spec["task"] == "sliding"
        if any((str(key) in registry) != expected for key in keys):
            raise ValueError(
                "Motion population does not match reward/critic task"
            )
        value = self.func(env, *args, **kwargs)
        if self.reward_name is not None:
            components = getattr(env, "_incremental_reward_components", None)
            if components is None:
                components = env._incremental_reward_components = {}
            raw = value.detach().clone()
            components[self.reward_name] = torch.stack(
                (
                    raw,
                    raw * self.weight,
                    raw * self.weight * float(env.step_dt),
                ),
                -1,
            )
            manager = getattr(env, "action_manager", None)
            if manager is not None:
                # Record before automatic reset zeroes action buffers.
                env._incremental_executed_action = (
                    manager.action.detach().clone()
                )
                env._incremental_joint_target = torch.cat(
                    [
                        manager.get_term(name)
                        .processed_actions.detach()
                        .clone()
                        for name in manager.active_terms
                    ],
                    dim=-1,
                )
        return value
