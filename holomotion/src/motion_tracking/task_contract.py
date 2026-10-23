"""Immutable task contracts and episode/version checks for incremental PPO."""

from __future__ import annotations
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from holomotion.src.motion_tracking.asset_contract import (
    asset_content_manifest,
)
from holomotion.src.motion_tracking.incremental_task import (
    TASK_IDS,
    config_hash,
    implementation_contract,
    select_reward_profile,
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    reward_id: str
    reward_config_hash: str
    reward_code_hash: str
    critic_id: str
    actor_schema_hash: str
    critic_schema_hash: str
    actor_model_hash: str
    critic_model_hash: str
    command_schema_hash: str
    reference_hash: str
    termination_hash: str
    action_contract_hash: str
    environment_contract_hash: str
    observation_generation_hash: str
    evaluation_contract_hash: str
    gamma: float
    gae_lambda: float
    timeout_semantics: str
    execution_profile: str = "training"

    def __post_init__(self):
        if self.task_id not in TASK_IDS:
            raise ValueError("Unknown task contract")
        if self.execution_profile not in {
            "training",
            "legacy_control",
            "automatic",
        }:
            raise ValueError("Unknown execution profile")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid task discount")
        for name, value in asdict(self).items():
            if name.endswith("_hash") and (
                len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"Malformed task hash: {name}")

    @property
    def digest(self):
        return config_hash(asdict(self))


def reference_manifest(command):
    """Content-address the bounded validation population before any run."""
    root = Path(__file__).resolve().parents[3]
    motion = command["ref_motion"]["params"]["motion_lib_cfg"]
    paths = set()
    for name in ("train_hdf5_roots", "val_hdf5_roots"):
        for entry in motion.get(name, []):
            directory = Path(
                entry["root"] if isinstance(entry, dict) else entry
            )
            directory = (
                directory if directory.is_absolute() else root / directory
            )
            if not directory.is_dir():
                raise ValueError(
                    f"Missing registered reference root: {directory}"
                )
            manifest_path = directory / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("version") != 2 or not manifest.get("clips"):
                    raise ValueError("Invalid registered reference manifest")
                paths.add(manifest_path.resolve())
                # Subset manifests can point to external shards and retain
                # many unused entries. Pin every shard reachable by a clip.
                for index in {c["shard"] for c in manifest["clips"].values()}:
                    shard = Path(manifest["hdf5_shards"][index]["file"])
                    shard = shard if shard.is_absolute() else directory / shard
                    if not shard.is_file():
                        raise ValueError(
                            f"Missing registered reference shard: {shard}"
                        )
                    paths.add(shard.resolve())
            else:
                paths.update(p.resolve() for p in directory.rglob("*.h5"))
    if (
        not paths
        or len(paths) > 128
        or sum(p.stat().st_size for p in paths) > 256 * 1024**2
    ):
        raise ValueError(
            "Reference population missing or exceeds "
            "G0 bounded-manifest budget"
        )
    return [
        {
            "path": str(p.resolve()),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(paths)
    ]


def evaluation_registry_key(task, profile):
    if profile == "training" or (
        task == "legacy" and profile == "legacy_control"
    ):
        return task
    return f"{task}:{profile}"


def evaluation_actor_config(actor, profile):
    """Produce an explicit route profile without changing native parameters."""
    if profile not in {"legacy_control", "automatic"}:
        raise ValueError("Unknown evaluation profile")
    actor = copy.deepcopy(actor)
    route = actor["internal_expansion"]
    route.update(
        mode="legacy" if profile == "legacy_control" else "cooperative",
        stage="eval",
        warmup=False,
        bootstrap_from_legacy=False,
    )
    if "admission" in route:
        route["admission"]["mode"] = (
            "closed" if profile == "legacy_control" else "learned"
        )
    return actor


EVALUATION_SEED = 42


def _simulation_contract(simulation):
    return {
        k: v
        for k, v in simulation.items()
        if k not in {"device", "compute_device", "rl_device"}
    }


def build_task_spec(
    actor, critic, env, algo, *, evaluation_profile=None, offline_eval=False
):
    """Bind the resolved run, reference contents and execution interface."""
    rewards = env["rewards"]
    selected = select_reward_profile(rewards)
    spec = critic["task_isolation"]
    if rewards["_config"]["incremental_task"] != spec:
        raise ValueError("Complete Reward/Critic task specification mismatch")
    task = spec["task"]
    # A state_dict hash cannot detect shape-compatible changes such as the
    # attention head count or normalization epsilon. Pin the native config.
    native_actor = copy.deepcopy(actor)
    native_actor.pop("internal_expansion")
    native_actor["type"] = "ReferenceRoutedGroupedMoETransformerPolicy"
    native_actor["obs_norm"]["update_at_train"] = True
    if config_hash(native_actor) != spec["legacy_actor_config_hash"]:
        raise ValueError("Legacy Actor configuration mismatch")
    native_critic = copy.deepcopy(critic)
    native_critic.pop("task_isolation")
    native_critic["type"] = "MLP"
    if config_hash(native_critic) != spec["legacy_critic_config_hash"]:
        raise ValueError("Legacy Critic configuration mismatch")
    route = actor["internal_expansion"]
    profile = evaluation_profile or (
        "legacy_control" if task == "legacy" else "training"
    )
    if profile == "legacy_control":
        if route["mode"] != "legacy" or route.get("warmup", False):
            raise ValueError(
                "Legacy task requires legacy routing with no warmup "
                "for legacy_control"
            )
    elif profile == "automatic":
        if not offline_eval:
            raise ValueError("Automatic task profile is evaluation only")
        if route["mode"] != "cooperative" or route.get("warmup", False):
            raise ValueError(
                "Automatic evaluation requires cooperative routing "
                "without warmup"
            )
        if (
            "admission" in route
            and route["admission"].get("mode") != "learned"
        ):
            raise ValueError("Automatic evaluation requires learned admission")
    elif profile != "training" or task != "sliding":
        raise ValueError("Only sliding has a training execution profile")
    code = implementation_contract()
    reward_code = {k: v for k, v in code.items() if "reward" in k}
    command = env["commands"]
    action = {
        "actions": env["actions"],
        "simulation": _simulation_contract(env["simulation"]),
        "dof_names": env["robot"]["dof_names"],
        "actuators": env["robot"]["actuators"],
        "init_state": env["robot"]["init_state"],
        "sampled_action": "Gaussian raw action; log_prob before affine/clip",
    }
    motion = command["ref_motion"]["params"]["motion_lib_cfg"]
    action["asset_content"] = asset_content_manifest(
        env["robot"]["asset"], motion
    )
    # Only placement, output and execution-size metadata are excluded. Bind
    # scene sensors, spacing, robot assets and all other environment inputs.
    runtime_fields = {
        "experiment_name",
        "num_envs",
        "headless",
        "num_processes",
        "main_process",
        "process_id",
        "ckpt_dir",
        "disable_ref_viz",
        "eval_log_dir",
        "save_rendering_dir",
        "simulation_device",
        "sim_device",
        "rl_device",
        "compute_device",
        "physx_device",
        "dump_npzs",
        "calc_per_clip_metrics",
        "generate_report",
        "export_policy",
        "export_only",
        "project_name",
    }
    environment = {
        k: copy.deepcopy(v) for k, v in env.items() if k not in runtime_fields
    }
    environment["simulation"] = _simulation_contract(env["simulation"])
    environment["rewards"] = selected
    environment["algo_seed"] = algo.get("seed")
    return TaskSpec(
        task_id=task,
        reward_id=f"{task}_reward_v1",
        critic_id=f"{task}_critic_v1",
        reward_config_hash=config_hash(selected),
        reward_code_hash=config_hash(reward_code),
        actor_schema_hash=config_hash(actor["obs_schema"]),
        critic_schema_hash=config_hash(critic["obs_schema"]),
        actor_model_hash=config_hash(native_actor),
        critic_model_hash=config_hash(native_critic),
        command_schema_hash=config_hash(command),
        reference_hash=config_hash(
            {
                "command": command["ref_motion"],
                "files": reference_manifest(command),
            }
        ),
        termination_hash=config_hash(env["terminations"]),
        action_contract_hash=config_hash(action),
        environment_contract_hash=config_hash(environment),
        observation_generation_hash=config_hash(
            {
                "obs": env["obs"],
                "normalization": env["normalization"],
            }
        ),
        evaluation_contract_hash=config_hash(
            {
                "deterministic_actions": True,
                "environment_seed": env.get("seed"),
                "algorithm_seed": algo.get("seed"),
                "reference": command["ref_motion"],
                "execution_profile": profile,
                "route": {
                    k: v
                    for k, v in route.items()
                    if k not in {"stage", "bootstrap_from_legacy"}
                },
                "closed_loop_required": True,
            }
        ),
        gamma=float(algo["gamma"]),
        gae_lambda=float(algo["lam"]),
        timeout_semantics=spec["timeout_semantics"]
        if task == "sliding"
        else "original_legacy",
        execution_profile=profile,
    )


def registered_evaluation_specs(
    actor, critic, env, algo, legacy_registration=None
):
    """Register control and automatic profiles over the same frozen inputs."""
    populations = {
        critic["task_isolation"]["task"]: (
            copy.deepcopy(critic),
            copy.deepcopy(env),
        )
    }
    if legacy_registration is not None:
        old_critic, old_env = copy.deepcopy(critic), copy.deepcopy(env)
        old_critic["task_isolation"]["task"] = "legacy"
        old_env["rewards"]["_config"]["incremental_task"] = copy.deepcopy(
            old_critic["task_isolation"]
        )
        motion = old_env["commands"]["ref_motion"]["params"]["motion_lib_cfg"]
        motion.update(
            train_hdf5_roots=list(legacy_registration["reference_roots"]),
            val_hdf5_roots=list(legacy_registration["reference_roots"]),
            handpicked_motion_names=None,
            max_frame_length=int(legacy_registration["max_frame_length"]),
        )
        motion["cache"]["swap_interval_steps"] = int(
            legacy_registration["max_frame_length"]
        )
        populations["legacy"] = (old_critic, old_env)
    result = {}
    algo = {**algo, "seed": EVALUATION_SEED}
    for task, (task_critic, task_env) in populations.items():
        task_env["seed"] = EVALUATION_SEED
        task_env["robot"]["motion"] = copy.deepcopy(
            task_env["commands"]["ref_motion"]["params"]["motion_lib_cfg"]
        )
        for profile in ("legacy_control", "automatic"):
            result[evaluation_registry_key(task, profile)] = asdict(
                build_task_spec(
                    evaluation_actor_config(actor, profile),
                    task_critic,
                    task_env,
                    algo,
                    evaluation_profile=profile,
                    offline_eval=True,
                )
            )
    return result


def check_episode_tasks(task_ids, dones):
    """Validate [T,E,1] identities before reshuffling sequences or GAE."""
    if task_ids.dtype != torch.int64 or task_ids.shape != dones.shape:
        raise ValueError("Task and episode boundary shape/dtype mismatch")
    if not torch.all((task_ids == 0) | (task_ids == 1)):
        raise ValueError("Unknown rollout task")
    if torch.any((task_ids[1:] != task_ids[:-1]) & ~dones[:-1].bool()):
        raise ValueError("Task changed without an episode reset")


class _BufferLeaseValue:
    """A stable identity token for one observed, bitwise buffer value."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.version = tensor._version
        self.layout = self.tensor_layout(tensor)
        self.snapshot = tensor.detach().clone()
        self.token = object()

    @staticmethod
    def tensor_layout(tensor):
        return tensor.dtype, tensor.device, tensor.shape, tensor.stride()


def _buffer_value_stamps(owner, named_buffers):
    """Accept DDP's value-preserving writes without hiding observed drift.

    Snapshots remain on the buffer device and outside the model state_dict.
    Unchanged versions need neither tensor operations nor synchronization.
    Changed versions are compared bitwise in one aggregate per device; a
    value-preserving broadcast reuses its old token without copying again.

    This checks observed values, not every intermediate write: changing and
    exactly restoring contents before the next check is indistinguishable
    from a DDP broadcast. Writes bypassing PyTorch version counters are not
    covered. Replacements and any previously observed drift invalidate an
    old token, even when the original value is subsequently restored.
    """
    current = list(named_buffers)
    cache = getattr(owner, "_incremental_buffer_lease_cache", None)
    if cache is None:
        cache = {}
        owner._incremental_buffer_lease_cache = cache
    changed = {}
    for name, tensor in current:
        previous = cache.get(name)
        if (
            previous is None
            or previous.tensor is not tensor
            or previous.layout != _BufferLeaseValue.tensor_layout(tensor)
        ):
            cache[name] = _BufferLeaseValue(tensor)
        elif previous.version != tensor._version:
            changed.setdefault(tensor.device, []).append((name, tensor))
    for entries in changed.values():
        same = torch.stack(
            [
                (
                    tensor.detach().reshape(-1).contiguous().view(torch.uint8)
                    == cache[name]
                    .snapshot.reshape(-1)
                    .contiguous()
                    .view(torch.uint8)
                ).all()
                for name, tensor in entries
            ]
        ).all()
        if bool(same):
            for name, tensor in entries:
                cache[name].version = tensor._version
        else:
            # One observed change invalidates the lease. Refreshing the
            # changed-version group avoids a scalar GPU sync for each buffer.
            for name, tensor in entries:
                cache[name] = _BufferLeaseValue(tensor)
    names = {name for name, _ in current}
    for name in cache.keys() - names:
        del cache[name]
    return tuple((name, cache[name].token) for name, _ in current)


def policy_stamp(actor):
    """Lease parameters strictly and buffer values across normal DDP sync."""
    layer = actor.actor_module.layers[0]
    gate = getattr(layer, "added_admission", None)
    buffers = [
        (name, buffer)
        for name, buffer in actor.named_buffers()
        # This is deliberately mutable new-expert diagnostic state. The live
        # KV/history tensors are not registered buffers. Every old registered
        # buffer, including router bias/EMA and RoPE lookup tables, is frozen.
        if not name.endswith(".added_assignment_counts")
    ]
    return (
        layer.route_mode,
        layer.stage,
        layer.warmup,
        gate.mode if gate is not None else None,
        getattr(actor.actor_module, "sequence_replay", "vectorized"),
        getattr(actor.actor_module, "replay_chunk_size", 0),
        tuple((id(p), p._version) for p in actor.parameters()),
        _buffer_value_stamps(actor, buffers),
    )


def critic_cycle_stamp(critic):
    """Lease every Critic parameter and buffer during collection/replay.

    The isolated MLP Critic has no KV cache or other mutable forward buffers.
    Include both task branches and their normalizers, without exclusions.
    """
    return (
        tuple(
            (name, id(parameter), parameter._version)
            for name, parameter in critic.named_parameters()
        ),
        _buffer_value_stamps(critic, critic.named_buffers()),
    )


def tag_rollout(
    td,
    contract,
    *,
    policy_version=0,
    route_version=0,
    normalizer_version=0,
    route_profile=1,
    episode_id=None,
    timestep=None,
):
    task = TASK_IDS[contract.task_id]
    entries = {
        "reward_id": task,
        "critic_id": task,
        "policy_version": policy_version,
        "route_version": route_version,
        "normalizer_version": normalizer_version,
        "route_profile": route_profile,
    }
    for key, value in entries.items():
        td.set(
            "incremental_" + key,
            torch.full(
                (*td.batch_size, 1),
                int(value),
                dtype=torch.int64,
                device=td.device,
            ),
        )
    td.set(
        "incremental_valid_mask",
        torch.ones((*td.batch_size, 1), dtype=torch.bool, device=td.device),
    )
    digest = torch.tensor(
        list(bytes.fromhex(contract.digest)),
        dtype=torch.uint8,
        device=td.device,
    )
    td.set(
        "incremental_contract_digest",
        digest.expand(*td.batch_size, 32).clone(),
    )
    for name, tensor in (("episode_id", episode_id), ("timestep", timestep)):
        value = torch.zeros(
            (*td.batch_size, 1), device=td.device, dtype=torch.int64
        )
        if tensor is not None:
            value.copy_(tensor.reshape_as(value))
        td.set("incremental_" + name, value)
    return td


def validate_rollout(
    td,
    contract,
    *,
    policy_version=None,
    route_version=None,
    normalizer_version=None,
    route_profile=None,
):
    valid = td.get("incremental_valid_mask", None)
    if (
        valid is None
        or valid.dtype != torch.bool
        or valid.shape != (*td.batch_size, 1)
        or not torch.all(valid)
    ):
        raise ValueError(
            "Fixed-task rollout requires real, unpadded transitions"
        )
    expected = {
        "task_id": TASK_IDS[contract.task_id],
        "reward_id": TASK_IDS[contract.task_id],
        "critic_id": TASK_IDS[contract.task_id],
    }
    if policy_version is not None:
        expected["policy_version"] = policy_version
    if route_version is not None:
        expected["route_version"] = route_version
    if normalizer_version is not None:
        expected["normalizer_version"] = normalizer_version
    if route_profile is not None:
        expected["route_profile"] = route_profile
    for key, value in expected.items():
        data = td.get("incremental_" + key, None)
        if (
            data is None
            or data.shape != (*td.batch_size, 1)
            or data.dtype != torch.int64
            or not torch.all(data == value)
        ):
            raise ValueError(f"Rollout {key} contract mismatch")
    digest = td.get("incremental_contract_digest", None)
    expected_digest = torch.tensor(
        list(bytes.fromhex(contract.digest)),
        dtype=torch.uint8,
        device=td.device,
    )
    if (
        digest is None
        or digest.shape != (*td.batch_size, 32)
        or not torch.all(digest == expected_digest)
    ):
        raise ValueError("Stored rollout TaskSpec mismatch")


def validate_episode_ledger(td, dones):
    for key in ("episode_id", "timestep"):
        data = td.get("incremental_" + key, None)
        if (
            data is None
            or data.shape != dones.shape
            or data.dtype != torch.int64
        ):
            raise ValueError("Malformed episode ledger")
        expected = data[:-1] + 1
        if key == "episode_id":
            expected = data[:-1] + dones[:-1].long()
        else:
            expected = torch.where(dones[:-1], 0, expected)
        if torch.any(data[1:] != expected):
            raise ValueError("Episode ID/timestep crossed a reset boundary")
