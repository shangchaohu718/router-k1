#!/usr/bin/env python3
"""Check shared GPU headroom, or run one bounded and monitored pilot.

Default mode only checks resources. --run is required to launch a pilot.
The monitor can stop only the process group it created. NVIDIA/PhysX memory
is monitored, not a hardware reservation; another process can grow at any time.
"""

from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_CONFIGS = {
    "admission": "motion_tracking_booster_k1_moonwalk_admission",
    "historical_single_critic": (
        "motion_tracking_booster_k1_moonwalk_internal_moe"
    ),
}


def selected_config_name(args):
    if getattr(args, "architecture", "external") != "internal":
        return "motion_tracking_booster_k1_moonwalk_frozen_router"
    profile = getattr(args, "internal_profile", "admission")
    if profile not in INTERNAL_CONFIGS:
        raise ValueError(f"Unknown internal launch profile: {profile}")
    return INTERNAL_CONFIGS[profile]


def compose_training_config(args):
    """Resolve the shell command's Hydra inputs without a simulator."""
    from hydra import compose, initialize_config_dir

    config_name = selected_config_name(args)
    command = training_command(args)
    # These are train_motion_tracking.sh's COMMON_ARGS and the single-process
    # runtime context. The script receives every trailing override unchanged.
    overrides = [
        f"num_envs={args.num_envs}",
        "headless=true",
        f"experiment_name={config_name}",
        *command[5:],
        "num_processes=1",
        "main_process=true",
        "process_id=0",
        "motion_h5_path=null",
        f"experiment_save_dir={args.output_dir / 'run'}",
    ]
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "holomotion/config"), version_base=None
    ):
        return compose(
            config_name=f"training/motion_tracking/{config_name}",
            overrides=overrides,
        )


def build_launch_contract(args):
    """CPU-reviewable configuration binding; this does not authorize launch."""
    from omegaconf import OmegaConf

    from holomotion.src.motion_tracking.incremental_task import (
        config_hash,
        select_reward_profile,
    )

    if getattr(args, "phase", "pilot") == "evaluation":
        return build_evaluation_contract(args)
    config = compose_training_config(args)
    actor = OmegaConf.to_container(config.modules.actor, resolve=True)
    critic = OmegaConf.to_container(config.modules.critic, resolve=True)
    rewards = OmegaConf.to_container(config.rewards, resolve=True)
    architecture = getattr(args, "architecture", "external")
    profile = getattr(args, "internal_profile", "admission")
    task = (
        OmegaConf.to_container(config.incremental_task, resolve=True)
        if ("incremental_task" in config)
        else None
    )
    isolated = architecture == "internal" and profile == "admission"
    if isolated:
        if (
            actor["type"] != "InternalExpandedMoEPolicy"
            or critic["type"] != "FrozenLegacySlidingCritic"
            or task is None
            or task["task"] != "sliding"
            or critic.get("task_isolation") != task
            or rewards.get("_config", {}).get("incremental_task") != task
        ):
            raise ValueError("Internal launch requires isolated reward/Critic")
        expansion = actor["internal_expansion"]
        if (
            expansion.get("sequence_replay") != "cached"
            or expansion.get("replay_chunk_size") != 0
            or expansion.get("admission", {}).get("mode") != "open"
            or int(config.algo.config.num_mini_batches) != 1
            or int(config.algo.config.rollout_forward_chunk_size) != 0
        ):
            raise ValueError(
                "Internal launch requires the audited cached layout"
            )
        if (
            config_hash(select_reward_profile(rewards))
            != task["sliding_reward_hash"]
        ):
            raise ValueError("Internal launch sliding reward hash mismatch")
    semantic = {
        "actor": actor,
        "critic": critic,
        "rewards": rewards,
        "task": task,
        "robot": OmegaConf.to_container(config.robot, resolve=True),
        "gamma": float(config.algo.config.gamma),
        "lambda": float(config.algo.config.lam),
        "environment": {
            key: OmegaConf.to_container(config.env.config[key], resolve=True)
            if OmegaConf.is_config(config.env.config[key])
            else config.env.config[key]
            for key in (
                "domain_rand",
                "terrain",
                "obs",
                "terminations",
                "simulation",
                "scene",
                "actions",
                "commands",
                "normalization",
                "resample_motion_when_training",
                "curriculum",
            )
        },
        "algorithm": {
            key: OmegaConf.to_container(config.algo.config[key], resolve=True)
            if OmegaConf.is_config(config.algo.config[key])
            else config.algo.config[key]
            for key in config.algo.config
            if key
            not in {
                "module_dict",
                "num_learning_iterations",
                "log_interval",
                "save_interval",
                "checkpoint_load_mode",
                "num_steps_per_env",
            }
        },
    }
    # Initialization flags legitimately differ between a pilot and its resume.
    # Preserve the substantive task/model recipe as the lineage identity.
    if architecture == "internal":
        actor["internal_expansion"].pop("bootstrap_from_legacy", None)
    if task is not None:
        for spec in (
            task,
            critic["task_isolation"],
            rewards["_config"]["incremental_task"],
        ):
            spec.pop("bootstrap_from_legacy", None)
    return {
        "version": 1,
        "architecture": architecture,
        "internal_profile": profile if architecture == "internal" else None,
        "config_name": selected_config_name(args),
        "task_isolated": isolated,
        "historical_reproduction_only": architecture == "internal"
        and not isolated,
        "critic_type": critic["type"],
        "task_id": None if task is None else task["task"],
        "profile_sha256": config_hash(semantic),
        "num_envs": int(config.num_envs),
        "rollout_steps": int(config.algo.config.num_steps_per_env),
        "num_mini_batches": int(config.algo.config.num_mini_batches),
        "rollout_forward_chunk_size": int(
            config.algo.config.get("rollout_forward_chunk_size", 0)
        ),
        "checkpoint": str((REPO_ROOT / config.checkpoint).resolve()),
        "checkpoint_load_mode": str(config.algo.config.checkpoint_load_mode),
    }


def checkpoint_files(checkpoint):
    """Pin config, .pt metadata and weight sidecars without loading pickle."""
    checkpoint = checkpoint.resolve()
    paths = [checkpoint, checkpoint.parent / "config.yaml"]
    for role in ("actor", "critic"):
        folder = checkpoint.with_suffix("") / role
        candidates = [
            folder / name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (folder / name).is_file()
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {role} weight sidecar")
        paths.extend(candidates)
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def validate_audit(path, architecture="external", launch_contract=None):
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (
        architecture == "internal"
        and audit.get("architecture") != "internal_moe"
    ):
        raise ValueError("Internal expansion requires its own CPU audit")
    if not audit.get("ready_for_pilot") or not audit.get("file_sha256"):
        raise ValueError("CPU audit has not approved a bounded pilot")
    if architecture == "internal":
        if (
            launch_contract is None
            or audit.get("launch_contract") != launch_contract
        ):
            raise ValueError(
                "CPU audit does not bind this exact launch contract"
            )
        config_path = (
            "holomotion/config/training/motion_tracking/"
            + launch_contract["config_name"]
            + ".yaml"
        )
        required = {
            "tools/booster_k1_retarget/sliding_router_job.py",
            config_path,
            "holomotion/scripts/training/train_motion_tracking.sh",
        }
        if launch_contract["task_isolated"]:
            from holomotion.src.motion_tracking.incremental_task import (
                implementation_contract,
            )

            required.update(implementation_contract())
        if not required.issubset(audit["file_sha256"]):
            raise ValueError(
                "CPU audit omits required launcher/model/config files"
            )
        if audit.get("checkpoint_files") != checkpoint_files(
            Path(launch_contract["checkpoint"])
        ):
            raise ValueError("CPU audit checkpoint lineage mismatch")
    for relative, expected in audit["file_sha256"].items():
        actual = hashlib.sha256(
            (REPO_ROOT / relative).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Audit is stale; review changed file: {relative}"
            )


def gpu_snapshot(index):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("Expected one unambiguous GPU snapshot")
    uuid, total, used, free, utilization = [v.strip() for v in rows[0]]
    return {
        "uuid": uuid,
        "total_mib": int(total),
        "used_mib": int(used),
        "free_mib": int(free),
        "utilization_percent": int(utilization),
    }


def group_gpu_memory(uuid, process_group):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    used = 0
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        gpu_uuid, pid, memory = [v.strip() for v in row]
        if gpu_uuid != uuid:
            continue
        try:
            if os.getpgid(int(pid)) == process_group:
                used += int(memory)
        except ProcessLookupError:
            continue
    return used


def admission_reason(
    snapshot, min_free_mib=10240, reserve_mib=4096, max_job_mib=6144
):
    needed = max(min_free_mib, reserve_mib + max_job_mib)
    if snapshot["free_mib"] < needed:
        return f"Need {needed} MiB free; only {snapshot['free_mib']} MiB free"
    return None


def stop_reason(snapshot, own_mib, reserve_mib=4096, max_job_mib=6144):
    if own_mib > max_job_mib:
        return f"Own process group exceeds budget: {own_mib} MiB"
    if snapshot["free_mib"] < reserve_mib:
        return f"Shared GPU reserve fell below {reserve_mib} MiB"
    return None


def stop_own_group(process):
    # start_new_session=True establishes a group owned by this exact child.
    # Never signal an arbitrary PID from nvidia-smi or an existing job.
    if process.pid == os.getpgrp():
        raise RuntimeError("Refusing to signal the guard's own process group")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        # The launcher may exit before workers: finish only this owned group.
    process.wait(timeout=10)


def write_status(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_measured_continuation(args):
    """A longer run needs an actually completed, frozen-base verified pilot."""
    if args.previous_pilot is None:
        raise ValueError("Continuation requires a previous measured pilot")
    pilot_dir = args.previous_pilot.resolve()
    pilot = json.loads((pilot_dir / "job.json").read_text())
    freeze = json.loads((pilot_dir / "freeze_verification.json").read_text())
    if pilot.get("status") != "completed" or pilot.get("exit_code") != 0:
        raise ValueError("Previous pilot did not complete successfully")
    if not 0 < pilot.get("peak_own_memory_mib", 0) < args.max_job_mib:
        raise ValueError("Previous pilot has no acceptable measured footprint")
    if not freeze.get("general_norm_sigma_bitwise_preserved"):
        raise ValueError("Previous pilot did not preserve the frozen base")
    if freeze.get("general_changed_tensors") or freeze.get(
        "gate_changed_tensors_since_first_save"
    ):
        raise ValueError("Previous pilot changed frozen parameters")
    expected = (
        REPO_ROOT / freeze["pilot_checkpoint"]
    ).parent.parent.with_suffix(".pt")
    if (
        args.checkpoint is None
        or args.checkpoint.resolve() != expected.resolve()
    ):
        raise ValueError(
            "Continuation checkpoint must match the verified pilot"
        )
    if not expected.is_file():
        raise FileNotFoundError(expected)
    if getattr(args, "architecture", "external") == "internal":
        current = build_launch_contract(args)
        previous = pilot.get("launch_contract", {})
        for field in (
            "architecture",
            "internal_profile",
            "config_name",
            "profile_sha256",
        ):
            if current[field] != previous.get(field):
                raise ValueError(
                    "Continuation pilot configuration lineage mismatch"
                )
        if current["task_isolated"] and not freeze.get(
            "legacy_critic_state_bitwise_preserved"
        ):
            raise ValueError(
                "Continuation needs frozen legacy Critic evidence"
            )
        if pilot.get("verified_artifacts", {}).get(
            "checkpoint_files"
        ) != checkpoint_files(expected):
            raise ValueError(
                "Continuation checkpoint/sidecar lineage mismatch"
            )


def training_command(args):
    phase = getattr(args, "phase", "pilot")
    if phase == "pilot" and getattr(args, "checkpoint", None) is not None:
        raise ValueError(
            "Pilot initialization uses the registered baseline; "
            "checkpoint continuation requires phase=specialist"
        )
    command = [
        "nice",
        "-n",
        "10",
        "bash",
        "holomotion/scripts/training/train_motion_tracking.sh",
        f"experiment_dir={args.output_dir / 'run'}",
        f"algo.config.num_learning_iterations={args.iterations}",
        f"algo.config.num_steps_per_env={getattr(args, 'steps_per_env', 8)}",
        "algo.config.num_mini_batches=1",
        f"algo.config.save_interval={100 if phase == 'specialist' else 10}",
        f"algo.config.log_interval={25 if phase == 'specialist' else 1}",
    ]
    internal = getattr(args, "architecture", "external") == "internal"
    if internal:
        command.append(
            "modules.actor.internal_expansion.warmup="
            + str(getattr(args, "warmup", "off") == "on").lower()
        )
    if phase == "specialist":
        bootstrap_override = (
            "modules.actor.internal_expansion.bootstrap_from_legacy=false"
            if getattr(args, "architecture", "external") == "internal"
            else "modules.actor.sliding_router.bootstrap_from_general=false"
        )
        command.extend(
            [
                f"checkpoint={args.checkpoint.resolve()}",
                bootstrap_override,
                "algo.config.checkpoint_load_mode=finetune",
            ]
        )
        if (
            internal
            and getattr(args, "internal_profile", "admission") == "admission"
        ):
            command.append("incremental_task.bootstrap_from_legacy=false")
    return command


def validate_internal_outputs(args):
    """Isaac can exit zero after an exception; require actual artifacts."""
    import numpy as np

    if args.phase == "evaluation":
        recording = args.output_dir / "ordered_observations.npz"
        with np.load(recording, allow_pickle=False) as data:
            obs = data["observations"]
            if (
                int(data["schema_version"]) != 1
                or obs.ndim != 2
                or not len(obs)
                or not np.isfinite(obs).all()
            ):
                raise ValueError("No valid ordered evaluation observations")
            frames = len(obs)
        trajectories = list(
            args.output_dir.glob("checkpoint/isaaclab_eval_output*/*.npz")
        )
        if not trajectories:
            raise ValueError("No native evaluation trajectory artifacts")
        return {
            "ordered_frames": frames,
            "native_trajectory_files": len(trajectories),
        }
    checkpoint = args.output_dir / "run" / f"model_{args.iterations - 1}.pt"
    weights = checkpoint.with_suffix("") / "actor"
    if not checkpoint.is_file() or not any(
        (weights / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise ValueError("Missing final internal training checkpoint/weights")
    return {
        "final_checkpoint": str(checkpoint),
        "checkpoint_files": checkpoint_files(checkpoint),
    }


def _evaluation_config(args):
    """Prepare task/profile semantics without creating output files."""
    from omegaconf import OmegaConf

    if args.checkpoint is None or args.dataset_root is None:
        raise ValueError("Evaluation requires checkpoint and dataset_root")
    source = args.checkpoint.resolve()
    if not source.is_file() or not args.dataset_root.is_dir():
        raise FileNotFoundError("Evaluation checkpoint or dataset is missing")
    config = OmegaConf.load(source.parent / "config.yaml")
    internal = str(config.modules.actor.type) == "InternalExpandedMoEPolicy"
    if internal:
        if args.route not in {"legacy", "cooperative", "new_only"}:
            raise ValueError(
                "Internal evaluation requires "
                "legacy/cooperative/new_only routing"
            )
        config.modules.actor.internal_expansion.mode = args.route
        config.modules.actor.internal_expansion.stage = "eval"
        config.modules.actor.internal_expansion.warmup = False
        config.modules.actor.internal_expansion.bootstrap_from_legacy = False
        if "incremental_task" in config:
            task = getattr(args, "task", None)
            profile = getattr(args, "evaluation_profile", None)
            expected_route = {
                "legacy_control": "legacy",
                "automatic": "cooperative",
            }.get(profile)
            if task not in {"legacy", "sliding"} or expected_route is None:
                raise ValueError(
                    "Isolated evaluation needs explicit task/profile"
                )
            if args.route != expected_route:
                raise ValueError("Evaluation route conflicts with its profile")
            from holomotion.src.motion_tracking.incremental_evaluation import (
                stage_registered_evaluation,
                validate_registered_evaluation_checkpoint,
            )

            config = stage_registered_evaluation(
                config,
                task=task,
                profile=profile,
                dataset_root=args.dataset_root,
                max_frame_length=args.max_eval_frames,
            )
            validate_registered_evaluation_checkpoint(config, source)
    elif str(config.modules.actor.type) == "FrozenBaseSlidingPolicy":
        config.modules.actor.sliding_router.mode = args.route
        config.modules.actor.sliding_router.bootstrap_from_general = False
    elif args.route != "base":
        raise ValueError("Original controller can only be evaluated as base")
    if getattr(args, "architecture", None) == "internal":
        isolated = (
            getattr(args, "internal_profile", "admission") == "admission"
        )
        actual = (
            "incremental_task" in config
            and str(config.modules.critic.type) == "FrozenLegacySlidingCritic"
        )
        if not internal or actual != isolated:
            raise ValueError(
                "Evaluation checkpoint does not match internal profile"
            )
    return source, config, internal


def build_evaluation_contract(args):
    from omegaconf import OmegaConf

    from holomotion.src.motion_tracking.incremental_task import config_hash

    source, config, internal = _evaluation_config(args)
    if not internal:
        raise ValueError(
            "Internal audit contract needs an internal checkpoint"
        )
    isolated = "incremental_task" in config
    semantic = {
        "actor": OmegaConf.to_container(config.modules.actor, resolve=True),
        "critic": OmegaConf.to_container(config.modules.critic, resolve=True),
        "rewards": OmegaConf.to_container(config.rewards, resolve=True),
        "task": getattr(args, "task", None),
        "evaluation_profile": getattr(args, "evaluation_profile", None),
        "dataset_root": str(args.dataset_root.resolve()),
        "max_eval_frames": args.max_eval_frames,
    }
    return {
        "version": 1,
        "architecture": "internal",
        "phase": "evaluation",
        "internal_profile": getattr(args, "internal_profile", "admission"),
        "config_name": selected_config_name(args),
        "task_isolated": isolated,
        "historical_reproduction_only": not isolated,
        "critic_type": str(config.modules.critic.type),
        "task_id": getattr(args, "task", None),
        "evaluation_profile": getattr(args, "evaluation_profile", None),
        "route": args.route,
        "profile_sha256": config_hash(semantic),
        "num_envs": args.num_envs,
        "checkpoint": str(source),
    }


def evaluation_command(args):
    """Use read-only weight links to preserve source run outputs."""
    from omegaconf import OmegaConf

    source, config, internal = _evaluation_config(args)
    stage = args.output_dir / "checkpoint"
    stage.mkdir(exist_ok=False)
    OmegaConf.save(config, stage / "config.yaml")
    staged = stage / source.name
    staged.symlink_to(source)
    (stage / source.stem).symlink_to(
        source.with_suffix(""), target_is_directory=True
    )
    # Fixed shell code; every dynamic argument is passed separately via "$@".
    return [
        "nice",
        "-n",
        "10",
        "bash",
        "-c",
        "source train.env >/dev/null\n"
        'exec "$Train_CONDA_PREFIX/bin/python" "$@"',
        "sliding-router-evaluation",
        "tools/booster_k1_retarget/evaluate_internal_moe.py"
        if internal
        else "tools/booster_k1_retarget/evaluate_sliding_router.py",
        "--config-name=evaluation/eval_isaaclab",
        "robot=booster/K1/22dof/22dof_training_isaaclab",
        "env/terminations=termination_motion_tracking",
        "headless=true",
        f"num_envs={args.num_envs}",
        f"checkpoint={staged}",
        f"motion_h5_path={json.dumps([str(args.dataset_root.resolve())])}",
        "project_name=HoloMotionK1InternalMoEEvaluation"
        if internal
        else "project_name=HoloMotionK1FrozenRouterEvaluation",
        "export_policy=false",
        "dump_npzs=true",
        "calc_per_clip_metrics=false",
        "generate_report=false",
        "robot.motion.handpicked_motion_names=null",
        "robot.motion.excluded_motion_names=null",
        f"robot.motion.max_frame_length={args.max_eval_frames}",
        f"+algo.config.offline_eval_max_steps={args.max_eval_frames}",
        "robot.motion.dataloader.num_workers=0",
        "robot.motion.dataloader.timeout=0",
    ]


def run(args):
    if getattr(args, "review_config", False):
        if args.run:
            raise ValueError("CPU configuration review cannot launch")
        print(json.dumps(build_launch_contract(args), indent=2))
        return 0
    if args.min_free_mib < 10240 or args.reserve_mib < 4096:
        raise ValueError(
            "This shared-GPU pilot requires 10 GiB admission and 4 GiB reserve"
        )
    if not 0 < args.max_job_mib <= 6144 or not 1 <= args.num_envs <= 16:
        raise ValueError("Pilot budget is at most 6 GiB and 16 environments")
    phase = getattr(args, "phase", "pilot")
    limit = 500 if phase == "specialist" else 20
    if not 1 <= args.iterations <= limit:
        raise ValueError(f"This {phase} run is limited to {limit} iterations")
    if not 1 <= getattr(args, "steps_per_env", 8) <= 32:
        raise ValueError(
            "At most 32 rollout steps per environment are allowed"
        )
    if phase == "specialist":
        validate_measured_continuation(args)
    if phase == "evaluation" and not 1 <= args.max_eval_frames <= 2000:
        raise ValueError("Bounded evaluation allows at most 2000 frames")
    if (
        phase == "evaluation"
        and getattr(args, "architecture", "external") == "internal"
        and args.num_envs != 1
    ):
        raise ValueError(
            "Internal ordered evaluation requires one environment"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = REPO_ROOT / "test_outputs/sliding_router_gpu_job.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                json.dumps(
                    {"status": "already_running", "training_started": False}
                )
            )
            return 0
        status_path = args.output_dir / "job.json"
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if previous.get("training_started") or previous.get("job_started"):
                if args.run:
                    raise ValueError(
                        "This output already contains a launched job; "
                        "choose a new output directory"
                    )
                print(json.dumps(previous, indent=2))
                return 0
        snapshot = gpu_snapshot(args.gpu)
        reason = admission_reason(
            snapshot, args.min_free_mib, args.reserve_mib, args.max_job_mib
        )
        status = {
            "status": "waiting_for_gpu" if reason else "resources_available",
            "training_started": False,
            "job_started": False,
            "gpu": snapshot,
            "reason": reason,
        }
        write_status(status_path, status)
        if reason or not args.run:
            print(json.dumps(status, indent=2))
            return 0
        architecture = getattr(args, "architecture", "external")
        launch_contract = (
            build_launch_contract(args) if architecture == "internal" else None
        )
        validate_audit(args.audit_report, architecture, launch_contract)
        # Sustained headroom, checked again immediately before spawn.
        for _ in range(2):
            time.sleep(5)
            snapshot = gpu_snapshot(args.gpu)
            reason = admission_reason(
                snapshot, args.min_free_mib, args.reserve_mib, args.max_job_mib
            )
            if reason:
                status.update(
                    status="waiting_for_gpu", gpu=snapshot, reason=reason
                )
                write_status(status_path, status)
                return 0
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=snapshot["uuid"],
            NUM_ENVS=str(args.num_envs),
            CONFIG_NAME=selected_config_name(args),
            TORCH_COMPILE_DISABLE="1",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            PYTHONPATH=str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        )
        env.pop("HOLOMOTION_TRAIN_DRY_RUN", None)
        current_contract = (
            build_launch_contract(args) if architecture == "internal" else None
        )
        if current_contract != launch_contract:
            raise ValueError(
                "Launch configuration changed during resource checks"
            )
        validate_audit(args.audit_report, architecture, current_contract)
        if phase == "evaluation":
            command = evaluation_command(args)
            if architecture == "internal":
                env["INTERNAL_MOE_RECORD_PATH"] = str(
                    args.output_dir / "ordered_observations.npz"
                )
            else:
                env["SLIDING_GATE_RECORD_PATH"] = str(
                    args.output_dir / "gate_samples.npz"
                )
        else:
            command = training_command(args)
        status.update(
            command=command,
            max_job_mib=args.max_job_mib,
            reserve_mib=args.reserve_mib,
            gpu=snapshot,
            phase=phase,
            launch_contract=launch_contract,
        )
        with (args.output_dir / "train.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                status.update(
                    status="running",
                    job_started=True,
                    training_started=phase != "evaluation",
                    pid=process.pid,
                )
                write_status(status_path, status)
                started = time.monotonic()
                while process.poll() is None:
                    snapshot = gpu_snapshot(args.gpu)
                    own = group_gpu_memory(snapshot["uuid"], process.pid)
                    reason = stop_reason(
                        snapshot, own, args.reserve_mib, args.max_job_mib
                    )
                    if time.monotonic() - started > getattr(
                        args, "max_runtime_seconds", 1800
                    ):
                        reason = "Own job exceeded its runtime limit"
                    status.update(
                        gpu=snapshot,
                        own_memory_mib=own,
                        peak_own_memory_mib=max(
                            own, status.get("peak_own_memory_mib", 0)
                        ),
                    )
                    if reason:
                        stop_own_group(process)
                        status.update(
                            status="stopped_for_memory", reason=reason
                        )
                        break
                    write_status(status_path, status)
                    time.sleep(2)
                else:
                    status["status"] = (
                        "completed" if process.returncode == 0 else "failed"
                    )
                status["exit_code"] = process.wait(timeout=10)
                stop_own_group(process)
                if (
                    status["status"] == "completed"
                    and architecture == "internal"
                ):
                    try:
                        status["verified_artifacts"] = (
                            validate_internal_outputs(args)
                        )
                    except (OSError, ValueError, KeyError) as error:
                        status.update(
                            status="failed_artifact_validation",
                            reason=repr(error),
                        )
            except BaseException as error:
                stop_own_group(process)
                status.update(
                    status="stopped_after_guard_error", reason=repr(error)
                )
                write_status(status_path, status)
                raise
            write_status(status_path, status)
        print(json.dumps(status, indent=2))
        return 0 if status["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--review-config",
        action="store_true",
        help="Print CPU launch contract without GPU checks or launching",
    )
    parser.add_argument(
        "--architecture", choices=["external", "internal"], default="external"
    )
    parser.add_argument(
        "--internal-profile",
        choices=list(INTERNAL_CONFIGS),
        default="admission",
    )
    parser.add_argument("--task", choices=["legacy", "sliding"])
    parser.add_argument(
        "--evaluation-profile", choices=["legacy_control", "automatic"]
    )
    parser.add_argument("--warmup", choices=["on", "off"], default="off")
    parser.add_argument(
        "--phase",
        choices=["pilot", "specialist", "evaluation"],
        default="pilot",
    )
    parser.add_argument("--steps-per-env", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--previous-pilot", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--route",
        choices=[
            "base",
            "specialist",
            "soft",
            "hard",
            "temporal_hard",
            "legacy",
            "cooperative",
            "new_only",
        ],
        default="base",
    )
    parser.add_argument("--max-eval-frames", type=int, default=1000)
    parser.add_argument("--max-runtime-seconds", type=int, default=1800)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--min-free-mib", type=int, default=10240)
    parser.add_argument("--reserve-mib", type=int, default=4096)
    parser.add_argument("--max-job-mib", type=int, default=6144)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=REPO_ROOT / "test_outputs/sliding_router_audit.json",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
