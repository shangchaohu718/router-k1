"""CPU-only preparation of explicitly registered incremental evaluations."""

from __future__ import annotations
import copy
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from holomotion.src.motion_tracking.task_contract import (
    EVALUATION_SEED,
    build_task_spec,
    evaluation_actor_config,
    evaluation_registry_key,
)


def _plain(value):
    return OmegaConf.to_container(value, resolve=True)


def stage_registered_evaluation(
    config, *, task, profile, dataset_root, max_frame_length
):
    """Resolve duplicated config views, preserving registered physics/obs."""
    if task not in {"legacy", "sliding"}:
        raise ValueError(
            "An explicit legacy/sliding evaluation task is required"
        )
    cfg = OmegaConf.create(_plain(config))
    if cfg.modules.critic.type != "FrozenLegacySlidingCritic":
        raise ValueError("Registered evaluation requires the isolated Critic")
    actor = evaluation_actor_config(_plain(cfg.modules.actor), profile)
    spec = _plain(cfg.modules.critic.task_isolation)
    spec.update(task=task, bootstrap_from_legacy=False)
    cfg.incremental_task = copy.deepcopy(spec)
    cfg.modules.critic.task_isolation = copy.deepcopy(spec)
    cfg.modules.actor = actor
    for rewards in (cfg.rewards, cfg.env.config.rewards):
        rewards._config.incremental_task = copy.deepcopy(spec)
    motion = cfg.env.config.commands.ref_motion.params.motion_lib_cfg
    if task == "legacy":
        registration = cfg.algo.config.incremental_legacy_evaluation
        motion.train_hdf5_roots = list(registration.reference_roots)
        motion.val_hdf5_roots = list(registration.reference_roots)
        motion.handpicked_motion_names = None
        motion.max_frame_length = int(registration.max_frame_length)
        motion.cache.swap_interval_steps = int(registration.max_frame_length)
    repo = Path(__file__).resolve().parents[3]
    expected = [(repo / str(root)).resolve() for root in motion.val_hdf5_roots]
    if expected != [Path(dataset_root).resolve()] or int(
        max_frame_length
    ) != int(motion.max_frame_length):
        raise ValueError(
            "Evaluation reference population/frame limit is not registered"
        )
    cfg.env.config.robot.motion = copy.deepcopy(motion)
    cfg.env.config.seed = EVALUATION_SEED
    cfg.algo.config.seed = EVALUATION_SEED
    cfg.robot.motion = copy.deepcopy(motion)
    cfg.train_hdf5_roots = list(motion.train_hdf5_roots)
    cfg.val_hdf5_roots = list(motion.val_hdf5_roots)
    cfg.algo.config.module_dict = copy.deepcopy(cfg.modules)
    cfg.algo.config.incremental_evaluation_profile = profile
    cfg.algo.config.checkpoint_load_mode = "resume"
    return cfg


def evaluation_task_spec(config):
    cfg = config
    profile = cfg.algo.config.get("incremental_evaluation_profile")
    if profile not in {"legacy_control", "automatic"}:
        raise ValueError(
            "Isolated evaluation requires an explicit registered profile"
        )
    return build_task_spec(
        _plain(cfg.modules.actor),
        _plain(cfg.modules.critic),
        _plain(cfg.env.config),
        {
            "gamma": float(cfg.algo.config.gamma),
            "lam": float(cfg.algo.config.lam),
            "seed": cfg.algo.config.get("seed"),
        },
        evaluation_profile=profile,
        offline_eval=True,
    )


def validate_registered_evaluation_checkpoint(config, checkpoint):
    """Fail on config/profile drift before constructing any simulator."""
    spec = evaluation_task_spec(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = evaluation_registry_key(spec.task_id, spec.execution_profile)
    if state.get("registered_task_specs", {}).get(key) != asdict(spec):
        raise ValueError(
            "Checkpoint does not register this complete evaluation TaskSpec"
        )
    return spec


def load_registered_incremental_eval_config(train_config, eval_config):
    """Keep the staged task contract; generic eval defaults are not applied.

    Only resource/output fields and the stopping budget are inherited from
    the generic evaluator. Physics, noise, observations, rewards, references
    and termination rules remain the explicitly registered staged contract.
    """
    cfg = OmegaConf.create(_plain(train_config))
    checkpoint = eval_config.checkpoint
    validate_registered_evaluation_checkpoint(cfg, checkpoint)
    for key in (
        "headless",
        "num_envs",
        "num_processes",
        "main_process",
        "process_id",
        "dump_npzs",
        "calc_per_clip_metrics",
        "generate_report",
        "project_name",
        "export_policy",
        "export_only",
    ):
        if key in eval_config and not OmegaConf.is_missing(eval_config, key):
            cfg[key] = eval_config[key]
            if key in cfg.env.config:
                cfg.env.config[key] = eval_config[key]
    budget = OmegaConf.select(
        eval_config, "algo.config.offline_eval_max_steps"
    )
    if budget is not None:
        cfg.algo.config.offline_eval_max_steps = int(budget)
    cfg.checkpoint = str(checkpoint)
    cfg.algo.config.checkpoint = str(checkpoint)
    cfg.algo.config.module_dict = copy.deepcopy(cfg.modules)
    # Runtime resource/output changes must not change the registered contract.
    validate_registered_evaluation_checkpoint(cfg, checkpoint)
    return cfg
