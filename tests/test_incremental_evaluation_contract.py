"""Observation generation and route profiles are checkpoint contracts."""

import copy
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from holomotion.src.algo.ppo import PPO
from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.motion_tracking.incremental_evaluation import (
    evaluation_task_spec,
    load_registered_incremental_eval_config,
    stage_registered_evaluation,
    validate_registered_evaluation_checkpoint,
)
from holomotion.src.motion_tracking.incremental_task import (
    implementation_contract,
)
from holomotion.src.motion_tracking.task_contract import (
    build_task_spec,
    evaluation_registry_key,
    registered_evaluation_specs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    with initialize_config_dir(
        config_dir=str(ROOT / "holomotion/config"), version_base=None
    ):
        return compose(
            config_name="training/motion_tracking/motion_tracking_booster_k1_moonwalk_admission",
            overrides=[
                "experiment_name=cpu_contract",
                "headless=true",
                "num_processes=1",
                "main_process=true",
                "process_id=0",
                f"experiment_save_dir={tmp_path}",
            ],
        )


def plain(value):
    return OmegaConf.to_container(value, resolve=True)


def parts(cfg):
    return (
        plain(cfg.modules.actor),
        plain(cfg.modules.critic),
        plain(cfg.env.config),
        {
            "gamma": cfg.algo.config.gamma,
            "lam": cfg.algo.config.lam,
            "seed": cfg.algo.config.get("seed"),
        },
    )


def registry(cfg):
    return registered_evaluation_specs(
        *parts(cfg), plain(cfg.algo.config.incremental_legacy_evaluation)
    )


def stage(cfg, task="legacy", profile="automatic"):
    if task == "legacy":
        registration = cfg.algo.config.incremental_legacy_evaluation
        root, length = (
            registration.reference_roots[0],
            registration.max_frame_length,
        )
    else:
        motion = cfg.env.config.commands.ref_motion.params.motion_lib_cfg
        root, length = motion.val_hdf5_roots[0], motion.max_frame_length
    return stage_registered_evaluation(
        cfg,
        task=task,
        profile=profile,
        dataset_root=ROOT / root,
        max_frame_length=length,
    )


@pytest.mark.parametrize(
    "mutation",
    ["noise", "function", "params", "history", "corruption", "clipping"],
)
def test_complete_observation_generation_changes_digest(cfg, mutation):
    actor, critic, env, algo = parts(cfg)
    before = build_task_spec(actor, critic, env, algo)
    group = env["obs"]["obs_groups"]["unified"]
    term = group["atomic_obs_list"][0]["actor_ref_gravity_projection_cur"]
    if mutation == "noise":
        assert term["noise"]["params"]["n_max"] == 0
        term["noise"]["params"]["n_max"] = 10
    elif mutation == "function":
        term["func"] = "ref_base_angvel_cur"
    elif mutation == "params":
        term["params"]["ref_prefix"] = "changed_"
    elif mutation == "history":
        term["history_length"] += 1
    elif mutation == "corruption":
        group["enable_corruption"] = not group["enable_corruption"]
    else:
        env["normalization"]["clip_observations"] = 3
    after = build_task_spec(actor, critic, env, algo)
    assert before.actor_schema_hash == after.actor_schema_hash
    assert (
        before.observation_generation_hash != after.observation_generation_hash
    )
    assert before.digest != after.digest


@pytest.mark.parametrize("task", ["legacy", "sliding"])
def test_control_and_automatic_are_separate_registered_profiles(cfg, task):
    registered = registry(cfg)
    specs = []
    for profile in ("legacy_control", "automatic"):
        staged = stage(cfg, task, profile)
        spec = evaluation_task_spec(staged)
        assert registered[evaluation_registry_key(task, profile)] == asdict(
            spec
        )
        assert staged.env.config.rewards._config.incremental_task.task == task
        assert (
            staged.algo.config.module_dict.critic.task_isolation.task == task
        )
        specs.append(spec)
    assert specs[0].digest != specs[1].digest
    assert (
        specs[0].observation_generation_hash
        == specs[1].observation_generation_hash
    )
    assert specs[0].reward_config_hash == specs[1].reward_config_hash
    assert specs[0].critic_id == specs[1].critic_id


def test_automatic_cannot_be_forced_open_warmup_or_training(cfg):
    staged = stage(cfg)
    for key, value in (("mode", "legacy"), ("warmup", True)):
        bad = copy.deepcopy(staged)
        bad.modules.actor.internal_expansion[key] = value
        with pytest.raises(ValueError, match="Automatic evaluation"):
            evaluation_task_spec(bad)
    staged.modules.actor.internal_expansion.admission.mode = "open"
    with pytest.raises(ValueError, match="learned admission"):
        evaluation_task_spec(staged)
    actor, critic, env, algo = parts(stage(cfg))
    with pytest.raises(ValueError, match="evaluation only"):
        build_task_spec(
            actor, critic, env, algo, evaluation_profile="automatic"
        )


def test_checkpoint_requires_matching_profile_and_full_input_config(
    cfg, tmp_path
):
    p = tmp_path / "model.pt"
    staged = stage(cfg)
    state = {"registered_task_specs": registry(cfg)}
    torch.save(state, p)
    validate_registered_evaluation_checkpoint(staged, p)
    wrong = copy.deepcopy(staged)
    wrong.env.config.obs.context_length += 1
    with pytest.raises(ValueError, match="complete evaluation TaskSpec"):
        validate_registered_evaluation_checkpoint(wrong, p)
    del state["registered_task_specs"]["legacy:automatic"]
    torch.save(state, p)
    with pytest.raises(ValueError, match="complete evaluation TaskSpec"):
        validate_registered_evaluation_checkpoint(staged, p)


def test_native_eval_loader_preserves_registered_noise_and_physics(
    cfg, tmp_path
):
    from holomotion.src.evaluation.eval_motion_tracking_single import (
        load_training_config,
    )

    staged = stage(cfg)
    p = tmp_path / "model.pt"
    torch.save({"registered_task_specs": registry(cfg)}, p)
    OmegaConf.save(staged, tmp_path / "config.yaml", resolve=True)
    # Generic evaluator normally applies these divergent defaults. The
    # isolated entry point deliberately takes physics from its registration.
    ev = OmegaConf.create(
        {
            "checkpoint": str(p),
            "num_envs": 1,
            "headless": True,
            "env": {
                "config": {
                    "domain_rand": {"different": True},
                    "simulation": {"episode_length_s": 36000},
                }
            },
            "algo": {"config": {"offline_eval_max_steps": 500}},
            "export_policy": False,
        }
    )
    loaded = load_training_config(str(p), ev)
    assert plain(loaded.env.config.obs) == plain(staged.env.config.obs)
    for name in ("simulation", "domain_rand", "terminations", "terrain"):
        assert plain(loaded.env.config[name]) == plain(staged.env.config[name])
    assert loaded.num_envs == 1
    assert loaded.algo.config.offline_eval_max_steps == 500
    assert (
        evaluation_task_spec(loaded).digest
        == evaluation_task_spec(staged).digest
    )
    assert plain(load_registered_incremental_eval_config(staged, ev)) == plain(
        loaded
    )


def test_ppo_load_rejects_changed_observations_before_parent_load(
    cfg, tmp_path
):
    staged = stage(cfg)
    spec = plain(staged.modules.critic.task_isolation)
    algo = PPOTF.__new__(PPOTF)
    algo.incremental_task_spec = spec
    algo.incremental_contract = evaluation_task_spec(staged)
    algo.is_offline_eval = True
    algo.config = staged.algo.config
    p = tmp_path / "model.pt"
    state = {
        "incremental_task_contract": {
            k: v for k, v in spec.items() if k != "bootstrap_from_legacy"
        },
        "registered_task_specs": registry(cfg),
        "incremental_implementation_contract": implementation_contract(),
        "incremental_actor_contract": plain(
            cfg.modules.actor.internal_expansion
        ),
    }
    torch.save(state, p)
    staged.env.config.obs.context_length += 1
    algo.incremental_contract = evaluation_task_spec(staged)
    with mock.patch.object(PPO, "load") as parent:
        with pytest.raises(ValueError, match="TaskSpec"):
            algo.load(str(p))
        parent.assert_not_called()


def test_relative_reference_roots_resolve_against_repository(
    cfg, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    staged = stage(cfg)
    assert evaluation_task_spec(staged).execution_profile == "automatic"


def test_device_compilation_preserves_contract_and_motion_views(cfg):
    from holomotion.src.utils.config import compile_config_devices

    staged = stage(cfg)
    staged.device = "cpu"
    before = evaluation_task_spec(staged)
    compiled = compile_config_devices(staged)
    assert evaluation_task_spec(compiled).digest == before.digest
    motion = plain(
        compiled.env.config.commands.ref_motion.params.motion_lib_cfg
    )
    assert plain(compiled.robot.motion) == motion
    assert plain(compiled.env.config.robot.motion) == motion
    assert compiled.env.config.seed == compiled.algo.config.seed == 42


@pytest.mark.parametrize(
    "key,value",
    [
        ("scene.contact_sensor.history_length", 17),
        ("env_spacing", 0.01),
        ("replicate_physics", False),
        ("seed", 43),
        ("simulation.dt", 0.01),
        ("robot.asset.angular_damping", 1.0),
    ],
)
def test_physical_environment_changes_are_detected(cfg, key, value):
    staged = stage(cfg)
    before = evaluation_task_spec(staged)
    assert OmegaConf.select(staged.env.config, key) != value
    OmegaConf.update(staged.env.config, key, value)
    assert evaluation_task_spec(staged).digest != before.digest


def test_algorithm_seed_is_registered(cfg):
    staged = stage(cfg)
    before = evaluation_task_spec(staged)
    staged.algo.config.seed += 1
    assert evaluation_task_spec(staged).digest != before.digest
