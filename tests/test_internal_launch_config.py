"""Real Hydra launcher resolution and lineage checks, without GPU calls."""

import argparse
import hashlib
import json
from dataclasses import asdict

import pytest
import torch
from omegaconf import OmegaConf

from holomotion.src.modules.incremental_critic import FrozenLegacySlidingCritic
from holomotion.src.motion_tracking import incremental_task
from tools.booster_k1_retarget import sliding_router_job as job
from tools.booster_k1_retarget.preflight_task_isolation import (
    sample_critic_obs,
)


def arguments(tmp_path, **changes):
    defaults = {
        "architecture": "internal",
        "internal_profile": "admission",
        "phase": "pilot",
        "output_dir": tmp_path / "output",
        "num_envs": 8,
        "iterations": 20,
        "steps_per_env": 8,
        "warmup": "off",
        "checkpoint": None,
        "run": False,
        "review_config": False,
    }
    return argparse.Namespace(**(defaults | changes))


def test_default_internal_command_resolves_real_isolated_critic_and_rewards(
    tmp_path,
):
    args = arguments(tmp_path)
    config = job.compose_training_config(args)
    critic_cfg = OmegaConf.to_container(config.modules.critic, resolve=True)
    task = OmegaConf.to_container(config.incremental_task, resolve=True)
    rewards = OmegaConf.to_container(config.rewards, resolve=True)
    assert critic_cfg["type"] == "FrozenLegacySlidingCritic"
    assert critic_cfg["task_isolation"] == task
    assert rewards["_config"]["incremental_task"] == task
    assert task["task"] == "sliding"
    sliding_reward = incremental_task.select_reward_profile(rewards)
    assert (
        incremental_task.config_hash(sliding_reward)
        == task["sliding_reward_hash"]
    )
    rewards["_config"]["incremental_task"]["task"] = "legacy"
    old_reward = incremental_task.select_reward_profile(rewards)
    assert (
        incremental_task.config_hash(old_reward) == task["legacy_reward_hash"]
    )
    assert set(sliding_reward) - set(old_reward) == set(
        incremental_task.SLIDING_TERMS
    )
    schema = critic_cfg["obs_schema"]
    sample = sample_critic_obs(
        schema,
        int(config.robot.actions_dim),
        int(config.robot.num_bodies),
        batch=1,
    )
    critic = FrozenLegacySlidingCritic(schema, critic_cfg, obs_example=sample)
    assert critic.task == "sliding"
    assert all(not p.requires_grad for p in critic.legacy.parameters())
    assert any(p.requires_grad for p in critic.sliding.parameters())
    assert {p.data_ptr() for p in critic.legacy.parameters()}.isdisjoint(
        {p.data_ptr() for p in critic.sliding.parameters()}
    )
    expansion = config.modules.actor.internal_expansion
    assert expansion.sequence_replay == "cached"
    assert expansion.replay_chunk_size == 0
    assert expansion.admission.mode == "open"
    assert config.algo.config.num_mini_batches == 1
    assert config.algo.config.rollout_forward_chunk_size == 0
    assert config.num_envs == args.num_envs


def test_historical_single_critic_requires_explicit_profile(tmp_path):
    args = arguments(tmp_path, internal_profile="historical_single_critic")
    cfg = job.compose_training_config(args)
    assert cfg.modules.critic.type == "MLP"
    assert "incremental_task" not in cfg
    assert "admission" not in cfg.modules.actor.internal_expansion
    contract = job.build_launch_contract(args)
    assert contract["historical_reproduction_only"]
    assert not contract["task_isolated"]
    assert contract["config_name"] != job.selected_config_name(
        arguments(tmp_path)
    )


def test_continuation_disables_actor_and_critic_bootstrap_in_composed_config(
    tmp_path,
):
    args = arguments(
        tmp_path, phase="specialist", checkpoint=tmp_path / "model.pt"
    )
    cfg = job.compose_training_config(args)
    assert not cfg.modules.actor.internal_expansion.bootstrap_from_legacy
    assert not cfg.incremental_task.bootstrap_from_legacy
    assert not cfg.modules.critic.task_isolation.bootstrap_from_legacy
    assert not cfg.rewards._config.incremental_task.bootstrap_from_legacy
    assert cfg.checkpoint == str(args.checkpoint)
    pilot = job.build_launch_contract(arguments(tmp_path))
    continuation = job.build_launch_contract(args)
    assert pilot["profile_sha256"] == continuation["profile_sha256"]


def test_cpu_config_review_never_inspects_gpu_or_starts_process(
    tmp_path, monkeypatch, capsys
):
    args = arguments(tmp_path, review_config=True)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Configuration-only review touched GPU or subprocess launch"
        )

    monkeypatch.setattr(job, "gpu_snapshot", forbidden)
    monkeypatch.setattr(job.subprocess, "Popen", forbidden)
    assert job.run(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["critic_type"] == "FrozenLegacySlidingCritic"
    assert result["task_isolated"]
    assert not args.output_dir.exists()


def prepared_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "REPO_ROOT", tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"metadata")
    (tmp_path / "config.yaml").write_text("modules: {}")
    for role in ("actor", "critic"):
        folder = checkpoint.with_suffix("") / role
        folder.mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(role.encode())
    contract = {
        "config_name": "motion_tracking_booster_k1_moonwalk_admission",
        "task_isolated": True,
        "checkpoint": str(checkpoint),
        "profile_sha256": "registered",
    }
    required = (
        "tools/booster_k1_retarget/sliding_router_job.py",
        "holomotion/scripts/training/train_motion_tracking.sh",
        "holomotion/config/training/motion_tracking/"
        + contract["config_name"]
        + ".yaml",
        "core.py",
    )
    hashes = {}
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    core_hash = hashes["core.py"]
    monkeypatch.setattr(
        incremental_task,
        "implementation_contract",
        lambda: {"core.py": core_hash},
    )
    audit = {
        "architecture": "internal_moe",
        "ready_for_pilot": True,
        "launch_contract": contract,
        "file_sha256": hashes,
        "checkpoint_files": job.checkpoint_files(checkpoint),
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    return path, contract, audit


def test_audit_requires_matching_profile_files_and_checkpoint_sidecars(
    tmp_path, monkeypatch
):
    path, contract, audit = prepared_audit(tmp_path, monkeypatch)
    job.validate_audit(path, "internal", contract)
    wrong = dict(contract, profile_sha256="historical_single_critic")
    with pytest.raises(ValueError, match="exact launch contract"):
        job.validate_audit(path, "internal", wrong)
    sidecar = tmp_path / "model/critic/model.safetensors"
    sidecar.write_bytes(b"different critic")
    with pytest.raises(ValueError, match="checkpoint lineage"):
        job.validate_audit(path, "internal", contract)
    sidecar.write_bytes(b"critic")
    audit["ready_for_pilot"] = False
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="not approved"):
        job.validate_audit(path, "internal", contract)


def test_old_ready_audit_does_not_authorize_new_default(tmp_path, monkeypatch):
    path, contract, audit = prepared_audit(tmp_path, monkeypatch)
    audit.pop("launch_contract")
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="exact launch contract"):
        job.validate_audit(path, "internal", contract)


def test_pilot_does_not_silently_ignore_a_checkpoint_argument(tmp_path):
    args = arguments(tmp_path, checkpoint=tmp_path / "different.pt")
    with pytest.raises(ValueError, match="phase=specialist"):
        job.compose_training_config(args)


def test_audit_cannot_omit_critical_configuration_hashes(
    tmp_path, monkeypatch
):
    path, contract, audit = prepared_audit(tmp_path, monkeypatch)
    audit["file_sha256"].pop("core.py")
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="omits required"):
        job.validate_audit(path, "internal", contract)


def test_continuation_checks_profile_critic_evidence_and_sidecar_lineage(
    tmp_path,
):
    pilot = tmp_path / "pilot"
    checkpoint = pilot / "run/model_19.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"saved metadata")
    (checkpoint.parent / "config.yaml").write_text("{}")
    for role in ("actor", "critic"):
        folder = checkpoint.with_suffix("") / role
        folder.mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(role.encode())
    args = arguments(
        tmp_path,
        phase="specialist",
        checkpoint=checkpoint,
        previous_pilot=pilot,
        max_job_mib=6144,
    )
    report = {
        "status": "completed",
        "exit_code": 0,
        "peak_own_memory_mib": 3000,
        "launch_contract": job.build_launch_contract(args),
        "verified_artifacts": {
            "checkpoint_files": job.checkpoint_files(checkpoint)
        },
    }
    freeze = {
        "general_norm_sigma_bitwise_preserved": True,
        "legacy_critic_state_bitwise_preserved": True,
        "pilot_checkpoint": str(
            checkpoint.with_suffix("") / "actor/model.safetensors"
        ),
    }
    (pilot / "job.json").write_text(json.dumps(report))
    freeze_path = pilot / "freeze_verification.json"
    freeze_path.write_text(json.dumps(freeze))
    job.validate_measured_continuation(args)
    freeze["legacy_critic_state_bitwise_preserved"] = False
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="legacy Critic evidence"):
        job.validate_measured_continuation(args)
    freeze["legacy_critic_state_bitwise_preserved"] = True
    freeze_path.write_text(json.dumps(freeze))
    report["launch_contract"]["internal_profile"] = "historical_single_critic"
    (pilot / "job.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="configuration lineage"):
        job.validate_measured_continuation(args)
    report["launch_contract"] = job.build_launch_contract(args)
    (pilot / "job.json").write_text(json.dumps(report))
    (checkpoint.with_suffix("") / "critic/model.safetensors").write_bytes(
        b"modified"
    )
    with pytest.raises(ValueError, match="checkpoint/sidecar lineage"):
        job.validate_measured_continuation(args)


def registered_evaluation(tmp_path, task, profile):
    from holomotion.src.motion_tracking.incremental_evaluation import (
        evaluation_task_spec,
        stage_registered_evaluation,
    )
    from holomotion.src.motion_tracking.task_contract import (
        evaluation_registry_key,
    )

    args = arguments(tmp_path)
    cfg = job.compose_training_config(args)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if task == "legacy":
        registry = cfg.algo.config.incremental_legacy_evaluation
        root, frames = registry.reference_roots[0], registry.max_frame_length
    else:
        motion = cfg.env.config.commands.ref_motion.params.motion_lib_cfg
        root, frames = motion.val_hdf5_roots[0], motion.max_frame_length
    dataset = (job.REPO_ROOT / root).resolve()
    staged = stage_registered_evaluation(
        cfg,
        task=task,
        profile=profile,
        dataset_root=dataset,
        max_frame_length=frames,
    )
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model.pt"
    checkpoint.with_suffix("").mkdir()
    key = evaluation_registry_key(task, profile)
    torch.save(
        {"registered_task_specs": {key: asdict(evaluation_task_spec(staged))}},
        checkpoint,
    )
    OmegaConf.save(cfg, source / "config.yaml", resolve=True)
    args.output_dir.mkdir()
    args.phase = "evaluation"
    args.checkpoint = checkpoint
    args.dataset_root = dataset
    args.max_eval_frames = frames
    args.task = task
    args.evaluation_profile = profile
    args.route = "legacy" if profile == "legacy_control" else "cooperative"
    args.num_envs = 1
    return args, cfg


@pytest.mark.parametrize("task", ["legacy", "sliding"])
@pytest.mark.parametrize("profile", ["legacy_control", "automatic"])
def test_resolved_eval_staging_preserves_registered_task_and_physics(
    tmp_path,
    task,
    profile,
):
    args, original = registered_evaluation(tmp_path, task, profile)
    command = job.evaluation_command(args)
    cfg = OmegaConf.load(args.output_dir / "checkpoint/config.yaml")
    assert cfg.incremental_task.task == task
    assert cfg.modules.critic.task_isolation.task == task
    assert cfg.algo.config.module_dict.critic.task_isolation.task == task
    assert cfg.rewards._config.incremental_task.task == task
    assert cfg.env.config.rewards._config.incremental_task.task == task
    assert cfg.env.config.simulation == original.env.config.simulation
    assert cfg.env.config.obs == original.env.config.obs
    assert cfg.env.config.terminations == original.env.config.terminations
    assert cfg.env.config.domain_rand == original.env.config.domain_rand
    assert cfg.modules.actor.internal_expansion.mode == args.route
    assert cfg.modules.actor.internal_expansion.admission.mode == (
        "closed" if profile == "legacy_control" else "learned"
    )
    assert cfg.algo.config.incremental_evaluation_profile == profile
    assert "export_policy=false" in command
    assert OmegaConf.load(args.checkpoint.parent / "config.yaml") == original
    contract = job.build_evaluation_contract(args)
    assert contract["task_id"] == task
    assert contract["evaluation_profile"] == profile


def test_eval_rejects_unregistered_checkpoint_before_creating_staging(
    tmp_path,
):
    args, _ = registered_evaluation(tmp_path, "legacy", "automatic")
    torch.save({"registered_task_specs": {}}, args.checkpoint)
    with pytest.raises(ValueError, match="does not register"):
        job.evaluation_command(args)
    assert not (args.output_dir / "checkpoint").exists()
