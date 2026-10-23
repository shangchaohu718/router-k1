"""Shared-resource admission and own-job-only stop logic, without a GPU."""

import argparse
import json
from unittest import mock

import pytest

from tools.booster_k1_retarget import sliding_router_job as job


def test_admission_reserves_space_for_both_job_and_others():
    assert job.admission_reason({"free_mib": 875})
    assert job.admission_reason({"free_mib": 10239})
    assert job.admission_reason({"free_mib": 10240}) is None
    assert job.stop_reason({"free_mib": 4096}, 6144) is None
    assert job.stop_reason({"free_mib": 4095}, 3000)
    assert job.stop_reason({"free_mib": 10000}, 6145)


def test_accounting_counts_only_owned_group_on_selected_gpu():
    output = "GPU-a, 11, 4000\nGPU-a, 12, 500\nGPU-b, 13, 9000\n"
    with (
        mock.patch.object(job.subprocess, "check_output", return_value=output),
        mock.patch.object(
            job.os, "getpgid", side_effect=lambda pid: 20 if pid == 11 else 99
        ),
    ):
        assert job.group_gpu_memory("GPU-a", 20) == 4000


def test_busy_gpu_never_spawns_a_process(tmp_path):
    args = argparse.Namespace(
        output_dir=tmp_path / "busy",
        run=True,
        gpu=0,
        num_envs=8,
        iterations=20,
        min_free_mib=10240,
        reserve_mib=4096,
        max_job_mib=6144,
        audit_report=tmp_path / "absent",
    )
    snapshot = {"uuid": "GPU-a", "free_mib": 875}
    with (
        mock.patch.object(job, "REPO_ROOT", tmp_path),
        mock.patch.object(job, "gpu_snapshot", return_value=snapshot),
        mock.patch.object(job.subprocess, "Popen") as spawn,
    ):
        assert job.run(args) == 0
        spawn.assert_not_called()
    status = json.loads((args.output_dir / "job.json").read_text())
    assert status["status"] == "waiting_for_gpu"
    assert not status["training_started"]


def test_stale_audit_cannot_launch(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {"ready_for_pilot": True, "file_sha256": {"model.py": "bad_hash"}}
        )
    )
    (tmp_path / "model.py").write_text("changed")
    with mock.patch.object(job, "REPO_ROOT", tmp_path):
        with pytest.raises(ValueError, match="Audit is stale"):
            job.validate_audit(audit)


def test_internal_expansion_cannot_reuse_external_audit(tmp_path):
    audit = tmp_path / "old.json"
    audit.write_text(
        json.dumps({"ready_for_pilot": True, "file_sha256": {"x": "y"}})
    )
    with pytest.raises(ValueError, match="own CPU audit"):
        job.validate_audit(audit, architecture="internal")


def test_internal_continuation_uses_internal_bootstrap_and_explicit_warmup(
    tmp_path,
):
    args = argparse.Namespace(
        architecture="internal",
        phase="specialist",
        output_dir=tmp_path,
        iterations=50,
        steps_per_env=8,
        checkpoint=tmp_path / "model.pt",
        warmup="off",
    )
    command = job.training_command(args)
    assert (
        "modules.actor.internal_expansion.bootstrap_from_legacy=false"
        in command
    )
    assert "modules.actor.internal_expansion.warmup=false" in command
    assert not any("sliding_router" in part for part in command)


def test_internal_evaluation_requires_artifacts():
    from pathlib import Path

    args = argparse.Namespace(
        phase="evaluation", output_dir=Path("/missing-internal-evaluation")
    )
    with pytest.raises(FileNotFoundError):
        job.validate_internal_outputs(args)


def test_stop_signals_only_the_group_created_by_guard():
    process = mock.Mock(pid=1234)
    with (
        mock.patch.object(job.os, "getpgrp", return_value=9876),
        mock.patch.object(job.os, "killpg") as kill,
    ):
        job.stop_own_group(process)
    assert {call.args[0] for call in kill.call_args_list} == {1234}


def test_stop_refuses_the_callers_process_group():
    with (
        mock.patch.object(job.os, "getpgrp", return_value=1234),
        mock.patch.object(job.os, "killpg") as kill,
    ):
        with pytest.raises(RuntimeError, match="own process group"):
            job.stop_own_group(mock.Mock(pid=1234))
        kill.assert_not_called()


def test_specialist_continuation_needs_verified_pilot(tmp_path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "job.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "peak_own_memory_mib": 3140,
            }
        )
    )
    checkpoint = pilot / "run/model_19.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    freeze = {
        "general_norm_sigma_bitwise_preserved": True,
        "general_changed_tensors": [],
        "gate_changed_tensors_since_first_save": [],
        "pilot_checkpoint": str(
            pilot / "run/model_19/actor/model.safetensors"
        ),
    }
    (pilot / "freeze_verification.json").write_text(json.dumps(freeze))
    args = argparse.Namespace(
        previous_pilot=pilot, checkpoint=checkpoint, max_job_mib=6144
    )
    job.validate_measured_continuation(args)
    freeze["general_changed_tensors"] = ["some_weight"]
    (pilot / "freeze_verification.json").write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="changed frozen"):
        job.validate_measured_continuation(args)


def test_continuation_keeps_composite_checkpoint_without_bootstrapping(
    tmp_path,
):
    args = argparse.Namespace(
        phase="specialist",
        output_dir=tmp_path,
        iterations=500,
        steps_per_env=32,
        checkpoint=tmp_path / "model_19.pt",
    )
    command = job.training_command(args)
    assert (
        "modules.actor.sliding_router.bootstrap_from_general=false" in command
    )
    assert "algo.config.checkpoint_load_mode=finetune" in command
    assert "algo.config.num_steps_per_env=32" in command


def test_evaluation_stages_links_and_changes_only_its_config(tmp_path):
    from omegaconf import OmegaConf

    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model_499.pt"
    checkpoint.touch()
    checkpoint.with_suffix("").mkdir()
    config = OmegaConf.create(
        {
            "modules": {
                "actor": {
                    "type": "FrozenBaseSlidingPolicy",
                    "sliding_router": {
                        "mode": "specialist",
                        "bootstrap_from_general": False,
                    },
                }
            }
        }
    )
    OmegaConf.save(config, source / "config.yaml")
    output = tmp_path / "evaluation"
    output.mkdir()
    data = tmp_path / "dataset"
    data.mkdir()
    args = argparse.Namespace(
        checkpoint=checkpoint,
        dataset_root=data,
        output_dir=output,
        route="base",
        num_envs=2,
        max_eval_frames=1000,
    )
    command = job.evaluation_command(args)
    staged = output / "checkpoint"
    assert (staged / "model_499.pt").is_symlink()
    staged_mode = OmegaConf.load(
        staged / "config.yaml"
    ).modules.actor.sliding_router.mode
    source_mode = OmegaConf.load(
        source / "config.yaml"
    ).modules.actor.sliding_router.mode
    assert staged_mode == "base" and source_mode == "specialist"
    assert "export_policy=false" in command
    assert "dump_npzs=true" in command
