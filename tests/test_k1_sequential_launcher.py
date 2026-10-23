"""Exercise orchestration with a fake trainer, never Isaac Lab or a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path("holomotion/scripts/training/train_booster_k1_curriculum_all.sh")
INITIAL = Path(
    "logs/HoloMotionBoosterK1LaFANWalkDanceJointRewardSmall/"
    "20261016_092811-motion_tracking_booster_k1_lafan_walk_dance_joint_reward_small/"
    "model_19999.pt"
)


@pytest.fixture
def mock_repo(tmp_path):
    script = tmp_path / SCRIPT
    script.parent.mkdir(parents=True)
    shutil.copyfile(REPO / SCRIPT, script)
    (tmp_path / "train.env").write_text(
        f"export Train_CONDA_PREFIX='{sys.prefix}'\n"
    )
    checkpoint = tmp_path / INITIAL
    checkpoint.parent.mkdir(parents=True)
    torch.save({"iter": 19999}, checkpoint)
    for role in ("actor", "critic"):
        folder = checkpoint.with_suffix("") / role
        folder.mkdir(parents=True)
        save_file({"weight": torch.ones(1)}, str(folder / "model.safetensors"))
    trainer = script.parent / "train_motion_tracking.sh"
    trainer.write_text(
        """#!/usr/bin/env bash
set -e
"$Train_CONDA_PREFIX/bin/python" - "$@" <<'PY'
import json, os, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
args = dict(arg.split("=", 1) for arg in sys.argv[1:])
args = {k: v.strip("'") for k, v in args.items()}
stage = int(os.environ["CONFIG_NAME"].split("_s")[1][0])
with open("calls.jsonl", "a") as out:
    out.write(json.dumps({"stage": stage, **args}) + "\\n")
if os.environ.get("HOLOMOTION_TRAIN_DRY_RUN") == "1":
    sys.exit(0)
if os.environ.get("FAIL_STAGE") == str(stage):
    sys.exit(7)
if os.environ.get("MISSING_STAGE") == str(stage):
    sys.exit(0)
iteration = int(args["algo.config.num_learning_iterations"]) - 1
path = Path(args["experiment_dir"]) / f"model_{iteration}.pt"
path.parent.mkdir(parents=True)
torch.save({"iter": iteration}, path)
for role in ("actor", "critic"):
    folder = path.with_suffix("") / role
    folder.mkdir(parents=True)
    weight = torch.ones(1)
    if os.environ.get("NAN_STAGE") == str(stage):
        weight.fill_(float("nan"))
    save_file({"weight": weight}, str(folder / "model.safetensors"))
PY
"""
    )
    return tmp_path


def run_mock(root, **overrides):
    env = dict(os.environ)
    env.pop("HOLOMOTION_TRAIN_DRY_RUN", None)
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(root / SCRIPT)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls = [
        json.loads(line)
        for line in (root / "calls.jsonl").read_text().splitlines()
    ]
    return result, calls


def test_success_passes_exact_predecessor_weights(mock_repo):
    result, calls = run_mock(mock_repo)
    assert result.returncode == 0, result.stderr
    assert [c["stage"] for c in calls] == [1, 2, 3]
    assert [c["algo.config.num_learning_iterations"] for c in calls] == [
        "3000",
        "5000",
        "12000",
    ]
    assert calls[1]["checkpoint"] == str(
        Path(calls[0]["experiment_dir"]) / "model_2999.pt"
    )
    assert calls[2]["checkpoint"] == str(
        Path(calls[1]["experiment_dir"]) / "model_4999.pt"
    )
    assert all(
        c["algo.config.finetune_continue_iteration"] == "false" for c in calls
    )


@pytest.mark.parametrize("fault", ["FAIL_STAGE", "MISSING_STAGE", "NAN_STAGE"])
def test_failure_never_advances_to_third_stage(mock_repo, fault):
    result, calls = run_mock(mock_repo, **{fault: "2"})
    assert result.returncode != 0
    assert [c["stage"] for c in calls] == [1, 2]


def test_dry_run_creates_no_training_directory(mock_repo):
    result, calls = run_mock(mock_repo, HOLOMOTION_TRAIN_DRY_RUN="1")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert not (mock_repo / "logs/HoloMotionK1SequentialCurriculum").exists()
