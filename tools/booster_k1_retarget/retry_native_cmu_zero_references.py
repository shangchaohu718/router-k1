"""Follow remaining CMU ankle-only zero recoveries with native foot frames.

Reuse the immutable variant/review/index workflow, replacing only the isolated
retarget worker. Other datasets and other failure classes stay in the v4 queue.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from tools.booster_k1_retarget import retry_unrecovered_references as workflow

REPO = Path(__file__).resolve().parents[2]
SCOPE = "Only completed v4 zero-recovery CMU references whose remaining flagged class is prolonged ankle saturation"


def targets(run):
    selected = []
    for file in sorted(run.glob("batch_*/status.json")):
        status = json.loads(file.read_text())
        if status["status"] != "complete":
            continue
        report = Path(status["report"])
        row = json.loads(report.read_text())["clips"][0]
        score, _ = workflow.score_review(report)
        if score[0] or row.get("source_error") is not None:
            continue
        if set(row["reason_intervals"]) != {"ankle_limit_saturation_for_0p5s"}:
            continue
        inp = file.parent / "input"
        if not (inp / "manifest.json").exists():
            continue
        manifest = json.loads((inp / "manifest.json").read_text())
        key = status["motion_keys"][0]
        if (
            Path(
                manifest["clips"][key]["metadata"]["source_path"]
            ).suffix.lower()
            != ".amc"
        ):
            continue
        selected.append((key, inp, False))
    return selected


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker = REPO / "tools/booster_k1_retarget/retry_cmu_native_foot_frame.py"
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    mode = {
        "method": "CMU native ankle/toe/toe-tip foot frame; unchanged guard, source/geometry gates and one-variant selection",
        "scope": SCOPE,
        "native_worker_sha256": digest,
        "training_started": False,
        "training_approved": False,
    }
    path = args.output / "native_foot_mode.json"
    if path.exists() and json.loads(path.read_text()) != mode:
        raise ValueError(
            "Native foot worker changed; use a new versioned output"
        )
    workflow.dump(path, mode)
    real_run = subprocess.run

    def run_native(command, **kwargs):
        command = list(command)
        if (
            len(command) > 2
            and command[1] == "-m"
            and command[2]
            == "tools.booster_k1_retarget.retry_guarded_reference"
        ):
            if hashlib.sha256(worker.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Native foot worker changed during run")
            command[2] = (
                "tools.booster_k1_retarget.retry_cmu_native_foot_frame"
            )
        return real_run(command, **kwargs)

    old_targets, old_subprocess = workflow.collect_targets, workflow.subprocess
    workflow.collect_targets = targets
    workflow.subprocess = SimpleNamespace(
        run=run_native,
        STDOUT=subprocess.STDOUT,
        CalledProcessError=subprocess.CalledProcessError,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    try:
        workflow.main()
    finally:
        workflow.collect_targets, workflow.subprocess = (
            old_targets,
            old_subprocess,
        )


if __name__ == "__main__":
    main()
