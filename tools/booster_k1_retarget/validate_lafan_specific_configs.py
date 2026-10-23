"""Bounded LaFAN configuration study; never bulk conversion or training."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.booster_k1_retarget.recover_short_reference_candidates import dump

KEYS = [
    "dance1_subject1",
    "dance2_subject1",
    "dance1_subject3",
    "dance2_subject4",
    "walk1_subject1",
    "walk2_subject1",
]
VARIANTS = [
    "current_full",
    "old_task_weights",
    "old_root_weight",
    "lafan_root100_arms10",
    "lafan_root100_arms20",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows", nargs="+", type=int, default=[0, 3000])
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    progress = {
        "training_started": False,
        "bulk_repair_started": False,
        "status": "validation_in_progress",
        "windows": a.windows,
        "keys": KEYS,
        "completed_windows": [],
        "completed_root_gate_reviews": [],
        "promotion_policy": "No automatic promotion, training or overwrite. Root/arm/leg/contact metrics and visual review required.",
    }
    dump(a.output / "progress.json", progress)
    for start in a.windows:
        root = a.output / f"frames_{start}_{start + 500}"
        commands = [
            [
                sys.executable,
                "-m",
                "tools.booster_k1_retarget.compare_lafan_retarget_pipelines",
                "--output",
                str(root),
                "--start-frame",
                str(start),
                "--keys",
                *KEYS,
                "--variants",
                *a.variants,
            ],
            [
                sys.executable,
                "-m",
                "tools.booster_k1_retarget.review_lafan_root_fidelity",
                "--root",
                str(root),
            ],
        ]
        with (a.output / f"window_{start}.log").open("a") as log:
            for cmd in commands:
                subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            info = json.loads((root / "comparison_report.json").read_text())
            for variant in info["variants"]:
                output = root / variant / "root_checked_review"
                if not (output / "report.json").exists():
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.booster_k1_retarget.salvage_lafan_root_checked",
                            "--input",
                            str(root / variant / "raw"),
                            "--output",
                            str(output),
                            "--smpl-model",
                            "/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                progress["completed_root_gate_reviews"].append(
                    str(output.resolve())
                )
                dump(a.output / "progress.json", progress)
        progress["completed_windows"].append(start)
        dump(a.output / "progress.json", progress)
    progress["status"] = (
        "small_sample_numerical_validation_complete_visual_and_dynamics_pending"
    )
    dump(a.output / "progress.json", progress)


if __name__ == "__main__":
    main()
