"""Collect bounded validation evidence and renders, never select a release."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--mid", type=Path)
    p.add_argument("--render-variant", default="old_task_weights")
    a = p.parse_args()
    deadline = time.monotonic() + 7200
    while True:
        state = json.loads((a.study / "progress.json").read_text())
        roots = [
            a.study / f"frames_{s}_{s + 500}"
            for s in state["completed_windows"]
        ]
        if (
            a.mid
            and a.mid not in roots
            and (a.mid / "root_fidelity_review.json").exists()
        ):
            roots.append(a.mid)
        comparisons = []
        rendered = []
        for root in roots:
            info = json.loads((root / "comparison_report.json").read_text())
            rows = json.loads(
                (root / "root_fidelity_review.json").read_text()
            )["clips"]
            for key, variants in rows.items():
                baseline = variants["old_reproduced"]
                for variant, metrics in variants.items():
                    risks = []
                    if metrics["root_p95_deg"] > 5:
                        risks.append(
                            "root_p95_over_provisional_5deg_review_target"
                        )
                    if (
                        metrics["all_frame_arm_direction_p95_deg"]
                        > baseline["all_frame_arm_direction_p95_deg"] + 5
                    ):
                        risks.append("arm_p95_worse_than_old_by_over_5deg")
                    if (
                        metrics["all_frame_leg_direction_p95_deg"]
                        > baseline["all_frame_leg_direction_p95_deg"] + 5
                    ):
                        risks.append("leg_p95_worse_than_old_by_over_5deg")
                    comparisons.append(
                        {
                            "window_start": info["start_frame_50hz"],
                            "motion": key,
                            "variant": variant,
                            "metrics": metrics,
                            "review_risks": risks,
                            "training_approved": False,
                        }
                    )
            if not (root / "visual/complete.json").exists():
                with (root / "render.log").open("a") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.booster_k1_retarget.render_lafan_ablation",
                            "--root",
                            str(root),
                            "--variants",
                            "old_reproduced",
                            "current_full"
                            if a.render_variant == "old_task_weights"
                            else "old_task_weights",
                            a.render_variant,
                            "--keys",
                            *info["keys"],
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        env={**os.environ, "MUJOCO_GL": "egl"},
                    )
                dump(
                    root / "visual/complete.json",
                    {
                        "generated": True,
                        "keys": info["keys"],
                        "training_approved": False,
                        "visual_approval": False,
                    },
                )
            rendered.append(str((root / "visual").resolve()))
        done = state["status"] != "validation_in_progress"
        dump(
            a.study / "evidence_summary.json",
            {
                "status": "evidence_collected_no_release_or_bulk_approval"
                if done
                else "collecting_independent_window_evidence",
                "training_started": False,
                "bulk_repair_started": False,
                "training_approved": False,
                "source_data_overwritten": False,
                "visual_roots": rendered,
                "comparisons": comparisons,
                "limitations": "Diagnostic thresholds are provisional. Global root drift is not hidden by yaw alignment. Visual generation is not visual approval. Per-frame gates preserve prior failure masks; no winner or bulk regeneration is automatically selected.",
            },
        )
        if done or time.monotonic() > deadline:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
