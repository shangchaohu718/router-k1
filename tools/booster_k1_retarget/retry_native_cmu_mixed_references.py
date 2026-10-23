"""Native-foot trials for CMU zero recoveries with mixed ankle failures.

Retain all source/geometry gates; do not join fragments or promote origins.
Ankle-only cases stay with the separate v5 worker.
"""

import json
from pathlib import Path

from tools.booster_k1_retarget import (
    retry_native_cmu_zero_references as native,
)


def eligible_row(row):
    reasons = set(row["reason_intervals"])
    return (
        row.get("source_error") is None
        and row["frames"] >= 50
        and "ankle_limit_saturation_for_0p5s" in reasons
        and len(reasons) > 1
    )


def targets(run):
    selected = []
    for file in sorted(run.glob("batch_*/status.json")):
        status = json.loads(file.read_text())
        if status["status"] != "complete":
            continue
        report = Path(status["report"])
        row = json.loads(report.read_text())["clips"][0]
        if not eligible_row(row):
            continue
        score, _ = native.workflow.score_review(report)
        if score[0]:
            continue
        inp = file.parent / "input"
        if not (inp / "manifest.json").exists():
            continue
        manifest = json.loads((inp / "manifest.json").read_text())
        key = status["motion_keys"][0]
        source = manifest["clips"][key]["metadata"]["source_path"]
        if Path(source).suffix.lower() == ".amc":
            selected.append((key, inp, False))
    return selected


def main():
    native.targets = targets
    native.SCOPE = (
        "Completed v4 zero-recovery CMU references with ankle saturation "
        "and other flagged classes, at least 50 source frames; all review "
        "gates unchanged. Ankle-only cases remain in v5."
    )
    native.main()


if __name__ == "__main__":
    main()
