"""Use latest corrected IK references for unresolved 80 + prior mapping cases."""

import json
from pathlib import Path

from localize_and_salvage import dump

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"


def main():
    current = json.loads(
        (ROOT / "organized_holdout_30min_v1/all_motions.json").read_text()
    )
    originals = {
        (r["source"], r["motion"]): r
        for r in current
        if "parent_motion" not in r
    }
    old = json.loads(
        (ROOT / "arm_repair_v1/remaining_review_decisions.json").read_text()
    )
    latest = json.loads(
        (ROOT / "remaining_repair_v1/repair_decisions.json").read_text()
    )
    targets = {
        (r["source"], r["motion"]) for r in old if r["quality"] == "quarantine"
    }
    targets |= {
        (r["source"], r["motion"]) for r in latest if not r["promoted"]
    }
    mapped = {}
    for batch in (
        "batch_armrepair_v1",
        "batch_rootrepair_supplement_v1",
        "batch_remaining_repair_v1",
    ):
        for path in (ROOT / batch).glob("*/*/status.json"):
            status = json.loads(path.read_text())
            if status["status"] != "complete":
                continue
            root = Path(status["raw_root"])
            m = json.loads((root / "manifest.json").read_text())
            source = {"cmu_asf": "CMU", "style100": "100STYLE"}[
                status["source"]
            ]
            report = {
                r["motion"]: r
                for r in json.loads(
                    (
                        Path(status["filtered_root"])
                        / "preparation_report.json"
                    ).read_text()
                )
            }
            for key, clip in m["clips"].items():
                identity = source, Path(clip["metadata"]["source_path"]).stem
                if identity not in targets:
                    continue
                row = dict(originals[identity])
                if row["quality"] != "quarantine":
                    continue
                row.update(
                    robot_root=str(root),
                    robot_motion_key=key,
                    frames=clip["length"],
                    reasons=report[key]["reasons"],
                    quality="quarantine",
                    repair_source_status=str(path),
                )
                mapped[identity] = row
    assert len(mapped) == len(targets), (len(mapped), len(targets))
    output = ROOT / "targeted_frame_repair_v1"
    output.mkdir(exist_ok=False)
    dump(output / "input/all_motions.json", list(mapped.values()))
    dump(
        output / "selection_summary.json",
        {
            "unique_parent_clips": len(mapped),
            "latest_80": sum(not r["promoted"] for r in latest),
            "previous_mapping_records": len(targets)
            - sum(not r["promoted"] for r in latest),
            "uses_latest_corrected_IK_raw": True,
            "training_started": False,
        },
    )
    print((output / "selection_summary.json").read_text())


if __name__ == "__main__":
    main()
