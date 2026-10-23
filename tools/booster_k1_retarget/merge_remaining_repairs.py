"""Promote only numerical+direction-pass new references; remove overlapping children."""

import argparse
import json
from pathlib import Path
import shutil

from localize_and_salvage import dump

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    audits = json.loads(a.audit.read_text())["clips"]
    selected = json.loads(
        (ROOT / "remaining_repair_v1/selection.json").read_text()
    )
    original = json.loads(
        (
            ROOT / "organized_arm_repaired_20261020_v2/all_motions.json"
        ).read_text()
    )
    lookup = {(r["source"], r["motion"]): r for r in original}
    audit_lookup = {(r["source"], r["motion"]): r for r in audits}
    numerical = {}
    for path in (ROOT / "batch_remaining_repair_v1").glob("*/*/status.json"):
        status = json.loads(path.read_text())
        if status["status"] != "complete":
            raise RuntimeError(f"Unfinished repair: {path}")
        raw = json.loads(
            (Path(status["raw_root"]) / "manifest.json").read_text()
        )
        report = json.loads(
            (
                Path(status["filtered_root"]) / "preparation_report.json"
            ).read_text()
        )
        source = {"cmu_asf": "CMU", "style100": "100STYLE"}[status["source"]]
        for item in report:
            name = Path(
                raw["clips"][item["motion"]]["metadata"]["source_path"]
            ).stem
            numerical[(source, name)] = item
    promoted, decisions = set(), []
    for chosen in selected:
        identity = chosen["source"], chosen["motion"]
        audit = audit_lookup.get(identity)
        reasons = []
        if not audit:
            reasons = numerical.get(identity, {}).get("reasons") or [
                "rebuild_missing_direction_audit"
            ]
        else:
            reasons += ["limb_direction:" + x for x in audit["flagged_limbs"]]
            if audit["root_orientation_error_deg"]["p95"] > 45:
                reasons.append("root_direction_p95_over_45deg")
        decision = {
            "source": identity[0],
            "motion": identity[1],
            "promoted": not reasons,
            "reasons": reasons,
            "audit": audit,
            "numerical_audit": numerical.get(identity),
        }
        if not reasons:
            row = lookup[identity]
            root = Path(audit["robot_root"])
            manifest = json.loads((root / "manifest.json").read_text())
            clip = manifest["clips"][audit["robot_motion_key"]]
            status = json.loads((root.parent / "status.json").read_text())
            assert status["split"] == row["split"]
            decision["previous_robot_root"] = row["robot_root"]
            for field in (
                "review_reasons",
                "review_evidence",
                "review_destination",
            ):
                row.pop(field, None)
            row.update(
                robot_root=str(root),
                robot_motion_key=audit["robot_motion_key"],
                frames=clip["length"],
                reasons=[],
                quality="numerical_pass_visual_physics_pending",
                repair_audit=audit,
            )
            promoted.add(identity)
        decisions.append(decision)
    output = ROOT / "remaining_repair_v1/merged_inputs"
    output.mkdir(exist_ok=False)
    dump(output / "original/all_motions.json", original)
    segments = json.loads(
        (ROOT / "frame_salvage_v1/segments.json").read_text()
    )
    kept = [
        r
        for r in segments
        if (r["source"], r["parent_motion"]) not in promoted
    ]
    dump(output / "salvage/segments.json", kept)
    shutil.copyfile(
        ROOT / "frame_salvage_v1/visual_review.json",
        output / "salvage/visual_review.json",
    )
    dump(ROOT / "remaining_repair_v1/repair_decisions.json", decisions)
    dump(
        ROOT / "remaining_repair_v1/merge_summary.json",
        {
            "attempted_parent_clips": len(selected),
            "promoted_full_clips": len(promoted),
            "still_held_parent_clips": len(selected) - len(promoted),
            "superseded_child_segments": len(segments) - len(kept),
            "remaining_child_segments": len(kept),
            "split_preserved": True,
            "originals_preserved": True,
            "training_started": False,
            "direction_audit": str(a.audit.resolve()),
        },
    )
    print((ROOT / "remaining_repair_v1/merge_summary.json").read_text())


if __name__ == "__main__":
    main()
