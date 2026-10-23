"""Promote only rechecked IK repairs; preserve every failed original/candidate."""

import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys

from organize_motion_curriculum import partition

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/holomotion_k1_v1"
REPORT = ROOT / "arm_repair_v1"
BATCH = ROOT / "batch_armrepair_v1"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) == value:
            return
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    completed = json.loads((BATCH / "completion.json").read_text())["outcomes"]
    completed += json.loads(
        (ROOT / "batch_rootrepair_supplement_v1/completion.json").read_text()
    )["outcomes"]
    if any(x["status"] != "complete" for x in completed):
        raise RuntimeError("Inspect failed repair groups before finalizing")
    baseline = {
        (x["source"], x["motion"]): x
        for x in json.loads(
            (REPORT / "baseline_limb_audit_v3.json").read_text()
        )["clips"]
    }
    new = {
        (x["source"], x["motion"]): x
        for x in json.loads(
            (REPORT / "repaired_limb_audit_dense.json").read_text()
        )["clips"]
    }
    old_decisions = json.loads(
        (ROOT / "review_20261020/review_decisions.json").read_text()
    )
    existing_decisions = {(x["source"], x["motion"]) for x in old_decisions}
    for identity, row in baseline.items():
        if (
            row["root_orientation_error_deg"]["p95"] > 45
            and identity not in existing_decisions
        ):
            old_decisions.append(
                {
                    "source": identity[0],
                    "motion": identity[1],
                    "quality": "quarantine",
                    "destination": "quarantine",
                    "reasons": ["root_orientation_error_over_45deg"],
                    "evidence": str(REPORT / "baseline_limb_audit_v3.json"),
                }
            )
    approved, results = [], []
    deferred_groups = defaultdict(set)
    for status in completed:
        source = {"cmu_asf": "CMU", "style100": "100STYLE"}[status["source"]]
        raw = Path(status["raw_root"])
        filtered = Path(status["filtered_root"])
        m = json.loads((raw / "manifest.json").read_text())
        checks = {
            x["motion"]: x
            for x in json.loads(
                (filtered / "preparation_report.json").read_text()
            )
        }
        for key, clip in m["clips"].items():
            name = Path(clip["metadata"]["source_path"]).stem
            identity = source, name
            old = baseline[identity]
            candidate = new.get(identity)
            reasons = list(checks[key]["reasons"])
            if candidate:
                reasons.extend(
                    "limb_direction:" + x for x in candidate["flagged_limbs"]
                )
                # Protect the lower-body reference while fixing the arms.
                for limb in (
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                ):
                    if (
                        candidate["limbs"][limb]["p95_deg"]
                        > old["limbs"][limb]["p95_deg"] + 10
                    ):
                        reasons.append("lower_body_regression:" + limb)
                if (
                    candidate["root_orientation_error_deg"]["p95"]
                    > old["root_orientation_error_deg"]["p95"] + 5
                ):
                    reasons.append("root_orientation_regression_over_5deg")
                if candidate["root_orientation_error_deg"]["p95"] > 45:
                    reasons.append("root_orientation_error_over_45deg")
            elif not reasons:
                reasons.append("missing_direction_audit")
            is_control = (
                not old["flagged_limbs"]
                and old["root_orientation_error_deg"]["p95"] <= 45
            )
            result = {
                "source": source,
                "motion": name,
                "is_control": is_control,
                "promoted": not reasons and not is_control,
                "reasons": reasons,
                "original_limb_metrics": old["limbs"],
                "repaired_limb_metrics": candidate["limbs"]
                if candidate
                else None,
                "original_root_error_deg": old["root_orientation_error_deg"],
                "repaired_root_error_deg": candidate[
                    "root_orientation_error_deg"
                ]
                if candidate
                else None,
                "quality": checks[key],
                "robot_root": str(filtered),
                "raw_root": str(raw),
                "robot_motion_key": key,
                "split": status["split"],
            }
            results.append(result)
            if result["promoted"]:
                approved.append(
                    {
                        k: result[k]
                        for k in (
                            "source",
                            "motion",
                            "robot_root",
                            "robot_motion_key",
                            "split",
                        )
                    }
                )
                approved[-1]["evidence"] = str(
                    REPORT / "repair_decisions.json"
                )
            if reasons and candidate:
                deferred_groups[str(filtered)].add(key)
    for source, keys in deferred_groups.items():
        p = Path(source)
        destination = (
            p.parents[2]
            / "direction_or_regression_quarantine"
            / p.parent.parent.name
            / p.parent.name
        )
        existing = destination / "manifest.json"
        if existing.exists():
            previous = json.loads(existing.read_text())
            assert set(previous["motion_keys"]) == keys
            assert previous["partition_source"] == str(p)
        else:
            partition(p, destination, keys)
    accepted = {(x["source"], x["motion"]) for x in approved}
    decisions = [
        d
        for d in old_decisions
        if (d["source"], d["motion"]) not in accepted
        or d["quality"] != "quarantine"
    ]
    save(REPORT / "repair_decisions.json", results)
    save(REPORT / "approved_replacements.json", approved)
    save(REPORT / "remaining_review_decisions.json", decisions)
    output = ROOT / "organized_arm_repaired_20261020_v2"
    subprocess.run(
        [
            sys.executable,
            "tools/booster_k1_retarget/organize_motion_curriculum.py",
            "--output",
            str(output),
            "--extra-batch-root",
            str(ROOT / "batch_recovery_v1"),
            "--approved-replacements",
            str(REPORT / "approved_replacements.json"),
            "--review-decisions",
            str(REPORT / "remaining_review_decisions.json"),
        ],
        cwd=REPO,
        check=True,
    )
    rows = json.loads((output / "all_motions.json").read_text())
    indexed = [
        row
        for p in output.glob("*/index.json")
        for row in json.loads(p.read_text())
    ]
    assert len(indexed) == len(rows)
    assert len({(r["source"], r["motion"]) for r in indexed}) == len(rows)
    references = []
    for path in output.glob("*/*/manifest.json"):
        manifest = json.loads(path.read_text())
        assert set(manifest["motion_keys"]) == set(manifest["clips"])
        for shard in manifest["hdf5_shards"]:
            assert (path.parent / shard["file"]).exists()
        references.extend(
            (manifest["partition_source"], key)
            for key in manifest["motion_keys"]
        )
    assert len(set(references)) == len(references)
    assert set(references) == {
        (r["robot_root"], r.get("robot_motion_key", r["motion"]))
        for r in rows
        if r.get("robot_root")
    }
    lookup = {(r["source"], r["motion"]): r for r in rows}
    for repair in approved:
        row = lookup[repair["source"], repair["motion"]]
        assert row["robot_root"] == repair["robot_root"]
        assert row["split"] == repair["split"]
        assert (row["robot_root"], row["robot_motion_key"]) in references
    pending = [r for r in rows if r["quality"] == "missing_fps_pending"]
    folder = output / "99_pending/missing_fps_sources"
    folder.mkdir(parents=True)
    for row in pending:
        p = Path(row["source_path"]).resolve()
        d = folder / p.parent.name
        d.mkdir(exist_ok=True)
        (d / p.name).symlink_to(p)
        asf = p.parent / (p.parent.name + ".asf")
        if asf.exists() and not (d / asf.name).exists():
            (d / asf.name).symlink_to(asf)
    save(folder / "index.json", pending)
    summary = {
        "root_cause": "Zero-yaw clip-start seed can trap bounded IK arms in a wrong local solution; calibrated first-reference root seed resolves tested cases.",
        "changed": "Only initial free-joint pose; no robot limits, IK weights, PPO rewards or motion timing changed.",
        "rebuilt": len(results),
        "controls": sum(x["is_control"] for x in results),
        "promoted_repairs": len(approved),
        "still_quarantined": sum(
            bool(x["reasons"]) and not x["is_control"] for x in results
        ),
        "control_regressions": [
            x["motion"] for x in results if x["is_control"] and x["reasons"]
        ],
        "remaining_reason_counts": dict(
            Counter(reason for x in results for reason in x["reasons"])
        ),
        "new_direction_audit": "Every 5th frame plus final frame; upper/lower arms and legs",
        "numerical_checks": "Every frame: finite arrays, joint limits/speed, sole-ground geometry, root tilt",
        "limitation": "Kinematic screening and sampled renders, not physical tracking approval",
        "snapshot": str(output),
        "originals_preserved": True,
        "training_started": False,
        "integrity": {
            "unique_source_records": len(rows),
            "unique_robot_references": len(references),
        },
    }
    save(REPORT / "summary.json", summary)
    save(
        ROOT / "active_review_snapshot.json",
        {
            "snapshot": str(output),
            "report": str(REPORT / "summary.json"),
            "training_approved": False,
            "physics_validation_pending": True,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
