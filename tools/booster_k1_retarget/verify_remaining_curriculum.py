"""Verify final candidate partition lineage and publish its local review pointer."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from localize_and_salvage import dump

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"
STAGES = {"01_basic", "02_locomotion", "03_dynamic", "04_coordination"}


def main():
    final = ROOT / "organized_remaining_repaired_v2"
    rows = json.loads((final / "all_motions.json").read_text())
    previous = json.loads(
        (ROOT / "organized_frame_salvage_v2/all_motions.json").read_text()
    )
    original = {
        (r["source"], r["motion"]): r
        for r in previous
        if "parent_motion" not in r
    }
    assert len(original) == 3813
    ordinary = [r for r in rows if r["curriculum_destination"] in STAGES]
    parent_groups = defaultdict(list)
    for row in rows:
        parent = row.get("parent_motion", row["motion"])
        assert row["split"] == original[row["source"], parent]["split"]
        if row["curriculum_destination"] in STAGES:
            assert not row.get("semantic_hold")
            assert row["quality"] in {
                "numerical_pass_visual_physics_pending",
                "segmented_numerical_pass_visual_physics_pending",
            }
            parent_groups[row["source"], parent].append(row)
    for group in parent_groups.values():
        if len(group) > 1:
            assert all("parent_motion" in r for r in group), (
                "Whole clip duplicated with children"
            )
            spans = sorted(
                (r["source_frame_start"], r["source_frame_end"]) for r in group
            )
            assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    expected = Counter(
        (
            r["curriculum_destination"],
            r["split"],
            str(Path(r["robot_root"]).resolve()),
            r.get("robot_motion_key", r["motion"]),
        )
        for r in rows
        if r.get("robot_root")
    )
    actual = Counter()
    partitions = 0
    for file in final.glob("*/*/partition_*/manifest.json"):
        manifest = json.loads(file.read_text())
        for key in manifest["motion_keys"]:
            clip = manifest["clips"][key]
            assert (
                file.parent / manifest["hdf5_shards"][clip["shard"]]["file"]
            ).is_file()
            actual[
                file.parents[2].name,
                file.parents[1].name,
                str(Path(manifest["partition_source"]).resolve()),
                key,
            ] += 1
        partitions += 1
    assert actual == expected, "Partition index and manifest disagree"
    assert all(n == 1 for n in expected.values())
    moon = next(r for r in rows if r["source"] == "Moonwalk")
    old_moon = original["Moonwalk", moon["motion"]]
    assert moon["robot_root"] == old_moon["robot_root"]
    assert moon["curriculum_destination"] == "99_pending"
    prior_ordinary = {
        (r["source"], r["motion"])
        for r in previous
        if r["curriculum_destination"] in STAGES
    }
    released_semantic = [
        r
        for r in ordinary
        if r.get("semantic_resolution")
        and (r["source"], r["motion"]) not in prior_ordinary
    ]
    report = {
        "partitions_checked": partitions,
        "robot_references_checked": sum(actual.values()),
        "ordinary_candidates": len(ordinary),
        "ordinary_train_candidates": sum(
            r["split"] == "train" for r in ordinary
        ),
        "ordinary_hours_all_splits": sum(r["seconds"] for r in ordinary)
        / 3600,
        "ordinary_train_hours": sum(
            r["seconds"] for r in ordinary if r["split"] == "train"
        )
        / 3600,
        "ordinary_counts_by_stage": dict(
            Counter(r["curriculum_destination"] for r in ordinary)
        ),
        "newly_released_semantic_records": len(released_semantic),
        "split_preserved": True,
        "no_ordinary_parent_child_duplication": True,
        "no_overlapping_ordinary_children": True,
        "moonwalk_reference_unchanged": True,
        "training_started": False,
        "training_approved": False,
        "review_limit": "Kinematic/numerical screening and sampled visual inspection, not full physical validation",
    }
    dump(ROOT / "remaining_repair_v1/final_integrity.json", report)
    dump(
        ROOT / "active_review_snapshot.json",
        {
            "snapshot": str(final),
            "report": str(final / "summary.json"),
            "integrity_report": str(
                ROOT / "remaining_repair_v1/final_integrity.json"
            ),
            "previous_snapshot": str(ROOT / "organized_frame_salvage_v2"),
            "frame_reports": str(ROOT / "frame_salvage_v1/frame_reports"),
            "preparation_status": "remaining_mapping_repair_and_semantic_triage_completed; held_records_excluded",
            "training_approved": False,
            "physics_validation_pending": True,
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
