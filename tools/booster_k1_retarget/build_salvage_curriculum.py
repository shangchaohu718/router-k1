"""Build split-preserving candidate curricula, with explicit semantic holds."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import h5py
import numpy as np

from organize_motion_curriculum import (
    difficulty,
    kinematic_difficulty,
    partition,
)
from localize_and_salvage import dump

REPO = Path(__file__).resolve().parents[2]


def semantic_hold(label):
    if re.search(r"calibrat|\bt.?pose\b|\brange of motion\b", label, re.I):
        return "calibration_not_training_motion"
    if re.search(
        r"wrestl|motorcycle|bicycle|ladder|stair|chair|uneven|terrain|obstacle|"
        r"\bsit(?:ting)?\b|climb|crawl|vault|two subjects|2 subjects|"
        r"partner|carry|pick.*box|lift.*(?:person|box)|\blbs?\b|suitcase|dribble|"
        r"sweep|wash windows|cleaning|^clean$|fishing",
        label,
        re.I,
    ):
        return "scene_or_object_interaction_review"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--salvage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction-audit", type=Path)
    parser.add_argument("--semantic-overrides", type=Path)
    args = parser.parse_args()
    overrides = (
        json.loads(args.semantic_overrides.read_text())
        if args.semantic_overrides
        else {}
    )
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.snapshot / "all_motions.json").read_text())
    recovered = json.loads((args.salvage / "segments.json").read_text())
    direction_rows = (
        json.loads(args.direction_audit.read_text())["clips"]
        if args.direction_audit
        else []
    )
    directions = {
        (r["robot_root"], r["robot_motion_key"]): r for r in direction_rows
    }
    visual_path = args.salvage / "visual_review.json"
    visual = {
        (r["source"], r["motion"]): r
        for r in (
            json.loads(visual_path.read_text())["clips"]
            if visual_path.exists()
            else []
        )
    }
    labels = json.loads(
        (
            REPO / "data/holomotion_k1_v1/catalog/cmu_motion_metadata.json"
        ).read_text()
    )["clips"]
    groups = defaultdict(set)
    rows = []
    stage_names = {
        "L1": "01_basic",
        "L2": "02_locomotion",
        "L3": "03_dynamic",
        "L4": "04_coordination",
    }
    for i, original_row in enumerate(original + recovered):
        row = dict(original_row)
        name = row.get("parent_motion", row["motion"])
        label = (
            labels.get(name, {}).get("description", name)
            if row["source"] == "CMU"
            else name
        )
        row["action_description"] = label
        semantic_override = overrides.get(row["source"] + ":" + name)
        if semantic_override:
            row["semantic_resolution"] = semantic_override
        hold = semantic_hold(label)
        visual_row = visual.get((row["source"], row["motion"]))
        if visual_row and visual_row.get("hold"):
            hold = visual_row["hold"]
        candidate = row["quality"] in {
            "numerical_pass_visual_physics_pending",
            "segmented_numerical_pass_visual_physics_pending",
        }
        destination = (
            "90_quarantine" if row["quality"] == "quarantine" else "99_pending"
        )
        root_text = row.get("robot_root")
        direction = directions.get(
            (root_text, row.get("robot_motion_key", row["motion"]))
        )
        direction_failed = bool(
            direction
            and (
                direction["flagged_limbs"]
                or direction["root_orientation_error_deg"]["p95"] > 45
            )
        )
        if direction:
            row["direction_audit"] = direction
        if direction_failed:
            candidate = False
            destination = "90_quarantine"
            row["quality"] = "recovered_segment_direction_review_failed"
        if candidate:
            root = Path(root_text)
            m = json.loads((root / "manifest.json").read_text())
            key = row.get("robot_motion_key", row["motion"])
            clip = m["clips"][key]
            s, n = clip["start"], clip["length"]
            # Full array read verifies every candidate is actually accessible.
            with h5py.File(
                root / m["hdf5_shards"][clip["shard"]]["file"], "r"
            ) as f:
                for field, width in (
                    ("ref_root_pos", 3),
                    ("ref_root_rot", 4),
                    ("ref_dof_pos", 22),
                ):
                    value = f[field][s : s + n]
                    if (
                        value.shape != (n, width)
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError(
                            f"Invalid candidate {root}:{key}:{field}"
                        )
            kin_level, metrics = kinematic_difficulty(root, m, clip)
            style, _, movement = name.rpartition("_")
            semantic_level, reason = difficulty(
                label, row["source"], style, movement
            )
            if semantic_override:
                semantic_level = semantic_override["difficulty"] or "pending"
                reason = semantic_override["category"]
                hold = hold or semantic_override["hold"]
            if semantic_level == "L5":
                hold = hold or "special_scene_or_floor_contact"
            if semantic_level == "pending":
                hold = hold or "unresolved_action_description"
            level = (
                max(kin_level, semantic_level)
                if semantic_level in stage_names
                else kin_level
            )
            row.update(
                difficulty=level,
                difficulty_reason=reason,
                kinematic_metrics=metrics,
                frames=n,
                seconds=n / float(clip["metadata"]["motion_fps"]),
                robot_motion_key=key,
            )
            destination = stage_names[level]
            if hold:
                destination = "05_scene_and_semantic_review"
                if hold in {
                    "missing_trial_semantics",
                    "manual_semantic_review",
                }:
                    destination = "99_semantic_pending"
                elif hold == "mixed_performance_requires_clip_review":
                    destination = "06_performance_review"
                elif hold == "source_explicit_bad_motion":
                    destination = "90_quarantine"
            elif row["split"] not in {"train", "val", "test"}:
                destination = "99_pending"
            row["semantic_hold"] = hold
            row["readback_validated"] = True
        elif row["quality"] == "special_scene_deferred":
            destination = "05_scene_and_semantic_review"
        row["curriculum_destination"] = destination
        row["difficulty_basis"] = (
            "semantic_label_and_kinematic_thresholds_not_policy_success"
        )
        row["training_approved"] = False
        if root_text:
            groups[
                (destination, row["split"], str(Path(root_text).resolve()))
            ].add(row.get("robot_motion_key", row["motion"]))
        rows.append(row)
        if i % 250 == 0:
            print(f"classified_and_read={i + 1}", flush=True)
    roots = defaultdict(list)
    for i, ((destination, split, root), keys) in enumerate(groups.items()):
        dest = args.output / destination / split / f"partition_{i:04d}"
        partition(Path(root), dest, keys)
        roots[(destination, split)].append(str(dest.resolve()))
    for destination in sorted({r["curriculum_destination"] for r in rows}):
        subset = [
            r for r in rows if r["curriculum_destination"] == destination
        ]
        dump(args.output / destination / "index.json", subset)
    for (stage, split), paths in roots.items():
        dump(
            args.output / "candidate_dataset_roots" / f"{stage}_{split}.json",
            paths,
        )
    summary = {
        "original_source_records": len(original),
        "derived_segment_records": len(recovered),
        "counts": dict(Counter(r["curriculum_destination"] for r in rows)),
        "stage_split_counts": dict(
            Counter(
                r["curriculum_destination"] + "/" + r["split"] for r in rows
            )
        ),
        "candidate_hours_by_stage_split": {
            stage + "/" + split: sum(
                r.get("seconds", 0)
                for r in rows
                if r["curriculum_destination"] == stage and r["split"] == split
            )
            / 3600
            for stage, split in roots
            if stage in stage_names.values()
        },
        "semantic_holds": dict(
            Counter(r["semantic_hold"] for r in rows if r.get("semantic_hold"))
        ),
        "training_started": False,
        "training_approved": False,
        "limitations": [
            "Difficulty is an explicit semantic/kinematic curriculum, not measured policy difficulty.",
            "Derived segments pass full numerical/FK checks but human-to-robot directional and visual review remain pending.",
            "Unreviewed scene/calibration/unknown-label clips excluded from ordinary stages.",
            "Source originals and parent split preserved; quarantined parents never included with recovered children in ordinary stages.",
        ],
        "parent_snapshot": str(args.snapshot.resolve()),
        "frame_audit": str(args.salvage.resolve()),
    }
    dump(args.output / "all_motions.json", rows)
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
