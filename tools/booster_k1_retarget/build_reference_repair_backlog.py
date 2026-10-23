"""Inventory training-side review pools without touching frozen holdouts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tools.booster_k1_retarget.holosmpl_h5_to_k1_h5 import K1_DOF_NAMES


def build(snapshot, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((snapshot / "all_motions.json").read_text())
    destinations = {
        "90_quarantine",
        "05_scene_and_semantic_review",
        "06_performance_review",
        "99_semantic_pending",
    }
    clips, shards, shard_ids, unresolved = {}, [], {}, []
    cache = {}
    seen = set()
    duplicates = []
    selected = [
        r
        for r in rows
        if r["split"] == "train"
        and r.get("curriculum_destination") in destinations
    ]
    for row in selected:
        root = Path(row["robot_root"])
        candidates = [root]
        if root.name == "robot_filtered":
            candidates.append(root.parent / "robot_raw")
        if "filtered_robot" in root.parts:
            candidates.append(
                Path(
                    str(root).replace("/filtered_robot/", "/candidate_robot/")
                )
            )
        key = row.get("robot_motion_key", row["motion"])
        found = None
        for candidate in candidates:
            manifest_path = candidate / "manifest.json"
            if not manifest_path.is_file():
                continue
            if manifest_path not in cache:
                cache[manifest_path] = json.loads(manifest_path.read_text())
            manifest = cache[manifest_path]
            if key not in manifest["clips"]:
                continue
            if manifest.get("dof_names") != list(K1_DOF_NAMES):
                continue
            found = (candidate, manifest, manifest["clips"][key])
            break
        if found is None:
            unresolved.append(
                {
                    "row": row,
                    "reason": "exact_reference_key_or_verified_dof_header_missing; no guessing",
                }
            )
            continue
        root, manifest, original = found
        entry = copy.deepcopy(original)
        metadata = entry["metadata"]
        if not metadata.get("source_path") or not metadata.get("source_fps"):
            unresolved.append(
                {"row": row, "reason": "source_path_or_fps_missing"}
            )
            continue
        shard_path = (
            root / manifest["hdf5_shards"][entry["shard"]]["file"]
        ).resolve()
        if not shard_path.is_file():
            unresolved.append({"row": row, "reason": "missing_original_shard"})
            continue
        identity = (
            str(shard_path),
            entry["start"],
            entry["length"],
            metadata.get("canonical_path"),
            row["curriculum_destination"],
        )
        if identity in seen:
            duplicates.append(
                {
                    "motion": row["motion"],
                    "reason": "identical immutable reference slice already inventoried",
                }
            )
            continue
        seen.add(identity)
        suffix = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:12]
        new_key = f"{row['source']}__{row['curriculum_destination']}__{key}__variant_{suffix}"
        if new_key in clips:
            raise ValueError("Duplicate backlog identity")
        if str(shard_path) not in shard_ids:
            shard_ids[str(shard_path)] = len(shards)
            shards.append(
                {
                    **manifest["hdf5_shards"][entry["shard"]],
                    "file": str(shard_path),
                }
            )
        entry["shard"] = shard_ids[str(shard_path)]
        metadata.update(
            geometry_review_source=row["source"],
            geometry_review_description=row.get(
                "action_description", row.get("description", row["motion"])
            ),
            geometry_review_parent_root=str(root.resolve()),
            geometry_review_original_destination=row["curriculum_destination"],
            backlog_original_motion_key=key,
            active_split="train",
            training_approved=False,
        )
        clips[new_key] = entry
    manifest = {
        "version": 2,
        "dataset_id": output.name,
        "clips": clips,
        "motion_keys": list(clips),
        "hdf5_shards": shards,
        "dof_names": list(K1_DOF_NAMES),
        "array_names": ["ref_root_pos", "ref_root_rot", "ref_dof_pos"],
        "training_approved": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    report = {
        "selected_review_rows": len(selected),
        "exact_references_found": len(clips),
        "unresolved_rows": unresolved,
        "duplicate_catalog_rows": duplicates,
        "training_approved": False,
        "scope": "Training-side quarantine, performance, semantics and scene review pools. Frozen val/test, holdout reserves, superseded duplicate parents, unknown-FPS unassigned sources and independent v14 are not reclassified. Repaired scenes/unknown semantics remain origin-review, never normal training candidates.",
    }
    (output / "input_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = build(args.snapshot, args.output)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "unresolved_rows"},
            indent=2,
        )
    )
    print("unresolved", len(report["unresolved_rows"]))
