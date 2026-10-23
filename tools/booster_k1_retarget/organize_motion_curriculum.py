"""Separate provisional difficulty, source splits, and data-quality quarantine.

Each robot partition is a manifest view of immutable source shards; no frame
data is duplicated. Raw unconverted motions receive searchable index records.
Difficulty is provisional until K1 metrics and tracking evaluations refine it.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/holomotion_k1_v1"
LEVELS = {
    "L1": "01_basic",
    "L2": "02_locomotion",
    "L3": "03_dynamic",
    "L4": "04_coordination",
    "L5": "05_special_scene",
    "quarantine": "90_quarantine",
    "pending": "99_pending",
}


def kinematic_difficulty(root, manifest, clip):
    start, length = clip["start"], clip["length"]
    with h5py.File(
        root / manifest["hdf5_shards"][clip["shard"]]["file"], "r"
    ) as f:
        pos = f["ref_root_pos"][start : start + length]
        quat = f["ref_root_rot"][start : start + length]
        q = f["ref_dof_pos"][start : start + length]
    fps = float(clip["metadata"]["motion_fps"])
    speed = float(
        np.percentile(
            np.linalg.norm(np.diff(pos[:, :2], axis=0) * fps, axis=1), 95
        )
    )
    joint_speed = float(
        np.percentile(np.max(np.abs(np.diff(q, axis=0) * fps), axis=1), 95)
    )
    rotation = Rotation.from_quat(quat)
    tilt = float(
        np.percentile(
            np.degrees(
                np.arccos(np.clip(rotation.as_matrix()[:, 2, 2], -1, 1))
            ),
            95,
        )
    )
    turn = float(
        np.percentile(
            (rotation[:-1].inv() * rotation[1:]).magnitude() * fps, 95
        )
    )
    height_range = float(
        np.percentile(pos[:, 2], 95) - np.percentile(pos[:, 2], 5)
    )
    if tilt > 45 or joint_speed > 8 or turn > 3:
        level = "L4"
    elif speed > 1.5 or joint_speed > 4 or height_range > 0.15:
        level = "L3"
    elif speed > 0.7 or turn > 1 or height_range > 0.08:
        level = "L2"
    else:
        level = "L1"
    return level, {
        "root_speed_p95_m_s": speed,
        "joint_speed_p95_rad_s": joint_speed,
        "root_tilt_p95_deg": tilt,
        "root_angular_speed_p95_rad_s": turn,
        "root_height_p95_p05_m": height_range,
    }


def difficulty(text, source="", style="", movement=""):
    t = text.lower()
    if re.search(
        r"stair|obstacle|ladder|climb|crawl|ground|\bfall(?:s|ing)?\b|get.?up|\bsit(?:ting)?\b|chair|vault|uneven|terrain",
        t,
    ):
        return "L5", "special_scene_or_floor_contact"
    if source == "AIST++" or "dance" in t or "moonwalk" in t:
        return "L4", "dance_or_sliding_coordination"
    if re.search(r"one leg|one foot|single.?leg|balance|spin|pirouette", t):
        return "L4", "single_support_or_rotation"
    if re.search(
        r"fight|punch|kick|boxing|martial|cartwheel|acrobat|sport", t
    ):
        return "L4", "whole_body_or_contact_coordination"
    if source == "100STYLE":
        if style not in ("Neutral", "ArmsBySide", "Akimbo"):
            return (
                "L4" if movement in ("BR", "FR", "SR") else "L3"
            ), "stylized_locomotion_provisional"
        if movement in ("BR", "FR", "SR"):
            return "L3", "running"
        if movement in ("BW", "SW") or movement.startswith("TR"):
            return "L2", "direction_change_or_transition"
        return "L1", "neutral_walk_or_idle"
    if re.search(r"run|sprint|jog|jump|hop|leap", t):
        return "L3", "dynamic_motion"
    if re.search(
        r"brisk|fast|backward|side|turn|push|stumble|transition|multiple", t
    ):
        return "L2", "locomotion_transition_or_balance"
    if re.search(r"walk|stand|idle|aiming", t):
        return "L1", "basic_pose_or_walk"
    return "pending", "insufficient_action_label"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def quality_bucket(reasons, level):
    scene_reasons = {
        "large_tilt_defer_to_complex_pool",
        "floor_alignment_ambiguous",
        "insufficient_ground_contact_review",
    }
    if reasons and level == "L5" and set(reasons).issubset(scene_reasons):
        return "special_scene_deferred"
    return "quarantine" if reasons else "numerical_pass_visual_physics_pending"


def partition(source, target, names):
    if (target / "manifest.json").exists():
        raise FileExistsError(f"Duplicate partition destination: {target}")
    m = json.loads((source / "manifest.json").read_text())
    keys = [k for k in m["motion_keys"] if k in names]
    if not keys:
        return
    target.mkdir(parents=True, exist_ok=True)
    for shard in m["hdf5_shards"]:
        link = target / shard["file"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to((source / shard["file"]).resolve())
    save(
        target / "manifest.json",
        {
            **m,
            "dataset_id": target.parent.name + "_" + target.name,
            "root": str(target.resolve()),
            "motion_keys": keys,
            "clips": {k: m["clips"][k] for k in keys},
            "partition_source": str(source),
            "quality_status": "provisional_partition_not_physics_certified",
        },
    )
    save(target / "nan_npz_paths.json", [])


def repair_partition_name(source, source_name, split):
    """Independent batches may repair different clips of the same subject."""
    source = Path(source)
    return f"repaired_{source_name}_{source.parents[2].name}_{source.parent.name}_{split}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--extra-batch-root", type=Path, action="append", default=[]
    )
    p.add_argument("--review-decisions", type=Path)
    p.add_argument("--approved-replacements", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = []
    groups = defaultdict(set)
    for split, suffix in [
        ("train", "train_subject1_4"),
        ("val", "val_subject5"),
    ]:
        source = REPO / ("data/lafan1_booster_k1_gmr_v2_h5_" + suffix)
        m = json.loads((source / "manifest.json").read_text())
        filtered = ROOT / "filtered_robot" / ("lafan_release_" + split)
        decisions = {}
        if (filtered / "preparation_report.json").exists():
            decisions = {
                r["motion"]: r
                for r in json.loads(
                    (filtered / "preparation_report.json").read_text()
                )
            }
        for key in m["motion_keys"]:
            level, reason = difficulty(key)
            row = {
                "source": "LaFAN1",
                "split": split,
                "motion": key,
                "difficulty": level,
                "difficulty_reason": reason,
                "quality": "existing_k1_dataset_reaudit_pending",
                "robot_root": str(source),
                "frames": m["clips"][key]["length"],
            }
            selected_source = source
            if key in decisions:
                reasons = decisions[key].get("reasons", ["invalid_arrays"])
                row.update(
                    quality=quality_bucket(reasons, level), reasons=reasons
                )
                if not reasons:
                    selected_source = filtered
                    motion_level, metrics = kinematic_difficulty(
                        source, m, m["clips"][key]
                    )
                    row["difficulty"] = max(level, motion_level)
                    row["kinematic_metrics"] = metrics
                    row["difficulty_basis"] = (
                        "semantic_and_kinematic_provisional_not_policy_score"
                    )
                row["robot_root"] = str(selected_source)
            rows.append(row)
            bucket = (
                "quarantine"
                if row["quality"] == "quarantine"
                else row["difficulty"]
            )
            suffix = (
                "_deferred"
                if row["quality"] == "special_scene_deferred"
                else ""
            )
            groups[
                (bucket, "lafan_" + split + suffix, str(selected_source))
            ].add(key)
    for split in ("train", "val", "test"):
        raw = ROOT / "candidate_robot" / ("aistpp_" + split)
        filtered = ROOT / "filtered_robot" / ("aistpp_" + split)
        m = json.loads((raw / "manifest.json").read_text())
        decisions = {
            r["motion"]: r
            for r in json.loads(
                (filtered / "preparation_report.json").read_text()
            )
        }
        for key in m["motion_keys"]:
            reasons = list(decisions[key].get("reasons", ["invalid_arrays"]))
            if key == "gJB_sBM_cAll_d07_mJB2_ch01":
                reasons.append("post_audit_insufficient_ground_contact")
            quality = (
                "quarantine"
                if reasons
                else "numerical_pass_visual_physics_pending"
            )
            level = "quarantine" if reasons else "L4"
            source = raw if reasons else filtered
            rows.append(
                {
                    "source": "AIST++",
                    "split": split,
                    "motion": key,
                    "difficulty": "L4",
                    "quality": quality,
                    "reasons": reasons,
                    "robot_root": str(source),
                    "frames": m["clips"][key]["length"],
                }
            )
            groups[(level, "aistpp_" + split, str(source))].add(key)
    source = ROOT / "filtered_robot/moonwalk_v14"
    m = json.loads((source / "manifest.json").read_text())
    for key in m["motion_keys"]:
        rows.append(
            {
                "source": "Moonwalk",
                "split": "train",
                "motion": key,
                "difficulty": "L4",
                "quality": "numerical_pass_physics_pending",
                "synthetic_repeat": True,
                "robot_root": str(source),
            }
        )
        groups[("L4", "moonwalk_train", str(source))].add(key)
    style = json.loads(
        (ROOT / "catalog/100style_motion_catalog.json").read_text()
    )
    for item in style["files"]:
        level, reason = difficulty(
            "",
            source="100STYLE",
            style=item["style"],
            movement=item["movement"],
        )
        rows.append(
            {
                "source": "100STYLE",
                "motion": Path(item["path"]).stem,
                "source_path": item["path"],
                "difficulty": level,
                "difficulty_reason": reason,
                "quality": "raw_adapter_validation_pending",
                "split": "unassigned",
                "problems": item["problems"],
            }
        )
    cmu_meta = json.loads(
        (ROOT / "catalog/cmu_motion_metadata.json").read_text()
    )["clips"]
    for path in sorted((ROOT / "raw/cmu").rglob("*.amc")):
        meta = cmu_meta.get(path.stem, {})
        level, reason = difficulty(meta.get("description", ""))
        rows.append(
            {
                "source": "CMU",
                "motion": path.stem,
                "source_path": str(path),
                "description": meta.get("description"),
                "fps": meta.get("fps"),
                "difficulty": level,
                "difficulty_reason": reason,
                "split": "unassigned",
                "quality": "raw_adapter_validation_pending"
                if meta.get("fps")
                else "missing_fps_pending",
            }
        )
    lookup = {(r["source"], r["motion"]): r for r in rows}
    status_paths = [
        path
        for batch in [ROOT / "batch_adaptation_v1", *a.extra_batch_root]
        for path in sorted(batch.glob("*/*/status.json"))
    ]
    seen_robot_sources = set()
    for status_path in status_paths:
        status = json.loads(status_path.read_text())
        if status["status"] != "complete":
            continue
        source_name = {"style100": "100STYLE", "cmu_asf": "CMU"}[
            status["source"]
        ]
        raw = Path(status["raw_root"])
        filtered = Path(status["filtered_root"])
        manifest = json.loads((raw / "manifest.json").read_text())
        decisions = {
            r["motion"]: r
            for r in json.loads(
                (filtered / "preparation_report.json").read_text()
            )
        }
        for key, clip in manifest["clips"].items():
            original = Path(clip["metadata"]["source_path"]).stem
            identity = (source_name, original)
            if identity in seen_robot_sources:
                raise ValueError(f"Duplicate completed source: {identity}")
            seen_robot_sources.add(identity)
            row = lookup[(source_name, original)]
            reasons = decisions[key].get("reasons", ["invalid_arrays"])
            source = raw if reasons else filtered
            row.update(
                quality=quality_bucket(reasons, row["difficulty"]),
                split=status["split"],
                reasons=reasons,
                robot_root=str(source),
                robot_motion_key=key,
            )
            if not reasons:
                motion_level, metrics = kinematic_difficulty(
                    raw, manifest, clip
                )
                semantic_level = row["difficulty"]
                row["difficulty"] = (
                    motion_level
                    if semantic_level == "pending"
                    else max(semantic_level, motion_level)
                )
                row["kinematic_metrics"] = metrics
                row["difficulty_basis"] = (
                    "semantic_and_kinematic_provisional_not_policy_score"
                )
            level = (
                "quarantine"
                if row["quality"] == "quarantine"
                else row["difficulty"]
            )
            name = (
                status["source"]
                + "_"
                + status["group"]
                + "_"
                + status["split"]
            )
            if row["quality"] == "special_scene_deferred":
                name += "_deferred"
            groups[(level, name, str(source))].add(key)
    if a.approved_replacements:
        for replacement in json.loads(a.approved_replacements.read_text()):
            row = lookup[(replacement["source"], replacement["motion"])]
            old_source = row["robot_root"]
            old_key = row.get("robot_motion_key", row["motion"])
            for (_, _, group_source), keys in groups.items():
                if old_source == group_source:
                    keys.discard(old_key)
            source = Path(replacement["robot_root"])
            key = replacement["robot_motion_key"]
            manifest = json.loads((source / "manifest.json").read_text())
            if row["split"] != replacement["split"]:
                raise ValueError("Repair cannot change train/val/test split")
            clip = manifest["clips"][key]
            level, metrics = kinematic_difficulty(source, manifest, clip)
            # Preserve existing style classifications; physical metrics may upgrade.
            level = (
                max(row["difficulty"], level)
                if row["difficulty"] != "pending"
                else level
            )
            row.update(
                robot_root=str(source),
                robot_motion_key=key,
                difficulty=level,
                quality="numerical_pass_visual_physics_pending",
                reasons=[],
                kinematic_metrics=metrics,
                replaced_robot_root=old_source,
                replacement_evidence=replacement["evidence"],
            )
            name = repair_partition_name(source, row["source"], row["split"])
            groups[(level, name, str(source))].add(key)
    if a.review_decisions:
        decisions = json.loads(a.review_decisions.read_text())
        for decision in decisions:
            row = lookup[(decision["source"], decision["motion"])]
            row.update(
                quality=decision["quality"],
                review_destination=decision["destination"],
                review_reasons=decision["reasons"],
                review_evidence=decision.get("evidence", ""),
            )
            key = row.get("robot_motion_key", row["motion"])
            source = row.get("robot_root")
            for (_, _, group_source), keys in groups.items():
                if source == group_source:
                    keys.discard(key)
            if source:
                level = decision["destination"]
                name = f"review_{row['source']}_{row['split']}"
                # Include source group to avoid colliding partition destinations.
                name += "_" + Path(source).parent.name
                groups[(level, name, source)].add(key)
    for (level, name, source), keys in groups.items():
        partition(Path(source), a.output / LEVELS[level] / name, keys)
    for level, folder in LEVELS.items():
        selected = [
            r
            for r in rows
            if (
                r["quality"] == "quarantine"
                if level == "quarantine"
                else (
                    (
                        r.get(
                            "review_destination",
                            "pending"
                            if r["quality"].endswith("_pending")
                            and not r.get("robot_root")
                            else r["difficulty"],
                        )
                        == level
                    )
                    and r["quality"] != "quarantine"
                )
            )
        ]
        save(a.output / folder / "index.json", selected)
    save(a.output / "all_motions.json", rows)
    report = {
        "difficulty_is_provisional": True,
        "total_records": len(rows),
        "difficulty_counts_excluding_quarantine": dict(
            Counter(
                r["difficulty"] for r in rows if r["quality"] != "quarantine"
            )
        ),
        "quality_counts": dict(Counter(r["quality"] for r in rows)),
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "preserves_existing_train_val_test_splits": True,
        "unassigned_sources_must_be_split_before_training": True,
        "special_scene_is_not_bad_data": True,
        "review_decisions": str(a.review_decisions)
        if a.review_decisions
        else None,
        "approved_replacements": str(a.approved_replacements)
        if a.approved_replacements
        else None,
        "note": "Robot subsets use manifest views with symlinked immutable shards. Do not delete original datasets. Difficulty folders are not automatic training authorization.",
    }
    save(a.output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
