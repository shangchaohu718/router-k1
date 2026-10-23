"""Build a reproducible two-per-category training subset for policy replay."""

import argparse
import copy
import json
import random
import re
from pathlib import Path

CATEGORIES = {
    "basic_pose_or_walk": "基础姿态与走路",
    "neutral_walk_or_idle": "普通走路与站立",
    "running": "跑步",
    "stylized_locomotion_provisional": "风格化移动",
    "dance_or_sliding_coordination": "舞蹈与滑步",
    "named_dance": "舞蹈",
    "dynamic_motion": "动态动作",
    "locomotion_transition_or_balance": "移动与平衡",
    "whole_body_or_contact_coordination": "全身协调",
    "gesture_or_upper_body_expression": "上肢手势",
    "single_support_or_rotation": "单脚支撑与转体",
    "stylized_steps_or_postural_transition": "风格步态与姿态切换",
    "rotation_label_without_direction": "转体",
    "direction_change_or_transition": "变向与过渡",
}


def prepare(snapshot: Path, output: Path, max_frames: int = 1000) -> dict:
    snapshot = snapshot.resolve()
    rows = json.loads((snapshot / "all_motions.json").read_text())
    rows = [
        r
        for r in rows
        if r["split"] == "train"
        and r.get("curriculum_destination")
        in {"01_basic", "02_locomotion", "03_dynamic", "04_coordination"}
        and r["frames"] >= 150
    ]
    indexed = {}
    for path in sorted(
        (snapshot / "candidate_dataset_roots").glob("0[1-4]*_train.json")
    ):
        for root in json.loads(path.read_text()):
            manifest = json.loads((Path(root) / "manifest.json").read_text())
            for key, clip in manifest["clips"].items():
                shard = manifest["hdf5_shards"][int(clip["shard"])]
                indexed[key] = (clip, (Path(root) / shard["file"]).resolve())
    rng = random.Random(20261021)
    selected = []
    shards = []
    shard_ids = {}
    clips = {}
    for category, label in CATEGORIES.items():
        candidates = sorted(
            [r for r in rows if r.get("difficulty_reason") == category],
            key=lambda r: (r["source"], r["motion"]),
        )
        rng.shuffle(candidates)
        parents = set()
        sources = set()
        chosen = []
        # Prefer different sources and never choose two slices of one capture.
        for diverse in (True, False):
            for row in candidates:
                parent = (
                    row["source"],
                    re.sub(
                        r"__frames_.*",
                        "",
                        row.get("parent_motion", row["motion"]),
                    ),
                )
                if parent in parents or (diverse and row["source"] in sources):
                    continue
                chosen.append(row)
                parents.add(parent)
                sources.add(row["source"])
                if len(chosen) == 2:
                    break
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise ValueError(
                f"Category lacks two distinct captures: {category}"
            )
        for row in chosen:
            key = row.get("robot_motion_key", row["motion"])
            clip, shard_path = indexed[key]
            if str(shard_path) not in shard_ids:
                shard_ids[str(shard_path)] = len(shards)
                shards.append({"file": str(shard_path)})
            entry = copy.deepcopy(clip)
            entry["shard"] = shard_ids[str(shard_path)]
            entry["length"] = min(int(entry["length"]), max_frames)
            entry["metadata"]["comparison_category"] = label
            clips[key] = entry
            selected.append(
                {
                    "motion_key": key,
                    "category": category,
                    "category_label": label,
                    "source": row["source"],
                    "description": row.get("action_description"),
                    "frames": entry["length"],
                    "source_frames": clip["length"],
                }
            )
    manifest = {
        "version": 2,
        "dataset_id": "k1_curriculum_comparison_20261021",
        "root": str(output.resolve()),
        "hdf5_shards": shards,
        "clips": clips,
    }
    selection = {
        "snapshot": str(snapshot),
        "seed": 20261021,
        "max_frames": max_frames,
        "selection_policy": "two distinct captures per existing semantic category, source diversity preferred; first <=20 seconds",
        "clips": selected,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("manifest.json", manifest),
        ("comparison_selection.json", selection),
    ]:
        path = output / name
        text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        if path.exists() and path.read_text() != text:
            raise FileExistsError(
                f"Refusing to replace a different comparison selection: {path}"
            )
        path.write_text(text)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.snapshot, args.output)
    print(f"Prepared {len(result['clips'])} clips in {args.output}")
