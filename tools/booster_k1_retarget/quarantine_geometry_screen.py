"""Create immutable curriculum views with geometry-review holds removed from train."""

import argparse
import json
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def partition(source, target, selected):
    m = json.loads((source / "manifest.json").read_text())
    clips = {k: v for k, v in m["clips"].items() if k in selected}
    if not clips:
        return False
    target.mkdir(parents=True, exist_ok=False)
    for shard in m["hdf5_shards"]:
        path = Path(shard["file"])
        if not path.is_absolute():
            link = target / path
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((source / path).resolve())
    m.update(
        clips=clips,
        motion_keys=list(clips),
        root=str(target.resolve()),
        dataset_id="_".join(target.parts[-3:]),
        partition_source=str(source.resolve()),
        quality_status="geometry_screening_review_not_physics_approval",
    )
    write_json(target / "manifest.json", m)
    write_json(target / "nan_npz_paths.json", [])
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--holds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    holds = json.loads(a.holds.read_text())
    stride = int(holds.get("frame_stride", 5))
    held = {r["motion_key"]: r for r in holds["clips"]}
    a.output.mkdir(parents=True, exist_ok=False)
    roots = {}
    found = set()
    ordinary = {"01_basic", "02_locomotion", "03_dynamic", "04_coordination"}
    for file in sorted(
        (a.snapshot / "candidate_dataset_roots").glob("*.json")
    ):
        dest, split = file.stem.rsplit("_", 1)
        for root in json.loads(file.read_text()):
            root = Path(root)
            m = json.loads((root / "manifest.json").read_text())
            keys = set(m["clips"])
            moved = (
                keys & set(held)
                if dest in ordinary and split == "train"
                else set()
            )
            found |= moved
            for folder, selected in [
                (dest, keys - moved),
                ("90_geometry_review", moved),
            ]:
                if not selected:
                    continue
                # Include original stage to keep moved partitions uniquely named.
                name = (
                    (dest + "_" + root.name)
                    if folder == "90_geometry_review"
                    else root.name
                )
                target = a.output / folder / split / name
                if partition(root, target, selected):
                    roots.setdefault(f"{folder}_{split}", []).append(
                        str(target.resolve())
                    )
    assert found == set(held), (
        "Held entries do not match normal training partitions"
    )
    records = json.loads((a.snapshot / "all_motions.json").read_text())
    train = []
    hold = []
    for row in records:
        key = row.get("robot_motion_key", row["motion"])
        if (
            row["split"] == "train"
            and row.get("curriculum_destination") in ordinary
        ):
            if key in held:
                row["previous_curriculum_destination"] = row[
                    "curriculum_destination"
                ]
                row["curriculum_destination"] = "90_geometry_review"
                row["quality"] = "geometry_review_hold"
                row["geometry_review"] = held[key]
                hold.append(row)
            else:
                row["geometry_review"] = (
                    f"stride{stride}_kinematic_screen_not_visual_or_physics_approval"
                )
                train.append(row)
    assert len(hold) == len(held)
    summary = {
        "previous_snapshot": str(a.snapshot.resolve()),
        "review_holds": str(a.holds.resolve()),
        "original_train_candidates": len(train) + len(hold),
        "retained_train_candidates": len(train),
        "held_train_candidates": len(hold),
        "retained_candidate_hours": sum(
            r["frames"] / 50 / 3600 for r in train
        ),
        "held_candidate_hours": sum(r["frames"] / 50 / 3600 for r in hold),
        "training_approved": False,
        "review_screen_stride": stride,
        "limitations": "Geometry review flags are not confirmed mapping failures. Genuine flights need native source review. No repaired candidate was auto-promoted; original frames and frozen holdouts are preserved.",
    }
    for key, value in roots.items():
        write_json(a.output / "candidate_dataset_roots" / f"{key}.json", value)
    write_json(a.output / "all_motions.json", records)
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
