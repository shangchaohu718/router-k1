"""Reduce active holdouts without moving held-out captures into training."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from localize_and_salvage import dump
from organize_motion_curriculum import partition

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"
STAGES = {"01_basic", "02_locomotion", "03_dynamic", "04_coordination"}


def identity(row):
    return row["source"], row["motion"]


def choose(rows, budget_frames=45000):
    """Round-robin source/difficulty coverage, deterministic whole-clip selection."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["source"], row["curriculum_destination"]].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: hashlib.sha256(
            ("k1_holdout_30min_v1:" + ":".join(identity(r))).encode()).hexdigest())
    chosen, used, parents = [], 0, set()
    while True:
        progressed = False
        for group in sorted(buckets):
            for i, row in enumerate(buckets[group]):
                parent = row["source"], row.get("parent_motion", row["motion"])
                if parent not in parents and row["frames"] <= budget_frames - used:
                    chosen.append(row)
                    used += row["frames"]
                    parents.add(parent)
                    buckets[group].pop(i)
                    progressed = True
                    break
        if not progressed:
            return chosen


def main():
    old = ROOT / "organized_remaining_repaired_v2"
    out = ROOT / "organized_holdout_30min_v1"
    out.mkdir(exist_ok=False)
    rows = json.loads((old / "all_motions.json").read_text())
    train_before = [r.copy() for r in rows if r["curriculum_destination"] in STAGES and r["split"] == "train"]
    selected = set()
    for split in ("val", "test"):
        candidates = [r for r in rows if r["curriculum_destination"] in STAGES and r["split"] == split]
        chosen = choose(candidates)
        assert sum(r["frames"] for r in chosen) <= 45000
        selected.update(identity(r) for r in chosen)
    for row in rows:
        if row["curriculum_destination"] in STAGES and row["split"] in {"val", "test"}:
            if identity(row) not in selected:
                row["reserved_from_stage"] = row["curriculum_destination"]
                row["curriculum_destination"] = "80_holdout_reserve"
                row["reserve_reason"] = "30_minute_active_holdout_budget; original split retained"
    train_after = [r for r in rows if r["curriculum_destination"] in STAGES and r["split"] == "train"]
    assert train_before == train_after
    groups = defaultdict(set)
    for row in rows:
        if row.get("robot_root"):
            groups[row["curriculum_destination"], row["split"], row["robot_root"]].add(
                row.get("robot_motion_key", row["motion"]))
    roots = defaultdict(list)
    reference_count = 0
    for i, ((stage, split, source), keys) in enumerate(groups.items()):
        dest = out / stage / split / f"partition_{i:04d}"
        partition(Path(source), dest, keys)
        m = json.loads((dest / "manifest.json").read_text())
        assert set(m["motion_keys"]) == keys
        for key in keys:
            c = m["clips"][key]
            assert (dest / m["hdf5_shards"][c["shard"]]["file"]).is_file()
        reference_count += len(keys)
        roots[stage, split].append(str(dest))
    for (stage, split), paths in roots.items():
        dump(out / "candidate_dataset_roots" / f"{stage}_{split}.json", paths)
    for stage in {r["curriculum_destination"] for r in rows}:
        dump(out / stage / "index.json", [r for r in rows if r["curriculum_destination"] == stage])
    active = [r for r in rows if r["curriculum_destination"] in STAGES]
    summary = {
        "source_snapshot": str(old), "training_unchanged": True,
        "active_splits": {
            split: {"clips": sum(r["split"] == split for r in active),
                    "seconds": sum(r["frames"]/50 for r in active if r["split"] == split)}
            for split in ("train", "val", "test")
        },
        "holdout_coverage": {
            split: dict(Counter(r["source"] + "/" + r["curriculum_destination"]
                               for r in active if r["split"] == split))
            for split in ("val", "test")
        },
        "reserve_clips": sum(r["curriculum_destination"] == "80_holdout_reserve" for r in rows),
        "reserve_seconds": sum(r["frames"]/50 for r in rows if r["curriculum_destination"] == "80_holdout_reserve"),
        "reference_count_verified": reference_count,
        "whole_clips_preserved": True, "original_splits_preserved": True,
        "training_started": False, "training_approved": False,
        "note": "Only active holdout manifests reduced; reserve is not training data. No source files deleted.",
    }
    assert reference_count == sum(bool(r.get("robot_root")) for r in rows)
    dump(out / "all_motions.json", rows)
    dump(out / "summary.json", summary)
    pointer = json.loads((ROOT / "active_review_snapshot.json").read_text())
    pointer.update(snapshot=str(out), report=str(out / "summary.json"),
                   integrity_report=str(out / "summary.json"), previous_snapshot=str(old),
                   preparation_status="active_holdout_reduced_to_30_minutes; reserves_excluded")
    dump(ROOT / "active_review_snapshot.json", pointer)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
