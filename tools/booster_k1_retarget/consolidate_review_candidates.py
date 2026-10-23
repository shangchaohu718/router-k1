"""Versioned, source-interval deduplicated candidate views; never training.

Select one coherent review per input reference before removing overlaps across
references. Preserve semantic/origin holds and all original rejected reports.
"""

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time

from tools.booster_k1_retarget.recover_short_reference_candidates import (
    dump,
    eligible,
)
from tools.booster_k1_retarget.organize_motion_curriculum import (
    kinematic_difficulty,
)

STAGES = ["01_basic", "02_locomotion", "03_dynamic", "04_coordination"]


def subtract(start, end, occupied):
    pieces = [(start, end)]
    for a, b in sorted(occupied):
        pieces = [
            span
            for s, e in pieces
            for span in (
                [(s, e)]
                if e <= a or s >= b
                else [(s, min(e, a)), (max(s, b), e)]
            )
            if span[1] > span[0]
        ]
    return pieces


def reviews(base):
    chosen = {}
    lineage = {}

    def consider(report, pool):
        report = Path(report)
        data = json.loads(report.read_text())
        candidates = defaultdict(list)
        for category in [
            "01_kinematic_candidates",
            "80_short_or_source_review",
        ]:
            root = report.parent / category
            manifest = json.loads((root / "manifest.json").read_text())
            rows = {r["motion_key"]: r for r in data["clips"]}
            for key, clip in manifest["clips"].items():
                parent = clip["metadata"]["parent_motion_key"]
                if category.startswith("80") and not eligible(
                    clip, rows[parent]
                ):
                    continue
                candidates[parent].append((root, manifest, key, clip))
        for row in data["clips"]:
            identity = (pool, row["motion_key"])
            entries = candidates[row["motion_key"]]
            score = sum(c[3]["length"] for c in entries)
            if identity not in chosen or score > chosen[identity][0]:
                chosen[identity] = (score, entries, str(report), row)

    for pool in [
        "batch_robot_proportions_source_checked_v3",
        "backlog_robot_proportions_source_checked_v3",
    ]:
        for f in sorted((base / pool).glob("batch_*/status.json")):
            s = json.loads(f.read_text())
            if s["status"] == "complete":
                consider(s["report"], pool)
    for name in [
        "unrecovered_retry_variants_v4",
        "native_cmu_ankle_retry_v5",
        "native_cmu_mixed_retry_v6",
    ]:
        for f in sorted((base / name).glob("batch_*/status.json")):
            s = json.loads(f.read_text())
            if s["status"] != "complete":
                continue
            key = s["motion_keys"][0]
            pool = s["original_pool"]
            if name == "unrecovered_retry_variants_v4":
                lineage[key] = pool
            else:
                pool = lineage[key]
            consider(s["report"], pool)
    # Independently repaired failures may not yet have been discovered by v4.
    summary = json.loads((base / "failed_retry_v4/summary.json").read_text())
    for s in summary["repairs"]:
        pool = (
            "backlog_robot_proportions_source_checked_v3"
            if s["motion"] in {"13_10", "62_11"}
            else "batch_robot_proportions_source_checked_v3"
        )
        consider(s["review_report"], pool)
    return chosen


def build(base, output):
    chosen = reviews(base)
    frozen_sources = set()
    for f in (base.parent / "organized_targeted_repaired_v1").glob(
        "*/*/partition_*/manifest.json"
    ):
        if f.parent.parent.name not in {"val", "test"}:
            continue
        for c in json.loads(f.read_text())["clips"].values():
            frozen_sources.add(
                str(Path(c["metadata"]["source_path"]).resolve())
            )
    occupied = defaultdict(list)
    groups = defaultdict(list)
    removed = too_short = 0
    holds = []
    # Ordinary main references own overlap ahead of held backlog variants.
    for (pool, parent), (score, entries, report, row) in sorted(
        chosen.items(),
        key=lambda item: (
            item[0][0] != "batch_robot_proportions_source_checked_v3",
            item[0],
        ),
    ):
        if not score:
            holds.append(
                {
                    "pool": pool,
                    "motion": parent,
                    "report": report,
                    "frames": row["frames"],
                    "reasons": row["reason_intervals"],
                    "source_error": row.get("source_error"),
                }
            )
        for root, manifest, key, original in entries:
            meta = original["metadata"]
            if meta.get("active_split", "train") != "train":
                raise ValueError(
                    "Frozen split cannot enter candidate consolidation"
                )
            source = str(Path(meta["source_path"]).resolve())
            if source in frozen_sources:
                continue
            start = int(meta["canonical_frame_start"])
            end = start + original["length"]
            pieces = subtract(start, end, occupied[source])
            removed += original["length"] - sum(e - s for s, e in pieces)
            for s, e in pieces:
                if e - s < 50:
                    too_short += e - s
                    continue
                clip = copy.deepcopy(original)
                clip["start"] += s - start
                clip["length"] = e - s
                cm = clip["metadata"]
                cm.update(
                    canonical_frame_start=s,
                    canonical_frame_count=e - s,
                    num_frames=e - s,
                    dedup_parent_manifest=str(root / "manifest.json"),
                    dedup_parent_key=key,
                    training_approved=False,
                )
                # Preserve frame lineage after cutting an overlap; never stitch.
                cm["parent_frame_start"] += s - start
                cm["parent_frame_end"] = cm["parent_frame_start"] + e - s
                stage = cm.get(
                    "geometry_review_original_destination",
                    Path(cm["geometry_review_parent_root"]).parts[-3],
                )
                level, metrics = kinematic_difficulty(root, manifest, clip)
                measured = STAGES[int(level[1]) - 1]
                ordinary = (
                    stage in STAGES
                    and pool == "batch_robot_proportions_source_checked_v3"
                )
                difficulty = (
                    STAGES[max(STAGES.index(stage), STAGES.index(measured))]
                    if stage in STAGES
                    else measured
                )
                category = "01_candidates" if ordinary else "80_origin_review"
                cm.update(
                    provisional_difficulty=difficulty,
                    difficulty_metrics=metrics,
                    origin_review_required=not ordinary,
                )
                shard = copy.deepcopy(manifest["hdf5_shards"][clip["shard"]])
                shard["file"] = str((root / shard["file"]).resolve())
                clip["shard"] = 0
                groups[(category, difficulty)].append(
                    (clip, shard, manifest["dof_names"])
                )
                occupied[source].append((s, e))
    stats = {}
    for (category, stage), values in groups.items():
        target = output / category / stage
        clips, shards = {}, []
        for clip, shard, names in values:
            clip["shard"] = len(shards)
            shards.append(shard)
            key = hashlib.sha256(
                json.dumps(
                    [
                        clip["metadata"]["source_path"],
                        clip["metadata"]["canonical_frame_start"],
                        clip["length"],
                        shard["file"],
                    ]
                ).encode()
            ).hexdigest()[:24]
            clips[key] = clip
        dump(
            target / "manifest.json",
            {
                "version": 2,
                "dataset_id": output.name + "_" + category + "_" + stage,
                "dof_names": names,
                "clips": clips,
                "motion_keys": list(clips),
                "hdf5_shards": shards,
                "training_approved": False,
            },
        )
        dump(target / "nan_npz_paths.json", [])
        frames = sum(c["length"] for c in clips.values())
        stats[category + "/" + stage] = {
            "segments": len(clips),
            "frames": frames,
            "hours": frames / 180000,
        }
    result = {
        "training_started": False,
        "training_approved": False,
        "physics_validation_pending": True,
        "stage_order": STAGES,
        "groups": stats,
        "duplicate_frames_not_counted": removed,
        "unique_tails_under_50_frames_held": too_short,
        "zero_recovery_references": len(holds),
        "dedup_scope": "Same resolved source capture and canonical 50 Hz frame intervals; not semantic or near-duplicate motion detection",
        "limitations": "Origin/scene/semantic holds preserved. Difficulty is provisional, never below inherited stage. No training or automatic activation. Immutable source H5 unchanged.",
    }
    dump(output / "zero_recovery_review.json", holds)
    dump(output / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    while True:
        signature = hashlib.sha256(
            "".join(
                str(f.stat().st_mtime_ns)
                for f in sorted(a.base.glob("*/progress.json"))
            ).encode()
        ).hexdigest()[:16]
        target = a.output / ("snapshot_" + signature)
        if not (target / "summary.json").exists():
            build(a.base, target)
        dump(
            a.output / "latest.json",
            {"snapshot": str(target.resolve()), "training_approved": False},
        )
        if not a.follow:
            return
        parents = [
            json.loads(f.read_text())
            for f in a.base.glob("*retry_v*/progress.json")
        ]
        if parents and all(
            s["status"] != "data_repair_in_progress" for s in parents
        ):
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
