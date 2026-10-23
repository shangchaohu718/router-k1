"""Integrate screened repairs and eligible reserves without leakage or overlap."""

from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

from audit_reserve_leakage import STAGES, group_keys, pose_hash, reasons
from localize_and_salvage import dump
from organize_motion_curriculum import partition

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"


def identity(r):
    return r["source"], r["motion"]


def best_nonoverlap(items):
    """Maximum retained frame count; prefer old versions on equal coverage."""
    ordered = sorted(
        items,
        key=lambda r: (r["canonical_end"], r["canonical_start"], r["motion"]),
    )
    ends = [r["canonical_end"] for r in ordered]
    scores, choices = [(0, 0)], [set()]
    for i, row in enumerate(ordered):
        before = bisect_right(ends, row["canonical_start"], hi=i)
        n = row["canonical_end"] - row["canonical_start"]
        take = (
            scores[before][0] + n,
            scores[before][1] + (0 if row.get("new_direction_repair") else n),
        )
        if take > scores[-1]:
            scores.append(take)
            choices.append(choices[before] | {identity(row)})
        else:
            scores.append(scores[-1])
            choices.append(choices[-1].copy())
    return choices[-1]


def set_span(row):
    root = Path(row["robot_root"])
    m = json.loads((root / "manifest.json").read_text())
    c = m["clips"][row.get("robot_motion_key", row["motion"])]
    meta = c["metadata"]
    start = int(meta.get("source_start_frame_trim", 0)) + int(
        meta.get("source_frame_start", 0)
    )
    row["canonical_start"], row["canonical_end"] = start, start + c["length"]


def main():
    old = ROOT / "organized_holdout_30min_v1"
    out = ROOT / "organized_targeted_repaired_v1"
    out.mkdir(exist_ok=False)
    rows = json.loads((old / "all_motions.json").read_text())
    new = json.loads(
        (
            ROOT / "targeted_frame_repair_v1/classified_v2/all_motions.json"
        ).read_text()
    )
    audit = json.loads(
        (
            ROOT / "targeted_frame_repair_v1/reserve_leakage_audit.json"
        ).read_text()
    )
    prior_repairs = json.loads(
        (ROOT / "arm_repair_v1/repair_decisions.json").read_text()
    )
    regression_holds = {
        (r["source"], r["motion"]): [
            why for why in r["reasons"] if "regression" in why
        ]
        for r in prior_repairs
        if any("regression" in why for why in r["reasons"])
    }
    protected, hashes = (
        set(audit["protected_group_keys"]),
        set(audit["protected_pose_hashes"]),
    )
    holdout_before = [
        dict(r)
        for r in rows
        if r["curriculum_destination"] in STAGES
        and r["split"] in {"val", "test"}
    ]
    reserve_decisions = {
        (r["source"], r["motion"]): r for r in audit["reserve_decisions"]
    }
    conflicts = {
        (r["source"], r["motion"]): r
        for r in audit["existing_train_conflicts"]
    }
    for row in rows:
        k = identity(row)
        if k in conflicts:
            row["reserved_from_stage"] = row["curriculum_destination"]
            row["curriculum_destination"] = "80_holdout_reserve"
            row["leakage_hold"] = conflicts[k]["reasons"]
        elif k in reserve_decisions:
            decision = reserve_decisions[k]
            if decision["eligible_for_train"]:
                row["original_split"] = row["split"]
                row["split"] = "train"
                row["curriculum_destination"] = row.pop("reserved_from_stage")
                row["reserve_promoted_to_train"] = True
                row.pop("reserve_reason", None)
            else:
                row["leakage_hold"] = decision["reasons"]
    for row in new:
        row["robot_motion_key"] = row.get("robot_motion_key", row["motion"])
        row["motion"] += "__direction_repaired_v1"
        row["new_direction_repair"] = True
        regression = regression_holds.get(
            (row["source"], row["parent_motion"])
        )
        if regression:
            row["prior_regression_hold"] = regression
            row["curriculum_destination"] = "90_quarantine"
            row["quality"] = (
                "segment_absolute_gate_pass_does_not_clear_prior_regression"
            )
        if row["curriculum_destination"] in STAGES:
            why = reasons(row, protected, hashes)
            row["original_split"] = row["split"]
            if why:
                row["reserved_from_stage"] = row["curriculum_destination"]
                row["curriculum_destination"] = "80_holdout_reserve"
                row["leakage_hold"] = why
            else:
                row["split"] = "train"
        rows.append(row)
    # Do not count alternative retargetings of the same capture interval twice.
    grouped = defaultdict(list)
    for row in rows:
        if row["curriculum_destination"] in STAGES and row["split"] == "train":
            set_span(row)
            grouped[
                row["source"], row.get("parent_motion", row["motion"])
            ].append(row)
    for group in grouped.values():
        kept = best_nonoverlap(group)
        for row in group:
            if identity(row) not in kept:
                row["superseded_from_stage"] = row["curriculum_destination"]
                row["curriculum_destination"] = "81_superseded_segments"
                row["superseded_reason"] = (
                    "overlap_alternative; maximum nonoverlapping frame coverage; prefer old on tie"
                )
    holdout_after = [
        r
        for r in rows
        if r["curriculum_destination"] in STAGES
        and r["split"] in {"val", "test"}
    ]
    assert holdout_before == holdout_after, "Frozen 30-minute holdout changed"
    train = [
        r
        for r in rows
        if r["curriculum_destination"] in STAGES and r["split"] == "train"
    ]
    assert len({identity(r) for r in rows}) == len(rows)
    for row in train:
        assert group_keys(row) is not None and not (
            group_keys(row) & protected
        )
        assert pose_hash(row) not in hashes
        assert not row.get("semantic_hold")
    for group in grouped.values():
        kept = sorted(
            (r["canonical_start"], r["canonical_end"])
            for r in group
            if r["curriculum_destination"] in STAGES
        )
        assert all(a[1] <= b[0] for a, b in zip(kept, kept[1:]))
    groups = defaultdict(set)
    for row in rows:
        if row.get("robot_root"):
            groups[
                row["curriculum_destination"], row["split"], row["robot_root"]
            ].add(row.get("robot_motion_key", row["motion"]))
    roots = defaultdict(list)
    refs = 0
    for i, ((stage, split, source), keys) in enumerate(groups.items()):
        dest = out / stage / split / f"partition_{i:04d}"
        partition(Path(source), dest, keys)
        m = json.loads((dest / "manifest.json").read_text())
        assert set(m["motion_keys"]) == keys
        m["active_split"] = split
        for k, c in m["clips"].items():
            assert (dest / m["hdf5_shards"][c["shard"]]["file"]).is_file()
            c["metadata"]["source_metadata_split"] = c["metadata"].get("split")
            c["metadata"]["active_split"] = split
        dump(dest / "manifest.json", m)
        roots[stage, split].append(str(dest))
        refs += len(keys)
    for (stage, split), paths in roots.items():
        dump(out / "candidate_dataset_roots" / f"{stage}_{split}.json", paths)
    for stage in {r["curriculum_destination"] for r in rows}:
        dump(
            out / stage / "index.json",
            [r for r in rows if r["curriculum_destination"] == stage],
        )
    final_new_train = [r for r in train if r.get("new_direction_repair")]
    summary = {
        "source_snapshot": str(old),
        "input_mapping_cases": 188,
        "new_segment_candidates": len(new),
        "new_repair_segments_in_train": len(final_new_train),
        "new_repair_train_seconds": sum(r["seconds"] for r in final_new_train),
        "new_segment_destinations": dict(
            Counter(
                r["curriculum_destination"]
                for r in rows
                if r.get("new_direction_repair")
            )
        ),
        "reserve_checked": 411,
        "reserve_eligible": audit["eligible"],
        "reserve_retained_in_train": sum(
            r.get("reserve_promoted_to_train", False) for r in train
        ),
        "reserve_blocked": audit["held"],
        "existing_train_conflicts_removed": len(conflicts),
        "active_train_records": len(train),
        "active_train_hours": sum(r["seconds"] for r in train) / 3600,
        "holdout_records": len(holdout_after),
        "holdout_seconds": sum(r["seconds"] for r in holdout_after),
        "counts": dict(Counter(r["curriculum_destination"] for r in rows)),
        "verified_robot_references": refs,
        "holdout_unchanged": True,
        "no_train_holdout_group_or_pose_hash_overlap": True,
        "no_overlapping_train_intervals_per_capture": True,
        "training_started": False,
        "training_approved": False,
        "limitations": "Numerical/FK and human-direction screening, not full dynamic tracking approval; hash/group isolation is not a guarantee against all undocumented near-duplicates",
    }
    dump(out / "all_motions.json", rows)
    dump(out / "summary.json", summary)
    dump(ROOT / "targeted_frame_repair_v1/integration_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
