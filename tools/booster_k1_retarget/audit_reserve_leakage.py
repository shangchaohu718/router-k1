"""Conservative holdout isolation: capture, person/style, family, pose sequence."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from localize_and_salvage import dump

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"
STAGES = {"01_basic", "02_locomotion", "03_dynamic", "04_coordination"}


def group_keys(row):
    source = row["source"]
    name = row.get("parent_motion", row["motion"])
    keys = {f"capture:{source}:{name}"}
    if source == "CMU":
        match = re.fullmatch(r"(\d+)_\d+", name)
        if not match:
            return None
        keys.add(f"subject:CMU:{int(match.group(1))}")
    elif source == "LaFAN1":
        match = re.search(r"subject(\d+)$", name)
        if not match:
            return None
        keys.add(f"subject:LaFAN1:{int(match.group(1))}")
    elif source == "100STYLE":
        style, sep, movement = name.rpartition("_")
        if not sep or not style or not movement:
            return None
        keys.add(f"style:100STYLE:{style}")
    elif source == "AIST++":
        match = re.search(r"_d(\d+)_", name)
        if not match:
            return None
        keys.add(f"dancer:AIST++:{int(match.group(1))}")
        # Conservatively hold identical filename families differing only in dancer.
        # This does not assert that the corresponding performances are identical.
        keys.add("filename_family:AIST++:" + re.sub(r"_d\d+_", "_dANY_", name))
    else:
        return None
    return keys


def pose_hash(row):
    root = Path(row["robot_root"])
    m = json.loads((root / "manifest.json").read_text())
    c = m["clips"][row.get("robot_motion_key", row["motion"])]
    start, n = c["start"], c["length"]
    h = hashlib.sha256(str(n).encode())
    with h5py.File(root / m["hdf5_shards"][c["shard"]]["file"], "r") as f:
        for key in ("ref_root_pos", "ref_root_rot", "ref_dof_pos"):
            x = f[key][start : start + n].astype(np.float64)
            if key == "ref_root_pos":
                x -= x[0]
            # Remove negative-zero representations after quantization.
            x = np.rint(x * 1e5).astype("<i8")
            h.update(x.tobytes())
    return h.hexdigest()


def reasons(row, protected, hashes):
    keys = group_keys(row)
    if keys is None:
        return ["group_identity_unresolved"]
    found = sorted(keys & protected)
    if pose_hash(row) in hashes:
        found.append("holdout_pose_sequence_duplicate_at_1e-5_resolution")
    return found


def main():
    rows = json.loads(
        (ROOT / "organized_holdout_30min_v1/all_motions.json").read_text()
    )
    holdout = [
        r
        for r in rows
        if r["curriculum_destination"] in STAGES
        and r["split"] in {"val", "test"}
    ]
    assert len(holdout) == 40
    protected = set().union(*(group_keys(r) for r in holdout))
    hashes = {pose_hash(r) for r in holdout}
    reserve = [
        r for r in rows if r["curriculum_destination"] == "80_holdout_reserve"
    ]
    decisions = []
    for row in reserve:
        why = reasons(row, protected, hashes)
        decisions.append(
            {
                "source": row["source"],
                "motion": row["motion"],
                "eligible_for_train": not why,
                "reasons": why,
            }
        )
    conflicts = []
    for row in rows:
        if row["curriculum_destination"] in STAGES and row["split"] == "train":
            why = reasons(row, protected, hashes)
            if why:
                conflicts.append(
                    {
                        "source": row["source"],
                        "motion": row["motion"],
                        "reasons": why,
                    }
                )
    report = {
        "reserve_records": len(reserve),
        "eligible": sum(r["eligible_for_train"] for r in decisions),
        "held": sum(not r["eligible_for_train"] for r in decisions),
        "existing_training_conflicts_under_stricter_policy": len(conflicts),
        "eligible_by_source": dict(
            Counter(r["source"] for r in decisions if r["eligible_for_train"])
        ),
        "protected_group_keys": sorted(protected),
        "protected_pose_hashes": sorted(hashes),
        "reserve_decisions": decisions,
        "existing_train_conflicts": conflicts,
        "limitations": "Conservative group and pose-hash screening; not proof against every possible near-duplicate or unrecorded capture relationship",
        "training_started": False,
    }
    dump(ROOT / "targeted_frame_repair_v1/reserve_leakage_audit.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in {
                    "reserve_decisions",
                    "existing_train_conflicts",
                    "protected_group_keys",
                    "protected_pose_hashes",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
