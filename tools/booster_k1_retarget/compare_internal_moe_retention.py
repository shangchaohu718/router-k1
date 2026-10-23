"""Match old-action trajectories and actual internal fine-expert selections."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from tools.booster_k1_retarget.compare_sliding_router_retention import (
    artifacts,
    compare_arrays,
)
from tools.booster_k1_retarget.summarize_sliding_router_trace import (
    first_episode,
)


def compare(base_job, candidate_job):
    base, candidate = artifacts(base_job), artifacts(candidate_job)
    if set(base) != set(candidate):
        raise ValueError("Missing or mismatched old-action clip coverage")
    rows = []
    valid_lengths = {}
    keys = (
        "robot_dof_pos",
        "ref_dof_pos",
        "robot_global_translation",
        "ref_global_translation",
    )
    for key in sorted(base):
        bp, bm = base[key]
        cp, cm = candidate[key]
        with (
            np.load(bp, allow_pickle=False) as b,
            np.load(cp, allow_pickle=False) as c,
        ):
            bn = min(
                len(b["robot_dof_pos"]),
                int(bm.get("valid_prefix_len", len(b["robot_dof_pos"]))),
            )
            cn = min(
                len(c["robot_dof_pos"]),
                int(cm.get("valid_prefix_len", len(c["robot_dof_pos"]))),
            )
            row = compare_arrays(
                {k: b[k][:bn] for k in keys}, {k: c[k][:cn] for k in keys}
            )
            row["robot_joint_max_difference_rad"] = float(
                np.abs(
                    b["robot_dof_pos"][: min(bn, cn)]
                    - c["robot_dof_pos"][: min(bn, cn)]
                ).max()
            )
        if bm["clip_length"] != cm["clip_length"]:
            raise ValueError("Old-action clip lengths do not match")
        expected = int(bm["clip_length"]) - 10
        row.update(
            motion_key=key,
            base_frames=bn,
            candidate_frames=cn,
            base_completed=bn >= expected,
            candidate_completed=cn >= expected,
        )
        rows.append(row)
        valid_lengths[key] = cn
    selected_steps, total_steps, assignments = 0, 0, 0
    with np.load(
        candidate_job / "ordered_observations.npz", allow_pickle=False
    ) as trace:
        old_experts = int(trace["old_fine_experts"])
        for batch in np.unique(trace["batch_ids"]):
            ids = np.flatnonzero(trace["batch_ids"] == batch)
            ids = first_episode(
                ids, trace["motion_keys"], trace["reference_frames"]
            )
            if not len(ids):
                continue
            key = str(trace["motion_keys"][ids[0]])
            if key not in valid_lengths:
                raise ValueError(
                    "Internal route recording has no native trajectory"
                )
            ids = ids[: valid_lengths[key]]
            new = trace["selected_experts"][ids] >= old_experts
            total_steps += len(ids)
            selected_steps += int(new.any(-1).sum())
            assignments += int(new.sum())
    return {
        "base_job": str(base_job),
        "candidate_job": str(candidate_job),
        "clip_count": len(rows),
        "base_completed": sum(row["base_completed"] for row in rows),
        "candidate_completed": sum(row["candidate_completed"] for row in rows),
        "all_robot_joint_prefixes_bitwise_equal": all(
            row["robot_joints_bitwise_equal_on_prefix"] for row in rows
        ),
        "macro_matched_joint_mae_increase_rad": float(
            np.mean([row["matched_joint_mae_increase_rad"] for row in rows])
        ),
        "valid_recorded_steps": total_steps,
        "steps_selecting_added_experts": selected_steps,
        "added_expert_assignments": assignments,
        "clips": rows,
        "scope": "Matched valid prefixes and first-episode actual internal routing; one seed and registered windows only, not unseen-motion or robot evidence.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_job", type=Path)
    parser.add_argument("candidate_job", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.base_job, args.candidate_job)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "clips"},
            indent=2,
        )
    )
