"""Compare old-motion trajectories on matched prefixes and actual gate use."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from tools.booster_k1_retarget.summarize_sliding_router_trace import (
    first_episode,
)


def compare_arrays(base, candidate):
    n = min(len(base["robot_dof_pos"]), len(candidate["robot_dof_pos"]))
    if not n:
        raise ValueError("No common valid prefix")
    ref_delta = float(
        np.abs(base["ref_dof_pos"][:n] - candidate["ref_dof_pos"][:n]).max()
    )
    if ref_delta > 1e-5:
        raise ValueError("Reference mismatch on paired trajectory prefix")
    result = {
        "matched_frames": n,
        "reference_joint_max_difference_rad": ref_delta,
        "robot_joints_bitwise_equal_on_prefix": np.array_equal(
            base["robot_dof_pos"][:n], candidate["robot_dof_pos"][:n]
        ),
    }
    for label, arrays in (("base", base), ("candidate", candidate)):
        result[label + "_matched_joint_mae_rad"] = float(
            np.abs(
                arrays["robot_dof_pos"][:n] - arrays["ref_dof_pos"][:n]
            ).mean()
        )
        result[label + "_matched_body_error_mm"] = float(
            np.linalg.norm(
                arrays["robot_global_translation"][:n]
                - arrays["ref_global_translation"][:n],
                axis=-1,
            ).mean()
            * 1000
        )
    result["matched_joint_mae_increase_rad"] = (
        result["candidate_matched_joint_mae_rad"]
        - result["base_matched_joint_mae_rad"]
    )
    return result


def artifacts(job):
    rows = {}
    for path in sorted(job.glob("checkpoint/isaaclab_eval_output*/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        key = metadata["motion_key"]
        if key in rows:
            raise ValueError("Duplicate native motion key; use one env")
        rows[key] = (path, metadata)
    if not rows:
        raise ValueError(f"No native trajectories: {job}")
    return rows


def compare(base_job, candidate_job):
    base, candidate = artifacts(base_job), artifacts(candidate_job)
    if set(base) != set(candidate):
        raise ValueError("Incomplete or mismatched clip coverage")
    rows = []
    for key in sorted(base):
        bp, bm = base[key]
        cp, cm = candidate[key]
        with (
            np.load(bp, allow_pickle=False) as b,
            np.load(cp, allow_pickle=False) as c,
        ):
            row = compare_arrays(b, c)
            bn, cn = len(b["robot_dof_pos"]), len(c["robot_dof_pos"])
        expected = int(bm["clip_length"]) - 10
        if int(cm["clip_length"]) != int(bm["clip_length"]):
            raise ValueError("Clip length mismatch")
        row.update(
            motion_key=key,
            base_frames=bn,
            candidate_frames=cn,
            base_completed=bn >= expected,
            candidate_completed=cn >= expected,
        )
        rows.append(row)
    valid_expert_samples, total_valid_samples = 0, 0
    valid_lengths = {r["motion_key"]: r["candidate_frames"] for r in rows}
    with np.load(
        candidate_job / "gate_samples.npz", allow_pickle=False
    ) as samples:
        for batch in np.unique(samples["batch_ids"]):
            for env in np.unique(samples["env_ids"]):
                ids = np.flatnonzero(
                    (samples["batch_ids"] == batch)
                    & (samples["env_ids"] == env)
                )
                ids = first_episode(
                    ids, samples["motion_keys"], samples["reference_frames"]
                )
                if not len(ids):
                    continue
                key = str(samples["motion_keys"][ids[0]])
                if key not in valid_lengths:
                    raise ValueError(
                        "Gate trace not covered by native outputs"
                    )
                ids = ids[: valid_lengths[key]]
                total_valid_samples += len(ids)
                valid_expert_samples += int(
                    np.count_nonzero(samples["gate_weight"][ids])
                )
    return {
        "base_job": str(base_job),
        "candidate_job": str(candidate_job),
        "clip_count": len(rows),
        "base_completed": sum(r["base_completed"] for r in rows),
        "candidate_completed": sum(r["candidate_completed"] for r in rows),
        "lost_completions": [
            r["motion_key"]
            for r in rows
            if r["base_completed"] and not r["candidate_completed"]
        ],
        "all_robot_joint_prefixes_bitwise_equal": all(
            r["robot_joints_bitwise_equal_on_prefix"] for r in rows
        ),
        "macro_matched_joint_mae_increase_rad": float(
            np.mean([r["matched_joint_mae_increase_rad"] for r in rows])
        ),
        "valid_gate_samples": total_valid_samples,
        "valid_expert_selections": valid_expert_samples,
        "clips": rows,
        "scope": (
            "Matched valid prefixes; raw outputs preserved. One seed and "
            "fixed reference windows, not a guarantee for unseen motions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_job", type=Path)
    parser.add_argument("candidate_job", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.base_job, args.candidate_job)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2)
    )


if __name__ == "__main__":
    main()
