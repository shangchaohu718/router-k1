"""Analyze temporal gate traces and the synthetic transition fixture."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from tools.booster_k1_retarget.summarize_sliding_eval import summarize


def first_episode(indices, keys, frames):
    if not len(indices):
        return indices
    for i in range(1, len(indices)):
        prev, cur = indices[i - 1], indices[i]
        if keys[cur] != keys[prev] or frames[cur] != frames[prev] + 1:
            return indices[:i]
    return indices


def analyze(job_dir, fixture=None):
    files = sorted(job_dir.glob("checkpoint/isaaclab_eval_output*/*.npz"))
    clips = [summarize(p) for p in files]
    result = {
        "clips": clips,
        "completion_rate": float(np.mean([r["completed"] for r in clips])),
        "gate_traces": [],
    }
    with np.load(job_dir / "gate_samples.npz", allow_pickle=False) as samples:
        if "batch_ids" not in samples:
            raise ValueError(
                "Need recorder batch IDs for unambiguous trace analysis"
            )
        keys, frames = samples["motion_keys"], samples["reference_frames"]
        for batch in np.unique(samples["batch_ids"]):
            for env in np.unique(samples["env_ids"]):
                ids = np.flatnonzero(
                    (samples["batch_ids"] == batch)
                    & (samples["env_ids"] == env)
                )
                ids = first_episode(ids, keys, frames)
                if not len(ids):
                    continue
                weight = samples["gate_weight"][ids]
                row = {
                    "batch_id": int(batch),
                    "env_id": int(env),
                    "motion_key": str(keys[ids[0]]),
                    "first_episode_samples": len(ids),
                    "active_fraction": float(weight.mean()),
                    "switches": [
                        {
                            "frame": int(frames[ids[t]]),
                            "expert_active": bool(weight[t]),
                        }
                        for t in np.flatnonzero(np.diff(weight)) + 1
                    ],
                }
                if fixture:
                    start, end = fixture["slide_interval"]
                    f = frames[ids]
                    slide = (f >= start) & (f < end)
                    # The fixture blends for 50 frames on each side.
                    ordinary = (f < start - 50) | (f >= end + 50)
                    row.update(
                        slide_selection_fraction=float(weight[slide].mean())
                        if slide.any()
                        else None,
                        ordinary_selection_fraction=float(
                            weight[ordinary].mean()
                        )
                        if ordinary.any()
                        else None,
                    )
                result["gate_traces"].append(row)
    if fixture:
        start, end = fixture["slide_interval"]
        # Native evaluator uses one filename per key, so padded duplicate
        # environments overwrite earlier trajectories. Explicitly report this
        # as one retained kinematic trajectory; trace statistics cover both.
        result["native_duplicate_limitation"] = (
            "Native NPZ uses motion key as filename; duplicate environments "
            "retain only the last trajectory. Gate traces retain each env."
        )
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                if len(data["robot_dof_pos"]) < end:
                    continue
                arrays = {
                    k: data[k][start:end].copy()
                    for k in data.files
                    if k != "metadata"
                }
                meta = json.loads(str(data["metadata"]))
            meta.update(
                motion_key="moonwalk_central_segment_of_transition",
                clip_length=end - start + 10,
                valid_prefix_len=end - start,
            )
            arrays["metadata"] = json.dumps(meta)
            analysis = job_dir / "analysis"
            analysis.mkdir(exist_ok=True)
            derived = analysis / "slide_segment.npz"
            if not derived.exists():
                with derived.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
            result["central_slide_mechanics"] = summarize(derived)
    result["limitations"] = (
        "Trace prefixes stop on reference reset/reassignment. Synthetic "
        "transition is a development stress test, not independent sliding "
        "generalization or verified contact-force evidence."
    )
    (job_dir / "trace_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text()) if args.fixture else None
    print(json.dumps(analyze(args.job_dir, fixture), indent=2))


if __name__ == "__main__":
    main()
