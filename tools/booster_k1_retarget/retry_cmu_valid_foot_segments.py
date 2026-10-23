"""Isolate degenerate native foot frames, retry only separate valid runs."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from holosmpl.converters.bvh_family.cmu_asf import load_cmu_fk
from tools.booster_k1_retarget.localize_and_salvage import intervals, padded
from tools.booster_k1_retarget.probe_native_position_retarget import (
    canonical_start,
)
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.retry_unrecovered_references import score_review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    a = p.parse_args()
    m = json.loads((a.input / "manifest.json").read_text())
    if len(m["clips"]) != 1:
        raise ValueError("One source per retry")
    key, clip = next(iter(m["clips"].items()))
    meta = clip["metadata"]
    _, starts, ends, _ = load_cmu_fk(Path(meta["source_path"]))
    t = (
        (canonical_start(meta) + np.arange(clip["length"]))
        * meta["source_fps"]
        / 50
    )
    bad = np.zeros(clip["length"], bool)
    for side in ["l", "r"]:
        values = [
            np.stack(
                [
                    np.interp(t, np.arange(len(v)), v[:, axis])
                    for axis in range(3)
                ],
                axis=1,
            )
            for v in [
                starts[side + "foot"],
                starts[side + "toes"],
                ends[side + "toes"],
            ]
        ]
        forward = values[2] - values[1]
        norm = np.linalg.norm(forward, axis=1)
        forward /= np.maximum(norm[:, None], 1e-12)
        bad |= (norm < 1e-6) | (
            np.linalg.norm(np.cross(values[0] - values[1], forward), axis=1)
            < 1e-6
        )
    spans = [(s, e) for s, e in intervals(~padded(bad, 5)) if e - s >= 50]
    result = {
        "motion": key,
        "training_started": False,
        "training_approved": False,
        "source_frames": clip["length"],
        "degenerate_intervals": intervals(bad),
        "padded_hold_intervals": intervals(padded(bad, 5)),
        "segments": [],
        "policy": "Separate valid source runs, 5-frame padding; unchanged standard quality gates. Never interpolate degenerate foot orientations or join across invalid frames.",
    }
    dump(a.output / "summary.json", result)
    for s, e in spans:
        root = a.output / f"source_frames_{s}_{e}"
        inp = root / "input"
        view = copy.deepcopy(m)
        c = copy.deepcopy(clip)
        c["start"] += s
        c["length"] = e - s
        c["metadata"].update(
            canonical_frame_start=canonical_start(meta) + s,
            canonical_frame_count=e - s,
            num_frames=e - s,
            native_source_valid_parent_span=[s, e],
            training_approved=False,
        )
        view.update(
            clips={key: c},
            motion_keys=[key],
            hdf5_shards=[
                {**shard, "file": str((a.input / shard["file"]).resolve())}
                for shard in m["hdf5_shards"]
            ],
        )
        dump(inp / "manifest.json", view)
        variants = []
        for cost in [0.5, 0.1, 2.0]:
            variant = root / f"orientation_{cost}"
            raw, review = variant / "raw", variant / "review"
            report = review / "report.json"
            if not report.exists():
                variant.mkdir(parents=True, exist_ok=True)
                with (variant / "worker.log").open("a") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.booster_k1_retarget.retry_cmu_native_foot_frame",
                            "--input",
                            str(inp),
                            "--output",
                            str(raw),
                            "--smpl-model",
                            str(a.smpl_model),
                            "--foot-orientation-cost",
                            str(cost),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.booster_k1_retarget.salvage_repaired_reference",
                            "--input",
                            str(raw),
                            "--output",
                            str(review),
                            "--smpl-model",
                            str(a.smpl_model),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            score, short = score_review(report)
            variants.append(
                {
                    "report": str(report.resolve()),
                    "cost": cost,
                    "score": score,
                    "short_frames": short,
                }
            )
        best = max(variants, key=lambda v: tuple(v["score"]))
        result["segments"].append(
            {"parent_span": [s, e], "selected": best, "variants": variants}
        )
        result["candidate_frames"] = sum(
            v["selected"]["score"][0] for v in result["segments"]
        )
        dump(a.output / "summary.json", result)


if __name__ == "__main__":
    main()
