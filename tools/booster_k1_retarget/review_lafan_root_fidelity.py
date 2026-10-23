"""Native-LaFAN source-root fidelity gate layered over existing review gates.

Diagnostic only: no original H5 edits, no candidate promotion or training.
The 10 degree gate is a provisional conservative review threshold, not a
published standard. Global yaw alignment and mirroring are never performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools.booster_k1_retarget import bvh_to_robot_dataset as old
from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import load
from tools.booster_k1_retarget.localize_and_salvage import intervals, padded
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def native_root_targets(meta, length):
    if meta.get("dataset") != "LaFAN1" or meta["source_fps"] != 30:
        raise ValueError(
            "Only checked native LaFAN 30 Hz calibration supported"
        )
    frames, _ = old.load_bvh_file(meta["source_path"], format="lafan1")
    native = Rotation.from_quat(
        np.array([f["Hips"][1] for f in frames]), scalar_first=True
    )
    cfg = json.loads(
        old.K1_BVH_CONFIG_PATH.with_name(
            "bvh_lafan1_to_official_k1_v2.json"
        ).read_text()
    )
    offset = Rotation.from_quat(
        cfg["ik_match_table1"]["trunk"][4], scalar_first=True
    )
    start = meta.get(
        "canonical_frame_start",
        int(meta.get("source_start_frame_trim", 0))
        + int(meta.get("source_frame_start", 0)),
    )
    times = (start + np.arange(length)) / 50
    return (
        Slerp(np.arange(len(frames)) / 30, native)(times) * offset
    ).as_quat()


def orientation_errors(actual, desired):
    actual, desired = np.asarray(actual), np.asarray(desired)
    if (
        actual.shape != desired.shape
        or actual.ndim != 2
        or actual.shape[1] != 4
    ):
        raise ValueError("Matched [T,4] xyzw quaternions required")
    if not np.isfinite(actual).all() or not np.isfinite(desired).all():
        raise ValueError("Nonfinite quaternion")
    return np.degrees(
        (
            Rotation.from_quat(desired).inv() * Rotation.from_quat(actual)
        ).magnitude()
    )


def retained_spans(row, errors, threshold=10.0):
    if row.get("source_error") is not None:
        return []
    if len(errors) != row["frames"]:
        raise ValueError("Frame correspondence mismatch")
    bad = np.asarray(errors) > threshold
    for spans in row["reason_intervals"].values():
        for s, e in spans:
            bad[s:e] = True
    return [(s, e) for s, e in intervals(~padded(bad, 5)) if e - s >= 50]


def review(root):
    info = json.loads((root / "comparison_report.json").read_text())
    inp = json.loads((root / "input/manifest.json").read_text())
    config = json.loads(Path(info["old_config"]).read_text())
    offset = Rotation.from_quat(
        config["ik_match_table1"]["trunk"][4], scalar_first=True
    )
    result = {
        "training_started": False,
        "training_approved": False,
        "provisional_root_threshold_deg": 10,
        "padding_frames": 5,
        "minimum_segment_frames": 50,
        "clips": {},
        "limitations": "Source-calibrated LaFAN only; retain all existing gates; no yaw normalization, stitching, dynamics approval or training activation.",
    }
    for key, c in inp["clips"].items():
        meta = c["metadata"]
        if meta.get("dataset") != "LaFAN1" or meta["source_fps"] != 30:
            raise ValueError(
                "Only checked native LaFAN 30 Hz calibration supported"
            )
        frames, _ = old.load_bvh_file(meta["source_path"], format="lafan1")
        native = Rotation.from_quat(
            np.array([f["Hips"][1] for f in frames]), scalar_first=True
        )
        times = (meta["canonical_frame_start"] + np.arange(c["length"])) / 50
        desired = (
            Slerp(np.arange(len(frames)) / 30, native)(times) * offset
        ).as_quat()
        scores = {}
        for name, report in info["variants"].items():
            row = next(r for r in report["clips"] if r["motion_key"] == key)
            actual = load(root / name / "raw", key)["ref_root_rot"]
            errors = orientation_errors(actual, desired)
            spans = retained_spans(row, errors)
            scores[name] = {
                "frames": len(errors),
                "root_mean_deg": float(errors.mean()),
                "root_p95_deg": float(np.percentile(errors, 95)),
                "root_max_deg": float(errors.max()),
                "root_error_over_10deg_intervals": intervals(errors > 10),
                "existing_candidate_frames": row["candidate_frames"],
                "root_checked_candidate_frames": sum(e - s for s, e in spans),
                "root_checked_candidate_spans": spans,
                "all_frame_arm_direction_p95_deg": row[
                    "source_upper_arm_direction_p95_deg"
                ],
                "all_frame_leg_direction_p95_deg": row[
                    "native_leg_direction_p95_deg"
                ],
            }
        result["clips"][key] = scores
    dump(root / "root_fidelity_review.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    review(p.parse_args().root)


if __name__ == "__main__":
    main()
