"""LaFAN-only source-root checked salvage, without modifying shared workers."""

import argparse
from pathlib import Path

import numpy as np

from tools.booster_k1_retarget import salvage_repaired_reference as salvage
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.review_lafan_root_fidelity import (
    native_root_targets,
    orientation_errors,
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    a, _ = parser.parse_known_args()
    source, classify = salvage.source_positions, salvage.classify_frames
    active = {}
    details = []

    def with_root(meta, length, converter):
        positions = source(meta, length, converter)
        active["desired"] = native_root_targets(meta, length)
        active["source_path"] = meta["source_path"]
        active["source_start"] = meta.get("canonical_frame_start", 0)
        return positions

    def classify_root(*args, **kwargs):
        reasons = classify(*args, **kwargs)
        errors = orientation_errors(args[6], active["desired"])
        reasons["source_root_orientation_error_over_10deg"] = errors > 10
        details.append(
            {
                "source_path": active["source_path"],
                "canonical_frame_start": active["source_start"],
                "root_error_mean_deg": float(errors.mean()),
                "root_error_p95_deg": float(np.percentile(errors, 95)),
                "root_error_max_deg": float(errors.max()),
                "frames_over_10deg": int(np.sum(errors > 10)),
            }
        )
        return reasons

    salvage.source_positions = with_root
    salvage.classify_frames = classify_root
    try:
        salvage.main()
    finally:
        salvage.source_positions = source
        salvage.classify_frames = classify
        if a.output.exists():
            dump(
                a.output / "root_fidelity_details.json",
                {
                    "training_started": False,
                    "training_approved": False,
                    "provisional_threshold_deg": 10,
                    "scope": "Native LaFAN 30Hz with explicit calibrated target; existing gates unchanged; no bulk promotion",
                    "clips": details,
                },
            )


if __name__ == "__main__":
    main()
