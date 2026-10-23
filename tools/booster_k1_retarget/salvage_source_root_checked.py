"""Source-root checked GMR salvage; non-LaFAN uses intended canonical target.

Canonical agreement is necessary, not proof of native skeleton correctness.
All existing geometric/source direction checks remain active.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget import salvage_repaired_reference as salvage
from tools.booster_k1_retarget.filter_expanded_root_candidates import roots
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.review_lafan_root_fidelity import (
    native_root_targets,
    orientation_errors,
)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output", type=Path, required=True)
    a, _ = p.parse_known_args()
    source, classify = salvage.source_positions, salvage.classify_frames
    active = {}
    details = []
    cfg = json.loads(
        Path(
            "tools/booster_k1_retarget/configs/gmr_smplx_to_official_k1_v2.json"
        ).read_text()
    )
    offset = Rotation.from_quat(
        cfg["ik_match_table1"]["trunk"][4], scalar_first=True
    )

    def with_root(meta, n, converter):
        positions = source(meta, n, converter)
        if meta["dataset"] == "LaFAN1":
            desired = native_root_targets(meta, n)
            origin = "native_lafan_calibrated_root"
        else:
            r = roots(
                meta.get("canonical_path"),
                meta["source_path"],
                meta["dataset"],
            )
            start = meta["canonical_frame_start"]
            if start < 0 or start + n > len(r):
                raise ValueError("Root correspondence mismatch")
            desired = (
                Rotation.from_rotvec(r[start : start + n]) * offset
            ).as_quat()
            origin = (
                "intended_canonical_GMR_root_not_native_adapter_certification"
            )
        active.update(
            desired=desired, origin=origin, source=meta["source_path"]
        )
        return positions

    def with_gate(*args, **kwargs):
        reasons = classify(*args, **kwargs)
        errors = orientation_errors(args[6], active["desired"])
        reasons["source_root_orientation_error_over_10deg"] = errors > 10
        details.append(
            {
                "source_path": active["source"],
                "origin": active["origin"],
                "p95_deg": float(np.percentile(errors, 95)),
                "max_deg": float(errors.max()),
            }
        )
        return reasons

    salvage.source_positions = with_root
    salvage.classify_frames = with_gate
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
                    "threshold_deg_provisional": 10,
                    "clips": details,
                },
            )


if __name__ == "__main__":
    main()
