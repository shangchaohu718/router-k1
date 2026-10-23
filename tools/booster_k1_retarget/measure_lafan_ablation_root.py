"""Root orientation fidelity to native BVH, independent of candidate counts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools.booster_k1_retarget import bvh_to_robot_dataset as old
from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import load
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.root / "comparison_report.json").read_text())
    inp = json.loads((a.root / "input/manifest.json").read_text())
    config = json.loads(Path(info["old_config"]).read_text())
    offset = Rotation.from_quat(
        config["ik_match_table1"]["trunk"][4], scalar_first=True
    )
    result = {
        "training_approved": False,
        "metric": "Angular distance of robot trunk/root quaternion to calibrated native BVH Hips target; same source times, no yaw normalization",
        "clips": {},
    }
    start = info.get("start_frame_50hz", 0)
    for key, c in inp["clips"].items():
        frames, _ = old.load_bvh_file(
            c["metadata"]["source_path"], format="lafan1"
        )
        rotations = Rotation.from_quat(
            np.array([f["Hips"][1] for f in frames]), scalar_first=True
        )
        desired = (
            Slerp(np.arange(len(frames)) / 30, rotations)(
                (start + np.arange(500)) / 50
            )
            * offset
        )
        scores = {}
        for variant in info["variants"]:
            arrays = load(a.root / variant / "raw", key)
            actual = Rotation.from_quat(arrays["ref_root_rot"])
            error = np.degrees((desired.inv() * actual).magnitude())
            tilt = np.degrees(
                np.arccos(np.clip(actual.as_matrix()[:, 2, 2], -1, 1))
            )
            scores[variant] = {
                "root_orientation_error_mean_deg": float(error.mean()),
                "root_orientation_error_p95_deg": float(
                    np.percentile(error, 95)
                ),
                "root_tilt_max_deg": float(tilt.max()),
            }
        result["clips"][key] = scores
    dump(a.root / "root_source_tracking.json", result)


if __name__ == "__main__":
    main()
