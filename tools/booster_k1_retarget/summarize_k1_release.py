"""Refresh the local preparation inventory; no downloads or training launch."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/holomotion_k1_v1"
PILOT = (
    REPO
    / "logs/HoloMotionK1ReleaseV1Pilot/20261020_093940-motion_tracking_booster_k1_release_v1_pilot"
)


def main():
    report = {
        "release_status": "PREPARATION_AND_SMOKE_TEST_ONLY_NOT_FORMAL_RELEASE",
        "original_data_preserved": True,
        "formal_long_training_started": False,
        "pilot_run": str(PILOT),
        "pilot_config": "motion_tracking_booster_k1_release_v1_pilot",
        "pilot_parent_checkpoint": "logs/HoloMotionBoosterK1LaFANWalkDanceJointRewardSmall/20261016_092811-motion_tracking_booster_k1_lafan_walk_dance_joint_reward_small/model_13000.pt",
        "pilot_iterations": 2,
        "pilot_num_envs": 64,
        "aistpp": {},
        "simulation_samples": [],
        "pending": [
            "100STYLE source-specific skeleton adapter and retarget validation",
            "CMU ASF/AMC adapter, per-clip FPS verification and action categorization",
            "Broader visual/dynamic QA before freezing a formal training dataset",
            "Full held-out validation and deployment parity checks",
        ],
    }
    for split in ("train", "val", "test"):
        root = ROOT / "filtered_robot" / f"aistpp_{split}"
        manifest = json.loads((root / "manifest.json").read_text())
        rows = json.loads((root / "preparation_report.json").read_text())
        report["aistpp"][split] = {
            "converted": len(rows),
            "initial_numerical_pass": len(manifest["motion_keys"]),
            "deferred": len(rows) - len(manifest["motion_keys"]),
            "frames": sum(c["length"] for c in manifest["clips"].values()),
            "reasons_nonexclusive": dict(
                Counter(reason for row in rows for reason in row["reasons"])
            ),
            "root": str(root),
        }
    report["aistpp"]["post_audit_excluded"] = ["gJB_sBM_cAll_d07_mJB2_ch01"]
    report["aistpp"]["eligible_train_after_post_audit"] = 135
    report["moonwalk"] = {
        "root": str(ROOT / "filtered_robot/moonwalk_v14"),
        "clips": 1,
        "frames": 500,
        "synthetic_repeats": True,
        "note": "Repeated repaired short reference, not independent 10-second capture; dynamic success not established.",
    }
    cmu = ROOT / "raw/cmu"
    report["cmu"] = {
        "amc_files": len(list(cmu.rglob("*.amc"))),
        "asf_files": len(list(cmu.rglob("*.asf"))),
        "status": "RAW_EXTRACTED_NOT_K1_RETARGETED",
        "new_download_bytes": 0,
    }
    style_receipt = ROOT / "downloads/100STYLE.download.json"
    if style_receipt.exists():
        report["style100"] = json.loads(style_receipt.read_text())
        report["style100"]["extracted_bvh"] = len(
            list((ROOT / "raw/100style").rglob("*.bvh"))
        )
    else:
        parts = list((ROOT / "downloads/100style_drive_parts").glob("*.part"))
        report["style100"] = {
            "status": "DOWNLOADING",
            "stored_range_bytes": sum(p.stat().st_size for p in parts),
        }
    for path in sorted(PILOT.glob("isaaclab_eval_output_model_1_*/*.npz")):
        with np.load(path, allow_pickle=False) as clip:
            meta = json.loads(str(clip["metadata"]))
            error = np.abs(clip["robot_dof_pos"] - clip["ref_dof_pos"])
            report["simulation_samples"].append(
                {
                    **meta,
                    "joint_position_mae_rad": float(error.mean()),
                    "joint_position_mae_legs_rad": float(error[:, 10:].mean()),
                    "path": str(path),
                    "note": "Valid recorded prefix; not a full-dataset completion metric.",
                }
            )
    destination = ROOT / "reports/preparation_status.json"
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
