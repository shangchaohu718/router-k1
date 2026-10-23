"""Bounded ablation of clip-start IK seeds; does not replace any dataset."""

import json
from pathlib import Path

import numpy as np

from holosmpl_h5_to_k1_h5 import HoloSmplToGmrConverter, create_k1_retargeter

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/holomotion_k1_v1"


def main():
    out = ROOT / "arm_repair_v1/probe"
    out.mkdir(parents=True, exist_ok=False)
    converter = HoloSmplToGmrConverter(
        Path(
            "/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
        )
    )
    samples = [
        ("style100", "Neutral", "Neutral_FR"),
        ("style100", "Neutral", "Neutral_FW"),
        ("style100", "Akimbo", "Akimbo_FR"),
        ("cmu_asf", "02", "02_03"),
    ]
    reports = []
    for source, group, name in samples:
        canonical = (
            ROOT / "batch_adaptation_v1" / source / group / "human/canonical"
        )
        m = json.loads((canonical / "manifest.json").read_text())
        entry = next(
            x for x in m["samples"] if x["clip_id"].startswith(name + "__")
        )
        with np.load(canonical / entry["canonical_path"]) as z:
            pose = np.c_[z["root_orient"], z["pose_body"]]
            trans = z["trans"]
        for mode in ("default", "root", "relaxed_arms", "root_relaxed_arms"):
            r = create_k1_retargeter(actual_human_height=1.8)
            h = converter(pose_aa=pose[0], root_trans=trans[0])
            r.update_targets(h)
            seed = r.configuration.data.qpos.copy()
            if mode.startswith("root"):
                seed[:3], seed[3:7] = r.scaled_human_data["pelvis"]
            if "relaxed_arms" in mode:
                for side, roll in [("left", -1.3), ("right", 1.3)]:
                    seed[
                        r.model.jnt_qposadr[
                            r.model.joint(side + "_shoulder_roll_joint").id
                        ]
                    ] = roll
            r.configuration.update(seed)
            errors = []
            qs = []
            for i in range(min(300, len(pose))):
                h = converter(pose_aa=pose[i], root_trans=trans[i])
                original = {
                    k: [v[0].copy(), v[1].copy()] for k, v in h.items()
                }
                q = r.retarget(h)
                qs.append(q)
                if i < 10:
                    continue
                frame = []
                for side, sign in [("left", 1), ("right", -1)]:
                    u = (
                        original[side + "_elbow"][0]
                        - original[side + "_shoulder"][0]
                    )
                    u /= np.linalg.norm(u)
                    bid = r.model.body(side + "_shoulder_roll_link").id
                    v = r.configuration.data.xmat[bid].reshape(
                        3, 3
                    ) @ np.array([0, sign, 0])
                    frame.append(
                        float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))
                    )
                errors.append(frame)
            report = {
                "motion": name,
                "mode": mode,
                "upper_arm_mean_deg": np.mean(errors, axis=0).tolist(),
                "upper_arm_p95_deg": np.percentile(
                    errors, 95, axis=0
                ).tolist(),
            }
            reports.append(report)
            np.savez_compressed(
                out / (name + "_" + mode + ".npz"), qpos=np.array(qs)
            )
            print(report, flush=True)
    (out / "report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
