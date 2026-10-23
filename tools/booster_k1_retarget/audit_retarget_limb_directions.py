"""Sample canonical-human versus K1 limb directions, preserving trim alignment.

Large errors trigger review, not a claim of physically impossible motion.
Neutral SMPL and robot segment anatomy differ. No motion is changed.
"""

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from holosmpl_h5_to_k1_h5 import (
    HoloSmplToGmrConverter,
    K1_XML_PATH,
    K1_IK_CONFIG_PATH,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-roots", nargs="+", type=Path, default=[])
    p.add_argument("--segment-index", type=Path)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--frame-stride",
        type=int,
        default=0,
        help="Positive: check every N frames plus final frame; 0: 24 samples",
    )
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    converter = HoloSmplToGmrConverter(a.smpl_model)
    model = mujoco.MjModel.from_xml_path(str(K1_XML_PATH))
    data = mujoco.MjData(model)
    config = json.loads(K1_IK_CONFIG_PATH.read_text())
    root_offset = Rotation.from_quat(
        config["ik_match_table1"][config["robot_root_name"]][4],
        scalar_first=True,
    )
    limbs = []
    for side in ("left", "right"):
        for first, second, b1, b2 in [
            ("shoulder", "elbow", "shoulder_roll_link", "elbow_pitch_link"),
            ("elbow", "wrist", "elbow_pitch_link", "elbow_yaw_link"),
            ("hip", "knee", "hip_roll_link", "knee_pitch_link"),
            ("knee", "ankle", "knee_pitch_link", "ankle_pitch_link"),
        ]:
            limbs.append(
                (
                    side + "_" + first,
                    side + "_" + second,
                    model.body(side + "_" + b1).id,
                    model.body(side + "_" + b2).id,
                )
            )
    rows = []
    statuses = [
        json.loads(path.read_text())
        for root in a.batch_roots
        for path in sorted(root.glob("*/*/status.json"))
    ]
    if a.segment_index:
        for row in json.loads(a.segment_index.read_text()):
            if row["source"] in {"CMU", "100STYLE"}:
                statuses.append(
                    {
                        "status": "complete",
                        "filtered_root": row["robot_root"],
                        "source": {"CMU": "cmu_asf", "100STYLE": "style100"}[
                            row["source"]
                        ],
                        "group": row["motion"],
                        "motion_keys": [
                            row.get("robot_motion_key", row["motion"])
                        ],
                    }
                )
    for status_group in [statuses]:
        for status in status_group:
            if status["status"] != "complete":
                continue
            folder = Path(status["filtered_root"])
            m = json.loads((folder / "manifest.json").read_text())
            for key, c in m["clips"].items():
                if (
                    status.get("motion_keys")
                    and key not in status["motion_keys"]
                ):
                    continue
                meta = c["metadata"]
                with np.load(meta["canonical_path"], allow_pickle=False) as z:
                    root_aa, body, trans = (
                        z["root_orient"],
                        z["pose_body"],
                        z["trans"],
                    )
                s, n = c["start"], c["length"]
                trim = meta.get("source_start_frame_trim", 0) + meta.get(
                    "source_frame_start", 0
                )
                if a.frame_stride < 0:
                    raise ValueError("frame-stride must be nonnegative")
                indices = (
                    np.unique(np.r_[np.arange(0, n, a.frame_stride), n - 1])
                    if a.frame_stride
                    else np.unique(np.linspace(0, n - 1, 24).astype(int))
                )
                with h5py.File(
                    folder / m["hdf5_shards"][c["shard"]]["file"], "r"
                ) as f:
                    xyz, quat, q = [
                        f[k][s + indices]
                        for k in (
                            "ref_root_pos",
                            "ref_root_rot",
                            "ref_dof_pos",
                        )
                    ]
                errors = {x[0]: [] for x in limbs}
                root_errors = []
                for j, idx in enumerate(indices):
                    i = idx + trim
                    human = converter(
                        pose_aa=np.r_[root_aa[i], body[i]], root_trans=trans[i]
                    )
                    target_root = (
                        Rotation.from_quat(
                            human[config["human_root_name"]][1],
                            scalar_first=True,
                        )
                        * root_offset
                    )
                    root_errors.append(
                        float(
                            np.degrees(
                                (
                                    target_root.inv()
                                    * Rotation.from_quat(quat[j])
                                ).magnitude()
                            )
                        )
                    )
                    data.qpos[:] = np.r_[xyz[j], quat[j, [3, 0, 1, 2]], q[j]]
                    mujoco.mj_forward(model, data)
                    for first, second, b1, b2 in limbs:
                        v = human[second][0] - human[first][0]
                        if second.endswith("_wrist"):
                            # The distal link extends along local +/-Y in the
                            # official K1 XML. Its origin is the elbow, not wrist.
                            sign = 1 if first.startswith("left_") else -1
                            w = data.xmat[b2].reshape(3, 3) @ np.array(
                                [0, sign, 0]
                            )
                        else:
                            w = data.xpos[b2] - data.xpos[b1]
                        cosine = np.dot(v, w) / (
                            np.linalg.norm(v) * np.linalg.norm(w)
                        )
                        errors[first].append(
                            float(
                                np.degrees(np.arccos(np.clip(cosine, -1, 1)))
                            )
                        )
                stats = {
                    k: {
                        "mean_deg": float(np.mean(v)),
                        "p95_deg": float(np.percentile(v, 95)),
                    }
                    for k, v in errors.items()
                }
                flagged = [
                    k
                    for k, v in stats.items()
                    if v["mean_deg"] > 45 or v["p95_deg"] > 75
                ]
                rows.append(
                    {
                        "source": {"cmu_asf": "CMU", "style100": "100STYLE"}[
                            status["source"]
                        ],
                        "motion": Path(meta["source_path"]).stem,
                        "robot_motion_key": key,
                        "robot_root": str(folder),
                        "limbs": stats,
                        "flagged_limbs": flagged,
                        "root_orientation_error_deg": {
                            "mean": float(np.mean(root_errors)),
                            "p95": float(np.percentile(root_errors, 95)),
                        },
                        "samples": len(indices),
                        "trim_frames": trim,
                        "sampled_error_frames": [
                            {
                                "segment_frame": int(idx),
                                "canonical_frame": int(idx + trim),
                                "root_error_deg": root_errors[j],
                                "limbs_over_75deg": [
                                    name
                                    for name, values in errors.items()
                                    if values[j] > 75
                                ],
                            }
                            for j, idx in enumerate(indices)
                            if root_errors[j] > 45
                            or any(
                                values[j] > 75 for values in errors.values()
                            )
                        ],
                    }
                )
            print(status["source"], status["group"], len(rows), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            {
                "method": "synchronized samples, world-frame limb directions; heuristic review gate",
                "frame_stride": a.frame_stride,
                "default_samples": 24,
                "threshold": "mean >45deg OR p95 >75deg",
                "clips": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
