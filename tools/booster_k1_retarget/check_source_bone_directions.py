"""Compare source FK limb directions with canonical SMPL FK (shape-independent)."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from holosmpl.converters.bvh_family.cmu_asf import load_cmu_fk
from holosmpl.converters.bvh_family.lafan1 import (
    _forward_kinematics,
    _local_rotations_and_positions,
)
from holosmpl.converters.bvh_family.noitom import load_bvh_motion
from holosmpl.converters.bvh_family.style100 import YUP_TO_ZUP
from holosmpl_h5_to_k1_h5 import HoloSmplToGmrConverter

LIMBS = [
    ("left_hip", "left_knee", "lfemur", "LeftHip", "LeftKnee"),
    ("right_hip", "right_knee", "rfemur", "RightHip", "RightKnee"),
    ("left_knee", "left_ankle", "ltibia", "LeftKnee", "LeftAnkle"),
    ("right_knee", "right_ankle", "rtibia", "RightKnee", "RightAnkle"),
    ("left_shoulder", "left_elbow", "lhumerus", "LeftShoulder", "LeftElbow"),
    (
        "right_shoulder",
        "right_elbow",
        "rhumerus",
        "RightShoulder",
        "RightElbow",
    ),
    ("left_elbow", "left_wrist", "lradius", "LeftElbow", "LeftWrist"),
    ("right_elbow", "right_wrist", "rradius", "RightElbow", "RightWrist"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    converter = HoloSmplToGmrConverter(a.smpl_model)
    rows = []
    manifest = json.loads((a.canonical / "manifest.json").read_text())
    for sample in manifest["samples"]:
        if sample.get("status") != "ok":
            continue
        with np.load(
            a.canonical / sample["canonical_path"], allow_pickle=False
        ) as z:
            root = z["root_orient"]
            body = z["pose_body"]
            trans = z["trans"]
            meta = json.loads(str(z["metadata"]))
        path = Path(meta["source_path"])
        fps = meta["source_fps"]
        offset = meta.get("trim_start", 0)
        if path.suffix == ".amc":
            _, start, end, _ = load_cmu_fk(path)
            vectors = {
                item[0]: end[item[2]] - start[item[2]] for item in LIMBS
            }
        else:
            motion = load_bvh_motion(path)
            lr, lp = _local_rotations_and_positions(motion)
            _, pos = _forward_kinematics(motion, lr, lp)
            vectors = {item[0]: pos[item[4]] - pos[item[3]] for item in LIMBS}
        errors = []
        for i in np.linspace(0, len(root) - 2, 12, dtype=int):
            source_index = min(
                int(round(i * fps / 50)) + offset,
                len(next(iter(vectors.values()))) - 1,
            )
            human = converter(
                pose_aa=np.r_[root[i], body[i]], root_trans=trans[i]
            )
            for first, second, *_ in LIMBS:
                v = YUP_TO_ZUP.apply(vectors[first][source_index])
                w = human[second][0] - human[first][0]
                cosine = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
                errors.append(
                    float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
                )
        rows.append(
            {
                "motion": sample["clip_id"],
                "mean_deg": float(np.mean(errors)),
                "p95_deg": float(np.percentile(errors, 95)),
                "max_deg": max(errors),
            }
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            {
                "check": "source_vs_canonical_limb_direction_not_dynamics",
                "clips": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
