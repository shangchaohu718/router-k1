"""Isolated CMU foot-frame correction from native ankle/toe/toe-tip geometry.

Does not force a world-up sole or pin feet to ground. Rotating the source
rotates the entire reconstructed frame, including deliberately inverted feet.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from holosmpl.converters.bvh_family.cmu_asf import load_cmu_fk
from holosmpl.converters.bvh_family.style100 import YUP_TO_ZUP
from tools.booster_k1_retarget import probe_native_position_retarget as probe
from tools.booster_k1_retarget import retry_guarded_reference as guarded


def foot_frame(ankle, toe, tip):
    points = np.asarray([ankle, toe, tip], dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("Non-finite native foot points")
    forward = points[2] - points[1]
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("Degenerate toe bone")
    forward /= np.linalg.norm(forward)
    lateral = np.cross(points[0] - points[1], forward)
    if np.linalg.norm(lateral) < 1e-6:
        raise ValueError("Collinear native foot points")
    lateral /= np.linalg.norm(lateral)
    up = np.cross(forward, lateral)
    return Rotation.from_matrix(np.stack([forward, lateral, up], axis=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a, _ = p.parse_known_args()
    m = json.loads((a.input / "manifest.json").read_text())
    if len(m["clips"]) != 1:
        raise ValueError("One CMU clip per isolated retry")
    clip = next(iter(m["clips"].values()))
    meta = clip["metadata"]
    path = Path(meta["source_path"])
    if path.suffix.lower() != ".amc":
        raise ValueError(
            "Native foot-frame adapter only validated for CMU ASF/AMC"
        )
    _, start, end, _ = load_cmu_fk(path)
    t = (
        (probe.canonical_start(meta) + np.arange(clip["length"]))
        * float(meta["source_fps"])
        / 50
    )
    poses = {}
    for side, prefix in [("left", "l"), ("right", "r")]:
        samples = []
        for values in (
            start[prefix + "foot"],
            start[prefix + "toes"],
            end[prefix + "toes"],
        ):
            v = YUP_TO_ZUP.apply(values)
            samples.append(
                np.stack(
                    [
                        np.interp(t, np.arange(len(v)), v[:, axis])
                        for axis in range(3)
                    ],
                    axis=1,
                )
            )
        poses[side] = [foot_frame(*points) for points in zip(*samples)]
    original = probe.proportion_lower_body
    count = 0
    differences = []

    def with_native_feet(human, retargeter, scale, rest_data):
        nonlocal count
        result = original(human, retargeter, scale, rest_data)
        for side in ("left", "right"):
            target = poses[side][count]
            offset = retargeter.rot_offsets1[side + "_foot"]
            old = (
                Rotation.from_quat(
                    result[side + "_ankle"][1], scalar_first=True
                )
                * offset
            )
            differences.append(
                float(np.degrees((target.inv() * old).magnitude()))
            )
            result[side + "_ankle"][1] = (target * offset.inv()).as_quat(
                scalar_first=True
            )
        count += 1
        return result

    probe.proportion_lower_body = with_native_feet
    try:
        guarded.main()
        manifest_path = a.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for c in manifest["clips"].values():
            c["metadata"].update(
                foot_orientation_source="native_cmu_ankle_toe_tip_plane_v1",
                training_approved=False,
            )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        probe.proportion_lower_body = original
        if a.output.exists():
            report = {
                "method": "Native CMU ankle, toe joint and toe tip define a rigid anatomical foot frame. No world-up replacement, ground pinning or changed robot limits.",
                "processed_frames": count,
                "mean_change_from_approximate_smpl_foot_deg": float(
                    np.mean(differences)
                )
                if differences
                else None,
                "training_started": False,
                "training_approved": False,
            }
            (a.output / "native_foot_frame_report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
