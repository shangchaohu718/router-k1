"""Create immutable CPU evaluation subsets and a synthetic switch fixture.

The switch reference interpolates existing K1 joint/root tracks. It is a
transition stress test, not a newly retargeted or physically certified motion.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from holomotion.src.training.data_production.robot_h5 import (
    ROBOT_H5_ARRAY_NAMES,
    RobotH5ShardWriter,
)


def read_clip(root, key, count, offset=0):
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["clips"][key]
    shard = Path(manifest["hdf5_shards"][entry["shard"]]["file"])
    if not shard.is_absolute():
        shard = root / shard
    if offset + count > entry["length"]:
        raise ValueError("Requested interval exceeds source clip")
    start = int(entry["start"]) + offset
    with h5py.File(shard, "r") as handle:
        arrays = {
            name: handle[name][start : start + count].copy()
            for name in manifest.get("array_names", ROBOT_H5_ARRAY_NAMES)
        }
    return (
        manifest,
        arrays,
        {
            "manifest": str(root / "manifest.json"),
            "motion_key": key,
            "offset": offset,
            "frames": count,
        },
    )


def transition_arrays(
    slide, neutral, hold_before=100, hold_after=200, blend=50
):
    n = len(slide["ref_dof_pos"])
    if 2 * blend >= n or blend < 2:
        raise ValueError("Need a nonempty unmodified slide interval")
    alpha = np.ones(n)
    u = np.linspace(0, 1, blend)
    ramp = 6 * u**5 - 15 * u**4 + 10 * u**3
    alpha[:blend], alpha[-blend:] = ramp, ramp[::-1]
    result = {}
    for name in ("ref_dof_pos", "ref_root_pos"):
        before = neutral[name][0].copy()
        after = before.copy()
        if name == "ref_root_pos":
            before[:2] = slide[name][0, :2]
            after[:2] = slide[name][-1, :2]
        resting = np.where((np.arange(n) < n // 2)[:, None], before, after)
        middle = resting * (1 - alpha[:, None]) + slide[name] * alpha[:, None]
        result[name] = np.concatenate(
            [
                np.repeat(before[None], hold_before, 0),
                middle,
                np.repeat(after[None], hold_after, 0),
            ]
        ).astype(np.float32)
    # Robot H5 uses xyzw quaternions (the source NPZ importer converts wxyz).
    # Interpolate on SO(3), never componentwise.
    rotations = slide["ref_root_rot"]
    middle_rot = []
    for i, a in enumerate(alpha):
        rest = rotations[0 if i < n // 2 else -1]
        pair = np.stack([rest, rotations[i]])
        q = Slerp([0, 1], Rotation.from_quat(pair))([a]).as_quat()[0]
        middle_rot.append(q)
    result["ref_root_rot"] = np.concatenate(
        [
            np.repeat(rotations[:1], hold_before, 0),
            np.array(middle_rot),
            np.repeat(rotations[-1:], hold_after, 0),
        ]
    ).astype(np.float32)
    return result, [hold_before + blend, hold_before + n - blend]


def write_dataset(destination, template, rows):
    destination.mkdir(parents=True, exist_ok=False)
    writer = RobotH5ShardWriter(
        destination / "shards/k1_000000.h5",
        dof_pos_dim=len(template["dof_names"]),
        compression="lzf",
        chunks_t=256,
    )
    manifest = {
        k: copy.deepcopy(v)
        for k, v in template.items()
        if k not in {"clips", "motion_keys", "hdf5_shards"}
    }
    manifest.update(root=str(destination.resolve()), clips={}, motion_keys=[])
    for i, (key, arrays, provenance) in enumerate(rows):
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Fixture contains nonfinite values")
        metadata = {
            "motion_fps": 50.0,
            "target_fps": 50.0,
            "evaluation_only": True,
            "provenance": provenance,
        }
        start, length = writer.append_motion(
            motion_id=i, arrays=arrays, metadata_json=json.dumps(metadata)
        )
        manifest["motion_keys"].append(key)
        manifest["clips"][key] = dict(
            motion_key=key,
            shard=0,
            clip_idx=i,
            start=start,
            length=length,
            metadata=metadata,
        )
    shard = writer.finalize()
    shard["file"] = "shards/k1_000000.h5"
    manifest["hdf5_shards"] = [shard]
    manifest["quality_status"] = (
        "evaluation_fixture_only_not_physics_certified"
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return hashlib.sha256(
        (destination / shard["file"]).read_bytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = Path("data/holomotion_k1_v1/filtered_robot")
    val = base / "lafan_release_val"
    manifest = json.loads((val / "manifest.json").read_text())
    rows = []
    for key in sorted(manifest["clips"]):
        _, arrays, provenance = read_clip(val, key, 500, offset=100)
        rows.append((key + "__router_holdout_100_600", arrays, provenance))
    old_hash = write_dataset(args.output_dir / "new_old", manifest, rows)
    slide_manifest, slide, slide_source = read_clip(
        base / "moonwalk_v14", "moonwalk_alex61_64_v14_synthetic", 500
    )
    neutral_root = Path(
        "test_outputs/current3998_regression_20261021/reference_subset"
    )
    neutral_manifest = json.loads((neutral_root / "manifest.json").read_text())
    neutral_key = next(
        k for k in neutral_manifest["clips"] if k.startswith("Neutral")
    )
    _, neutral, neutral_source = read_clip(neutral_root, neutral_key, 1)
    if neutral_manifest["dof_names"] != slide_manifest["dof_names"]:
        raise ValueError("Joint order mismatch between fixture sources")
    mixed, interval = transition_arrays(slide, neutral)
    mixed_hash = write_dataset(
        args.output_dir / "switch",
        slide_manifest,
        [
            (
                "router_transition_stand_slide_stand",
                mixed,
                {
                    "slide": slide_source,
                    "neutral": neutral_source,
                    "unmodified_slide_interval": interval,
                    "blend_frames": 50,
                },
            )
        ],
    )
    report = {
        "new_old_clips": 5,
        "frames_per_old_clip": 500,
        "old_shard_sha256": old_hash,
        "switch_shard_sha256": mixed_hash,
        "switch_frames": len(mixed["ref_dof_pos"]),
        "slide_interval": interval,
        "limitations": (
            "Held-out ordinary motion groups for this gate only. Transition "
            "reference is synthetic and not physically certified; a failure "
            "may reflect reference quality as well as controller switching. "
            "No unseen sliding motion was available in this fixture."
        ),
    }
    (args.output_dir / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
