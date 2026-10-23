#!/usr/bin/env python3
"""Render one Booster K1 HDF5 v2 clip to MP4 with MuJoCo."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
K1_XML_PATH = REPO_ROOT / "assets" / "robots" / "booster_k1" / "K1_22dof.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    motion_key = manifest["motion_keys"][args.clip_index]
    clip = manifest["clips"][motion_key]
    shard_path = (
        args.dataset_root / manifest["hdf5_shards"][clip["shard"]]["file"]
    )
    start = int(clip["start"])
    length = int(clip["length"])
    if args.max_frames > 0:
        length = min(length, args.max_frames)
    with h5py.File(shard_path, "r") as handle:
        root_pos = np.asarray(handle["ref_root_pos"][start : start + length])
        root_rot_xyzw = np.asarray(
            handle["ref_root_rot"][start : start + length]
        )
        dof_pos = np.asarray(handle["ref_dof_pos"][start : start + length])

    model = mujoco.MjModel.from_xml_path(str(K1_XML_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 2.2
    camera.azimuth = 135.0
    camera.elevation = -12.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fps = float(clip["metadata"]["motion_fps"])
    with imageio.get_writer(args.output, fps=fps, codec="libx264") as writer:
        for frame in range(length):
            data.qpos[:3] = root_pos[frame]
            data.qpos[3:7] = root_rot_xyzw[frame, [3, 0, 1, 2]]
            data.qpos[7:] = dof_pos[frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] += 0.15
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    renderer.close()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "motion_key": motion_key,
                "frames": length,
                "fps": fps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
