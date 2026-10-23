#!/usr/bin/env python3
"""Play Booster K1 HDF5 clips sequentially in the interactive MuJoCo viewer."""

from __future__ import annotations
import argparse
import fnmatch
import json
import time
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
K1_XML_PATH = REPO_ROOT / "assets" / "robots" / "booster_k1" / "K1_22dof.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="HDF5 v2 dataset directory containing manifest.json.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start at this index in the filtered playlist.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Play at most this many clips; 0 plays every selected clip.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Shell-style motion-name filter, for example 'walk*'.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier; 1.0 is real time.",
    )
    parser.add_argument(
        "--pause-between",
        type=float,
        default=0.25,
        help="Seconds to hold the last frame before the next clip.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart the playlist after its last clip.",
    )
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Do not make the viewer camera follow the robot root.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the selected playlist without opening a GUI.",
    )
    return parser.parse_args()


def load_manifest(dataset_root: Path) -> dict[str, object]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def select_motion_keys(
    manifest: dict[str, object],
    *,
    pattern: str,
    start_index: int,
    max_clips: int,
) -> list[str]:
    keys = [
        str(key)
        for key in manifest["motion_keys"]
        if fnmatch.fnmatchcase(str(key), pattern)
    ]
    if not keys:
        raise ValueError(f"no motion keys match pattern {pattern!r}")
    if start_index < 0 or start_index >= len(keys):
        raise ValueError(
            f"start-index must be in [0, {len(keys) - 1}], got {start_index}"
        )
    keys = keys[start_index:]
    if max_clips > 0:
        keys = keys[:max_clips]
    return keys


def load_clip(
    dataset_root: Path,
    manifest: dict[str, object],
    motion_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    clip = manifest["clips"][motion_key]
    shard = manifest["hdf5_shards"][int(clip["shard"])]
    shard_path = dataset_root / shard["file"]
    start = int(clip["start"])
    end = start + int(clip["length"])
    with h5py.File(shard_path, "r") as handle:
        root_pos = np.asarray(handle["ref_root_pos"][start:end])
        root_rot_xyzw = np.asarray(handle["ref_root_rot"][start:end])
        dof_pos = np.asarray(handle["ref_dof_pos"][start:end])
    if root_pos.shape != (end - start, 3):
        raise ValueError(f"invalid root position shape: {root_pos.shape}")
    if root_rot_xyzw.shape != (end - start, 4):
        raise ValueError(f"invalid root rotation shape: {root_rot_xyzw.shape}")
    if dof_pos.shape != (end - start, 22):
        raise ValueError(f"invalid K1 joint position shape: {dof_pos.shape}")
    if not all(
        np.isfinite(array).all()
        for array in (root_pos, root_rot_xyzw, dof_pos)
    ):
        raise ValueError(f"clip contains non-finite values: {motion_key}")
    metadata = clip.get("metadata", {})
    fps = float(metadata.get("motion_fps", manifest.get("target_fps", 50.0)))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid motion FPS for {motion_key}: {fps}")
    return root_pos, root_rot_xyzw, dof_pos, fps


def hold_last_frame(viewer, seconds: float) -> bool:
    deadline = time.perf_counter() + max(0.0, seconds)
    while viewer.is_running() and time.perf_counter() < deadline:
        viewer.sync()
        time.sleep(0.01)
    return viewer.is_running()


def play_clip(
    *,
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    fps: float,
    speed: float,
    fixed_camera: bool,
) -> bool:
    frame_period = 1.0 / (fps * speed)
    next_frame_time = time.perf_counter()
    for frame in range(len(root_pos)):
        if not viewer.is_running():
            return False
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_rot_xyzw[frame, [3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[frame]
        mujoco.mj_forward(model, data)
        if not fixed_camera:
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.cam.lookat[2] += 0.15
        viewer.sync()

        next_frame_time += frame_period
        delay = next_frame_time - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
        elif delay < -0.5:
            next_frame_time = time.perf_counter()
    return viewer.is_running()


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError(f"speed must be positive, got {args.speed}")
    if args.max_clips < 0:
        raise ValueError("max-clips must not be negative")
    dataset_root = args.dataset_root.resolve()
    manifest = load_manifest(dataset_root)
    motion_keys = select_motion_keys(
        manifest,
        pattern=args.pattern,
        start_index=args.start_index,
        max_clips=args.max_clips,
    )
    print(f"Selected {len(motion_keys)} clips from {dataset_root}")
    for index, motion_key in enumerate(motion_keys):
        print(f"  [{index:03d}] {motion_key}")
    if args.list_only:
        return

    model = mujoco.MjModel.from_xml_path(str(K1_XML_PATH))
    if model.nq != 29:
        raise ValueError(f"expected K1 nq=29, got {model.nq}")
    data = mujoco.MjData(model)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -12.0
            while viewer.is_running():
                for playlist_index, motion_key in enumerate(motion_keys):
                    if not viewer.is_running():
                        return
                    root_pos, root_rot, dof_pos, fps = load_clip(
                        dataset_root, manifest, motion_key
                    )
                    duration = len(root_pos) / fps
                    print(
                        f"Playing [{playlist_index + 1}/{len(motion_keys)}] "
                        f"{motion_key}: {len(root_pos)} frames, "
                        f"{duration:.2f}s at {fps:g} FPS"
                    )
                    if not play_clip(
                        viewer=viewer,
                        model=model,
                        data=data,
                        root_pos=root_pos,
                        root_rot_xyzw=root_rot,
                        dof_pos=dof_pos,
                        fps=fps,
                        speed=args.speed,
                        fixed_camera=args.fixed_camera,
                    ):
                        return
                    if not hold_last_frame(viewer, args.pause_between):
                        return
                if not args.loop:
                    print("Playlist finished. Close the viewer to exit.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    return
                print("Restarting playlist...")
    except KeyboardInterrupt:
        print("Playback stopped.")


if __name__ == "__main__":
    main()
