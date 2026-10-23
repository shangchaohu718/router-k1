#!/usr/bin/env python3
"""Desktop viewer for K1 training-reference and policy-rollout comparisons."""

from __future__ import annotations
import argparse
import json
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import cv2
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_XML = (
    REPO_ROOT / "assets" / "robots" / "booster_k1" / "K1_22dof.xml"
)
REQUIRED_POLICY_KEYS = (
    "robot_dof_pos",
    "robot_global_translation",
    "robot_global_rotation_quat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-npz-dir",
        type=Path,
        required=True,
        help=(
            "Isaac Lab offline-evaluation directory containing one NPZ per "
            "clip."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Training HDF5 v2 dataset root containing manifest.json.",
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=DEFAULT_ROBOT_XML,
        help="Booster K1 MuJoCo XML.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=(
            "MP4 output directory (default: "
            "<policy-npz-dir>/comparison_videos)."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--title", default="Booster K1 训练动作 / 当前策略对比"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help=(
            "Validate inputs and print available clips without opening a "
            "window."
        ),
    )
    return parser.parse_args()


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    value = np.asarray(raw).reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return dict(json.loads(value))
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported NPZ metadata type: {type(value)}")


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("._")
    return cleaned or "clip"


def align_initial_heading(
    positions: np.ndarray, rotations_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ONE rigid yaw rotation to the entire clip, without reflection.

    Align initial forward direction to +X and initial XY to the origin.
    Keep height, roll/pitch, later turns and all joint angles unchanged.
    """
    rotations = Rotation.from_quat(rotations_xyzw)
    forward = rotations[0].apply([1.0, 0.0, 0.0])
    yaw = np.arctan2(forward[1], forward[0])
    correction = Rotation.from_rotvec([0.0, 0.0, -yaw])
    origin = np.array([positions[0, 0], positions[0, 1], 0.0])
    return correction.apply(positions - origin), (
        correction * rotations
    ).as_quat()


class TrainingDataset:
    """Read full K1 reference qpos arrays from an HDF5 v2 dataset."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing dataset manifest: {manifest_path}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.clips = self.manifest["clips"]
        self.shards = self.manifest["hdf5_shards"]

    def _resolve_key(self, candidates: list[str]) -> str:
        for candidate in candidates:
            if candidate in self.clips:
                return candidate
        normalized = {
            key.replace("/", "+").replace(" ", "_"): key for key in self.clips
        }
        for candidate in candidates:
            if candidate in normalized:
                return normalized[candidate]
        raise KeyError(
            "Clip is missing from the training dataset. Tried: "
            + ", ".join(candidates)
        )

    def load_reference(
        self, candidates: list[str]
    ) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
        motion_key = self._resolve_key(candidates)
        clip = self.clips[motion_key]
        shard = self.shards[int(clip["shard"])]
        shard_path = self.root / shard["file"]
        start = int(clip["start"])
        stop = start + int(clip["length"])
        with h5py.File(shard_path, "r") as handle:
            arrays = {
                "root_pos": np.asarray(handle["ref_root_pos"][start:stop]),
                "root_rot": np.asarray(handle["ref_root_rot"][start:stop]),
                "dof_pos": np.asarray(handle["ref_dof_pos"][start:stop]),
            }
        return motion_key, dict(clip.get("metadata", {})), arrays


class PolicyClipIndex:
    """Index policy rollout NPZs by their metadata motion key."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Policy NPZ directory does not exist: {root}"
            )
        self.records: list[dict[str, Any]] = []
        selection_path = self.root / "comparison_selection.json"
        labels = {}
        descriptions = {}
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            labels = {
                r["motion_key"]: r["category_label"]
                for r in selection["clips"]
            }
            descriptions = {
                r["motion_key"]: str(r.get("description") or "")
                for r in selection["clips"]
            }
        for path in sorted(self.root.glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    metadata = _parse_metadata(archive.get("metadata"))
                    missing = [
                        key
                        for key in REQUIRED_POLICY_KEYS
                        if key not in archive
                    ]
                    if missing:
                        continue
                motion_key = str(metadata.get("motion_key", path.stem))
                raw_motion_key = str(
                    metadata.get("raw_motion_key", motion_key)
                )
                self.records.append(
                    {
                        "path": path,
                        "motion_key": motion_key,
                        "raw_motion_key": raw_motion_key,
                        "metadata": metadata,
                        "label": labels.get(raw_motion_key, ""),
                        "description": descriptions.get(raw_motion_key, ""),
                    }
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        self.records.sort(key=lambda r: (r["label"], r["raw_motion_key"]))
        if not self.records:
            raise ValueError(
                "No policy rollout NPZs with "
                f"{REQUIRED_POLICY_KEYS} found in {root}"
            )

    def load(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        with np.load(record["path"], allow_pickle=False) as archive:
            return {
                key: np.asarray(archive[key]) for key in REQUIRED_POLICY_KEYS
            }


class ComparisonRenderer:
    def __init__(self, robot_xml: Path, width: int, height: int):
        if not robot_xml.is_file():
            raise FileNotFoundError(f"Missing K1 MuJoCo XML: {robot_xml}")
        self.model = mujoco.MjModel.from_xml_path(str(robot_xml.resolve()))
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), height
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width = width
        self.height = height
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 2.2
        self.camera.azimuth = 135.0
        self.camera.elevation = -12.0

    def render_pose(
        self,
        root_pos: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot_xyzw[[3, 0, 1, 2]]
        self.data.qpos[7:] = dof_pos
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = root_pos
        self.camera.lookat[2] += 0.15
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _label_frame(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 64), (20, 20, 20), -1)
    cv2.putText(
        result,
        title,
        (18, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        subtitle,
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return result


class ComparisonViewer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset = TrainingDataset(args.dataset_root)
        self.policy_index = PolicyClipIndex(args.policy_npz_dir)
        self.renderer = ComparisonRenderer(
            args.robot_xml, args.width, args.height
        )
        self.record_dir = (
            args.record_dir.resolve()
            if args.record_dir is not None
            else args.policy_npz_dir.resolve() / "comparison_videos"
        )

        self.root = tk.Tk()
        self.root.title(args.title)
        self.root.geometry("1540x860")
        self.root.minsize(1100, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.filtered_records = list(self.policy_index.records)
        self.current_record: dict[str, Any] | None = None
        self.reference: dict[str, np.ndarray] | None = None
        self.policy: dict[str, np.ndarray] | None = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 50.0
        self.playing = False
        self.recording = False
        self.video_writer: cv2.VideoWriter | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._slider_updating = False

        self._build_ui()
        self._refresh_clip_list()
        if self.filtered_records:
            self.clip_list.selection_set(0)
            self._select_record(0)

    def _build_ui(self) -> None:
        sidebar = ttk.Frame(self.root, padding=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(sidebar, text="训练切片", font=("Sans", 14, "bold")).pack(
            anchor=tk.W, pady=(0, 8)
        )
        self.search_var = tk.StringVar()
        search = ttk.Entry(sidebar, textvariable=self.search_var, width=32)
        search.pack(fill=tk.X, pady=(0, 8))
        search.bind("<KeyRelease>", lambda _event: self._refresh_clip_list())

        list_frame = ttk.Frame(sidebar)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.clip_list = tk.Listbox(
            list_frame,
            width=38,
            height=35,
            exportselection=False,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.clip_list.yview)
        self.clip_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.clip_list.bind("<<ListboxSelect>>", self._on_clip_selected)

        self.info_var = tk.StringVar(value="尚未选择切片")
        ttk.Label(
            sidebar,
            textvariable=self.info_var,
            wraplength=275,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(10, 0))

        content = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(content, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(content)
        controls.pack(fill=tk.X, pady=(8, 0))
        self.play_button = ttk.Button(
            controls, text="播放", command=self.toggle_play
        )
        self.play_button.pack(side=tk.LEFT)
        ttk.Button(controls, text="回到开头", command=self.restart).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        self.record_button = ttk.Button(
            controls, text="开始录像", command=self.toggle_recording
        )
        self.record_button.pack(side=tk.LEFT, padx=(16, 0))
        ttk.Button(
            controls, text="导出完整切片", command=self.export_full_clip
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.frame_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(
            controls,
            from_=0,
            to=1,
            variable=self.frame_var,
            command=self._on_slider,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=16)
        self.frame_label = ttk.Label(controls, text="0 / 0")
        self.frame_label.pack(side=tk.RIGHT)

    def _refresh_clip_list(self) -> None:
        query = self.search_var.get().strip().lower()
        self.filtered_records = [
            record
            for record in self.policy_index.records
            if query in record["motion_key"].lower()
            or query in record["raw_motion_key"].lower()
            or query in record["label"].lower()
            or query in record["description"].lower()
        ]
        self.clip_list.delete(0, tk.END)
        for record in self.filtered_records:
            label = record["label"]
            self.clip_list.insert(
                tk.END,
                f"{label} | {record['raw_motion_key']}"
                if label
                else record["raw_motion_key"],
            )

    def _on_clip_selected(self, _event: tk.Event) -> None:
        selected = self.clip_list.curselection()
        if selected:
            self._select_record(int(selected[0]))

    def _select_record(self, index: int) -> None:
        self._stop_recording()
        self.playing = False
        self.play_button.config(text="播放")
        record = self.filtered_records[index]
        policy = self.policy_index.load(record)
        candidates = [
            record["motion_key"],
            record["raw_motion_key"],
            record["path"].stem,
        ]
        try:
            motion_key, ref_meta, reference = self.dataset.load_reference(
                candidates
            )
        except (KeyError, OSError, ValueError) as exc:
            messagebox.showerror("无法加载训练参考", str(exc))
            return

        self.current_record = record
        reference["root_pos"], reference["root_rot"] = align_initial_heading(
            reference["root_pos"], reference["root_rot"]
        )
        policy = {key: value.copy() for key, value in policy.items()}
        (
            policy["robot_global_translation"][:, 0],
            policy["robot_global_rotation_quat"][:, 0],
        ) = align_initial_heading(
            policy["robot_global_translation"][:, 0],
            policy["robot_global_rotation_quat"][:, 0],
        )
        self.reference = reference
        self.policy = policy
        self.current_frame = 0
        self.total_frames = int(reference["dof_pos"].shape[0])
        self.fps = float(
            ref_meta.get(
                "motion_fps", record["metadata"].get("motion_fps", 50.0)
            )
        )
        policy_frames = int(policy["robot_dof_pos"].shape[0])
        self._slider_updating = True
        self.slider.configure(to=max(1, self.total_frames - 1))
        self.frame_var.set(0)
        self._slider_updating = False
        self.info_var.set(
            f"切片：{motion_key}\n"
            f"参考：{self.total_frames} 帧\n"
            f"策略：{policy_frames} 帧\n"
            f"帧率：{self.fps:g} Hz\n"
            f"类别：{record['label']}\n"
            f"动作说明：{record['description']}\n"
            "两侧初始朝向已对齐（保留后续转身）"
        )
        self._show_current_frame()

    def _render_current_frame(self) -> np.ndarray:
        if self.reference is None or self.policy is None:
            raise RuntimeError("No clip selected")
        ref_idx = min(
            self.current_frame, self.reference["dof_pos"].shape[0] - 1
        )
        policy_len = self.policy["robot_dof_pos"].shape[0]
        policy_idx = min(self.current_frame, policy_len - 1)

        ref_frame = self.renderer.render_pose(
            self.reference["root_pos"][ref_idx],
            self.reference["root_rot"][ref_idx],
            self.reference["dof_pos"][ref_idx],
        )
        robot_pos = self.policy["robot_global_translation"][policy_idx, 0]
        robot_rot = self.policy["robot_global_rotation_quat"][policy_idx, 0]
        policy_frame = self.renderer.render_pose(
            robot_pos,
            robot_rot,
            self.policy["robot_dof_pos"][policy_idx],
        )
        elapsed = self.current_frame / self.fps
        ref_frame = _label_frame(
            ref_frame,
            "REFERENCE MOTION",
            f"frame {ref_idx + 1}/{self.total_frames}   t={elapsed:.2f}s",
        )
        policy_status = (
            "POLICY ROLLOUT ENDED - holding last state"
            if self.current_frame >= policy_len
            else f"frame {policy_idx + 1}/{policy_len}   t={elapsed:.2f}s"
        )
        policy_frame = _label_frame(
            policy_frame, "CURRENT POLICY", policy_status
        )
        separator = np.full((self.args.height, 6, 3), 235, dtype=np.uint8)
        return np.concatenate([ref_frame, separator, policy_frame], axis=1)

    def _show_current_frame(self) -> None:
        if self.reference is None:
            return
        frame = self._render_current_frame()
        if self.recording and self.video_writer is not None:
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        available_width = max(640, self.root.winfo_width() - 330)
        available_height = max(480, self.root.winfo_height() - 100)
        scale = min(
            available_width / frame.shape[1],
            available_height / frame.shape[0],
            1.0,
        )
        if scale < 1.0:
            shown = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            shown = frame
        self._photo = ImageTk.PhotoImage(Image.fromarray(shown))
        self.image_label.configure(image=self._photo)
        self._slider_updating = True
        self.frame_var.set(self.current_frame)
        self._slider_updating = False
        self.frame_label.configure(
            text=f"{self.current_frame + 1} / {self.total_frames}"
        )

    def _on_slider(self, value: str) -> None:
        if self._slider_updating or self.reference is None:
            return
        self.current_frame = min(int(float(value)), self.total_frames - 1)
        self._show_current_frame()

    def toggle_play(self) -> None:
        if self.reference is None:
            return
        self.playing = not self.playing
        self.play_button.config(text="暂停" if self.playing else "播放")
        if self.playing:
            if self.current_frame >= self.total_frames - 1:
                self.current_frame = 0
            self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        self._show_current_frame()
        if self.current_frame >= self.total_frames - 1:
            self.playing = False
            self.play_button.config(text="播放")
            self._stop_recording()
            return
        self.current_frame += 1
        self.root.after(max(1, round(1000.0 / self.fps)), self._tick)

    def restart(self) -> None:
        self.current_frame = 0
        self._show_current_frame()

    def _new_recording_path(self) -> Path:
        if self.current_record is None:
            raise RuntimeError("No clip selected")
        self.record_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = _sanitize_filename(self.current_record["motion_key"])
        return self.record_dir / f"{name}_{stamp}.mp4"

    def toggle_recording(self) -> None:
        if self.recording:
            self._stop_recording()
            return
        if self.reference is None:
            return
        path = self._new_recording_path()
        size = (self.args.width * 2 + 6, self.args.height)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
        )
        if not writer.isOpened():
            messagebox.showerror("录像失败", f"无法创建视频：{path}")
            return
        self.video_writer = writer
        self.recording = True
        self.record_button.config(text="停止录像")
        if not self.playing:
            self.toggle_play()

    def _stop_recording(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        if hasattr(self, "record_button"):
            self.record_button.config(text="开始录像")

    def export_full_clip(self) -> None:
        if self.reference is None:
            return
        was_playing = self.playing
        self.playing = False
        path = self._new_recording_path()
        self.record_dir.mkdir(parents=True, exist_ok=True)
        size = (self.args.width * 2 + 6, self.args.height)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
        )
        if not writer.isOpened():
            messagebox.showerror("导出失败", f"无法创建视频：{path}")
            return
        previous = self.current_frame
        try:
            for frame_idx in range(self.total_frames):
                self.current_frame = frame_idx
                rgb = self._render_current_frame()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if frame_idx % 25 == 0:
                    self.frame_label.configure(
                        text=f"正在导出 {frame_idx + 1} / {self.total_frames}"
                    )
                    self.root.update_idletasks()
        finally:
            writer.release()
            self.current_frame = previous
            self.playing = was_playing
            self._show_current_frame()
        messagebox.showinfo("导出完成", f"视频已保存：\n{path}")

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.playing = False
        self._stop_recording()
        self.renderer.close()
        self.root.destroy()


def main() -> None:
    args = parse_args()
    dataset = TrainingDataset(args.dataset_root)
    index = PolicyClipIndex(args.policy_npz_dir)
    if args.list_only:
        print(
            json.dumps(
                {
                    "dataset_root": str(dataset.root),
                    "policy_npz_dir": str(index.root),
                    "num_clips": len(index.records),
                    "clips": [
                        record["motion_key"] for record in index.records
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    ComparisonViewer(args).run()


if __name__ == "__main__":
    main()
