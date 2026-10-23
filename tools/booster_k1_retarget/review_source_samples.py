"""Reproducible K1 kinematic previews; never claims physical tracking success."""

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_release_h5 import audit
from organize_motion_curriculum import partition

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/holomotion_k1_v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--selection", nargs="+", help="Explicit SOURCE:MOTION pairs"
    )
    p.add_argument("--replacement-batch-root", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = json.loads((args.snapshot / "all_motions.json").read_text())
    if args.replacement_batch_root:
        lookup = {(r["source"], r["motion"]): r for r in rows}
        for path in args.replacement_batch_root.glob("*/*/status.json"):
            status = json.loads(path.read_text())
            if status["status"] != "complete":
                continue
            root = Path(status["filtered_root"])
            m = json.loads((root / "manifest.json").read_text())
            source = {"cmu_asf": "CMU", "style100": "100STYLE"}[
                status["source"]
            ]
            for key, c in m["clips"].items():
                row = lookup[source, Path(c["metadata"]["source_path"]).stem]
                row.update(robot_root=str(root), robot_motion_key=key)
    # Stratify by source and behavior, not just easy walking.
    chosen = [
        ("CMU", "113_21"),
        ("CMU", "02_01"),
        ("CMU", "02_03"),
        ("CMU", "02_04"),
        ("CMU", "02_05"),
        ("CMU", "111_09"),
        ("100STYLE", "Neutral_ID"),
        ("100STYLE", "Neutral_FW"),
        ("100STYLE", "Neutral_FR"),
        ("100STYLE", "Neutral_BW"),
        ("100STYLE", "Aeroplane_FW"),
        ("100STYLE", "Akimbo_FR"),
    ]
    if args.selection:
        chosen = [tuple(item.split(":", 1)) for item in args.selection]
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "assets/robots/booster_k1/K1_22dof.xml")
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=300, width=400)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.azimuth, camera.elevation = 2.3, 135, -15
    reports = []
    for source, name in chosen:
        row = next(
            r for r in rows if r["source"] == source and r["motion"] == name
        )
        root = Path(row["robot_root"])
        key = row["robot_motion_key"]
        dest = args.output / (source + "_" + name)
        partition(root, dest / "data", {key})
        metrics = audit(dest / "data", stride=1)
        m = json.loads((root / "manifest.json").read_text())
        c = m["clips"][key]
        assert m["dof_names"] == [
            model.joint(i).name
            for i in range(model.njnt)
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        ]
        s, n = c["start"], c["length"]
        with h5py.File(root / m["hdf5_shards"][c["shard"]]["file"], "r") as f:
            xyz, quat, q = [
                f[k][s : s + n]
                for k in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
            ]
        fps = float(c["metadata"]["motion_fps"])
        sheet = Image.new("RGB", (1600, 990), "white")
        draw = ImageDraw.Draw(sheet)
        for j, idx in enumerate(np.linspace(0, n - 1, 12).astype(int)):
            data.qpos[:] = np.r_[xyz[idx], quat[idx, [3, 0, 1, 2]], q[idx]]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = xyz[idx]
            renderer.update_scene(data, camera=camera)
            x, y = (j % 4) * 400, (j // 4) * 330
            sheet.paste(Image.fromarray(renderer.render()), (x, y))
            draw.text(
                (x + 6, y + 303), f"{name} | {idx / fps:.2f}s", fill="black"
            )
        sheet.save(dest / "full_clip_samples.jpg")
        # First 30 seconds at 25 Hz; sheet above covers the full clip.
        with imageio.get_writer(
            dest / "preview.mp4", fps=fps / 2, codec="libx264"
        ) as writer:
            for idx in range(0, min(n, int(30 * fps)), 2):
                data.qpos[:] = np.r_[xyz[idx], quat[idx, [3, 0, 1, 2]], q[idx]]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = xyz[idx]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
        report = {
            "source_record": row,
            "numerical_full_frame_audit": metrics,
            "preview_seconds": min(n / fps, 30),
            "full_clip_seconds": n / fps,
            "rendering": "kinematic_qpos_replay_NOT_physics_rollout",
            "visual_decision": "pending_manual_inspection",
        }
        (dest / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append({"source": source, "motion": name, "folder": str(dest)})
        (args.output / "samples.json").write_text(
            json.dumps(reports, indent=2) + "\n"
        )
        print(source, name, metrics["clips"][0], flush=True)
    renderer.close()


if __name__ == "__main__":
    main()
