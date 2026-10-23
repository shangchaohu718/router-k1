"""Render matched original/repaired reference frames, never a policy rollout."""

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]


def load(root, manifest, key):
    clip = manifest["clips"][key]
    s, n = clip["start"], clip["length"]
    with h5py.File(
        root / manifest["hdf5_shards"][clip["shard"]]["file"], "r"
    ) as f:
        pos, rot, q = (
            f[name][s : s + n]
            for name in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
        )
    return np.c_[pos, rot[:, [3, 0, 1, 2]], q]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--repair-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--keys", nargs="+", required=True)
    p.add_argument("--video", action="store_true")
    p.add_argument("--same-clip-keys", action="store_true", help="Compare full clips with identical keys and source-frame correspondence")
    p.add_argument("--align-floor-for-review", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    old = json.loads((a.original_root / "manifest.json").read_text())
    new = json.loads((a.repair_root / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "assets/robots/booster_k1/K1_22dof.xml")
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=400, height=300)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.azimuth, camera.elevation = 1.9, 135, -15
    output = []

    def render(pose):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        camera.lookat[:] = pose[:3]
        camera.lookat[2] = 0.55
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    try:
        for key in a.keys:
            meta = new["clips"][key]["metadata"]
            parent = key if a.same_clip_keys else meta["parent_motion_key"]
            original = load(a.original_root, old, parent)
            repaired = load(a.repair_root, new, key)
            if a.same_clip_keys:
                original_meta = old['clips'][parent]['metadata']
                if (original_meta['source_path'], original_meta['canonical_frame_start']) != (meta['source_path'], meta['canonical_frame_start']):
                    raise ValueError('Source/frame correspondence differs')
                start, end = 0, len(repaired)
                if len(original) != end:
                    raise ValueError('Full-clip lengths differ')
            else:
                start, end = meta["parent_frame_start"], meta["parent_frame_end"]
            original = original[start:end]
            if original.shape != repaired.shape:
                raise ValueError(
                    "Original/repair frame correspondence differs"
                )
            floor_shifts = []
            if a.align_floor_for_review:
                foot_geoms = [
                    g for g in range(model.ngeom)
                    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                    and model.body(int(model.geom_bodyid[g])).name in {"left_ankle_roll_link", "right_ankle_roll_link"}
                ]
                for poses in (original, repaired):
                    soles = []
                    for pose in poses:
                        data.qpos[:] = pose
                        mujoco.mj_forward(model, data)
                        soles.append(min(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g] for g in foot_geoms))
                    shift = float(np.percentile(soles, 1))
                    poses[:, 2] -= shift
                    floor_shifts.append(shift)
            sheet = Image.new("RGB", (400 * 6, 330 * 2), "white")
            draw = ImageDraw.Draw(sheet)
            frames = np.unique(
                np.linspace(0, len(repaired) - 1, 6).astype(int)
            )
            for col, frame in enumerate(frames):
                for row, (label, poses) in enumerate(
                    (
                        ("ORIGINAL REFERENCE", original),
                        ("REPAIRED REFERENCE CANDIDATE", repaired),
                    )
                ):
                    sheet.paste(
                        Image.fromarray(render(poses[frame])),
                        (col * 400, row * 330 + 30),
                    )
                    draw.text(
                        (col * 400 + 5, row * 330 + 5),
                        f"{label} frame={start + frame}",
                        fill="black",
                    )
            png = a.output / (key + ".jpg")
            sheet.save(png, quality=90)
            if a.video:
                with imageio.get_writer(
                    a.output / (key + ".mp4"), fps=25, codec="libx264"
                ) as writer:
                    for frame in range(0, len(repaired), 2):
                        canvas = Image.new("RGB", (800, 336), "white")
                        for x, poses in ((0, original), (400, repaired)):
                            canvas.paste(
                                Image.fromarray(render(poses[frame])), (x, 32)
                            )
                        text = ImageDraw.Draw(canvas)
                        text.text((5, 8), "ORIGINAL REFERENCE", fill="black")
                        text.text(
                            (405, 8),
                            "REPAIRED CANDIDATE - NOT POLICY",
                            fill="black",
                        )
                        writer.append_data(np.asarray(canvas))
            output.append(
                {
                    "key": key,
                    "original_parent": parent,
                    "original_frame_span": [start, end],
                    "frames": len(repaired),
                    "visual_only_constant_floor_shifts_m": floor_shifts,
                    "contact_sheet": str(png.resolve()),
                    "meaning": "matched kinematic references, not policy or dynamics; same world camera axes, no mirroring or independent yaw normalization",
                }
            )
            print(key, "rendered", flush=True)
    finally:
        renderer.close()
    (a.output / "render_report.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
