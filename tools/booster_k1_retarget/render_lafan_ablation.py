"""Same-world-axis diagnostic renders, no policy and no dynamics."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import (
    REPO,
    load,
)


def poses(root, key):
    output = np.full((500, 29), np.nan)
    for name in [
        "01_kinematic_candidates",
        "80_short_or_source_review",
        "90_rejected_intervals",
    ]:
        folder = root / "review" / name
        m = json.loads((folder / "manifest.json").read_text())
        for k, c in m["clips"].items():
            if c["metadata"]["parent_motion_key"] != key:
                continue
            arrays = load(folder, k)
            s, e = (
                c["metadata"]["parent_frame_start"],
                c["metadata"]["parent_frame_end"],
            )
            output[s:e] = np.c_[
                arrays["ref_root_pos"],
                arrays["ref_root_rot"][:, [3, 0, 1, 2]],
                arrays["ref_dof_pos"],
            ]
    if not np.isfinite(output).all():
        raise ValueError("Missing matched review frames")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--video", action="store_true")
    p.add_argument(
        "--variants",
        nargs=3,
        default=["old_reproduced", "current_full", "no_proportions"],
    )
    p.add_argument(
        "--frames", nargs=6, type=int, default=[0, 99, 199, 299, 399, 499]
    )
    p.add_argument("--subdir", default="visual")
    p.add_argument(
        "--keys", nargs="+", default=["dance1_subject1", "dance2_subject1"]
    )
    a = p.parse_args()
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "assets/robots/booster_k1/K1_22dof.xml")
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=360, height=280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.azimuth, camera.elevation = 1.9, 135, -15
    out = a.root / a.subdir
    out.mkdir(exist_ok=True)

    def render(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        camera.lookat[:] = [q[0], q[1], 0.55]
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    try:
        for key in a.keys:
            rows = [(v, poses(a.root / v, key)) for v in a.variants]
            sheet = Image.new("RGB", (2160, 930), "white")
            text = ImageDraw.Draw(sheet)
            for row, (name, values) in enumerate(rows):
                for col, frame in enumerate(a.frames):
                    sheet.paste(
                        Image.fromarray(render(values[frame])),
                        (col * 360, row * 310 + 30),
                    )
                    text.text(
                        (col * 360 + 4, row * 310 + 7),
                        f"{name} local frame {frame}",
                        fill="black",
                    )
            sheet.save(out / (key + ".jpg"), quality=92)
            if a.video:
                with imageio.get_writer(
                    out / (key + ".mp4"), fps=25, codec="libx264"
                ) as writer:
                    for frame in range(0, 500, 2):
                        canvas = Image.new("RGB", (720, 320), "white")
                        label = ImageDraw.Draw(canvas)
                        for col, (name, values) in enumerate(rows[:2]):
                            canvas.paste(
                                Image.fromarray(render(values[frame])),
                                (col * 360, 32),
                            )
                            label.text((col * 360 + 4, 6), name, fill="black")
                        writer.append_data(np.asarray(canvas))
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
