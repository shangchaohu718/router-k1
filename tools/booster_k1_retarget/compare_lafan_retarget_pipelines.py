"""Isolated old LaFAN GMR vs current processing ablation; no active writes."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from tools.booster_k1_retarget.recover_short_reference_candidates import dump

REPO = Path(__file__).resolve().parents[2]
SMPL = Path(
    "/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
)
VARIANTS = [
    "current_full",
    "no_proportions",
    "no_collision",
    "no_native_positions",
    "old_foot_weight",
    "old_root_weight",
    "old_root_and_foot_weight",
    "root_weight_30",
    "root_weight_30_foot2",
    "old_task_weights",
    "lafan_root100_arms10",
    "lafan_root100_arms20",
    "lafan_arm_lengths",
    "lafan_arm_lengths_pos40",
    "lafan_arm_direction",
    "lafan_arm_direction_rootpos200",
]


def load(root, key):
    m = json.loads((root / "manifest.json").read_text())
    c = m["clips"][key]
    with h5py.File(root / m["hdf5_shards"][c["shard"]]["file"], "r") as f:
        s, e = c["start"], c["start"] + c["length"]
        return {
            n: f[n][s:e]
            for n in ["ref_root_pos", "ref_root_rot", "ref_dof_pos"]
        }


def worker(a):
    from tools.booster_k1_retarget import retry_guarded_reference as guard

    probe = guard.probe
    original = probe.run

    def run(settings):
        if a.worker == "root_weight_30_foot2":
            settings.foot_orientation_cost = 2.0
        if a.worker in {
            "old_foot_weight",
            "old_root_and_foot_weight",
            "old_task_weights",
            "lafan_root100_arms10",
            "lafan_root100_arms20",
            "lafan_arm_lengths",
            "lafan_arm_lengths_pos40",
            "lafan_arm_direction",
            "lafan_arm_direction_rootpos200",
        }:
            settings.foot_orientation_cost = 10.0
        if a.worker == "no_proportions":
            settings.proportional_skeleton = False
        if a.worker == "no_collision":
            settings.collision = settings.line_search = False
        if a.worker == "no_native_positions":
            settings.native = False
        original(settings)

    create = probe.create_k1_retargeter
    proportion = probe.proportion_lower_body

    def with_arm_lengths(human, retargeter, scale, rest):
        h = proportion(human, retargeter, scale, rest)
        if a.worker not in {
            "lafan_arm_lengths",
            "lafan_arm_lengths_pos40",
            "lafan_arm_direction",
            "lafan_arm_direction_rootpos200",
        }:
            return h
        from scipy.spatial.transform import Rotation

        model = retargeter.model
        trunk = rest.xpos[model.body("trunk").id]
        rotation = (
            Rotation.from_quat(h["pelvis"][1], scalar_first=True)
            * retargeter.rot_offsets1["pelvis"]
        )
        root = h["pelvis"][0] * scale
        for side in ["left", "right"]:
            shoulder = rest.xpos[model.body(side + "_shoulder_roll_link").id]
            elbow = rest.xpos[model.body(side + "_elbow_yaw_link").id]
            direction = h[side + "_elbow"][0] - h[side + "_shoulder"][0]
            length = np.linalg.norm(direction)
            if length < 1e-6:
                raise ValueError("Degenerate source upper arm")
            target = root + rotation.apply(shoulder - trunk)
            h[side + "_shoulder"][0] = target / scale
            h[side + "_elbow"][0] = (
                target + direction / length * np.linalg.norm(elbow - shoulder)
            ) / scale
        return h

    probe.proportion_lower_body = with_arm_lengths

    def guarded_create(**kwargs):
        r = create(**kwargs)
        if a.worker in {
            "old_task_weights",
            "lafan_root100_arms10",
            "lafan_root100_arms20",
            "lafan_arm_lengths",
            "lafan_arm_lengths_pos40",
            "lafan_arm_direction",
            "lafan_arm_direction_rootpos200",
        }:
            cfg = REPO / "tools/booster_k1_retarget/configs"
            old_cfg = json.loads(
                (cfg / "bvh_lafan1_to_official_k1_v2.json").read_text()
            )
            new_cfg = json.loads(
                (cfg / "gmr_smplx_to_official_k1_v2.json").read_text()
            )
            for label, tasks in [
                ("ik_match_table1", r.human_body_to_task1),
                ("ik_match_table2", r.human_body_to_task2),
            ]:
                for robot, entry in new_cfg[label].items():
                    task = tasks[entry[0]]
                    weights = old_cfg[label].get(robot, [None, 0, 0])
                    task.set_position_cost(weights[1])
                    task.set_orientation_cost(weights[2])
                    if a.worker.startswith("lafan_arm_direction"):
                        if "shoulder_roll" in robot:
                            task.set_orientation_cost(0.0)
                        if "elbow_yaw" in robot:
                            task.set_position_cost(200.0)
                        if (
                            a.worker == "lafan_arm_direction_rootpos200"
                            and robot == "trunk"
                        ):
                            task.set_position_cost(200.0)
                    if (
                        a.worker == "lafan_arm_lengths_pos40"
                        and "elbow_yaw" in robot
                    ):
                        task.set_position_cost(
                            40.0 if label == "ik_match_table2" else 0.0
                        )
                    if a.worker.startswith("lafan_root100_arms") and any(
                        part in robot
                        for part in ("shoulder_roll", "elbow_yaw")
                    ):
                        task.set_orientation_cost(
                            float(a.worker.rsplit("arms", 1)[1])
                        )
        if a.worker in {"root_weight_30", "root_weight_30_foot2"}:
            for table in (r.human_body_to_task1, r.human_body_to_task2):
                table["pelvis"].set_orientation_cost(30.0)
        if a.worker in {"old_root_weight", "old_root_and_foot_weight"}:
            for table in (r.human_body_to_task1, r.human_body_to_task2):
                table["pelvis"].set_orientation_cost(100.0)
        config = r.configuration

        def integrate(velocity, dt):
            q, _, _ = guard.bounded_step(
                r.model, config.data.qpos.copy(), velocity, dt
            )
            config.update(q)

        config.integrate_inplace = integrate
        return r

    probe.create_k1_retargeter = guarded_create
    worker_manifest = json.loads(
        (a.output / "input/manifest.json").read_text()
    )
    worker_native = all(
        Path(c["metadata"]["source_path"]).suffix.lower() in {".bvh", ".amc"}
        for c in worker_manifest["clips"].values()
    )
    run(
        argparse.Namespace(
            input=a.output / "input",
            output=a.output / a.worker / "raw",
            smpl_model=SMPL,
            keys=[""],
            max_frames=a.max_frames,
            native=worker_native,
            collision=True,
            line_search=True,
            anatomical_scale=True,
            proportional_skeleton=True,
            foot_orientation_cost=0.5,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", choices=VARIANTS)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument(
        "--keys", nargs="+", default=["dance1_subject1", "dance2_subject1"]
    )
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.worker:
        return worker(a)
    from tools.booster_k1_retarget import bvh_to_robot_dataset as old
    from holomotion.src.training.data_production.robot_h5 import (
        RobotH5ShardWriter,
    )

    src = REPO / "data/lafan1_booster_k1_gmr_v2_h5_train_subject1_4"
    m = json.loads((src / "manifest.json").read_text())
    keys = a.keys
    inp = copy.deepcopy(m)
    inp["clips"] = {k: copy.deepcopy(m["clips"][k]) for k in keys}
    for c in inp["clips"].values():
        c["start"] += a.start_frame
        c["length"] = 500
        c["metadata"].update(
            canonical_frame_start=a.start_frame,
            canonical_frame_count=500,
            num_frames=500,
            active_split="train",
        )
    inp["motion_keys"] = keys
    inp["hdf5_shards"] = [
        {**s, "file": str((src / s["file"]).resolve())}
        for s in m["hdf5_shards"]
    ]
    dump(a.output / "input/manifest.json", inp)
    config = Path(m["retarget_config"]["ik_config"])
    old.configure_k1_gmr(config)
    baseline = a.output / "old_reproduced/raw"
    if not (baseline / "manifest.json").exists():
        baseline.mkdir(parents=True, exist_ok=False)
        writer = RobotH5ShardWriter(baseline / "shards/old.h5", dof_pos_dim=22)
        view = copy.deepcopy(inp)
        for i, key in enumerate(keys):
            # Preserve old solver history from source frame zero.
            q = old.retarget_bvh(
                Path(m["clips"][key]["metadata"]["source_path"]),
                max_frames=int(np.ceil((a.start_frame + 500) * 30 / 50)) + 1,
            )
            q = old.resample_qpos(q, source_fps=30, target_fps=50)[
                a.start_frame : a.start_frame + 500
            ]
            if len(q) != 500:
                raise ValueError("Old/source frame correspondence mismatch")
            arrays = old.qpos_to_robot_arrays(q)
            start, length = writer.append_motion(
                motion_id=i,
                arrays=arrays,
                metadata_json=json.dumps(view["clips"][key]["metadata"]),
            )
            view["clips"][key].update(
                shard=0, clip_idx=i, start=start, length=length
            )
        writer.finalize()
        view["hdf5_shards"] = [
            {
                "file": "shards/old.h5",
                "num_frames": 500 * len(keys),
                "num_clips": len(keys),
            }
        ]
        dump(baseline / "manifest.json", view)
    variants = ["old_reproduced"] + a.variants
    failures = {}
    for variant in variants:
        root = a.output / variant
        root.mkdir(parents=True, exist_ok=True)
        try:
            with (root / "run.log").open("a") as log:
                if not (root / "raw/manifest.json").exists():
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.booster_k1_retarget.compare_lafan_retarget_pipelines",
                            "--output",
                            str(a.output),
                            "--worker",
                            variant,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                if not (root / "review/report.json").exists():
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.booster_k1_retarget.salvage_repaired_reference",
                            "--input",
                            str(root / "raw"),
                            "--output",
                            str(root / "review"),
                            "--smpl-model",
                            str(SMPL),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
        except subprocess.CalledProcessError as e:
            failures[variant] = str(e)
    result = {
        "training_started": False,
        "training_approved": False,
        "frames_per_clip": 500,
        "start_frame_50hz": a.start_frame,
        "keys": keys,
        "old_config": str(config),
        "same_robot_xml": str(old.K1_XML_PATH),
        "scope": "Matched 10-second windows. Old at 30 Hz then resampled, current IK at 50 Hz; that timing distinction and fresh current initialization at window start are part of the pipeline, not a controlled single-change comparison. Paired current ablations isolate flags; no dynamic validation.",
        "variants": {},
        "failures": failures,
    }
    for variant in variants:
        path = a.output / variant / "review/report.json"
        if path.exists():
            result["variants"][variant] = json.loads(path.read_text())
    result["old_reproduction_error"] = {}
    for key in keys:
        stored = load(src, key)
        fresh = load(baseline, key)
        result["old_reproduction_error"][key] = {
            n: float(
                np.max(
                    np.abs(
                        stored[n][a.start_frame : a.start_frame + 500]
                        - fresh[n]
                    )
                )
            )
            for n in fresh
        }
    dump(a.output / "comparison_report.json", result)


if __name__ == "__main__":
    main()
