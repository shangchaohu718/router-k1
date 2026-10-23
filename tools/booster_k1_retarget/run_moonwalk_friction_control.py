"""Run a sequential friction-only control, then a 2x2 policy/physics evaluation.

Waits for the explicitly selected teacher pilot; never signals other processes.
Does not modify either baseline checkpoint or any training data.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def analyze_and_render(status, output):
    # Grounding/anchor metrics below are explicitly geometry/velocity proxies;
    # exported policy NPZs do not contain measured per-foot ground forces.
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation
    from tools.booster_k1_retarget.k1_policy_comparison_viewer import (
        ComparisonRenderer,
        DEFAULT_ROBOT_XML,
        align_initial_heading,
        _label_frame,
    )

    with np.load(
        REPO
        / "data/holomotion_k1_v1/moonwalk_lower_body_teacher_v56_v1/teacher.npz",
        allow_pickle=False,
    ) as bank:
        roles = bank["role_code"].copy()
        reference_q = bank["reference_dof_pos"].copy()
        offset = bank["foot_offsets"].copy()
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_ROBOT_XML))
    data = mujoco.MjData(model)
    feet = [model.body(s + "_ankle_roll_link").id for s in ("left", "right")]
    summary = {}
    for name, info in status["evaluations"].items():
        folder = Path(info["policy_npz_root"]).parent
        with np.load(
            next(Path(info["policy_npz_root"]).glob("*.npz")),
            allow_pickle=False,
        ) as f:
            metadata = json.loads(str(f["metadata"]))
            n = min(
                len(f["robot_dof_pos"]),
                int(metadata.get("valid_prefix_len", len(f["robot_dof_pos"]))),
            )
            if n < 3:
                raise ValueError("Too few valid evaluation frames")
            q = f["robot_dof_pos"][:n].copy()
            pos = f["robot_global_translation"][:n, 0].copy()
            rot = f["robot_global_rotation_quat"][:n, 0].copy()
            rp = f["ref_global_translation"][:n, 0].copy()
            rr = f["ref_global_rotation_quat"][:n, 0].copy()
            rq = f["ref_dof_pos"][:n].copy()
        error = np.max(np.abs(rq[:, None] - reference_q[None]), axis=-1)
        phase = error.argmin(-1)
        if np.max(error[np.arange(n), phase]) > 1e-5:
            raise ValueError(
                "Evaluation reference no longer matches teacher phase labels"
            )
        role = roles[phase]
        points = []
        for i in range(n):
            data.qpos[:] = np.r_[pos[i], rot[i, [3, 0, 1, 2]], q[i]]
            mujoco.mj_forward(model, data)
            points.append(
                [
                    data.xpos[b] + offset[j] @ data.xmat[b].reshape(3, 3).T
                    for j, b in enumerate(feet)
                ]
            )
        points = np.array(points)
        velocity = np.gradient(points, 0.02, axis=0)
        forward = Rotation.from_quat(rot).apply(
            np.tile([1.0, 0.0, 0.0], (n, 1))
        )
        forward[:, -1] = 0
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
        vx = (velocity.mean(2) * forward[:, None]).sum(-1)
        toe_speed = np.linalg.norm(velocity[:, :, 0, :2], axis=-1)
        gap = points[:, :, 1, 2] - points[:, :, 0, 2]
        slide = role == 2
        anchor = role == 1
        grounded = (points[:, :, 0, 2] < 0.025) & (
            points[:, :, :, 2].min(-1) > -0.015
        )
        qualified_slide = grounded & (vx < -0.03) & (np.abs(gap) < 0.035)
        qualified_anchor = grounded & (gap > 0.01) & (toe_speed < 0.1)
        paired = ((role[:, 0] == 1) & (role[:, 1] == 2)) | (
            (role[:, 0] == 2) & (role[:, 1] == 1)
        )
        success = np.where(
            slide, qualified_slide, np.where(anchor, qualified_anchor, False)
        ).all(-1)
        summary[name] = {
            "valid_frames": n,
            "duration_s": n / 50,
            "support_toe_speed_mean_m_s": float(toe_speed[anchor].mean()),
            "support_toe_speed_p95_m_s": float(
                np.percentile(toe_speed[anchor], 95)
            ),
            "support_slip_over_0p1_m_s_fraction": float(
                (toe_speed[anchor] > 0.1).mean()
            ),
            "slide_backward_speed_mean_m_s": float((-vx[slide]).mean()),
            "slide_grounded_kinematic_proxy_fraction": float(
                grounded[slide].mean()
            ),
            "anchor_heel_gap_mean_mm": float(gap[anchor].mean() * 1000),
            "paired_mechanism_kinematic_proxy_fraction": float(
                success[paired].mean()
            ),
            "backward_root_distance_m": float(
                -(pos[-1] - pos[0]) @ forward[0]
            ),
            "notes": "Toe-near-ground and toe velocity are proxies, not measured ground-contact force.",
        }
        pos, rot = align_initial_heading(pos, rot)
        rp, rr = align_initial_heading(rp, rr)
        renderer = ComparisonRenderer(DEFAULT_ROBOT_XML, 400, 400)
        try:
            with imageio.get_writer(
                folder / "comparison.mp4",
                fps=25,
                codec="libx264",
                macro_block_size=2,
            ) as writer:
                for i in range(0, n, 2):
                    left = _label_frame(
                        renderer.render_pose(rp[i], rr[i], rq[i]),
                        "REFERENCE",
                        f"frame {i}",
                    )
                    right = _label_frame(
                        renderer.render_pose(pos[i], rot[i], q[i]),
                        name,
                        f"frame {i}",
                    )
                    writer.append_data(
                        np.concatenate(
                            [
                                left,
                                np.full((400, 6, 3), 235, dtype=np.uint8),
                                right,
                            ],
                            axis=1,
                        )
                    )
        finally:
            renderer.close()
    (output / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    status = {
        "baseline": str(a.baseline.resolve()),
        "baseline_pid": a.baseline_pid,
        "low_friction": {"static": 0.3, "dynamic": 0.15},
        "num_envs": 64,
        "iterations": 500,
        "evaluations": {},
    }

    def state(stage, **extra):
        status.update(stage=stage, updated_unix_seconds=time.time(), **extra)
        tmp = a.output / "status.tmp"
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(a.output / "status.json")
        print(stage, flush=True)

    try:
        state("waiting_for_baseline_to_release_gpu")
        proc = Path(f"/proc/{a.baseline_pid}/cmdline")
        identity = proc.read_bytes() if proc.exists() else b""
        if (
            identity
            and b"motion_tracking_booster_k1_moonwalk_teacher_aux"
            not in identity
        ):
            raise ValueError(
                "Selected PID is not the requested baseline pilot"
            )
        deadline = time.monotonic() + 8 * 3600
        while identity and proc.exists() and proc.read_bytes() == identity:
            if time.monotonic() > deadline:
                raise TimeoutError("Baseline wait exceeded 8 hours")
            time.sleep(10)
        baseline = a.baseline / "model_499.pt"
        if (
            not baseline.is_file()
            or not (a.baseline / "model_499/actor/model.safetensors").is_file()
        ):
            raise RuntimeError(
                "Baseline did not finish/save model_499; no control submitted"
            )
        state("waiting_for_gpu_memory")
        while True:
            memory = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            if int(memory.splitlines()[0]) >= 3800:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Not enough free GPU memory")
            time.sleep(15)
        env = os.environ.copy()
        env.update(
            CONFIG_NAME="motion_tracking_booster_k1_moonwalk_teacher_lowfriction",
            NUM_ENVS="64",
            TORCH_COMPILE_DISABLE="1",
        )
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        parent = REPO / "logs/HoloMotionK1MoonwalkTeacherLowFriction"
        before = set(parent.glob("*")) if parent.exists() else set()
        state("training_low_friction_control")
        with (a.output / "training.log").open("w") as log:
            subprocess.run(
                [
                    "bash",
                    "holomotion/scripts/training/train_motion_tracking.sh",
                ],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        runs = set(parent.glob("*")) - before
        if len(runs) != 1:
            raise RuntimeError(
                f"Could not identify unique control run: {runs}"
            )
        low = next(iter(runs))
        status["low_friction_run"] = str(low.resolve())
        # Fail closed if the actual PhysX material audit did not execute.
        if (
            "FIXED_FOOT_FRICTION_AUDIT"
            not in (a.output / "training.log").read_text()
        ):
            raise RuntimeError("Material startup audit missing")
        for label, checkpoint in [
            ("A_baseline", baseline),
            ("B_lowtrained", low / "model_499.pt"),
        ]:
            for condition in ("normal", "low"):
                name = label + "_" + condition
                state("evaluating_" + name)
                folder = a.output / name
                folder.mkdir()
                cfg = (
                    "eval_isaaclab"
                    if condition == "normal"
                    else "eval_booster_k1_moonwalk_lowfriction"
                )
                command = [
                    sys.executable,
                    "holomotion/src/evaluation/eval_motion_tracking_single.py",
                    f"--config-name=evaluation/{cfg}",
                    "robot=booster/K1/22dof/22dof_training_isaaclab",
                    "env/terminations=termination_motion_tracking",
                    "headless=true",
                    "num_envs=1",
                    "export_policy=false",
                    "dump_npzs=true",
                    "calc_per_clip_metrics=false",
                    "generate_report=false",
                    "motion_h5_path=['data/holomotion_k1_v1/filtered_robot/moonwalk_v14']",
                    "+use_kv_cache=true",
                    "export_only=false",
                    f"checkpoint={checkpoint}",
                    "project_name=HoloMotionK1MoonwalkFrictionComparison",
                    "robot.motion.max_frame_length=500",
                    "+algo.config.offline_eval_max_steps=500",
                    "robot.motion.excluded_motion_names=null",
                    "robot.motion.handpicked_motion_names=['moonwalk_alex61_64_v14_synthetic']",
                    "robot.motion.dataloader.num_workers=0",
                    "robot.motion.dataloader.timeout=0",
                ]
                with (folder / "evaluation.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=REPO,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                if (
                    condition == "low"
                    and "FIXED_FOOT_FRICTION_AUDIT"
                    not in (folder / "evaluation.log").read_text()
                ):
                    raise RuntimeError(
                        "Low-friction evaluation did not apply physical material"
                    )
                exported = (
                    checkpoint.parent
                    / f"isaaclab_eval_output_{checkpoint.stem}_moonwalk_v14"
                )
                files = list(exported.glob("*.npz"))
                if len(files) != 1:
                    raise RuntimeError("Expected exactly one v14 rollout")
                dst = folder / "policy_npzs"
                dst.mkdir()
                shutil.copy2(files[0], dst / files[0].name)
                status["evaluations"][name] = {
                    "checkpoint": str(checkpoint.resolve()),
                    "policy_npz_root": str(dst.resolve()),
                    "friction_condition": condition,
                }
        state("analyzing_and_rendering")
        analyze_and_render(status, a.output)
        state(
            "complete",
            metrics=str((a.output / "comparison_metrics.json").resolve()),
        )
    except Exception as exc:
        state("failed_no_other_jobs_modified", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
