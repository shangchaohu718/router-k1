"""Isolated retry with bounded internal IK integration, not relaxed limits.

The original worker and shared Mink/GMR packages are never modified. A whole
integration step is backtracked; individual joints are never clipped.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from tools.booster_k1_retarget import probe_native_position_retarget as probe


def bounded_step(model, previous, velocity, dt):
    ids = np.flatnonzero(
        model.jnt_limited & (model.jnt_type != mujoco.mjtJoint.mjJNT_FREE)
    )
    addresses = model.jnt_qposadr[ids]
    limits = model.jnt_range[ids]
    if (
        not np.isfinite(previous).all()
        or not np.isfinite(velocity).all()
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Invalid IK integration state")
    values = previous[addresses]
    if np.any(values < limits[:, 0] - 1e-10) or np.any(
        values > limits[:, 1] + 1e-10
    ):
        raise ValueError(
            "Internal guard requires a valid previous configuration"
        )
    proposed = previous.copy()
    mujoco.mj_integratePos(model, proposed, velocity, dt)
    q = proposed[addresses]
    violation = np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0)
    worst = int(np.argmax(violation))
    diagnostic = {
        "max_proposed_violation_rad": float(violation[worst]),
        "joint": model.joint(int(ids[worst])).name,
        "proposed_value_rad": float(q[worst]),
        "lower_rad": float(limits[worst, 0]),
        "upper_rad": float(limits[worst, 1]),
    }
    for attempt in range(30):
        alpha = 0.5**attempt
        candidate = previous.copy()
        mujoco.mj_integratePos(model, candidate, velocity, dt * alpha)
        values = candidate[addresses]
        if np.all(values >= limits[:, 0]) and np.all(values <= limits[:, 1]):
            return candidate, alpha, diagnostic
    return previous.copy(), 0.0, diagnostic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--foot-orientation-cost", type=float, default=0.5)
    a = p.parse_args()
    manifest = json.loads((a.input / "manifest.json").read_text())
    if len(manifest["clips"]) != 1:
        raise ValueError("Retry one source clip per immutable attempt")
    meta = next(iter(manifest["clips"].values()))["metadata"]
    native = Path(meta["source_path"]).suffix.lower() in {".amc", ".bvh"}
    create_original = probe.create_k1_retargeter
    stats = []

    def create_guarded(**kwargs):
        retargeter = create_original(**kwargs)
        config = retargeter.configuration
        record = {
            "internal_steps": 0,
            "backtracked_steps": 0,
            "stalled_steps": 0,
            "worst_proposed_violation": None,
        }
        stats.append(record)

        def integrate_guarded(velocity, dt):
            result, alpha, diagnostic = bounded_step(
                retargeter.model, config.data.qpos.copy(), velocity, dt
            )
            record["internal_steps"] += 1
            record["backtracked_steps"] += alpha < 1
            record["stalled_steps"] += alpha == 0
            old = record["worst_proposed_violation"]
            if (
                old is None
                or diagnostic["max_proposed_violation_rad"]
                > old["max_proposed_violation_rad"]
            ):
                record["worst_proposed_violation"] = diagnostic
            config.update(result)

        config.integrate_inplace = integrate_guarded
        return retargeter

    probe.create_k1_retargeter = create_guarded
    try:
        probe.run(
            SimpleNamespace(
                input=a.input,
                output=a.output,
                smpl_model=a.smpl_model,
                keys=[""],
                max_frames=10_000_000,
                native=native,
                collision=True,
                line_search=True,
                anatomical_scale=True,
                proportional_skeleton=True,
                foot_orientation_cost=a.foot_orientation_cost,
            )
        )
    finally:
        probe.create_k1_retargeter = create_original
        if a.output.exists():
            report = {
                "method": "Whole-step backtracking of internal IK integration, real joint limits unchanged, no per-joint clipping; standard subsequent source/geometry gates still required",
                "training_started": False,
                "training_approved": False,
                "foot_orientation_cost": a.foot_orientation_cost,
                "retargeters": stats,
            }
            (a.output / "inner_limit_guard_report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
