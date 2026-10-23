"""Optional lower-body trajectory teacher, not cross-policy action cloning.

Only enabled by the dedicated teacher config. Targets are tied to the exact
v14 reference/window/frame, never to episode elapsed time or a guessed cycle.
No teacher root trajectory, upper-body joints, or motor commands are loaded.
"""

import json
from pathlib import Path

import numpy as np
import torch


def quat_apply(q, v):
    xyz = q[..., 1:]
    uv = torch.linalg.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(xyz, uv, dim=-1))


def heading_xy(v, root_quat):
    w, x, y, z = root_quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    while yaw.ndim < v.ndim - 1:
        yaw = yaw.unsqueeze(-1)
    c, s = yaw.cos(), yaw.sin()
    return torch.stack(
        (c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]), -1
    )


def leg_score(actual, target, quality, std):
    if std <= 0:
        raise ValueError("Teacher joint scale must be positive")
    per_leg = (1 / (1 + ((actual - target) / std).square())).mean(-1)
    return (per_leg * quality).mean(-1)


def sole_score(
    actual_xy, actual_z, teacher_xy, teacher_z, quality, xy_scale, z_scale
):
    if min(xy_scale, z_scale) <= 0:
        raise ValueError("Teacher sole scales must be positive")
    error = ((actual_xy - teacher_xy) / xy_scale).square().sum(-1) + (
        (actual_z - teacher_z) / z_scale
    ).square()
    return ((1 / (1 + error)).mean(-1) * quality).mean(-1)


def mechanism_score(
    points_z,
    velocity_xy,
    contact_force,
    target_z,
    target_velocity,
    roles,
    quality,
    velocity_scale=0.2,
):
    """A paired bonus; stopping/backwalking cannot replace role-specific slide.

    Whole-foot force + toe-near-ground is a contact proxy, not a measured
    forefoot/heel force decomposition. Independent leg/sole teachers stay dense.
    """
    if velocity_scale <= 0:
        raise ValueError("Teacher velocity scale must be positive")
    gap = points_z[..., 1] - points_z[..., 0]
    target_gap = target_z[..., 1] - target_z[..., 0]
    # Normalize the soft force gate: exactly zero force must pay zero bonus.
    zero_force = torch.sigmoid(contact_force.new_tensor(-2.5))
    contact = (
        (torch.sigmoid((contact_force - 5.0) / 2.0) - zero_force)
        / (1 - zero_force)
    ).clamp(0, 1)
    near_toe = 1 / (
        1 + ((points_z[..., 0] - target_z[..., 0]) / 0.015).square()
    )
    speed = 1 / (
        1
        + ((velocity_xy.mean(-2) - target_velocity.mean(-2)) / velocity_scale)
        .square()
        .sum(-1)
    )
    backward = torch.sigmoid((-velocity_xy[..., 0].mean(-1) - 0.03) / 0.03)
    flat_tracking = 1 / (1 + ((gap - target_gap) / 0.015).square())
    slide = speed * backward * contact * near_toe * flat_tracking
    # Flat feet receive zero support lift, unlike averaging lift/contact scores.
    lift = (gap / target_gap.clamp_min(0.015)).clamp(0, 1)
    still = 1 / (1 + (velocity_xy.mean(-2) / 0.15).square().sum(-1))
    anchor = lift * near_toe * contact * still
    foot = torch.where(
        roles == 2,
        slide,
        torch.where(roles == 1, anchor, torch.zeros_like(slide)),
    )
    paired = ((roles[:, 0] == 1) & (roles[:, 1] == 2)) | (
        (roles[:, 0] == 2) & (roles[:, 1] == 1)
    )
    return torch.where(
        paired, (foot * quality).prod(-1), torch.zeros_like(foot[:, 0])
    )


def windowed_bonus(
    score, frames, roles, state=None, window=8, reset_mask=None
):
    """Reset on random starts, reference jumps and left/right role switches."""
    if window < 2:
        raise ValueError("Teacher qualification window must be at least two")
    if state is None:
        state = {
            "frame": frames - 2,
            "roles": torch.full_like(roles, -1),
            "count": torch.zeros_like(frames),
            "history": score.new_zeros((len(score), window)),
        }
    continuous = (frames == state["frame"] + 1) & (
        roles == state["roles"]
    ).all(-1)
    if reset_mask is not None:
        continuous &= ~reset_mask
    count = torch.where(
        continuous, state["count"] + 1, torch.ones_like(frames)
    ).clamp(max=window)
    history = torch.where(
        continuous[:, None],
        state["history"],
        torch.zeros_like(state["history"]),
    )
    history = torch.cat((history[:, 1:], score.detach()[:, None]), dim=1)
    value = torch.where(
        count >= window, history.mean(-1), torch.zeros_like(score)
    )
    return value, {
        "frame": frames.clone(),
        "roles": roles.clone(),
        "count": count,
        "history": history,
    }


class TeacherBank:
    def __init__(self, path, robot):
        self.path = str(Path(path).resolve())
        device = robot.data.joint_pos.device
        with np.load(self.path, allow_pickle=False) as a:
            self.metadata = json.loads(str(a["metadata"]))
            names = a["joint_names"].tolist()
            self.targets = {
                k: torch.as_tensor(a[k].copy(), device=device)
                for k in (
                    "joint_pos",
                    "sole_relative_xy",
                    "sole_ground_z",
                    "sole_velocity_heading_xy",
                    "role_code",
                    "quality",
                    "reference_dof_pos",
                )
            }
            self.offsets = torch.as_tensor(
                a["foot_offsets"].copy(), device=device
            )
        if (
            self.metadata["schema"] != "k1_lower_body_teacher_v1"
            or self.metadata["fps"] != 50
        ):
            raise ValueError("Unsupported teacher schema/rate")
        if len(names) != 12 or any(
            not any(part in n for part in ("hip_", "knee_", "ankle_"))
            for n in names
        ):
            raise ValueError("Teacher may supervise only the 12 leg joints")
        self.joint_ids = [robot.joint_names.index(n) for n in names]
        self.foot_ids = [
            robot.body_names.index(s + "_ankle_roll_link")
            for s in ("left", "right")
        ]
        self.validated_batch = None

    def select(self, command):
        batch = command._motion_cache.current_batch
        if batch is not self.validated_batch:
            if (
                float(command.cfg.target_fps) != 50
                or int(command.cfg.n_fut_frames) < 1
            ):
                raise ValueError(
                    "Teacher requires 50 Hz and immediate-next reference support"
                )
            key, n = (
                self.metadata["reference_motion_key"],
                self.metadata["frames"],
            )
            # This first experiment deliberately supports only one full window.
            # Fail closed if someone later adds other clips/crops to the config.
            if set(batch.raw_motion_keys) != {key} or set(
                batch.motion_keys
            ) != {self.metadata["expected_window_key"]}:
                raise ValueError(
                    "Teacher/reference window mismatch; do not apply v14 teacher to other motions"
                )
            if not torch.all(batch.lengths == n):
                raise ValueError("Teacher/reference frame length mismatch")
            if (
                list(command.urdf_dof_names)
                != self.metadata["reference_dof_names"]
            ):
                raise ValueError("Teacher reference DOF ordering mismatch")
            ref = batch.tensors["ref_dof_pos"]
            expected = self.targets["reference_dof_pos"].to(ref.device)
            if ref.shape[1:] != expected.shape or not torch.allclose(
                ref, expected.unsqueeze(0).expand_as(ref), atol=1e-6, rtol=0
            ):
                raise ValueError("Teacher source reference changed")
            self.validated_batch = batch
            self.window_state = None
        # Same +1/clamp as _get_immediate_next_ref_state_array used by rewards.
        index = (
            (command._frame_indices + 1)
            .clamp(max=self.metadata["frames"] - 1)
            .long()
        )
        return {
            k: value[index]
            for k, value in self.targets.items()
            if k != "reference_dof_pos"
        }

    def actual_soles(self, env, command):
        data = command.robot.data
        ids = self.foot_ids
        offset = self.offsets[None].expand(len(data.joint_pos), -1, -1, -1)
        rot = data.body_quat_w[:, ids, None, :].expand(-1, -1, 2, -1)
        world_offset = quat_apply(rot, offset)
        points = data.body_pos_w[:, ids, None, :] + world_offset
        vel = data.body_link_lin_vel_w[:, ids, None, :] + torch.linalg.cross(
            data.body_link_ang_vel_w[:, ids, None, :].expand_as(world_offset),
            world_offset,
            dim=-1,
        )
        xy = heading_xy(
            points - data.root_pos_w[:, None, None, :], data.root_quat_w
        )
        z = points[..., 2] - env.scene.env_origins[:, None, None, 2]
        return xy, z, heading_xy(vel, data.root_quat_w)


def resolve(env, teacher_path, command_name):
    command = env.command_manager.get_term(command_name)
    banks = getattr(env, "_moonwalk_auxiliary_teachers", None)
    if banks is None:
        banks = {}
        env._moonwalk_auxiliary_teachers = banks
    key = (str(Path(teacher_path).resolve()), command_name)
    if key not in banks:
        banks[key] = TeacherBank(teacher_path, command.robot)
    bank = banks[key]
    return command, bank, bank.select(command)


def moonwalk_teacher_leg_tracking(
    env, teacher_path, std=0.20, command_name="ref_motion"
):
    command, bank, target = resolve(env, teacher_path, command_name)
    actual = command.robot.data.joint_pos[:, bank.joint_ids].reshape(-1, 2, 6)
    return leg_score(actual, target["joint_pos"], target["quality"], std)


def moonwalk_teacher_sole_tracking(
    env, teacher_path, xy_scale=0.06, z_scale=0.02, command_name="ref_motion"
):
    command, bank, target = resolve(env, teacher_path, command_name)
    xy, z, _ = bank.actual_soles(env, command)
    return sole_score(
        xy,
        z,
        target["sole_relative_xy"],
        target["sole_ground_z"],
        target["quality"],
        xy_scale,
        z_scale,
    )


def moonwalk_teacher_slide_mechanism(
    env,
    teacher_path,
    sensor_name="contact_forces",
    velocity_scale=0.20,
    window_steps=8,
    command_name="ref_motion",
):
    command, bank, target = resolve(env, teacher_path, command_name)
    _, z, velocity = bank.actual_soles(env, command)
    sensor = env.scene.sensors[sensor_name]
    sensor_ids = [
        sensor.body_names.index(side + "_ankle_roll_link")
        for side in ("left", "right")
    ]
    forces = sensor.data.net_forces_w[:, sensor_ids].norm(dim=-1)
    score = mechanism_score(
        z,
        velocity,
        forces,
        target["sole_ground_z"],
        target["sole_velocity_heading_xy"],
        target["role_code"],
        target["quality"],
        velocity_scale,
    )
    frames = (
        (command._frame_indices + 1)
        .clamp(max=bank.metadata["frames"] - 1)
        .long()
    )
    result, state = windowed_bonus(
        score,
        frames,
        target["role_code"],
        getattr(bank, "window_state", None),
        window_steps,
        reset_mask=env.episode_length_buf <= 1,
    )
    bank.window_state = state
    return result
