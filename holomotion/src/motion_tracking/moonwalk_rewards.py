"""Reference-conditioned sole rewards; pure tensor math without simulator imports.

Sole tensors are [batch, left/right, toe/heel, xyz]. Targets come from the
current reference, so random start frames cannot desynchronize a phase clock.
"""

import torch


def sole_pose_score(
    actual_relative: torch.Tensor,
    reference_relative: torch.Tensor,
    actual_world: torch.Tensor,
    reference_world: torch.Tensor,
    xy_scale: float,
    z_scale: float,
) -> torch.Tensor:
    if min(xy_scale, z_scale) <= 0:
        raise ValueError("Sole pose scales must be positive")
    xy_error = (
        ((actual_relative[..., :2] - reference_relative[..., :2]) / xy_scale)
        .square()
        .sum(-1)
    )
    z_error = (
        (actual_world[..., 2] - reference_world[..., 2]) / z_scale
    ).square()
    return (1.0 / (1.0 + xy_error + z_error)).mean(dim=(-1, -2))


def sole_velocity_score(
    actual_velocity: torch.Tensor,
    reference_velocity: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Track nonzero sliding velocity and reference support drift alike."""
    if scale <= 0:
        raise ValueError("Sole velocity scale must be positive")
    error = (
        ((actual_velocity[..., :2] - reference_velocity[..., :2]) / scale)
        .square()
        .sum(-1)
    )
    return (1.0 / (1.0 + error)).mean(dim=(-1, -2))


def heel_contact_score(
    actual_points: torch.Tensor,
    reference_points: torch.Tensor,
    contact_force: torch.Tensor,
    ground_z: torch.Tensor,
    lift_scale: float,
    toe_scale: float,
    contact_height: float,
    force_threshold: float,
    force_temperature: float,
) -> torch.Tensor:
    """Independent heel gap, toe height and whole-foot contact signals.

    Whole-foot force does not establish a measured toe/heel force split.
    Do not multiply all signals: coarse lift improvements should still count
    before contact and toe height are perfect.
    """
    if min(lift_scale, toe_scale, force_temperature) <= 0:
        raise ValueError("Heel/contact scales must be positive")
    actual_gap = actual_points[:, :, 1, 2] - actual_points[:, :, 0, 2]
    reference_gap = reference_points[:, :, 1, 2] - reference_points[:, :, 0, 2]
    lift = 1.0 / (1.0 + ((actual_gap - reference_gap) / lift_scale).square())
    toe = 1.0 / (
        1.0
        + (
            (actual_points[:, :, 0, 2] - reference_points[:, :, 0, 2])
            / toe_scale
        ).square()
    )
    expected_contact = (
        reference_points[..., 2].amin(-1) - ground_z[:, None] <= contact_height
    )
    measured_contact = torch.sigmoid(
        (contact_force - force_threshold) / force_temperature
    )
    contact = torch.where(
        expected_contact, measured_contact, 1.0 - measured_contact
    )
    return ((lift + toe + contact) / 3.0).mean(-1)
