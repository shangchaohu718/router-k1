import torch

from holomotion.src.motion_tracking.moonwalk_rewards import (
    heel_contact_score,
    sole_pose_score,
    sole_velocity_score,
)


def points():
    return torch.tensor(
        [
            [
                [[0.08, 0.1, 0.0], [-0.06, 0.1, 0.04]],
                [[0.08, -0.1, 0.0], [-0.06, -0.1, 0.0]],
            ]
        ]
    )


def contact_score(actual, ref, force=None, ground=None):
    return heel_contact_score(
        actual,
        ref,
        torch.full((1, 2), 100.0) if force is None else force,
        torch.zeros(1) if ground is None else ground,
        0.015,
        0.015,
        0.025,
        5.0,
        2.0,
    )


def test_exact_reference_is_optimal():
    p = points()
    torch.testing.assert_close(
        sole_pose_score(p, p, p, p, 0.06, 0.015), torch.ones(1)
    )
    torch.testing.assert_close(contact_score(p, p), torch.ones(1))


def test_intended_slide_is_rewarded_over_stationary_or_reversed_feet():
    ref = torch.zeros_like(points())
    ref[:, 1, :, 0] = -0.6
    exact = sole_velocity_score(ref, ref, 0.3)
    stopped = sole_velocity_score(torch.zeros_like(ref), ref, 0.3)
    reversed_slide = sole_velocity_score(-ref, ref, 0.3)
    assert exact > stopped > reversed_slide


def test_absolute_body_relative_offset_is_not_cancelled_at_slide_onset():
    ref = points()
    wrong = ref.clone()
    wrong[:, 0, :, 0] -= 0.12
    assert sole_pose_score(wrong, ref, wrong, ref, 0.06, 0.015) < 0.7


def test_whole_foot_lift_cannot_fake_a_contacting_toe_pivot():
    ref = points()
    lifted = ref + torch.tensor([0.0, 0.0, 0.08])
    assert contact_score(lifted, ref, torch.zeros(1, 2)) < 0.5
    assert sole_pose_score(lifted, ref, lifted, ref, 0.06, 0.015) < 0.1


def test_heel_lift_follows_reference_side_without_an_external_phase_clock():
    first = points()
    second = first.flip(1).clone()
    assert contact_score(second, second) > contact_score(first, second)


def test_world_translation_and_environment_ground_offset_do_not_change_scores():
    ref = points()
    actual = ref.clone()
    actual[:, 0, 1, 2] += 0.01
    shift = torch.tensor([32.0, -17.0, 3.0])
    torch.testing.assert_close(
        contact_score(actual, ref),
        contact_score(actual + shift, ref + shift, ground=torch.tensor([3.0])),
    )
    torch.testing.assert_close(
        sole_pose_score(actual, ref, actual, ref, 0.06, 0.015),
        sole_pose_score(actual, ref, actual + shift, ref + shift, 0.06, 0.015),
    )
