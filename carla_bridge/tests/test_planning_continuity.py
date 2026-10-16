"""Physical-time continuity costs do not depend on action names or ego frames."""
import math

import numpy as np
import pytest

from carla_bridge.planning.continuity import TrajectoryReference, trajectory_continuity_costs


def straight_reference(timestamp_us=1000000, dt=0.1, count=41):
    times = np.arange(count) * dt
    states = np.column_stack((2 * times, np.zeros(count), np.zeros(count)))
    return TrajectoryReference(states, np.full(count, 2.0), timestamp_us, dt)


def evaluate(states, speeds, previous, **kwargs):
    return trajectory_continuity_costs(states, speeds, previous=previous,
                                       timestamp_us=1200000, sample_interval_s=0.1, **kwargs)


def test_receding_curved_plan_is_aligned_in_absolute_time_not_sample_index():
    old_times = np.arange(21) * 0.2
    angles = old_times * 0.2
    old_states = np.column_stack((10 * np.sin(angles), 10 * (1 - np.cos(angles)), angles))
    previous = TrajectoryReference(old_states, np.full(21, 2.0), 1000000, 0.2)
    new_times = 0.2 + np.arange(21) * 0.1
    states = np.column_stack([np.interp(new_times, old_times, old_states[:, i]) for i in range(3)])
    costs, context = evaluate(states[None], np.full((1, 21), 2.0), previous)
    assert costs[0] == pytest.approx(0, abs=1e-12)
    assert context["compared_samples"] == 10
    assert context["reference_age_s"] == pytest.approx(0.2)


def test_equivalent_heading_across_pi_boundary_has_no_artificial_turn_penalty():
    previous = straight_reference()
    previous.states[:, 2] = math.pi - 0.1 + np.arange(41) * 0.02
    states = previous.states[2:23].copy()
    states[:, 2] = (states[:, 2] + math.pi) % (2 * math.pi) - math.pi
    costs, context = evaluate(states[None], np.full((1, 21), 2.0), previous)
    assert costs[0] == pytest.approx(0, abs=1e-12)
    assert context["candidates"][0]["mean_heading_deviation_rad"] < 1e-12


def test_identical_physical_candidates_have_equal_cost_without_label_or_id_state():
    previous = straight_reference()
    states = np.repeat(previous.states[2:23][None], 2, axis=0)
    states[:, :, 1] = 0.2
    speeds = np.full((2, 21), 2.0)
    original = states.copy()
    first, first_context = evaluate(states, speeds, previous)
    second, second_context = evaluate(states, speeds, previous)
    assert 0 < first[0] == first[1] < 0.15
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(states, original)
    assert first_context == second_context


def test_cost_is_invariant_to_world_translation_and_rotation():
    previous = straight_reference()
    states = previous.states[2:23].copy()
    states[:, 1] += 0.3
    speeds = np.full((1, 21), 3.0)
    expected, _ = evaluate(states[None], speeds, previous)
    angle = 1.2
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    moved_previous = previous.states.copy()
    moved_states = states.copy()
    for poses in (moved_previous, moved_states):
        poses[:, :2] = poses[:, :2] @ rotation.T + np.array([123.0, -456.0])
        poses[:, 2] += angle
    moved_reference = TrajectoryReference(moved_previous, previous.speeds, previous.timestamp_us, 0.1)
    actual, _ = evaluate(moved_states[None], speeds, moved_reference)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_future_divergence_outside_overlap_does_not_penalize_current_continuation():
    previous = straight_reference()
    states = previous.states[2:23].copy()
    speeds = np.full((1, 21), 2.0)
    states[11:, 1] = 10.0
    speeds[:, 11:] = 10.0
    costs, _ = evaluate(states[None], speeds, previous)
    assert costs[0] == pytest.approx(0)


def test_large_physical_change_has_bounded_soft_cost():
    previous = straight_reference()
    states = previous.states[2:23].copy()
    states[:, 1] += 20
    states[:, 2] += math.pi / 2
    costs, _ = evaluate(states[None], np.full((1, 21), 20.0), previous, weight=0.15)
    assert costs[0] == pytest.approx(0.15)


@pytest.mark.parametrize("timestamp,reason", [(0, "stale_previous"), (1300000, "clock_rollback")])
def test_missing_physical_time_context_disables_continuity(timestamp, reason):
    previous = straight_reference(timestamp_us=timestamp)
    costs, context = evaluate(previous.states[None], previous.speeds[None], previous)
    assert costs.tolist() == [0.0]
    assert context["reason"] == reason
    assert context["applied"] is False


def test_startup_and_short_expired_reference_do_not_extrapolate_a_plan():
    previous = straight_reference(count=3)
    states = np.zeros((1, 11, 3))
    speeds = np.zeros((1, 11))
    for reference, expected in ((None, "no_previous"), (previous, "no_future_overlap")):
        costs, context = evaluate(states, speeds, reference)
        assert costs.tolist() == [0.0]
        assert context["reason"] == expected


def test_nonfinite_reference_is_ignored_and_nonfinite_candidate_does_not_contaminate_others():
    previous = straight_reference()
    states = np.repeat(previous.states[2:23][None], 2, axis=0)
    states[0, 4, 0] = np.nan
    states[1, :, 1] = 0.3
    speeds = np.full((2, 21), 2.0)
    costs, context = evaluate(states, speeds, previous)
    assert costs[0] == 0 and costs[1] > 0
    assert context["ignored_non_finite_candidates"] == [0]
    previous.speeds[4] = np.inf
    costs, context = evaluate(states, speeds, previous)
    assert costs.tolist() == [0.0, 0.0]
    assert context["reason"] == "non_finite_or_invalid_previous"
