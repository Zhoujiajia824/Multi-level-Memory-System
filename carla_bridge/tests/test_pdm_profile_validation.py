"""A PDM terminal stop is not a zero speed cap for its entire trajectory."""
from copy import deepcopy

import pytest

from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.control.trajectory_validation import TrajectoryValidator


def _accelerate_then_stop():
    """Rest -> +2 m/s2 for 8/3 s -> -4 m/s2 -> rest at t=4 s."""
    switch = 8.0 / 3.0
    points = []
    for index in range(41):
        t = index / 10.0
        if t <= switch:
            x, speed = t * t, 2 * t
        else:
            elapsed = t - switch
            x = switch ** 2 + 2 * switch * elapsed - 2 * elapsed ** 2
            speed = max(0.0, 2 * switch - 4 * elapsed)
        points.append({"t": t, "x": x, "y": 0.0, "optional_v": speed})
    return points


def _stationary():
    return [{"t": index / 10, "x": 0.0, "y": 0.0, "optional_v": 0.0}
            for index in range(41)]


def _closed(backend="pdm_closed"):
    # Exercise production validation / stop detection without CARLA setup.
    closed = ClosedLoop.__new__(ClosedLoop)
    closed.planner_backend = backend
    closed.trajectory_validator = TrajectoryValidator()
    closed._execution_horizon = 3.0
    closed._pdm_interval = 0.2
    closed.fixed_delta = 0.1
    closed._stuck_speed_thresh = 0.3
    return closed


def test_profile_accepts_acceleration_then_stop_with_zero_terminal_speed():
    points = _accelerate_then_stop()
    before = deepcopy(points)
    result = TrajectoryValidator().validate(
        points, target_speed=0.0, current_speed=0.0, required_horizon_s=3.0,
        use_speed_profile=True,
    )
    assert result.accepted, result.to_dict()
    assert result.details["path_length_m"] == pytest.approx(32 / 3, abs=1e-4)
    assert result.details["speed_profile_peak_mps"] > 5.0
    assert points == before


def test_legacy_keeps_target_speed_cap_without_profile_opt_in():
    result = TrajectoryValidator().validate(_accelerate_then_stop(), 0.0, 0.0, 3.0)
    assert not result.accepted
    assert "unreachable_distance" in result.reasons
    assert "distance_speed_mismatch" in result.reasons


@pytest.mark.parametrize("value,reason", [
    (None, "missing_speed_profile"),
    (float("nan"), "invalid_waypoint_speed"),
    (float("inf"), "invalid_waypoint_speed"),
    (True, "invalid_waypoint_speed"),
    (False, "invalid_waypoint_speed"),
])
def test_profile_mode_rejects_missing_nonfinite_and_boolean_speeds(value, reason):
    points = _accelerate_then_stop()
    if value is None:
        points[7].pop("optional_v")
    else:
        points[7]["optional_v"] = value
    result = TrajectoryValidator().validate(points, 0.0, 0.0, 3.0, use_speed_profile=True)
    assert not result.accepted
    assert reason in result.reasons


def test_profile_peak_does_not_authorize_instantaneous_acceleration():
    # Correct position/speed trapezoids, but an impossible 0 -> 8 m/s in 0.1 s.
    points = [{"t": 0.0, "x": 0.0, "y": 0.0, "optional_v": 0.0}]
    points.extend({"t": index / 10, "x": 0.4 + 0.8 * (index - 1),
                   "y": 0.0, "optional_v": 8.0} for index in range(1, 41))
    result = TrajectoryValidator().validate(points, 8.0, 0.0, 3.0, use_speed_profile=True)
    assert not result.accepted
    assert "waypoint_acceleration_inconsistent" in result.reasons
    assert "unreachable_distance" in result.reasons


@pytest.mark.parametrize("backend,accepted", [("pdm_closed", True), ("vlm_trajectory", False)])
def test_closed_loop_only_enables_profile_cap_for_pdm(backend, accepted):
    validation = _closed(backend)._validate_decision(
        {"trajectory": _accelerate_then_stop(), "target_speed": 0.0},
        {"ego_state": {"speed": 0.0}},
    )
    assert validation["accepted"] is accepted


def test_terminal_stop_does_not_disable_stuck_detection_during_near_term_acceleration():
    decision = {"trajectory": _accelerate_then_stop(), "target_speed": 0.0}
    assert decision["trajectory"][-1]["optional_v"] == pytest.approx(0.0)
    assert _closed()._is_planned_stop(decision) is False


def test_stationary_profile_is_a_planned_stop():
    assert _closed()._is_planned_stop({"trajectory": _stationary(), "target_speed": 0.0}) is True


def test_stop_detection_includes_next_planning_interval_plus_one_tick():
    points = _stationary()
    points[3]["optional_v"] = 0.4  # t=0.3 s is inside the next control window.
    assert _closed()._is_planned_stop({"trajectory": points, "target_speed": 0.0}) is False
    points[3]["optional_v"] = 0.0
    points[4]["optional_v"] = 0.4  # t=0.4 s is outside this window.
    assert _closed()._is_planned_stop({"trajectory": points, "target_speed": 0.0}) is True


def test_legacy_planned_stop_still_uses_target_speed():
    decision = {"trajectory": _accelerate_then_stop(), "target_speed": 0.0}
    assert _closed("vlm_trajectory")._is_planned_stop(decision) is True
