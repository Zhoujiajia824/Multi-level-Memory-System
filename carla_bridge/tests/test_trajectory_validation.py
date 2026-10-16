"""Physical sanity checks for VLM trajectories (no CARLA server required)."""
import math

import pytest

from carla_bridge.control.trajectory_validation import TrajectoryValidator


def _straight(speed=6.0, duration=3.0):
    return [{"t": i * duration / 20, "x": speed * i * duration / 20, "y": 0.0}
            for i in range(21)]


def _validate(points, target=6.0, current=6.0, horizon=3.0, config=None):
    return TrajectoryValidator(config).validate(points, target, current, horizon)


def test_valid_constant_speed_and_no_input_mutation():
    points = _straight()
    original = [dict(point) for point in points]
    result = _validate(points)
    assert result.accepted, result.to_dict()
    assert result.details["path_length_m"] == 18.0
    assert points == original


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_valid_ninety_degree_turns(side):
    radius = 12.0
    speed = math.pi * radius / 6.0
    points = [{"t": i * 0.15, "x": radius * math.sin(speed * i * 0.15 / radius),
               "y": side * radius * (1 - math.cos(speed * i * 0.15 / radius))}
              for i in range(21)]
    result = _validate(points, target=speed, current=speed)
    assert result.accepted, result.to_dict()


def test_target_speed_does_not_invalidate_feasible_deceleration():
    # Current speed is 10 m/s, desired speed is 2 m/s: 18 m is physically
    # correct over 3 s, despite target_speed * horizon being only 6 m.
    points = []
    for i in range(21):
        t = i * 0.15
        points.append({"t": t, "x": 10 * t - 0.5 * (8 / 3) * t ** 2,
                       "y": 0.0, "optional_v": 10 - (8 / 3) * t})
    result = _validate(points, target=2, current=10)
    assert result.accepted, result.to_dict()


def test_valid_acceleration_from_rest():
    points = []
    for i in range(21):
        t = i * 0.15
        points.append({"t": t, "x": t ** 2, "y": 0.0, "optional_v": 2 * t})
    result = _validate(points, target=6, current=0)
    assert result.accepted, result.to_dict()


def test_zero_speed_stationary_plan():
    result = _validate(_straight(0), target=0, current=0)
    assert result.accepted, result.to_dict()


def test_zero_target_speed_with_real_braking_distance():
    points = []
    for i in range(21):
        t = i * 0.15
        braking_t = min(t, 1.5)
        points.append({"t": t, "x": 6 * braking_t - 2 * braking_t ** 2,
                       "y": 0.0, "optional_v": max(0.0, 6 - 4 * t)})
    result = _validate(points, target=0, current=6)
    assert result.accepted, result.to_dict()


def test_reported_83_metre_in_three_seconds_plan_is_rejected():
    points = [{"t": i * 0.15, "x": 53.5 * i / 20, "y": -63.9 * i / 20}
              for i in range(21)]
    result = _validate(points, target=6, current=6)
    assert not result.accepted
    assert "distance_speed_mismatch" in result.reasons
    assert "unreachable_distance" in result.reasons


@pytest.mark.parametrize("key,value", [("x", float("nan")), ("y", float("inf")),
                                       ("t", float("-inf")), ("x", True), ("y", "1.0")])
def test_non_numeric_or_nonfinite_waypoints(key, value):
    points = _straight()
    points[5][key] = value
    result = _validate(points)
    assert not result.accepted
    assert "nonfinite_or_missing_coordinate" in result.reasons


@pytest.mark.parametrize("points", [None, [], {}, ["bad"], [{"x": 1.0, "y": 0.0}]])
def test_malformed_trajectory(points):
    assert not _validate(points).accepted


@pytest.mark.parametrize("speed", [float("nan"), float("inf"), -1, True, "6", None])
def test_invalid_target_speed(speed):
    result = _validate(_straight(), target=speed)
    assert not result.accepted
    assert "invalid_target_speed" in result.reasons


@pytest.mark.parametrize("speed", [float("nan"), float("inf"), -1, True])
def test_invalid_waypoint_speed(speed):
    points = _straight()
    points[5]["optional_v"] = speed
    assert "invalid_waypoint_speed" in _validate(points).reasons


@pytest.mark.parametrize("timestamp", [-1.0, 0.0, 0.15])
def test_non_monotonic_time(timestamp):
    points = _straight()
    points[2]["t"] = timestamp
    assert "non_monotonic_time" in _validate(points).reasons


def test_horizon_must_cover_actual_control_ticks():
    result = _validate(_straight(duration=1), horizon=3)
    assert "insufficient_horizon" in result.reasons


def test_future_first_point_does_not_require_explicit_zero_anchor():
    result = _validate(_straight()[1:])
    assert result.accepted, result.to_dict()


def test_cannot_start_trajectory_far_from_capture_pose():
    points = _straight()
    for point in points:
        point["x"] += 10
    assert "nonzero_origin" in _validate(points).reasons


def test_acceleration_from_rest_cannot_jump_to_six_metres_per_second():
    result = _validate(_straight(), current=0)
    assert not result.accepted
    assert "excessive_acceleration" in result.reasons


def test_current_motion_cannot_disappear_instantly():
    result = _validate(_straight(0), current=10, target=0)
    assert not result.accepted
    assert "instantaneous_stop" in result.reasons


def test_vehicle_already_above_limit_can_follow_a_braking_plan():
    points = [{"t": i * 0.15, "x": 28 * i * 0.15 - (i * 0.15) ** 2, "y": 0.0}
              for i in range(21)]
    result = _validate(points, target=22, current=28)
    assert result.accepted, result.to_dict()


def test_reverse_is_not_supported_by_forward_controller():
    result = _validate(_straight(-3), target=3, current=3)
    assert "not_forward_drivable" in result.reasons


def test_path_cusp_is_not_a_forward_turn():
    points = [{"t": t, "x": x, "y": 0.0}
              for t, x in [(0, 0), (1, 3), (2, 6), (3, 3)]]
    assert "direction_reversal" in _validate(points, target=3, current=3).reasons


def test_sharp_high_speed_corner_is_rejected():
    points = _straight(12)
    for point in points[11:]:
        point["y"] = point["x"] - 18
        point["x"] = 18
    result = _validate(points, target=12, current=12)
    assert "excessive_lateral_acceleration" in result.reasons


def test_optional_speed_must_match_waypoint_distances():
    points = _straight()
    for point in points:
        point["optional_v"] = 1
    assert "waypoint_speed_inconsistent" in _validate(points).reasons


def test_limits_are_configurable_and_invalid_limits_fail_early():
    assert "target_speed_exceeds_limit" in _validate(_straight(), config={"max_speed_mps": 5}).reasons
    for value in (0, -1, float("nan"), True):
        with pytest.raises(ValueError):
            TrajectoryValidator({"max_acceleration_mps2": value})


def test_rejection_diagnostics_are_strict_json_serializable():
    import json
    result = _validate([{ "t": 1, "x": float("nan"), "y": 0}])
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("point", [{"t": 3, "x": 10 ** 1000, "y": 0},
                                   {"t": 1e308, "x": 0, "y": 0},
                                   {"t": 1e-308, "x": 1e308, "y": 1e308}])
def test_extreme_json_numbers_fail_closed_without_nonfinite_diagnostics(point):
    import json
    result = _validate([point], current=0)
    assert not result.accepted
    json.dumps(result.to_dict(), allow_nan=False)
