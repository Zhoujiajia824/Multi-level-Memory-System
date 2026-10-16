"""Control-time speed profiles without a running CARLA server."""
import copy

import pytest

from carla_bridge.control.trajectory_tracker import TrajectoryTracker


def tracker(dt=0.1):
    return TrajectoryTracker(
        {"pid": {"kp": 1.0, "ki": 0.0, "kd": 0.0, "deadband_mps": 0.01}},
        control_dt_s=dt,
    )


def trajectory(speeds, times=None, y=0.0):
    times = times if times is not None else [float(i) for i in range(len(speeds))]
    return [{"t": t, "x": float(i * 5), "y": y, "optional_v": v}
            for i, (t, v) in enumerate(zip(times, speeds))]


def set_profile(controller, points, target_speed=99.0):
    controller.set_trajectory(points, 0.0, 0.0, 0.0, target_speed,
                              follow_speed_profile=True)


def test_zero_initial_speed_can_start_and_interpolates_next_tick():
    controller = tracker()
    set_profile(controller, trajectory([0.0, 2.0, 4.0]))
    control = controller.compute_control(0.0, 0.0, 0.0, 0.0)
    assert control.throttle == pytest.approx(0.2)
    assert control.brake == 0.0
    control = controller.compute_control(0.0, 0.0, 0.0, 0.0)
    assert control.throttle == pytest.approx(0.4)


def test_decelerating_profile_uses_time_speed_instead_of_final_target():
    controller = tracker()
    set_profile(controller, trajectory([5.0, 0.0, 0.0]), target_speed=0.0)
    control = controller.compute_control(0.0, 0.0, 0.0, 5.0)
    assert control.throttle == 0.0
    assert control.brake == pytest.approx(0.5)


def test_profile_expires_and_does_not_keep_terminal_speed():
    controller = tracker(dt=0.25)
    set_profile(controller, trajectory([2.0, 2.0], [0.0, 0.5]))
    assert controller.compute_control(0.0, 0.0, 0.0, 1.0).throttle > 0.0
    assert controller.compute_control(0.0, 0.0, 0.0, 1.0).throttle > 0.0
    expired = controller.compute_control(0.0, 0.0, 0.0, 1.0)
    assert expired.brake == 1.0
    assert expired.throttle == 0.0


def test_new_plan_resets_profile_clock_after_expiry():
    controller = tracker(dt=0.5)
    points = trajectory([0.0, 1.0])
    set_profile(controller, points)
    for _ in range(3):
        controller.compute_control(0.0, 0.0, 0.0, 0.0)
    assert controller.last_control_diagnostics["reason"] == "profile_expired"
    assert controller.pid._i == 0.0
    set_profile(controller, points)
    assert controller._profile_elapsed_s == 0.0
    assert controller.pid._prev_err is None
    assert controller.compute_control(0.0, 0.0, 0.0, 0.0).throttle == pytest.approx(0.5)


def test_stop_profile_without_forward_point_brakes_until_stopped():
    controller = tracker()
    points = trajectory([0.0, 0.0, 0.0])
    for point in points:
        point["x"] = 0.0
    set_profile(controller, points)
    controller.pid._i = 5.0
    for speed in [1.0, 0.1, 0.0]:
        control = controller.compute_control(0.0, 0.0, 0.0, speed)
        assert control.brake == 1.0
        assert control.throttle == 0.0


def test_nonzero_profile_without_forward_point_still_brakes():
    controller = tracker()
    set_profile(controller, trajectory([5.0, 5.0]))
    control = controller.compute_control(6.0, 0.0, 0.0, 0.0)
    assert control.brake == 1.0
    assert control.throttle == 0.0


def test_profile_preserves_left_positive_ego_coordinates():
    controller = tracker()
    set_profile(controller, trajectory([5.0, 5.0, 5.0], y=1.0))
    assert controller.compute_control(0.0, 0.0, 0.0, 5.0).steer < 0.0


def test_legacy_constant_target_ignores_profile_fields_and_does_not_expire():
    controller = tracker(dt=1.0)
    points = [{"t": "ignored", "x": 0.0, "y": 0.0, "optional_v": float("nan")},
              {"x": 10.0, "y": 0.0}]
    controller.set_trajectory(points, 0.0, 0.0, 0.0, 0.5)
    for _ in range(4):
        control = controller.compute_control(0.0, 0.0, 0.0, 0.0)
        assert control.throttle == pytest.approx(0.5)


def test_returning_to_constant_target_disables_previous_speed_profile():
    controller = tracker(dt=1.0)
    points = trajectory([0.0, 0.0])
    set_profile(controller, points)
    controller.set_trajectory(points, 0.0, 0.0, 0.0, 0.5)
    for _ in range(3):
        assert controller.compute_control(0.0, 0.0, 0.0, 0.0).throttle == pytest.approx(0.5)


@pytest.mark.parametrize("field,value", [
    ("t", float("nan")), ("t", float("inf")), ("t", -1.0),
    ("t", 0.0), ("t", None),
    ("optional_v", float("nan")), ("optional_v", float("inf")),
    ("optional_v", -1.0), ("optional_v", None),
])
def test_invalid_profile_clears_stale_plan(field, value):
    controller = tracker()
    points = trajectory([1.0, 1.0])
    set_profile(controller, points)
    invalid = copy.deepcopy(points)
    invalid[1][field] = value
    with pytest.raises(ValueError):
        set_profile(controller, invalid)
    assert not controller.has_trajectory()
    assert controller.compute_control(0.0, 0.0, 0.0, 0.0).brake == 1.0


def test_missing_speed_rejected_but_empty_profile_stops():
    controller = tracker()
    with pytest.raises(ValueError):
        set_profile(controller, [{"t": 0.0, "x": 0.0, "y": 0.0}])
    set_profile(controller, [])
    assert controller.compute_control(0.0, 0.0, 0.0, 1.0).brake == 1.0


def test_profile_replan_preserves_measured_velocity_feedback():
    cfg = {"pid": {"kp": 0.5, "ki": 0.0, "kd": 0.1, "deadband_mps": 0.0}}
    uninterrupted = TrajectoryTracker(cfg, control_dt_s=0.1)
    replanned = TrajectoryTracker(cfg, control_dt_s=0.1)
    points = trajectory([2.0, 2.0, 2.0])
    for controller in (uninterrupted, replanned):
        set_profile(controller, points)
        controller.compute_control(0.0, 0.0, 0.0, 1.0)
    set_profile(replanned, points)
    expected = uninterrupted.compute_control(0.1, 0.0, 0.0, 1.2)
    actual = replanned.compute_control(0.1, 0.0, 0.0, 1.2)
    assert actual.throttle == pytest.approx(expected.throttle)
    assert actual.throttle == pytest.approx(0.2)
    assert actual.brake == expected.brake == 0.0


def test_zero_speed_with_valid_forward_path_brakes_without_emergency_jump():
    controller = tracker()
    set_profile(controller, trajectory([0.2, 0.0, 0.0], times=[0.0, 0.1, 0.2]))
    control = controller.compute_control(0.0, 0.0, 0.0, 0.2)
    assert control.throttle == 0.0
    assert control.brake == pytest.approx(0.2)
    assert controller.last_control_diagnostics["reason"] == "planned_stop"
    assert controller.last_control_diagnostics["target_speed_mps"] == 0.0
    control = controller.compute_control(0.01, 0.0, 0.0, 0.01)
    assert control.brake == 1.0
    assert controller.last_control_diagnostics["reason"] == "stationary_hold"


@pytest.mark.parametrize("failure", ["empty", "expired", "past_endpoint"])
def test_failsafe_brakes_immediately_and_discards_old_throttle_state(failure):
    controller = tracker(dt=0.1)
    set_profile(controller, trajectory([2.0, 2.0], times=[0.0, 0.1]))
    controller.compute_control(0.0, 0.0, 0.0, 0.0)
    controller.pid._i = 4.0
    x = 0.0
    if failure == "empty":
        set_profile(controller, [])
    elif failure == "past_endpoint":
        set_profile(controller, trajectory([2.0, 2.0]))
        x = 6.0
    control = controller.compute_control(x, 0.0, 0.0, 1.0)
    assert control.brake == 1.0 and control.throttle == 0.0
    assert controller.pid._i == 0.0
    assert controller.pid._prev_measurement is None
    assert controller.last_control_diagnostics["reason"] in {
        "no_trajectory", "profile_expired", "no_forward_target"}


def test_control_diagnostics_report_target_at_executed_tick():
    controller = tracker()
    set_profile(controller, trajectory([0.0, 2.0, 4.0]))
    control = controller.compute_control(0.0, 0.0, 0.0, 0.0)
    diagnostics = controller.last_control_diagnostics
    assert diagnostics["reason"] == "speed_tracking"
    assert diagnostics["profile_elapsed_s"] == pytest.approx(0.1)
    assert diagnostics["target_speed_mps"] == pytest.approx(0.2)
    assert diagnostics["speed_error_mps"] == pytest.approx(0.2)
    assert diagnostics["throttle"] == pytest.approx(control.throttle)


def stopping_profile(deceleration, initial_speed=4.0):
    """A kinematically consistent stop, followed by repeated stationary points."""
    points = []
    for index in range(41):
        t = index * 0.1
        speed = max(0.0, initial_speed - deceleration * t)
        distance = (initial_speed ** 2 - speed ** 2) / (2.0 * deceleration)
        points.append({"t": t, "x": distance, "y": 0.0, "optional_v": speed})
    return points


@pytest.mark.parametrize("deceleration", [2.0, 4.0])
@pytest.mark.parametrize("replan", [False, True])
def test_realistic_stop_tail_keeps_forward_target_until_stationary(deceleration, replan):
    # Ideal measured motion isolates trajectory/endpoint handling from vehicle
    # brake calibration; it is not evidence of CARLA's real braking performance.
    controller = TrajectoryTracker(
        {"pid": {"kp": 1.6, "ki": 0.05, "kd": 0.05, "deadband_mps": 0.15}},
        control_dt_s=0.1,
    )
    points = stopping_profile(deceleration)
    set_profile(controller, points, target_speed=0.0)
    stop_index = int(4.0 / deceleration / 0.1)
    for index in range(stop_index + 3):
        point = points[index]
        if replan and index > 0 and index % 2 == 0:
            remaining = [{**sample, "t": sample["t"] - point["t"],
                          "x": sample["x"] - point["x"]}
                         for sample in points[index:]]
            controller.set_trajectory(remaining, point["x"], 0.0, 0.0, 0.0,
                                      follow_speed_profile=True)
        control = controller.compute_control(point["x"], 0.0, 0.0, point["optional_v"])
        assert control.throttle == 0.0
        if point["optional_v"] > 0.05:
            assert 0.0 < control.brake < 1.0
            assert controller.last_control_diagnostics["endpoint_forward_m"] > 0.001
            assert controller.last_control_diagnostics["profile_acceleration_mps2"] == pytest.approx(-deceleration)
        else:
            assert control.brake == 1.0
            assert controller.last_control_diagnostics["reason"] == "stationary_hold"


def test_stronger_stop_profile_requests_more_brake_before_stationary_hold():
    brakes = []
    for deceleration in (2.0, 4.0):
        controller = TrajectoryTracker({"pid": {"kp": 1.6}}, control_dt_s=0.1)
        set_profile(controller, stopping_profile(deceleration), target_speed=0.0)
        brakes.append(controller.compute_control(0.0, 0.0, 0.0, 4.0).brake)
    assert 0.0 < brakes[0] < brakes[1] < 1.0


@pytest.mark.parametrize("acceleration", [-0.5, 0.5])
def test_gentle_speed_change_is_not_erased_by_cruise_deadband(acceleration):
    controller = TrajectoryTracker(
        {"pid": {"kp": 1.0, "ki": 0.0, "kd": 0.0, "deadband_mps": 0.15}},
        control_dt_s=0.1,
    )
    points = [{"t": t, "x": 4.0 * t + 0.5 * acceleration * t ** 2,
               "y": 0.0, "optional_v": 4.0 + acceleration * t}
              for t in (0.0, 0.1, 0.2, 1.0)]
    set_profile(controller, points)
    control = controller.compute_control(0.0, 0.0, 0.0, 4.0)
    assert (control.brake if acceleration < 0.0 else control.throttle) == pytest.approx(0.05)
    assert controller.last_control_diagnostics["profile_acceleration_mps2"] == pytest.approx(acceleration)


def test_constant_cruise_profile_still_coasts_in_deadband():
    controller = TrajectoryTracker(
        {"pid": {"kp": 1.0, "ki": 0.0, "kd": 0.0, "deadband_mps": 0.15}},
        control_dt_s=0.1,
    )
    set_profile(controller, trajectory([4.0, 4.0, 4.0]))
    control = controller.compute_control(0.0, 0.0, 0.0, 3.95)
    assert control.throttle == control.brake == 0.0


def test_stop_endpoint_overshoot_still_full_brakes_with_diagnostics():
    controller = tracker()
    points = stopping_profile(2.0)
    set_profile(controller, points)
    control = controller.compute_control(points[-1]["x"] + 0.1, 0.0, 0.0, 0.3)
    assert control.throttle == 0.0 and control.brake == 1.0
    assert controller.last_control_diagnostics["reason"] == "no_forward_target"
    assert controller.last_control_diagnostics["endpoint_forward_m"] == pytest.approx(-0.1)
    assert controller.last_control_diagnostics["endpoint_distance_m"] == pytest.approx(0.1)
