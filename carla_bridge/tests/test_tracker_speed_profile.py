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


def test_new_plan_resets_profile_clock_and_pid_derivative():
    controller = tracker(dt=0.5)
    points = trajectory([0.0, 1.0])
    set_profile(controller, points)
    for _ in range(3):
        controller.compute_control(0.0, 0.0, 0.0, 0.0)
    controller.pid._i = 2.0
    set_profile(controller, points)
    assert controller.pid._i == 2.0
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
