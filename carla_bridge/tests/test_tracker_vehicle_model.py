"""Vehicle-specific steering conversion; no running CARLA server is required."""
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock

import carla
import pytest

from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.control.steering_model import SteeringModel
from carla_bridge.control.trajectory_tracker import TrajectoryTracker
from carla_bridge.state.ego_state_provider import EgoStateProvider


def test_steering_curve_uses_kmh_and_clamps_to_endpoint_knots():
    model = SteeringModel(1.2, [(0, 1), (36, .5), (72, .25)])
    assert model.limit_at(0) == pytest.approx(1.2)
    assert model.limit_at(5) == pytest.approx(.9)
    assert model.limit_at(10) == pytest.approx(.6)
    assert model.limit_at(100) == pytest.approx(.3)
    assert model.limit_at(-5) == pytest.approx(.9)


@pytest.mark.parametrize("curve", [[(0, 0)], [(0, float('nan'))],
                                  [(10, 1), (0, .5)], [(0, 1), (0, .5)]])
def test_invalid_physics_mapping_is_not_silently_accepted(curve):
    with pytest.raises(ValueError):
        SteeringModel(1.2, curve)


def test_binding_uses_pdm_geometry_and_observes_physics_without_mutation():
    loop = ClosedLoop.__new__(ClosedLoop)
    loop.tracker = TrajectoryTracker({"wheelbase_m": 2.8, "max_steer_rad": .6})
    loop.ego_provider = EgoStateProvider(max_steer_rad=.6)
    model = SteeringModel(math.radians(70), [(0, 1), (36, .5)],
                          wheelbase_m=3.0046469567, front_track_m=1.666828)
    loop.pdm_planner = NS(vehicle_parameters=NS(wheel_base=3.0046469567),
                          max_steer=model.max_steer_rad, steering_model=model,
                          rear_axle_actor_xyz_m=(-1.3864, 0, .368))
    ego = Mock()
    ego.get_physics_control.return_value = NS(steering_curve=[NS(x=0, y=1), NS(x=36, y=.5)])
    loop._bind_pdm_controller(ego)
    assert loop.tracker.pp.L == pytest.approx(3.0046469567)
    assert loop.tracker.max_steer_rad == pytest.approx(model.inner_to_bicycle(math.radians(70)))
    assert loop.tracker.steering_model.limit_at(5) == pytest.approx(model.inner_to_bicycle(math.radians(70) * .75))
    assert loop.tracker.steering_model is model
    assert loop.tracker.rear_axle_actor_xyz_m == (-1.3864, 0, .368)
    assert loop.ego_provider.max_steer_rad == loop.pdm_planner.max_steer
    assert loop.ego_provider.steering_model is loop.tracker.steering_model
    assert loop.ego_provider.steering_model.curve == ((0, 1), (36, .5))
    assert loop.tracker.vehicle_model_source == "carla_pdm_vehicle"
    ego.apply_physics_control.assert_not_called()
    ego.set_transform.assert_not_called()


def test_turn_command_tracks_same_curvature_with_speed_dependent_physical_limit():
    """A circular path must not double curvature because max steer was 0.6."""
    speed, radius, wheelbase = 5.0, 20.0, 3.0046469567
    max_angle = math.radians(70)
    curve = [(0, 1), (36, .5)]
    controller = TrajectoryTracker({"lookahead_min_m": 3, "lookahead_k": .5})
    controller.bind_vehicle_model(wheelbase, max_angle, curve)
    # Left turn in ego coordinates; CARLA world path has negative y.
    points = [{"t": i * .1, "x": radius * math.sin(speed * i * .1 / radius),
               "y": radius * (1 - math.cos(speed * i * .1 / radius)),
               "optional_v": speed} for i in range(81)]
    controller.set_trajectory(points, 0, 0, 0, speed, follow_speed_profile=True)
    x = y = yaw = 0.0
    physical_limit = max_angle * .75  # 18 km/h: halfway between the knots.
    errors = []
    for step in range(40):
        control = controller.compute_control(x, y, yaw, speed)
        assert control.steer < 0
        road_angle = control.steer * physical_limit
        yaw_rate = speed * math.tan(road_angle) / wheelbase
        x += speed * math.cos(yaw + yaw_rate * .05) * .1
        y += speed * math.sin(yaw + yaw_rate * .05) * .1
        yaw += yaw_rate * .1
        errors.append(abs(math.hypot(x, y + radius) - radius))
    assert max(errors) < .05
    audit = controller.last_control_diagnostics
    assert audit["steering_limit_at_speed_rad"] == pytest.approx(physical_limit)
    assert audit["vehicle_model_source"] == "carla_pdm_vehicle"


def test_legacy_tracker_keeps_configured_model_until_explicitly_bound():
    tracker = TrajectoryTracker({"wheelbase_m": 2.8, "max_steer_rad": .6})
    assert tracker.pp.L == 2.8
    assert tracker.steering_model.limit_at(10) == .6
    assert tracker.vehicle_model_source == "controller_config"


@pytest.mark.parametrize("speed,expected_angle", [(0, -.3), (5, -.225), (10, -.15)])
def test_real_ego_provider_uses_shared_curve_for_observed_steering_estimate(speed, expected_angle):
    model = SteeringModel(1.2, [(0, 1), (36, .5)])
    provider = EgoStateProvider(steering_model=model)
    vehicle = NS(get_transform=lambda: carla.Transform(),
                 get_velocity=lambda: carla.Vector3D(x=speed),
                 get_acceleration=lambda: carla.Vector3D(),
                 get_control=lambda: carla.VehicleControl(steer=.25))
    state = provider.build(vehicle, 1.0)
    assert state.speed == speed
    assert state.steering_angle == pytest.approx(expected_angle)
    assert provider.steering_model is model


@pytest.mark.parametrize("speed", [0, 10])
def test_real_legacy_ego_provider_keeps_static_steering_conversion(speed):
    provider = EgoStateProvider(max_steer_rad=.6)
    vehicle = NS(get_transform=lambda: carla.Transform(),
                 get_velocity=lambda: carla.Vector3D(x=speed),
                 get_acceleration=lambda: carla.Vector3D(),
                 get_control=lambda: carla.VehicleControl(steer=.25))
    assert provider.build(vehicle, 1.0).steering_angle == pytest.approx(-.15)
    assert provider.steering_model is None


@pytest.mark.parametrize("inner_deg", [-70, -40, -16.16028023, 0, 16.16028023, 40, 70])
def test_ackermann_roundtrip_and_center_limit(inner_deg):
    model = SteeringModel(math.radians(70), wheelbase_m=3.0046469567, front_track_m=1.666828)
    inner = math.radians(inner_deg)
    center = model.inner_to_bicycle(inner)
    assert model.bicycle_to_inner(center) == pytest.approx(inner)
    assert abs(center) <= abs(inner)
    assert model.limit_at(0) == pytest.approx(math.radians(57.3261101))
    assert model.physical_limit_at(0) == pytest.approx(math.radians(70))
    assert model.bicycle_to_normalized(math.radians(80), 0) == pytest.approx(1)
    for speed in (0, 4.1, 15):
        for command in (-1, -.25, 0, .25, 1):
            assert model.bicycle_to_normalized(model.normalized_to_bicycle(command, speed), speed) == pytest.approx(command)


def test_recorded_carla_front_wheels_require_center_angle_conversion():
    """Measured frame 3398997; unlike averaging wheel angles, cot mapping is exact."""
    model = SteeringModel(math.radians(70), [(0, 1), (20, .9), (60, .8), (120, .7)],
                          wheelbase_m=3.0046469567, front_track_m=1.666828)
    center = model.front_wheels_to_bicycle(math.radians(14.0170545578), math.radians(16.1602802277))
    assert math.degrees(center) == pytest.approx(15.0143106, abs=1e-6)
    assert model.front_wheels_to_bicycle(-math.radians(16.1602802277), -math.radians(14.0170545578)) == pytest.approx(-center)
    assert model.inner_to_bicycle(math.radians(16.1602802277)) == pytest.approx(center, abs=1e-7)
    assert model.bicycle_to_normalized(math.radians(15), 4.0998787) == pytest.approx(.2490008848)
    assert model.front_wheels_to_bicycle(0, 0) == 0


@pytest.mark.parametrize("angles", [(0, .1), (-.1, .1), (float('nan'), 0), (0, math.pi / 2)])
def test_invalid_measured_angles_do_not_become_valid_state(angles):
    with pytest.raises(ValueError):
        SteeringModel(1.2).front_wheels_to_bicycle(*angles)


def test_bound_provider_uses_ackermann_forward_mapping_for_semantic_estimate():
    model = SteeringModel(math.radians(70), [(0, 1), (20, .9)],
                          wheelbase_m=3.0046469567, front_track_m=1.666828)
    provider = EgoStateProvider(steering_model=model)
    vehicle = NS(get_transform=lambda: carla.Transform(), get_velocity=lambda: carla.Vector3D(x=4.1),
                 get_acceleration=lambda: carla.Vector3D(), get_control=lambda: carla.VehicleControl(steer=.25))
    assert provider.build(vehicle, 1).steering_angle == pytest.approx(-model.normalized_to_bicycle(.25, 4.1))


@pytest.mark.parametrize("capture_yaw", [0, math.pi / 2, -2.8])
@pytest.mark.parametrize("direction", [-1, 1])
def test_pdm_actor_trajectory_and_pose_are_both_tracked_at_rear_axle(capture_yaw, direction):
    """Integrate a real rear-axle bicycle; actor lies 1.3864 m ahead, not at the axle."""
    radius, speed, wheelbase, track, ahead = 8.0, 3.0, 3.0046469567, 1.666828, 1.386424422
    model = SteeringModel(math.radians(70), [(0, 1), (20, .9)], wheelbase_m=wheelbase, front_track_m=track)
    tracker = TrajectoryTracker({"lookahead_min_m": 3, "lookahead_k": .5})
    tracker.bind_vehicle_model(wheelbase, steering_model=model, rear_axle_actor_xyz_m=(-ahead, 0, .368))
    points = []
    for i in range(101):
        angle = speed * i * .1 / radius
        points.append({"t": i * .1, "x": radius * math.sin(angle) + ahead * math.cos(angle) - ahead,
                       "y": direction * (radius * (1 - math.cos(angle)) + ahead * math.sin(angle)),
                       "optional_heading_rad": direction * angle, "optional_v": speed})
    capture_x, capture_y = 20.0, -17.0
    tracker.set_trajectory(points, capture_x, capture_y, capture_yaw, speed, follow_speed_profile=True)
    rear_x = capture_x - ahead * math.cos(capture_yaw)
    rear_y = capture_y - ahead * math.sin(capture_yaw)
    center_x = rear_x + direction * radius * math.sin(capture_yaw)
    center_y = rear_y - direction * radius * math.cos(capture_yaw)
    yaw, errors = capture_yaw, []
    for _ in range(45):
        actor_x, actor_y = rear_x + ahead * math.cos(yaw), rear_y + ahead * math.sin(yaw)
        control = tracker.compute_control(actor_x, actor_y, yaw, speed)
        assert control.steer * direction < 0
        # Independent PhysX per-wheel Ackermann construction, then equivalent curvature.
        inner = abs(control.steer) * model.physical_limit_at(speed)
        outer = math.atan(wheelbase / (track + wheelbase / math.tan(inner)))
        curvature = math.copysign(2 / (wheelbase * (1 / math.tan(inner) + 1 / math.tan(outer))), control.steer)
        rate = speed * curvature
        rear_x += speed * math.cos(yaw + rate * .05) * .1
        rear_y += speed * math.sin(yaw + rate * .05) * .1
        yaw += rate * .1
        errors.append(abs(math.hypot(rear_x - center_x, rear_y - center_y) - radius))
    assert max(errors) < .025
    assert tracker.last_control_diagnostics["tracking_reference"] == "pdm_rear_axle"
    assert tracker._actor_global_wps[0]["x"] == pytest.approx(capture_x)
    assert tracker._global_wps[0]["x"] == pytest.approx(capture_x - ahead * math.cos(capture_yaw))


@pytest.mark.parametrize("heading", [None, float('nan'), float('inf'), 'not-an-angle'])
def test_rear_axle_binding_requires_explicit_valid_body_heading_and_clears_stale_plan(heading):
    tracker = TrajectoryTracker({})
    tracker.bind_vehicle_model(3, 1.2, rear_axle_actor_xyz_m=(-1.3, 0, .3))
    tracker.set_trajectory([{"x": 0, "y": 0, "optional_heading_rad": 0},
                            {"x": 10, "y": 0, "optional_heading_rad": 0}], 0, 0, 0, 3)
    broken = {"x": 10, "y": 0}
    if heading is not None:
        broken["optional_heading_rad"] = heading
    with pytest.raises(ValueError, match="heading"):
        tracker.set_trajectory([broken], 0, 0, 0, 3)
    assert not tracker.has_trajectory()
    assert tracker.compute_control(0, 0, 0, 0).brake == 1


def test_legacy_trajectory_does_not_require_body_heading():
    tracker = TrajectoryTracker({})
    tracker.set_trajectory([{"x": 0, "y": 0}, {"x": 10, "y": 0}], 0, 0, 0, 3)
    assert tracker.compute_control(0, 0, 0, 1).brake < 1
