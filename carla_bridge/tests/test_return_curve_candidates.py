"""Real Town10HD return corner replay: native rollout, without world RPC.

The physical run returned toward the correct lane, but its only target-center
curve swept the front-left corner across the opposing-direction solid line.
This fixture preserves that pose and steering feedback. It is not a full live
traffic/physics test; independent physical completion is checked separately.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carla_bridge.tests.test_recovery_planner_integration import (
    make_snapshot_planner, road_snapshot)


def return_snapshot(road_snapshot, *, quality=True):
    import carla
    road, _ = road_snapshot
    snap = json.loads((Path(__file__).parent / "fixtures" /
                       "town10hd_return_boundary.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(road.to_opendrive().encode()).hexdigest() == snap["map_sha256"]
    snap["planning"]["lane_quality"] = {"enabled": quality}
    planner, clock = make_snapshot_planner((road, snap))
    clock.time = snap["timestamp_s"]
    ego = planner.ego
    ego.get_velocity = lambda: carla.Vector3D(x=snap["velocity_xy"][0], y=snap["velocity_xy"][1])
    ego.get_control = lambda: carla.VehicleControl(steer=snap["steering"]["normalized_command"])
    ego.get_wheel_steer_angle = lambda wheel: snap["steering"][
        "front_left_deg" if wheel == carla.VehicleWheelLocation.FL_Wheel else "front_right_deg"]
    state, _ = planner._ego_state(round(clock.time * 1e6))
    planner._geometry_diagnostics = {}
    planner._road_geometry(state)
    lm = planner.lane_maneuver
    reference = lm._planning_reference(planner._resolved_route_waypoints)
    xy = np.array([[w.transform.location.x, -w.transform.location.y] for w in reference])
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    active = deepcopy(snap["maneuver"])
    built = lm._build(reference, stations, active["anchor"], active["length"], active["side"],
                      min(stations[-1], active["anchor"] + 90), returning=True)
    active.update(points=built[0], sample_s=built[1], widths=built[2], corridor_end=stations[-1])
    lm._active = active
    return planner, clock


def advance_native_feedback(planner, clock, rollout_state, elapsed_s):
    """Execute only a prefix of the selected native prediction in model space.

    This is an offline receding-horizon contract check, not CARLA physics. Keep
    axle/actor transforms, centre velocity and measured wheel feedback coherent
    so that a follow-up call actually replans from its own preceding execution.
    """
    import carla
    si, api = planner.api.StateIndex, planner.api
    value = rollout_state.copy()
    state = api.EgoState.build_from_rear_axle(
        rear_axle_pose=api.StateSE2(*value[si.STATE_SE2]),
        rear_axle_velocity_2d=api.StateVector2D(*value[si.VELOCITY_2D]),
        rear_axle_acceleration_2d=api.StateVector2D(*value[si.ACCELERATION_2D]),
        tire_steering_angle=value[si.STEERING_ANGLE],
        time_point=api.TimePoint(round((clock.time + elapsed_s) * 1e6)),
        vehicle_parameters=planner.vehicle_parameters,
        angular_vel=value[si.ANGULAR_VELOCITY], angular_accel=value[si.ANGULAR_ACCELERATION])
    yaw = -state.rear_axle.heading
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rear = planner.rear_axle_actor_xyz_m
    transform = carla.Transform(carla.Location(
        x=state.rear_axle.x - (cosine*rear[0] - sine*rear[1]),
        y=-state.rear_axle.y - (sine*rear[0] + cosine*rear[1]), z=.002),
        carla.Rotation(yaw=math.degrees(yaw)))

    def carla_vector(local):
        return carla.Vector3D(x=cosine*local.x + sine*local.y,
                              y=sine*local.x - cosine*local.y)

    velocity = carla_vector(state.dynamic_car_state.center_velocity_2d)
    acceleration = carla_vector(state.dynamic_car_state.center_acceleration_2d)
    delta = -value[si.STEERING_ANGLE]
    curvature = math.tan(delta) / planner.vehicle_parameters.wheel_base
    tangent = planner.vehicle_parameters.wheel_base * curvature
    half_track_curvature = planner.front_track_m * .5 * curvature
    front_left = math.degrees(math.atan(tangent / (1 + half_track_curvature)))
    front_right = math.degrees(math.atan(tangent / (1 - half_track_curvature)))
    command = planner.steering_model.bicycle_to_normalized(delta, state.dynamic_car_state.speed)
    ego = planner.ego
    ego.get_location = lambda: transform.location
    ego.get_transform = lambda: transform
    ego.get_velocity = lambda: velocity
    ego.get_acceleration = lambda: acceleration
    ego.get_angular_velocity = lambda: carla.Vector3D(z=-math.degrees(value[si.ANGULAR_VELOCITY]))
    ego.get_control = lambda: carla.VehicleControl(steer=command)
    ego.get_wheel_steer_angle = lambda wheel: (
        front_left if wheel == carla.VehicleWheelLocation.FL_Wheel else front_right)
    clock.time += elapsed_s
    actual, _ = planner._ego_state(round(clock.time * 1e6))
    np.testing.assert_allclose(actual.rear_axle.array, state.rear_axle.array, atol=1e-5)
    np.testing.assert_allclose(actual.dynamic_car_state.rear_axle_velocity_2d.array,
                               state.dynamic_car_state.rear_axle_velocity_2d.array, atol=1e-5)
    assert actual.tire_steering_angle == pytest.approx(value[si.STEERING_ANGLE])


def test_real_return_curve_keeps_progress_and_selects_a_body_clear_target_lane_variant(road_snapshot):
    planner, clock = return_snapshot(road_snapshot)
    nominal = planner.lane_maneuver._active["points"].copy()
    anchor = planner.lane_maneuver._active["anchor"]
    length = planner.lane_maneuver._active["length"]
    diagnostic = planner.plan(None, semantic_enabled=False, sim_time_s=clock.time)["planner_diagnostics"]
    original = next(c for c in diagnostic["candidates"] if c["id"] == "center:ACCELERATE")
    assert original["lane_quality"]["minimum_footprint_clearance_m"] < -.10
    winner = next(c for c in diagnostic["candidates"] if c["id"] == diagnostic["selected_id"])
    assert winner["physical_lane_side"] == "center", winner
    assert winner["base_score"] > .9
    # Return speed is now bound to the same 5 m/s used by the gap check. The
    # recorded ego starts at 7.25 m/s: it must decelerate, not retain the old
    # ordinary 8 m/s target merely to pass a distance assertion.
    cap = planner.lane_maneuver._active["speed_cap"]
    assert winner["maneuver_execution_limit"]
    assert winner["desired_speed_mps"] == cap
    assert winner["forward_progress_m"] > .9 * cap * planner.horizon
    assert winner["terminal_speed_mps"] == pytest.approx(cap, abs=.1)
    assert winner["acceleration_range_mps2"][0] >= -winner["idm_decel_max_mps2"] - 1e-8
    assert winner["lane_quality"]["minimum_footprint_clearance_m"] > .05
    assert winner["lane_quality"]["footprint_boundary_crossing_fraction"] == 0.
    assert all(c["semantic_bonus"] == 0 for c in diagnostic["candidates"])
    assert diagnostic["safety_gate_applied"] is False
    # The real winning offset must become execution state. A scored good path
    # that cannot pass the commit interface would be rebuilt/lost next tick.
    commit = diagnostic["lane_maneuver"]["execution_commit"]
    assert commit["committed"] and commit["actual_selected_id"] == winner["id"]
    actual = planner._evaluated_lateral_paths[winner["lateral_path_index"]]
    expected = np.array([[p.x, p.y] for p in actual.discrete_path])
    active = planner.lane_maneuver._active
    np.testing.assert_array_equal(active["executed_selected_points"], expected)
    np.testing.assert_array_equal(active["nominal_points"], nominal)
    assert active["anchor"] == anchor and active["length"] == length


def test_real_return_curve_completes_in_receding_native_rollouts_without_stale_state(road_snapshot):
    """The capped body-clear choice must lead back to sustained route driving."""
    from carla_bridge.planning.lane_maneuver import project_polyline
    planner, clock = return_snapshot(road_snapshot)
    active = planner.lane_maneuver._active
    anchor, length = active["anchor"], active["length"]
    completion_time, completed_samples = None, []
    dt, execute_steps = .4, 4
    for _ in range(16):
        diagnostic = planner.plan(None, semantic_enabled=False, sim_time_s=clock.time)["planner_diagnostics"]
        winner = next(c for c in diagnostic["candidates"] if c["id"] == diagnostic["selected_id"])
        # Lane-quality evaluates a two-second actual footprint prefix. Each
        # executed .4-second prefix is inside that interval, not an unscored
        # path interpolation or a fresh teleport to the target lane centre.
        assert winner["lane_quality"]["evaluated_horizon_s"] >= dt
        assert winner["lane_quality"]["minimum_footprint_clearance_m"] > .05
        assert winner["lane_quality"]["footprint_boundary_crossing_fraction"] == 0.
        assert winner["forward_progress_m"] > .9 * 5. * planner.horizon
        if planner.lane_maneuver._active is not None:
            assert planner.lane_maneuver._active["anchor"] == anchor
            assert planner.lane_maneuver._active["length"] == length
            assert winner["desired_speed_mps"] <= 5.
        else:
            completion_time = clock.time if completion_time is None else completion_time
            reference = planner.lane_maneuver._last_reference
            xy = np.array([[w.transform.location.x, -w.transform.location.y] for w in reference])
            tf = planner.ego.get_transform()
            station, lateral, heading = project_polyline(tf.location.x, -tf.location.y, xy)
            assert station >= anchor + length
            assert lateral < .35
            assert abs(math.atan2(math.sin(-math.radians(tf.rotation.yaw)-heading),
                                  math.cos(-math.radians(tf.rotation.yaw)-heading))) < .15
            assert winner["physical_lane_side"] == "center"
            assert not winner["maneuver_execution_limit"]
            completed_samples.append(clock.time)
            if len(completed_samples) >= 4:
                break
        rollout = planner.scorer._states[winner["index"]]
        assert np.isfinite(rollout).all()
        advance_native_feedback(planner, clock, rollout[execute_steps], dt)
    assert completion_time is not None, "return state never released after native execution"
    assert completion_time - 12.2 < 5.
    assert len(completed_samples) >= 4 and completed_samples[-1] - completed_samples[0] >= 1.1
    assert planner.lane_maneuver._active is None


def test_return_curve_soft_quality_preserves_native_scores_and_simulated_geometry(road_snapshot):
    results = []
    for enabled in (False, True):
        planner, clock = return_snapshot(road_snapshot, quality=enabled)
        results.append(planner.plan(None, semantic_enabled=False, sim_time_s=clock.time)["planner_diagnostics"])
    before, after = ({c["id"]: c for c in d["candidates"]} for d in results)
    assert before.keys() == after.keys()
    for label, original in before.items():
        assert after[label]["base_score"] == original["base_score"]
        assert after[label]["trajectory_sha256"] == original["trajectory_sha256"]
