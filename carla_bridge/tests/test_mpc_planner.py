"""CARLA-shaped geometry through the real independent numerical MPC adapter."""
from dataclasses import asdict
from types import SimpleNamespace as NS
import json
import math

import carla
import numpy as np
import pytest

from carla_bridge.planning.mpc_planner import CarlaMPCPlanner, effective_mpc_config
from carla_bridge.tests.test_pdm_planner import Road, Vehicle, Waypoint, RedLight


def fixture(*others, speed=0., x=0., y=0., yaw=0., config=None, allow_change=True):
    road, ego = Road(allow_change=allow_change), Vehicle(speed=speed, x=x, y=y, yaw=yaw)
    world = NS(get_map=lambda: road, get_actors=lambda: [ego, *others],
               get_weather=lambda: NS(fog_density=0.),
               get_snapshot=lambda: NS(timestamp=NS(elapsed_seconds=10.)))
    reference = [Waypoint(road, value) for value in range(-10, 160, 2)]
    route = NS(_route=[wp.transform.location for wp in reference], _route_waypoints=reference,
        _nearest_index=lambda p: min(len(reference)-2, max(0, int((p.x+10)/2))),
        distance_completed_m=lambda: max(0., ego.x), nav_instruction=lambda: "STRAIGHT")
    defaults = {"backend": "mpc", "max_speed_mps": 8., "semantic_policy": "selective",
                "mpc": {"dt": .2, "horizon_steps": 20, "max_iterations": 70}}
    defaults.update(config or {})
    return CarlaMPCPlanner(world, ego, route, defaults)


def test_effective_config_is_pure_and_rejects_unknown_solver_keys():
    values = asdict(effective_mpc_config({"horizon_s": 4., "sample_interval_s": .1}))
    assert values["dt"] == .2 and values["horizon_steps"] == 20
    assert asdict(effective_mpc_config({"mpc": values})) == values
    with pytest.raises(ValueError, match="Unknown MPC"):
        effective_mpc_config({"mpc": {"unrecognized": True}})


@pytest.mark.parametrize("dt", [0., -.1, True, float("nan"), float("inf"), .6])
def test_adapter_rejects_unexecutable_control_profile_intervals_before_driving(dt):
    with pytest.raises(ValueError, match="CARLA MPC dt"):
        effective_mpc_config({"mpc": {"dt": dt}})


def test_forward_plan_optimizes_controls_and_uses_actual_rear_axle():
    planner = fixture()
    plan = planner.plan(None, False, sim_time_s=0.)
    d = plan["planner_diagnostics"]
    assert d["backend"] == "mpc" and not d["provenance"]["upstream_pdm_used"]
    assert d["solver_feasibility_gate_applied"]
    assert not d["emergency_brake"]
    assert plan["trajectory"][0]["x"] == pytest.approx(0, abs=1e-9)
    assert plan["trajectory"][0]["y"] == pytest.approx(0, abs=1e-9)
    assert plan["trajectory"][-1]["x"] > 10.
    assert d["vehicle"]["rear_axle_actor_xyz_m"][0] == pytest.approx(-1.2)
    profile = d["mpc_tracking"]
    assert profile["active"] and profile["source"] == "nonlinear_mpc_optimized_controls"
    assert len(profile["bicycle_angles_rad"]) == 20
    assert max(profile["accelerations_mps2"]) <= 2.+1e-6
    assert len(plan["trajectory"]) == 41
    json.dumps(plan, allow_nan=False)


def test_replay_world_and_execution_coordinates_use_the_same_rigid_transform():
    planner = fixture(speed=2.)
    planner.configure_trajectory_visualization({"enabled": True})
    plan = planner.plan(None, False, sim_time_s=0.)
    snapshot = planner.last_visualization_snapshot
    selected = next(c for c in snapshot["candidates"] if c["id"] == plan["planner_diagnostics"]["selected_id"])
    for i, point in enumerate(selected["points_world"]):
        execution = plan["trajectory"][2*i]
        assert execution["x"] == pytest.approx(point[0], abs=1e-9)
        assert execution["y"] == pytest.approx(-point[1], abs=1e-9)
    assert snapshot["source"] == "nonlinear_mpc_optimized_rollout"


def test_red_signal_is_an_optimizer_stop_constraint_and_does_not_query_model():
    planner = fixture(speed=2.)
    light = RedLight(planner.map, 14.)
    planner.world.get_actors = lambda: [planner.ego, light]
    plan = planner.plan(None, False, sim_time_s=0.)
    d = plan["planner_diagnostics"]
    assert not d["emergency_brake"]
    line = d["traffic_stop_line"]
    assert line and line["id"].startswith("stop_100_")
    # Actor-origin conversion puts front bumper .2+2.3 metres ahead.
    assert max(p["x"]+2.5 for p in plan["trajectory"]) < 14.05
    query = planner.query_decision(0.)
    assert not query["query"] and query["reason"] == "expected_signal_or_queue_wait"


def test_no_feasible_solution_brakes_and_never_executes_model_preference():
    obstacle = Vehicle(actor_id=7, x=1., speed=0.)
    planner = fixture(obstacle, speed=4.)
    plan = planner.plan({"lateral_decision": "CHANGE_LANE_LEFT", "longitudinal_decision": "ACCELERATE"},
                        semantic_enabled=True, sim_time_s=0., semantic_update=True)
    d = plan["planner_diagnostics"]
    assert d["emergency_brake"] and d["failure_reason"] == "no_feasible_mpc_solution"
    assert not plan["trajectory"] and plan["target_speed"] == 0.
    assert all(not c["eligible"] for c in d["candidates"])
    assert any(c["trajectory_finite"] and not c["optimization_feasible"] for c in d["candidates"])
    json.dumps(plan, allow_nan=False)


def test_legal_near_walking_pedestrian_gets_an_optimized_bypass_without_model():
    person = Vehicle(actor_id=7, x=8., speed=.7)
    person.type_id = "walker.pedestrian.0001"
    person.bounding_box = carla.BoundingBox(carla.Location(z=.9), carla.Vector3D(x=.25, y=.25, z=.9))
    planner = fixture(person, speed=.5)
    plan = None
    for now in (0., .2, .4, .6, .8, 1.):
        plan = planner.plan(None, False, sim_time_s=now)
    d = plan["planner_diagnostics"]
    assert d["lane_maneuver"]["reason"] in ("persistent_slow_obstacle", "committed_maneuver")
    assert any(c["maneuver"] for c in d["candidates"])
    baseline = next(c for c in d["candidates"] if c["id"] == d["selected_id"])
    assert baseline["maneuver"] and baseline["optimization_feasible"], d
    assert max(p["y"] for p in plan["trajectory"]) > 1.4
    center = [c["forward_progress_m"] for c in d["candidates"] if c["eligible"] and not c["maneuver"]]
    assert not center or baseline["forward_progress_m"] > max(center)
    assert d["lane_maneuver"]["committed_after_selection"]
    assert d["semantic_enabled"] is False
    from carla_bridge.control.trajectory_validation import TrajectoryValidator
    validation = TrajectoryValidator().validate(plan["trajectory"], plan["target_speed"],
                                                .5, 3., use_speed_profile=True)
    assert validation.accepted, validation.to_dict()
    assert d["lane_maneuver"]["execution_commit"]["geometry_role"] == \
        "committed_maneuver_reference_not_optimized_execution"


def test_solid_boundary_does_not_generate_a_model_requested_lane_change():
    person = Vehicle(actor_id=7, x=8., speed=.7)
    person.type_id = "walker.pedestrian.0001"
    person.bounding_box = carla.BoundingBox(carla.Location(z=.9), carla.Vector3D(x=.25, y=.25, z=.9))
    planner = fixture(person, speed=.5, allow_change=False)
    plan = None
    for now in (0., .2, .4, .6, .8, 1.):
        plan = planner.plan({"lateral_decision": "CHANGE_LANE_LEFT", "longitudinal_decision": "ACCELERATE"},
                            True, sim_time_s=now)
    assert not any(c["maneuver"] for c in plan["planner_diagnostics"]["candidates"])


def test_shadow_restores_baseline_instead_of_executing_semantic_stop():
    planner = fixture(config={"semantic_policy": "shadow"})
    plan = planner.plan({"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "STOP"},
                        True, semantic_update=True, sim_time_s=0.)
    d = plan["planner_diagnostics"]
    assert d["shadow_only"] and d["selected_id"] == d["baseline_selected_id"]
    assert not d["governor"]["adopted"]


@pytest.mark.parametrize("mode,expected", [("off", 3), ("gate", 3), ("full", 4)])
def test_execution_supported_memory_adds_only_bounded_current_state_optimization(mode, expected):
    planner = fixture(config={"planning_memory": mode})
    evidence = {"parameter_priors": [{"status": "execution_supported", "support_count": 3,
        "parameters": {"target_velocity_mps": 500.}, "record_ids": ["verified1", "verified2", "verified3"]}]}
    memory = NS(retrieve=lambda context: evidence)
    planner.experience_memory = memory
    plan = planner.plan(None, False, sim_time_s=0.)
    candidates = plan["planner_diagnostics"]["candidates"]
    assert len(candidates) == expected
    assert all(c["desired_speed_mps"] <= planner.max_speed for c in candidates)
    if mode == "full":
        c = next(c for c in candidates if ":MEMORY" in c["id"])
        assert c["optimizer_status"] in ("converged", "feasible_not_converged")
        assert c["parameter_model"] == "nonlinear_mpc_not_IDM"


def test_speed_dependent_carla_curve_is_bound_before_optimization():
    original = fixture()
    getter = original.ego.get_physics_control
    def physical():
        model = getter()
        model.steering_curve = [NS(x=0., y=1.), NS(x=20., y=.6), NS(x=60., y=.4)]
        return model
    original.ego.get_physics_control = physical
    planner = CarlaMPCPlanner(original.world, original.ego, original.route, original.config)
    plan = planner.plan(None, False, sim_time_s=0.)
    guard = plan["planner_diagnostics"]["speed_dependent_steering"]
    assert guard["effective_bicycle_limit_rad"] < planner.max_steer
    assert planner.solver.config.steer_max == pytest.approx(guard["effective_bicycle_limit_rad"])
    assert planner.solver.config.steer_max <= planner.steering_model.limit_at(planner.max_speed)


def test_expansion_reoptimizes_without_widening_legal_corridor_and_shadow_restores(monkeypatch):
    planner = fixture(config={"semantic_policy": "shadow"})
    # Authorization is the separate, extensively tested governor contract. This
    # adapter test forces that branch but runs every additional SLSQP unchanged.
    monkeypatch.setattr(planner.governor, "recovery_request", lambda *_: {
        "authorized": True, "reason": "test_authorized_generation"})
    requests = []
    solve = planner.solver.solve
    def tracked(problem):
        requests.append(problem)
        return solve(problem)
    monkeypatch.setattr(planner.solver, "solve", tracked)
    plan = planner.plan(None, True, sim_time_s=0.)
    d = plan["planner_diagnostics"]
    assert len(requests) > 3 and d["recovery"]["expanded"]
    assert not d["recovery"]["constraints_relaxed"]
    assert len(d["candidates"]) == 3 and len(d["shadow_candidates"]) > 3
    assert d["candidate_pool_sha256"] == d["original_candidate_pool_sha256"]
    assert d["selected_id"] == d["original_baseline_selected_id"]
    assert len(planner.last_candidates) == 3
    original_width = requests[0].reference.half_width
    for problem in requests[3:]:
        ref = problem.reference
        assert ref.metadata["recovery"]
        offset = abs(ref.metadata["lane_offset_m"])
        assert np.all(np.asarray(ref.half_width)+offset <= original_width+1e-9)
        assert ref.speed_limit == requests[0].reference.speed_limit


def test_half_step_execution_uses_bicycle_rollout_rather_than_straight_chord():
    planner = fixture(speed=3.)
    from carla_bridge.planning.mpc_solver import rollout, MPCInitialState
    initial = MPCInitialState(-1.2, 0., 0., 3.)
    controls = np.tile([0., .3], (planner.samples, 1))
    states = rollout(initial, controls, planner.numerical_config)
    trajectory = planner._execution(NS(states=states, controls=controls), planner.ego.get_transform())
    heading = trajectory[1]["optional_heading_rad"]
    assert heading < 0.  # CARLA positive-right becomes project negative-left.
    # Actor yaw adds the physical rear->actor rotation. A linear chord would
    # average the endpoints and miss this exact fractional bicycle integration.
    chord = (trajectory[0]["y"]+trajectory[2]["y"])/2
    assert abs(trajectory[1]["y"]-chord) > 1e-4


def test_red_stop_before_a_turn_does_not_use_the_future_crossroads_normal():
    from carla_bridge.planning.mpc_solver import MPCInitialState, MPCProblem
    planner = fixture(speed=2.7)
    planner._speed_limit = 8.
    points = np.array([[-10., 0.], [0., 0.], [8., 0.], [10., 2.], [10., 10.], [10., 30.]])
    headings = np.array([0., 0., 0., math.pi/2, math.pi/2, math.pi/2])
    initial = MPCInitialState(-1.2, 0., 0., 2.7)
    reference = planner._reference("red_before_turn", points, initial, 8., 3.5,
        {"label": "KEEP_LANE", "semantics": ["KEEP_LANE"], "longitudinal": "ACCELERATE",
         "desired_speed_mps": 8.}, headings=headings,
        stop_line={"point": [4., 0.], "normal": [1., 0.]})
    assert np.max(reference.points[:, 0]) <= 4.+1e-8
    assert np.max(np.abs(reference.points[:, 1])) < 1e-8
    result = planner.solver.solve(MPCProblem(initial, reference))
    assert result.feasible, result.diagnostics
    assert np.max(result.states[:, 0]) <= 4.+.002


def test_physical_stationary_deceleration_is_not_a_previous_acceleration_command():
    planner = fixture()
    planner.ego.get_acceleration = lambda: carla.Vector3D(x=-12.7805)
    plan = planner.plan(None, False, sim_time_s=3.9)
    diagnostic = plan["planner_diagnostics"]
    assert not diagnostic["emergency_brake"], diagnostic
    state = diagnostic["steering_state"]
    assert state["measured_longitudinal_acceleration_mps2"] == pytest.approx(-12.7805)
    assert state["optimization_initial_acceleration_mps2"] == 0.
    # Merely choosing an optimized candidate must not claim it was dispatched.
    assert planner._executed_desired_acceleration["sim_time_s"] is None


def test_only_actual_optimized_control_dispatch_updates_previous_desired_input():
    planner = fixture()
    planner.ego.get_acceleration = lambda: carla.Vector3D(x=-12.7805)
    diagnostics = {"mpc_tracking": {"accelerations_mps2": [.4, 1.2, 1.6]}}
    planner.record_control_execution(diagnostics, {"applied": True, "profile_sample_index": 1},
                                     now_s=1., source="tracker")
    initial, _ = planner._initial_state(1.1)
    assert initial.acceleration == pytest.approx(1.2)
    assert planner._steering_state_diagnostics["previous_desired_acceleration"]["source"] == \
        "dispatched_optimized_desired_acceleration"
    assert planner._steering_state_diagnostics["measured_longitudinal_acceleration_mps2"] == pytest.approx(-12.7805)
    assert planner._initial_state(1.9)[0].acceleration == 0.  # Stale requests are not reused.
    assert planner._initial_state(.9)[0].acceleration == 0.  # A different run clock is not reused.


@pytest.mark.parametrize("source,audit,tracker,reason", [
    ("direct_brake", {"applied": True, "profile_sample_index": 0}, {}, "direct_brake_reset_desired_input"),
    ("tracker", {"applied": True, "profile_sample_index": 0}, {"reason": "stationary_hold"},
        "stationary_hold_zero_desired_acceleration"),
    ("tracker", {"applied": False}, {}, "mpc_control_override_rejected_reset_desired_input"),
    ("tracker", {"applied": True, "profile_sample_index": 8}, {},
        "invalid_dispatched_profile_reset_desired_input"),
    ("tracker", {"applied": True, "profile_sample_index": True}, {},
        "invalid_dispatched_profile_reset_desired_input"),
])
def test_hold_and_interruption_reset_previous_desired_input(source, audit, tracker, reason):
    planner = fixture()
    diagnostics = {"mpc_tracking": {"accelerations_mps2": [1.2]}}
    planner.record_control_execution(diagnostics, {"applied": True, "profile_sample_index": 0},
                                     now_s=1., source="tracker")
    assert planner._initial_state(1.1)[0].acceleration == pytest.approx(1.2)
    planner.record_control_execution(diagnostics, audit, now_s=1.1, source=source,
                                     tracker_diagnostics=tracker)
    assert planner._initial_state(1.2)[0].acceleration == 0.
    assert planner._executed_desired_acceleration["source"] == reason


def test_lane_identity_brackets_a_connected_road_seam_without_assuming_an_adjacent_lane():
    planner = fixture(x=.05)
    planner.map.seam = .02
    reference = [Waypoint(planner.map, x) for x in (-2., -.01, 2., 4.)]
    assert reference[1].road_id != planner.map.get_waypoint(planner.ego.get_location()).road_id
    assert planner._side(reference) == "center"
    planner.ego.y = -3.5
    assert planner._side(reference) == "left"
    planner.ego.y = 3.5
    assert planner._side(reference) == "right"


def test_coincident_initial_junction_exit_uses_verified_connected_nonjunction_topology():
    planner = fixture()
    reference = [Waypoint(planner.map, float(x)) for x in range(0, 100, 2)]
    reference[0].road_id, reference[0].is_junction = 68, True
    planner.route._route_waypoints = reference
    planner.route._route = [wp.transform.location for wp in reference]
    planner.route._nearest_index = lambda _: 0
    resolved, _ = planner._resolve_reference()
    assert not resolved[0].is_junction
    assert resolved[0].road_id == reference[1].road_id
    assert reference[0].is_junction and reference[0].road_id == 68  # Original checkpoints stay intact.
    assert planner._reference_topology_corrections[0]["evaluation_route_modified"] is False
    # The corrected private copy persists after the vehicle leaves the endpoint.
    planner.ego.x = .2
    assert not planner._resolve_reference()[0][0].is_junction


@pytest.mark.parametrize("actual_junction,disconnected,offset", [(True, False, 0.), (False, True, 0.),
                                                               (False, False, .2)])
def test_endpoint_correction_never_hides_real_junction_disconnection_or_distinct_location(
        actual_junction, disconnected, offset):
    planner = fixture()
    reference = [Waypoint(planner.map, float(x)) for x in range(0, 100, 2)]
    reference[0].road_id, reference[0].is_junction = 68, True
    reference[0].transform.location.x = offset
    planner.route._route_waypoints = reference
    planner.route._route = [wp.transform.location for wp in reference]
    planner.route._nearest_index = lambda _: 0
    original = planner.map.get_waypoint
    def map_waypoint(location):
        current = original(location)
        current.is_junction = actual_junction
        if disconnected:
            current.next = lambda distance: []
        return current
    planner.map.get_waypoint = map_waypoint
    assert planner._resolve_reference()[0][0].is_junction
    assert not planner._reference_topology_corrections


def test_missing_shapely_is_rejected_during_dependency_preflight(monkeypatch):
    import builtins
    from carla_bridge.planning.mpc_planner import validate_dependencies
    imported = builtins.__import__
    def without_geometry(name, *args, **kwargs):
        if name == "shapely" or name.startswith("shapely."):
            raise ModuleNotFoundError("No module named 'shapely'")
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_geometry)
    with pytest.raises(RuntimeError, match="MPC geometry dependency Shapely/GEOS.*requirements-mpc"):
        validate_dependencies()


def test_same_frame_baseline_then_semantic_replanning_cannot_compound_jerk_history():
    planner = fixture(config={"semantic_policy": "always"})
    first = planner.plan(None, False, sim_time_s=1.)
    first_d = first["planner_diagnostics"]
    planner.record_control_execution(first_d, {"applied": True, "profile_sample_index": 0},
                                     now_s=1., source="tracker")
    # A cognition cycle may replan and dispatch again before world.tick().
    second = planner.plan({"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "ACCELERATE"},
                          True, semantic_update=True, sim_time_s=1.)
    second_d = second["planner_diagnostics"]
    assert second_d["steering_state"]["optimization_initial_acceleration_mps2"] == 0.
    assert planner._executed_desired_acceleration["sim_time_s"] is None
    assert planner._pending_desired_acceleration["sim_time_s"] == 1.
    rate_bound = planner.numerical_config.jerk_max*planner.dt
    assert abs(second_d["mpc_tracking"]["accelerations_mps2"][0]) <= rate_bound+1e-6
    # Only the final dispatch in that frame becomes previous at the next tick.
    planner.record_control_execution({"mpc_tracking": {"accelerations_mps2": [.8]}},
        {"applied": True, "profile_sample_index": 0}, now_s=1., source="tracker")
    assert planner._initial_state(1.)[0].acceleration == 0.
    assert planner._initial_state(1.1)[0].acceleration == pytest.approx(.8)


def test_later_dispatch_promotes_only_the_preceding_physical_tick():
    planner = fixture()
    valid = {"applied": True, "profile_sample_index": 0}
    planner.record_control_execution({"mpc_tracking": {"accelerations_mps2": [.8]}},
                                     valid, now_s=1., source="tracker")
    planner.record_control_execution({"mpc_tracking": {"accelerations_mps2": [1.6]}},
                                     valid, now_s=1.1, source="tracker")
    assert planner._initial_state(1.1)[0].acceleration == pytest.approx(.8)
    assert planner._initial_state(1.2)[0].acceleration == pytest.approx(1.6)


def test_planar_vehicle_memory_identity_survives_spawn_yaw_micro_noise_and_suspension_height():
    from carla_bridge.planning.experience_memory import execution_planner_version
    signatures, hashes = [], []
    for x, y, yaw, tiny, rear_z in [(0., 0., 0., 0., .3886),
                                  (44., -39., -90.337, 2e-6, .4024),
                                  (-164., -95., -179.913, -2e-6, .41)]:
        planner = fixture(x=x, y=y, yaw=yaw)
        # These micro-metre differences reproduce world->actor roundoff; z
        # changes are a suspension/terrain measurement, not planar dynamics.
        planner.vehicle_parameters.wheel_base += tiny
        planner.front_track_m += tiny
        planner.rear_axle_to_center += tiny
        rear = planner.rear_axle_actor_xyz_m
        planner.rear_axle_actor_xyz_m = (rear[0]+tiny, rear[1]-tiny, rear_z)
        signature = planner.execution_vehicle_signature()
        signatures.append(signature)
        hashes.append(execution_planner_version(planner.config, vehicle_model=signature))
        assert signature["measurement_resolution_m"] == .001
        assert "rear_axle_actor_xyz_m" not in signature
        assert len(signature["rear_axle_actor_xy_m"]) == 2
        # The function must return a detached value, without mutating telemetry.
        assert planner.rear_axle_actor_xyz_m[2] == rear_z
    assert all(signature == signatures[0] for signature in signatures)
    assert len(set(hashes)) == 1


@pytest.mark.parametrize("changed", ["wheelbase", "length", "width", "track", "rear_center", "rear_x",
                                   "rear_y", "physical_steer", "steering_curve", "blueprint"])
def test_actual_planar_geometry_or_steering_changes_partition_execution_memory(changed):
    from carla_bridge.planning.experience_memory import execution_planner_version
    planner = fixture()
    original = planner.execution_vehicle_signature()
    before = execution_planner_version(planner.config, vehicle_model=original)
    if changed == "wheelbase":
        planner.vehicle_parameters.wheel_base += .01
    elif changed in ("length", "width"):
        setattr(planner.vehicle_parameters, changed, getattr(planner.vehicle_parameters, changed)+.01)
    elif changed == "track":
        planner.front_track_m += .01
    elif changed == "rear_center":
        planner.rear_axle_to_center += .01
    elif changed.startswith("rear_"):
        rear = list(planner.rear_axle_actor_xyz_m)
        rear[0 if changed == "rear_x" else 1] += .01
        planner.rear_axle_actor_xyz_m = tuple(rear)
    elif changed == "physical_steer":
        planner.physical_max_steer += .01
    elif changed == "steering_curve":
        planner.steering_model.curve = ((0., 1.), (20., .6), (60., .4))
    elif changed == "blueprint":
        planner.ego.type_id = "vehicle.lincoln.mkz_2020"
    updated = planner.execution_vehicle_signature()
    assert updated != original
    assert execution_planner_version(planner.config, vehicle_model=updated) != before
