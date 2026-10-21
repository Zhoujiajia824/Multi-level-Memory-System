"""The actor-origin evaluation start must not attract a stopped rear axle."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from carla_bridge.planning.mpc_solver import MPCProblem
from carla_bridge.tests.test_mpc_planner import fixture
from carla_bridge.tests.test_pdm_planner import Waypoint


def spawn_route():
    planner = fixture()
    reference = [Waypoint(planner.map, value) for value in range(0,100,2)]
    planner.route._route_waypoints = reference
    planner.route._route = [waypoint.transform.location for waypoint in reference]
    planner.route._nearest_index = lambda location: 0
    return planner


def stop_reference(planner, local, initial):
    points = np.array([[wp.transform.location.x,wp.transform.location.y] for wp in local])
    planner._speed_limit = 8.
    return planner._reference("center:STOP", points, initial, 0., 3.5, {},
        headings=np.radians([wp.transform.rotation.yaw for wp in local]))


def test_spawn_actor_route_gets_only_private_verified_physical_predecessors():
    planner = spawn_route()
    original_route, original_waypoints = planner.route._route, planner.route._route_waypoints
    local, index = planner._resolve_reference()
    assert index == 0
    assert planner.route._route is original_route
    assert planner.route._route_waypoints is original_waypoints
    assert planner.route._route[0].x == 0.
    assert planner._resolved_route_waypoints[0].transform.location.x == 0.
    assert local[0].transform.location.x <= -8.
    assert planner._reference_start_extension["point_count"] == 4
    assert not planner._reference_start_extension["evaluation_route_modified"]
    # The same cached private prefix is reused without duplicating route points.
    again, _ = planner._resolve_reference()
    assert len(again) == len(local)


def test_stop_at_spawn_targets_actual_rear_station_and_executes_zero_speed_hold():
    planner = spawn_route()
    initial, _ = planner._initial_state(0.)
    assert initial.x < -1., "The physical rear axle really precedes evaluation station zero"
    local, _ = planner._resolve_reference()
    reference = stop_reference(planner, local, initial)
    assert reference.points[:,0] == pytest.approx(np.full(len(reference.points),initial.x))
    result = planner.solver.solve(MPCProblem(initial, reference))
    assert result.feasible and result.converged
    assert max(result.states[:,3]) < 1e-7
    assert result.states[-1,0] == pytest.approx(initial.x,abs=1e-7)


def test_ambiguous_or_disconnected_predecessor_is_not_guessed():
    for mode in ("ambiguous","disconnected"):
        planner = spawn_route()
        first = planner.route._route_waypoints[0]
        candidate = Waypoint(planner.map,-2.)
        if mode == "ambiguous":
            other = Waypoint(planner.map,-2.,1)
            first.previous = lambda distance:[candidate,other]
        else:
            candidate.next = lambda distance:[]
            first.previous = lambda distance:[candidate]
        local, _ = planner._resolve_reference()
        assert local[0].transform.location.x == 0.
        assert planner._reference_start_extension["point_count"] == 0
        assert planner._reference_start_extension["stop_reason"] == (
            "ambiguous_predecessors" if mode=="ambiguous" else "predecessor_not_verified_connected")


def test_provider_without_previous_keeps_real_stop_objective_and_original_road_corridor():
    planner = spawn_route()
    planner.route._route_waypoints[0].previous = None
    initial, _ = planner._initial_state(0.)
    local, _ = planner._resolve_reference()
    reference = stop_reference(planner,local,initial)
    assert reference.points[0,0] == 0., "Missing topology must not be fabricated"
    assert reference.metadata["stationary_hold_pose"] == [initial.x,initial.y,initial.yaw]
    result = planner.solver.solve(MPCProblem(initial,reference))
    assert result.feasible and result.converged
    assert max(result.states[:,3]) < 1e-7
    assert result.states[-1,0] == pytest.approx(initial.x,abs=1e-7)


def test_physical_prefix_follows_connected_geometry_across_only_verified_seam():
    planner = spawn_route()
    planner.map.seam = -1.
    planner.route._route_waypoints = [Waypoint(planner.map,value) for value in range(0,100,2)]
    planner.route._route = [wp.transform.location for wp in planner.route._route_waypoints]
    local, _ = planner._resolve_reference()
    assert local[0].road_id != planner._resolved_route_waypoints[0].road_id
    assert planner._reference_start_extension["point_count"] == 4
    assert local[-1].transform.location.x > 10.
