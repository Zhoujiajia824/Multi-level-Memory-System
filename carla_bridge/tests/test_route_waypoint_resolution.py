"""Connected route identity survives overlapping junction connections."""
import math
import os
from pathlib import Path
from types import SimpleNamespace

import carla
import pytest

from carla_bridge.planning.route_waypoints import resolve_route_waypoints


class ForkWaypoint:
    def __init__(self, road, x, branch=0):
        self.road, self.x, self.branch = road, x, branch
        self.road_id, self.section_id, self.lane_id = (10 if branch == 0 else 20 + branch), 0, -1
        self.lane_type, self.is_junction = carla.LaneType.Driving, branch != 0
        y = 0.18 * max(0, x - 4) ** 2 if branch == 2 else 0
        slope = 0.36 * max(0, x - 4) if branch == 2 else 0
        self.transform = carla.Transform(carla.Location(x=x, y=y),
                                         carla.Rotation(yaw=math.degrees(math.atan(slope))))

    def next(self, distance):
        x = self.x + distance
        if self.branch == 0 and x >= 2:
            return [ForkWaypoint(self.road, x, 1), ForkWaypoint(self.road, x, 2)]
        return [ForkWaypoint(self.road, x, self.branch)]


class ForkRoad:
    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        # Both connection lines overlap until x=4. Nearest lookup arbitrarily
        # chooses straight; only the later ordered points determine the route.
        branch = 0 if location.x < 2 else 1 if abs(location.y) < 0.01 else 2
        return ForkWaypoint(self, location.x, branch)


def fork_points():
    road = ForkRoad()
    return road, [ForkWaypoint(road, x, 0 if x < 2 else 2).transform.location for x in range(10)]


def test_overlapping_fork_uses_future_route_points_without_lane_jump():
    road, points = fork_points()
    before = [(p.x, p.y, p.z) for p in points]
    wps, info = resolve_route_waypoints(road, points)
    assert [wp.branch for wp in wps] == [0, 0] + [2] * 8
    assert info["reprojection_mismatch_indices"] == [2, 3, 4]
    assert info["max_point_error_m"] < 0.001
    assert [(p.x, p.y, p.z) for p in points] == before
    assert info["evaluation_points_unchanged"]


def test_single_candidate_greedy_lookup_cannot_recover_this_fork():
    road, points = fork_points()
    with pytest.raises(ValueError, match="cannot be reached through driving-lane successors"):
        resolve_route_waypoints(road, points, beam_width=1)


@pytest.mark.parametrize("offset", [(0, 3.5, 0), (0, 0, 4), (-7, 0, 0)])
def test_unconnected_neighbor_deck_or_reverse_point_is_rejected(offset):
    road = ForkRoad()
    points = [carla.Location(x=0), carla.Location(x=1), carla.Location(x=2 + offset[0], y=offset[1], z=offset[2])]
    with pytest.raises(ValueError, match="cannot be reached through driving-lane successors"):
        resolve_route_waypoints(road, points)


def test_duplicate_checkpoint_is_preserved_without_advancing_branch():
    road, points = fork_points()
    points.insert(3, points[2])
    wps, info = resolve_route_waypoints(road, points)
    assert len(wps) == len(points)
    assert wps[2] is wps[3]
    assert info["point_count"] == len(points)


def test_existing_fixed_scenario_retains_coordinates_and_resolves_navigation():
    from carla_bridge.env.route_planner import RoutePlanner, NAV_RIGHT
    road, locations = fork_points()
    original = [{"x": p.x, "y": p.y, "z": p.z} for p in locations]
    ego = SimpleNamespace(get_location=lambda: locations[0])
    route = RoutePlanner(SimpleNamespace(get_map=lambda: road), ego)
    route.load_fixed_points(original)
    assert [{"x": p.x, "y": p.y, "z": p.z} for p in route._route] == original
    assert [wp.branch for wp in route._route_waypoints] == [0, 0] + [2] * 8
    assert NAV_RIGHT in route._options
    assert route._waypoint_resolution["reprojection_mismatch_count"] == 3


@pytest.mark.parametrize("points", [[], [carla.Location(x=float("nan"))]])
def test_missing_or_nonfinite_route_is_rejected(points):
    with pytest.raises(ValueError):
        resolve_route_waypoints(ForkRoad(), points)


TOWN05 = Path(os.environ.get("CARLA_TOWN05_XODR", ""))
TURN_XYZ = [(-208.76048, 94.82786, 0), (-206.76048, 94.83014, 0),
            (-204.76048, 94.83243, 0), (-202.76047, 94.83472, 0),
            (-201.10419, 94.97231, 0), (-199.56129, 95.45192, 0.00115),
            (-198.15848, 96.25355, 0.02936), (-196.96066, 97.34045, 0.05499),
            (-196.01897, 98.66209, 0.05545), (-195.37907, 100.15341, 0.05545),
            (-195.07011, 101.74654, 0.05545), (-195.04254, 103.59914, 0.05545),
            (-195.04199, 105.59914, 0.05545)]


@pytest.mark.skipif(not TOWN05.is_file(), reason="optional CARLA Town05 OpenDRIVE is not installed")
def test_real_town05_fixed_route_uses_706_at_both_overlapping_seams():
    road = carla.Map("Town05", TOWN05.read_text())
    # Recorded fixed-route checkpoints at the stalled right turn. Several
    # connections overlap at both ends of road 706 in the real CARLA map.
    points = [carla.Location(x=x, y=y, z=z) for x, y, z in TURN_XYZ]
    assert road.get_waypoint(points[3]).road_id == 607
    assert road.get_waypoint(points[11]).road_id == 694
    wps, info = resolve_route_waypoints(road, points)
    assert list(dict.fromkeys(wp.road_id for wp in wps)) == [24, 706, 42]
    assert wps[3].lane_id == wps[11].lane_id == 1
    assert info["max_point_error_m"] < 0.2  # first nearest point has a station bias
    assert info["reprojection_mismatch_indices"] == [3, 11]
    from carla_bridge.planning.pdm_planner import _connected
    assert all(_connected(a, b) for a, b in zip(wps, wps[1:]))


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for optional real-map regression")
@pytest.mark.parametrize("length,width,wheelbase", [
    pytest.param(4.6, 1.8, 2.8, id="narrow_fixture"),
    pytest.param(4.8, 2.2, 3.004646956723217, id="wide_assumption_archive_wheelbase",
        marks=pytest.mark.xfail(strict=True, raises=AssertionError,
            reason="Known wide-vehicle continuous-turn stall; real vehicle geometry and "
                   "trajectory generation/prediction consistency remain unresolved")),
])
def test_real_town05_continuous_low_speed_turn_with_pinned_pdm(length, width, wheelbase):
    """Ideal reference execution isolates planning from CARLA actuator physics.

    The regression must pass the whole bend, not just emit one moving plan.
    Actual server route completion and actuator comfort remain separate checks.
    The wide envelope is an explicit assumption, not measured Tesla dimensions:
    only its wheelbase is available from the supplied archive. Keep that known
    failing case visible until real geometry and planning consistency are fixed.
    """
    pytest.importorskip("tuplan_garage")
    pytest.importorskip("nuplan")
    from carla_bridge.tests.test_pdm_planner import Vehicle, semantic
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    from carla_bridge.state.coords import ego_to_global

    road = carla.Map("Town05", TOWN05.read_text())
    points = [carla.Location(x=x, y=y, z=z) for x, y, z in TURN_XYZ]
    ego = Vehicle(x=-205.2371368408203, y=94.58367156982422, z=.00167799,
                  yaw=math.degrees(.01468131), speed=0)
    ego.bounding_box = carla.BoundingBox(carla.Location(x=.2, z=.8),
        carla.Vector3D(x=length / 2, y=width / 2, z=.8))
    ego.get_physics_control = lambda: SimpleNamespace(wheels=[
        SimpleNamespace(position=carla.Vector3D(x=x, y=y), max_steer_angle=40 if x > 0 else 0)
        for x, y in ((wheelbase * 50, -80), (wheelbase * 50, 80),
                     (-wheelbase * 50, -80), (-wheelbase * 50, 80))])
    ego.steer = 0.0
    ego.get_control = lambda: carla.VehicleControl(steer=ego.steer)
    ego.get_velocity = lambda: carla.Vector3D(x=ego.speed * math.cos(math.radians(ego.yaw)),
                                             y=ego.speed * math.sin(math.radians(ego.yaw)))
    world = SimpleNamespace(get_map=lambda: road, get_actors=lambda: [ego],
                            get_snapshot=lambda: SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2)))
    route = SimpleNamespace(_route=points,
        _nearest_index=lambda location: min(range(len(points)), key=lambda i: points[i].distance(location)))
    planner = CarlaPDMPlanner(world, ego, route)
    # Recorded suspension/standstill acceleration must not be silently removed
    # to pass the map check. A valid driving surface accepts this snapshot too.
    ego.get_acceleration = lambda: carla.Vector3D(x=.627 * math.cos(math.radians(ego.yaw)),
                                                  y=.627 * math.sin(math.radians(ego.yaw)))
    initial = planner.plan(semantic("YIELD"))["planner_diagnostics"]
    baseline = planner.plan(semantic("YIELD"), semantic_enabled=False)["planner_diagnostics"]
    assert not initial["emergency_brake"]
    assert initial["selected_id"] == "center:YIELD"
    assert initial["candidate_pool_sha256"] == baseline["candidate_pool_sha256"]
    assert [c["safety"] for c in initial["candidates"]] == [c["safety"] for c in baseline["candidates"]]
    ego.get_acceleration = lambda: carla.Vector3D()
    planner = CarlaPDMPlanner(world, ego, route)
    speeds, choices, rejected_moving = [], [], False
    for step in range(90):
        world.get_snapshot = lambda: SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2 + step * .2))
        plan = planner.plan(semantic("YIELD"))
        diagnostics = plan["planner_diagnostics"]
        assert plan["trajectory"], (step, diagnostics)
        assert diagnostics["geometry"]["lateral_paths"]["center"]["cut"] is None
        choice = next(c for c in diagnostics["candidates"] if c["id"] == diagnostics["selected_id"])
        assert choice["safe"]
        rejected_moving |= any(not c["safety"]["drivable_area"] and c["desired_speed_mps"] > 2
                               for c in diagnostics["candidates"])
        point = plan["trajectory"][2]
        x, y = ego_to_global(point["x"], point["y"], ego.x, ego.y, math.radians(ego.yaw))
        heading = planner.generator._state_array[choice["index"], 2, planner.api.StateIndex.HEADING]
        yaw = -math.degrees(heading)
        distance = max(1e-5, math.hypot(x - ego.x, y - ego.y))
        curvature = math.radians(yaw - ego.yaw) / distance
        ego.steer = max(-1, min(1, math.atan(planner.vehicle_parameters.wheel_base * curvature) / planner.max_steer))
        ego.x, ego.y, ego.yaw, ego.speed = x, y, yaw, point["optional_v"]
        ego.z = road.get_waypoint(ego.get_location()).transform.location.z + .0017
        speeds.append(ego.speed)
        choices.append(diagnostics["selected_id"])
    assert ego.y > 110 and 80 < ego.yaw < 100, (ego.x, ego.y, ego.yaw)
    assert min(speeds[5:]) > .5
    assert not any("STOP" in choice for choice in choices)
    assert rejected_moving  # actual fast off-road proposals still fail safety
