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


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for real-map topology regression")
def test_real_town05_left_lane_cannot_take_the_reference_right_exit():
    road = carla.Map("Town05", TOWN05.read_text())
    lane = road.get_waypoint(carla.Location(x=-208.76, y=91.334))
    branches = [[lane]]
    for _ in range(25):
        branches = [path + [following] for path in branches for following in path[-1].next(2)]
    chains = [list(dict.fromkeys((wp.road_id, wp.lane_id) for wp in path)) for path in branches]
    assert chains == [[(24, 1), (607, 1), (23, 1)], [(24, 1), (646, 1), (43, -1)]]
    assert not any(706 in [road_id for road_id, lane_id in chain] for chain in chains)


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for real-map late-exit regression")
def test_real_town05_vehicle_already_at_the_wrong_exit_gets_a_finite_stop_diagnostic():
    pytest.importorskip("tuplan_garage")
    pytest.importorskip("nuplan")
    from carla_bridge.tests.test_pdm_planner import Vehicle, semantic
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    road = carla.Map("Town05", TOWN05.read_text())
    waypoints, _ = resolve_route_waypoints(road, [carla.Location(x=x, y=y, z=z) for x, y, z in TURN_XYZ])
    points = [wp.transform.location for wp in waypoints]
    ego = Vehicle(x=-207.6, y=91.334, speed=0)
    world = SimpleNamespace(get_map=lambda: road, get_actors=lambda: [ego],
        get_snapshot=lambda: SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2)))
    route = SimpleNamespace(_route=points, _route_waypoints=waypoints,
        _nearest_index=lambda location: min(range(len(points)), key=lambda i: points[i].distance(location)))
    diagnostics = CarlaPDMPlanner(world, ego, route).plan(semantic("MAINTAIN"))["planner_diagnostics"]
    left = diagnostics["geometry"]["lateral_paths"]["left"]
    assert left["route_connection"]["reason"] == "route_exit_return_is_too_late"
    assert left["route_connection"]["finite_stopping_corridor_retained"]
    assert left["cut"]["last_lane_key"] == "24:0:1"
    assert left["path_length_m"] < 12
    assert diagnostics["route_recovery_gate_applied"] is False


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for real-map road-surface regression")
def test_real_town05_exit_surface_includes_lane_before_its_first_sparse_reference_sample():
    from shapely.geometry import Point
    from carla_bridge.planning.road_area import road_driving_polygons
    road = carla.Map("Town05", TOWN05.read_text())
    first_outer = road.get_waypoint(carla.Location(x=-195.04199, y=105.59914))
    assert (first_outer.road_id, first_outer.lane_id) == (42, 2)
    inner = first_outer.get_left_lane()
    assert (inner.road_id, inner.lane_id) == (42, 1)
    polygons, audit = road_driving_polygons(road, [first_outer, inner])
    # This actual LQR footprint corner was rejected by sparse neighbour strips:
    # 706 has no corresponding neighbour, so 42:1 previously began at y=105.6.
    corner = carla.Location(x=-193.1166961, y=104.1564308)
    assert road.get_waypoint(corner, project_to_road=False, lane_type=carla.LaneType.Driving) is not None
    assert polygons["42:0:1"].contains(Point(corner.x, -corner.y))
    assert not polygons["42:0:1"].contains(Point(-188, -104.1564308))
    assert audit["polygon_expansion_m"] == 0


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for physical-stall surface regression")
def test_real_town05_road_end_includes_the_recorded_73_percent_stall_corner():
    from shapely.geometry import Point
    from carla_bridge.planning.road_area import road_driving_polygons
    road = carla.Map("Town05", TOWN05.read_text())
    seeds = [road.get_waypoint_xodr(0, lane, 52) for lane in (1, 2)]
    cache = {}
    polygons, audit = road_driving_polygons(road, seeds, cache=cache)
    # The actual low-speed candidate in the local CARLA physics run was
    # rejected at s=53.54449, beyond the old final 53.5 grid cross-section.
    # Both corners are on real ordinary-road lanes near their junction seam.
    for lane, x, y in ((1, -36.3563600514671, 4.147040209108387),
                       (2, -35.284437351304824, 6.026268146036446)):
        actual = road.get_waypoint(carla.Location(x=x, y=y), project_to_road=False,
                                  lane_type=carla.LaneType.Driving)
        assert (actual.road_id, actual.lane_id) == (0, lane)
        assert polygons[f"0:0:{lane}"].contains(Point(x, -y))
    assert audit["exact_opendrive_endpoints"]
    assert audit["polygon_expansion_m"] == 0
    cached, _ = road_driving_polygons(road, seeds, cache=cache)
    assert all(cached[key] is polygon for key, polygon in polygons.items())


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for recorded physics rollout regression")
@pytest.mark.parametrize("preview_s", [1., 2.])
@pytest.mark.parametrize("points,pose,velocity,acceleration,steer,sign", [
    pytest.param([
        (-43.534,17.39394,.05545),(-43.56283,15.37315,.05545),(-43.59929,13.35888,.05545),
        (-43.50749,11.69512,.05545),(-43.05963,10.16536,.05544),(-42.2748,8.778,.04002),
        (-41.20385,7.62043,.0096),(-39.89649,6.76994,0),(-38.4197,6.26832,0),
        (-36.77868,6.13886,0),(-34.77869,6.13399,0),(-32.77869,6.12912,0),
        (-30.77871,6.12425,0),(-28.77871,6.11939,0),(-26.77872,6.11452,0),
        (-24.77872,6.10965,0),(-22.77873,6.10478,0),(-20.77874,6.09991,0),
        (-18.77874,6.09505,0),(-16.77875,6.09018,0),(-14.77875,6.08531,0),
        (-12.77876,6.08044,0),(-10.77876,6.07557,0),(-8.77877,6.0707,0),
        (-6.77878,6.06584,0),(-4.77878,6.06097,0),(-2.77879,6.0561,0),(-.77879,6.05123,0)],
        (-38.015556335,6.654981613,.008506292,-.406312762),
        (.108365335,-.019177094),(.17014198,-.033901185),.377732784,1, id="recorded_73_percent_exit"),
    pytest.param([
        (24.5471,162.59673,.00002),(24.5463,164.59673,.00002),(24.5455,166.59673,.00002),
        (24.5447,168.59673,.00002),(24.54391,170.59673,.00002),(24.54311,172.59673,.00002),
        (24.54231,174.59673,.00002),(24.54151,176.59673,.00002),(24.54071,178.59671,.00002),
        (24.51324,180.44435,.00002),(24.19544,182.01494,.00002),(23.53488,183.47488,.00001),
        (22.56493,184.7504,.00001),(21.30518,185.80922,.00001),(19.82754,186.66284,0),
        (18.22882,187.25961,0),(16.55326,187.58301,0)],
        (24.543762207,178.95880127,.001745507,1.569208816),(0,0),(0,0),0,-1,
        id="recorded_98_percent_exit"),
])
def test_real_town05_native_offsets_provide_a_road_compliant_moving_rollout(points, pose, velocity, acceleration, steer, sign, preview_s):
    """Frozen real physical states; improve proposals without changing scores."""
    pytest.importorskip("tuplan_garage")
    pytest.importorskip("nuplan")
    from carla_bridge.tests.test_pdm_planner import Vehicle, semantic, assert_score_only_selection
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    road = carla.Map("Town05", TOWN05.read_text())
    locations = [carla.Location(x=x,y=y,z=z) for x,y,z in points]
    waypoints, _ = resolve_route_waypoints(road, locations)
    x,y,z,yaw = pose
    ego = Vehicle(x=x,y=y,z=z,yaw=math.degrees(yaw),speed=math.hypot(*velocity))
    ego.bounding_box = carla.BoundingBox(carla.Location(x=.0292187948,z=.735860527),
        carla.Vector3D(x=4.791779518127441/2,y=2.163450002670288/2,z=.75))
    ego.wheelbase_m,ego.max_steer_deg,ego.rear_axle_local_x_m = 3.004646956723217,70,-1.386424422
    ego.get_velocity = lambda: carla.Vector3D(x=velocity[0],y=velocity[1])
    ego.get_acceleration = lambda: carla.Vector3D(x=acceleration[0],y=acceleration[1])
    ego.get_control = lambda: carla.VehicleControl(steer=steer)
    world = SimpleNamespace(get_map=lambda:road,get_actors=lambda:[ego],
        get_snapshot=lambda:SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2)))
    route = SimpleNamespace(_route=locations,_route_waypoints=waypoints,
        _nearest_index=lambda loc:min(range(len(locations)),key=lambda i:locations[i].distance(loc)))
    planner = CarlaPDMPlanner(world,ego,route,{"lqr_lateral_horizon_s":preview_s})
    diagnostics = [planner.plan(semantic("MAINTAIN","FOLLOW_ROUTE"),semantic_enabled=enabled)["planner_diagnostics"]
                   for enabled in (False,True)]
    assert diagnostics[0]["candidate_pool_sha256"] == diagnostics[1]["candidate_pool_sha256"]
    for audit in diagnostics:
        assert_score_only_selection(audit)
        original = next(c for c in audit["candidates"] if c["id"] == "center:MAINTAIN")
        if preview_s == 1.:
            assert original["safety"]["drivable_area"] == 0  # The original failure is not suppressed.
        chosen = next(c for c in audit["candidates"] if c["id"] == audit["selected_id"])
        if preview_s == 1.:
            assert chosen["lane_offset_m"] * sign > 0
        assert chosen["physical_lane_side"] == "center" and chosen["route_target"]
        assert chosen["safety"]["drivable_area"] == 1 and chosen["base_score"] > .5
        assert chosen["forward_progress_m"] > 5 and chosen["terminal_speed_mps"] > 1


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for frozen restart regression")
def test_real_town05_zero_wheel_angle_restart_has_a_native_road_compliant_choice_with_two_second_preview():
    """Recorded 98.8456% absorbing state: real dimensions, actual wheels zero.

    One-second native LQR preview reproduces the all-moving-road-zero failure.
    A separate two-second lateral preview and gentle native IDM proposal admit
    a moving choice, while longitudinal control retains its one-second horizon.
    This frozen-state check does not claim a whole physical route has passed.
    """
    pytest.importorskip("tuplan_garage")
    pytest.importorskip("nuplan")
    from carla_bridge.tests.test_pdm_planner import Vehicle, semantic, assert_score_only_selection
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    road=carla.Map("Town05",TOWN05.read_text())
    locations=[carla.Location(x=x,y=y,z=z)for x,y,z in [
        (24.5471,162.59673,.00002),(24.5463,164.59673,.00002),(24.5455,166.59673,.00002),
        (24.5447,168.59673,.00002),(24.54391,170.59673,.00002),(24.54311,172.59673,.00002),
        (24.54231,174.59673,.00002),(24.54151,176.59673,.00002),(24.54071,178.59671,.00002),
        (24.51324,180.44435,.00002),(24.19544,182.01494,.00002),(23.53488,183.47488,.00001),
        (22.56493,184.7504,.00001),(21.30518,185.80922,.00001),(19.82754,186.66284,0),
        (18.22882,187.25961,0),(16.55326,187.58301,0)]]
    ego=Vehicle(x=24.1381587982,y=181.3146209717,z=.001758209,yaw=math.degrees(1.64027367226))
    ego.bounding_box=carla.BoundingBox(carla.Location(x=.029218795,y=.000000925,z=.735860527),
        carla.Vector3D(x=2.395889759,y=1.081725001,z=.743830025))
    ego.wheelbase_m,ego.max_steer_deg,ego.rear_axle_local_x_m=3.004646956723217,70,-1.386424422
    def physics():
        wheels=[]
        for i in range(4):
            point=ego.get_transform().transform(carla.Location(
                x=ego.rear_axle_local_x_m+(ego.wheelbase_m if i<2 else 0),
                y=(-1 if i%2==0 else 1)*1.666828/2,z=.3708472))
            wheels.append(SimpleNamespace(position=carla.Vector3D(x=100*point.x,y=100*point.y,z=100*point.z),
                                          max_steer_angle=70 if i<2 else 0))
        return SimpleNamespace(wheels=wheels)
    ego.get_physics_control=physics
    ego.get_wheel_steer_angle=lambda which:0.
    world=SimpleNamespace(get_map=lambda:road,get_actors=lambda:[ego],
        get_snapshot=lambda:SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2)))
    route=SimpleNamespace(_route=locations,
        _nearest_index=lambda loc:min(range(len(locations)),key=lambda i:locations[i].distance(loc)))
    outcomes=[]
    for preview in (1.,2.):
        planner=CarlaPDMPlanner(world,ego,route,{"lqr_lateral_horizon_s":preview})
        audits=[planner.plan(semantic("MAINTAIN"),semantic_enabled=enabled)["planner_diagnostics"]for enabled in (False,True)]
        assert audits[0]["candidate_pool_sha256"]==audits[1]["candidate_pool_sha256"]
        for audit in audits:
            assert_score_only_selection(audit)
            assert audit["steering_state"]["source"]=="carla_measured_front_wheels"
            assert audit["steering_state"]["bicycle_angle_rad"]==0.
            moving=[c for c in audit["candidates"]if c["longitudinal"]not in ("STOP","DECELERATE")]
            if preview==1.:
                assert all(c["safety"]["drivable_area"]==0 for c in moving)
            else:
                chosen=next(c for c in audit["candidates"]if c["id"]==audit["selected_id"])
                assert chosen in moving
                assert chosen["safety"]["drivable_area"]==1 and chosen["base_score"]>0
                assert chosen["terminal_speed_mps"]>.5
                assert chosen["longitudinal_policy_id"]=="YIELD_GENTLE"
                assert audit["lqr_longitudinal_horizon_s"]==1.
        outcomes.append(audits[0])
    assert outcomes[0]["selected_id"].endswith(":STOP")
    assert not outcomes[1]["selected_id"].endswith(":STOP")


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for real-map route-exit regression")
@pytest.mark.parametrize("enabled", [True, False])
def test_real_town05_adjacent_proposal_returns_before_required_right_exit(enabled):
    """Measured Tesla footprint, rolling replans and native PDM scores.

    This proves legal lane selection before the exit. It does not claim the
    following tight right bend or CARLA actuator physics have passed.
    """
    pytest.importorskip("tuplan_garage")
    pytest.importorskip("nuplan")
    from carla_bridge.tests.test_pdm_planner import Vehicle, semantic, assert_score_only_selection
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    from carla_bridge.state.coords import ego_to_global
    road = carla.Map("Town05", TOWN05.read_text())
    waypoints, _ = resolve_route_waypoints(road, [carla.Location(x=x, y=y, z=z) for x, y, z in TURN_XYZ])
    for _ in range(40):
        waypoints.insert(0, waypoints[0].previous(2)[0])
    points = [wp.transform.location for wp in waypoints]
    ego = Vehicle(x=-242, y=91.33, z=.002, yaw=.06557, speed=4)
    ego.bounding_box = carla.BoundingBox(carla.Location(x=.0292187948, z=.735860527),
        carla.Vector3D(x=4.791779518127441 / 2, y=2.163450002670288 / 2, z=.75))
    wheelbase = 3.004646956723217
    ego.wheelbase_m, ego.max_steer_deg, ego.rear_axle_local_x_m = wheelbase, 70, -1.3864
    ego.steer = 0.0
    ego.get_control = lambda: carla.VehicleControl(steer=ego.steer)
    ego.get_velocity = lambda: carla.Vector3D(x=ego.speed * math.cos(math.radians(ego.yaw)),
                                             y=ego.speed * math.sin(math.radians(ego.yaw)))
    world = SimpleNamespace(get_map=lambda: road, get_actors=lambda: [ego],
                            get_snapshot=lambda: SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2)))
    route = SimpleNamespace(_route=points, _route_waypoints=waypoints,
        _nearest_index=lambda location: min(range(len(points)), key=lambda i: points[i].distance(location)))
    planner = CarlaPDMPlanner(world, ego, route)
    saw_return_proposal = False
    for step in range(35):
        world.get_snapshot = lambda: SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2 + .2 * step))
        plan = planner.plan(semantic("MAINTAIN", "KEEP_LANE"), semantic_enabled=enabled)
        diagnostics = plan["planner_diagnostics"]
        assert_score_only_selection(diagnostics)
        # STOP is a braking profile, not an instantaneous standstill. Native
        # ranking may select it briefly before a return; detect actual stalled
        # execution while retaining the same 35-step legal-return deadline.
        assert ego.speed > .5
        left = diagnostics["geometry"]["lateral_paths"]["left"]
        saw_return_proposal |= left.get("route_connection", {}).get("method") == "legal_return_in_spatial_proposal"
        choice = next(c for c in diagnostics["candidates"] if c["id"] == diagnostics["selected_id"])
        assert choice["safety"]["drivable_area"] == 1
        point = plan["trajectory"][2]
        x, y = ego_to_global(point["x"], point["y"], ego.x, ego.y, math.radians(ego.yaw))
        yaw = -math.degrees(planner.generator._state_array[choice["index"], 2, planner.api.StateIndex.HEADING])
        curvature = math.radians(yaw - ego.yaw) / max(1e-5, math.hypot(x - ego.x, y - ego.y))
        ego.steer = planner.steering_model.bicycle_to_normalized(math.atan(wheelbase * curvature), point["optional_v"])
        ego.x, ego.y, ego.yaw, ego.speed = x, y, yaw, point["optional_v"]
        assert ego.speed > .5
        ego.z = road.get_waypoint(ego.get_location()).transform.location.z + .0017
        if ego.x > -210 and ego.y > 94 and road.get_waypoint(ego.get_location()).lane_id == 2:
            # Acceptance here is the complete legal return before the turn;
            # full bend execution is covered separately below and in CARLA.
            break
    assert saw_return_proposal
    assert ego.x > -210 and ego.y > 94, (ego.x, ego.y)
    assert road.get_waypoint(ego.get_location()).lane_id == 2


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for optional real-map regression")
@pytest.mark.parametrize("length,width,wheelbase", [
    pytest.param(4.6, 1.8, 2.8, id="narrow_fixture"),
    pytest.param(4.8, 2.2, 3.004646956723217, id="wide_assumption_archive_wheelbase"),
])
def test_real_town05_continuous_low_speed_turn_with_pinned_pdm(length, width, wheelbase, record_property):
    """Ideal reference execution isolates planning from CARLA actuator physics.

    The regression must pass the whole bend, not just emit one moving plan.
    Actual server route completion and actuator comfort remain separate checks.
    The wide envelope is an explicit assumption, not measured Tesla dimensions:
    only its wheelbase is available from the supplied archive.
    The score-only experiment does not assert collision-free or road-compliant
    execution: it measures predicted road failures and actual origin deviation
    separately from completion, so removal of a veto is not called a safety fix.
    """
    pytest.importorskip("tuplan_garage")
    pytest.importorskip("nuplan")
    from carla_bridge.tests.test_pdm_planner import Vehicle, semantic, assert_score_only_selection
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    from carla_bridge.state.coords import ego_to_global

    road = carla.Map("Town05", TOWN05.read_text())
    points = [carla.Location(x=x, y=y, z=z) for x, y, z in TURN_XYZ]
    ego = Vehicle(x=-205.2371368408203, y=94.58367156982422, z=.00167799,
                  yaw=math.degrees(.01468131), speed=0)
    ego.bounding_box = carla.BoundingBox(carla.Location(x=.2, z=.8),
        carla.Vector3D(x=length / 2, y=width / 2, z=.8))
    ego.wheelbase_m, ego.max_steer_deg = wheelbase, 40
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
    initial_choice = next(c for c in initial["candidates"] if c["id"] == initial["selected_id"])
    assert initial_choice["physical_lane_side"] == "center"
    assert initial_choice["longitudinal"] == "YIELD"
    assert initial["candidate_pool_sha256"] == baseline["candidate_pool_sha256"]
    assert [c["safety"] for c in initial["candidates"]] == [c["safety"] for c in baseline["candidates"]]
    ego.get_acceleration = lambda: carla.Vector3D()
    planner = CarlaPDMPlanner(world, ego, route)
    speeds, choices, scored_offroad = [], [], False
    selected_offroad_steps, offsets = 0, []
    for step in range(90):
        world.get_snapshot = lambda: SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2 + step * .2))
        plan = planner.plan(semantic("YIELD"))
        diagnostics = plan["planner_diagnostics"]
        assert plan["trajectory"], (step, diagnostics)
        assert diagnostics["geometry"]["lateral_paths"]["center"]["cut"] is None
        choice = next(c for c in diagnostics["candidates"] if c["id"] == diagnostics["selected_id"])
        assert choice["eligible"]
        assert_score_only_selection(diagnostics)
        selected_offroad_steps += choice["safety"]["drivable_area"] == 0
        scored_offroad |= any(not c["safety"]["drivable_area"] and c["desired_speed_mps"] > 2
                               for c in diagnostics["candidates"])
        point = plan["trajectory"][2]
        x, y = ego_to_global(point["x"], point["y"], ego.x, ego.y, math.radians(ego.yaw))
        heading = planner.generator._state_array[choice["index"], 2, planner.api.StateIndex.HEADING]
        yaw = -math.degrees(heading)
        distance = max(1e-5, math.hypot(x - ego.x, y - ego.y))
        curvature = math.radians(yaw - ego.yaw) / distance
        ego.steer = planner.steering_model.bicycle_to_normalized(
            math.atan(planner.vehicle_parameters.wheel_base * curvature), point["optional_v"])
        ego.x, ego.y, ego.yaw, ego.speed = x, y, yaw, point["optional_v"]
        ego.z = road.get_waypoint(ego.get_location()).transform.location.z + .0017
        nearest = road.get_waypoint(ego.get_location()).transform.location
        offsets.append(math.hypot(ego.x - nearest.x, ego.y - nearest.y))
        speeds.append(ego.speed)
        choices.append(diagnostics["selected_id"])
    measurements = {"final_x_m": ego.x, "final_y_m": ego.y, "final_yaw_deg": ego.yaw,
                    "max_origin_lane_center_offset_m": max(offsets),
                    "selected_predicted_offroad_steps": selected_offroad_steps,
                    "min_speed_after_start_mps": min(speeds[5:]),
                    "stop_selection_count": sum("STOP" in name for name in choices),
                    "final_selected_id": choice["id"],
                    "final_candidate_score_audit": [{"id": c["id"], "base_score": c["base_score"],
                        "semantic_bonus": c["semantic_bonus"], "final_score": c["final_score"],
                        "drivable_area": c["safety"]["drivable_area"], "eligible": c["eligible"]}
                        for c in diagnostics["candidates"]]}
    for name, value in measurements.items():
        record_property(name, value)
    assert ego.y > 110 and 80 < ego.yaw < 100, (ego.x, ego.y, ego.yaw)
    assert min(speeds[5:]) > .5
    assert not any("STOP" in choice for choice in choices)
    assert scored_offroad  # Road failures still affect native scores, not eligibility.
