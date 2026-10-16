"""Real pinned PDM numerics against a deterministic CARLA-shaped road fixture.

No simulator or VLM is contacted. Install the optional PDM sources to run these
tests; missing optional sources are skipped, broken installed sources fail.
"""
from importlib.util import find_spec
import math
from types import SimpleNamespace as NS

import carla
import pytest

from carla_bridge.planning.pdm_planner import CarlaPDMPlanner, validate_dependencies

pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


class Waypoint:
    def __init__(self, road, x, lane=0):
        self.road, self.x, self.lane = road, float(x), lane
        self.road_id = 2 if road.seam and x >= road.seam else 1
        self.section_id, self.lane_id = 0, -lane - 4
        self.lane_width, self.is_junction = 3.5, False
        self.lane_type = carla.LaneType.Driving
        self.lane_change = carla.LaneChange.Both if road.allow_change else carla.LaneChange.NONE
        self.transform = carla.Transform(carla.Location(x=x, y=-3.5 * lane), carla.Rotation())

    def next(self, distance):
        return [Waypoint(self.road, self.x + distance, self.lane)]

    def previous(self, distance):
        return [Waypoint(self.road, self.x - distance, self.lane)]

    def get_left_lane(self):
        return Waypoint(self.road, self.x, self.lane + 1) if self.lane < 1 else None

    def get_right_lane(self):
        return Waypoint(self.road, self.x, self.lane - 1) if self.lane > -1 else None


class Road:
    def __init__(self, allow_change=True, seam=None):
        self.allow_change, self.seam = allow_change, seam

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        lane = min(1, max(-1, round(-location.y / 3.5)))
        return Waypoint(self, location.x, lane)


class Vehicle:
    type_id = "vehicle.tesla.model3"

    def __init__(self, actor_id=1, x=0, y=0, z=0.2, speed=0, yaw=0):
        self.id, self.x, self.y, self.z, self.speed, self.yaw = actor_id, x, y, z, speed, yaw
        self.bounding_box = carla.BoundingBox(carla.Location(x=0.2, z=0.8),
                                              carla.Vector3D(x=2.3, y=0.9, z=0.8))

    def get_location(self):
        return carla.Location(x=self.x, y=self.y, z=self.z)

    def get_transform(self):
        return carla.Transform(self.get_location(), carla.Rotation(yaw=self.yaw))

    def get_velocity(self):
        return carla.Vector3D(x=self.speed)

    def get_acceleration(self):
        return carla.Vector3D()

    def get_angular_velocity(self):
        return carla.Vector3D()

    def get_control(self):
        return carla.VehicleControl()

    def get_physics_control(self):
        return NS(wheels=[NS(position=carla.Vector3D(x=x, y=y), max_steer_angle=40 if x > 0 else 0)
                          for x, y in ((140, -80), (140, 80), (-140, -80), (-140, 80))])

    def get_speed_limit(self):
        return 30.0


class RedLight:
    id, type_id, state = 100, "traffic.traffic_light", carla.TrafficLightState.Red

    def __init__(self, road, x):
        self.road, self.x = road, x

    def get_stop_waypoints(self):
        return [Waypoint(self.road, self.x, lane) for lane in (-1, 0, 1)]


def make_planner(speed=0, x=0, length=100, allow_change=True, seam=None, others=None):
    road, ego = Road(allow_change, seam), Vehicle(speed=speed, x=x)
    world = NS(get_map=lambda: road, get_actors=lambda: [ego] + (others or []),
               get_snapshot=lambda: NS(timestamp=NS(elapsed_seconds=2.0)))
    route = NS(_route=[carla.Location(x=float(value)) for value in range(0, length + 1, 2)],
               _nearest_index=lambda location: min(length // 2 - 1, max(0, int(location.x / 2))))
    return CarlaPDMPlanner(world, ego, route, {"horizon_s": 4.0})


def semantic(longitudinal="MAINTAIN", lateral="FOLLOW_ROUTE"):
    return {"lateral_decision": lateral, "longitudinal_decision": longitudinal}


def test_real_core_sources_and_coordinates():
    assert validate_dependencies()["upstream_source_files_verified"] >= 30
    plan = make_planner().plan(semantic(), semantic_enabled=False)
    assert plan["trajectory"]
    assert plan["trajectory"][0]["x"] == pytest.approx(0, abs=1e-3)
    assert plan["trajectory"][0]["y"] == pytest.approx(0, abs=1e-3)
    assert plan["trajectory"][-1]["t"] >= 4
    from carla_bridge.control.trajectory_validation import TrajectoryValidator
    result = TrajectoryValidator().validate(plan["trajectory"], plan["target_speed"], 0, 3)
    assert result.accepted, result.to_dict()


def test_real_core_stop_ablation_changes_choice_only():
    planner = make_planner()
    on = planner.plan(semantic("STOP"), semantic_enabled=True)
    off = planner.plan(semantic("STOP"), semantic_enabled=False)
    assert on["selected_longitudinal"] == "STOP"
    assert on["target_speed"] == 0
    assert off["target_speed"] > 2
    first, second = on["planner_diagnostics"], off["planner_diagnostics"]
    assert first["candidate_pool_sha256"] == second["candidate_pool_sha256"]
    assert first["baseline_selected_id"] == second["selected_id"]


def test_maintain_starts_normal_cruise():
    plan = make_planner().plan(semantic("MAINTAIN"))
    assert plan["selected_longitudinal"] == "MAINTAIN"
    assert plan["target_speed"] > 3


def test_real_core_legal_left_change_and_forbidden_request():
    plan = make_planner(speed=4).plan(semantic(lateral="CHANGE_LANE_LEFT"))
    assert plan["selected_lateral"] == "CHANGE_LANE_LEFT"
    assert plan["trajectory"][-1]["y"] > 0.5  # Project frame y points LEFT.
    forbidden = make_planner(speed=4, allow_change=False).plan(semantic(lateral="CHANGE_LANE_LEFT"))
    assert forbidden["selected_lateral"] != "CHANGE_LANE_LEFT"
    assert forbidden["planner_diagnostics"]["unavailable_lateral_request"] == "CHANGE_LANE_LEFT"


def test_short_route_can_reach_last_checkpoint():
    plan = make_planner(length=6).plan(semantic(), semantic_enabled=False)
    assert plan["trajectory"][-1]["x"] > 6


def test_front_rear_road_seam_does_not_reject():
    plan = make_planner(speed=4, x=10.3, seam=10).plan(semantic())
    assert plan["trajectory"]


def test_initial_collision_cannot_be_ignored_by_upstream():
    plan = make_planner(others=[Vehicle(actor_id=9, x=2)]).plan(semantic("ACCELERATE"))
    assert plan["trajectory"] == []
    assert plan["planner_diagnostics"]["initial_collision_tokens"] == ["9"]
    assert not any(c["safe"] for c in plan["planner_diagnostics"]["candidates"])


def test_separate_bridge_level_does_not_create_false_collision():
    plan = make_planner(others=[Vehicle(actor_id=9, x=2, z=10)]).plan(semantic())
    assert plan["trajectory"]
    assert not plan["planner_diagnostics"]["initial_collision_tokens"]


def test_red_light_constraint_applies_with_semantics_on_and_off():
    planner = make_planner(speed=4)
    light = RedLight(planner.map, 16)
    planner.world.get_actors = lambda: [planner.ego, light]
    for enabled in (True, False):
        plan = planner.plan(semantic("ACCELERATE", "CHANGE_LANE_LEFT"), semantic_enabled=enabled)
        assert plan["trajectory"]
        assert max(point["x"] for point in plan["trajectory"]) < 16 - 2.3
        chosen = next(c for c in plan["planner_diagnostics"]["candidates"]
                      if c["id"] == plan["planner_diagnostics"]["selected_id"])
        assert chosen["safety"]["traffic_light"] is True


class CurvedWaypoint(Waypoint):
    def __init__(self, road, arc, lane=0):
        super().__init__(road, arc, lane)
        self.arc = arc
        radius = road.radius
        angle = max(0.0, min(math.pi / 2, arc / radius))
        if arc < 0:
            x, y = arc, 0.0
        elif arc > radius * math.pi / 2:
            x, y = radius, radius + arc - radius * math.pi / 2
        else:
            x, y = radius * math.sin(angle), radius * (1 - math.cos(angle))
        x += 3.5 * lane * math.sin(angle)
        y -= 3.5 * lane * math.cos(angle)
        self.transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation(yaw=math.degrees(angle)))
        self.is_junction = 0 < arc < radius * math.pi / 2

    def next(self, distance):
        return [CurvedWaypoint(self.road, self.arc + distance, self.lane)]

    def previous(self, distance):
        return [CurvedWaypoint(self.road, self.arc - distance, self.lane)]

    def get_left_lane(self):
        return CurvedWaypoint(self.road, self.arc, self.lane + 1) if self.lane < 1 else None

    def get_right_lane(self):
        return CurvedWaypoint(self.road, self.arc, self.lane - 1) if self.lane > -1 else None


class CurvedRoad(Road):
    radius = 18.0

    def parameter(self, location):
        if location.x < 0:
            return location.x
        if location.y > self.radius:
            return self.radius * math.pi / 2 + location.y - self.radius
        return self.radius * math.atan2(location.x, self.radius - location.y)

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        arc = self.parameter(location)
        center = CurvedWaypoint(self, arc)
        yaw = math.radians(center.transform.rotation.yaw)
        d = location - center.transform.location
        lateral = d.x * math.sin(yaw) - d.y * math.cos(yaw)
        return CurvedWaypoint(self, arc, max(-1, min(1, round(lateral / 3.5))))


def test_real_core_continuous_ninety_degree_intersection_turn():
    road = CurvedRoad()
    arc, speed = 6.0, 4.0
    wp = CurvedWaypoint(road, arc)
    ego = Vehicle(x=wp.transform.location.x, y=wp.transform.location.y,
                  yaw=wp.transform.rotation.yaw, speed=speed)
    yaw = math.radians(ego.yaw)
    ego.get_velocity = lambda: carla.Vector3D(x=speed * math.cos(yaw), y=speed * math.sin(yaw))
    ego.get_angular_velocity = lambda: carla.Vector3D(z=math.degrees(speed / road.radius))
    ego.get_control = lambda: carla.VehicleControl(steer=math.atan(2.8 / road.radius) / math.radians(40))
    world = NS(get_map=lambda: road, get_actors=lambda: [ego],
               get_snapshot=lambda: NS(timestamp=NS(elapsed_seconds=2.0)))
    route = NS(_route=[CurvedWaypoint(road, value).transform.location for value in range(-10, 92, 2)],
               _nearest_index=lambda location: max(0, int((road.parameter(location) + 10) / 2)))
    planner = CarlaPDMPlanner(world, ego, route, {"horizon_s": 4.0})
    plan = planner.plan(semantic())
    assert plan["trajectory"]
    assert plan["trajectory"][-1]["x"] > 5
    assert plan["trajectory"][-1]["y"] < -2  # CARLA right turn -> ego negative y.
    assert all(c["lateral"] == "KEEP_LANE" for c in plan["planner_diagnostics"]["candidates"])
    from carla_bridge.control.trajectory_validation import TrajectoryValidator
    valid = TrajectoryValidator().validate(plan["trajectory"], plan["target_speed"], speed, 3,
                                           use_speed_profile=True)
    assert valid.accepted, valid.to_dict()


@pytest.mark.parametrize("lateral", ["CHANGE_LANE_LEFT", "FOLLOW_ROUTE"])
def test_repeated_point_two_second_replans_preserve_maneuver_and_cruise(lateral):
    """Ideal reference execution isolates replanning from CARLA actuator error.

    Before the fix a fresh zero-tangent lateral blend reset every lane change,
    and upstream's default two-sample leader update applied a spurious initial
    brake each replan. In eight seconds it travelled 12.8m / shifted only 0.04m.
    """
    from carla_bridge.state.coords import ego_to_global
    planner = make_planner(speed=5)
    ego = planner.ego
    ego.get_velocity = lambda: carla.Vector3D(x=ego.speed * math.cos(math.radians(ego.yaw)),
                                             y=ego.speed * math.sin(math.radians(ego.yaw)))
    speeds = []
    for step in range(40):
        planner.world.get_snapshot = lambda: NS(timestamp=NS(elapsed_seconds=2 + step * 0.2))
        plan = planner.plan(semantic(lateral=lateral))
        assert plan["trajectory"], (step, plan["planner_diagnostics"])
        point = plan["trajectory"][2]
        assert point["t"] == pytest.approx(0.2)
        x, y = ego_to_global(point["x"], point["y"], ego.x, ego.y, math.radians(ego.yaw))
        candidate = next(c for c in plan["planner_diagnostics"]["candidates"]
                         if c["id"] == plan["planner_diagnostics"]["selected_id"])
        # Use the upstream reference vehicle orientation, not an assumed
        # sideslip-free center-path tangent, when advancing the snapshot.
        heading = planner.generator._state_array[candidate["index"], 2, planner.api.StateIndex.HEADING]
        ego.x, ego.y, ego.yaw, ego.speed = x, y, -math.degrees(heading), point["optional_v"]
        speeds.append(ego.speed)
    assert ego.x > 40
    assert min(speeds) >= 4.9
    assert ego.speed > 6
    if lateral == "CHANGE_LANE_LEFT":
        assert -4.0 < ego.y < -2.5
    else:
        assert abs(ego.y) < 0.1


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("barrier", ["end", "junction"])
def test_recover_reference_lane_before_return_opportunity_closes(enabled, barrier):
    from carla_bridge.tests.test_pdm_route_geometry import EndingRoad, ending_planner
    road = EndingRoad(end=60 if barrier == "end" else 1000,
                      junction=60 if barrier == "junction" else 1000)
    planner = ending_planner(road)
    plan = planner.plan(semantic(lateral="KEEP_LANE"), semantic_enabled=enabled)
    diagnostics = plan["planner_diagnostics"]
    assert plan["trajectory"]
    assert diagnostics["route_recovery_active"]
    assert diagnostics["selected_id"].startswith("center:")
    assert plan["selected_lateral"] == "CHANGE_LANE_RIGHT"
    assert diagnostics["geometry"]["lateral_paths"]["left"]["return_barrier"]["distance_ahead_m"] == 60
    other = planner.plan(semantic(lateral="KEEP_LANE"), semantic_enabled=not enabled)
    assert diagnostics["candidate_pool_sha256"] == other["planner_diagnostics"]["candidate_pool_sha256"]


def test_short_current_lane_remains_a_braking_path_when_return_is_forbidden():
    from carla_bridge.tests.test_pdm_route_geometry import EndingRoad, ending_planner
    planner = ending_planner(EndingRoad(end=30, junction=24), speed=2)
    plan = planner.plan(semantic("STOP", "KEEP_LANE"))
    assert plan["trajectory"]
    assert plan["target_speed"] == 0
    diagnostics = plan["planner_diagnostics"]
    assert diagnostics["selected_id"] == "left:STOP"
    assert diagnostics["excluded_lateral_paths"]["center"] == "lane_change_not_permitted_or_intersection"
    assert diagnostics["geometry"]["lateral_paths"]["left"]["cut"]["distance_ahead_m"] == 30


def test_repeated_keep_lane_commands_do_not_interrupt_necessary_recovery():
    from carla_bridge.tests.test_pdm_route_geometry import ending_planner
    from carla_bridge.state.coords import ego_to_global
    planner = ending_planner()
    ego = planner.ego
    ego.get_velocity = lambda: carla.Vector3D(x=ego.speed * math.cos(math.radians(ego.yaw)),
                                             y=ego.speed * math.sin(math.radians(ego.yaw)))
    for step in range(45):
        plan = planner.plan(semantic(lateral="KEEP_LANE"))
        assert plan["trajectory"], (step, plan["planner_diagnostics"])
        point = plan["trajectory"][2]
        x, y = ego_to_global(point["x"], point["y"], ego.x, ego.y, math.radians(ego.yaw))
        candidate = next(c for c in plan["planner_diagnostics"]["candidates"]
                         if c["id"] == plan["planner_diagnostics"]["selected_id"])
        heading = planner.generator._state_array[candidate["index"], 2, planner.api.StateIndex.HEADING]
        ego.x, ego.y, ego.yaw, ego.speed = x, y, -math.degrees(heading), point["optional_v"]
    assert ego.x > 45
    assert abs(ego.y) < 0.5


def test_return_boundary_limits_current_lane_when_recovery_is_forbidden():
    from carla_bridge.tests.test_pdm_route_geometry import EndingRoad, ending_planner
    # The neighboring lane geometrically continues through the junction. Its
    # candidate must stop before the return boundary, not follow that branch.
    planner = ending_planner(EndingRoad(end=1000, junction=20), speed=3)
    plan = planner.plan(semantic(lateral="KEEP_LANE"))
    assert plan["trajectory"]
    assert max(p["x"] for p in plan["trajectory"]) < 18
    assert plan["planner_diagnostics"]["geometry"]["lateral_paths"]["left"]["bounded_before_return_barrier"]
    assert plan["planner_diagnostics"]["excluded_lateral_paths"]["center"] == "lane_change_not_permitted_or_intersection"


def test_real_geometry_exception_contains_all_candidate_exclusions():
    from carla_bridge.planning.pdm_planner import RoadGeometryError
    from carla_bridge.tests.test_pdm_route_geometry import EndingRoad, ending_planner
    planner = ending_planner(EndingRoad(end=2, junction=0), x=1.9)
    with pytest.raises(RoadGeometryError) as error:
        planner.plan(semantic())
    details = error.value.diagnostics["geometry"]
    assert details["current_side"] == "left"
    assert details["ego_lane_key"]
    assert details["reference_lane_key"]
    assert set(details["excluded_lateral_paths"]) == {"center", "left", "right"}


def test_cannot_enter_lane_too_short_to_finish_change_before_return_boundary(monkeypatch):
    from carla_bridge.tests.test_pdm_route_geometry import EndingRoad, EndingWaypoint, ending_planner
    original = EndingWaypoint.__init__

    def asymmetric_marking(self, road, x, lane=0):
        original(self, road, x, lane)
        if lane == 1 and x >= 24:
            self.lane_change = carla.LaneChange.Left  # Outbound change is allowed; return is not.

    monkeypatch.setattr(EndingWaypoint, "__init__", asymmetric_marking)
    planner = ending_planner(EndingRoad(end=1000), y=0)
    plan = planner.plan(semantic(lateral="CHANGE_LANE_LEFT"))
    assert plan["trajectory"]
    assert plan["planner_diagnostics"]["excluded_lateral_paths"]["left"] == "insufficient_lane_change_corridor"


@pytest.mark.parametrize("enabled", [True, False])
def test_overlapping_junction_connector_id_does_not_stop_route_following(enabled, monkeypatch):
    from carla_bridge.planning.pdm_planner import RoadGeometryError
    from carla_bridge.tests.test_pdm_route_geometry import OverlappingJunctionRoad, ending_planner
    planner = ending_planner(OverlappingJunctionRoad(), x=12.1, y=-0.02, speed=2)
    with monkeypatch.context() as old_lookup:
        old_lookup.setattr("carla_bridge.planning.pdm_planner._junction_lane_match",
                           lambda *args: (None, {"method": "disabled_for_regression"}))
        with pytest.raises(RoadGeometryError, match="Ego has left the reference route"):
            planner.plan(semantic("YIELD"), semantic_enabled=enabled)
    plan = planner.plan(semantic("YIELD"), semantic_enabled=enabled)
    assert plan["trajectory"]
    geometry = plan["planner_diagnostics"]["geometry"]
    assert geometry["ego_lane_key"] == "489:0:-1"
    assert geometry["ego_lane_key"] != geometry["reference_lane_key"]
    assert geometry["current_side"] == "center"
    assert geometry["lane_match"]["method"] == "junction_geometry"
    assert plan["trajectory"][-1]["x"] > 3
