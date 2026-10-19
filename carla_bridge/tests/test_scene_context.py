"""Exercise route geometry using current-state CARLA-shaped objects, without CARLA.

Projection and forecasting run unchanged: these are geometry tests, not mocked
conflict labels. The forecast remains a constant-velocity advisory proxy.
"""
from types import SimpleNamespace as NS
import math

import pytest

from carla_bridge.planning.scene_context import (
    build_scene_context, constant_velocity_conflict, point_at, project_polyline,
)
from carla_bridge.planning.semantic_governor import SelectiveSemanticGovernor


def location(x, y=0.0, z=0.0):
    return NS(x=float(x), y=float(y), z=float(z))


def waypoint(x, y=0.0, z=0.0, yaw=0.0, lane=1, road=1):
    return NS(transform=NS(location=location(x, y, z), rotation=NS(yaw=yaw)),
              road_id=road, section_id=0, lane_id=lane, lane_width=3.5,
              is_junction=False, get_left_lane=lambda: None, get_right_lane=lambda: None)


def vehicle(identity, x, y=0.0, z=0.0, yaw=0.0, vx=0.0, vy=0.0,
            lane=1, road=1, type_id="vehicle.test"):
    pose = NS(location=location(x, y, z), rotation=NS(yaw=yaw))
    velocity = NS(x=float(vx), y=float(vy), z=0.0)
    actor = NS(id=identity, type_id=type_id,
               bounding_box=NS(extent=NS(x=2.4, y=1.1, z=0.8)),
               get_transform=lambda: pose, get_location=lambda: pose.location,
               get_velocity=lambda: velocity, get_speed_limit=lambda: 36.0)
    actor.map_waypoint = waypoint(x, y, z, yaw, lane, road)
    return actor


def traffic_light(identity, *stops, state="Red"):
    return NS(id=identity, type_id="traffic.traffic_light.test",
              state="TrafficLightState." + state, get_stop_waypoints=lambda: stops)


def scene(*actors, ego=None, reference=None, nearest=0, progress=0.0,
          fog_density=0.0, retained_reference=None, special_situations=()):
    ego = ego or vehicle(1, 0, vx=5.0)
    reference = reference or [waypoint(x) for x in range(0, 202, 2)]
    geometries = [ego] + [a for a in actors if hasattr(a, "map_waypoint")]
    map_waypoints = {
        (a.get_location().x, a.get_location().y, a.get_location().z): a.map_waypoint
        for a in geometries
    }
    carla_map = NS(name="Town05", get_waypoint=lambda p: map_waypoints.get((p.x, p.y, p.z)))
    world = NS(get_map=lambda: carla_map, get_actors=lambda: [ego, *actors],
               get_weather=lambda: NS(fog_density=fog_density))
    route = NS(_nearest_index=lambda p: nearest,
               _route_waypoints=retained_reference or reference,
               distance_completed_m=lambda: progress, nav_instruction=lambda: "STRAIGHT")
    return build_scene_context(world, ego, route, 10.0, reference=reference,
                               special_situations=special_situations)


def test_neighboring_opposing_vehicle_does_not_conflict_or_explain_stopping():
    opposing = vehicle(7, 20, y=3.5, yaw=180, vx=-5, lane=-1)
    context = scene(opposing)
    actor = context["actors"][0]
    assert actor["relative_s_m"] == 20
    assert abs(actor["relative_l_m"]) == 3.5
    assert not actor["same_lane"] and not actor["conflict"]
    assert actor["ttc_s"] is None
    assert not context["blocked"] and not context["baseline_stop_expected"]


def test_same_lane_label_alone_does_not_make_opposing_heading_a_queue():
    opposing = vehicle(7, 8, y=3.5, yaw=180, vx=0, lane=1)
    context = scene(opposing)
    assert not context["actors"][0]["same_lane"]
    assert not context["actors"][0]["conflict"]
    assert not context["baseline_stop_expected"]


def test_crossing_vehicle_conflicts_only_at_shared_future_time():
    crossing = vehicle(7, 10, y=-10, yaw=90, vy=5, road=2)
    context = scene(crossing)
    actor = context["actors"][0]
    assert actor["conflict"] and 0 < actor["ttc_s"] <= 2.0
    assert not actor["same_lane"]
    assert context["prediction_kind"] == "constant_velocity_proxy_not_future_GT"
    assert "proxy" in context["source"]


def test_same_geometric_crossing_at_a_different_time_is_not_conflict():
    late_crossing = vehicle(7, 10, y=-25, yaw=90, vy=5, road=2)
    context = scene(late_crossing)
    assert not context["actors"][0]["conflict"]
    assert context["actors"][0]["ttc_s"] is None
    assert not context["blocked"] and not context["baseline_stop_expected"]


def test_stationary_neighbor_is_neither_a_route_block_nor_a_queue():
    neighbor = vehicle(7, 8, y=3.5, lane=2)
    context = scene(neighbor)
    assert not context["blocked"]
    assert not context["baseline_stop_expected"]
    assert not context["actors"][0]["path_blocked"]


def test_one_stopped_vehicle_cannot_distinguish_queue_from_parked_obstacle():
    near = scene(vehicle(7, 8))
    far = scene(vehicle(7, 20))
    assert near["blocked"] and not near["baseline_stop_expected"]
    assert far["blocked"] and not far["baseline_stop_expected"]
    candidates = [{"id": "move", "base_score": 1.0, "forward_progress_m": 20,
                   "lateral_semantics": ["FOLLOW_ROUTE"], "longitudinal": "MAINTAIN"}]
    request = SelectiveSemanticGovernor().query_decision(far, candidates, 10.0)
    assert request["query"] and request["reason"] == "path_blocked"


@pytest.mark.parametrize("kind", ["static.prop.trafficcone01", "walker.pedestrian.0001"])
def test_near_obstacle_or_pedestrian_is_not_misclassified_as_a_vehicle_queue(kind):
    context = scene(vehicle(7, 8, type_id=kind), vehicle(8, 16, type_id=kind))
    assert context["blocked"] and not context["baseline_stop_expected"]


@pytest.mark.parametrize("speed", [0.0, 0.6])
def test_two_nearby_same_direction_slow_vehicles_explain_queue_wait(speed):
    context = scene(vehicle(7, 8, vx=speed), vehicle(8, 16, vx=speed))
    assert context["baseline_stop_expected"]


@pytest.mark.parametrize("second", [vehicle(8, 29), vehicle(8, 16, yaw=180),
                                    vehicle(8, 16, y=3.5, lane=2),
                                    vehicle(8, 16, vx=5)])
def test_sparse_opposing_neighboring_or_fast_vehicles_do_not_make_a_queue(second):
    context = scene(vehicle(7, 4), second)
    assert not context["baseline_stop_expected"]


@pytest.mark.parametrize("state, expected", [("Red", True), ("Yellow", True),
                                            ("Green", False), ("Off", False)])
def test_applicable_near_stop_line_is_bumper_relative(state, expected):
    context = scene(traffic_light(11, waypoint(10), state=state))
    assert len(context["signals"]) == 1
    assert context["signals"][0]["id"] == "stop_11_0"
    assert context["signals"][0]["stop_line_distance_m"] == pytest.approx(7.6)
    assert context["baseline_stop_expected"] is expected


@pytest.mark.parametrize("stop", [waypoint(10, lane=2), waypoint(10, road=2),
                                  waypoint(10, yaw=180), waypoint(10, y=3.5),
                                  waypoint(10, z=5)])
def test_wrong_lane_direction_lateral_position_or_bridge_stop_line_is_excluded(stop):
    context = scene(traffic_light(11, stop))
    assert context["signals"] == []
    assert not context["baseline_stop_expected"]


def test_far_red_line_is_recorded_but_does_not_explain_current_stop():
    context = scene(traffic_light(11, waypoint(50)))
    assert context["signals"][0]["stop_line_distance_m"] == pytest.approx(47.6)
    assert not context["baseline_stop_expected"]


@pytest.mark.parametrize("stop_x", [1.5, 1.9, 2.39])
def test_front_bumper_already_past_stop_line_does_not_explain_stopping(stop_x):
    context = scene(traffic_light(11, waypoint(stop_x)))
    assert context["signals"][0]["stop_line_distance_m"] < 0
    assert not context["baseline_stop_expected"]


def test_stop_line_clearly_behind_ego_is_excluded():
    ego = vehicle(1, 20, vx=5)
    context = scene(traffic_light(11, waypoint(15)), ego=ego, nearest=10, progress=20)
    assert context["signals"] == []
    assert not context["baseline_stop_expected"]


def test_other_bridge_actor_at_same_xy_does_not_block_or_conflict():
    above = vehicle(7, 8, z=6)
    context = scene(above)
    assert context["actors"] == []
    assert not context["blocked"] and not context["baseline_stop_expected"]


def test_explicit_global_reference_matches_global_route_index_and_progress():
    reference = [waypoint(x) for x in range(0, 402, 2)]
    ego = vehicle(1, 200, vx=5)
    lead = vehicle(7, 210, vx=5)
    light = traffic_light(11, waypoint(220))
    context = scene(lead, light, ego=ego, reference=reference, nearest=100, progress=200,
                    retained_reference=[waypoint(-100), waypoint(-90)])
    assert context["route_progress_m"] == 200
    assert context["actors"][0]["route_distance_m"] == pytest.approx(10)
    assert context["signals"][0]["stop_line_distance_m"] == pytest.approx(17.6)
    assert not context["blocked"] and not context["baseline_stop_expected"]


def test_near_route_end_and_clamped_behind_projection_do_not_invent_a_block():
    reference = [waypoint(x) for x in range(0, 402, 2)]
    ego = vehicle(1, 398, vx=5)
    behind = vehicle(7, 386)
    context = scene(behind, traffic_light(11, waypoint(399)), ego=ego,
                    reference=reference, nearest=199, progress=398)
    actor = context["actors"][0]
    assert actor["relative_s_m"] == -12
    # The local route starts at x=392, so the behind actor's projection clamps
    # to that endpoint. Its along-route sign must nevertheless remain behind.
    assert actor["route_distance_m"] < 0
    assert not actor["path_blocked"] and not context["blocked"]
    assert context["signals"][0]["stop_line_distance_m"] == pytest.approx(-1.4)
    assert not context["baseline_stop_expected"]


def test_last_route_index_still_uses_two_distinct_local_points():
    reference = [waypoint(0), waypoint(10), waypoint(20), waypoint(30)]
    context = scene(ego=vehicle(1, 30), reference=reference, nearest=3, progress=30)
    assert context["route_progress_m"] == 30
    assert context["signals"] == []


def test_visibility_is_observed_but_does_not_authorize_stop_by_itself():
    context = scene(fog_density=50, special_situations=("road_work", "road_work"))
    assert context["special_situations"] == ["limited_visibility_fog", "road_work"]
    assert not context["baseline_stop_expected"]


def test_degenerate_local_route_fails_explicitly_instead_of_claiming_clear():
    with pytest.raises(ValueError, match="distinct local segments"):
        scene(reference=[waypoint(0), waypoint(0)])
    with pytest.raises(ValueError, match="ordered map route"):
        scene(reference=[waypoint(0)])


def test_projection_and_forecast_follow_a_bend_with_matching_time():
    points = [(0, 0), (10, 0), (10, 20), (10, 20)]
    assert project_polyline(10, 5, points) == pytest.approx((15, 0, math.pi / 2))
    assert point_at(15, points) == (10, 5)
    # Ego turns north at t=2 and meets the westbound crossing near t=3.
    conflict, ttc = constant_velocity_conflict(0, 5, points, (25, 5), (-5, 0))
    assert conflict and 2 < ttc <= 3


def test_stationary_ego_prediction_is_explicit_possible_proceeding_proxy():
    points = [(0, 0), (100, 0)]
    conflict, ttc = constant_velocity_conflict(0, 0, points, (4, -4), (0, 2))
    assert conflict and 0 < ttc <= 2
