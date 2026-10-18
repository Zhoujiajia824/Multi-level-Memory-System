"""Only current map facts, without equating lane IDs with traffic permission."""
from types import SimpleNamespace as NS

import carla

from carla_bridge.planning.navigation_context import build_navigation_context
from carla_bridge.tests.test_pdm_route_geometry import EndingRoad, EndingWaypoint


def test_navigation_facts_distinguish_current_lane_from_route_lane():
    road = EndingRoad(junction=20)
    points = [EndingWaypoint(road, x) for x in range(0, 50, 2)]
    route = NS(_route_waypoints=points, _nearest_index=lambda loc: 3, map=road)
    ego = NS(get_location=lambda: carla.Location(x=6, y=-3.5))
    context = build_navigation_context(route, ego, 3.1)
    assert context["ego_lane_key"] != context["reference_lane_key"]
    assert context["current_side"] == "left"
    assert context["route_distance_to_junction_m"] == 14
    assert context["snapshot_time_s"] == 3.1
    assert "available_lateral_semantics" not in context


def test_junction_does_not_make_nearest_road_query_an_authoritative_lane_match():
    road = EndingRoad(junction=0)
    route = NS(_route_waypoints=[EndingWaypoint(road, x) for x in range(0, 20, 2)],
               _nearest_index=lambda loc: 2, map=road)
    context = build_navigation_context(route, NS(get_location=lambda: carla.Location(x=4)), 1)
    assert context["current_side"] == "unknown"
    assert context["route_distance_to_junction_m"] == 0


def test_context_absence_cannot_mean_route_blocked():
    assert build_navigation_context(NS(_route_waypoints=None), None, 0) == {
        "source": "CARLA_GT_and_ordered_route", "snapshot_time_s": 0.0}
