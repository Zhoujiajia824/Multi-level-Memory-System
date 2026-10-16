"""Lane endpoints and road seams, using CARLA-shaped map fixtures."""
from types import SimpleNamespace as NS
import math

import carla
import numpy as np
import pytest
from carla_bridge.planning.pdm_planner import _lane_corridor, _junction_lane_match
from carla_bridge.tests.test_pdm_planner import Waypoint, Road, Vehicle


class EndingWaypoint(Waypoint):
    def __init__(self, road, x, lane=0):
        super().__init__(road, x, lane)
        self.is_junction = x >= road.junction
        if self.is_junction:
            self.lane_change = carla.LaneChange.NONE
        if lane == 1 and x >= road.disconnected:
            self.road_id = 99

    def next(self, distance):
        if self.lane == 1 and self.x < self.road.disconnected <= self.x + distance:
            return []
        if self.lane == 1 and self.x + distance >= self.road.end:
            return []
        return [EndingWaypoint(self.road, self.x + distance, self.lane)]

    def previous(self, distance):
        return [EndingWaypoint(self.road, self.x - distance, self.lane)]

    def get_left_lane(self):
        if self.lane < 1 and self.road.start <= self.x < self.road.end:
            return EndingWaypoint(self.road, self.x, self.lane + 1)
        return None

    def get_right_lane(self):
        return EndingWaypoint(self.road, self.x, self.lane - 1) if self.lane > -1 else None


class EndingRoad(Road):
    def __init__(self, start=-1000, end=60, junction=1000, disconnected=1000):
        super().__init__()
        self.start, self.end, self.junction, self.disconnected = start, end, junction, disconnected

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        lane = min(1, max(-1, round(-location.y / 3.5)))
        return EndingWaypoint(self, location.x, lane)


def ending_planner(road=None, x=0, y=-3.5, speed=5):
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    road, ego = road or EndingRoad(), Vehicle(x=x, y=y, speed=speed)
    world = NS(get_map=lambda: road, get_actors=lambda: [ego],
               get_snapshot=lambda: NS(timestamp=NS(elapsed_seconds=2.0)))
    route = NS(_route=[carla.Location(x=float(value)) for value in range(-20, 160, 2)],
               _nearest_index=lambda location: max(0, int((location.x + 20) / 2)))
    return CarlaPDMPlanner(world, ego, route, {"horizon_s": 4.0})


def test_missing_adjacent_lane_behind_ego_does_not_discard_current_lane():
    road = EndingRoad(start=0, end=30)
    reference = [EndingWaypoint(road, x) for x in range(-10, 50, 2)]
    corridor, cut = _lane_corridor(reference, 8, "left", np.arange(len(reference)) * 2)
    assert corridor[0].x == 0
    assert corridor[-1].x == 28
    assert cut["distance_ahead_m"] == 24


def test_unconnected_successor_is_not_joined_even_when_nearby():
    road = EndingRoad(disconnected=20)
    reference = [EndingWaypoint(road, x) for x in range(0, 50, 2)]
    corridor, cut = _lane_corridor(reference, 0, "left", np.arange(len(reference)) * 2)
    assert corridor[-1].x == 18
    assert cut["reason"] == "disconnected_lane_successor"


def test_connected_road_seam_remains_continuous():
    road = Road(seam=20)
    reference = [Waypoint(road, x) for x in range(0, 50, 2)]
    corridor, cut = _lane_corridor(reference, 0, "left", np.arange(len(reference)) * 2)
    assert len(corridor) == len(reference)
    assert cut is None


class OverlappingJunctionRoad(EndingRoad):
    def __init__(self):
        super().__init__(end=1000, junction=10)
        self.seam = 10

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        wp = super().get_waypoint(location, project_to_road, lane_type)
        # Exact route samples map to the intended connection. A small lateral
        # tracking error picks the crossing connector in CARLA's nearest query.
        if location.x >= 10 and abs(location.y) > 0.001:
            wp.road_id, wp.lane_id = 489, -1
            wp.transform.rotation.yaw = 90
        return wp


@pytest.mark.parametrize("y,heading,height,expected", [
    (-0.02, 0, 0.2, "center"), (-3.48, 0, 0.2, "left"),
    (3.48, 0, 0.2, "right"), (-7.0, 0, 0.2, None),
    (-0.02, math.pi / 2, 0.2, None), (-0.02, math.pi, 0.2, None),
    (-0.02, 0, 8.0, None),
])
def test_junction_match_requires_containment_direction_and_height(y, heading, height, expected):
    road = OverlappingJunctionRoad()
    reference = [EndingWaypoint(road, x) for x in range(0, 50, 2)]
    side, evidence = _junction_lane_match(reference, 6, np.array([12.1, -y]), heading, height)
    assert side == expected
    if side:
        assert evidence["matches"][side]["offset_m"] < 0.03


def test_geometry_fallback_is_not_used_on_nonjunction_roads():
    road = EndingRoad(junction=1000)
    reference = [EndingWaypoint(road, x) for x in range(0, 50, 2)]
    side, _ = _junction_lane_match(reference, 6, np.array([12.1, 0.0]), 0, 0.2)
    assert side is None


def test_ambiguous_overlapping_route_lanes_fail_closed(monkeypatch):
    road = OverlappingJunctionRoad()
    reference = [EndingWaypoint(road, x) for x in range(0, 50, 2)]
    monkeypatch.setattr(EndingWaypoint, "get_left_lane", lambda self: self)
    side, evidence = _junction_lane_match(reference, 6, np.array([12.1, 0.0]), 0, 0.2)
    assert side is None
    assert {"left", "center"} <= set(evidence["matches"])
