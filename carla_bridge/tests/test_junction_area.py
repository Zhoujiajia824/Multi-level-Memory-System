import os
from pathlib import Path
from types import SimpleNamespace

import carla
import pytest
from shapely.geometry import Point
from shapely.ops import unary_union

from carla_bridge.planning.junction_area import junction_driving_polygons


class JunctionWaypoint:
    def __init__(self, junction, lane, s):
        self.junction, self.lane, self.s = junction, lane, s
        self.road_id, self.section_id, self.lane_id = lane[0], 0, lane[3]
        self.lane_type, self.lane_width = carla.LaneType.Driving, 3.0
        self.is_junction = 0 <= s <= 10
        self.transform = carla.Transform(carla.Location(x=s, y=lane[1], z=lane[2]),
                                         carla.Rotation(yaw=0 if lane[3] < 0 else 180))

    def next(self, distance):
        return [JunctionWaypoint(self.junction, self.lane, self.s + distance * (-1 if self.lane_id > 0 else 1))]

    def get_junction(self):
        return self.junction


class Junction:
    id = 9

    def __init__(self):
        self.lanes = [(1, 0, 0, -1), (2, 4, 0, 1), (3, 0, 5, -1)]
        self.calls = 0

    def get_waypoints(self, lane_type):
        self.calls += 1
        return [(JunctionWaypoint(self, lane, 0 if lane[3] < 0 else 10),
                 JunctionWaypoint(self, lane, 10 if lane[3] < 0 else 0)) for lane in self.lanes]


def test_other_junction_driving_lanes_are_included_without_filling_sidewalks():
    junction = Junction()
    reference = [JunctionWaypoint(junction, junction.lanes[0], s) for s in (0, 5, 10)]
    polygons, diagnostics = junction_driving_polygons(reference)
    assert set(polygons) == {"1:0:-1", "2:0:1"}
    area = unary_union(list(polygons.values()))
    assert area.covers(Point(5, -4))
    assert not area.covers(Point(5, -2))  # no convex-hull filling between lanes
    assert not area.covers(Point(5, -7))
    assert diagnostics["excluded_height_segments"] > 0


def test_cached_samples_reapply_deck_filter_and_reuse_lane_unions():
    junction, cache = Junction(), {}
    lower = [JunctionWaypoint(junction, junction.lanes[0], 5)]
    upper = [JunctionWaypoint(junction, junction.lanes[2], 5)]
    a, _ = junction_driving_polygons(lower, cache=cache)
    b, info = junction_driving_polygons(upper, cache=cache)
    c, _ = junction_driving_polygons(lower, cache=cache)
    assert set(a) == set(c) == {"1:0:-1", "2:0:1"}
    assert set(b) == {"3:0:-1"}
    assert junction.calls == 1
    assert info["cache_hits"] == 1


def test_geometry_sampling_has_an_explicit_budget():
    junction = Junction()
    with pytest.raises(ValueError, match="geometry budget"):
        junction_driving_polygons([JunctionWaypoint(junction, junction.lanes[0], 5)],
                                  max_segments_per_junction=2)


def test_nonjunction_or_fixture_without_junction_api_has_no_extras():
    wp = SimpleNamespace(is_junction=True, transform=carla.Transform())
    polygons, info = junction_driving_polygons([wp])
    assert polygons == {}
    assert info["junction_ids"] == []
    assert junction_driving_polygons([])[0] == {}


TOWN05 = Path(os.environ.get("CARLA_TOWN05_XODR", ""))


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for optional real-map regression")
def test_real_town05_crossing_lane_is_drivable_but_sidewalk_is_not():
    road = carla.Map("Town05", TOWN05.read_text())
    reference = [road.get_waypoint_xodr(706, 1, s) for s in range(1, 17)]
    polygons, info = junction_driving_polygons(reference)
    assert "607:0:2" in polygons
    # Recorded YIELD front-left corner falsely outside the old route ribbon.
    corner = Point(-198.04979684300542, -94.24268700273116)
    assert polygons["607:0:2"].covers(corner)
    area = unary_union(list(polygons.values()))
    # A faster candidate truly cuts outside the outgoing road; adding real
    # junction lanes must never fill this sidewalk region.
    sidewalk = Point(-197.24395836417582, -109.67701607454234)
    assert road.get_waypoint(carla.Location(x=sidewalk.x, y=-sidewalk.y, z=.05545),
                             project_to_road=False, lane_type=carla.LaneType.Driving) is None
    assert not area.covers(sidewalk)
    assert info["junction_ids"] == [562]
