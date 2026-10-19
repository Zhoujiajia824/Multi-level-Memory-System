"""A projected route point must retain its resolved connection identity."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carla_bridge.planning.pdm_planner import _reference_projection_lane, RoadGeometryError
from carla_bridge.tests.test_pdm_planner import Road, Waypoint, make_planner
from carla_bridge.tests.test_recovery_planner_integration import road_snapshot, make_snapshot_planner


def test_crossing_connection_cannot_become_the_reference_base_lane():
    road = Road()
    reference = [Waypoint(road, x) for x in (0., 2., 4.)]
    crossing = Waypoint(road, 1.)
    crossing.road_id = 99
    resolved, audit = _reference_projection_lane(reference, np.array([0., 2., 4.]), 1., crossing)
    assert resolved is reference[0]
    assert not audit["nearest_map_identity_accepted"]
    assert audit["projected_lane_key"] == "99:0:-4"
    assert audit["reference_segment_lane_keys"] == ["1:0:-4", "1:0:-4"]


@pytest.mark.parametrize("station,seam", [(9.8, 9.9), (8.2, 8.1)])
def test_both_true_road_seam_identities_survive_nearest_checkpoint_disagreement(station, seam):
    road = Road(seam=seam)
    reference = [Waypoint(road, x) for x in (8., 10., 12.)]
    projected = Waypoint(road, station)
    nearest = min(reference, key=lambda wp: abs(wp.x - station))
    assert projected.road_id != nearest.road_id
    resolved, audit = _reference_projection_lane(reference, np.array([0., 2., 4.]), station - 8., projected)
    assert resolved is projected
    assert audit["nearest_map_identity_accepted"]


@pytest.mark.parametrize("station,connected,accepted", [(2.02, True, True), (2.06, True, False),
                                                         (2.02, False, False)])
def test_checkpoint_roundoff_uses_only_a_nearby_connected_seam(station, connected, accepted):
    class ShortRoadWaypoint(Waypoint):
        def __init__(self, road, x):
            super().__init__(road, x)
            self.road_id = 1 + int(x // 2)

        def next(self, distance):
            return [ShortRoadWaypoint(self.road, self.x + distance)]

    road = Road()
    reference = [ShortRoadWaypoint(road, x) for x in (0., 2., 4., 6.)]
    if not connected:
        reference[0].next = lambda distance: []
    predecessor_station = ShortRoadWaypoint(road, 1.99)
    result, audit = _reference_projection_lane(reference, np.array([0., 2., 4., 6.]),
                                              station, predecessor_station)
    assert audit["nearest_map_identity_accepted"] is accepted
    assert result is (predecessor_station if accepted else reference[1])


@pytest.mark.parametrize("y,side", [(0., "center"), (-3.5, "left"), (3.5, "right")])
def test_nonjunction_seam_preserves_actual_current_and_adjacent_lanes(y, side):
    planner = make_planner(x=9.8, speed=2., seam=9.9)
    planner.ego.y = y
    geometry = planner.plan(None, semantic_enabled=False)["planner_diagnostics"]["geometry"]
    assert geometry["reference_lane_key"] == "1:0:-4"
    assert geometry["current_side"] == side
    assert geometry["lane_match"]["method"] == "lane_id"
    assert geometry["reference_projection"]["nearest_map_identity_accepted"]


def test_real_town10_exit_drift_is_off_reference_instead_of_an_invented_left_lane(road_snapshot):
    # This physical tail pose really is 1.774 m from a 3.5-metre lane centre.
    # Rejecting the crossing-road identity must not hide that actual drift.
    failure = json.loads((Path(__file__).parent / "fixtures" / "town10hd_exit_drift.json").read_text(encoding="utf-8"))
    road, original = road_snapshot
    snapshot = deepcopy(original)
    snapshot.update({key: failure[key] for key in ("pose", "route_progress", "vehicle")})
    planner, _ = make_snapshot_planner((road, snapshot))
    with pytest.raises(RoadGeometryError, match="Ego has left the reference route") as caught:
        planner.plan(None, semantic_enabled=False, sim_time_s=failure["sim_time_s"])
    geometry = caught.value.diagnostics["geometry"]
    assert geometry["ego_lane_key"] == "934:0:-1"
    assert geometry["reference_lane_key"] == "939:0:-1"
    assert geometry["reference_projection"]["projected_lane_key"] == "934:0:-2"
    assert not geometry["reference_projection"]["nearest_map_identity_accepted"]
    assert geometry["current_side"] is None
    assert geometry["lane_match"] == {"method": "junction_geometry_rejected", "matches": {}}
