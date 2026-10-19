"""Every numerical lane change uses the same directional map permission."""
from importlib.util import find_spec
from types import SimpleNamespace as NS

import carla
import pytest


@pytest.mark.parametrize("kind,marking_permission,side,allowed", [
    ("Solid", carla.LaneChange.Both, "left", False),
    ("SolidSolid", carla.LaneChange.Both, "left", False),
    ("Curb", carla.LaneChange.Both, "left", False),
    ("Grass", carla.LaneChange.Both, "left", False),
    ("Broken", carla.LaneChange.Both, "left", True),
    ("BrokenSolid", carla.LaneChange.Left, "left", True),
    ("BrokenSolid", carla.LaneChange.Right, "left", False),
    ("SolidBroken", carla.LaneChange.Right, "right", True),
    ("SolidBroken", carla.LaneChange.Left, "right", False),
])
def test_solid_marking_and_directional_permission_override_a_broad_lane_flag(
        kind, marking_permission, side, allowed):
    from carla_bridge.planning.lane_maneuver import lane_change_permitted
    source = NS(lane_type=carla.LaneType.Driving, lane_change=carla.LaneChange.Both,
                is_junction=False, transform=carla.Transform())
    target = NS(lane_type=carla.LaneType.Driving, is_junction=False, transform=carla.Transform())
    setattr(source, side + "_lane_marking", NS(type=kind, lane_change=marking_permission))
    assert lane_change_permitted(source, target, side) is allowed


optional_pdm = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                 reason="optional pinned PDM dependencies are not installed")


@optional_pdm
def test_generic_native_lane_changes_cannot_cross_a_solid_line_with_a_broad_flag(monkeypatch):
    from carla_bridge.tests.test_pdm_planner import make_planner, Waypoint
    original_init = Waypoint.__init__

    def marked(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.left_lane_marking = NS(type="Solid", lane_change=carla.LaneChange.Both)
        self.right_lane_marking = NS(type="Broken", lane_change=carla.LaneChange.Both)

    monkeypatch.setattr(Waypoint, "__init__", marked)
    planner = make_planner(speed=3.)
    diagnostic = planner.plan(None, semantic_enabled=False)["planner_diagnostics"]
    assert diagnostic["excluded_lateral_paths"]["left"] == "lane_change_not_permitted_or_intersection"
    assert not any(item["physical_lane_side"] == "left" for item in diagnostic["candidates"])
    assert any(item["physical_lane_side"] == "right" for item in diagnostic["candidates"])
    assert diagnostic["geometry"]["lateral_paths"]["left"]["permission_failure"][
        "source_lane_marking_type"] == "Solid"
    assert diagnostic["safety_gate_applied"] is False


@optional_pdm
def test_legal_broken_line_keeps_native_changing_candidates_in_the_pool(monkeypatch):
    from carla_bridge.tests.test_pdm_planner import make_planner, Waypoint
    original_init = Waypoint.__init__

    def marked(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.left_lane_marking = NS(type="BrokenSolid", lane_change=carla.LaneChange.Left)
        self.right_lane_marking = NS(type="SolidBroken", lane_change=carla.LaneChange.Right)

    monkeypatch.setattr(Waypoint, "__init__", marked)
    planner = make_planner(speed=3.)
    diagnostic = planner.plan(None, semantic_enabled=False)["planner_diagnostics"]
    assert {item["physical_lane_side"] for item in diagnostic["candidates"]} == {"center", "left", "right"}
