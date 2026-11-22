"""Offline snapshot adaptation: archived evidence must not become planning."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from shapely.geometry import Polygon

from epviz.library_scene import build_library_scene


def actor(identity, x, y=0., **extra):
    return dict(id=identity, kind="vehicle", position=[x, y], heading=0.,
                velocity=[4., 0.], acceleration=[0., 0.], length=4., width=2., **extra)


def scene():
    road = Polygon([[-20., -6.], [80., -6.], [80., 6.], [-20., 6.]],
                   holes=[[[30., 2.], [34., 2.], [34., 4.], [30., 4.]]])
    return dict(frame_id="recorded:12", time=12.5, scenario="openscene", coordinates="world_xy_m",
        ego=actor("ego", 0.), actors=[actor("lead", 10.), actor("other", 25., -3.)],
        lanes=[dict(id="lane-0", center=[[-20., 0.], [80., 0.]], polygon=road.wkt,
            width=4., successors=[], boundaries=[dict(id="solid-0", points=[[-20., -2.], [80., -2.]],
                line_type="solid", crossable=False), dict(id="unknown-0", points=[[-20., 2.], [80., 2.]],
                line_type=0, crossable=None)])],
        drivable=road.wkt, route=[[-20., 0.], [80., 0.]], route_lane_ids=["lane-0"],
        rules={}, assets={}, diagnostics=dict(lane_map_complete=True))


def grid():
    slots = [[dict(role=role, state="empty", id=None, kind=None) for role in row] for row in
        (("left_front", "front", "right_front"), ("left_side", "ego", "right_side"),
         ("left_rear", "rear", "right_rear"))]
    slots[0][1].update(state="occupied", id="lead", kind="vehicle")
    slots[1][1].update(state="occupied", id="ego", kind="vehicle")
    return dict(available=True, slots=slots)


class LibrarySceneTests(unittest.TestCase):
    def test_archived_road_holes_markings_and_object_footprints(self):
        result = build_library_scene(scene(), grid())
        self.assertTrue(result["available"])
        vis = result["visualization"]
        surface = vis["render_geometry"]["road_surface"]
        self.assertTrue(surface["holes_preserved"])
        self.assertTrue(surface["recorded_surface_only"])
        self.assertEqual(len(surface["polygons"][0]["holes"]), 1)
        self.assertEqual(len(vis["render_geometry"]["markings"]), 1)
        self.assertEqual(len(vis["objects"]), 2)
        self.assertEqual(vis["objects"][0]["center"][:2], [10., 0.])
        self.assertEqual(len(vis["objects"][0]["corners"]), 8)
        self.assertEqual(vis["objects"][0]["geometry_quality"], "illustrative_2_5d")

    def test_grid_roles_are_archived_tags_not_apf_or_attention(self):
        result = build_library_scene(scene(), grid())
        vis = result["visualization"]
        lead, other = vis["objects"]
        self.assertEqual(lead["grid_roles"], ["front"])
        self.assertTrue(lead["grid_matched"])
        self.assertEqual(other["grid_roles"], [])
        for obj in vis["objects"]:
            self.assertFalse(obj["selected"])
            self.assertFalse(obj["potential_source"])
            self.assertFalse(obj["attention"])
        self.assertEqual(vis["regions"], [])
        self.assertFalse(vis["region_status"]["available"])
        self.assertEqual(len(vis["grid_matches"]), 9)

    def test_route_history_and_unrelated_diagnostics_are_not_plans(self):
        value = scene()
        value["ego"]["history"] = [dict(time=11., position=[-4., 0.]), dict(time=12., position=[-2., 0.])]
        value["diagnostics"].update(previous_control=[2., 0.], goal=[50., 0.], attention=["lead"],
                                    potential_field={"version": "lane-frenet-center-ellipse-1.0"})
        result = build_library_scene(value, grid())
        self.assertIsNone(result["plan"])
        self.assertEqual(result["scene"]["route"], value["route"])
        self.assertEqual(result["scene"]["ego"]["history"], value["ego"]["history"])
        self.assertEqual(result["visualization"]["trajectory"], [])
        self.assertEqual(result["visualization"]["corridor"], [])
        self.assertIsNone(result["visualization"]["goal"])
        self.assertFalse(result["potential"]["available"])
        self.assertFalse(result["potential"]["deferred"])
        self.assertEqual(result["cameras"], [])
        self.assertEqual(result["camera_overlays"], {})
        self.assertTrue(result["snapshot_only"])

    def test_missing_object_is_not_reconstructed_from_relative_grid_values(self):
        value = grid()
        value["slots"][2][1].update(state="occupied", id="absent", kind="vehicle", values=[-10., 0.] + [0.] * 5)
        result = build_library_scene(scene(), value)
        self.assertEqual(len(result["visualization"]["objects"]), 2)
        match = next(row for row in result["visualization"]["grid_matches"] if row["id"] == "absent")
        self.assertFalse(match["actor_available"])
        self.assertIsNone(match["position"])
        self.assertTrue(any("不在本证据scene" in warning for warning in result["warnings"]))

    def test_legacy_slot_dictionary_accepted_without_rerunning_matcher(self):
        value = {"slots": {"front": {"state": "occupied", "id": "other", "kind": "vehicle"}}}
        with patch("epviz.geometry._engine", side_effect=AssertionError("must not import driving engine")):
            result = build_library_scene(scene(), value, engine_root="unused-main-worktree")
        other = next(obj for obj in result["visualization"]["objects"] if obj["id"] == "other")
        self.assertEqual(other["grid_roles"], ["front"])

    def test_diagnostic_only_grid_does_not_create_empty_regions(self):
        result = build_library_scene(scene(), {"available": False, "reason": "invalid", "slots": []})
        self.assertEqual(result["visualization"]["grid_matches"], [])
        self.assertFalse(result["library_scene"]["capabilities"]["grid_object_matches"]["available"])

    def test_no_display_stitch_is_added_to_archived_road(self):
        value = scene()
        value["lanes"] = [dict(id="a", center=[[0., 0.], [10., 0.]],
                polygon=[[0., -2.], [10., -2.], [10., 2.], [0., 2.]], width=4., successors=["b"]),
            dict(id="b", center=[[10.3, 0.], [20., 0.]],
                polygon=[[10.3, -2.], [20., -2.], [20., 2.], [10.3, 2.]], width=4., successors=[])]
        value.pop("drivable")
        result = build_library_scene(value)
        render = result["visualization"]["render_geometry"]
        self.assertEqual(render["connections"], [])
        self.assertEqual(len(render["road_surface"]["polygons"]), 2)
        self.assertEqual(render["road_surface"]["recorded_area_m2"], render["road_surface"]["display_area_m2"])

    def test_input_is_unchanged_and_heights_are_not_inferred(self):
        value, match = scene(), grid()
        original_scene, original_grid = deepcopy(value), deepcopy(match)
        build_library_scene(value, match)
        self.assertEqual(value, original_scene)
        self.assertEqual(match, original_grid)
        self.assertNotIn("height", value["actors"][0])

    def test_recorded_true_height_is_used_only_when_explicit(self):
        value = scene()
        value["actors"][0].update(z=1., height=2.)
        result = build_library_scene(value)
        obj = result["visualization"]["objects"][0]
        self.assertEqual(obj["geometry_quality"], "recorded_3d")
        self.assertEqual(obj["height"], 2.)
        self.assertEqual(result["camera_overlays"], {})

    def test_missing_dimensions_do_not_create_default_physical_boxes(self):
        value = scene(); value["actors"][0].pop("width")
        result = build_library_scene(value)
        self.assertEqual(len(result["visualization"]["objects"]), 1)
        counts = result["library_scene"]["capabilities"]["current_objects"]
        self.assertEqual(counts["archived_count"], 2)
        self.assertEqual(counts["rendered_count"], 1)
        self.assertTrue(any("默认实体尺寸" in warning for warning in result["warnings"]))

    def test_missing_actor_list_is_not_claimed_as_empty_scene(self):
        value = scene(); value.pop("actors")
        result = build_library_scene(value)
        self.assertFalse(result["available"])
        self.assertIn("不能当作对象为空", result["reason"])

    def test_invalid_identity_time_and_ego_fail_explicitly(self):
        for value in [None, {}, {**scene(), "frame_id": None}, {**scene(), "time": None},
                      {**scene(), "ego": {"position": [float("nan"), 0.]}}]:
            with self.subTest(scene=value):
                result = build_library_scene(value)
                self.assertFalse(result["available"])
                self.assertTrue(result["reason"])


if __name__ == "__main__":
    unittest.main()
