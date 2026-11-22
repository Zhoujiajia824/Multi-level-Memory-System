"""Local archived road display and strictly retrospective execution geometry."""
from copy import deepcopy
import unittest

from shapely.geometry import Polygon

from epviz.library_roads import build_library_roads
from epviz.library_scene import build_library_scene


def snapshot(scenario="openscene"):
    return dict(frame_id="anchor", time=10., coordinates="world_xy_m", scenario=scenario,
        ego=dict(id="ego", kind="vehicle", position=[0., 0.], heading=0., width=2., length=4.),
        actors=[], lanes=[dict(id="a", center=[[0., 0.], [10., 0.]],
            polygon=[[0., -2.], [10., -2.], [10., 2.], [0., 2.]], width=4., successors=["b"],
            boundaries=[dict(id="edge-a", points=[[0., 2.], [10., 2.]],
                line_type=3, source="nuplan_map_boundary", permission_source="unknown_boundary_type_not_permission")]),
            dict(id="b", center=[[10., 0.], [1000., 0.]],
                polygon=[[10., -2.], [1000., -2.], [1000., 2.], [10., 2.]], width=4., successors=[])],
        route=[[0., 0.], [1000., 0.]], rules={}, diagnostics=dict(lane_map_complete=True,
            lane_source_metadata={"a": {"kind": "connector"}, "b": {"kind": "lane"}}))


def supervision(current=None):
    value = current or snapshot()
    return dict(available=True, sha256_verified=True, source="recorded_human_driving_not_optimality_proof",
        source_path="audit/supervision.json", anchor_frame_id="anchor", anchor_time=10.,
        evidence_frame_id=value["frame_id"], evidence_time=value["time"],
        ego_trajectory=[dict(time_s=10. + i * .5, position=[i * 2., 0.], heading=0.) for i in range(7)])


class LibraryRoadTests(unittest.TestCase):
    def test_unknown_connector_boundaries_never_become_painted_lines(self):
        scene = snapshot()
        render, _ = build_library_roads(scene)
        self.assertEqual(render["markings"], [])
        self.assertEqual(render["marking_diagnostics"]["ignored_by_reason"]["unknown_or_virtual_boundary"], 1)
        self.assertEqual(render["topology"]["edges"][0]["kind"], "connector")
        self.assertTrue(render["topology"]["edges"][0]["display_only"])

    def test_unverified_nuplan_integer_two_is_not_assumed_solid(self):
        scene = snapshot()
        edge = scene["lanes"][0]["boundaries"][0]
        edge.pop("permission_source"); edge["line_type"] = 2
        scene["diagnostics"]["lane_source_metadata"]["a"]["kind"] = "lane"
        render, _ = build_library_roads(scene)
        self.assertEqual(render["markings"], [])
        self.assertEqual(render["marking_diagnostics"]["ignored_by_reason"]["unverified_numeric_or_unknown_marking_type"], 1)

    def test_known_highway_numeric_types_and_nuscenes_dashed_are_preserved(self):
        scene = snapshot("highway")
        scene["lanes"][0]["boundaries"] = [dict(id="simulation-edge", line_type=3,
            crossable=False, points=[[0., 2.], [10., 2.]])]
        render, _ = build_library_roads(scene)
        self.assertEqual(render["markings"][0]["kind"], "solid")
        scene["scenario"] = "nuscenes"
        scene["lanes"][0]["boundaries"][0]["line_type"] = "DOUBLE_DASHED_WHITE"
        render, _ = build_library_roads(scene)
        self.assertEqual(render["markings"][0]["kind"], "broken")
        self.assertEqual(render["markings"][0]["color"], "#f0f3f2")

    def test_display_clips_actual_surface_without_changing_original_domain(self):
        scene = snapshot()
        render, original = build_library_roads(scene, half_extent_m=30.)
        self.assertEqual(original.bounds, (0., -2., 1000., 2.))
        self.assertAlmostEqual(render["road_surface"]["recorded_area_m2"], 4000.)
        self.assertAlmostEqual(render["road_surface"]["display_area_m2"], 120.)
        coords = render["road_surface"]["polygons"][0]["outer"]
        self.assertEqual(max(point[0] for point in coords), 30.)
        self.assertEqual(render["display_domain"]["source"], "display_window_only_not_planning_or_object_matching_scope")
        self.assertFalse(render["connection_policy"]["arbitrary_centerline_buffers"])

    def test_holes_survive_local_clipping(self):
        scene = snapshot()
        scene["drivable"] = Polygon([[-10., -10.], [50., -10.], [50., 10.], [-10., 10.]],
            holes=[[[4., 2.], [6., 2.], [6., 4.], [4., 4.]]]).wkt
        render, _ = build_library_roads(scene, half_extent_m=20.)
        self.assertEqual(len(render["road_surface"]["polygons"][0]["holes"]), 1)

    def test_topology_uses_recorded_paths_and_does_not_create_road_connections(self):
        scene = snapshot()
        render, _ = build_library_roads(scene)
        self.assertEqual(render["connections"], [])
        self.assertEqual(render["topology"]["connections"][0]["status"], "coincident_recorded_endpoints")
        self.assertEqual(render["topology"]["routes"][0]["kind"], "recorded_navigation_reference")
        self.assertEqual(len(render["topology"]["edges"][0]["path"][0]), 3)
        scene["lanes"][1]["center"][0] = [11., 0.]
        render, _ = build_library_roads(scene)
        self.assertEqual(render["topology"]["connections"], [])
        self.assertFalse(render["topology"]["unresolved_connections"][0]["drawn"])

    def test_missing_successor_is_not_joined_to_nearest_parallel_road(self):
        scene = snapshot()
        scene["lanes"][0]["successors"] = ["outside_local_map"]
        render, _ = build_library_roads(scene)
        self.assertEqual(render["topology"]["targets_outside_recorded_local_map"], 1)
        self.assertEqual(render["topology"]["connections"], [])

    def test_coarse_map_quality_is_explicit_without_replacing_geometry(self):
        scene = snapshot("nuscenes")
        scene["diagnostics"].update(lane_map_complete=False, lane_width_source="polygon_area_divided_by_centerline_length")
        scene["lanes"][0]["width"] = 22.
        render, _ = build_library_roads(scene)
        self.assertEqual(render["map_quality"]["wide_archived_lane_values"][0]["archived_width_m"], 22.)
        self.assertFalse(render["map_quality"]["lane_map_complete"])
        self.assertAlmostEqual(render["road_surface"]["recorded_area_m2"], 4000.)

    def test_inputs_are_unchanged(self):
        scene = snapshot(); before = deepcopy(scene)
        build_library_roads(scene)
        self.assertEqual(scene, before)


class RecordedExecutionTests(unittest.TestCase):
    def test_future_execution_is_shown_with_distinct_provenance_and_geometric_ribbon(self):
        scene = snapshot(); future = supervision()
        frame = build_library_scene(scene, future=future)
        vis = frame["visualization"]
        self.assertEqual(len(vis["trajectory"]), 7)
        self.assertEqual(vis["trajectory_segments"], [vis["trajectory"]])
        provenance = vis["trajectory_provenance"]
        self.assertEqual(provenance["kind"], "recorded_future_execution")
        self.assertEqual(provenance["actual_source"], future["source"])
        self.assertFalse(provenance["online_input"])
        self.assertIsNone(frame["plan"])
        self.assertTrue(vis["corridor_ribbon"]["available"])
        self.assertEqual(vis["corridor_ribbon"]["half_width_m"], 1.)
        self.assertIn("不是规划安全走廊", vis["corridor_ribbon"]["note"])
        self.assertFalse(frame["library_scene"]["capabilities"]["planned_trajectory"]["available"])

    def test_later_evidence_crops_verified_actual_samples(self):
        scene = snapshot(); scene.update(frame_id="evidence-1", time=11.)
        scene["ego"]["position"] = [4., 0.]
        frame = build_library_scene(scene, future=supervision(scene))
        self.assertEqual(len(frame["visualization"]["trajectory"]), 5)
        self.assertEqual(frame["visualization"]["trajectory"][0][:2], [4., 0.])
        provenance = frame["visualization"]["trajectory_provenance"]
        self.assertEqual(provenance["anchor_frame_id"], "evidence-1")
        self.assertEqual(provenance["record_anchor_frame_id"], "anchor")

    def test_last_evidence_has_no_fabricated_forward_extension(self):
        scene = snapshot(); scene.update(frame_id="evidence-3", time=13.)
        scene["ego"]["position"] = [12., 0.]
        frame = build_library_scene(scene, future=supervision(scene))
        self.assertEqual(frame["visualization"]["trajectory"], [])
        self.assertFalse(frame["visualization"]["corridor_ribbon"]["available"])
        self.assertFalse(frame["visualization"]["trajectory_provenance"]["available"])

    def test_future_must_match_current_frame_time_and_position(self):
        scene = snapshot()
        for change in [{"sha256_verified": False}, {"evidence_frame_id": "wrong"}, {"evidence_time": 10.2}]:
            with self.subTest(change=change):
                frame = build_library_scene(scene, future={**supervision(), **change})
                self.assertEqual(frame["visualization"]["trajectory"], [])
        future = supervision(); future["ego_trajectory"][0]["position"] = [2., 0.]
        frame = build_library_scene(scene, future=future)
        self.assertEqual(frame["visualization"]["trajectory"], [])

    def test_no_interpolation_to_unobserved_evidence_time(self):
        scene = snapshot(); scene.update(time=10.25); scene["ego"]["position"] = [1., 0.]
        frame = build_library_scene(scene, future=supervision(scene))
        self.assertEqual(frame["visualization"]["trajectory"], [])
        self.assertIn("未插值", frame["visualization"]["trajectory_provenance"]["reason"])

    def test_missing_sample_or_large_gap_is_not_bridged(self):
        for mode in ("missing", "gap"):
            future = supervision()
            if mode == "missing":
                future["ego_trajectory"][3] = None
            else:
                del future["ego_trajectory"][3]
            frame = build_library_scene(snapshot(), future=future)
            self.assertEqual(len(frame["visualization"]["trajectory"]), 3)
            self.assertIsNotNone(frame["visualization"]["trajectory_provenance"]["truncated_reason"])

    def test_unverified_future_cannot_turn_navigation_into_execution(self):
        frame = build_library_scene(snapshot(), future={"ego_trajectory": supervision()["ego_trajectory"]})
        self.assertEqual(frame["visualization"]["trajectory"], [])
        self.assertEqual(len(frame["visualization"]["topology"]["routes"]), 1)


if __name__ == "__main__":
    unittest.main()
