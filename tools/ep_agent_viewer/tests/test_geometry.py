"""Offline regression checks; no simulator, model or experiment writes."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from epviz.geometry import (box_corners, build_potential, enrich_frame,
                            project_segment, project_world_point, _project_segment_progress)


WORKSPACE = Path(__file__).resolve().parents[2]
ENGINE = WORKSPACE / ".worktrees" / "EP-Driver"


def actor(identity, x=0., y=0., **values):
    return dict(id=identity, kind="vehicle", position=[x, y], velocity=[0., 0.],
                acceleration=[0., 0.], heading=0., length=4., width=2.,
                lane_id="center", visible=True, **values)


def camera(**values):
    return dict(view="front", frame_id="episode:1", time=0., frame=100,
                aligned=True, calibrated=True, width=800, height=600, fov_deg=90.,
                world_to_camera_ue=np.eye(4).tolist(), **values)


def frame():
    return dict(frame_id="episode:1", time=0., ego=actor("ego"),
                actors=[actor("lead", 10.)], plan={"states": [[0., 0., 0., 4.], [10., 0., 0., 4.]],
                    "diagnostics": {"planning_object_selection": {"selected_ids": ["lead"]},
                                    "potential_field": {"version": "lane-frenet-center-ellipse-1.0"}}},
                cameras=[], model_output=None, map={}, scene={})


def lane_frame(curved=False):
    result = frame()
    xs = np.linspace(-60, 100, 81)
    if curved:
        angles = np.linspace(-math.pi / 3, math.pi / 3, 81)
        center = np.column_stack([30 * np.sin(angles), 30 * (1 - np.cos(angles))])
        left = np.column_stack([26 * np.sin(angles), 30 - 26 * np.cos(angles)])
        right = np.column_stack([34 * np.sin(angles), 30 - 34 * np.cos(angles)])
        boundaries = [np.column_stack([r * np.sin(angles), 30 - r * np.cos(angles)]) for r in (28, 32)]
    else:
        center = np.column_stack([xs, np.zeros(len(xs))])
        left = center + [0., 4.]
        right = center - [0., 4.]
        boundaries = [center + [0., 2.], center - [0., 2.]]
    edge_left = dict(id="left_edge", points=boundaries[0].tolist(), crossable=True)
    edge_right = dict(id="right_edge", points=boundaries[1].tolist(), crossable=False)
    def lane(identity, points, left_id, right_id, edges):
        polygon = LineString(points).buffer(2., cap_style=2)
        return dict(id=identity, center=points.tolist(), polygon_wkt=polygon.wkt,
                    width=4., speed_limit=15., left=left_id, right=right_id,
                    successors=[], boundaries=edges)
    lanes = [lane("center", center, "left", "right", [edge_left, edge_right]),
             lane("left", left, None, "center", [edge_left]),
             lane("right", right, "center", None, [edge_right])]
    drivable = unary_union([LineString(points).buffer(2., cap_style=2) for points in (center, left, right)])
    result["scene"] = dict(ego=result["ego"], actors=result["actors"], lanes=lanes,
        route=center.tolist(), route_lane_ids=["center"], drivable_wkt=drivable.wkt,
        diagnostics={"lane_map_complete": True, "object_scope": {"settings": {"front_m": 40., "rear_m": 20.}}})
    return result


class CameraProjectionTests(unittest.TestCase):
    def test_left_positive_world_is_left_in_image(self):
        self.assertTrue(np.allclose(project_world_point([10, 0, 0], camera()), [400, 300]))
        self.assertTrue(np.allclose(project_world_point([10, 2, 0], camera()), [320, 300]))
        self.assertTrue(np.allclose(project_world_point([10, 0, 2], camera()), [400, 220]))

    def test_depth_clip_rejects_behind_camera_and_keeps_crossing(self):
        self.assertIsNone(project_world_point([-1, 0, 0], camera()))
        self.assertIsNone(project_segment([-2, 0, 0], [-1, 0, 0], camera()))
        segment = project_segment([-1, 0, 0], [10, -2, 0], camera())
        self.assertIsNotNone(segment)
        self.assertTrue(np.isfinite(segment).all())
        self.assertTrue(all(0 <= x <= 800 and 0 <= y <= 600 for x, y in segment))

    def test_viewport_clip_keeps_edge_with_both_endpoints_outside(self):
        segment = project_segment([10, 100, 0], [10, -100, 0], camera())
        self.assertTrue(np.allclose(segment, [[0, 300], [800, 300]]))

    def test_extrinsic_translation_uses_unreal_basis(self):
        calibrated = camera()
        calibrated["world_to_camera_ue"][0][3] = -5.
        self.assertTrue(np.allclose(project_world_point([15, 2, 0], calibrated), [320, 300]))

    def test_clipped_colour_progress_uses_world_not_screen_fraction(self):
        pixels, fractions = _project_segment_progress([2., 6., 0.], [10., -6., 0.], camera())
        self.assertTrue(np.allclose(pixels, [[0., 300.], [640., 300.]]))
        self.assertTrue(np.allclose(fractions, [.2, 1.]))
        # Near-plane clipping also keeps the original segment parameter.
        _, fractions = _project_segment_progress([-1., 0., 0.], [10., 0., 0.], camera())
        self.assertTrue(np.allclose(fractions, [.1, 1.]))

    def test_visible_plan_tail_retains_source_indices_and_endpoint_colours(self):
        value = frame()
        value['plan']['states'] = [[x, 0., 0., 4.] for x in (-10., -5., 10., 20.)]
        value['cameras'] = [camera()]
        value['camera_state'] = dict(frame=100, time=0., ego=actor('ego', z=1., height_m=2.))
        overlay = enrich_frame(value)['camera_overlays']['front']
        self.assertEqual(overlay['trajectory_source_point_count'], 4)
        self.assertEqual(len(overlay['trajectory_segments']), 2)
        self.assertGreater(overlay['trajectory_segments_progress'][0][0], 1. / 3.)
        self.assertEqual(overlay['trajectory_segments_progress'][-1][-1], 1.)
        self.assertEqual([p['index'] for p in overlay['trajectory_samples']], [2, 3])
        self.assertTrue(np.allclose([p['progress'] for p in overlay['trajectory_samples']], [2. / 3., 1.]))

    def test_single_plan_point_keeps_green_start_progress(self):
        value = frame()
        value['plan']['states'] = [[10., 0., 0., 4.]]
        value['cameras'] = [camera()]
        value['camera_state'] = dict(frame=100, time=0., ego=actor('ego', z=1., height_m=2.))
        overlay = enrich_frame(value)['camera_overlays']['front']
        self.assertEqual(overlay['trajectory_source_point_count'], 1)
        self.assertEqual(overlay['trajectory_segments'], [])
        self.assertEqual(overlay['trajectory_samples'], [{'point': [400., 300.], 'index': 0, 'progress': 0.}])

    def test_bad_calibration_fails_explicitly(self):
        with self.assertRaises(ValueError):
            project_world_point([1, 0, 0], {"width": 800, "height": 600})

    def test_only_recorded_3d_boxes_are_projected(self):
        value = frame()
        value["cameras"] = [camera()]
        value["camera_state"] = {"frame": 100, "time": 0., "ego": actor("ego", z=1., height_m=2.),
                                 "actors": [actor("lead", 10., z=1., height_m=2.)]}
        enriched = enrich_frame(value)
        obj = enriched["visualization"]["objects"][0]
        self.assertEqual(obj["geometry_quality"], "recorded_3d")
        self.assertEqual(len(enriched["camera_overlays"]["front"]["boxes"]), 1)
        value.pop("camera_state")
        enriched = enrich_frame(value)
        self.assertEqual(enriched["visualization"]["objects"][0]["geometry_quality"], "illustrative_2_5d")
        self.assertEqual(enriched["camera_overlays"]["front"]["boxes"], [])

    def test_mismatched_source_frame_disables_overlay(self):
        value = frame()
        value["cameras"] = [camera()]
        value["camera_state"] = {"frame": 101, "time": 0., "actors": [actor("lead", 10., z=1., height_m=2.)]}
        enriched = enrich_frame(value)
        self.assertFalse(enriched["camera_overlays"]["front"]["valid"])
        self.assertIn("帧号", enriched["camera_overlays"]["front"]["reason"])

    def test_mismatched_time_never_borrows_an_image(self):
        value = frame()
        value["cameras"] = [camera()]
        value["cameras"][0]["time"] = .1
        enriched = enrich_frame(value)
        self.assertFalse(enriched["camera_overlays"]["front"]["valid"])
        self.assertEqual(enriched["camera_overlays"]["front"]["boxes"], [])


class GeometryTests(unittest.TestCase):
    def test_enrich_does_not_mutate_source(self):
        value = frame()
        before = deepcopy(value)
        enrich_frame(value)
        self.assertEqual(value, before)

    def test_eight_role_labels_follow_recorded_ids_not_all_planning_sources(self):
        value = frame()
        value['actors'].append(actor('attention_extra', 15., 3.))
        selection = value['plan']['diagnostics']['planning_object_selection']
        selection.update(selected_ids=['lead', 'attention_extra'], region_objects={'front': 'lead'})
        enriched = enrich_frame(value)
        objects = {obj['id']: obj for obj in enriched['visualization']['objects']}
        self.assertEqual(objects['lead']['grid_roles'], ['front'])
        self.assertTrue(objects['lead']['grid_matched'])
        self.assertFalse(objects['attention_extra']['grid_matched'])
        self.assertTrue(objects['attention_extra']['selected'])
        self.assertNotIn('grid_roles', value['actors'][0])

    def test_camera_keeps_same_recorded_eight_match_tags(self):
        value = frame()
        value['plan']['diagnostics']['planning_object_selection']['region_objects'] = {'front': 'lead'}
        value['cameras'] = [camera()]
        value['camera_state'] = dict(frame=100, time=0., ego=actor('ego', z=1., height_m=2.),
                                    actors=[actor('lead', 10., z=1., height_m=2.)])
        overlay = enrich_frame(value)['camera_overlays']['front']
        self.assertEqual(overlay['boxes'][0]['grid_roles'], ['front'])
        self.assertTrue(overlay['boxes'][0]['grid_matched'])

    def test_box_rotates_without_changing_dimensions(self):
        points = box_corners([1, 2, 3], 4., 2., 2., math.pi / 2)
        self.assertTrue(np.allclose(np.ptp(points, axis=0), [2, 4, 2]))
        self.assertTrue(np.allclose(points.mean(axis=0), [1, 2, 3]))

    def test_corridor_is_width_based_ribbon_and_body_sweep_is_separate(self):
        result = enrich_frame(frame())["visualization"]
        vertices = np.asarray(result["corridor"][0])
        self.assertAlmostEqual(vertices[:, 1].min(), -1.)
        self.assertAlmostEqual(vertices[:, 1].max(), 1.)
        self.assertAlmostEqual(vertices[:, 0].min(), 0.)
        self.assertAlmostEqual(vertices[:, 0].max(), 10.)
        swept = np.asarray(result["body_sweep"]["polygons"][0])
        self.assertAlmostEqual(swept[:, 0].min(), -2.)
        self.assertAlmostEqual(swept[:, 0].max(), 12.)
        self.assertEqual(result["corridor_ribbon"]["display_margin_m"], 0.)
        self.assertIn("不代表", result["corridor_note"])

    def test_curved_ribbon_is_valid_continuous_and_clipped_to_road(self):
        value = lane_frame(curved=True)
        angle = np.linspace(0., math.pi / 4, 40)
        xy = np.column_stack([30 * np.sin(angle), 30 * (1 - np.cos(angle))])
        value["plan"]["states"] = np.column_stack([xy, angle, np.full(len(xy), 4.)]).tolist()
        result = enrich_frame(value)["visualization"]
        ribbon = result["corridor_ribbon"]
        self.assertTrue(ribbon["valid"])
        self.assertTrue(ribbon["trajectory_simple"])
        self.assertTrue(LineString(np.asarray(ribbon["left"])[:, :2]).is_simple)
        self.assertTrue(LineString(np.asarray(ribbon["right"])[:, :2]).is_simple)
        self.assertTrue(ribbon["clipped_to_recorded_road"])
        for part in ribbon["polygons"]:
            self.assertTrue(Polygon(np.asarray(part["outer"])[:, :2], [np.asarray(x)[:, :2] for x in part["holes"]]).is_valid)

    def test_road_union_preserves_island_holes(self):
        value = frame()
        road = Point(0, 0).buffer(20).difference(Point(0, 0).buffer(10))
        value["map"] = {"drivable": road.wkt}
        surface = enrich_frame(value)["visualization"]["render_geometry"]["road_surface"]
        self.assertEqual(len(surface["polygons"]), 1)
        self.assertEqual(len(surface["polygons"][0]["holes"]), 1)
        self.assertAlmostEqual(surface["recorded_area_m2"], road.area)

    def test_road_markings_deduplicate_reversed_edges_and_ignore_unmarked(self):
        value = lane_frame()
        line = {"id": "paint", "points": [[-10, 0], [10, 0]], "line_type": 1, "crossable": True}
        reverse = {**line, "id": "paint_reverse", "points": [[10, 0], [-10, 0]]}
        absent = {**line, "id": "no_paint", "points": [[-10, 1], [10, 1]], "line_type": 0}
        for lane in value["scene"]["lanes"]:
            lane["boundaries"] = []
        value["scene"]["lanes"][0]["boundaries"] = [line, reverse, absent]
        render = enrich_frame(value)["visualization"]["render_geometry"]
        self.assertEqual(len(render["markings"]), 1)
        self.assertEqual(render["markings"][0]["kind"], "broken")
        self.assertEqual(render["marking_diagnostics"]["ignored_unmarked_or_virtual"], 1)

    def test_small_connection_requires_topology_and_aligned_endpoints(self):
        value = frame()
        def lane(identity, start, end, successors):
            points = [start, end]
            return {"id": identity, "center": points, "width": 2., "successors": successors,
                    "polygon": list(LineString(points).buffer(1., cap_style=2).exterior.coords)}
        a = lane("a", [0, 0], [10, 0], ["b"])
        b = lane("b", [10.2, 0], [20, 0], [])
        value["map"] = {"lanes": [a, b]}
        render = enrich_frame(value)["visualization"]["render_geometry"]
        self.assertEqual(len(render["connections"]), 1)
        a["successors"] = []
        self.assertEqual(enrich_frame(value)["visualization"]["render_geometry"]["connections"], [])
        a["successors"] = ["b"]
        b["center"] = [[10.2, 0], [10.2, 10]]
        self.assertEqual(enrich_frame(value)["visualization"]["render_geometry"]["connections"], [])

    @unittest.skipUnless((ENGINE / "src/nine_grid/contracts.py").exists(), "matching local engine absent")
    def test_regions_follow_permitted_lanes_and_arc_length(self):
        result = enrich_frame(lane_frame(), ENGINE)["visualization"]
        regions = {row["name"]: row for row in result["regions"]}
        self.assertEqual(set(regions), {"front", "rear", "left_front", "left_side", "left_rear"})
        front = np.asarray(regions["front"]["polygons"][0])
        self.assertAlmostEqual(front[:, 0].max(), 40.)
        rear = np.asarray(regions["rear"]["polygons"][0])
        self.assertAlmostEqual(rear[:, 0].min(), -20.)
        side = np.asarray(regions["left_side"]["polygons"][0])
        self.assertAlmostEqual(side[:, 0].min(), -2.)
        self.assertAlmostEqual(side[:, 0].max(), 2.)
        self.assertFalse(result["region_status"]["roads"]["right"]["crossable"])

    @unittest.skipUnless((ENGINE / "src/nine_grid/contracts.py").exists(), "matching local engine absent")
    def test_curved_lane_regions_follow_road_instead_of_straight_rectangle(self):
        value = lane_frame(curved=True)
        result = enrich_frame(value, ENGINE)["visualization"]
        front = next(row for row in result["regions"] if row["name"] == "front")
        shape = unary_union([Polygon(ring) for ring in front["polygons"]])
        self.assertTrue(shape.covers(Point(20, 30 - math.sqrt(30**2 - 20**2))))
        self.assertFalse(shape.covers(Point(20, 0)))

    def test_missing_engine_reports_regions_unavailable(self):
        result = enrich_frame(frame())["visualization"]
        self.assertEqual(result["regions"], [])
        self.assertFalse(result["region_status"]["available"])

    def test_wkt_lanes_are_normalized_without_requiring_engine(self):
        value = lane_frame()
        result = enrich_frame(value)["visualization"]
        self.assertEqual(len(result["roads"]), 3)
        self.assertTrue(all(len(point) == 3 for road in result["roads"] for ring in road["polygons"] for point in ring))
        self.assertEqual(value["scene"]["lanes"][0].get("polygon"), None)

    def test_old_potential_evidence_is_not_reinterpreted(self):
        value = frame()
        value["plan"]["diagnostics"]["potential_field"] = {"version": "old_cartesian_rectangular"}
        result = build_potential(value, ENGINE)
        self.assertFalse(result["available"])
        self.assertIn("旧版", result["reason"])

    @unittest.skipUnless((ENGINE / "src/nine_grid/contracts.py").exists(), "matching local engine absent")
    def test_potential_is_core_formula_with_domain_mask_and_current_sources(self):
        from epviz.geometry import _engine, _scene
        value = lane_frame()
        core = _engine(ENGINE)
        domain, evidence = core["potential_domain"].build_lane_potential_domain(_scene(value, core))
        value["plan"]["diagnostics"]["potential_field"] = {
            "version": "lane-frenet-center-ellipse-1.0", "geometry": {"version": "lane-frenet-centre-potential-1.0"},
            "domain": evidence, "classes": {"vehicle": {"longitudinal_m": 20., "lateral_m": 6., "peak": 8.}},
            "attention_increment": 1., "outer_risk_weight": 2.}
        value["plan"]["diagnostics"]["planning_object_selection"]["selected_state"] = {
            "lead": {"memberships": {"front": {"delta_s_m": 10.}}}}
        result = build_potential(value, ENGINE, resolution=36)
        self.assertTrue(result["available"], result.get("reason"))
        self.assertEqual(result["sources"][0]["id"], "lead")
        self.assertTrue(any(x is None for row in result["values"] for x in row))
        self.assertGreater(result["maximum"], 5.)
        self.assertAlmostEqual(max(x for row in result["weighted_values"] for x in row if x is not None), 2 * result["maximum"])
        self.assertEqual(result["goal"], None)
        # An attention source doubles its peak with configured increment one.
        value["model_output"] = {"frame_id": value["frame_id"], "attention": [{"id": "lead", "rank": 1}]}
        attention = build_potential(value, ENGINE, resolution=36)
        self.assertAlmostEqual(attention["maximum"], 2 * result["maximum"])
        self.assertEqual(attention["sources"][0]["attention_gain"], 2.)
        self.assertGreaterEqual(result["display_window"]["front_m"], 70.)
        self.assertGreaterEqual(result["display_window"]["rear_m"], 30.)
        value["plan"]["diagnostics"]["potential_field"].pop("outer_risk_weight")
        self.assertFalse(build_potential(value, ENGINE)["available"])

    @unittest.skipUnless((ENGINE / "src/nine_grid/contracts.py").exists(), "matching local engine absent")
    def test_goal_field_has_negative_valley_and_disappears_when_retired(self):
        from epviz.geometry import _engine, _scene
        value = lane_frame()
        core = _engine(ENGINE)
        _, evidence = core["potential_domain"].build_lane_potential_domain(_scene(value, core))
        diag = value["plan"]["diagnostics"]
        diag["planning_object_selection"] = {"selected_ids": []}
        diag["potential_field"] = {"version": "lane-frenet-center-ellipse-1.0",
            "geometry": {"version": "lane-frenet-centre-potential-1.0"}, "domain": evidence,
            "classes": {}, "attention_increment": 1., "outer_risk_weight": 2.}
        # Route begins at x=-60; goal x=20 therefore has polyline s=80.
        diag["goal_state"] = {"active_goal": {"world_goal": [20., 0.], "last_s": 60.,
            "s_goal_m": 80., "goal_reference": {"points": value["scene"]["route"]}}}
        value["configuration"] = {"parameters": {"local_goal": {"attraction_weight": 3.,
            "longitudinal_scale_m": 30., "lateral_scale_m": 15., "potential_shift": 6.}}}
        result = build_potential(value, ENGINE, resolution=40)
        self.assertTrue(result["available"], result.get("reason"))
        self.assertLess(result["minimum"], -5.9)
        diag["goal_state"] = {"active_goal": None}
        diag["active_goal"] = False
        retired = build_potential(value, ENGINE, resolution=40)
        self.assertEqual(retired["minimum"], 0.)

    @unittest.skipUnless((ENGINE / "src/nine_grid/contracts.py").exists(), "matching local engine absent")
    def test_raster_matches_analytic_raw_and_weighted_energies(self):
        """Check independently evaluated energies, not just internal ratios."""
        from epviz.geometry import _engine, _scene
        value = lane_frame()
        core = _engine(ENGINE)
        _, evidence = core["potential_domain"].build_lane_potential_domain(_scene(value, core))
        diag = value["plan"]["diagnostics"]
        diag["planning_object_selection"]["selected_state"] = {
            "lead": {"memberships": {"front": {"delta_s_m": 10.}}}}
        diag["potential_field"] = {
            "version": "lane-frenet-center-ellipse-1.0", "geometry": {"version": "lane-frenet-centre-potential-1.0"},
            "domain": evidence, "classes": {"vehicle": {"longitudinal_m": 20., "lateral_m": 6., "peak": 8.}},
            "attention_increment": 1., "outer_risk_weight": 2.5}
        diag["goal_state"] = {"active_goal": {"world_goal": [20., 0.], "last_s": 60.,
            "s_goal_m": 80., "goal_reference": {"points": value["scene"]["route"]}}}
        value["configuration"] = {"parameters": {"local_goal": {"attraction_weight": 3.,
            "longitudinal_scale_m": 30., "lateral_scale_m": 15., "potential_shift": 6.}}}
        value["model_output"] = {"frame_id": value["frame_id"], "attention": [{"id": "lead", "rank": 2}]}
        result = build_potential(value, ENGINE, resolution=67)
        self.assertTrue(result["available"], result.get("reason"))
        xx, yy = np.meshgrid(result["xs"], result["ys"])
        rho = np.sqrt(((xx - 10.) / 20.)**2 + (yy / 6.)**2)
        t = np.maximum(1. - rho, 0.)
        repulsion = 12. * t**3 * (1. + 3. * rho + 6. * rho**2)
        attraction = 1.5 * (((xx - 20.) / 30.)**2 + (yy / 15.)**2) - 6.
        mask = np.asarray(result["display_mask"], bool)
        matrices = {name: np.asarray(result[name], float) for name in
                    ("values", "weighted_values", "repulsion_values", "attraction_values")}
        self.assertTrue(np.allclose(matrices["values"][mask], (repulsion + attraction)[mask], atol=1e-10))
        self.assertTrue(np.allclose(matrices["weighted_values"][mask], (2.5 * repulsion + attraction)[mask], atol=1e-10))
        self.assertTrue(np.allclose(matrices["repulsion_values"][mask], repulsion[mask], atol=1e-10))
        self.assertTrue(np.allclose(matrices["attraction_values"][mask], attraction[mask], atol=1e-10))
        self.assertTrue(np.all(np.isnan(matrices["values"][~mask])))
        outside_support = mask & (rho >= 1.)
        self.assertTrue(outside_support.any())
        self.assertTrue(np.all(matrices["repulsion_values"][outside_support] == 0.))
        self.assertTrue(np.any(~np.asarray(result["domain_mask"], bool)))

    @unittest.skipUnless((ENGINE / "src/nine_grid/contracts.py").exists(), "matching local engine absent")
    def test_unselected_objects_and_nonsemantic_goal_do_not_create_fields(self):
        from epviz.geometry import _engine, _scene
        value = lane_frame()
        core = _engine(ENGINE)
        _, evidence = core["potential_domain"].build_lane_potential_domain(_scene(value, core))
        diag = value["plan"]["diagnostics"]
        diag["planning_object_selection"] = {"selected_ids": []}
        diag["semantic_enabled"] = False
        diag["potential_field"] = {
            "version": "lane-frenet-center-ellipse-1.0", "geometry": {"version": "lane-frenet-centre-potential-1.0"},
            "domain": evidence, "classes": {"vehicle": {"longitudinal_m": 20., "lateral_m": 6., "peak": 8.}},
            "attention_increment": 1., "outer_risk_weight": 2.}
        value["model_output"] = {"frame_id": value["frame_id"], "world_goal": [20., 0.],
                                 "attention": [{"id": "lead", "rank": 1}]}
        result = build_potential(value, ENGINE, resolution=36)
        self.assertTrue(result["available"], result.get("reason"))
        self.assertEqual(result["sources"], [])
        self.assertIsNone(result["goal"])
        self.assertTrue(result["empty_source_field"])
        self.assertEqual(result["minimum"], 0.)
        self.assertEqual(result["maximum"], 0.)

    @unittest.skipUnless((ENGINE / "src/nine_grid/contracts.py").exists(), "matching local engine absent")
    def test_curved_raster_matches_planner_ego_coordinate_kernel(self):
        """The viewer uses world coordinates; the planner uses a rigid ego frame."""
        from epviz.geometry import _engine, _scene
        value = lane_frame(curved=True)
        core = _engine(ENGINE)
        scene = _scene(value, core)
        domain, evidence = core["potential_domain"].build_lane_potential_domain(scene)
        settings = {"longitudinal_m": 20., "lateral_m": 6., "peak": 8.}
        diag = value["plan"]["diagnostics"]
        diag["planning_object_selection"]["selected_state"] = {
            "lead": {"memberships": {"front": {"delta_s_m": 10.}}}}
        diag["potential_field"] = {
            "version": "lane-frenet-center-ellipse-1.0", "geometry": {"version": "lane-frenet-centre-potential-1.0"},
            "domain": evidence, "classes": {"vehicle": settings},
            "attention_increment": 1., "outer_risk_weight": 2.}
        result = build_potential(value, ENGINE, resolution=42)
        self.assertTrue(result["available"], result.get("reason"))
        xx, yy = np.meshgrid(result["xs"], result["ys"])
        points = np.column_stack([xx.ravel(), yy.ravel()])
        origin, heading = np.asarray([3., -7.]), .71
        c, s = math.cos(heading), math.sin(heading)
        rotation = np.asarray([[c, -s], [s, c]])
        local = lambda xy: (np.asarray(xy) - origin) @ rotation
        geometry = core["lane_potential"].LanePotentialGeometry(local(scene.route), domain,
                                                                      domain_pose=(origin, heading))
        old_s = core["lane_roles"].project(scene.route, scene.ego.position)[0]
        anchor = float(geometry.chart_arc_from_polyline_s(old_s))
        source_anchor = float(np.clip(anchor + 10., 0., geometry.length))
        source = geometry.project_sequence(local([scene.actors[0].position]), source_anchor,
                    require_domain=False, initial_window_m=10.)
        query = geometry.project(local(points), anchors=anchor)
        expected = geometry.repulsion_from_projection(query, source, settings)["energy"]
        mask = np.asarray(result["display_mask"], bool).ravel()
        actual = np.asarray(result["values"], float).ravel()
        self.assertTrue(mask.any())
        self.assertGreater(float(np.nanmax(actual)), 0.)
        self.assertTrue(np.allclose(actual[mask], expected[mask], atol=1e-9))

    def test_attention_follows_explicit_plan_source_binding(self):
        value = frame()
        value["model_source_frame_id"] = "episode:0"
        value["active_model_output"] = {"frame_id": "episode:0", "answer": {"attention": [{"id": "lead", "rank": 2}]}}
        obj = enrich_frame(value)["visualization"]["objects"][0]
        self.assertTrue(obj["attention"])
        self.assertEqual(obj["attention_rank"], 2)
        value["plan"]["diagnostics"]["semantic_enabled"] = False
        obj = enrich_frame(value)["visualization"]["objects"][0]
        self.assertFalse(obj["attention"])

    def test_wrong_bound_model_source_does_not_highlight_attention(self):
        value = frame()
        value["model_source_frame_id"] = "episode:0"
        value["active_model_output"] = {"frame_id": "another_episode:0", "attention": [{"id": "lead"}]}
        self.assertFalse(enrich_frame(value)["visualization"]["objects"][0]["attention"])

    def test_active_goal_comes_from_goal_manager_and_retired_goal_is_not_reused(self):
        value = frame()
        value["plan"]["diagnostics"]["goal_state"] = {"active_goal": {"world_goal": [12., 3.], "source_frame_id": "episode:0"}}
        self.assertEqual(enrich_frame(value)["visualization"]["goal"], [12., 3., 0.])
        value["plan"]["diagnostics"]["goal_state"] = {"active_goal": None}
        value["plan"]["diagnostics"]["active_goal"] = False
        value["model_output"] = {"frame_id": value["frame_id"], "world_goal": [12., 3.]}
        self.assertIsNone(enrich_frame(value)["visualization"]["goal"])


if __name__ == "__main__":
    unittest.main()
