"""Native camera projection must stay bound to one authenticated evidence."""
from copy import deepcopy
import unittest

import numpy as np

from epviz.library_projection import (attach_library_cameras, camera_model,
    project_segment, quaternion_matrix, _camera_point, _pixel, _project_segment_progress)


def frame():
    return dict(available=True, frame_id="normalized:12", time=12.,
        actors=[dict(id="lead", position=[10., 0.])],
        visualization=dict(trajectory=[[0., 0., 0.], [20., 0., 0.]],
            trajectory_segments=[[[0., 0., 0.], [20., 0., 0.]]],
            trajectory_provenance=dict(kind="recorded_future_execution", anchor_frame_id="normalized:12", time=12.),
            corridor_ribbon=dict(source="recorded_future_execution_schematic_ribbon",
                polygons=[dict(outer=[[1., -1., 0.], [20., -1., 0.], [20., 1., 0.], [1., 1., 0.]], holes=[])])),
        library_scene=dict(capabilities={}))


def data(source="openscene"):
    view = "CAM_F0" if source == "openscene" else "CAM_FRONT"
    camera = dict(view=view, width=100, height=100, timestamp_s=12., sample_timestamp_s=12.,
        time_offset_s=0., exposure_timestamp_s=None, sha256="abc", source_filename="raw/view.jpg",
        calibration=dict(rotation=[.5, -.5, .5, -.5], translation=[0., 0., 1.5],
            camera_intrinsic=[[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]),
        ego_pose=dict(rotation=[1., 0., 0., 0.], translation=[0., 0., 0.]))
    context = dict(timestamp_s=12.,
        raw_ego_pose=dict(rotation=[1., 0., 0., 0.], translation=[0., 0., 0.]),
        visible_annotation_candidates=[dict(source_id="lead", time=12., position=[10., 0.],
            position_3d_for_camera=[10., 0., 1.], raw_local_box=[10., 0., 1., 4., 2., 2., 0.])])
    return dict(available=True, source=source, frame_id="normalized:12", time=12., evidence_index=0,
        integrity="verified_current_evidence_assets", manifest=[camera],
        exports=[dict(view=view, exported_filename="camera_00.jpg", source_filename="raw/view.jpg",
                      sha256="abc", timestamp_s=12.)],
        original_scene=dict(source=source, frame_id="raw-source-token", current_context=context),
        images=[dict(view=view, path="records/r/evidence/0/source/000_camera_00.jpg",
                     url="/api/library/asset?registered=1", sha256="abc", registered=True, sha256_verified=True)])


class LibraryProjectionTests(unittest.TestCase):
    def test_native_camera_to_ego_and_world_to_exposure_camera(self):
        model = camera_model(data()["manifest"][0])
        np.testing.assert_allclose(_camera_point([10., 0., 0.], model), [0., 1.5, 10.], atol=1e-12)
        np.testing.assert_allclose(_pixel(_camera_point([10., 0., 0.], model), model), [50., 65.])
        # Changing the image's own exposure pose must alter projection, without
        # borrowing the normalized ego body's center or current state's pose.
        camera = data()["manifest"][0]
        camera["ego_pose"]["translation"] = [1., 0., 0.]
        np.testing.assert_allclose(_camera_point([10., 0., 0.], camera_model(camera)), [0., 1.5, 9.])

    def test_openscene_measured_box_and_recorded_future_approximation(self):
        original_frame, original_data = frame(), data()
        before_frame, before_data = deepcopy(original_frame), deepcopy(original_data)
        result = attach_library_cameras(original_frame, original_data)
        overlay = result["camera_overlays"]["CAM_F0"]
        self.assertTrue(overlay["valid"])
        self.assertEqual(len(overlay["boxes"]), 1)
        self.assertEqual(overlay["boxes"][0]["geometry_quality"], "measured_source_3d_box")
        self.assertTrue(overlay["trajectory_segments"])
        self.assertTrue(overlay["corridor_polygons"])
        self.assertFalse(overlay["planned_trajectory"])
        self.assertIn("不是在线预测或规划", overlay["projection_note"])
        self.assertIn("地平面", overlay["projection_note"])
        self.assertEqual(original_frame, before_frame)
        self.assertEqual(original_data, before_data)
        self.assertEqual(result["cameras"][0]["frame_id"], original_frame["frame_id"])
        self.assertEqual(result["cameras"][0]["time"], original_frame["time"])

    def test_raw_sensor_to_lidar_is_not_substituted_for_composed_calibration(self):
        value = data()
        value["manifest"][0].update(raw_sensor2lidar_rotation=np.eye(3).tolist(),
                                   raw_sensor2lidar_translation=[999., 0., 0.])
        result = attach_library_cameras(frame(), value)
        self.assertTrue(result["camera_overlays"]["CAM_F0"]["boxes"])

    def test_archived_grid_tags_are_copied_by_id_without_rematching(self):
        current = frame()
        current["visualization"]["objects"] = [dict(id="lead", grid_matched=True, grid_roles=["front"])]
        box = attach_library_cameras(current, data())["camera_overlays"]["CAM_F0"]["boxes"][0]
        self.assertTrue(box["grid_matched"])
        self.assertEqual(box["grid_roles"], ["front"])
        self.assertFalse(box["potential_source"])
        self.assertFalse(box["attention"])

    def test_nuscenes_missing_z_never_becomes_fake_measured_box(self):
        value = data("nuscenes")
        value["original_scene"]["current_context"]["visible_annotation_candidates"] = [
            dict(source_id="lead", position=[10., 0.], height=1.6, length=4., width=2.)]
        value["manifest"][0].update(timestamp_s=11.96342, time_offset_s=-.03658)
        value["exports"][0]["timestamp_s"] = 11.96342
        result = attach_library_cameras(frame(), value)
        overlay = result["camera_overlays"]["CAM_FRONT"]
        self.assertTrue(overlay["valid"])
        self.assertEqual(overlay["boxes"], [])
        self.assertTrue(overlay["trajectory_segments"])
        self.assertIn("-0.036580s", overlay["projection_note"])
        self.assertIn("缺对象z", overlay["projection_note"])
        self.assertEqual(result["cameras"][0]["time"], 12.)
        self.assertEqual(result["cameras"][0]["timestamp_s"], 11.96342)

    def test_evidence_identity_and_time_mismatches_rejected(self):
        for mutation in (lambda value: value.update(frame_id="other"),
                         lambda value: value.update(time=13.),
                         lambda value: value.update(integrity="unchecked"),
                         lambda value: value["original_scene"]["current_context"].update(timestamp_s=13.)):
            with self.subTest(mutation=mutation):
                value = data()
                mutation(value)
                result = attach_library_cameras(frame(), value)
                self.assertFalse(result["library_camera_status"]["available"])
                self.assertEqual(result["cameras"], [])

    def test_image_checksum_source_and_duplicate_mapping_rejected(self):
        for mutation in (lambda value: value["images"][0].update(sha256="wrong"),
                         lambda value: value["images"][0].update(sha256_verified=False),
                         lambda value: value["manifest"][0].update(source_filename="other.jpg"),
                         lambda value: value["images"].append(deepcopy(value["images"][0]))):
            value = data()
            mutation(value)
            self.assertEqual(attach_library_cameras(frame(), value)["cameras"], [])

    def test_invalid_exposure_offset_keeps_image_without_overlays(self):
        value = data()
        value["manifest"][0]["time_offset_s"] = 2.
        result = attach_library_cameras(frame(), value)
        self.assertEqual(len(result["cameras"]), 1)
        overlay = result["camera_overlays"]["CAM_F0"]
        self.assertFalse(overlay["valid"])
        self.assertEqual(overlay["boxes"], [])
        self.assertEqual(overlay["trajectory_segments"], [])
        self.assertEqual(overlay["trajectory_segments_progress"], [])
        self.assertEqual(overlay["trajectory_samples"], [])
        self.assertEqual(overlay["trajectory_source_point_count"], 0)

    def test_mismatched_actor_not_recovered_from_other_evidence(self):
        value = data()
        value["original_scene"]["current_context"]["visible_annotation_candidates"][0]["position"] = [11., 0.]
        overlay = attach_library_cameras(frame(), value)["camera_overlays"]["CAM_F0"]
        self.assertEqual(overlay["boxes"], [])

    def test_route_or_mismatched_future_not_projected_as_trajectory(self):
        for provenance in (dict(kind="navigation_route", anchor_frame_id="normalized:12", time=12.),
                           dict(kind="recorded_future_execution", anchor_frame_id="other", time=12.)):
            current = frame()
            current["visualization"]["trajectory_provenance"] = provenance
            overlay = attach_library_cameras(current, data())["camera_overlays"]["CAM_F0"]
            self.assertEqual(overlay["trajectory_segments"], [])
            self.assertEqual(overlay["corridor_polygons"], [])
            self.assertEqual(overlay["trajectory_samples"], [])
            self.assertEqual(overlay["trajectory_source_point_count"], 0)

    def test_near_and_image_clipping_preserves_crossing_lines(self):
        model = camera_model(data()["manifest"][0])
        self.assertIsNone(project_segment([-10., 0., 0.], [-1., 0., 0.], model))
        self.assertIsNotNone(project_segment([-1., 0., 1.5], [10., 0., 1.5], model))
        line = project_segment([10., -100., 1.5], [10., 100., 1.5], model)
        np.testing.assert_allclose(sorted(point[0] for point in line), [0., 100.], atol=1e-10)
        self.assertTrue(all(0 <= point[1] <= 100 for point in line))

    def test_viewport_progress_uses_original_world_fraction_not_screen_fraction(self):
        model = camera_model(data()["manifest"][0])
        pixels, fractions = _project_segment_progress([2., 6., 1.5], [10., -6., 1.5], model)
        np.testing.assert_allclose(pixels, [[0., 50.], [100., 50.]], atol=1e-12)
        np.testing.assert_allclose(fractions, [5./16., 7./8.], atol=1e-12)
        # The screen clipping fraction is 250/360, quite different from the
        # original 3D line's 5/16. Verify the clipped point through the camera.
        a, b = np.array([2., 6., 1.5]), np.array([10., -6., 1.5])
        for pixel, fraction in zip(pixels, fractions):
            np.testing.assert_allclose(_pixel(_camera_point(a + fraction*(b-a), model), model), pixel,
                                       atol=1e-12)

    def test_near_clip_preserves_original_progress_in_both_directions(self):
        model = camera_model(data()["manifest"][0])
        _, forward = _project_segment_progress([-1., 0., 1.5], [10., 0., 1.5], model)
        _, backward = _project_segment_progress([10., 0., 1.5], [-1., 0., 1.5], model)
        np.testing.assert_allclose(forward, [.1, 1.], atol=1e-12)
        np.testing.assert_allclose(backward, [0., .9], atol=1e-12)

    def test_visible_future_tail_retains_hidden_source_indices(self):
        current = frame()
        points = [[x, 0., 0.] for x in (-10., -5., 10., 20.)]
        current["visualization"].update(trajectory=points, trajectory_segments=[points], corridor_ribbon={})
        overlay = attach_library_cameras(current, data())["camera_overlays"]["CAM_F0"]
        self.assertEqual(overlay["trajectory_source_point_count"], 4)
        self.assertEqual(len(overlay["trajectory_segments"]), 2)
        self.assertEqual(len(overlay["trajectory_segments_progress"]), 2)
        self.assertGreater(overlay["trajectory_segments_progress"][0][0], 1./3.)
        self.assertEqual(overlay["trajectory_segments_progress"][-1][-1], 1.)
        self.assertEqual([point["index"] for point in overlay["trajectory_samples"]], [2, 3])
        np.testing.assert_allclose([point["progress"] for point in overlay["trajectory_samples"]], [2./3., 1.])
        self.assertFalse(overlay["planned_trajectory"])
        self.assertTrue(overlay["recorded_future_execution"])

    def test_remaining_evidence_future_gets_green_first_and_yellow_last(self):
        current = frame()
        points = [[10., 0., 0.], [20., 0., 0.], [30., 0., 0.]]
        current["visualization"].update(trajectory=points, trajectory_segments=[points], corridor_ribbon={})
        overlay = attach_library_cameras(current, data())["camera_overlays"]["CAM_F0"]
        self.assertEqual(overlay["trajectory_source_point_count"], 3)
        self.assertEqual(overlay["trajectory_segments_progress"], [[0., .5], [.5, 1.]])
        self.assertEqual([point["progress"] for point in overlay["trajectory_samples"]], [0., .5, 1.])

    def test_source_gap_is_not_connected_or_renumbered(self):
        current = frame()
        points = [[float(x), 0., 0.] for x in (10, 20, 30, 40, 50, 60)]
        current["visualization"].update(trajectory=points, trajectory_segments=[points[:2], points[4:]],
                                         corridor_ribbon={})
        overlay = attach_library_cameras(current, data())["camera_overlays"]["CAM_F0"]
        self.assertEqual(overlay["trajectory_source_point_count"], 6)
        np.testing.assert_allclose(overlay["trajectory_segments_progress"], [[0., .2], [.8, 1.]])
        self.assertEqual(len(overlay["trajectory_segments"]), 2)

    def test_single_visible_recorded_point_is_green_without_inventing_a_segment(self):
        current = frame()
        points = [[10., 0., 0.]]
        current["visualization"].update(trajectory=points, trajectory_segments=[points], corridor_ribbon={})
        overlay = attach_library_cameras(current, data())["camera_overlays"]["CAM_F0"]
        self.assertEqual(overlay["trajectory_source_point_count"], 1)
        self.assertEqual(overlay["trajectory_segments"], [])
        self.assertEqual(overlay["trajectory_segments_progress"], [])
        self.assertEqual(overlay["trajectory_samples"], [{"point": [50., 65.], "index": 0, "progress": 0.}])

    def test_short_future_under_front_hood_is_not_extended_into_image(self):
        current = frame()
        current["visualization"].update(trajectory_segments=[[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]],
                                         corridor_ribbon={})
        overlay = attach_library_cameras(current, data())["camera_overlays"]["CAM_F0"]
        self.assertTrue(overlay["valid"])
        self.assertTrue(overlay["recorded_future_execution"])
        self.assertEqual(overlay["trajectory_segments"], [])

    def test_quaternion_rejects_unproven_order_or_invalid_rotation(self):
        with self.assertRaises(ValueError):
            quaternion_matrix([2., 0., 0., 0.])

    def test_highway_has_no_invented_front_camera(self):
        result = attach_library_cameras(frame(), dict(available=True, source="highway-env"))
        self.assertEqual(result["cameras"], [])
        self.assertIn("没有前视相机", result["library_camera_status"]["reason"])


if __name__ == "__main__":
    unittest.main()
