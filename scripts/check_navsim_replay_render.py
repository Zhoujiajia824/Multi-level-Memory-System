"""Offline checks for rendering captured evaluation states; no NAVSIM imports."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ep_driver_navsim.replay_render import VERSION, COLORS, _Canvas, footprint, load_replay, render_replay


def fixture():
    states = [[2 + i * 0.2, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0] for i in range(41)]
    actors = [[dict(id="reactive_car", type="vehicle", center=[18 + i * .08, 4, 0], length_m=4.5, width_m=1.8),
               dict(id="pedestrian", type="pedestrian", center=[12, 9 - i * .03, 1.57], length_m=.7, width_m=.6)]
              for i in range(41)]
    return dict(version=VERSION, token="offline_fixture", stage="original", dt_s=.1,
        ego=dict(states=states, length_m=4.5, width_m=1.8, rear_axle_to_center_m=1.4),
        planned_poses_world=[[2 + i * .22, 4 + .15 * np.sin(i / 40 * np.pi), 0] for i in range(41)],
        human_poses_world=[[2 + i * .18, 3.7, 0] for i in range(41)],
        map=dict(polygons=[dict(wkt="POLYGON ((-10 0, 40 0, 40 12, -10 12, -10 0))", kind="roadblock", token="road")],
                 centerline=[[-10, 4], [40, 4]]), actors=actors,
        red_lights=[[dict(wkt="POLYGON ((30 0, 31 0, 31 12, 30 12, 30 0))", token="signal")] for _ in range(41)],
        metrics=dict(official_final=dict(no_at_fault_collisions=1., drivable_area_compliance=1.,
            driving_direction_compliance=1., traffic_light_compliance=1., ego_progress=.8,
            time_to_collision_within_bound=1., lane_keeping=1., history_comfort=1.), raw={}),
        events=[dict(frame_index=20, type="raw_first_ttc", actor_id="reactive_car")],
        provenance=dict(source="offline renderer fixture; not a real driving evaluation"))


class ReplayRenderChecks(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / "outputs/navsim_check_temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temp_root)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, value):
        path = self.directory / "trace.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(value, stream)
        return path

    def test_rear_axle_center_and_rotated_footprint(self):
        corners = footprint([100, 200, np.pi / 2], 4.5, 1.8, 1.4)
        np.testing.assert_allclose(corners.mean(axis=0), [100, 201.4])
        np.testing.assert_allclose(np.ptp(corners, axis=0), [1.8, 4.5])
        actor = footprint([100, 200, np.pi / 2], 4.5, 1.8)
        np.testing.assert_allclose(actor.mean(axis=0), [100, 200])

    def test_no_imputed_missing_traffic_frames(self):
        replay = fixture()
        replay["actors"].pop()
        with self.assertRaisesRegex(ValueError, "not aligned"):
            load_replay(self.save(replay))

    def test_cached_labels_are_pixel_identical_to_direct_text(self):
        canvas = _Canvas(fixture(), 900)
        cached = Image.new("RGB", (900, 150), "white")
        direct = cached.copy()
        cached_draw, direct_draw = ImageDraw.Draw(cached), ImageDraw.Draw(direct)
        for row, (font, text, color) in enumerate((("normal", "Ego glyphs: 0.0 / 4.0s", COLORS["ink"]),
                ("small", "RAW contact [vehicle]", COLORS["red"]),
                ("metric", "NC 1.00", COLORS["ink"]),
                ("normal", "Ego glyphs: 0.0 / 4.0s", COLORS["muted"]))):
            xy = (20, 10 + row * 30)
            canvas.text(cached_draw, xy, text, font, color)
            direct_draw.text(xy, text, font=canvas.fonts[font], fill=color)
        np.testing.assert_array_equal(np.asarray(cached), np.asarray(direct))

    def test_synthetic_without_human_reference_and_plan_alignment(self):
        replay = fixture()
        replay.update(stage="synthetic", human_poses_world=None)
        captured = load_replay(self.save(replay))
        self.assertEqual(_Canvas(captured, 900).frame(40).size, (900, 900))
        replay["planned_poses_world"].pop()
        with self.assertRaisesRegex(ValueError, "plan is not aligned"):
            load_replay(self.save(replay))

    def test_reject_invalid_timing_geometry_and_events(self):
        edits = (("timing", lambda v: v.update(dt_s=.5)),
                 ("ego states", lambda v: v["ego"]["states"][2].__setitem__(0, float("nan"))),
                 ("map geometry", lambda v: v["map"]["polygons"][0].update(wkt="POLYGON EMPTY")),
                 ("event frame", lambda v: v["events"][0].update(frame_index=41)))
        for reason, edit in edits:
            with self.subTest(reason=reason):
                replay = fixture()
                edit(replay)
                with self.assertRaises(ValueError):
                    load_replay(self.save(replay))

    def test_mp4_frames_exactly_match_actual_endpoint_inclusive_samples(self):
        replay = fixture()
        expected = copy.deepcopy(replay)
        path = self.save(replay)
        # Exercise long worktree/sample paths: final basenames fit MAX_PATH,
        # while verbose 32-character UUID temporary names would exceed it.
        destination = self.directory / ("n" * max(1, 225 - len(str(self.directory)) - 1))
        status = render_replay(path, destination, size=900)
        self.assertEqual(status["decoded_frames"], 41)
        self.assertEqual(status["simulated_span_s"], 4.)
        self.assertEqual(status["encoded_duration_s"], 4.1)
        self.assertTrue(status["playback_verified"])
        self.assertEqual(load_replay(path), expected)
        capture = cv2.VideoCapture(status["video"])
        try:
            self.assertTrue(capture.isOpened())
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        self.assertEqual(len(frames), 41)
        self.assertEqual(frames[0].shape, (900, 900, 3))
        self.assertGreater(np.mean(np.abs(frames[0].astype(float) - frames[-1].astype(float))), .1)
        recorded = json.loads((destination / "render_status.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["source"], "actual_official_scoring_execution")


if __name__ == "__main__":
    unittest.main(verbosity=2)
