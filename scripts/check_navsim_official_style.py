"""Verify plot semantics offline; optionally audit one already exported real trace.

Does not invoke a model, planner, SDK scorer or renderer. Supply --status to
independently inspect the SDK call evidence and fully decode its H.264 export.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ep_driver_navsim.official_render import _camera_path, local_poses, read_packet
from ep_driver_navsim.replay_gallery import update_gallery
from ep_driver_navsim.replay_render import RENDERER_VERSION, load_replay
from nine_grid.paths import filesystem_path


class OfficialStyleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = ROOT / "outputs/navsim_check_temp"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_north_facing_source_keeps_forward_up_and_left_positive(self):
        local = local_poses([[100, 202, np.pi / 2], [98, 200, np.pi]], np.array([100, 200, np.pi / 2]))
        np.testing.assert_allclose(local, [[2, 0, 0], [0, 2, np.pi / 2]], atol=1e-10)

    def test_archived_packet_prefers_gzip_and_relocated_camera(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            sample = Path(tmp)
            (sample / "input_packet.json").write_text('{"source":"old"}', encoding="utf-8")
            with gzip.open(sample / "input_packet.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(dict(source="compressed"), stream)
            self.assertEqual(read_packet(sample / "input_packet.json")["source"], "compressed")
            (sample / "cameras").mkdir()
            image = sample / "cameras/cam_f0.jpg"
            image.write_bytes(b"relocated archive")
            self.assertEqual(_camera_path(dict(view="cam_f0", path="/obsolete/windows/path.jpg"), sample), filesystem_path(image))

    def test_gallery_does_not_count_old_custom_video_as_official_style(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root = Path(tmp)
            for name, official in (("old", False), ("new", True)):
                replay = root / "samples" / name / "official_replay"
                replay.mkdir(parents=True)
                (replay / "official_bev.mp4").write_bytes(b"test status identity only")
                value = dict(status="complete", token=name)
                if official:
                    value.update(renderer_version=RENDERER_VERSION, official_visualization=True)
                (replay / "render_status.json").write_text(json.dumps(value), encoding="utf-8")
            update_gallery(root)
            summary = json.loads((root / "visualizations/replay_summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["complete_videos"], summary["legacy_style_videos"]), (1, 1))
            document = (root / "visualizations/index.html").read_text(encoding="utf-8")
            self.assertIn("static", document.lower())
            self.assertIn('data-official="false"', document)


def verify_export(status_path):
    status = json.loads(filesystem_path(status_path).read_text(encoding="utf-8"))
    assert status["renderer_version"] == RENDERER_VERSION and status["official_visualization"]
    assert status["source"] == "actual_official_scoring_execution"
    assert not any(status[key] for key in ("planner_called", "model_called", "scoring_called", "camera_future_generated"))
    trace_path = filesystem_path(status["replay_path"])
    trace = load_replay(trace_path)
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == status["trace_sha256"]
    for path, checksum in status["official_visualization_source_sha256"].items():
        assert hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest() == checksum
    calls = status["official_functions_called"]
    assert calls["add_map_to_bev_ax"] == 1 and calls["add_camera_ax"] == 8
    assert calls["add_annotations_to_bev_ax"] == len(trace["actors"]) + 1
    assert calls["add_oriented_box_to_bev_ax"] == len(trace["ego"]["states"]) + 1
    video = cv2.VideoCapture(str(filesystem_path(status["video"])))
    count = 0
    previous = None
    changed = 0
    try:
        assert video.isOpened() and abs(video.get(cv2.CAP_PROP_FPS) - 10) < 1e-5
        while True:
            ok, frame = video.read()
            if not ok:
                break
            assert list(frame.shape[:2]) == list(reversed(status["size"]))
            if previous is not None:
                changed += bool(np.any(frame != previous))
            previous = frame
            count += 1
    finally:
        video.release()
    assert count == 41 == status["decoded_frames"] and changed > 0
    assert filesystem_path(status["camera_panel"]).stat().st_size > 0
    print(json.dumps(dict(status="passed", token=trace["token"], decoded_frames=count,
        sdk_plot_calls=calls, sdk_and_trace_hashes_unchanged=True, no_scoring_or_model_calls=True)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OfficialStyleChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.status:
        verify_export(args.status)
