"""Read-only artifact ingestion regressions; no simulator or model is started."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from epviz.episodes import EpisodeStore


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


def plan(frame_id, time, answer_source=None):
    return dict(frame_id=frame_id, time=time, answer_source=answer_source,
                plan=dict(frame_id=frame_id, time=time, valid=True,
                          states=[[1, 2, 0, 3]], controls=[[0, 0]], diagnostics={}))


class EpisodeStoreTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures in the selected workspace; sandbox app temp directories
        # may not grant the Python child process access to their nested paths.
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "group" / "experiment_001"
        self.episode.mkdir(parents=True)
        write_json(self.episode / "configuration.json", {"parameters": {"planner": {"hz": 10}}})
        append(self.episode / "plans.jsonl", plan("episode:000000", 0))
        self.store = EpisodeStore(self.root)
        self.identifier = self.store.discover()[0]["id"]

    def tearDown(self):
        self.assertEqual(self.root.resolve().parent, Path(__file__).resolve().parent)
        self.temporary.cleanup()

    def test_discovery_pending_planning_timeline_and_nested_quality(self):
        append(self.episode / "plans.jsonl", plan("episode:000001", .1))
        append(self.episode / "frames.jsonl", dict(frame_id="episode:000000", time=0,
                                                  ego={"position": [1, 2]}, actors=[], actual={"time": .1}))
        append(self.episode / "quality.jsonl", {"time": 0, "quality": {"frame_id": "episode:000000", "time": 0, "score": .8}})
        manifest = self.store.get_manifest(self.identifier)
        self.assertEqual(manifest["times"], [0, .1])
        self.assertFalse(manifest["frames"][0]["pending"])
        self.assertTrue(manifest["frames"][1]["pending"])
        frame = self.store.get_frame(self.identifier, 0)
        self.assertEqual(frame["quality"]["score"], .8)
        self.assertEqual(frame["time"], 0)
        self.assertEqual(frame["ego"]["position"], [1, 2])
        self.assertEqual(frame["plan"]["controls"], [[0, 0]])
        self.assertIsNone(self.store.get_frame(self.identifier, 1)["ego"])

    def test_incremental_half_line_complete_line_error_and_append(self):
        path = self.episode / "plans.jsonl"
        encoded = json.dumps(plan("episode:000001", .1))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded[:20])
        self.assertEqual(self.store.refresh(self.identifier)["frame_count"], 1)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded[20:] + "\nnot json\n")
        append(path, plan("episode:000002", .2))
        result = self.store.get_manifest(self.identifier)
        self.assertEqual(result["times"], [0, .1, .2])
        self.assertTrue(any("invalid complete line" in warning for warning in result["warnings"]))

    def test_truncation_rebuilds_frame_index(self):
        append(self.episode / "plans.jsonl", plan("episode:000001", .1))
        self.store.refresh(self.identifier)
        (self.episode / "plans.jsonl").write_text(json.dumps(plan("replacement:1", 8)) + "\n", encoding="utf-8")
        manifest = self.store.get_manifest(self.identifier)
        self.assertEqual(manifest["times"], [8])
        self.assertEqual(self.store.get_frame(self.identifier, 0)["frame_id"], "replacement:1")

    def test_unchanged_large_plan_file_is_not_reparsed(self):
        large = plan("episode:000001", .1)
        large["plan"]["states"] *= 20000
        append(self.episode / "plans.jsonl", large)
        self.store.refresh(self.identifier)
        stream = self.store._episodes[self.identifier].streams["plans"]
        read_before = stream.bytes_read
        self.store.get_manifest(self.identifier)
        self.store.get_frame(self.identifier, 0)
        self.assertEqual(stream.bytes_read, read_before)

    def test_join_requires_both_frame_id_and_source_time(self):
        append(self.episode / "timing.jsonl", {"frame_id": "another:0", "time": 0, "mpc_s": 9})
        append(self.episode / "timing.jsonl", {"frame_id": "episode:000000", "time": .1, "mpc_s": 9})
        frame = self.store.get_frame(self.identifier, 0)
        self.assertIsNone(frame["timing"])
        self.assertTrue(any("different/missing source time" in warning for warning in frame["warnings"]))

    def _camera(self, timestamp=123, frame=100):
        folder = self.episode / "camera_frames" / "00000100"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "front.png").write_bytes(b"image fixture")
        write_json(folder / "manifest.json", dict(frame_id="episode:000000", time=0,
                   carla_frame=100, carla_timestamp=123,
                   cameras=[dict(view="front", frame=frame, timestamp=timestamp, width=640, height=360,
                                 rgb_path="/server/outputs/experiment_001/camera_frames/00000100/front.png",
                                 world_to_camera_ue=[[1, 0, 0, 0]] * 4)]))
        write_json(folder / "current_observed_state.json", dict(frame=100, timestamp=123,
                   ego={"z": 1}, actors=[{"height_m": 1.8}]))
        return folder

    def test_camera_exact_match_and_low_rate_archive_does_not_borrow(self):
        self._camera()
        append(self.episode / "plans.jsonl", plan("episode:000001", .1))
        frame = self.store.get_frame(self.identifier, 0)
        self.assertEqual(len(frame["cameras"]), 1)
        self.assertEqual(frame["camera_state"]["ego"]["z"], 1)
        self.assertTrue(frame["cameras"][0]["calibrated"])
        self.assertEqual(frame["cameras"][0]["source_time"], 0)
        next_frame = self.store.get_frame(self.identifier, 1)
        self.assertEqual(next_frame["cameras"], [])
        self.assertIsNone(next_frame["camera_state"])
        self.assertIn("lower rate", " ".join(next_frame["warnings"]))

    def test_camera_same_timestamp_but_wrong_frame_is_excluded(self):
        self._camera(frame=99)
        frame = self.store.get_frame(self.identifier, 0)
        self.assertEqual(frame["cameras"], [])
        self.assertIn("mismatch", " ".join(frame["warnings"]))

    def test_relocated_windows_asset_and_unauthorized_paths(self):
        folder = self.episode / "captures" / "000000"
        folder.mkdir(parents=True)
        (folder / "goal_overlay.png").write_bytes(b"image")
        write_json(folder / "scene.json", dict(frame_id="episode:000000", time=0, ego={}, actors=[], rules={}))
        payload = plan("episode:000000", 0)
        payload["goal_overlay_path"] = r"\\?\D:\old\experiment_001\captures\000000\goal_overlay.png"
        (self.episode / "plans.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
        frame = self.store.get_frame(self.identifier, 0)
        self.assertEqual(frame["assets"]["goal_overlay"], "captures/000000/goal_overlay.png")
        self.assertEqual(self.store.resolve_asset(self.identifier, frame["assets"]["goal_overlay"]), folder / "goal_overlay.png")
        for invalid in ("../configuration.json", "/etc/passwd", "C:/Windows/system.ini", r"..\secret", "//server/secret"):
            with self.assertRaises(ValueError, msg=invalid):
                self.store.resolve_asset(self.identifier, invalid)
        self.assertIsNone(self.store._relative(self.store._episode(self.identifier), "/etc/passwd"))

    def test_symlink_cannot_escape_episode(self):
        outside = self.root / "private.txt"
        outside.write_text("private", encoding="utf-8")
        link = self.episode / "escape.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Host does not allow symlinks")
        with self.assertRaises(ValueError):
            self.store.resolve_asset(self.identifier, "escape.txt")

    def test_explicit_model_binding_and_no_implicit_latest_response(self):
        append(self.episode / "model_inputs.jsonl", dict(frame_id="episode:000000", time=0, prompt="Real input"))
        append(self.episode / "model_outputs.jsonl", dict(frame_id="episode:000000", time=0, raw="Real answer", answer={"objects": []}))
        append(self.episode / "plans.jsonl", plan("episode:000001", .1, answer_source="episode:000000"))
        append(self.episode / "plans.jsonl", plan("episode:000002", .2))
        bound = self.store.get_frame(self.identifier, 1)
        self.assertIsNone(bound["model_output"])
        self.assertEqual(bound["active_model_output"]["raw"], "Real answer")
        self.assertEqual(bound["active_model_input"]["time"], 0)
        self.assertIsNone(self.store.get_frame(self.identifier, 2)["active_model_output"])

    def test_single_episode_root_and_nested_discovery(self):
        direct = EpisodeStore(self.episode)
        identifier = direct.discover()[0]["id"]
        self.assertEqual(direct.get_manifest(identifier)["frame_count"], 1)
        second = self.root / "other" / "nested" / "experiment_002"
        second.mkdir(parents=True)
        append(second / "plans.jsonl", plan("other:1", 0))
        self.assertEqual(len(self.store.discover()), 2)

    def test_model_source_images_relocate_without_serving_external_files(self):
        folder = self.episode / "captures" / "000000"
        folder.mkdir(parents=True)
        (folder / "bev.png").write_bytes(b"image")
        (folder / "topdown.png").write_bytes(b"image")
        images = ["/old/server/experiment_001/captures/000000/bev.png",
                  r"D:\old\experiment_001\captures\000000\topdown.png",
                  "/etc/passwd", "../private.png"]
        append(self.episode / "model_inputs.jsonl", dict(frame_id="episode:000000", time=0,
                                                        prompt="original", images=images))
        append(self.episode / "plans.jsonl", plan("episode:000001", .1, answer_source="episode:000000"))
        original = self.store.get_frame(self.identifier, 0)["model_input"]
        active = self.store.get_frame(self.identifier, 1)["active_model_input"]
        self.assertEqual(original["images"], images)
        self.assertEqual(original["image_paths"], ["captures/000000/bev.png", "captures/000000/topdown.png"])
        self.assertEqual(active["image_paths"], original["image_paths"])
        for relative in active["image_paths"]:
            self.assertTrue(self.store.resolve_asset(self.identifier, relative).is_file())


if __name__ == "__main__":
    unittest.main()
