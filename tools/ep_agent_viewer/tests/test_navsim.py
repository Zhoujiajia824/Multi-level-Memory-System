"""Offline-only NAVSIM ingestion, coordinate and HTTP contract regressions."""
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen

from epviz.episodes import EpisodeStore
from epviz.server import ViewerServer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)


class NavsimTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.sample = self.root / "run" / "samples" / "sample_digest"
        self.origin = [10., 20., math.pi / 2]
        self.packet = dict(protocol="ep-driver-navsim-privileged-history-1.0", identity=dict(token="tokenA"),
            vehicle=dict(rear_axle_to_center_m=1.),
            ego=dict(id="ego", position=[1., 0.], heading=0., length=5., width=2.),
            actors=[], lanes=[], route=[], route_lane_ids=[], rules={"must_wait": False},
            diagnostics=dict(source_rear_axle_global_pose=self.origin),
            camera_images=[dict(view="cam_f0", path="C:/old/run/samples/sample_digest/cameras/cam_f0.jpg",
                                frame_id="tokenA", time=0.)])
        self.evidence = dict(identity=dict(token="tokenA", timestamp_us=123456),
            parameters=dict(mpc=dict(step_s=.1)), plan=dict(frame_id="tokenA", time=0., valid=True,
                states=[[1. + .1 * i, 0., 0., 1.] for i in range(41)], controls=[[0, 0]] * 40,
                diagnostics=dict(safety_feasible=False)),
            internal_quality=dict(valid=False, score=.2), model_raw="ORIGINAL RAW", model_answer={"explanation": "answer"},
            retrieval=dict(frame_id="tokenA", records=[dict(id="memory1", experience="original text",
                _representations=dict(features=[1] * 10000), scene=dict(lanes=[1] * 10000))], scores=[dict(bev=.8, grid=.7, occupancy=.9)]))
        self.trace = dict(token="tokenA", version="navsim-official-replay-1.0", dt_s=.1,
            ego=dict(states=[[10., 20. + .1 * i, math.pi / 2, 1., 0., 0., 0., 0., 0., 0., 0.] for i in range(41)],
                     length_m=5., width_m=2., rear_axle_to_center_m=1.),
            actors=[[dict(id="actor", type="vehicle", center=[8., 23., math.pi], length_m=4., width_m=2.)] for _ in range(41)],
            metrics=dict(official_final=dict(pdm_score=.6)),
            provenance=dict(coordinates="global_map_xy_m; ego/planned/human poses at rear axle; actor poses at box center"))
        write(self.sample / "input_packet.json", self.packet)
        write(self.sample / "planning_evidence.json", self.evidence)
        write(self.sample / "trajectory.json", dict(frame_id="tokenA", poses=[[.5 * i, 0., 0.] for i in range(1, 9)],
              sampling=dict(horizon_s=4., interval_s=.5)))
        write(self.sample / "official_replay/trace.json.gz", self.trace)
        write(self.sample / "official_replay/render_status.json", dict(status="complete", token="tokenA", encoded_duration_s=4.1))
        (self.sample / "official_replay/official_bev.mp4").write_bytes(b"video")
        (self.sample / "model_prompt.txt").write_text("ORIGINAL PROMPT", encoding="utf-8")
        (self.sample / "cameras").mkdir()
        (self.sample / "cameras/cam_f0.jpg").write_bytes(b"image")
        write(self.root / "run/benchmark_result.json", dict(official_two_stage_epdms=.4, evaluated_tokens=2,
              official_csv="C:/old/run/official_scores/score.csv"))
        (self.root / "run/official_scores").mkdir()
        (self.root / "run/official_scores/score.csv").write_text("token,valid,score\ntokenA,True,0.7\n", encoding="utf-8")
        self.store = EpisodeStore(self.root)
        self.identifier = self.store.discover()[0]["id"]

    def tearDown(self):
        self.temp.cleanup()

    def test_nonzero_global_yaw_rear_axle_conversion_and_actor_center(self):
        manifest = self.store.get_manifest(self.identifier)
        self.assertEqual(len(manifest["frames"]), 41)
        self.assertEqual(manifest["times"], [round(.1 * i, 9) for i in range(41)])
        frame = self.store.get_frame(self.identifier, 10)
        self.assertAlmostEqual(frame["ego"]["position"][0], 2.)
        self.assertAlmostEqual(frame["ego"]["position"][1], 0.)
        actor = frame["actors"][0]
        self.assertAlmostEqual(actor["position"][0], 3.)
        self.assertAlmostEqual(actor["position"][1], 2.)
        self.assertAlmostEqual(actor["heading"], math.pi / 2)
        self.assertEqual(len(actor["position"]), 2)  # trace's third column is yaw, never height.
        paths = frame["navsim"]["trajectories"]
        self.assertEqual(paths["source_plan_center"][0], [1., 0., 0.])
        self.assertEqual(len(paths["submitted_rear_axle"]), 8)
        self.assertAlmostEqual(paths["official_simulated_rear_axle"][10][0], 1.)
        self.assertEqual(len(frame["plan"]["states"][0]), 4)

    def test_source_images_and_models_are_never_future_observations(self):
        source, future = (self.store.get_frame(self.identifier, i) for i in (0, 1))
        self.assertEqual(source["cameras"][0]["path"], "cameras/cam_f0.jpg")
        self.assertFalse(source["cameras"][0]["calibrated"])
        self.assertFalse(source["cameras"][0]["projection_available"])
        self.assertEqual(future["cameras"], [])
        self.assertEqual(future["assets"], {})
        self.assertIsNone(future["rules"])
        self.assertIsNone(future["model_input"])
        self.assertEqual(future["active_model_input"]["prompt"], "ORIGINAL PROMPT")
        self.assertEqual(source["model_output"]["raw"], "ORIGINAL RAW")
        self.assertEqual(source["model_input"]["image_paths"], [])
        self.assertFalse(source["model_input"]["image_manifest_available"])
        record = source["retrieval"]["records"][0]
        self.assertEqual(record["experience"], "original text")
        self.assertNotIn("_representations", record)
        self.assertNotIn("scene", record)
        self.assertEqual(source["retrieval"]["scores"][0], dict(bev=.8, grid=.7, occupancy=.9))

    def test_scores_keep_internal_anchor_csv_and_run_aggregate_distinct(self):
        scores = self.store.get_frame(self.identifier, 1)["navsim"]["scores"]
        self.assertEqual(scores["internal_quality"]["score"], .2)
        self.assertFalse(scores["internal_quality_is_official_metric"])
        self.assertEqual(scores["official_anchor"]["pdm_score"], .6)
        self.assertEqual(scores["official_csv_anchor"]["score"], "0.7")
        self.assertEqual(scores["official_two_stage_aggregate"]["score"], .4)
        # Selecting a sample alone does not authorize reading a parent run.
        isolated = EpisodeStore(self.sample)
        item = isolated.discover()[0]
        self.assertIsNone(isolated.get_frame(item["id"], 0)["navsim"]["scores"]["official_two_stage_aggregate"]["score"])

    def test_gzip_input_and_missing_trace_fall_back_to_one_anchor(self):
        write(self.sample / "input_packet.json.gz", self.packet)
        (self.sample / "input_packet.json").unlink()
        (self.sample / "official_replay/trace.json.gz").unlink()
        manifest = self.store.get_manifest(self.identifier)
        self.assertEqual(manifest["times"], [0.])
        self.assertFalse(manifest["navsim"]["trace_available"])
        self.assertEqual(len(self.store.get_frame(self.identifier, 0)["cameras"]), 1)

    def test_video_renderer_identity_is_explicit_and_legacy_is_not_official(self):
        video = self.store.get_manifest(self.identifier)["video"]
        self.assertTrue(video["available"])
        self.assertFalse(video["official_visualization"])
        self.assertIn("旧版", video["visualization_label"])
        self.assertEqual(video["path"], "official_replay/official_bev.mp4")
        status = dict(status="complete", token="tokenA", renderer_version="navsim-sdk-plots-1.0", official_visualization=True)
        write(self.sample / "official_replay/render_status.json", status)
        video = self.store.get_manifest(self.identifier)["video"]
        self.assertTrue(video["official_visualization"])
        self.assertIn("SDK", video["visualization_label"])
        for changes in (dict(official_visualization=False), dict(renderer_version="unknown"), dict(token="another")):
            write(self.sample / "official_replay/render_status.json", {**status, **changes})
            unverified = self.store.get_manifest(self.identifier)["video"]
            self.assertFalse(unverified["official_visualization"])
            self.assertIn("未认证", unverified["visualization_label"])

    def test_same_token_in_distinct_runs_gets_distinct_video_asset_urls(self):
        import shutil
        second = self.root / "runB/samples/sample_digest"
        second.parent.mkdir(parents=True)
        shutil.copytree(self.sample, second)
        server = ViewerServer(("127.0.0.1", 0), self.root)
        try:
            items = server.store.discover()
            self.assertEqual(len(items), 2)
            manifests = [server.decorate(item["id"], server.store.get_manifest(item["id"])) for item in items]
            self.assertEqual(manifests[0]["navsim"]["anchor_token"], manifests[1]["navsim"]["anchor_token"])
            self.assertNotEqual(manifests[0]["video"]["url"], manifests[1]["video"]["url"])
            self.assertNotEqual(manifests[0]["episode_id"], manifests[1]["episode_id"])
            first_url = manifests[0]["video"]["url"]
            first = server.store._episode(items[0]["id"])
            (first.path / "official_replay/official_bev.mp4").write_bytes(b"replacement-video")
            revised = server.decorate(items[0]["id"], server.store.get_manifest(items[0]["id"]))
            self.assertNotEqual(first_url, revised["video"]["url"])
        finally:
            server.server_close()

    def test_mismatched_token_coordinate_provenance_and_partial_gzip_excluded(self):
        for change in (dict(token="wrong"), dict(provenance={})):
            write(self.sample / "official_replay/trace.json.gz", {**self.trace, **change})
            self.assertEqual(len(self.store.get_manifest(self.identifier)["frames"]), 1)
        (self.sample / "official_replay/trace.json.gz").write_bytes(b"\x1f\x8bpartial")
        self.assertEqual(len(self.store.get_manifest(self.identifier)["frames"]), 1)

    def test_samples_stay_independent_and_reads_do_not_modify_artifacts(self):
        other = self.root / "run/samples/another_digest"
        write(other / "input_packet.json", {**self.packet, "identity": {"token": "tokenB"}})
        write(other / "trajectory.json", {"frame_id": "tokenB", "poses": [[1, 0, 0]]})
        files = list(self.root.rglob("*"))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
        items = self.store.discover()
        self.assertEqual(len(items), 2)
        self.assertEqual(sorted(item["frame_count"] for item in items), [1, 41])
        for item in items:
            self.store.get_manifest(item["id"])
            self.store.get_frame(item["id"], 0)
            with self.assertRaises(IndexError):
                self.store.get_frame(item["id"], item["frame_count"])
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()})

    def test_http_source_urls_future_no_projection_and_potential_disabled(self):
        server = ViewerServer(("127.0.0.1", 0), self.root, engine_root="nonexistent-engine")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def get(path):
            with urlopen(base + path, timeout=10) as response:
                return json.load(response)
        try:
            identifier = get("/api/episodes")["episodes"][0]["id"]
            source = get(f"/api/episodes/{identifier}/frames/0")
            future = get(f"/api/episodes/{identifier}/frames/1")
            self.assertIn("/api/asset", source["cameras"][0]["url"])
            self.assertFalse(source["camera_overlays"]["cam_f0"]["valid"])
            self.assertEqual(future["cameras"], [])
            self.assertEqual(len(future["visualization"]["trajectory"]), 41)
            self.assertIsNone(future["map"])
            self.assertNotIn("lanes", future["scene"])
            self.assertEqual(future["visualization"]["trajectory_provenance"]["kind"], "recorded_future_execution")
            self.assertFalse(get(f"/api/episodes/{identifier}/potential/0")["available"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)


@unittest.skipUnless(os.environ.get("EPVIZ_NAVSIM_REAL_ROOT"), "Set EPVIZ_NAVSIM_REAL_ROOT for existing saved-run regression")
class RealNavsimArtifacts(unittest.TestCase):
    def test_discover_and_read_every_saved_anchor_without_running_driver(self):
        root = Path(os.environ["EPVIZ_NAVSIM_REAL_ROOT"])
        store = EpisodeStore(root)
        episodes = store.discover()
        self.assertTrue(episodes)
        for item in episodes:
            manifest = store.get_manifest(item["id"])
            source = store.get_frame(item["id"], 0)
            last = store.get_frame(item["id"], len(manifest["frames"]) - 1)
            self.assertEqual(source["navsim"]["anchor_token"], last["navsim"]["anchor_token"])
            self.assertEqual(len(source["navsim"]["trajectories"]["submitted_rear_axle"]), 8)
            if len(manifest["frames"]) > 1:
                self.assertEqual(last["cameras"], [])
                self.assertAlmostEqual(last["time"], 4.)


if __name__ == "__main__":
    unittest.main()
