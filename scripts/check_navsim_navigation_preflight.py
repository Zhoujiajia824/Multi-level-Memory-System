"""Offline navigation checkpoint regressions; no real maps, models or scores."""
import copy
import hashlib
from concurrent.futures import Future
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ep_driver_navsim import nav_preflight as preflight
from ep_driver_navsim.protocol import digest, write_json
from nine_grid.paths import filesystem_path


class CurrentOnlyRows:
    def __init__(self, current):
        self.current = current

    def __getitem__(self, index):
        if index != 3:
            raise AssertionError("Preflight inspected a noncurrent frame")
        return self.current


def evidence(task, contract):
    return dict(route_evidence=dict(topology_correction_changed=task["token"] == "b",
                                    current_lane_id="lane"),
                usable_lane_count=2, route_point_count=3, skipped_lane_ids=[])


class NavigationCheckpointTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / "outputs/navsim_check_temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.temp.name = str(filesystem_path(self.temp.name))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = dict(evaluation_tokens=["c", "a", "b"], output_dir=str(self.root / "A"),
            metric_cache_path=str(self.root / "official_cache"), sdk_root=str(self.root / "sdk"),
            maps_runtime_root=str(self.root / "maps"), data_root=str(self.root / "data"),
            split="fake", desired_speed_mps=10., map_radius_m=160.,
            official_commit="sdk-commit", official_source_sha256="sdk-source",
            numerical_source_sha256="adapter-source", map_source_sha256={"map.gpkg": "map-source"},
            scene_filter_sha256="filter", mapping_sha256="mapping")
        self.tasks = {token: dict(token=token, origin=[1., 2., .3], public_roadblock_ids=["R1"],
            map_name="fake-map", map_version=preflight.MAP_VERSION) for token in ["a", "b", "c"]}
        self.runtime = dict(python="offline", packages={"shapely": "fake"})
        self.addCleanup(patch.stopall)
        patch.object(preflight, "verified_runtime", side_effect=lambda settings: copy.deepcopy(self.runtime)).start()
        patch.object(preflight, "extract_task", side_effect=lambda loader, token: copy.deepcopy(self.tasks[token])).start()

    def run_check(self, checker=evidence, settings=None):
        return preflight.inspect_navigation(settings or self.settings, object(), map_checker=checker)

    def report(self, settings=None):
        return json.loads((Path((settings or self.settings)["output_dir"]) / "navigation_preflight.json").read_text())

    def cache_files(self):
        return sorted(filesystem_path(self.root / "official_cache_ep_navigation_preflight").glob("*/*.json"))

    def test_complete_and_cross_group_reuse_without_official_metric_cache(self):
        calls = []
        result = self.run_check(lambda task, contract: (calls.append(task["token"]) or evidence(task, contract)))
        self.assertEqual(calls, ["a", "b", "c"])
        self.assertEqual(result["computed_tokens"], 3)
        report = self.report()
        self.assertTrue(report["complete"])
        self.assertEqual(report["checked_tokens_sha256"], digest(["a", "b", "c"]))
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["official_metric_cache"])
        self.assertEqual(len(self.cache_files()), 3)
        other = dict(self.settings, output_dir=str(self.root / "B"))
        result = self.run_check(lambda *args: self.fail("A/B identical map check was repeated"), other)
        self.assertEqual(result["reused_tokens"], 3)
        self.assertEqual(self.report(other)["topology_corrected_tokens"][0]["token"], "b")
        self.assertFalse((self.root / "official_cache/metadata").exists())
        self.assertFalse((self.root / "official_cache").exists())
        # A successful check must not prevent first-time official cache creation.
        from ep_driver_navsim.launch import verify_cache_contract
        self.assertEqual(verify_cache_contract(self.settings, creating=True),
                         Path(self.settings["metric_cache_path"]) / "ep_navsim_cache_contract.json")

    def test_explicit_navigation_cache_cannot_pollute_official_cache(self):
        settings = dict(self.settings, navigation_preflight_cache=str(self.root / "official_cache" / "custom"))
        with self.assertRaisesRegex(ValueError, "outside the official"):
            self.run_check(settings=settings)
        self.assertFalse((self.root / "official_cache").exists())

    def test_partial_interruption_is_not_passed_and_resumes_remaining(self):
        def stop(task, contract):
            if task["token"] == "b":
                raise KeyboardInterrupt("simulated interruption")
            return evidence(task, contract)
        with self.assertRaises(KeyboardInterrupt):
            self.run_check(stop)
        report = self.report()
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "in_progress")
        self.assertEqual(report["checked_tokens"], 1)
        self.assertEqual(len(self.cache_files()), 1)
        result = self.run_check()
        self.assertEqual((result["reused_tokens"], result["computed_tokens"]), (1, 2))

    def test_failure_durable_immediately_and_reused_as_failure(self):
        def fail_then_stop(task, contract):
            if task["token"] == "a":
                raise ValueError("unconnectable public route")
            report = self.report()
            self.assertEqual(report["failed_tokens"][0]["token"], "a")
            self.assertEqual(report["failed_tokens"][0]["error"]["message"], "unconnectable public route")
            raise KeyboardInterrupt("after first failure")
        with self.assertRaises(KeyboardInterrupt):
            self.run_check(fail_then_stop)
        with self.assertRaisesRegex(RuntimeError, "failed for 1/3"):
            self.run_check()
        report = self.report()
        self.assertTrue(report["complete"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reused_tokens"], 1)
        self.assertEqual(report["failed_tokens"][0]["task"], self.tasks["a"])
        with self.assertRaisesRegex(RuntimeError, "failed for 1/3"):
            self.run_check(lambda *args: self.fail("Cached failure must not disappear"))

    def test_corrupt_and_partial_records_recomputed(self):
        self.run_check()
        files = self.cache_files()
        files[0].write_text('{"truncated":', encoding="utf-8")
        row = json.loads(files[1].read_text())
        row["complete"] = False
        write_json(files[1], row)
        result = self.run_check()
        self.assertEqual((result["reused_tokens"], result["computed_tokens"]), (1, 2))

    def test_changed_pose_public_route_map_or_anchor_invalidates_only_token(self):
        self.run_check()
        for field, value in [("origin", [9., 2., .3]), ("public_roadblock_ids", ["R2"]),
                             ("map_name", "changed-map"), ("timestamp_us", 999)]:
            self.tasks["a"][field] = value
            result = self.run_check()
            self.assertEqual((result["reused_tokens"], result["computed_tokens"]), (2, 1))

    def test_token_set_and_source_contract_changes_invalidate_all(self):
        self.run_check()
        for key in ["official_commit", "official_source_sha256", "numerical_source_sha256",
                    "scene_filter_sha256", "mapping_sha256", "map_radius_m", "desired_speed_mps"]:
            changed = copy.deepcopy(self.settings)
            changed[key] = changed[key] + "changed" if isinstance(changed[key], str) else changed[key] + 1
            self.assertEqual(self.run_check(settings=changed)["computed_tokens"], 3)
        changed = dict(self.settings, evaluation_tokens=["a", "b"])
        self.assertEqual(self.run_check(settings=changed)["computed_tokens"], 2)
        self.runtime["packages"]["shapely"] = "changed"
        self.assertEqual(self.run_check()["computed_tokens"], 3)

    def test_changed_map_hash_and_actual_adapter_source_invalidate(self):
        self.run_check()
        changed = dict(self.settings, map_source_sha256={"map.gpkg": "different"})
        self.assertEqual(self.run_check(settings=changed)["computed_tokens"], 3)
        original_read = Path.read_bytes
        def changed_source(path):
            data = original_read(path)
            return data + b"\n# changed source\n" if path.name == "input_adapter.py" else data
        with patch.object(Path, "read_bytes", changed_source):
            self.assertEqual(self.run_check()["computed_tokens"], 3)

    def test_extraction_failure_never_hidden_or_reused(self):
        with patch.object(preflight, "extract_task", side_effect=ValueError("bad current row")):
            with self.assertRaisesRegex(RuntimeError, "failed for 3/3"):
                self.run_check()
        result = self.run_check()
        self.assertEqual((result["reused_tokens"], result["computed_tokens"]), (0, 3))

    def test_source_verification_error_replaces_old_pass_report(self):
        self.run_check()
        with patch.object(preflight, "verified_runtime", side_effect=ValueError("SDK changed")):
            with self.assertRaisesRegex(ValueError, "SDK changed"):
                self.run_check()
        self.assertFalse(self.report()["complete"])
        self.assertEqual(self.report()["status"], "in_progress")

    def test_runtime_changed_mid_check_never_publishes_pass(self):
        def change(task, contract):
            self.runtime["packages"]["shapely"] = "replaced-during-check"
            return evidence(task, contract)
        with self.assertRaisesRegex(RuntimeError, "changed during preflight"):
            self.run_check(change)
        self.assertFalse(self.report()["complete"])
        self.assertEqual(self.report()["status"], "in_progress")

    def test_duplicate_empty_selection_rejected_and_worker_validation(self):
        for tokens in [[], ["a", "a"]]:
            with self.assertRaises(ValueError):
                self.run_check(settings=dict(self.settings, evaluation_tokens=tokens))
        for count in [0, True, 33, "8"]:
            with self.assertRaises(ValueError):
                preflight.worker_count(dict(navigation_preflight_workers=count))
        with patch.object(preflight.os, "name", "nt"):
            self.assertEqual(preflight.worker_count(dict(navigation_preflight_workers=8)), 1)

    def test_spawn_dispatch_contains_only_public_task_and_contract(self):
        submits = []
        class FakePool:
            def __init__(pool, max_workers, mp_context):
                self.assertEqual(max_workers, 2)
                self.assertEqual(mp_context.get_start_method(), "spawn")
            def submit(pool, function, task, contract):
                submits.append((task, contract))
                future = Future()
                future.set_result(evidence(task, contract))
                return future
            def shutdown(pool, **kwargs):
                pass
        with patch.object(preflight, "worker_count", return_value=2), patch.object(preflight, "ProcessPoolExecutor", FakePool):
            result = preflight.inspect_navigation(self.settings, object())
        self.assertEqual(result["computed_tokens"], 3)
        self.assertEqual(len(submits), 3)
        self.assertEqual(submits[0][0], self.tasks["a"])
        self.assertFalse(submits[0][1]["official_metric_cache"])


class CurrentTaskExtractionTests(unittest.TestCase):
    def test_original_only_current_fields_and_official_quaternion(self):
        row = dict(token="a", timestamp=123, ego2global_translation=[1., 2., 0.],
                   ego2global_rotation=[1., 0., 0., 0.], map_location="map", roadblock_ids=["R"])
        loader = NS(synthetic_scenes={}, scene_frames_dicts={"a": CurrentOnlyRows(row)})
        fake = NS(Quaternion=lambda *args: NS(yaw_pitch_roll=(.25, 0, 0)))
        with patch.dict(sys.modules, pyquaternion=fake):
            task = preflight.extract_task(loader, "a")
        self.assertEqual(task["origin"], [1., 2., .25])
        row["token"] = "wrong"
        with patch.dict(sys.modules, pyquaternion=fake), self.assertRaisesRegex(ValueError, "Requested"):
            preflight.extract_task(loader, "a")

    def test_synthetic_no_future_rows_and_mismatched_anchor_rejected(self):
        value = dict(scene_metadata=dict(num_history_frames=4, map_name="map", initial_token="a"),
            frames=CurrentOnlyRows(dict(token="a", timestamp=123, ego_status=dict(ego_pose=[1., 2., .3]), roadblock_ids=["R"])))
        temp_root = ROOT / "outputs/navsim_check_temp"
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / "scene.pkl"
            def save():
                path.write_bytes(pickle.dumps(value))
            save()
            loader = NS(synthetic_scenes={"a": (path, None)})
            self.assertEqual(preflight.extract_task(loader, "a")["origin"], [1., 2., .3])
            value["scene_metadata"]["initial_token"] = "wrong"
            save()
            with self.assertRaisesRegex(ValueError, "metadata/current"):
                preflight.extract_task(loader, "a")


class ActualSourceVerificationTests(unittest.TestCase):
    def test_changed_actual_sdk_adapter_or_map_bytes_rejected(self):
        temp_root = ROOT / "outputs/navsim_check_temp"
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            (root / "map.gpkg").write_bytes(b"original-map")
            settings = dict(sdk_root="fake-sdk", official_source_sha256="sdk-source",
                numerical_source_sha256="adapter-source", maps_runtime_root=str(root),
                map_source_sha256={"map.gpkg": hashlib.sha256(b"original-map").hexdigest()})
            with patch("ep_driver_navsim.launch.source_fingerprint", side_effect=[(1, "sdk-source"), (1, "adapter-source")]):
                runtime = preflight.verified_runtime(settings)
                self.assertTrue(runtime["packages"]["shapely"])
                self.assertTrue(runtime["geos_version"])
            for sources, reason in [([(1, "changed-sdk")], "SDK source"),
                                    ([(1, "sdk-source"), (1, "changed-adapter")], "adapter source")]:
                with patch("ep_driver_navsim.launch.source_fingerprint", side_effect=sources):
                    with self.assertRaisesRegex(ValueError, reason):
                        preflight.verified_runtime(settings)
            (root / "map.gpkg").write_bytes(b"map-replaced")
            with patch("ep_driver_navsim.launch.source_fingerprint", side_effect=[(1, "sdk-source"), (1, "adapter-source")]):
                with self.assertRaisesRegex(ValueError, "map source"):
                    preflight.verified_runtime(settings)

    def test_map_checker_rebuilds_official_api_and_delegates_identical_logic(self):
        from ep_driver_navsim import input_adapter
        seen = []
        def get_maps_api(*args):
            api = NS(identity=len(seen))
            seen.append((args, api))
            return api
        task = dict(token="a", map_name="map", map_version=preflight.MAP_VERSION,
                    origin=[1., 2., .3], public_roadblock_ids=["R"])
        contract = dict(maps_root="maps", desired_speed_mps=10., map_radius_m=160.)
        called = []
        def map_packet(api, origin, blocks, speed, radius):
            called.append((api, origin.tolist(), blocks, speed, radius))
            return ["lane"], [[0., 0.], [1., 0.]], None, None, None, None, None, [], dict(topology_correction_changed=False)
        factory = NS(get_maps_api=get_maps_api)
        with patch.dict(sys.modules, {"nuplan.common.maps.nuplan_map.map_factory": factory}), patch.object(input_adapter, "_map_packet", map_packet):
            preflight.check_map_task(task, contract)
            preflight.check_map_task(task, contract)
        self.assertEqual(seen[0][0], ("maps", preflight.MAP_VERSION, "map"))
        self.assertIsNot(seen[0][1], seen[1][1])
        self.assertEqual(called[0][1:], ([1., 2., .3], ["R"], 10., 160.))


if __name__ == "__main__":
    unittest.main()
