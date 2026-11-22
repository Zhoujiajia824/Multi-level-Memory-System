"""Targeted offline checkpoint/archive regressions: no model, SDK or scoring calls."""
import json
import os
from pathlib import Path
import sys
import tempfile
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ep_driver_navsim import worker
from ep_driver_navsim.archive import read_input_packet, read_json, serialize_retrieval, write_input_packet
from ep_driver_navsim.launch import source_fingerprint
from ep_driver_navsim.protocol import digest, horizon_contract, relative_asset_path, write_json
from ep_driver_navsim import protocol
from ep_driver_navsim.model import StructuredServerVisionClient, structured_response_format
from ep_driver_navsim.resume import (COMPLETION_FILE, begin_sample, build_run_identity,
                                    load_completed_sample, sample_identity, _local_asset)
from nine_grid.contracts import ModelAnswer, PlanResult, PredictionBundle, Representations, RetrievalResult
from nine_grid.paths import filesystem_path
from nine_grid.models import CommunicationFailure, _goal_reference, parse_answer
from nine_grid.decision_protocol import LATERAL_ACTIONS, LONGITUDINAL_ACTIONS
from check_navsim_integration import packet_fixture

TEMP_ROOT = ROOT / "outputs/navsim_check_temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class ResumeStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.root = Path(self.temp.name).resolve()
        # tempfile's cleanup must also traverse the >MAX_PATH regression tree.
        # Keep self.root ordinary so the worker still receives the mixed form.
        self.temp.name = str(filesystem_path(self.temp.name))
        self.addCleanup(self.temp.cleanup)
        self.sample = self.root / "sample"
        self.sample.mkdir()
        self.packet = packet_fixture()
        for row in self.packet["camera_images"]:
            path = self.sample / "cameras" / (row["view"] + ".jpg")
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (8, 6)).save(path)
            row["path"] = str(path)
        self.model_dir = self.root / "dino"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        (self.model_dir / "model.safetensors").write_bytes(b"offline fingerprint fixture")
        self.config = worker.planning_config(json.loads((ROOT / "config/parameters.json").read_text(
            encoding="utf-8")), self.model_dir)
        self.settings = dict(dino_model=str(self.model_dir), dino_device="cpu", fast="off",
            resume=True, retrieve=False, numerical_source_sha256=source_fingerprint(ROOT, "src")[1],
            config_file_sha256=dict(parameters="fixture-parameters", prompts_config="fixture-prompts"),
            model_revision=None, model_service={})
        self.prompts = dict(fast="fixture system prompt", reflection="fixture reflection prompt")
        self.model = dict(model="offline fixture", base_url="http://localhost:1234/v1",
                          api_key_env="TEST_FAST_SECRET")
        self.pipeline = worker.NumericalPipeline.__new__(worker.NumericalPipeline)
        self.pipeline.config = self.config
        self.pipeline.settings = self.settings
        self.pipeline.resume_enabled = True
        self.pipeline.memory_snapshot_id = "fixture-memory"
        self.pipeline.records = []
        self.pipeline.extractor = NS(signature="test:offline-fixture")
        self.pipeline.fast = NS(enabled=False)
        self.update_identity()
        self.calls = []

    def update_identity(self):
        self.pipeline.run_identity = build_run_identity(self.settings, self.config, self.model,
                                                        self.prompts, self.pipeline.memory_snapshot_id)

    def representation(self, scene, directory, *args):
        self.calls.append("representations")
        # Match the real representation builder, which emits extended Windows paths.
        directory = filesystem_path(directory)
        directory.mkdir(exist_ok=True)
        name = digest(scene.frame_id)[:20]
        paths = {key: directory / (name + suffix) for key, suffix in
                 dict(bev="_bev.png", feature="_feature.png", grid="_grid.npz", occupancy="_occupancy.npz").items()}
        for key in ("bev", "feature"):
            Image.new("RGB", (8, 6)).save(paths[key])
        for key in ("grid", "occupancy"):
            np.savez_compressed(paths[key], fixture=np.zeros((2, 2)))
        return Representations(scene.frame_id, 0., str(paths["bev"]), str(paths["feature"]),
            np.zeros((2, 2)), np.ones((2, 2)), {}, np.zeros(768),
            dino_signature="test:offline-fixture", grid_path=str(paths["grid"]))

    def context(self, scene, config, answer=None):
        times = np.arange(50) * .1
        trajectory = np.c_[scene.ego.position[0] + 5 * times, np.zeros(50),
                           np.zeros(50), np.full(50, 5.)]
        return scene, PredictionBundle(scene.frame_id, 0., times, {"car": trajectory}, {})

    def planner(self, config):
        def solve(scene, predictions, answer=None):
            self.calls.append("planner")
            times = np.arange(41) * .1
            states = np.c_[scene.ego.position[0] + 5 * times, np.zeros(41),
                           np.zeros(41), np.full(41, 5.)]
            return PlanResult(scene.frame_id, 0., True, states, np.zeros((40, 2)),
                              diagnostics=dict(execution_mode="normal", safety_feasible=True))
        return NS(solve=solve)

    def compute(self):
        answer = ModelAnswer(self.packet["identity"]["token"], 0., [], {}, {}, np.zeros(2), {}, "fixture")
        with patch.object(worker, "build_representations", side_effect=self.representation), \
                patch.object(worker, "perception_scene", side_effect=lambda scene, config: (scene, {})), \
                patch.object(worker, "build_prompt", return_value="fixture prompt"), \
                patch.object(worker, "parse_answer", return_value=answer), \
                patch.object(worker, "prepare_planning_context", side_effect=self.context), \
                patch.object(worker, "create_planner", side_effect=self.planner), \
                patch.object(worker, "evaluate", return_value={"fixture": True}), \
                patch.object(worker, "execution_audit", return_value={"alarm": False}), \
                patch.object(worker, "render_goal_overlay", return_value=Image.new("RGB", (8, 6))):
            return self.pipeline.compute(self.packet, self.sample)

    def test_complete_sample_reuses_without_representation_planner_or_model(self):
        first = self.compute()
        self.assertFalse(first["resume"]["reused"])
        self.assertEqual(self.calls, ["representations", "planner", "planner"])
        self.calls.clear()
        second = self.compute()
        self.assertTrue(second["resume"]["reused"])
        self.assertEqual(self.calls, [])
        np.testing.assert_array_equal(first["poses"], second["poses"])
        self.assertTrue((self.sample / "input_packet.json.gz").is_file())
        self.assertFalse((self.sample / "input_packet.json").exists())

    def test_independent_alarm_braking_retains_original_attention(self):
        answer = ModelAnswer(self.packet["identity"]["token"], 0., [{"id": "car"}], {}, {},
                             np.zeros(2), {}, "fixture")
        self.pipeline.fast = NS(enabled=True, call=Mock(return_value="fixture answer"))
        scene = worker.snapshot_from_packet(self.packet, self.config)
        emergency_plan = self.planner(self.config).solve(scene, self.context(scene, self.config)[1])
        emergency = Mock(return_value=emergency_plan)

        def planner(config):
            instance = self.planner(config)
            instance.emergency_brake = emergency
            return instance

        with patch.object(worker, "build_representations", side_effect=self.representation), \
                patch.object(worker, "perception_scene", return_value=(scene, {})), \
                patch.object(worker, "build_prompt", return_value="fixture prompt"), \
                patch.object(worker, "parse_answer", return_value=answer), \
                patch.object(worker, "prepare_planning_context", side_effect=self.context), \
                patch.object(worker, "create_planner", side_effect=planner), \
                patch.object(worker, "evaluate", return_value={}), \
                patch.object(worker, "execution_audit", side_effect=[{"alarm": True}, {"alarm": False}]), \
                patch.object(worker, "render_goal_overlay", return_value=Image.new("RGB", (8, 6))):
            self.pipeline.compute(self.packet, self.sample)
        emergency.assert_called_once()
        self.assertIs(emergency.call_args.kwargs["answer"], answer)
        self.assertEqual(emergency.call_args.args[2], "independent_predicted_execution_alarm")

    def test_missing_corrupt_or_unmarked_sample_recomputes(self):
        self.compute()
        marker = self.sample / COMPLETION_FILE
        original = marker.read_bytes()
        for corruption in (b"{partial", b"{}", b"[]"):
            marker.write_bytes(corruption)
            self.assertIsNone(load_completed_sample(self.sample, sample_identity(self.packet, self.pipeline.run_identity)))
        marker.write_bytes(original)
        (self.sample / "trajectory_bev.png").unlink()
        self.calls.clear()
        self.compute()
        self.assertEqual(self.calls, ["representations", "planner", "planner"])
        marker.unlink()
        self.assertIsNone(load_completed_sample(self.sample, sample_identity(self.packet, self.pipeline.run_identity)))

    def test_camera_same_path_modified_content_invalidates(self):
        self.compute()
        old_identity = sample_identity(self.packet, self.pipeline.run_identity)
        camera = Path(self.packet["camera_images"][0]["path"])
        camera.write_bytes(camera.read_bytes() + b"different pixels payload")
        self.assertNotEqual(old_identity, sample_identity(self.packet, self.pipeline.run_identity))
        self.assertIsNone(load_completed_sample(self.sample, old_identity))
        self.calls.clear()
        self.compute()
        self.assertIn("representations", self.calls)

    def test_numerical_binding_and_secrets(self):
        original = self.pipeline.run_identity
        with patch.dict(os.environ, TEST_FAST_SECRET="first-secret"):
            self.update_identity()
        with patch.dict(os.environ, TEST_FAST_SECRET="second-secret"):
            self.assertEqual(original, build_run_identity(self.settings, self.config, self.model,
                self.prompts, self.pipeline.memory_snapshot_id))
        self.assertNotIn("secret", json.dumps(original))
        for field in ("numerical_source_sha256", "dino_device"):
            changed = dict(self.settings, **{field: "changed"})
            self.assertNotEqual(original, build_run_identity(changed, self.config, self.model,
                self.prompts, self.pipeline.memory_snapshot_id))
        self.assertNotEqual(original, build_run_identity(self.settings, self.config, self.model,
            dict(self.prompts, fast="changed"), self.pipeline.memory_snapshot_id))
        self.assertNotEqual(original, build_run_identity(self.settings, self.config, self.model,
            self.prompts, "changed-memory"))
        (self.model_dir / "model.safetensors").write_bytes(b"different weights")
        self.update_identity()
        self.assertNotEqual(original, self.pipeline.run_identity)
        with self.assertRaisesRegex(ValueError, "credentials"):
            build_run_identity(self.settings, self.config, dict(self.model, api_key="do-not-store"),
                               self.prompts, self.pipeline.memory_snapshot_id)

    def test_model_revision_and_service_preprocessing_bound(self):
        enabled = dict(self.settings, fast="server")
        with self.assertRaisesRegex(ValueError, "model_revision"):
            build_run_identity(enabled, self.config, self.model, self.prompts, "memory")
        enabled["model_revision"] = "sha256:first-checkpoint"
        first = build_run_identity(enabled, self.config, self.model, self.prompts, "memory")
        for change in (dict(model_revision="sha256:second-checkpoint"),
                       dict(model_service={"max_pixels_per_image": 602112}),
                       dict(structured_model_output=True)):
            self.assertNotEqual(first, build_run_identity(dict(enabled, **change), self.config,
                self.model, self.prompts, "memory"))

    def schema_fixture(self, scene=None):
        scene = scene or worker.snapshot_from_packet(self.packet, self.config)
        reference = _goal_reference(scene, self.config)
        response_format = structured_response_format(scene, self.config, reference)
        payload = dict(scene=dict(description="当前道路", reason="当前可见状态"),
            decision=dict(lateral=dict(action="保持车道", reason="沿当前导航"),
                          longitudinal=dict(action="减速", reason="关注前车")),
            attention=[dict(id="car", reason="当前前车")],
            local_goal=dict(delta_s_m=10., d_m=0., reason="当前参考线前方引导点"))
        return scene, reference, response_format, payload

    def test_structured_schema_uses_current_ids_core_actions_and_original_limits(self):
        scene, reference, response_format, payload = self.schema_fixture()
        schema = response_format["json_schema"]["schema"]
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)
        self.assertTrue(response_format["json_schema"]["strict"])
        props = schema["properties"]
        self.assertEqual(props["attention"]["items"]["properties"]["id"]["enum"], ["car"])
        self.assertEqual(props["decision"]["properties"]["lateral"]["properties"]["action"]["enum"], list(LATERAL_ACTIONS))
        self.assertEqual(props["decision"]["properties"]["longitudinal"]["properties"]["action"]["enum"], list(LONGITUDINAL_ACTIONS))
        self.assertEqual(props["local_goal"]["properties"]["d_m"], {"type": "number"})
        self.assertEqual(props["local_goal"]["properties"]["delta_s_m"]["minimum"], reference["limits"]["minimum_delta_s_m"])
        answer = parse_answer(json.dumps(payload), scene, reference=reference, config=self.config)
        self.assertEqual(answer.attention[0]["id"], "car")
        self.assertEqual(answer.local_goal, {"delta_s_m": 10., "d_m": 0.})
        shortened = dict(reference, points=[[0., 0.], [40., 0.]], s_ego_m=30.)
        limited = structured_response_format(scene, self.config, shortened)
        self.assertAlmostEqual(limited["json_schema"]["schema"]["properties"]["local_goal"]["properties"]["delta_s_m"]["maximum"], 10.-1e-6)
        with self.assertRaisesRegex(ValueError, "no legal longitudinal"):
            structured_response_format(scene, self.config, dict(shortened, s_ego_m=39.))

    def test_empty_scene_schema_requires_empty_attention(self):
        scene = replace(worker.snapshot_from_packet(self.packet, self.config), actors=[])
        _, _, response_format, payload = self.schema_fixture(scene)
        schema = response_format["json_schema"]["schema"]
        Draft202012Validator.check_schema(schema)
        self.assertEqual(schema["properties"]["attention"]["maxItems"], 0)
        payload["attention"] = []
        Draft202012Validator(schema).validate(payload)

    def test_navsim_model_and_prediction_keep_bev_candidates_outside_lane_scope(self):
        from nine_grid.object_scope import scope_scene
        from nine_grid.planning_objects import planning_view
        scene = worker.snapshot_from_packet(self.packet, self.config)
        side = replace(scene.actors[0], id="bev_side", position=np.array([18., 8.]),
                       velocity=np.zeros(2), lane_id=None)
        scene = replace(scene, actors=scene.actors + [side])
        road, _ = scope_scene(scene, self.config)
        self.assertNotIn(side.id, {actor.id for actor in road.actors})
        perception, _ = worker.perception_scene(scene, self.config)
        _, reference, response_format, payload = self.schema_fixture(perception)
        ids = response_format["json_schema"]["schema"]["properties"]["attention"]["items"]["properties"]["id"]["enum"]
        self.assertIn(side.id, ids)
        scoped, prediction = worker.prepare_planning_context(scene, self.config)
        self.assertIn(side.id, prediction.trajectories)
        base, _, _ = planning_view(scoped, prediction, self.config)
        self.assertNotIn(side.id, {actor.id for actor in base.actors})
        payload["attention"] = [dict(id=side.id, reason="BEV内需关注的对象")]
        answer = parse_answer(json.dumps(payload), perception, reference=reference, config=self.config)
        guided, _, _ = planning_view(scoped, prediction, self.config, answer)
        self.assertIn(side.id, {actor.id for actor in guided.actors})

    def test_structured_transport_sends_nine_images_and_never_downgrades_request(self):
        _, _, response_format, payload = self.schema_fixture()
        raw = json.dumps(payload, ensure_ascii=False)
        create = Mock(return_value=NS(choices=[NS(finish_reason="stop", message=NS(content=raw))]))
        transport = NS(chat=NS(completions=NS(create=create)))
        client = StructuredServerVisionClient("fast", dict(self.model, backend="server", system_prompt="fixture system"),
                                              transport=transport)
        images = [row["path"] for row in self.packet["camera_images"]]
        images.append(images[0])  # The ninth input represents the current BEV.
        self.assertEqual(client.call_structured("same prompt", images, response_format), raw)
        request = create.call_args.kwargs
        self.assertEqual(request["response_format"], response_format)
        self.assertEqual(len(request["messages"][1]["content"]), 10)
        self.assertEqual(request["messages"][0]["content"], "fixture system")
        self.assertIs(client.client, transport)
        create.reset_mock()
        create.side_effect = ValueError("unsupported response_format with sensitive remote text")
        with self.assertRaises(CommunicationFailure) as captured:
            client.call_structured("same prompt", images, response_format)
        self.assertNotIn("sensitive", str(captured.exception))
        self.assertEqual(create.call_count, 1)
        self.assertEqual(create.call_args.kwargs["response_format"], response_format)
        self.assertIs(client.client, transport)
        with self.assertRaisesRegex(ValueError, "server backend"):
            StructuredServerVisionClient("fast", dict(self.model, backend="api"), transport=transport)

    def test_structured_output_does_not_repair_invalid_ids_goals_or_actions(self):
        scene, reference, _, original = self.schema_fixture()
        for identity in ("ego", "car_1"):
            payload = json.loads(json.dumps(original))
            payload["attention"][0]["id"] = identity
            with self.assertRaisesRegex(ValueError, "关注 ID"):
                parse_answer(json.dumps(payload), scene, reference=reference, config=self.config)
            self.assertEqual(payload["attention"][0]["id"], identity)
        changes = [dict(d_m=100.), dict(delta_s_m=True), dict(delta_s_m=1000.)]
        for change in changes:
            payload = json.loads(json.dumps(original))
            payload["local_goal"].update(change)
            with self.assertRaises(ValueError):
                parse_answer(json.dumps(payload), scene, reference=reference, config=self.config)
        payload = json.loads(json.dumps(original))
        payload["decision"]["lateral"]["action"] = "invented"
        with self.assertRaises(ValueError):
            parse_answer(json.dumps(payload), scene, reference=reference, config=self.config)

    def test_structured_worker_archives_request_schema_and_binds_resume(self):
        self.settings.update(fast="server", model_revision="fixture-checkpoint", structured_model_output=True)
        self.update_identity()
        model_call = Mock(return_value="fixture raw structured answer")
        self.pipeline.fast = NS(enabled=True, call_structured=model_call)
        first = self.compute()
        self.assertTrue(first["condition_audit"]["structured_model_output"])
        request_schema = read_json(self.sample / "model_response_format.json")
        self.assertEqual(model_call.call_args.args[2], request_schema)
        evidence = read_json(self.sample / "planning_evidence.json")
        self.assertEqual(evidence["model_response_format_sha256"], digest(request_schema))
        self.calls.clear()
        self.assertTrue(self.compute()["resume"]["reused"])
        self.assertEqual(model_call.call_count, 1)
        self.assertEqual(self.calls, [])

    def test_only_local_parse_errors_retain_detail(self):
        self.settings.update(fast="server", model_revision="fixture-revision")
        self.update_identity()
        self.pipeline.fast = NS(enabled=True, call=lambda *args: "fixture raw")
        failure = ValueError("关注 ID 不存在、不可用、自车或重复")
        with patch.object(worker, "parse_answer", side_effect=failure):
            with patch.object(worker, "build_representations", side_effect=self.representation), \
                    patch.object(worker, "perception_scene", side_effect=lambda scene, config: (scene, {})), \
                    patch.object(worker, "build_prompt", return_value="fixture prompt"), \
                    patch.object(worker, "prepare_planning_context", side_effect=self.context), \
                    patch.object(worker, "create_planner", side_effect=self.planner), \
                    patch.object(worker, "evaluate", return_value={}), \
                    patch.object(worker, "execution_audit", return_value={"alarm": False}), \
                    patch.object(worker, "render_goal_overlay", return_value=Image.new("RGB", (8, 6))):
                result = self.pipeline.compute(self.packet, self.sample)
        self.assertEqual(result["condition_audit"]["model_error_detail"], str(failure))
        self.assertFalse(result["condition_audit"]["model_answer_valid"])

    def test_model_failure_never_reused_but_model_success_is(self):
        self.settings.update(fast="api", model_revision="fixture-revision")
        self.update_identity()
        self.pipeline.fast = NS(enabled=True, call=lambda *args: "fixture raw model response")
        self.compute()
        self.calls.clear()
        self.assertTrue(self.compute()["resume"]["reused"])
        self.assertEqual(self.calls, [])
        begin_sample(self.sample)
        def fail(*args):
            raise RuntimeError("provider could expose a credential here")
        self.pipeline.fast.call = fail
        first = self.compute()
        self.assertFalse(first["condition_audit"]["model_answer_valid"])
        self.assertEqual(first["condition_audit"]["model_error"], "RuntimeError")
        self.assertFalse(read_json(self.sample / COMPLETION_FILE)["reusable"])
        self.calls.clear()
        self.compute()
        self.assertIn("representations", self.calls)

    def test_retrieval_condition_model_fallback_is_not_reusable(self):
        self.settings.update(fast="server", model_revision="fixture-checkpoint", retrieve=True)
        self.update_identity()
        def fail(*args):
            raise RuntimeError("offline fake failure")
        self.pipeline.fast = NS(enabled=True, call=fail)
        retrieved = RetrievalResult("anchor", records=[dict(id="memory", revision=1,
            source="highway-env", answer={"explanation": "fixture memory"})],
            snapshot_id=self.pipeline.memory_snapshot_id)
        with patch.object(worker, "retrieve", return_value=retrieved):
            result = self.compute()
        self.assertTrue(result["condition_audit"]["retrieval_enabled"])
        self.assertEqual(result["condition_audit"]["retrieved_memory_ids"], ["memory"])
        self.assertFalse(read_json(self.sample / COMPLETION_FILE)["reusable"])
        self.assertIsNone(load_completed_sample(self.sample, sample_identity(self.packet, self.pipeline.run_identity)))

    def test_four_second_actual_prediction_and_reference_grids(self):
        self.assertEqual(self.config["prediction"]["diagnostic_horizon_s"], 4.9)
        self.compute()
        evidence = read_json(self.sample / "planning_evidence.json")
        self.assertEqual(len(evidence["plan"]["states"]), 41)
        self.assertEqual(len(evidence["reference"]["states"]), 41)
        self.assertEqual(evidence["predictions"]["times"][40], 4.)
        self.assertEqual(evidence["reference_prediction_times_s"][40], 4.)
        self.assertEqual(evidence["horizon_contract"]["main_horizon_s"], 4.)
        self.assertEqual(evidence["horizon_contract"]["diagnostic_horizon_s"], 4.9)
        self.assertFalse(evidence["horizon_contract"]["extrapolation"])
        scene = worker.snapshot_from_packet(self.packet, self.config)
        _, prediction = self.context(scene, self.config)
        prediction.times = np.arange(40) * .1
        with self.assertRaisesRegex(ValueError, "configured TTC diagnostic grid"):
            horizon_contract(self.config, predictions=dict(prediction=prediction))

    def test_packet_archive_legacy_and_gzip_round_trip(self):
        path = self.root / "legacy"
        path.mkdir()
        write_json(path / "input_packet.json", self.packet)
        self.assertEqual(read_input_packet(path), self.packet)
        write_input_packet(path, self.packet)
        self.assertFalse((path / "input_packet.json").exists())
        self.assertEqual(read_input_packet(path / "input_packet.json"), self.packet)
        self.assertEqual(read_input_packet(path / "input_packet.json.gz"), self.packet)

    def test_failed_atomic_publication_preserves_previous_json(self):
        target = self.root / "atomic.json"
        write_json(target, {"previous": True})
        with patch.object(protocol.os, "replace", side_effect=OSError("simulated publication interruption")):
            with self.assertRaises(OSError):
                write_json(target, {"partial": False})
        self.assertEqual(read_json(target), {"previous": True})
        self.assertEqual(list(self.root.glob(".atomic.json.*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows extended-path contract")
    def test_extended_and_ordinary_paths_share_relative_asset_boundary(self):
        asset = Path(self.packet["camera_images"][0]["path"])
        extended_asset = filesystem_path(asset)
        extended_root = filesystem_path(self.sample)
        self.assertTrue(str(extended_asset).startswith("\\\\?\\"))
        expected = "cameras/" + asset.name
        self.assertEqual(relative_asset_path(extended_asset, self.sample), expected)
        self.assertEqual(relative_asset_path(asset, extended_root), expected)
        self.assertEqual(_local_asset(self.sample, expected), extended_asset)
        # Similar textual prefixes and extended absolute paths cannot escape.
        sibling = self.sample.with_name(self.sample.name + "-sibling") / "asset.json"
        for candidate in (sibling, filesystem_path(sibling), self.sample / "../outside.json"):
            with self.assertRaises(ValueError):
                relative_asset_path(candidate, extended_root)
        for name in ("../outside.json", str(filesystem_path(sibling))):
            with self.assertRaises(ValueError):
                _local_asset(self.sample, name)

    @unittest.skipUnless(os.name == "nt", "Windows MAX_PATH regression")
    def test_worker_completes_and_resumes_beyond_windows_max_path(self):
        # Exercise the original failure: ordinary output root, extended representation
        # filenames, compressed archives, atomic writes, completion and subsequent reuse.
        ordinary_sample = self.root / ("long-directory-" + "a" * 70) / ("unicode-目录-" + "b" * 70) / "sample"
        self.assertGreater(len(str(ordinary_sample / "planning_evidence.json")), 260)
        filesystem_path(ordinary_sample).mkdir(parents=True)
        for row in self.packet["camera_images"]:
            old = Path(row["path"])
            replacement = ordinary_sample / "cameras" / old.name
            protocol.write_bytes(replacement, old.read_bytes())
            row["path"] = str(replacement)
        self.sample = ordinary_sample
        first = self.compute()
        self.assertFalse(first["resume"]["reused"])
        evidence = read_json(self.sample / "planning_evidence.json")
        self.assertTrue(all(name.startswith("representations/") for name in evidence["representation_assets"].values()))
        self.assertTrue(filesystem_path(self.sample / COMPLETION_FILE).is_file())
        self.calls.clear()
        self.assertTrue(self.compute()["resume"]["reused"])
        self.assertEqual(self.calls, [])

    def test_interrupted_recompute_clears_old_completion(self):
        self.compute()
        self.pipeline.resume_enabled = False
        with patch.object(worker, "build_representations", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.pipeline.compute(self.packet, self.sample)
        self.assertFalse((self.sample / COMPLETION_FILE).exists())
        self.assertIsNone(load_completed_sample(self.sample, sample_identity(self.packet, self.pipeline.run_identity)))

    def test_real_numerical_backend_covers_main_and_diagnostic_horizons(self):
        # Real predictor + optimizer on a tiny synthetic packet; no DINO/model/SDK.
        scene = worker.snapshot_from_packet(self.packet, self.config)
        scoped, predictions = worker.prepare_planning_context(scene, self.config)
        plan = worker.create_planner(self.config).solve(scoped, predictions)
        reference = worker.create_planner(self.config).solve(scoped, predictions)
        checked = horizon_contract(self.config, dict(plan=plan, reference=reference),
                                   dict(prediction=predictions))
        self.assertTrue(plan.valid)
        self.assertTrue(reference.valid)
        self.assertEqual(checked["plans"]["plan"]["state_count"], 41)
        self.assertEqual(checked["plans"]["reference"]["horizon_s"], 4.)
        self.assertEqual(checked["predictions"]["prediction"]["main_state_count"], 41)
        self.assertEqual(checked["predictions"]["prediction"]["diagnostic_horizon_s"], 4.9)

    def test_retrieval_archive_keeps_prose_scores_assets_without_internal_tensor_copy(self):
        scene = worker.snapshot_from_packet(self.packet, self.config)
        reps = self.representation(scene, self.sample / "representations")
        record = dict(id="memory", revision=7, source="highway-env", answer={"explanation": "retain prose"},
            assets={"bev": "assets/bev.png"}, checksums={"assets/bev.png": "fixture-checksum"},
            _asset_root="readonly/library/memory", _representations=reps)
        diagnostic = dict(experience_id="memory", accepted=True, eligible=True,
                          channels={"grid": {"matches": [{"query_id": "car", "memory_id": "old_car"}]}})
        retrieved = RetrievalResult("anchor", records=[record], scores=[dict(id="memory", grid=.8)],
            diagnostics=dict(all_records=[diagnostic], candidates=[diagnostic], candidate_ids=["memory"]))
        archived = serialize_retrieval(retrieved, self.sample)
        row = archived["records"][0]
        self.assertNotIn("_representations", row)
        self.assertEqual(row["answer"], record["answer"])
        self.assertEqual(row["revision"], 7)
        self.assertEqual(archived["scores"], retrieved.scores)
        self.assertEqual(read_json(self.sample / "retrieval_diagnostics.json.gz"), retrieved.diagnostics)
        self.assertIs(record["_representations"], reps)
        self.assertIn("all_records", retrieved.diagnostics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
