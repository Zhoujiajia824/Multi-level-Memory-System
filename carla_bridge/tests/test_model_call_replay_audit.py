"""Evidence follows real client dispatch boundaries, without a remote model."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from carla_bridge.model_call_audit import ModelCallAudit
from src.vla_memory.common.model_call_audit import observe_model_requests
from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def response(text):
    return NS(choices=[NS(message=NS(content=text))])


def client(results):
    value = OpenAICompatibleVLMClient(retry_times=2, retry_interval_seconds=0,
                                     system_prompt="精确系统提示词")
    create = Mock(side_effect=results)
    value._client = NS(chat=NS(completions=NS(create=create)))
    return value, create


def test_scene_decision_attempts_keep_exact_text_and_portable_images(tmp_path):
    original = tmp_path / "bank" / "input.jpg"
    original.parent.mkdir()
    original.write_bytes(b"exact-input-image-bytes")
    run = tmp_path / "run"
    audit = ModelCallAudit(run)
    vlm, create = client([response('{"scene":1}'), TimeoutError("timeout"), response('{"decision":1}')])
    kf = {"sample_token": "token", "sim_time_s": 8.5, "carla_frame_id": 523,
          "image_path": str(original), "planning_constraints": {"planner_context": {"red": False}}}
    with audit.consultation(kf, {"reason": "planner_stalled"}, {"x": 12, "y": -3}) as evidence:
        vlm.understand_scene(str(original), "场景提示词", extra_context={"nav": "left"})
        result = vlm.decide("决策提示词", image_paths=[str(original)])
        evidence["record"] = {"raw_response": result, "decision_validation": {"accepted": True}}
    requests = rows(run / "model_requests.jsonl")
    starts = [row for row in requests if row["event"] == "request_start"]
    assert create.call_count == 3 == len(starts)
    assert [row["stage"] for row in starts] == ["scene", "decision", "decision"]
    assert [row["attempt"] for row in starts] == [1, 1, 2]
    assert starts[1]["request_id"] == starts[2]["request_id"] != starts[0]["request_id"]
    assert starts[0]["prompt"] == "场景提示词\n\n额外上下文:\nnav: left"
    assert starts[0]["system_prompt"] == "精确系统提示词"
    assert all(row["carla_frame_id"] == 523 and row["sim_time_s"] == 8.5 for row in requests)
    asset = run / starts[0]["image_paths"][0]
    assert asset.read_bytes() == original.read_bytes()
    assert "base64" not in (run / "model_requests.jsonl").read_text(encoding="utf-8")
    original.unlink()
    assert asset.is_file()
    consultations = rows(run / "model_calls.jsonl")
    assert len(consultations) == 2
    assert consultations[0]["call_id"] == consultations[1]["call_id"]
    assert consultations[-1]["record"]["raw_response"] == result
    assert consultations[-1]["status"] == "completed"


def test_dispatch_failure_and_interruption_are_distinct_from_capture_only(tmp_path, monkeypatch):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    audit = ModelCallAudit(tmp_path / "run")
    vlm, _ = client([TimeoutError("secret-value"), TimeoutError("secret-value")])
    monkeypatch.setenv(vlm.api_key_env, "secret-value")
    with pytest.raises(RuntimeError):
        with audit.consultation({"image_path": str(image)}, {"reason": "interaction"}, {}) as evidence:
            vlm.decide("input", image_paths=[str(image)])
    request_rows = rows(audit.run_dir / "model_requests.jsonl")
    assert len(request_rows) == 4
    assert all("secret-value" not in row.get("error", "") for row in request_rows)
    assert rows(audit.run_dir / "model_calls.jsonl")[-1]["status"] == "failed"
    assert "secret-value" not in (audit.run_dir / "model_calls.jsonl").read_text(encoding="utf-8")
    with pytest.raises(KeyboardInterrupt):
        with audit.consultation({}, {}, {}):
            raise KeyboardInterrupt()
    assert rows(audit.run_dir / "model_calls.jsonl")[-1]["status"] == "interrupted"
    # A consultation can end before any API dispatch (e.g. feature extraction fails).
    assert len(rows(audit.run_dir / "model_requests.jsonl")) == 4


def test_failed_observer_never_repeats_a_successful_inference(tmp_path):
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    vlm, create = client([response("actual response"), response("second response")])
    observer = Mock(side_effect=OSError("disk full"))
    with observe_model_requests(observer):
        assert vlm.decide("prompt", image_paths=[str(image)]) == "actual response"
    assert create.call_count == 1
    # Context observer does not escape its scope.
    assert vlm.decide("next prompt", image_paths=[]) == "second response"
    assert observer.call_count == 2


def test_closed_loop_integration_links_trigger_pose_and_model_output(tmp_path):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop._model_call_audit = ModelCallAudit(tmp_path)
    loop.loop.step.return_value = {"current_scene": {"scene_description": "前方施工"},
        "decision_validation": {"accepted": True}, "raw_response": "real-output-fixture"}
    kf = {"sample_token": "sample", "carla_frame_id": 88, "sim_time_s": 5.5}
    record = loop._run_cognition(kf, {"query": True, "reason": "blocked_lane"})
    end = rows(tmp_path / "model_calls.jsonl")[-1]
    assert end["carla_frame_id"] == 88
    assert end["query"]["reason"] == "blocked_lane"
    assert end["ego"]["speed_mps"] == 5
    assert end["record"] == record
    assert end["status"] == "accepted"
    assert loop.carla_client.tick.call_count == 0


def test_pose_audit_failure_does_not_cancel_inference(tmp_path):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop._model_call_audit = ModelCallAudit(tmp_path)
    loop.scenario.ego_vehicle.get_transform.side_effect = RuntimeError("telemetry unavailable")
    loop.loop.step.return_value = {"decision_validation": {"accepted": True}}
    assert loop._run_cognition({}, {}) == loop.loop.step.return_value
    assert rows(tmp_path / "model_calls.jsonl")[-1]["ego"]["error"].startswith("RuntimeError")


def test_visualization_transform_failure_cannot_reject_valid_plan(tmp_path):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop.config.data["output_dir"] = tmp_path
    loop.trajectory_renderer = Mock()
    loop.scenario.ego_vehicle.get_transform.side_effect = RuntimeError("visual telemetry failed")
    loop._audit_pdm_plan({"planner_diagnostics": {}}, {"accepted": True}, "control_update")
    assert rows(tmp_path / "planner.jsonl")[-1]["validation"]["accepted"] is True
    loop.scenario.ego_vehicle.apply_control.assert_not_called()
    loop.trajectory_renderer.draw.assert_not_called()
