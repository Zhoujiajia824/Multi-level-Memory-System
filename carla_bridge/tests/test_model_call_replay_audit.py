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
    audit = rows(tmp_path / "planner.jsonl")[-1]["trajectory_visualization"]
    assert audit["enabled"] and audit["error"]["type"] == "RuntimeError"
    loop.recorder = Mock()
    loop._record_tick(loop.scenario.ego_vehicle)
    assert loop.recorder.on_tick.call_args.kwargs["overlay"]["trajectory_visualization"] == audit
    loop.scenario.ego_vehicle.apply_control.assert_not_called()
    loop.trajectory_renderer.draw.assert_not_called()


@pytest.mark.parametrize("accepted", [True, False])
def test_visualization_audit_is_saved_with_the_same_planner_sequence(tmp_path, accepted):
    from copy import deepcopy
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop.config.data["output_dir"] = tmp_path
    display = {"enabled": True, "style_version": "fixture", "drawn_candidates": 6,
               "candidate_pool_count": 40, "execution_drawn": accepted,
               "rejected_drawn": not accepted, "groups": [{"id": "left", "count": 12}]}
    loop.trajectory_renderer = Mock()
    loop.trajectory_renderer.draw.return_value = display
    snapshot = {"source": "fixture", "candidates": []}
    loop.pdm_planner = NS(last_visualization_snapshot=snapshot)
    decision = {"trajectory": [{"x": 0., "y": 0.}, {"x": 3., "y": 0.}],
                "planner_diagnostics": {"selected_id": "center:CRUISE"}}
    before = deepcopy(decision)
    validation = {"accepted": accepted}
    loop._audit_pdm_plan(decision, validation, "control_update")
    journal = rows(tmp_path / "planner.jsonl")
    assert len(journal) == 1
    assert journal[0]["sequence"] == 1
    assert journal[0]["trajectory_visualization"] == display
    assert loop._last_trajectory_visualization_audit == display
    assert journal[0]["validation"] == validation
    assert journal[0]["trajectory"] == decision["trajectory"]
    assert decision == before
    call = loop.trajectory_renderer.draw.call_args
    assert call.args[0] is snapshot and call.args[1] is decision
    assert call.kwargs["accepted"] is accepted
    loop.scenario.ego_vehicle.apply_control.assert_not_called()
    loop.carla_client.tick.assert_not_called()


def test_latest_trajectory_legend_overlay_is_a_detached_planner_display_audit(tmp_path):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop.config.data["output_dir"] = tmp_path
    display = {"enabled": True, "style": "paper", "drawn_candidates": 8,
               "candidate_pool_count": 40, "merged_candidates": 20, "hidden_candidates": 12}
    loop.trajectory_renderer = Mock()
    loop.trajectory_renderer.draw.return_value = display
    loop._audit_pdm_plan({"planner_diagnostics": {"selected_id": "left:CRUISE"}},
                         {"accepted": True}, "control_update")
    original = dict(display)
    display["drawn_candidates"] = 999
    loop.recorder = Mock()
    loop._record_tick(loop.scenario.ego_vehicle)
    overlay = loop.recorder.on_tick.call_args.kwargs["overlay"]
    assert overlay["selected_id"] == "left:CRUISE"
    assert overlay["trajectory_visualization"] == original
    overlay["trajectory_visualization"]["hidden_candidates"] = 999
    assert loop._last_trajectory_visualization_audit == original
    assert rows(tmp_path / "planner.jsonl")[-1]["trajectory_visualization"] == original
    loop.scenario.ego_vehicle.apply_control.assert_not_called()
    loop.carla_client.tick.assert_not_called()


def test_video_projection_snapshot_and_ego_mask_are_detached_from_planning(tmp_path):
    from copy import deepcopy
    from carla_bridge.tests.test_route_acceptance import fake_loop
    from carla_bridge.visualization.trajectory_renderer import DEFAULTS, DISPLAY_COLORS
    loop = fake_loop()
    loop.config.data["output_dir"] = tmp_path
    paths = {"style": "paper", "paths": [{"kind": "selected", "rgb": list(DISPLAY_COLORS["selected"]),
             "points_world": [[2., 3., .156], [5., 3., .156]], "width_m": DEFAULTS["selected_thickness"]}]}
    original = deepcopy(paths)
    loop.trajectory_renderer = NS(last_display_snapshot=paths)
    loop._last_trajectory_visualization_audit = {"enabled": True, "style": "paper"}
    loop.recorder = Mock()
    ego = loop.scenario.ego_vehicle
    transform = ego.get_transform.return_value
    ego.bounding_box.get_world_vertices.return_value = [NS(x=1., y=2., z=3.), NS(x=4., y=5., z=6.)]
    loop._record_tick(ego)
    overlay = loop.recorder.on_tick.call_args.kwargs["overlay"]
    assert overlay["trajectory_display"] == {**original, "ego_box_world": [[1., 2., 3.], [4., 5., 6.]]}
    ego.bounding_box.get_world_vertices.assert_called_once_with(transform)
    overlay["trajectory_display"]["paths"][0]["points_world"][0][0] = 999.
    assert paths == original
    assert "trajectory_display" not in loop._last_trajectory_visualization_audit
    assert not (tmp_path / "planner.jsonl").exists()
    ego.apply_control.assert_not_called()
    loop.carla_client.tick.assert_not_called()


def test_live_display_without_control_journal_still_uses_exact_server_frame_and_plan_metadata(tmp_path):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop.config.data["run_id"] = "live-frame-fixture"
    loop._control_audit_path = None
    loop.metrics = None
    loop._world_clock_audit = Mock(return_value={"frame": 4321, "elapsed_seconds": 57.2})
    loop.recorder = Mock()
    loop._elapsed = 5.7
    loop._last_control_command = {"planner_sequence": 19}
    loop._last_plan_diagnostics = {"selected_id": "left:passing"}
    loop._last_trajectory_visualization_audit = {"enabled": False}
    ego = loop.scenario.ego_vehicle
    loop._record_tick(ego)
    request = loop.recorder.on_tick.call_args.kwargs
    assert request["frame_id"] == 4321 and request["sim_time_s"] == 5.7
    assert request["overlay"]["run_id"] == "live-frame-fixture"
    assert request["overlay"]["planner_sequence"] == 19
    assert request["overlay"]["selected_id"] == "left:passing"
    assert "trajectory_display" not in request["overlay"]
    loop._world_clock_audit.assert_called_once()
    ego.get_transform.assert_not_called()
    ego.apply_control.assert_not_called()
    loop.carla_client.tick.assert_not_called()


def test_same_frame_overlay_keeps_model_consultation_identity_without_requerying_model(tmp_path):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop._world_clock_audit = Mock(return_value={"frame": 512})
    loop.recorder = Mock()
    loop._model_call_audit = NS(consultation_count=3,
        latest={"call_id": "consultation-3", "query": {"reason": "blocked_lane"}})
    loop._record_tick(loop.scenario.ego_vehicle)
    overlay = loop.recorder.on_tick.call_args.kwargs["overlay"]
    assert overlay["model_call_count"] == 3
    assert overlay["latest_call_id"] == "consultation-3"
    assert overlay["latest_trigger"] == "blocked_lane"
    loop.loop.step.assert_not_called()
    loop.cameras.capture.assert_not_called()
    loop.carla_client.tick.assert_not_called()


def display_setup_loop(tmp_path, monkeypatch, *, live_enabled=True, record_video=False):
    """Use real setup/cleanup flow with no simulator, inference, or HTTP service."""
    from carla_bridge import closed_loop as closed
    from carla_bridge.planning import pdm_planner
    from carla_bridge.video import recorder as video
    from carla_bridge.visualization import live_view
    from src.vla_memory.common.config import Config

    config = Config(overrides={"output_dir": str(tmp_path), "carla": {
        "planning": {"backend": "pdm_closed", "semantic_influence": False, "model_inference": False},
        "video": {"enabled": record_video}, "replay": {"enabled": False},
        "visualization": {"live": {"enabled": live_enabled}, "trajectories": {"enabled": False}}}})
    world, ego, client = Mock(), Mock(), Mock()
    scenario = NS(world=world, ego_vehicle=ego, cfg={}, route=Mock(), setup=Mock(),
                  destroy=Mock(return_value=True), scenario_name=lambda: "display_setup")
    monkeypatch.setattr(closed, "CarlaClient", Mock(return_value=client))
    monkeypatch.setattr(closed.ScenarioManager, "from_yaml", Mock(return_value=scenario))
    monkeypatch.setattr(pdm_planner, "validate_dependencies", Mock())
    monkeypatch.setattr(pdm_planner, "CarlaPDMPlanner", Mock(return_value=Mock()))
    monkeypatch.setattr(closed, "TrajectoryTracker", Mock(return_value=Mock()))
    metrics = Mock()
    metrics.summary.return_value = {}
    monkeypatch.setattr(closed, "MetricsCollector", Mock(return_value=metrics))
    live, recorder = Mock(), Mock()
    live.start.return_value = "http://127.0.0.1:8765/"
    live_factory, camera_factory = Mock(return_value=live), Mock(return_value=recorder)
    monkeypatch.setattr(live_view, "LiveTrajectoryView", live_factory)
    monkeypatch.setattr(video, "VideoRecorder", camera_factory)
    loop = closed.ClosedLoop(config, "unused", mode="memory_off")
    loop._bind_pdm_controller = Mock()
    loop._setup_collision_sensor = Mock()
    loop._write_control_context = Mock()
    loop._write_report = Mock()
    return NS(loop=loop, live=live, recorder=recorder, live_factory=live_factory,
              camera_factory=camera_factory, metrics=metrics, client=client, scenario=scenario)


def fail_display_directory(monkeypatch, error):
    from pathlib import Path
    mkdir = Path.mkdir

    def fail(path, *args, **kwargs):
        if path.name == "carla_videos":
            raise error
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail)


def test_both_display_sinks_off_allocate_neither_http_service_nor_display_camera(tmp_path, monkeypatch):
    state = display_setup_loop(tmp_path, monkeypatch, live_enabled=False, record_video=False)
    try:
        state.loop.setup()
        assert state.loop.live_view is None and state.loop.recorder is None
        state.live_factory.assert_not_called()
        state.camera_factory.assert_not_called()
        assert state.loop.metrics is state.metrics
        assert state.client.tick.call_count == 15  # Existing traffic warmup only.
    finally:
        state.loop.close()
    state.scenario.destroy.assert_called_once()
    state.client.restore.assert_called_once()


@pytest.mark.parametrize("phase", ["construct", "start"])
def test_optional_live_service_failure_cannot_abort_closed_loop_setup(tmp_path, monkeypatch, phase):
    state = display_setup_loop(tmp_path, monkeypatch)
    error = RuntimeError("optional live service %s failed" % phase)
    if phase == "construct":
        state.live_factory.side_effect = error
    else:
        state.live.start.side_effect = error
    try:
        state.loop.setup()
        assert state.loop.live_view is None and state.loop.recorder is None
        assert state.loop.metrics is state.metrics
        state.camera_factory.assert_not_called()
        if phase == "start":
            state.live.close.assert_called_once()
        else:
            state.live.close.assert_not_called()
    finally:
        state.loop.close()
    state.client.restore.assert_called_once()


@pytest.mark.parametrize("phase", ["mkdir", "construct", "start", "start_cleanup_error"])
def test_live_only_display_camera_failure_cleans_optional_sinks_and_continues(tmp_path, monkeypatch, phase):
    state = display_setup_loop(tmp_path, monkeypatch)
    error = RuntimeError("optional display camera %s failed" % phase)
    if phase == "mkdir":
        fail_display_directory(monkeypatch, error)
    elif phase == "construct":
        state.camera_factory.side_effect = error
    else:
        state.recorder.start.side_effect = error
        if phase == "start_cleanup_error":
            state.recorder.flush.side_effect = RuntimeError("partial display cleanup failed")
    try:
        state.loop.setup()
        assert state.loop.live_view is None and state.loop.recorder is None
        assert state.loop.metrics is state.metrics
        state.live.close.assert_called_once()
        if phase in ("start", "start_cleanup_error"):
            state.recorder.flush.assert_called_once()
        else:
            state.recorder.flush.assert_not_called()
        if phase == "mkdir":
            state.camera_factory.assert_not_called()
        assert state.client.tick.call_count == 15
        # Optional display failure did not alter setup's planner or controller.
        assert state.loop.pdm_planner is not None and state.loop.tracker is not None
    finally:
        state.loop.close()
    state.live.close.assert_called_once()
    state.client.restore.assert_called_once()


@pytest.mark.parametrize("phase", ["mkdir", "construct", "start"])
def test_required_recording_failure_still_propagates_and_is_cleaned_on_close(tmp_path, monkeypatch, phase):
    state = display_setup_loop(tmp_path, monkeypatch, record_video=True)
    error = RuntimeError("required recording %s failed" % phase)
    if phase == "mkdir":
        fail_display_directory(monkeypatch, error)
    elif phase == "construct":
        state.camera_factory.side_effect = error
    else:
        state.recorder.start.side_effect = error
    try:
        with pytest.raises(RuntimeError, match="required recording"):
            state.loop.setup()
    finally:
        state.loop.close()
    state.live.close.assert_called_once()
    if phase == "start":
        state.recorder.flush.assert_called_once()
    state.scenario.destroy.assert_called_once()
    state.client.restore.assert_called_once()


@pytest.mark.parametrize("snapshot", [None, {}])
def test_missing_or_unmaskable_video_snapshot_keeps_recording(tmp_path, snapshot):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop.trajectory_renderer = NS(last_display_snapshot=snapshot)
    loop._last_trajectory_visualization_audit = {"enabled": True, "style": "paper"}
    loop.recorder = Mock()
    ego = loop.scenario.ego_vehicle
    ego.bounding_box.get_world_vertices.side_effect = RuntimeError("bbox unavailable")
    loop._record_tick(ego)
    overlay = loop.recorder.on_tick.call_args.kwargs["overlay"]
    if snapshot is None:
        assert "trajectory_display" not in overlay
        ego.get_transform.assert_not_called()
    else:
        assert overlay["trajectory_display"] == {}
    ego.apply_control.assert_not_called()


def test_noncopyable_optional_projection_snapshot_cannot_stop_control_tick(tmp_path):
    from carla_bridge.tests.test_route_acceptance import fake_loop

    class InvalidDisplay(dict):
        def __deepcopy__(self, memo):
            raise RuntimeError("invalid optional display")

    loop = fake_loop()
    loop.trajectory_renderer = NS(last_display_snapshot=InvalidDisplay(paths=[]))
    loop._last_trajectory_visualization_audit = {"enabled": True, "style": "paper"}
    loop.recorder = Mock()
    loop._record_tick(loop.scenario.ego_vehicle)
    assert "trajectory_display" not in loop.recorder.on_tick.call_args.kwargs["overlay"]
    loop.scenario.ego_vehicle.apply_control.assert_not_called()


@pytest.mark.parametrize("latest_audit", [{"enabled": False},
    {"enabled": True, "error": {"type": "RuntimeError", "message": "new plan transform unavailable"}}])
def test_failed_or_disabled_latest_drawing_cannot_project_a_stale_previous_plan(tmp_path, latest_audit):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop.trajectory_renderer = NS(last_display_snapshot={"style": "paper", "paths": [{"old": "plan"}]})
    loop._last_trajectory_visualization_audit = latest_audit
    loop.recorder = Mock()
    loop._record_tick(loop.scenario.ego_vehicle)
    assert "trajectory_display" not in loop.recorder.on_tick.call_args.kwargs["overlay"]
    loop.scenario.ego_vehicle.get_transform.assert_not_called()
    loop.scenario.ego_vehicle.apply_control.assert_not_called()


@pytest.mark.parametrize("bad_display", [None, {"drawn_segments": float("nan")}, {"object": object()}])
def test_malformed_optional_visualization_audit_does_not_reject_a_plan(tmp_path, bad_display):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop.config.data["output_dir"] = tmp_path
    loop.trajectory_renderer = Mock()
    loop.trajectory_renderer.draw.return_value = bad_display
    loop._audit_pdm_plan({"planner_diagnostics": {}}, {"accepted": True}, "control_update")
    audit = rows(tmp_path / "planner.jsonl")[-1]
    assert audit["validation"]["accepted"] is True
    assert audit["trajectory_visualization"]["error"]["type"] in ("TypeError", "ValueError")
    loop.scenario.ego_vehicle.apply_control.assert_not_called()


def test_disabled_visualization_is_explicit_in_planner_journal(tmp_path):
    from carla_bridge.tests.test_route_acceptance import fake_loop
    loop = fake_loop()
    loop.config.data["output_dir"] = tmp_path
    loop._audit_pdm_plan({"planner_diagnostics": {}}, {"accepted": True}, "control_update")
    assert rows(tmp_path / "planner.jsonl")[-1]["trajectory_visualization"] == {"enabled": False}
    loop.scenario.ego_vehicle.get_transform.assert_not_called()
