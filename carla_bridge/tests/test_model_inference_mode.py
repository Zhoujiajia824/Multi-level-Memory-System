"""Pure PDM bypass: no fabricated model output and unchanged physics scheduling."""
from copy import deepcopy
import builtins
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.inference_mode import model_inference_enabled
from carla_bridge.run_session import RunSession
from carla_bridge.tests.test_route_acceptance import fake_loop
from src.vla_memory.common.config import Config


def bypass_planning(**overrides):
    return {"backend": "pdm_closed", "semantic_influence": False,
            "model_inference": False, **overrides}


def trajectory():
    return {
        "trajectory": [{"t": i / 10, "x": i / 2, "y": 0.0, "optional_v": 5.0}
                       for i in range(41)],
        "target_speed": 5.0,
        "selected_lateral": "FOLLOW_ROUTE",
        "planner_diagnostics": {"selected_id": "center:CRUISE",
                                "selected_lateral": "FOLLOW_ROUTE"},
    }


def bypass_loop(tmp_path, *, complete_at=None, duration=1.0):
    loop = fake_loop(complete_at=complete_at, duration=duration)
    loop.config.data["output_dir"] = tmp_path
    loop.config.data["carla"]["planning"] = bypass_planning()
    loop.model_inference = False
    loop.mode = "memory_off"
    loop.planner_backend = "pdm_closed"
    loop.semantic_influence = False
    loop.loop = None
    loop.cameras = None
    loop.history = None
    loop.perception = None
    loop.kf_builder = None
    loop._raw_capture = Mock(side_effect=AssertionError("pure PDM captured model images"))
    loop._replan_cycle = Mock(side_effect=AssertionError("pure PDM entered cognition cycle"))
    loop.pdm_planner = Mock()
    loop.pdm_planner.plan.side_effect = lambda *args, **kwargs: deepcopy(trajectory())
    return loop


def test_existing_modes_keep_model_inference_by_default():
    assert model_inference_enabled({}, "memory_on") is True
    assert model_inference_enabled({"semantic_influence": False}, "memory_off") is True
    assert model_inference_enabled(bypass_planning(model_inference=True), "memory_on") is True
    assert model_inference_enabled(bypass_planning(), "memory_off") is False


def test_fresh_pure_pdm_entry_import_does_not_import_model_or_memory_implementations():
    # A new process prevents cached modules from concealing eager imports.
    code = r'''
import builtins
import json
import sys
sys.path[:] = json.loads(sys.argv[1])
original_import = builtins.__import__
blocked = ("torch", "transformers", "faiss", "src.vla_memory.pipeline.online_loop",
           "src.vla_memory.perception.dinov2_extractor",
           "src.vla_memory.perception.openai_compatible_client",
           "src.vla_memory.perception.scene_understanding",
           "src.vla_memory.decision", "src.vla_memory.memory",
           "carla_bridge.memory_adapter.loop_runner")
def guarded(name, *args, **kwargs):
    if name.startswith(blocked):
        raise AssertionError("Pure PDM imported model or memory implementation: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from carla_bridge import run_carla_demo
from src.vla_memory.common.config import Config
cfg = Config(overrides={"carla": {"planning": {"backend": "pdm_closed",
    "semantic_influence": False, "model_inference": False}}})
closed = run_carla_demo.ClosedLoop(cfg, "unused", mode="memory_off")
assert closed.model_inference is False and closed.loop is None
assert not any(name.startswith(blocked) for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code, json.dumps(sys.path)],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_bypass_setup_skips_cameras_and_pipeline_but_preserves_planner_and_metrics(tmp_path, monkeypatch):
    from carla_bridge import closed_loop as module
    from carla_bridge.planning import pdm_planner

    cfg = Config(overrides={"output_dir": str(tmp_path),
                            "carla": {"planning": bypass_planning(), "video": {"enabled": False}}})
    world, ego, client = Mock(), Mock(), Mock()
    scenario = SimpleNamespace(world=world, ego_vehicle=ego, cfg={}, route=Mock(),
                               setup=Mock(), destroy=Mock(return_value=True),
                               scenario_name=Mock(return_value="bypass_setup"))
    monkeypatch.setattr(module, "CarlaClient", Mock(return_value=client))
    monkeypatch.setattr(module.ScenarioManager, "from_yaml", Mock(return_value=scenario))
    forbidden = Mock(side_effect=AssertionError("pure PDM initialized cognition resources"))
    for name in ("CameraManager", "HistoryBuffer", "PerceptionProvider", "KeyframeBuilder"):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(pdm_planner, "validate_dependencies", Mock())
    planner = Mock()
    monkeypatch.setattr(pdm_planner, "CarlaPDMPlanner", Mock(return_value=planner))
    metrics = Mock()
    metrics.summary.return_value = {}
    monkeypatch.setattr(module, "MetricsCollector", Mock(return_value=metrics))
    tracker = Mock()
    monkeypatch.setattr(module, "TrajectoryTracker", Mock(return_value=tracker))
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in ("carla_bridge.memory_adapter.loop_runner", "src.vla_memory.pipeline.online_loop"):
            raise AssertionError("pure PDM imported the cognition pipeline")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    loop = ClosedLoop(cfg, "unused", mode="memory_off")
    loop._bind_pdm_controller = Mock()
    loop._setup_collision_sensor = Mock()
    loop._write_control_context = Mock()
    loop._make_output_path = forbidden
    try:
        loop.setup()
        assert loop.loop is loop.cameras is loop.perception is loop.history is loop.kf_builder is None
        assert loop.pdm_planner is planner and loop.tracker is tracker and loop.metrics is metrics
        loop._setup_collision_sensor.assert_called_once_with(world, ego)
        forbidden.assert_not_called()
        client.connect.assert_called_once()
        scenario.setup.assert_called_once()
    finally:
        loop.close()
    scenario.destroy.assert_called_once()
    client.restore.assert_called_once()


def test_legacy_semantic_off_still_calls_cognition_by_default(tmp_path):
    loop = fake_loop(duration=3.0)
    loop.config.data["output_dir"] = tmp_path
    loop.planner_backend = "pdm_closed"
    loop.semantic_influence = False
    loop.scenario.route._route_waypoints = []
    assert loop.model_inference is True
    loop._replan_cycle()
    loop.loop.step.assert_called_once()
    loop.cameras.capture.assert_called_once()


@pytest.mark.parametrize("invalid", [None, 0, 1, "off", "false", [], {}])
def test_model_switch_requires_a_boolean(invalid):
    with pytest.raises(ValueError, match="model_inference"):
        model_inference_enabled(bypass_planning(model_inference=invalid), "memory_off")


@pytest.mark.parametrize("planning,mode", [
    (bypass_planning(backend="vlm_trajectory"), "memory_off"),
    (bypass_planning(backend="unknown"), "memory_off"),
    (bypass_planning(semantic_influence=True), "memory_off"),
    (bypass_planning(semantic_influence="off"), "memory_off"),
    (bypass_planning(), "memory_on"),
])
def test_inconsistent_modes_fail_before_runtime_or_output_creation(tmp_path, planning, mode):
    cfg = Config(overrides={"output_dir": str(tmp_path / "runs"),
                            "carla": {"planning": planning}})
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("scenario_name: validation\n", encoding="utf-8")
    with pytest.raises(ValueError):
        model_inference_enabled(planning, mode)
    with pytest.raises(ValueError):
        ClosedLoop(cfg, str(scenario), mode=mode)
    with pytest.raises(ValueError):
        RunSession(cfg, scenario, mode)
    assert not (tmp_path / "runs").exists()


def test_bypass_replans_every_point_two_seconds_without_cognition_or_extra_ticks(tmp_path):
    loop = bypass_loop(tmp_path, complete_at=1.0, duration=1.0)
    loop.metrics = Mock()
    loop.recorder = Mock()
    loop.event_scheduler = Mock()
    loop.run()

    assert loop.run_status == "route_completed"
    assert loop._elapsed == pytest.approx(1.0)
    assert loop.carla_client.tick.call_count == 10
    assert loop.metrics.on_tick.call_count == 10
    assert loop.recorder.on_tick.call_count == 10
    assert loop.event_scheduler.tick.call_count == 10
    loop._replan_cycle.assert_not_called()
    loop._raw_capture.assert_not_called()
    assert loop.pdm_planner.plan.call_count == 5
    for call in loop.pdm_planner.plan.call_args_list:
        assert call.args == (None,)
        assert call.kwargs == {"semantic_enabled": False}
    rows = [json.loads(line) for line in (tmp_path / "planner.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert [row["sim_time_s"] for row in rows] == pytest.approx([.1, .3, .5, .7, .9])
    assert all(row["semantic_decision"] is None and row["semantic_frame_id"] is None for row in rows)
    assert all(row["model_inference"] is False and row["semantic_influence"] is False for row in rows)
    assert not list(tmp_path.glob("decisions*.jsonl"))
    assert loop.outcome()["model_inference"] is False
    assert loop.outcome()["success"] is True


@pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                    reason="optional pinned PDM sources are not installed")
def test_real_pdm_with_no_model_request_matches_existing_semantic_off_baseline():
    from carla_bridge.tests.test_pdm_planner import make_planner, semantic

    planner = make_planner(speed=4)
    existing = planner.plan(semantic("STOP", "CHANGE_LANE_LEFT"), semantic_enabled=False)
    bypass = planner.plan(None, semantic_enabled=False)
    old_diag, new_diag = existing["planner_diagnostics"], bypass["planner_diagnostics"]
    assert new_diag["semantic_requested"] == {}
    assert new_diag["semantic_enabled"] is False
    assert old_diag["candidate_pool_sha256"] == new_diag["candidate_pool_sha256"]
    assert old_diag["selected_id"] == new_diag["selected_id"]
    assert existing["trajectory"] == bypass["trajectory"]
    assert existing["target_speed"] == bypass["target_speed"]
    assert all(candidate["semantic_bonus"] == 0 for candidate in new_diag["candidates"])


def test_bypass_planner_errors_brake_and_terminate_after_three_rejections(tmp_path):
    loop = bypass_loop(tmp_path, duration=5.0)
    loop.pdm_planner.plan.side_effect = RuntimeError("broken map")
    loop.run()

    assert loop.run_status == "invalid_trajectory"
    assert not loop.outcome()["success"]
    assert loop._trajectory_rejections == 3
    assert loop.pdm_planner.plan.call_count == 3
    assert loop._elapsed == pytest.approx(.5)
    assert loop.carla_client.tick.call_count == 5
    assert all(call.args[0] == [] and call.args[-1] == 0.0
               for call in loop.tracker.set_trajectory.call_args_list)
    assert loop.scenario.ego_vehicle.apply_control.call_args[0][0].brake == 1.0
    rows = [json.loads(line) for line in (tmp_path / "planner.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert all(row["semantic_decision"] is None and not row["validation"]["accepted"] for row in rows)


def test_bypass_session_records_mode_without_creating_a_memory_bank(tmp_path):
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("scenario_name: no_model\n", encoding="utf-8")
    cfg = Config(overrides={"output_dir": str(tmp_path / "runs"),
                            "carla": {"planning": bypass_planning()}})
    session = RunSession(cfg, scenario, "memory_off")
    try:
        session.prepare()
        manifest = json.loads((session.run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "model_off" in session.run_dir.name
        assert manifest["model_inference"] is False
        assert manifest["config"]["carla"]["planning"]["model_inference"] is False
        assert manifest["memory_access"] == "off"
        assert cfg.get("memory_access") == {"retrieve": False, "write": False, "short_term": False}
        assert not cfg.get("memory_db_dir").exists()
    finally:
        session.close()


def test_bypass_report_marks_cognition_disabled_and_preserves_route_result(tmp_path):
    loop = bypass_loop(tmp_path, complete_at=.5)
    loop.run()
    loop.metrics = Mock()
    loop.metrics.summary.return_value = {"route_completion": 1.0}
    loop._write_report()
    report = json.loads(next(tmp_path.glob("carla_run_*.json")).read_text(encoding="utf-8"))
    info = report["run_info"]
    assert info["model_inference"] is False
    assert info["semantic_influence"] is False
    assert info["replan_interval_s"] is None
    assert info["pdm_replan_interval_s"] == pytest.approx(.2)
    assert info["success"] and report["metrics"]["route_completion"] == 1.0
    markdown = next(tmp_path.glob("carla_run_*.md")).read_text(encoding="utf-8")
    assert "None s" not in markdown


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_without_api_keys_reaches_setup_only_when_models_are_disabled(tmp_path, monkeypatch, enabled):
    from carla_bridge import run_carla_demo as entry

    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("scenario_name: cli_no_model\n", encoding="utf-8")
    real_config = Config
    configs = []

    def load_config(*, overrides):
        cfg = real_config(overrides={**overrides, "output_dir": str(tmp_path / "runs")})
        for block in ("scene_understanding", "decision"):
            cfg.data.setdefault(block, {})["api_key_env"] = "CARLA_BYPASS_TEST_ABSENT_KEY"
        configs.append(cfg)
        return cfg

    monkeypatch.delenv("CARLA_BYPASS_TEST_ABSENT_KEY", raising=False)
    monkeypatch.setattr(entry, "load_config", load_config)
    monkeypatch.setattr(entry, "set_log_directory", Mock())
    monkeypatch.setattr(entry, "setup_logger", Mock())
    loop = Mock()
    loop.setup.side_effect = RuntimeError("test reached CARLA setup")
    loop.outcome.return_value = {}
    factory = Mock(return_value=loop)
    monkeypatch.setattr(entry, "ClosedLoop", factory)
    monkeypatch.setattr("sys.argv", ["run_carla_demo", "--scenario", str(scenario),
                                    "--mode", "memory_off", "--planner", "pdm_closed",
                                    "--semantic-influence", "off", "--model-inference",
                                    "on" if enabled else "off"])
    if enabled:
        with pytest.raises(EnvironmentError, match="API Key"):
            entry.main()
        factory.assert_not_called()
        assert not (tmp_path / "runs").exists()
    else:
        with pytest.raises(RuntimeError, match="test reached CARLA setup"):
            entry.main()
        loop.setup.assert_called_once()
        loop.close.assert_called_once()
        assert configs[0].get_nested("carla", "planning", "model_inference") is False


def test_paper_summary_distinguishes_bypass_and_legacy_model_calls(tmp_path):
    import csv
    import yaml
    from carla_bridge.paper_suite import summarize_runs
    from carla_bridge.tests.test_paper_suite import cases

    scenario = cases("eval")["R05_multilane_eval_s101"]
    for label, planning in (("new_bypass", bypass_planning()),
                            ("legacy_sem_off", {"backend": "pdm_closed", "semantic_influence": False})):
        directory = tmp_path / "runs" / label
        directory.mkdir(parents=True)
        manifest = {"run_id": label, "status": "failed", "mode": "memory_off",
                    "scenario_yaml": yaml.safe_dump(scenario),
                    "config": {"carla": {"planning": planning}}}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "results.csv"
    assert summarize_runs(tmp_path / "runs", output) == 2
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    assert rows["new_bypass"]["model_inference"] == "False"
    assert rows["legacy_sem_off"]["model_inference"] == "True"
    assert all(row["status"] == "failed" for row in rows.values())
