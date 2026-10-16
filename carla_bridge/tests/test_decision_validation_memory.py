"""Rejected actuation plans are audited without becoming stored experience."""
import json
from unittest.mock import Mock

import pytest

from carla_bridge.control.trajectory_validation import TrajectoryValidator
from src.vla_memory.common.config import Config
from src.vla_memory.decision.prompt_builder import DecisionPromptBuilder
from src.vla_memory.pipeline import online_loop as module


def _loop(tmp_path, monkeypatch, validator, event_mode=False):
    monkeypatch.setattr(module, "DINOv2Extractor", Mock())
    monkeypatch.setattr(module, "OpenAICompatibleVLMClient", Mock())
    config = Config(overrides={
        "output_dir": str(tmp_path), "memory_db_dir": str(tmp_path / "bank"),
        "feature_dir": str(tmp_path / "features"),
        "mid_term": {"persistence": {"auto_load_on_init": False, "save_on_close": False},
                     "capacity": {"enabled": False}, "admission": {"enabled": False},
                     "event_memory": {"enabled": event_mode}},
    })
    loop = module.OnlineDrivingLoop(config, "memory_on", resume=False,
                                    decision_validator=validator)
    loop.setup()
    loop.scene_pipeline.process_frame = Mock(return_value={
        "scene_understanding": {"scene_description": "road", "scene_id": "unknown"}})
    loop.decision_client.decide = Mock(return_value="original VLM text")
    parsed = {"behavior": "KEEP_LANE", "risk_level": "low", "trajectory": [],
              "target_speed": 5}
    loop._parse_or_fallback = Mock(return_value=(parsed, [], "success", False))
    loop.mid_term.add_record = Mock(wraps=loop.mid_term.add_record)
    loop._event_manager = Mock()
    loop._event_manager.on_frame.return_value = None
    loop._event_manager.flush.return_value = None
    loop._update_enabled = True
    loop._update_manager = Mock()
    loop._update_manager.process.return_value = []
    return loop, parsed


@pytest.mark.parametrize("event_mode", [False, True])
def test_rejected_plan_never_updates_midterm_events_or_conflicts(tmp_path, monkeypatch, event_mode):
    validation = {"accepted": False, "reasons": ["unreachable_distance"], "details": {}}
    validator = Mock(return_value=validation)
    loop, parsed = _loop(tmp_path, monkeypatch, validator, event_mode)
    try:
        kf = {"sample_token": "bad_plan", "timestamp": 100, "scene_token": "new_scene",
              "nav_instruction": "straight", "planning_constraints": {"required_horizon_s": 3}}
        loop._prev_scene_token = "old_scene"
        record = loop.step(kf)
        assert record["decision_output"] == parsed
        assert record["raw_response"] == "original VLM text"
        assert record["decision_validation"] == validation
        assert record["mid_term_memory_added"] is False
        assert record["memory_admission_should_store"] is False
        assert record["memory_update_enabled"] is False
        assert record["memory_admission_reject_reasons"] == ["decision_validation_rejected"]
        loop.mid_term.add_record.assert_not_called()
        loop._event_manager.on_frame.assert_not_called()
        loop._event_manager.flush.assert_not_called()
        loop._update_manager.process.assert_not_called()
        assert len(loop.short_term) == 1
        assert loop._prev_frame_ctx["trajectory"] is None
        validator.assert_called_once_with(parsed, kf)
        assert loop.decision_client.decide.call_args.kwargs["planning_constraints"] == kf["planning_constraints"]
        saved = json.loads(loop.output_jsonl_path.read_text(encoding="utf-8").strip())
        assert saved["decision_validation"] == validation
        assert saved["mid_term_memory_added"] is False
    finally:
        loop.close()


@pytest.mark.parametrize("response", [None, {}, {"accepted": "yes", "reasons": [], "details": {}},
                                     {"accepted": True, "reasons": [], "details": []}])
def test_invalid_validator_result_fails_closed(tmp_path, monkeypatch, response):
    loop, _ = _loop(tmp_path, monkeypatch, Mock(return_value=response))
    try:
        record = loop.step({"sample_token": "invalid_callback", "timestamp": 100})
        assert record["decision_validation"]["accepted"] is False
        assert record["decision_validation"]["reasons"] == ["decision_validator_error"]
        loop.mid_term.add_record.assert_not_called()
    finally:
        loop.close()


def test_validator_exception_fails_closed(tmp_path, monkeypatch):
    loop, _ = _loop(tmp_path, monkeypatch, Mock(side_effect=RuntimeError("guard failed")))
    try:
        record = loop.step({"sample_token": "callback_error", "timestamp": 100})
        assert record["decision_validation"]["accepted"] is False
        assert record["mid_term_memory_added"] is False
        loop.mid_term.add_record.assert_not_called()
    finally:
        loop.close()


@pytest.mark.parametrize("validator", [None, lambda parsed, kf: {
    "accepted": True, "reasons": [], "details": {}}])
def test_accepted_and_legacy_paths_preserve_memory_updates(tmp_path, monkeypatch, validator):
    loop, _ = _loop(tmp_path, monkeypatch, validator)
    try:
        record = loop.step({"sample_token": "accepted", "timestamp": 100})
        assert record["mid_term_memory_added"] is True
        loop.mid_term.add_record.assert_called_once()
        loop._update_manager.process.assert_called_once()
    finally:
        loop.close()


def test_carla_prompt_includes_physics_without_changing_offline_prompt():
    builder = DecisionPromptBuilder()
    old_prompt = builder.build({})
    assert builder.build({}, planning_constraints=None) == old_prompt
    constraints = dict(TrajectoryValidator.DEFAULTS, required_horizon_s=4.0)
    prompt = builder.build({}, ego_state={"speed": 8}, planning_constraints=constraints)
    assert "CARLA 闭环轨迹物理约束" not in old_prompt
    assert "当前实测速度是 8.000 m/s" in prompt
    assert "末点 t=4 秒" in prompt
    assert "route_center 是前方路线的空间参考" in prompt
    assert "当前速度和制动距离" in prompt
