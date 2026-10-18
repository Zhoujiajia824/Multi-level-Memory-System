"""Semantic planning, memory admission and two-rate control without PDM/VLM servers."""
from copy import deepcopy
import json
from unittest.mock import Mock

import carla
import pytest

from carla_bridge.control.trajectory_tracker import TrajectoryTracker
from carla_bridge.tests.test_decision_validation_memory import _loop
from carla_bridge.tests.test_route_acceptance import fake_loop
from src.vla_memory.decision.semantic_output_parser import parse_semantic_decision_output
from src.vla_memory.memory.admission import MemoryAdmissionResult
from src.vla_memory.memory.event_memory import EventMemoryManager
from src.vla_memory.pipeline.online_loop import OnlineDrivingLoop


SEMANTIC = {
    "lateral_decision": "CHANGE_LANE_LEFT", "longitudinal_decision": "DECELERATE",
    "behavior_reason": "Leave room for the merging vehicle.",
    "risk_level": "medium", "safety_notes": ["Keep a safe gap."],
}


def planned_trajectory():
    return {
        "trajectory": [{"t": i * 0.1, "x": i * 0.5, "y": 0.0, "optional_v": 5.0}
                       for i in range(31)],
        "target_speed": 5.0,
        "selected_lateral": "FOLLOW_ROUTE",
        "planner_diagnostics": {"selected_id": 4, "selected_lateral": "FOLLOW_ROUTE"},
    }


def semantic_keyframe(token="semantic_frame", timestamp=100):
    return {"sample_token": token, "timestamp": timestamp,
            "ego_state": {"speed": 5.0}, "nav_instruction": "straight",
            "planning_constraints": {"decision_mode": "semantic", "required_horizon_s": 3.0}}


def planning_loop(tmp_path, enabled=True):
    closed = fake_loop(duration=30)
    closed.config.data["output_dir"] = tmp_path / "closed"
    closed.planner_backend = "pdm_closed"
    closed.semantic_influence = enabled
    # This scheduling fixture has no map geometry. Do not let Mock fabricate
    # a truthy waypoint collection for the optional current navigation facts.
    closed.scenario.route._route_waypoints = []
    closed.pdm_planner = Mock()
    closed.pdm_planner.plan.side_effect = lambda *args, **kwargs: deepcopy(planned_trajectory())
    return closed


def semantic_memory_loop(tmp_path, monkeypatch, closed, event_mode=False):
    online, _ = _loop(tmp_path, monkeypatch, closed._validate_decision, event_mode)
    # Exercise the production semantic parser and processor, while replacing
    # only model/network dependencies and numerical proposal generation.
    online._parse_or_fallback = OnlineDrivingLoop._parse_or_fallback.__get__(online)
    online.decision_client.decide.return_value = json.dumps(SEMANTIC)
    online.decision_processor = closed._plan_semantic_decision
    return online


@pytest.mark.parametrize("event_mode", [False, True])
@pytest.mark.parametrize("semantic_enabled", [False, True])
def test_planning_precedes_admission_and_preserves_semantics_in_memory(
        tmp_path, monkeypatch, event_mode, semantic_enabled):
    closed = planning_loop(tmp_path, semantic_enabled)
    online = semantic_memory_loop(tmp_path, monkeypatch, closed, event_mode)
    online._admission_enabled = True
    online._admission_controller = Mock()

    def admit(ctx, previous):
        # Admission must see the selected, validated numerical plan, while the
        # model intent remains independently available for future retrieval.
        closed.pdm_planner.plan.assert_called_once()
        assert ctx["parsed"]["trajectory"] == planned_trajectory()["trajectory"]
        assert ctx["parsed"]["semantic_decision"] == SEMANTIC
        return MemoryAdmissionResult(should_store=True, admission_score=1.0)

    online._admission_controller.decide.side_effect = admit
    if event_mode:
        online._event_manager = EventMemoryManager({
            "min_event_length_frames": 1, "max_event_length_frames": 1,
        })
    try:
        record = online.step(semantic_keyframe())
        assert record["decision_validation"]["accepted"]
        assert record["mid_term_memory_added"]
        assert record["model_decision_output"]["trajectory"] == []
        assert "target_speed" not in record["model_decision_output"]
        assert "planner_diagnostics" not in record["model_decision_output"]
        assert record["model_decision_output"]["behavior"] == "SLOW_DOWN"
        assert record["decision_output"]["behavior"] == "KEEP_LANE"
        closed.pdm_planner.plan.assert_called_once_with(
            SEMANTIC, semantic_enabled=semantic_enabled)
        online.mid_term.add_record.assert_called_once()
        stored = online.mid_term.add_record.call_args.args[0]
        assert stored.semantic_decision == SEMANTIC
        assert stored.trajectory == planned_trajectory()["trajectory"]
        assert stored.planning_context == {
            "backend": "pdm_closed", "semantic_influence": semantic_enabled,
            "selected_id": 4, "executed_behavior": "KEEP_LANE",
            "selected_lateral": "FOLLOW_ROUTE", "selected_longitudinal": "CRUISE",
            "record_stage": "selected_before_execution",
        }
        assert stored.memory_type == ("event_memory" if event_mode else "frame_memory")
        saved = json.loads(online.output_jsonl_path.read_text(encoding="utf-8").strip())
        assert saved["model_decision_output"]["trajectory"] == []
        assert saved["decision_output"]["trajectory_source"] == "pdm_closed"
        assert saved["planning_output"] == planned_trajectory()["planner_diagnostics"]
    finally:
        online.close()


@pytest.mark.parametrize("failure", ["planner_exception", "invalid_semantics", "missing_processor",
                                    "malformed_processor_result", "unsafe_trajectory"])
@pytest.mark.parametrize("event_mode", [False, True])
def test_failed_semantic_plan_never_becomes_memory(tmp_path, monkeypatch, failure, event_mode):
    closed = planning_loop(tmp_path)
    online = semantic_memory_loop(tmp_path, monkeypatch, closed, event_mode)
    online._admission_enabled = True
    online._admission_controller = Mock()
    if failure == "planner_exception":
        closed.pdm_planner.plan.side_effect = RuntimeError("planner unavailable")
    elif failure == "invalid_semantics":
        online.decision_client.decide.return_value = json.dumps({**SEMANTIC, "trajectory": []})
        closed._active_semantics = deepcopy(SEMANTIC)
    elif failure == "missing_processor":
        online.decision_processor = None
    elif failure == "malformed_processor_result":
        online.decision_processor = Mock(return_value=[])
    elif failure == "unsafe_trajectory":
        invalid = planned_trajectory()
        invalid["trajectory"][-1]["x"] = 999.0
        closed.pdm_planner.plan.side_effect = None
        closed.pdm_planner.plan.return_value = invalid
    try:
        record = online.step(semantic_keyframe())
        assert record["decision_validation"]["accepted"] is False
        assert record["mid_term_memory_added"] is False
        assert record["memory_update_enabled"] is False
        online._admission_controller.decide.assert_not_called()
        online.mid_term.add_record.assert_not_called()
        online._event_manager.on_frame.assert_not_called()
        online._update_manager.process.assert_not_called()
        assert online._prev_frame_ctx["trajectory"] is None
        assert len(online.short_term) == 1  # Objective observations remain useful.
        if failure == "invalid_semantics":
            closed.pdm_planner.plan.assert_not_called()
            assert closed._active_semantics is None
            assert record["parser_status"] == "failed"
        if failure == "missing_processor":
            assert record["decision_validation"]["reasons"] == ["semantic_planner_missing"]
        if failure == "unsafe_trajectory":
            assert closed._active_semantics is None
    finally:
        online.close()


def test_mutating_processor_cannot_rewrite_original_model_audit(tmp_path, monkeypatch):
    closed = planning_loop(tmp_path)
    online = semantic_memory_loop(tmp_path, monkeypatch, closed)

    def mutating_processor(parsed, kf):
        parsed["safety_notes"].append("planner-only note")
        return closed._plan_semantic_decision(parsed, kf)

    online.decision_processor = mutating_processor
    try:
        record = online.step(semantic_keyframe())
        assert record["model_decision_output"]["safety_notes"] == SEMANTIC["safety_notes"]
        assert record["decision_output"]["safety_notes"][-1] == "planner-only note"
    finally:
        online.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_three_second_semantics_with_point_two_second_planning_no_extra_ticks(tmp_path, enabled):
    closed = planning_loop(tmp_path, enabled)
    decision, errors = parse_semantic_decision_output(json.dumps(SEMANTIC))
    assert not errors
    plan_times = []

    def plan(*args, **kwargs):
        plan_times.append(closed._elapsed)
        assert args[0] == SEMANTIC
        assert kwargs["semantic_enabled"] is enabled
        return deepcopy(planned_trajectory())

    def semantic_step(kf):
        result = closed._plan_semantic_decision(deepcopy(decision), kf)
        return {"decision_output": result,
                "decision_validation": closed._validate_decision(result, kf)}

    closed.pdm_planner.plan.side_effect = plan
    closed.loop.step.side_effect = semantic_step
    closed._replan_cycle()
    closed._replan_cycle()
    assert closed.carla_client.tick.call_count == 60
    semantic_frames = [call.args[0] for call in closed.loop.step.call_args_list]
    assert [kf["sim_time_s"] for kf in semantic_frames] == pytest.approx([0.1, 3.1])
    assert all(kf["planning_constraints"]["decision_mode"] == "semantic" for kf in semantic_frames)
    assert plan_times == pytest.approx([0.1 + 0.2 * i for i in range(30)])
    assert all(call.kwargs["follow_speed_profile"] for call in closed.tracker.set_trajectory.call_args_list)
    journal = [json.loads(line) for line in (tmp_path / "closed" / "planner.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(journal) == 30
    assert sum(row["source"] == "semantic_update" for row in journal) == 2
    assert all(row["semantic_influence"] is enabled for row in journal)


def test_fast_planner_failure_clears_stale_control_and_brakes(tmp_path):
    closed = planning_loop(tmp_path)
    closed.tracker = TrajectoryTracker({}, control_dt_s=0.1)
    decision, _ = parse_semantic_decision_output(json.dumps(SEMANTIC))

    def semantic_step(kf):
        result = closed._plan_semantic_decision(deepcopy(decision), kf)
        # The initial plan is valid; only subsequent fast updates fail.
        closed.pdm_planner.plan.side_effect = RuntimeError("new actor state unavailable")
        return {"decision_output": result,
                "decision_validation": closed._validate_decision(result, kf)}

    closed.loop.step.side_effect = semantic_step
    closed._replan_cycle()
    assert closed._early_terminated
    assert closed._termination_reason == "invalid_trajectory"
    assert closed._trajectory_rejections == 3
    assert not closed.tracker.has_trajectory()
    assert closed.scenario.ego_vehicle.apply_control.call_args.args[0].brake == 1.0
    assert closed.carla_client.tick.call_count == 7
    assert closed._elapsed == pytest.approx(0.7)


def test_no_fast_plan_without_valid_active_semantics(tmp_path):
    closed = planning_loop(tmp_path)
    closed._elapsed = 1.0
    closed._refresh_pdm_control(carla.Transform(), 0.0, 5.0)
    closed.pdm_planner.plan.assert_not_called()
    closed.tracker.set_trajectory.assert_not_called()


@pytest.mark.parametrize("selected_longitudinal,behavior", [
    ("CRUISE", "KEEP_LANE"), ("DECELERATE", "SLOW_DOWN"),
    ("FOLLOW", "FOLLOW"), ("YIELD", "YIELD"), ("STOP", "STOP"),
])
@pytest.mark.parametrize("label_source", ["plan", "diagnostics"])
def test_memory_behavior_describes_selected_plan_not_requested_intent(
        tmp_path, selected_longitudinal, behavior, label_source):
    closed = planning_loop(tmp_path, enabled=False)
    plan = planned_trajectory()
    destination = plan if label_source == "plan" else plan["planner_diagnostics"]
    destination["selected_longitudinal"] = selected_longitudinal
    result = closed._merge_plan({}, deepcopy(SEMANTIC), plan)
    assert result["behavior"] == behavior
    assert result["planning_context"]["executed_behavior"] == behavior
    assert result["planning_context"]["selected_longitudinal"] == selected_longitudinal
    assert result["semantic_decision"] == SEMANTIC
    assert result["planning_context"]["record_stage"] == "selected_before_execution"


def test_full_audit_disk_still_brakes_and_terminates_after_three_failures(tmp_path):
    closed = planning_loop(tmp_path)
    closed.tracker = TrajectoryTracker({}, control_dt_s=0.1)
    closed.tracker.set_trajectory(planned_trajectory()["trajectory"], 0.0, 0.0, 0.0, 5.0,
                                  follow_speed_profile=True)
    closed._active_semantics = deepcopy(SEMANTIC)
    closed._audit_pdm_plan = Mock(side_effect=OSError("disk full"))
    for index in range(1, 4):
        closed._elapsed = index * 0.2
        closed._refresh_pdm_control(carla.Transform(), 0.0, 5.0)
        assert closed.scenario.ego_vehicle.apply_control.call_args.args[0].brake == 1.0
        assert not closed.tracker.has_trajectory()
        assert closed._trajectory_rejections == index
        assert closed._early_terminated is (index == 3)
    assert closed._termination_reason == "invalid_trajectory"
    assert closed._audit_pdm_plan.call_count == 6  # Plan audit and error audit both fail.
    assert closed.scenario.ego_vehicle.apply_control.call_count == 3
    closed.carla_client.tick.assert_not_called()


@pytest.mark.parametrize("source", ["semantic", "control"])
def test_geometry_failure_preserves_diagnostics_and_brakes(tmp_path, source):
    from carla_bridge.planning.pdm_planner import RoadGeometryError
    closed = planning_loop(tmp_path)
    diagnostics = {"failure_reason": "road_geometry_error", "geometry": {
        "current_side": "left", "excluded_lateral_paths": {"center": "lane_change_not_permitted"}}}
    closed.pdm_planner.plan.side_effect = RoadGeometryError("No legal lateral path", diagnostics)
    if source == "semantic":
        decision, _ = parse_semantic_decision_output(json.dumps(SEMANTIC))
        with pytest.raises(RoadGeometryError):
            closed._plan_semantic_decision(decision, semantic_keyframe())
        assert closed._active_semantics is None
    else:
        closed._active_semantics = deepcopy(SEMANTIC)
        closed._elapsed = 0.2
        closed._refresh_pdm_control(carla.Transform(), 0, 5)
    audit = json.loads((tmp_path / "closed" / "planner.jsonl").read_text(encoding="utf-8"))
    assert audit["source"] == source + "_error"
    assert audit["planner_diagnostics"] == diagnostics
    assert audit["semantic_decision"] == SEMANTIC
    assert not audit["validation"]["accepted"]
    assert closed.scenario.ego_vehicle.apply_control.call_args.args[0].brake == 1.0


def test_expected_emergency_is_not_reported_as_malformed_trajectory(tmp_path):
    closed = planning_loop(tmp_path)
    plan = {"trajectory": [], "target_speed": 0.0, "planner_diagnostics": {
        "emergency_brake": True, "failure_reason": "no_finite_candidate"}}
    validation = closed._validate_decision(plan, semantic_keyframe())
    assert validation == {"accepted": False, "reasons": ["pdm_emergency_brake"],
                          "details": {"failure_reason": "no_finite_candidate"}}
