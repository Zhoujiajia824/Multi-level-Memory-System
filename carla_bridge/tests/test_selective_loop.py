"""Selective cognition integration: real scheduling, stubbed model and CARLA I/O.

These checks establish software contracts, not improved closed-loop driving.
Pinned PDM numerics are used for the shadow-execution test when installed.
"""
from copy import deepcopy
from importlib.util import find_spec
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import carla
import pytest

from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.planning.experience_memory import ExecutionExperienceMemory
from carla_bridge.tests.test_model_inference_mode import trajectory
from carla_bridge.tests.test_route_acceptance import fake_loop
from src.vla_memory.common.config import Config


def selective_loop(tmp_path, *, complete_at=None, duration=1.0):
    loop = fake_loop(complete_at=complete_at, duration=duration)
    loop.config.data["output_dir"] = tmp_path
    loop.planner_backend = "pdm_closed"
    loop.semantic_policy = "selective"
    loop.semantic_influence = loop.model_inference = loop._selective_execution = True
    loop.mode = "memory_off"
    loop._raw_capture = Mock(side_effect=AssertionError("legacy capture path used"))
    loop._replan_cycle = Mock(side_effect=AssertionError("legacy cognition cycle used"))
    loop.pdm_planner = Mock()
    loop.pdm_planner.last_context = {"schema_version": 1, "topology_signature": "straight"}
    loop.pdm_planner.query_decision.return_value = {"query": False, "reason": "routine"}
    loop.pdm_planner.plan.side_effect = lambda *a, **kw: deepcopy(trajectory())
    return loop


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_no_query_avoids_models_but_replans_every_point_two_seconds(tmp_path):
    loop = selective_loop(tmp_path, complete_at=1.0)
    loop.run()
    assert loop.run_status == "route_completed"
    assert loop.outcome()["success"] is True
    assert loop.carla_client.tick.call_count == 10
    assert loop.pdm_planner.plan.call_count == 5
    loop.loop.step.assert_not_called()
    loop.cameras.capture.assert_not_called()
    loop._replan_cycle.assert_not_called()
    loop._raw_capture.assert_not_called()
    assert [r["sim_time_s"] for r in rows(tmp_path / "planner.jsonl")] == pytest.approx([.1, .3, .5, .7, .9])
    assert loop.pdm_planner.query_decision.call_count == 2
    assert loop.advisory_stats["model_queries"] == 0
    assert loop.advisory_stats["query_checks_skipped"] == 2


@pytest.mark.parametrize("failure", ["model_exception", "parse_failure", "missing_validation"])
def test_unavailable_model_continues_numerical_plan_without_emergency_stop(tmp_path, failure):
    loop = selective_loop(tmp_path)
    loop.pdm_planner.query_decision.return_value = {"query": True, "reason": "interaction"}
    loop._advisory_keyframe = Mock(return_value={"nav_instruction": "straight"})
    if failure == "model_exception":
        loop.loop.step.side_effect = TimeoutError("model unavailable")
    elif failure == "parse_failure":
        loop.loop.step.return_value = {"decision_output": {"parser_status": "failed"},
                                       "decision_validation": {"accepted": False, "reasons": ["parse"]}}
    else:
        loop.loop.step.return_value = {}
    loop._advisory_tick()
    assert loop.advisory_stats["model_failures"] == 1
    assert loop._active_semantics is None
    assert loop._trajectory_rejections == 0
    assert not loop._early_terminated
    assert loop.pdm_planner.plan.call_count == 2  # normal plan + refreshed fallback
    loop.pdm_planner.governor.submit.assert_called_once_with(None, loop.pdm_planner.last_context, .1)
    assert loop.tracker.set_trajectory.call_count == 2
    assert all(call.args[0] and call.kwargs["follow_speed_profile"]
               for call in loop.tracker.set_trajectory.call_args_list)
    assert all(call.args[0].brake < 1 for call in loop.scenario.ego_vehicle.apply_control.call_args_list)
    assert rows(tmp_path / "advisory.jsonl")[-1]["action"] == "unavailable"


def test_semantic_callback_preserves_intervention_evidence_by_value(tmp_path):
    loop = selective_loop(tmp_path)
    intervention = {"reason": "crossing_conflict", "target_actor_ids": [12], "valid_for_s": 1.0}
    decision = {"parser_status": "success", "decision_mode": "semantic",
                "lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "MAINTAIN",
                "behavior_reason": "visible conflict", "risk_level": "medium", "safety_notes": [],
                "intervention": intervention}
    result = loop._plan_semantic_decision(decision, {"sample_token": "frame-1", "ego_state": {"speed": 5}})
    sent = loop.pdm_planner.plan.call_args.args[0]
    assert sent["intervention"] == intervention
    assert sent["intervention"] is not intervention
    assert result["semantic_decision"]["intervention"] == intervention
    assert loop._active_semantics["intervention"] == intervention
    assert rows(tmp_path / "planner.jsonl")[-1]["semantic_decision"]["intervention"] == intervention


@pytest.mark.parametrize("complete_at,expected", [(None, "timeout"), (1.0, "route_completed")])
def test_selective_mode_requires_full_route_for_success(tmp_path, complete_at, expected):
    loop = selective_loop(tmp_path, complete_at=complete_at)
    loop.run()
    assert loop.run_status == expected
    assert loop.outcome()["success"] == (expected == "route_completed")
    assert loop.route_completion == (1.0 if expected == "route_completed" else .4)


def experience_decision(action="MAINTAIN", speed=5.0):
    return {"semantic_decision": {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": action},
            "planner_diagnostics": {
                "selected_id": "center:" + action, "baseline_selected_id": "center:CRUISE",
                "planner_context": {"schema_version": 1, "sim_time_s": 0.0, "ego_speed_mps": 5.0,
                    "topology_signature": "Town05:straight", "navigation": "straight", "in_junction": False,
                    "blocked": False, "special_situations": [], "baseline_stop_expected": False,
                    "actors": [], "signals": []},
                "candidates": [{"id": "center:" + action, "lateral": "FOLLOW_ROUTE",
                    "longitudinal": action, "desired_speed_mps": speed, "idm_headway_s": 1.5,
                    "idm_min_gap_m": 1.5, "idm_accel_max_mps2": 1.5, "idm_decel_max_mps2": 3.0}]}}


def execution_loop(tmp_path, *, memory_mode="read_write"):
    loop = selective_loop(tmp_path / "run")
    loop.metrics = NS(on_tick=Mock(), red_light_violations=0, wrong_way_violations=0)
    loop.scenario.route.distance_completed_m.side_effect = lambda: 5 * loop._elapsed
    loop.experience_memory = ExecutionExperienceMemory(tmp_path / "bank", memory_mode=memory_mode,
        run_id="integration", run_dir=tmp_path / "audit", planner_version="carla_pdm_selective_v1")
    return loop


def test_actual_action_change_censors_pending_window_before_new_execution(tmp_path):
    loop = execution_loop(tmp_path)
    loop._elapsed = .1
    loop._begin_execution_experience(experience_decision())
    old_id = next(iter(loop.experience_memory.pending))
    loop._elapsed = .2
    loop._record_tick(loop.scenario.ego_vehicle)
    loop._elapsed = .3
    loop._begin_execution_experience(experience_decision("DECELERATE", 3.0))
    assert len(loop.experience_memory.pending) == 1
    assert old_id not in loop.experience_memory.pending
    records = rows(tmp_path / "bank" / "planning_experiences.jsonl")
    old = [record for record in records if record["id"] == old_id][-1]
    assert old["status"] == "censored"
    assert old["outcome"]["termination_reason"] == "policy_changed"
    assert old["decision"]["actual_action"]["longitudinal_decision"] == "MAINTAIN"
    assert old["outcome"]["duration_s"] == pytest.approx(.1)
    new = next(iter(loop.experience_memory.pending.values()))["record"]
    assert new["decision"]["actual_action"]["longitudinal_decision"] == "DECELERATE"


def test_unfinished_route_and_interrupted_window_never_become_learned_success(tmp_path):
    loop = execution_loop(tmp_path)
    loop._elapsed = .1
    d = experience_decision()
    loop._begin_execution_experience(d)
    for t in (.2, .3, .4):
        loop._elapsed = t
        loop._record_tick(loop.scenario.ego_vehicle)
    loop.experience_memory.finish("timeout")
    evidence = loop.experience_memory.retrieve(d["planner_diagnostics"]["planner_context"])
    assert evidence["status"] == "unknown"
    assert not evidence["parameter_priors"]
    assert evidence["causal_advantage"]["status"] == "unknown"
    assert rows(tmp_path / "bank" / "planning_experiences.jsonl")[-1]["status"] == "censored"


def test_expansion_records_original_baseline_and_ends_window_when_geometry_changes(tmp_path):
    loop = execution_loop(tmp_path)
    decision = experience_decision()
    diagnostics = decision["planner_diagnostics"]
    diagnostics["original_baseline_selected_id"] = "center:STOP"
    diagnostics["baseline_selected_id"] = diagnostics["selected_id"]
    candidate = diagnostics["candidates"][0]
    candidate.update(recovery=True, recovery_recipe="short_pose_rejoin",
                     recovery_transition_m=3.0, lane_offset_m=-.5)
    loop._elapsed = .1
    loop._begin_execution_experience(decision)
    old_id, pending = next(iter(loop.experience_memory.pending.items()))
    record = pending["record"]
    assert record["baseline_candidate_id"] == "center:STOP"
    assert record["decision"]["adopted"] is True
    assert record["decision"]["selected_geometry"]["recovery_transition_m"] == 3.0
    loop._elapsed = .2
    loop._record_tick(loop.scenario.ego_vehicle)
    candidate["recovery_transition_m"] = 4.0
    loop._elapsed = .3
    loop._begin_execution_experience(decision)
    assert old_id not in loop.experience_memory.pending
    records = rows(tmp_path / "bank" / "planning_experiences.jsonl")
    assert [r for r in records if r["id"] == old_id][-1]["status"] == "censored"


def test_read_only_observation_audit_does_not_create_or_modify_bank(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    path = bank / "planning_experiences.jsonl"
    path.write_bytes(b"malformed existing line\n")
    before = path.read_bytes()
    loop = execution_loop(tmp_path, memory_mode="read_only")
    d = experience_decision()
    loop.experience_memory.retrieve(d["planner_diagnostics"]["planner_context"])
    loop._begin_execution_experience(d)
    loop._elapsed = .1
    loop._record_tick(loop.scenario.ego_vehicle)
    loop.experience_memory.finish("interrupted")
    assert path.read_bytes() == before
    assert sorted(p.name for p in bank.iterdir()) == ["planning_experiences.jsonl"]
    assert (tmp_path / "audit" / "planning_experience_audit.jsonl").is_file()
    assert loop.experience_memory.summarize()["malformed_lines"] == 1


def test_memory_off_setup_never_constructs_execution_bank(tmp_path, monkeypatch):
    from carla_bridge import closed_loop as module
    from carla_bridge.planning import pdm_planner, experience_memory
    cfg = Config(overrides={"output_dir": str(tmp_path / "run"), "memory_db_dir": str(tmp_path / "bank"),
        "carla": {"planning": {"backend": "pdm_closed", "model_inference": False,
            "semantic_influence": False, "semantic_policy": "selective", "planning_memory": "full"},
            "video": {"enabled": False}}})
    scenario = NS(world=Mock(), ego_vehicle=Mock(), cfg={}, route=Mock(), setup=Mock(),
                  scenario_name=Mock(return_value="memory_off"), destroy=Mock(return_value=True))
    monkeypatch.setattr(module, "CarlaClient", Mock(return_value=Mock()))
    monkeypatch.setattr(module.ScenarioManager, "from_yaml", Mock(return_value=scenario))
    monkeypatch.setattr(pdm_planner, "validate_dependencies", Mock())
    monkeypatch.setattr(pdm_planner, "CarlaPDMPlanner", Mock(return_value=Mock()))
    monkeypatch.setattr(module, "MetricsCollector", Mock())
    forbidden = Mock(side_effect=AssertionError("memory_off accessed execution memory"))
    monkeypatch.setattr(experience_memory, "ExecutionExperienceMemory", forbidden)
    loop = ClosedLoop(cfg, "unused", mode="memory_off")
    loop._bind_pdm_controller = loop._setup_collision_sensor = loop._write_control_context = Mock()
    loop.setup()
    forbidden.assert_not_called()
    assert loop.experience_memory is None
    assert not (tmp_path / "bank").exists()


@pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                    reason="optional pinned PDM sources are not installed")
def test_real_shadow_executes_identical_baseline_and_retains_would_select(tmp_path, monkeypatch):
    from carla_bridge.tests.test_pdm_planner import make_planner, semantic
    from carla_bridge.planning.selection import choose_candidate
    from carla_bridge.planning import scene_context
    planner = make_planner(speed=4)
    baseline = planner.plan(None, semantic_enabled=False)
    planner.semantic_policy = "shadow"
    planner.governor = Mock()
    planner.governor.config = {"moving_option_progress_m": 2.0}
    planner.governor.observe_planner_health.return_value = {"sustained": False}
    planner.governor.recovery_request.return_value = {"authorized": False}
    context = experience_decision()["planner_diagnostics"]["planner_context"]
    monkeypatch.setattr(scene_context, "build_scene_context", lambda *a, **kw: deepcopy(context))
    def forced_selection(candidates, *_a, **_kw):
        selected, diagnostics = choose_candidate(candidates, semantic("STOP"), True, 0.0, 2.0)
        diagnostics["governor"] = {"authorized": True, "adopted": True, "reason": "test_evidence"}
        return selected, diagnostics

    planner.governor.select.side_effect = forced_selection
    shadow = planner.plan(semantic("STOP"), semantic_enabled=True, semantic_update=True)
    d = shadow["planner_diagnostics"]
    assert d["shadow_only"] is True
    assert d["shadow_selected_id"] != d["baseline_selected_id"]
    assert d["selected_id"] == d["baseline_selected_id"] == baseline["planner_diagnostics"]["selected_id"]
    assert shadow["trajectory"] == baseline["trajectory"]
    assert shadow["target_speed"] == baseline["target_speed"]
    assert all(c["semantic_bonus"] == 0 for c in d["candidates"])
    assert any(c["semantic_bonus"] > 0 for c in d["shadow_candidates"])
    assert d["governor"]["would_adopt"] is True
    assert d["governor"]["adopted"] is False
    planner.governor.submit.assert_called_once()
