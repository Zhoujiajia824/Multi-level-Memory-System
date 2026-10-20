"""A semantic request can expand numerical search, never supply a trajectory."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.vla_memory.decision.prompt_builder import DecisionPromptBuilder
from src.vla_memory.decision.semantic_output_parser import (
    MAX_SEMANTIC_OUTPUT_CHARS, parse_semantic_decision_output,
)


def request(**changes):
    value = {"needed": True, "reason_code": "planner_degraded", "target_actor_ids": [],
             "target_stop_line_id": None, "activation": "immediate",
             "release": "condition_cleared", "valid_for_s": 2.0,
             "planner_request": "expand_candidates"}
    value.update(changes)
    return value


def output(**changes):
    value = {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "MAINTAIN",
             "behavior_reason": "前进候选持续退化，请重新搜索可跟踪路径。",
             "risk_level": "low", "safety_notes": [], "intervention": request()}
    value.update(changes)
    return value


def parse(value):
    return parse_semantic_decision_output(json.dumps(value), require_intervention=True)


@pytest.mark.parametrize("lateral", ["FOLLOW_ROUTE", "KEEP_LANE"])
@pytest.mark.parametrize("longitudinal", ["MAINTAIN", "ACCELERATE", "FOLLOW"])
def test_expansion_retains_semantics_and_request_without_numerical_plan(lateral, longitudinal):
    value = output(lateral_decision=lateral, longitudinal_decision=longitudinal)
    parsed, errors = parse(value)
    assert not errors
    assert parsed["intervention"] == value["intervention"]
    assert parsed["lateral_decision"] == lateral
    assert parsed["longitudinal_decision"] == longitudinal
    assert parsed["trajectory"] == []


def test_old_seven_field_requests_keep_exact_values_and_default_to_no_expansion():
    old = request(needed=False, reason_code="none")
    old.pop("planner_request")
    parsed, errors = parse(output(intervention=old))
    assert not errors
    assert parsed["intervention"] == old
    assert parsed["intervention"].get("planner_request", "none") == "none"


@pytest.mark.parametrize("value", [None, True, 0, 1.0, [], {}, "expand", "EXPAND_CANDIDATES",
                                   "expand_candidates ", "none " , "x" * 2000])
def test_planner_request_is_a_strict_bounded_enum(value):
    parsed, errors = parse(output(intervention=request(planner_request=value)))
    assert parsed is None and errors


@pytest.mark.parametrize("changes", [
    {"needed": False, "reason_code": "none"},
    {"reason_code": "path_blocked"}, {"reason_code": "route_change"},
    {"reason_code": "traffic_signal"}, {"reason_code": "none"},
    {"planner_request": "none"}, {"target_actor_ids": [123]},
    {"target_stop_line_id": "stop_123_0"},
    {"activation": "while_conflict"}, {"activation": "approach_stop_line"},
    {"release": "signal_changed"}, {"release": "maneuver_complete"},
    {"release": "target_passed"},
])
def test_expansion_cannot_impersonate_actor_stopline_or_traffic_wait_request(changes):
    parsed, errors = parse(output(intervention=request(**changes)))
    assert parsed is None and errors


@pytest.mark.parametrize("changes", [
    {"lateral_decision": "CHANGE_LANE_LEFT"}, {"lateral_decision": "CHANGE_LANE_RIGHT"},
    {"longitudinal_decision": "STOP"}, {"longitudinal_decision": "YIELD"},
    {"longitudinal_decision": "DECELERATE"},
])
def test_expansion_is_separate_from_stop_yield_or_forced_lane_change(changes):
    parsed, errors = parse(output(**changes))
    assert parsed is None and errors


@pytest.mark.parametrize("field,value", [
    ("trajectory", [{"x": 1, "y": 0}]), ("target_speed_mps", 3),
    ("steering_angle", .1), ("recovery_length_m", 12), ("planner_parameters", {}),
])
def test_expansion_never_accepts_numerical_model_commands(field, value):
    parsed, errors = parse(output(intervention=request(**{field: value})))
    assert parsed is None and errors


def test_request_stays_nested_and_duplicate_request_keys_are_rejected():
    parsed, errors = parse(output(planner_request="expand_candidates"))
    assert parsed is None and errors
    raw = json.dumps(output()).replace('"planner_request": "expand_candidates"',
                                      '"planner_request": "none", "planner_request": "expand_candidates"')
    parsed, errors = parse_semantic_decision_output(raw, require_intervention=True)
    assert parsed is None and errors


def test_oversized_model_output_is_rejected_before_json_decoding():
    value = output(behavior_reason="x" * MAX_SEMANTIC_OUTPUT_CHARS)
    parsed, errors = parse(value)
    assert parsed is None and any("16384" in item for item in errors)


@pytest.mark.parametrize("policy", ["selective", "shadow"])
def test_pipeline_passes_expansion_through_without_synthesizing_trajectory(policy):
    from src.vla_memory.pipeline.online_loop import OnlineDrivingLoop
    parsed, errors, status, fallback = OnlineDrivingLoop._parse_or_fallback(
        SimpleNamespace(), json.dumps(output()), {}, "straight",
        decision_mode="semantic", semantic_policy=policy,
    )
    assert not errors and status == "success" and not fallback
    assert parsed["intervention"]["planner_request"] == "expand_candidates"
    assert parsed["intervention"]["reason_code"] == "planner_degraded"
    assert parsed["trajectory"] == []


def test_planner_health_survives_overfull_actor_context_without_mutating_input():
    # Producers may append health after many actors; it is needed to judge a
    # degraded-planner request and must precede large optional actor details.
    context = {"actors": [{"id": i, "details": "x" * 2000} for i in range(1000)],
               "planner_degradation": {"degraded": True, "sustained": True, "duration_s": 5.1,
                                       "observation_count": 12},
               "planner_health": {"source": "native_pdm_baseline", "positive_progressing_count": 0}}
    before = copy.deepcopy(context)
    rendered = DecisionPromptBuilder._render_planner_context(context)
    payload = json.loads(rendered.splitlines()[1])
    assert payload["snapshot"]["planner_health"] == context["planner_health"]
    assert payload["snapshot"]["planner_degradation"] == context["planner_degradation"]
    assert payload["context_limited"] is True
    assert len(rendered) < 12000
    assert context == before


@pytest.mark.parametrize("policy", ["selective", "shadow"])
def test_prompts_describe_health_gate_and_keep_parseable_nonintervention_default(policy):
    prompt = DecisionPromptBuilder().build(
        {}, planning_constraints={"decision_mode": "semantic", "semantic_policy": policy,
                                  "planner_context": {"planner_health": {"sustained_failure": True}}},
    )
    for phrase in ("planner_health", "持续退化", "红灯、排队", "不保证起步",
                   "当前后端约束核验", "expand_candidates", "planner_degraded"):
        assert phrase in prompt
    section = prompt.split("## 输出要求", 1)[1]
    example, _ = json.JSONDecoder().raw_decode(section[section.index("{"):])
    parsed, errors = parse(example)
    assert not errors
    assert parsed["intervention"]["needed"] is False
    assert parsed["intervention"]["planner_request"] == "none"
