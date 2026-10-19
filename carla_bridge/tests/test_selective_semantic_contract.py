"""Intervention requests are explicit structured evidence, never inferred intent."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.vla_memory.common.config import Config
from src.vla_memory.decision.config_access import (
    get_decision_system_prompt, get_semantic_prompt_template,
)
from src.vla_memory.decision.prompt_builder import DecisionPromptBuilder
from src.vla_memory.decision.semantic_output_parser import (
    INTERVENTION_ACTIVATIONS, INTERVENTION_FIELDS, INTERVENTION_REASON_CODES,
    INTERVENTION_RELEASES, parse_semantic_decision_output,
)


def decision(**changes):
    value = {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "MAINTAIN",
             "behavior_reason": "当前无额外语义信息。", "risk_level": "low", "safety_notes": []}
    value.update(changes)
    return value


def intervention(**changes):
    value = {"needed": False, "reason_code": "none", "target_actor_ids": [],
             "target_stop_line_id": None, "activation": "immediate",
             "release": "condition_cleared", "valid_for_s": 1.0}
    value.update(changes)
    return value


def parse_request(request):
    return parse_semantic_decision_output(
        json.dumps(decision(intervention=request)), require_intervention=True,
    )


@pytest.mark.parametrize("duration", [.2, 1, 6.0])
def test_explicit_no_intervention_preserved_without_inferred_targets(duration):
    value = intervention(valid_for_s=duration)
    parsed, errors = parse_request(value)
    assert not errors
    assert parsed["intervention"] == value
    assert parsed["trajectory"] == []


@pytest.mark.parametrize("reason", [r for r in INTERVENTION_REASON_CODES if r != "none"])
def test_parser_validates_syntax_without_inventing_physical_grounding(reason):
    # An unknown ID is retained for the current-state governor to reject. The
    # parser cannot claim a visibility/special-rule request has physical support.
    value = intervention(needed=True, reason_code=reason, target_actor_ids=[88321])
    parsed, errors = parse_request(value)
    assert not errors and parsed["intervention"] == value


@pytest.mark.parametrize("field", sorted(INTERVENTION_FIELDS))
def test_partial_intervention_is_rejected_even_when_optional_in_legacy_mode(field):
    request = intervention()
    request.pop(field)
    parsed, errors = parse_semantic_decision_output(json.dumps(decision(intervention=request)))
    assert parsed is None and errors


@pytest.mark.parametrize("invalid_request", [None, [], "none", False, 0])
def test_intervention_requires_object(invalid_request):
    parsed, errors = parse_request(invalid_request)
    assert parsed is None and errors


@pytest.mark.parametrize("field,value", [
    ("needed", 1), ("needed", "false"), ("needed", None),
    ("reason_code", "unknown"), ("reason_code", False),
    ("target_actor_ids", [True]), ("target_actor_ids", [1.0]),
    ("target_actor_ids", ["123"]), ("target_actor_ids", "123"),
    ("target_actor_ids", [1, 1]), ("target_actor_ids", list(range(9))),
    ("target_stop_line_id", 123), ("target_stop_line_id", " "),
    ("activation", "always"), ("release", "never"),
    ("valid_for_s", True), ("valid_for_s", None), ("valid_for_s", ".2"),
    ("valid_for_s", .199), ("valid_for_s", 6.01),
    ("valid_for_s", float("nan")), ("valid_for_s", float("inf")),
    ("valid_for_s", float("-inf")), ("valid_for_s", 10 ** 400),
    ("trajectory", []), ("target_speed_mps", 3), ("distance_m", 10),
])
def test_intervention_rejects_ambiguous_ids_numbers_and_extra_fields(field, value):
    parsed, errors = parse_request(intervention(**{field: value}))
    assert parsed is None and errors


@pytest.mark.parametrize("changes", [
    {"needed": True}, {"reason_code": "interaction"},
    {"target_actor_ids": [1]}, {"target_stop_line_id": "stop_1_0"},
])
def test_no_intervention_and_reason_targets_must_be_consistent(changes):
    parsed, errors = parse_request(intervention(**changes))
    assert parsed is None and errors


@pytest.mark.parametrize("activation", INTERVENTION_ACTIVATIONS)
@pytest.mark.parametrize("release", INTERVENTION_RELEASES)
def test_exact_activation_and_release_enums_are_preserved(activation, release):
    value = intervention(needed=True, reason_code="traffic_signal",
                         target_stop_line_id="stop_42_0", activation=activation,
                         release=release)
    parsed, errors = parse_request(value)
    assert not errors and parsed["intervention"] == value


def test_nested_duplicate_and_overflow_numeric_json_are_rejected():
    raw = json.dumps(decision(intervention=intervention()))
    duplicate = raw.replace('"needed": false', '"needed": false, "needed": true')
    overflow = raw.replace('"valid_for_s": 1.0', '"valid_for_s": 1e309')
    for invalid in (duplicate, overflow):
        parsed, errors = parse_semantic_decision_output(invalid, require_intervention=True)
        assert parsed is None and errors


def test_legacy_does_not_acquire_intervention_but_selective_rejects_missing_authorization():
    raw = json.dumps(decision(longitudinal_decision="STOP", behavior_reason="Stop for actor 123"))
    legacy, errors = parse_semantic_decision_output(raw)
    assert not errors and "intervention" not in legacy
    parsed, errors = parse_semantic_decision_output(raw, require_intervention=True)
    assert parsed is None and errors


@pytest.mark.parametrize("policy", ["selective", "shadow"])
def test_v2_template_has_strict_parseable_example_and_current_target_context(policy):
    context = {"source": "CARLA_GT_route_constant_velocity_proxy", "sim_time_s": 7.1,
               "actors": [{"id": 314, "conflict": False}],
               "signals": [{"id": "stop_271_0", "state": "Red"}],
               "candidates_summary": {"baseline_candidate_id": "route_center", "baseline_score": 1.0}}
    prompt = DecisionPromptBuilder().build(
        {"scene_description": "CURRENT_SCENE"}, nav_instruction="NAVIGATION",
        planning_constraints={"decision_mode": "semantic", "semantic_policy": policy,
                              "planner_context": context},
    )
    for text in ("CURRENT_SCENE", "NAVIGATION", "stop_271_0", '"id": 314',
                 "baseline_candidate_id", "相邻对向车辆未与路线产生时空冲突时正常行驶",
                 "不要等待整个路口清空", "制动时机与位置由数值规划决定",
                 "历史请求不代表执行成功", "未经验证的经验不作为停车依据"):
        assert text in prompt
    section = prompt.split("## 输出要求", 1)[1]
    example, _ = json.JSONDecoder().raw_decode(section[section.index("{"):])
    parsed, errors = parse_semantic_decision_output(json.dumps(example), require_intervention=True)
    assert not errors and parsed["intervention"]["needed"] is False
    assert parsed["lateral_decision"] == "FOLLOW_ROUTE"
    assert parsed["longitudinal_decision"] == "MAINTAIN"
    for forbidden in ('"trajectory":', '"target_speed":', '"optional_v":', "CARLA 闭环轨迹物理约束"):
        assert forbidden not in prompt


@pytest.mark.parametrize("policy", ["always", "selective", "shadow"])
def test_system_and_user_templates_select_same_policy_contract(policy):
    cfg = Config(overrides={"carla": {"planning": {"backend": "pdm_closed", "semantic_policy": policy}}})
    system = get_decision_system_prompt(cfg)
    user = get_semantic_prompt_template(policy)
    assert ("intervention" in system) == (policy != "always")
    assert ("intervention" in user) == (policy != "always")


def test_snapshot_json_is_bounded_complete_finite_and_does_not_mutate_input():
    context = {"actors": [{"id": i, "details": {"text": "\x00" * 2000,
                "rows": list(range(1000))}} for i in range(1000)],
               "signals": [{"id": "stop_123_0", "state": "Red"}],
               "source": "CURRENT_GT", "sim_time_s": 10,
               "candidates_summary": {"baseline_score": 1.0, "invalid": float("nan")}}
    before = copy.deepcopy(context)
    rendered = DecisionPromptBuilder._render_planner_context(context)
    document = rendered.splitlines()[1]
    parsed = json.loads(document)
    assert len(document) < 12000
    assert parsed["context_limited"] is True
    assert parsed["snapshot"]["source"] == "CURRENT_GT"
    assert parsed["snapshot"]["signals"][0]["id"] == "stop_123_0"
    assert "invalid" not in parsed["snapshot"]["candidates_summary"]
    assert len(parsed["snapshot"]["actors"]) <= 24
    assert context["actors"] == before["actors"]
    assert "缺失或因长度省略的目标均为未知" in rendered


def test_missing_context_never_invents_empty_road_or_ids():
    rendered = DecisionPromptBuilder._render_planner_context(None)
    assert "未知" in rendered and "不得编造ID" in rendered
    assert "道路畅通" not in rendered


@pytest.mark.parametrize("policy,expected", [("always", "success"), ("selective", "failed"), ("shadow", "failed")])
def test_pipeline_parser_policy_does_not_synthesize_a_v2_request(policy, expected):
    from src.vla_memory.pipeline.online_loop import OnlineDrivingLoop
    parsed, errors, status, fallback = OnlineDrivingLoop._parse_or_fallback(
        SimpleNamespace(), json.dumps(decision()), {}, "straight",
        decision_mode="semantic", semantic_policy=policy,
    )
    assert status == expected and not fallback
    assert "intervention" not in parsed
    assert bool(errors) == (expected == "failed")
