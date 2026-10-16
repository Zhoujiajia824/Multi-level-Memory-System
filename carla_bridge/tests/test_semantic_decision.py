"""Semantic output never becomes a model-generated numerical trajectory."""
import json
from types import SimpleNamespace

import pytest

from src.vla_memory.decision.prompt_builder import DecisionPromptBuilder
from src.vla_memory.decision.semantic_output_parser import (
    LATERAL_DECISIONS,
    LONGITUDINAL_DECISIONS,
    parse_semantic_decision_output,
)


def _decision(**changes):
    result = {
        "lateral_decision": "FOLLOW_ROUTE",
        "longitudinal_decision": "MAINTAIN",
        "behavior_reason": "Follow the signed route; no conflict observed.",
        "risk_level": "low",
        "safety_notes": [],
    }
    result.update(changes)
    return result


@pytest.mark.parametrize("lateral", LATERAL_DECISIONS)
@pytest.mark.parametrize("longitudinal", LONGITUDINAL_DECISIONS)
def test_all_intents_preserved_without_generated_numeric_values(lateral, longitudinal):
    raw = json.dumps(_decision(lateral_decision=lateral, longitudinal_decision=longitudinal))
    parsed, errors = parse_semantic_decision_output(raw)
    assert not errors
    assert parsed["lateral_decision"] == lateral
    assert parsed["longitudinal_decision"] == longitudinal
    assert parsed["trajectory"] == []
    assert "target_speed" not in parsed
    assert parsed["raw_response"] == raw
    assert parsed["fallback_used"] is False
    assert parsed["parser_status"] == "success"
    assert parsed["decision_mode"] == "semantic"


@pytest.mark.parametrize("field", list(_decision()))
def test_missing_field_rejected_without_defaults(field):
    value = _decision()
    value.pop(field)
    parsed, errors = parse_semantic_decision_output(json.dumps(value))
    assert parsed is None
    assert errors


@pytest.mark.parametrize("field,value", [
    ("trajectory", []), ("target_speed", 5), ("behavior", "KEEP_LANE"),
    ("optional_v", 6), ("confidence", 0.9), ("fallback_used", True),
])
def test_numerical_or_extra_output_rejected(field, value):
    parsed, errors = parse_semantic_decision_output(json.dumps(_decision(**{field: value})))
    assert parsed is None
    assert errors


@pytest.mark.parametrize("field,value", [
    ("lateral_decision", "TURN_LEFT"), ("lateral_decision", "keep_lane"),
    ("lateral_decision", ["KEEP_LANE"]), ("longitudinal_decision", "STOP "),
    ("longitudinal_decision", True), ("longitudinal_decision", 0),
    ("risk_level", "unknown"), ("risk_level", None),
    ("behavior_reason", "  "), ("behavior_reason", 123),
    ("safety_notes", "look ahead"), ("safety_notes", [None]),
    ("safety_notes", [""]),
])
def test_malformed_semantic_fields_do_not_guess_intent(field, value):
    parsed, errors = parse_semantic_decision_output(json.dumps(_decision(**{field: value})))
    assert parsed is None
    assert errors


@pytest.mark.parametrize("raw", [
    None, "", " ", "[]", "null", '{"lateral_decision":',
    '{"lateral_decision":"KEEP_LANE","lateral_decision":"FOLLOW_ROUTE"}',
    '{"risk_level":NaN}',
])
def test_invalid_or_ambiguous_json_rejected(raw):
    parsed, errors = parse_semantic_decision_output(raw)
    assert parsed is None
    assert errors


def test_code_fence_accepted_but_multiple_answers_rejected():
    raw = json.dumps(_decision())
    parsed, errors = parse_semantic_decision_output("```json\n" + raw + "\n```")
    assert parsed is not None and not errors
    parsed, errors = parse_semantic_decision_output(raw + "\n" + raw)
    assert parsed is None and errors


def test_compatibility_behavior_does_not_erase_lateral_semantics():
    parsed, errors = parse_semantic_decision_output(json.dumps(_decision(
        lateral_decision="CHANGE_LANE_LEFT", longitudinal_decision="STOP",
    )))
    assert not errors
    assert parsed["behavior"] == "STOP"
    assert parsed["lateral_decision"] == "CHANGE_LANE_LEFT"


def test_semantic_prompt_keeps_all_context_without_legacy_output_contract():
    prompt = DecisionPromptBuilder().build(
        scene_understanding={"scene_description": "CURRENT_SCENE"},
        ego_state={"speed": 7.5},
        history_trajectory=[{"x": -1, "y": 0}],
        nav_instruction="ROUTE_TURN_LEFT",
        short_term_summary="SHORT_MEMORY",
        mid_term_memories=[{"record": {"decision_reason": "MID_MEMORY"}}],
        long_term_rules_text="LONG_MEMORY",
        perception_objects=[{
            "category": "CURRENT_ORACLE_VEHICLE", "position_ego": [20, 0],
        }],
        image_layout="SIX_CAMERA_MOSAIC",
        # Semantic mode does not need any trajectory physics placeholders.
        planning_constraints={"decision_mode": "semantic"},
    )
    for expected in (
        "CURRENT_SCENE", "ROUTE_TURN_LEFT", "SHORT_MEMORY", "MID_MEMORY",
        "LONG_MEMORY", "CURRENT_ORACLE_VEHICLE", "SIX_CAMERA_MOSAIC", "7.50 m/s",
    ):
        assert expected in prompt
    for forbidden in ('"trajectory":', '"target_speed":', '"optional_v":',
                      "CARLA 闭环轨迹物理约束", "20 到 30 个 waypoint"):
        assert forbidden not in prompt
    assert '"lateral_decision": "FOLLOW_ROUTE"' in prompt
    assert '"longitudinal_decision": "MAINTAIN"' in prompt
    assert "{scene_block}" not in prompt


def test_semantic_memory_off_has_no_empty_memory_sections():
    prompt = DecisionPromptBuilder().build(
        {}, planning_constraints={"decision_mode": "semantic"},
    )
    assert "## 短期记忆" not in prompt
    assert "## 相似历史经验" not in prompt
    assert "## 长期记忆" not in prompt


def test_legacy_generic_prompt_still_requests_numerical_trajectory():
    prompt = DecisionPromptBuilder().build({})
    assert '"trajectory":' in prompt
    assert '"target_speed":' in prompt
    assert '"lateral_decision":' not in prompt


@pytest.mark.parametrize("object_record", [False, True])
@pytest.mark.parametrize("influence", [False, True])
def test_memory_distinguishes_model_request_from_selected_plan(object_record, influence):
    record = {
        "behavior": "KEEP_LANE",
        "decision_reason": "Safe planner candidate chosen.",
        "semantic_decision": _decision(
            lateral_decision="CHANGE_LANE_LEFT", longitudinal_decision="ACCELERATE",
            behavior_reason="MODEL_REQUEST_REASON",
        ),
        "planning_context": {
            "backend": "PDM_TEST_BACKEND", "semantic_influence": influence,
            "selected_id": "center_cruise", "executed_behavior": "KEEP_LANE",
        },
    }
    source = SimpleNamespace(**record) if object_record else record
    prompt = DecisionPromptBuilder().build(
        {}, mid_term_memories=[{"record": source}],
        planning_constraints={"decision_mode": "semantic"},
    )
    assert "请求意图，不代表已执行或成功" in prompt
    assert '"lateral_decision": "CHANGE_LANE_LEFT"' in prompt
    assert '"longitudinal_decision": "ACCELERATE"' in prompt
    assert "MODEL_REQUEST_REASON" in prompt
    assert "PDM_TEST_BACKEND" in prompt
    assert '"selected_id": "center_cruise"' in prompt
    assert '"executed_behavior": "KEEP_LANE"' in prompt
    assert "上述选择记录不能单独证明" in prompt
    if influence:
        assert "请求意图仍受规划器安全约束限制" in prompt
    else:
        assert "模型建议未参与该帧候选评分" in prompt


def test_old_memory_does_not_acquire_invented_planning_history():
    context = DecisionPromptBuilder._render_planning_memory_context({
        "behavior": "KEEP_LANE", "decision_reason": "Old memory.",
    })
    assert context == ""
