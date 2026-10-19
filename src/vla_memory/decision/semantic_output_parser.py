"""Strict semantic decisions for a downstream numerical trajectory planner.

The model chooses intent only. Numerical trajectories and target speeds must
come from the planner; accepting them here would silently restore the old
model-as-trajectory-generator path.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.vla_memory.schemas.decision import VALID_RISK_LEVELS


LATERAL_DECISIONS = (
    "FOLLOW_ROUTE", "KEEP_LANE", "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT",
)
LONGITUDINAL_DECISIONS = (
    "MAINTAIN", "ACCELERATE", "DECELERATE", "FOLLOW", "YIELD", "STOP",
)
SEMANTIC_FIELDS = frozenset((
    "lateral_decision", "longitudinal_decision", "behavior_reason",
    "risk_level", "safety_notes",
))
INTERVENTION_FIELDS = frozenset((
    "needed", "reason_code", "target_actor_ids", "target_stop_line_id",
    "activation", "release", "valid_for_s",
))
INTERVENTION_REASON_CODES = (
    "none", "path_blocked", "interaction", "traffic_signal", "visibility",
    "special_rule", "route_change",
)
INTERVENTION_ACTIVATIONS = ("immediate", "while_conflict", "approach_stop_line")
INTERVENTION_RELEASES = (
    "condition_cleared", "target_passed", "signal_changed", "maneuver_complete",
)


def _validate_intervention(value: Any) -> List[str]:
    """Validate a request, never infer permission or physical applicability."""
    if not isinstance(value, dict):
        return ["intervention 必须为完整对象"]
    errors = []
    missing = sorted(INTERVENTION_FIELDS - value.keys())
    unexpected = sorted(value.keys() - INTERVENTION_FIELDS)
    if missing:
        errors.append("intervention 缺少字段: " + ", ".join(missing))
    if unexpected:
        errors.append("intervention 包含不允许的字段: " + ", ".join(unexpected))
    if type(value.get("needed")) is not bool:
        errors.append("intervention.needed 必须为布尔值")
    for field, choices in (
        ("reason_code", INTERVENTION_REASON_CODES),
        ("activation", INTERVENTION_ACTIVATIONS),
        ("release", INTERVENTION_RELEASES),
    ):
        if field in value and (
            not isinstance(value[field], str) or value[field] not in choices
        ):
            errors.append("无效的 intervention." + field + ": " + repr(value[field]))
    actor_ids = value.get("target_actor_ids")
    if not isinstance(actor_ids, list) or any(type(item) is not int for item in actor_ids):
        errors.append("intervention.target_actor_ids 必须为整数列表")
    elif len(actor_ids) > 8 or len(set(actor_ids)) != len(actor_ids):
        errors.append("intervention.target_actor_ids 最多8项且不得重复")
    stop_line_id = value.get("target_stop_line_id")
    if stop_line_id is not None and (
        not isinstance(stop_line_id, str) or not stop_line_id.strip()
    ):
        errors.append("intervention.target_stop_line_id 必须为非空字符串或 null")
    duration = value.get("valid_for_s")
    # Check the bounded range before isfinite so enormous JSON integers cannot
    # overflow the int-to-double conversion inside math.isfinite.
    if (type(duration) not in (int, float) or not 0.2 <= duration <= 6.0
            or not math.isfinite(duration)):
        errors.append("intervention.valid_for_s 必须为0.2至6秒的有限数字")
    if value.get("needed") is False:
        if value.get("reason_code") != "none":
            errors.append("无需介入时 intervention.reason_code 必须为 none")
        if actor_ids != [] or stop_line_id is not None:
            errors.append("无需介入时 intervention 目标必须为空")
    elif value.get("needed") is True and value.get("reason_code") == "none":
        errors.append("需要介入时 intervention.reason_code 不得为 none")
    return errors


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("重复 JSON 字段: " + key)
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ValueError("JSON 中不允许非有限数: " + value)


def semantic_behavior(lateral: str, longitudinal: str) -> str:
    """Legacy memory label; the two semantic fields remain authoritative.

    Existing memory records need one behavior label. Longitudinal safety
    actions have precedence over lateral intent for this summary only.
    """
    longitudinal_behavior = {
        "STOP": "STOP", "YIELD": "YIELD", "DECELERATE": "SLOW_DOWN",
        "FOLLOW": "FOLLOW",
    }
    if longitudinal in longitudinal_behavior:
        return longitudinal_behavior[longitudinal]
    if lateral in ("CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"):
        return lateral
    return "KEEP_LANE"


def parse_semantic_decision_output(
    raw_text: str,
    *,
    require_intervention: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse legacy five fields or explicit v2 intervention, without defaults.

    A complete JSON object, optionally enclosed in one Markdown code fence,
    is accepted. Extra fields, duplicate keys, numeric outputs and malformed
    enums are rejected. Selective/shadow callers require the v2 object; legacy
    output stays compatible but never acquires an inferred intervention. The
    internal empty ``trajectory`` only preserves
    compatibility with existing memory consumers; it is never executable.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, ["VLM 语义输出为空或非字符串"]
    text = raw_text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return None, ["VLM 语义输出无法解析为唯一有效 JSON 对象: " + str(exc)]
    if not isinstance(parsed, dict):
        return None, ["语义决策输出不是字典类型"]

    errors: List[str] = []
    missing = sorted(SEMANTIC_FIELDS - parsed.keys())
    unexpected = sorted(parsed.keys() - SEMANTIC_FIELDS - {"intervention"})
    if missing:
        errors.append("缺少语义字段: " + ", ".join(missing))
    if unexpected:
        errors.append("语义输出包含不允许的字段: " + ", ".join(unexpected))
    if "intervention" in parsed:
        errors.extend(_validate_intervention(parsed["intervention"]))
    elif require_intervention:
        errors.append("选择性语义模式缺少 intervention；不会从自由文本推断介入授权")
    for field, choices in (
        ("lateral_decision", LATERAL_DECISIONS),
        ("longitudinal_decision", LONGITUDINAL_DECISIONS),
        ("risk_level", VALID_RISK_LEVELS),
    ):
        if field in parsed and (
            not isinstance(parsed[field], str) or parsed[field] not in choices
        ):
            errors.append("无效的 " + field + ": " + repr(parsed[field]))
    reason = parsed.get("behavior_reason")
    if "behavior_reason" in parsed and (
        not isinstance(reason, str) or not reason.strip()
    ):
        errors.append("behavior_reason 必须为非空字符串")
    notes = parsed.get("safety_notes")
    if "safety_notes" in parsed and (
        not isinstance(notes, list)
        or any(not isinstance(item, str) or not item.strip() for item in notes)
    ):
        errors.append("safety_notes 必须为字符串列表，可为空列表")
    if errors:
        return None, errors

    parsed.update({
        "behavior": semantic_behavior(
            parsed["lateral_decision"], parsed["longitudinal_decision"],
        ),
        "trajectory": [],
        "decision_mode": "semantic",
        "raw_response": raw_text,
        "fallback_used": False,
        "parser_status": "success",
    })
    return parsed, []
