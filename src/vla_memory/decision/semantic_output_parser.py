"""Strict semantic decisions for a downstream numerical trajectory planner.

The model chooses intent only. Numerical trajectories and target speeds must
come from the planner; accepting them here would silently restore the old
model-as-trajectory-generator path.
"""
from __future__ import annotations

import json
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
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse the five-field semantic schema without filling missing intent.

    A complete JSON object, optionally enclosed in one Markdown code fence,
    is accepted. Extra fields, duplicate keys, numeric outputs and malformed
    enums are rejected. The internal empty ``trajectory`` only preserves
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
    unexpected = sorted(parsed.keys() - SEMANTIC_FIELDS)
    if missing:
        errors.append("缺少语义字段: " + ", ".join(missing))
    if unexpected:
        errors.append("语义输出包含不允许的字段: " + ", ".join(unexpected))
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
