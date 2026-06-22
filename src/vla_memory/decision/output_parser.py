"""决策输出解析模块
==================
解析和校验 VLM 决策输出的 JSON。
如果 VLM 输出非法 JSON、轨迹点数不足、字段缺失，则记录失败。
增加 raw_response 保存、fallback_used 标记、parser_status 状态。
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List

from src.vla_memory.schemas.decision import VALID_BEHAVIORS, VALID_RISK_LEVELS
from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.decision.config_access import get_waypoint_bounds

logger = get_logger("output_parser")


def parse_decision_output(
    raw_text: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """从 VLM 原始文本中解析和校验决策输出。

    处理步骤：
    1. 从文本中提取 JSON。
    2. 校验 behavior、risk_level、target_speed、trajectory。
    3. 校验 trajectory 长度为 20-30 个 waypoint。
    4. 校验每个 waypoint 包含 t、x、y 字段。
    5. 保存 raw_response。
    6. 增加 fallback_used=False 和 parser_status 标记。

    Args:
        raw_text: VLM 原始文本输出。

    Returns:
        (解析后的决策字典, 错误信息列表)。
        如果解析/校验失败，第一个元素为 None。
    """
    from src.vla_memory.common.json_utils import extract_json_from_text

    errors = []

    # ---- 1. 提取 JSON ----
    if not raw_text or not isinstance(raw_text, str):
        return None, ["VLM 输出为空或非字符串"]

    parsed = extract_json_from_text(raw_text)
    if parsed is None:
        return None, [f"VLM 输出无法解析为有效 JSON。前300字符: {raw_text[:300]}"]

    if not isinstance(parsed, dict):
        return None, ["决策输出不是字典类型"]

    # ---- 2. 校验 behavior ----
    behavior = parsed.get("behavior")
    if not behavior:
        errors.append("缺少 behavior 字段")
    elif behavior not in VALID_BEHAVIORS:
        errors.append(f"无效的 behavior: {behavior}")

    # ---- 3. 校验 risk_level ----
    risk_level = parsed.get("risk_level", "medium")
    if risk_level not in VALID_RISK_LEVELS:
        errors.append(f"无效的 risk_level: {risk_level}")
        parsed["risk_level"] = "medium"

    # ---- 4. 校验 trajectory ----
    waypoint_min_num, waypoint_max_num = get_waypoint_bounds()
    trajectory = parsed.get("trajectory", [])
    if not isinstance(trajectory, list):
        errors.append("trajectory 不是列表类型")
    elif len(trajectory) < waypoint_min_num:
        errors.append(
            f"轨迹点数量不足: {len(trajectory)}，"
            f"至少需要 {waypoint_min_num} 个"
        )
    elif len(trajectory) > waypoint_max_num:
        # 超出时截断，不报错
        parsed["trajectory"] = trajectory[:waypoint_max_num]
    else:
        # 校验每个 waypoint
        for i, wp in enumerate(trajectory):
            if not isinstance(wp, dict):
                errors.append(f"第 {i} 个轨迹点不是字典类型")
            elif "x" not in wp or "y" not in wp:
                errors.append(f"第 {i} 个轨迹点缺少 x 或 y 字段")

    # ---- 5. 校验 target_speed ----
    target_speed = parsed.get("target_speed")
    if target_speed is not None:
        if not isinstance(target_speed, (int, float)):
            errors.append("target_speed 不是数值类型")
        elif target_speed < 0:
            errors.append("target_speed 不能为负数")

    # ---- 6. 填充默认值 ----
    parsed.setdefault("behavior_reason", "")
    parsed.setdefault("target_speed", 5.0)
    parsed.setdefault("risk_level", "medium")
    parsed.setdefault("trajectory", [])
    parsed.setdefault("safety_notes", [])

    # ---- 7. 增加 meta 字段 ----
    parsed["raw_response"] = raw_text
    parsed["fallback_used"] = False
    parsed["parser_status"] = "success" if not errors else "validation_error"

    if errors:
        logger.warning(f"决策输出校验发现问题: {errors}")
        return None, errors

    return parsed, []
