"""规则 Fallback 模块
====================
当 VLM 输出格式错误时，使用规则 fallback 生成可评测轨迹。
Fallback 只能用于 VLM 输出格式错误时，不能用于替代 VLM 决策。
所有行为（含转弯、变道、停车等）都会生成对应方向的轨迹。
输出中 fallback_used 标记为 true。
参数从 config/decision.yaml 读取。
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("rule_fallback")

# 默认参数（与 config/decision.yaml 保持一致）
DEFAULT_SPEED = 5.0
DEFAULT_BEHAVIOR = "KEEP_LANE"
DEFAULT_HORIZON_SECONDS = 3.0
DEFAULT_WAYPOINT_NUM = 25


def generate_fallback_trajectory(
    speed: float = DEFAULT_SPEED,
    horizon_seconds: float = DEFAULT_HORIZON_SECONDS,
    waypoint_num: int = DEFAULT_WAYPOINT_NUM,
    behavior: str = DEFAULT_BEHAVIOR,
) -> List[Dict[str, Any]]:
    """生成 fallback 轨迹。

    根据行为类型生成不同方向的轨迹：
    - 直行类（KEEP_LANE, FOLLOW, SLOW_DOWN）: y=0 直线前进
    - 左转/左变道（TURN_LEFT, CHANGE_LANE_LEFT）: y 正向偏移
    - 右转/右变道（TURN_RIGHT, CHANGE_LANE_RIGHT）: y 负向偏移
    - 停车（STOP）: x/y 均为 0 附近，速度为 0
    - 绕行（AVOID_OBSTACLE）: 先 y 正向再回归
    - 让行（YIELD）: 减速至 0

    Args:
        speed: 默认速度（m/s）。
        horizon_seconds: 时间范围（秒）。
        waypoint_num: 目标 waypoint 数量（默认 25）。
        behavior: 行为类型，影响轨迹形状。

    Returns:
        轨迹点列表，每个点包含 {t, x, y, optional_v}。
    """
    trajectory = []
    actual_dt = horizon_seconds / waypoint_num

    for i in range(waypoint_num):
        t = (i + 1) * actual_dt
        progress = (i + 1) / waypoint_num  # 0→1 进度

        if behavior == "STOP":
            # 停车：所有点在原点附近
            x = max(0.0, speed * 0.2 * (1.0 - progress))
            y = 0.0
            v = max(0.0, speed * (1.0 - progress))

        elif behavior == "YIELD":
            # 让行：减速至接近 0
            x = speed * 0.5 * t * (1.0 - progress * 0.8)
            y = 0.0
            v = max(0.0, speed * (1.0 - progress * 0.9))

        elif behavior == "TURN_LEFT":
            # 左转：x 前进 + y 正向偏移（左向为正）
            x = speed * t * 0.8
            y = speed * t * 0.5 * progress
            v = speed * 0.7

        elif behavior == "TURN_RIGHT":
            # 右转：x 前进 + y 负向偏移（右向为负）
            x = speed * t * 0.8
            y = -speed * t * 0.5 * progress
            v = speed * 0.7

        elif behavior == "CHANGE_LANE_LEFT":
            # 左变道：x 前进 + y 缓慢正向偏移
            x = speed * t
            y = 3.5 * progress  # 约 3.5 米横向偏移
            v = speed

        elif behavior == "CHANGE_LANE_RIGHT":
            # 右变道：x 前进 + y 缓慢负向偏移
            x = speed * t
            y = -3.5 * progress
            v = speed

        elif behavior == "AVOID_OBSTACLE":
            # 绕行：先偏移再回归，形成弧线
            x = speed * t
            offset = 2.0 * (1.0 - abs(2.0 * progress - 1.0))  # 先增后减
            y = offset
            v = speed * 0.8

        else:
            # KEEP_LANE / FOLLOW / SLOW_DOWN / UNKNOWN: 直线前进
            x = speed * t
            y = 0.0
            v = speed

        trajectory.append({
            "t": round(t, 3),
            "x": round(x, 3),
            "y": round(y, 3),
            "optional_v": round(v, 2),
        })

    logger.info(
        f"生成 fallback 轨迹: {len(trajectory)} 个 waypoint, "
        f"行为={behavior}, 速度={speed} m/s"
    )
    return trajectory


def generate_fallback_decision(
    ego_state: Optional[Dict] = None,
    nav_instruction: str = "",
) -> Dict[str, Any]:
    """生成 fallback 决策结果。

    仅在 VLM 输出格式错误时使用，不能替代 VLM 决策。
    输出中 fallback_used 标记为 true。

    Args:
        ego_state: 自车状态字典。
        nav_instruction: 导航语义。

    Returns:
        Fallback 决策字典，包含 fallback_used=True 标记。
    """
    speed = DEFAULT_SPEED
    if ego_state:
        speed = ego_state.get("speed", DEFAULT_SPEED)
        speed = max(0.0, min(speed, 30.0))

    # 根据导航语义和速度选择行为
    if speed < 1.0:
        behavior = "STOP"
    else:
        behavior_map = {
            "left_turn": "TURN_LEFT",
            "right_turn": "TURN_RIGHT",
            "lane_change_left": "CHANGE_LANE_LEFT",
            "lane_change_right": "CHANGE_LANE_RIGHT",
            "slow_or_stop": "SLOW_DOWN",
            "straight": "KEEP_LANE",
            "lane_follow": "KEEP_LANE",
        }
        behavior = behavior_map.get(nav_instruction, DEFAULT_BEHAVIOR)

    trajectory = generate_fallback_trajectory(
        speed=speed,
        behavior=behavior,
    )

    logger.warning(
        f"使用规则 Fallback 生成决策（仅因 VLM 输出格式错误）: "
        f"behavior={behavior}, speed={speed:.1f} m/s"
    )

    return {
        "behavior": behavior,
        "behavior_reason": "[Fallback] VLM 输出格式错误，使用规则生成",
        "target_speed": speed,
        "risk_level": "medium",
        "trajectory": trajectory,
        "safety_notes": ["此决策由规则 Fallback 生成，非 VLM 输出"],
        "fallback_used": True,
        "parser_status": "fallback",
    }
