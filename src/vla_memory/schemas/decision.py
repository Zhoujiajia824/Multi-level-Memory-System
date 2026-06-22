"""
决策输出数据模型
================
定义决策模块输出的数据结构 DecisionOutput。
包含行为决策、原因摘要、目标速度、风险等级、轨迹、安全备注。
behavior 和 risk_level 使用严格枚举校验。trajectory 支持 waypoint 数量校验。
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator


# ===================== 有效枚举值常量 =====================

VALID_BEHAVIORS = [
    "KEEP_LANE", "FOLLOW", "SLOW_DOWN", "STOP",
    "TURN_LEFT", "TURN_RIGHT",
    "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT",
    "AVOID_OBSTACLE", "YIELD", "UNKNOWN",
]
"""有效的行为决策枚举列表。"""

VALID_RISK_LEVELS = ["low", "medium", "high"]
"""有效的风险等级枚举列表。"""


class DecisionOutput(BaseModel):
    """决策输出数据模型。

    VLM 输出的驾驶决策结果，包含行为、轨迹和安全评估。
    必须通过 JSON schema 校验后才能被后续模块使用。

    Attributes:
        behavior: 行为决策，必须是 VALID_BEHAVIORS 之一。
        behavior_reason: 可审计的因果摘要，不输出冗长思维链。
        target_speed: 目标速度（m/s）。
        risk_level: 风险等级，必须是 VALID_RISK_LEVELS 之一。
        trajectory: 轨迹点列表，每个点必须包含 x 和 y 字段。
        safety_notes: 安全注意事项列表。
    """
    behavior: str = Field(..., description="行为决策")
    behavior_reason: str = Field("", description="行为原因摘要")
    target_speed: float = Field(5.0, description="目标速度 m/s")
    risk_level: str = Field("medium", description="风险等级")
    trajectory: List[dict] = Field(..., description="轨迹点列表")
    safety_notes: List[str] = Field(default_factory=list, description="安全注意事项")

    @validator("behavior")
    def _validate_behavior(cls, v):
        """校验行为决策是否合法。"""
        if v not in VALID_BEHAVIORS:
            raise ValueError(f"无效的行为决策: '{v}'，有效值为: {VALID_BEHAVIORS}")
        return v

    @validator("risk_level")
    def _validate_risk_level(cls, v):
        """校验风险等级是否合法。"""
        if v not in VALID_RISK_LEVELS:
            raise ValueError(f"无效的风险等级: '{v}'，有效值为: {VALID_RISK_LEVELS}")
        return v

    @validator("target_speed")
    def _validate_target_speed(cls, v):
        """校验目标速度。"""
        if v < 0:
            raise ValueError(f"目标速度不能为负数: {v}")
        if v > 50.0:
            raise ValueError(f"目标速度异常大: {v} m/s，上限 50 m/s")
        return v

    @validator("trajectory")
    def _validate_trajectory(cls, v):
        """校验轨迹点数量和字段完整性。"""
        if not isinstance(v, list):
            raise ValueError("轨迹必须是列表类型")
        if len(v) < 5:
            raise ValueError(f"轨迹点数量过少: {len(v)}，至少需要 5 个")
        for i, wp in enumerate(v):
            if not isinstance(wp, dict):
                raise ValueError(f"第 {i} 个轨迹点不是字典类型")
            if "x" not in wp or "y" not in wp:
                raise ValueError(f"第 {i} 个轨迹点缺少 x 或 y 字段")
        return v

    def waypoint_count(self) -> int:
        """返回轨迹点数量。"""
        return len(self.trajectory)

    def check_waypoint_range(self, min_num: Optional[int] = None, max_num: Optional[int] = None) -> bool:
        """检查轨迹点数量是否在 [min_num, max_num] 范围内。

        默认值从 config/decision.yaml -> trajectory.waypoint_min_num / waypoint_max_num 读取，
        使路点约束的"单一来源"在全链生效。
        """
        if min_num is None or max_num is None:
            # 懒加载避免与 decision/config_access -> schemas/decision 互相 import 形成循环
            from src.vla_memory.decision.config_access import get_waypoint_bounds
            _min, _max = get_waypoint_bounds()
            min_num = min_num if min_num is not None else _min
            max_num = max_num if max_num is not None else _max
        return min_num <= len(self.trajectory) <= max_num
