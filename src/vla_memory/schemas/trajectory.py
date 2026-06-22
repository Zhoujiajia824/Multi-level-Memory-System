"""
轨迹数据模型
============
定义 TrajectoryPoint 和 Trajectory 数据结构。
坐标系为 ego-centric：x 前向，y 左向，单位米。
支持 waypoint 数量校验和轨迹有效性判断。
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TrajectoryPoint(BaseModel):
    """单个轨迹点。

    Attributes:
        t: 相对当前帧的时间（秒），正值表示未来，负值表示过去。
        x: ego-centric x 坐标（米），前向为正。
        y: ego-centric y 坐标（米），左向为正。
        optional_v: 可选的瞬时速度（m/s）。
    """
    t: float = Field(..., description="相对时间（秒）")
    x: float = Field(..., description="ego-centric x 坐标（米），前向为正")
    y: float = Field(..., description="ego-centric y 坐标（米），左向为正")
    optional_v: Optional[float] = Field(None, description="可选瞬时速度（m/s）")

    def to_dict(self) -> dict:
        """转换为字典。"""
        d = {"t": round(self.t, 4), "x": round(self.x, 4), "y": round(self.y, 4)}
        if self.optional_v is not None:
            d["optional_v"] = round(self.optional_v, 4)
        return d


class Trajectory(BaseModel):
    """轨迹数据模型。

    表示一组有序的轨迹点，包含坐标系说明和可选的实际历史时长。

    Attributes:
        coordinate_system: 坐标系名称，默认 'ego_centric'。
        points: 轨迹点列表。
        history_seconds_actual: 实际覆盖的历史时长（秒），可选。
    """
    coordinate_system: str = Field(
        default="ego_centric",
        description="坐标系: ego_centric（x前向，y左向，单位米）",
    )
    points: List[TrajectoryPoint] = Field(
        ..., description="轨迹点列表",
    )
    history_seconds_actual: Optional[float] = Field(
        None, description="实际覆盖的历史时长（秒）",
    )

    # P4 起不对 points 做非空校验：空轨迹是合法状态（无历史/无未来）。
    # 下游使用方（full_demo_pipeline.enrich_keyframes_with_state）已通过
    # try/except 兜底空轨迹情况。

    def waypoint_count(self) -> int:
        """返回轨迹点数量。"""
        return len(self.points)

    def check_min_waypoints(self, min_num: int = 20) -> bool:
        """检查轨迹点数量是否满足最小要求。

        Args:
            min_num: 最小 waypoint 数量。

        Returns:
            是否满足要求。
        """
        return len(self.points) >= min_num

    def check_waypoint_range(self, min_num: int = 20, max_num: int = 30) -> bool:
        """检查轨迹点数量是否在 [min_num, max_num] 范围内。

        Args:
            min_num: 最小 waypoint 数量。
            max_num: 最大 waypoint 数量。

        Returns:
            是否在有效范围内。
        """
        return min_num <= len(self.points) <= max_num

    def to_list(self) -> List[dict]:
        """转换为字典列表，用于 JSON 序列化。"""
        return [p.to_dict() for p in self.points]

    @classmethod
    def from_list(cls, data: List[dict]) -> "Trajectory":
        """从字典列表创建轨迹。

        Args:
            data: 包含 t, x, y 等键的字典列表。

        Returns:
            Trajectory 实例。
        """
        points = [TrajectoryPoint(**wp) for wp in data]
        return cls(points=points)
