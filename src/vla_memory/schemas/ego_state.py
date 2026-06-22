"""
自车状态数据模型
================
定义 EgoState 数据结构，包含位置、航向角、速度、加速度等字段。
第一版由 ego_pose 时间序列差分估计速度和加速度。
后续可接入 CAN bus expansion 获取更准确的车辆状态。
"""
from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, validator


class EgoState(BaseModel):
    """自车状态数据模型。

    表示某一时刻自车的完整运动状态信息。
    坐标系为全局坐标系（nuScenes 坐标系）。

    Attributes:
        timestamp: 时间戳（微秒）。
        x: 自车位置 x 坐标（米）。
        y: 自车位置 y 坐标（米）。
        z: 自车位置 z 坐标（米）。
        yaw: 航向角（弧度），从四元数转换而来，范围 [-pi, pi]。
        vx: x 方向速度（m/s），由相邻帧差分估计。
        vy: y 方向速度（m/s），由相邻帧差分估计。
        speed: 标量速度（m/s），sqrt(vx^2 + vy^2)。
        ax: x 方向加速度（m/s²），由三帧差分估计。
        ay: y 方向加速度（m/s²），由三帧差分估计。
        acceleration: 标量加速度（m/s²），可正可负。
        yaw_rate: 偏航角速率（rad/s）。P3 起来自 CAN bus 真值，差分回退时为 None。
        steering_angle: 方向盘转角（rad）。仅 CAN bus 可用，差分回退时为 None。
        throttle: 油门 [0,1]。仅 CAN bus 可用。
        brake: 刹车 [0,1]。仅 CAN bus 可用。
        gear: 档位标识（字符串）。仅 CAN bus 可用。
        source: 数据来源标识，``"pose_diff"`` | ``"can_bus"`` | ``"can_bus_pose_only"``。
    """
    timestamp: int = Field(0, description="时间戳（微秒）")
    x: float = Field(0.0, description="自车位置 x 坐标（米）")
    y: float = Field(0.0, description="自车位置 y 坐标（米）")
    z: float = Field(0.0, description="自车位置 z 坐标（米）")
    yaw: float = Field(0.0, description="航向角（弧度）")
    vx: float = Field(0.0, description="x 方向速度（m/s）")
    vy: float = Field(0.0, description="y 方向速度（m/s）")
    speed: float = Field(0.0, description="标量速度（m/s）")
    ax: float = Field(0.0, description="x 方向加速度（m/s²）")
    ay: float = Field(0.0, description="y 方向加速度（m/s²）")
    acceleration: float = Field(0.0, description="标量加速度（m/s²）")
    # P3 起 CAN bus 真值字段（可选）
    yaw_rate: Optional[float] = Field(None, description="偏航角速率 (rad/s)")
    steering_angle: Optional[float] = Field(None, description="方向盘转角 (rad)")
    throttle: Optional[float] = Field(None, description="油门 [0,1]")
    brake: Optional[float] = Field(None, description="刹车 [0,1]")
    gear: Optional[str] = Field(None, description="档位")
    source: str = Field(
        "pose_diff",
        description="数据来源: pose_diff | can_bus | can_bus_pose_only",
    )

    @validator("speed", pre=False, always=True)
    def _validate_speed(cls, v, values):
        """速度不可为负数。"""
        if v < 0:
            raise ValueError(f"速度不能为负数: {v}")
        return v

    def to_dict(self) -> dict:
        """转换为普通字典，用于 JSON 序列化和 VLM prompt 构建。

        约定：CAN bus 字段（yaw_rate / steering_angle / throttle / brake / gear）
        为 None 时不出现在输出中，保持 prompt 紧凑。
        ``source`` 始终输出，便于审计。
        """
        out = {
            "timestamp": self.timestamp,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "yaw": round(self.yaw, 6),
            "vx": round(self.vx, 4),
            "vy": round(self.vy, 4),
            "speed": round(self.speed, 4),
            "ax": round(self.ax, 4),
            "ay": round(self.ay, 4),
            "acceleration": round(self.acceleration, 4),
            "source": self.source,
        }
        if self.yaw_rate is not None:
            out["yaw_rate"] = round(self.yaw_rate, 6)
        if self.steering_angle is not None:
            out["steering_angle"] = round(self.steering_angle, 6)
        if self.throttle is not None:
            out["throttle"] = round(self.throttle, 4)
        if self.brake is not None:
            out["brake"] = round(self.brake, 4)
        if self.gear is not None:
            out["gear"] = self.gear
        return out

    def to_ego_centric(self) -> dict:
        """转换为 ego-centric 坐标系的描述字典，用于 VLM prompt。"""
        return self.to_dict()

    @staticmethod
    def quat_to_yaw(quat: list) -> float:
        """从 nuScenes 四元数 [w, x, y, z] 提取 yaw 角。

        Args:
            quat: 四元数列表 [w, x, y, z]。

        Returns:
            yaw 角（弧度），范围 [-pi, pi]。
        """
        if not quat or len(quat) < 4:
            return 0.0
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
