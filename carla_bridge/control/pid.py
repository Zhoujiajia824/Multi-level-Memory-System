"""PID 纵向速度跟踪
==================
目标速度 - 当前速度 -> throttle / brake。``u > 0`` 踩油门，``u < 0`` 踩刹车，
二者互斥（不同时为正）。含积分抗饱和。

纯数学，不依赖 carla。
"""
from __future__ import annotations

from typing import Tuple


class PID:
    """纵向速度 PID。"""

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.05,
        kd: float = 0.05,
        max_throttle: float = 0.75,
        max_brake: float = 0.8,
        deadband_mps: float = 0.15,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.deadband = deadband_mps
        self._i = 0.0
        self._prev_err = None

    def reset(self) -> None:
        self._i = 0.0
        self._prev_err = None

    def soft_reset(self) -> None:
        """重规划切换轨迹时调用：清微分项防踢，**保留积分**（起步仍有 I 项助力）。"""
        self._prev_err = None

    def compute(self, speed_err: float, dt: float = 0.05) -> Tuple[float, float]:
        """``speed_err = target - current``。返回 ``(throttle, brake)``，均 ∈ [0,1]。

        死区：|误差| < deadband 时输出 (0, 0) 滑行，防油门/刹车高频抖动。
        """
        if abs(speed_err) < self.deadband:
            return 0.0, 0.0
        self._i += speed_err * dt
        self._i = max(-5.0, min(5.0, self._i))  # 抗积分饱和
        d = 0.0 if self._prev_err is None else (speed_err - self._prev_err) / dt
        self._prev_err = speed_err
        u = self.kp * speed_err + self.ki * self._i + self.kd * d
        if u > 0:
            return min(self.max_throttle, u), 0.0
        return 0.0, min(self.max_brake, -u)
