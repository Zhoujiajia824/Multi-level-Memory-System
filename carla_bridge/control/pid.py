"""PID 纵向速度跟踪
==================
目标速度 - 当前速度 -> throttle / brake。``u > 0`` 踩油门，``u < 0`` 踩刹车，
二者互斥（不同时为正）。含积分抗饱和。

纯数学，不依赖 carla。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple


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
        self._prev_measurement = None

    def reset(self) -> None:
        self._i = 0.0
        self._prev_err = None
        self._prev_measurement = None

    def soft_reset(self) -> None:
        """旧版定速轨迹切换时清微分基准，保留未饱和的积分。"""
        self._prev_err = None
        self._prev_measurement = None

    def compute(
        self, speed_err: float, dt: float = 0.05, *,
        current_speed: Optional[float] = None, allow_coast: bool = True,
    ) -> Tuple[float, float]:
        """``speed_err = target - current``。返回 ``(throttle, brake)``，均 ∈ [0,1]。

        提供 current_speed 时对测量值求微分，避免重规划目标速度跳变的微分踢动。
        死区内可滑行；有计划停车时 allow_coast=False，继续跟踪小速度误差。
        饱和时停止同向积分，误差反向时释放旧积分，避免减速命令仍输出油门。
        """
        if not math.isfinite(speed_err) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("PID requires finite speed error and positive finite dt")
        if current_speed is not None:
            if not math.isfinite(current_speed):
                raise ValueError("PID requires finite measured speed")
            d = (0.0 if self._prev_measurement is None else
                 -(current_speed - self._prev_measurement) / dt)
        else:
            d = 0.0 if self._prev_err is None else (speed_err - self._prev_err) / dt
        # Update feedback even while coasting; resuming control must not compare
        # against a stale sample from several ticks ago.
        self._prev_measurement = current_speed
        self._prev_err = speed_err
        if allow_coast and abs(speed_err) < self.deadband:
            self._i = 0.0
            return 0.0, 0.0
        if self._i * speed_err < 0.0:
            self._i = 0.0
        proposed_i = max(-5.0, min(5.0, self._i + speed_err * dt))
        proposed_u = self.kp * speed_err + self.ki * proposed_i + self.kd * d
        if not ((proposed_u > self.max_throttle and speed_err > 0.0) or
                (proposed_u < -self.max_brake and speed_err < 0.0)):
            self._i = proposed_i
        u = self.kp * speed_err + self.ki * self._i + self.kd * d
        # Neither stale integral nor derivative damping may accelerate an
        # overspeed vehicle, or brake a vehicle that is below its target speed.
        if speed_err <= 0.0:
            u = min(0.0, u)
        if speed_err >= 0.0:
            u = max(0.0, u)
        if u > 0:
            return min(self.max_throttle, u), 0.0
        return 0.0, min(self.max_brake, -u)
