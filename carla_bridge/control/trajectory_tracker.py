"""轨迹跟踪器：决策轨迹 -> carla.VehicleControl
================================================
把决策的 ego-centric 轨迹在**捕获时刻**一次性转成全局 waypoint（``coords``），
之后每个控制 tick 用 Pure Pursuit(横向) + PID(纵向) 算 ``carla.VehicleControl``。

默认目标速度取决策的 ``target_speed``；规划器可通过 ``follow_speed_profile=True``
启用 waypoint 的 ``t`` / ``optional_v`` 速度曲线。曲线按控制 tick 的仿真时间推进，
重规划时重新计时，超出曲线有效期立即刹车。
方向盘左正(项目约定)经 ``steer_sign`` 转成 CARLA 归一化 steer ∈ [-1,1]：
CARLA ``VehicleControl.steer`` 正值=右转，与项目左正相反，故 ``steer_sign=-1``。
"""
from __future__ import annotations

from bisect import bisect_right
import math
from typing import List, Optional, Tuple

import carla  # 须在 mulmem_carla(3.9) 运行

from carla_bridge.state import coords
from carla_bridge.control.pure_pursuit import PurePursuit
from carla_bridge.control.pid import PID


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TrajectoryTracker:
    """决策轨迹 -> 每_tick 的 carla.VehicleControl。"""

    def __init__(self, controller_cfg: dict, control_dt_s: float = 0.05):
        self.pp = PurePursuit(
            wheelbase_m=controller_cfg.get("wheelbase_m", 2.8),
            lookahead_min=controller_cfg.get("lookahead_min_m", 3.0),
            lookahead_max=controller_cfg.get("lookahead_max_m", 12.0),
            lookahead_k=controller_cfg.get("lookahead_k", 0.5),
        )
        pid_cfg = controller_cfg.get("pid", {}) or {}
        self.pid = PID(
            kp=pid_cfg.get("kp", 1.0),
            ki=pid_cfg.get("ki", 0.05),
            kd=pid_cfg.get("kd", 0.05),
            max_throttle=pid_cfg.get("max_throttle", 0.75),
            max_brake=pid_cfg.get("max_brake", 0.8),
            deadband_mps=pid_cfg.get("deadband_mps", 0.15),
        )
        self.max_steer_rad = controller_cfg.get("max_steer_rad", 0.6)
        # 项目 steer_rad 左正；CARLA VehicleControl.steer 右正 → 必须取反（+yaw=右转，实测）
        self.steer_sign = controller_cfg.get("steer_sign", -1)
        self.control_dt = control_dt_s
        self._global_wps: Optional[List[dict]] = None
        self._target_speed: float = 0.0
        self._speed_profile: Optional[List[Tuple[float, float]]] = None
        self._profile_elapsed_s: float = 0.0

    def set_trajectory(
        self,
        ego_centric_trajectory: List[dict],
        capture_ego_x: float,
        capture_ego_y: float,
        capture_ego_yaw_rad: float,
        target_speed: float,
        *,
        follow_speed_profile: bool = False,
    ) -> None:
        """设定本周期轨迹；启用速度曲线时每个点必须有有效的时间和非负速度。

        无效曲线抛出 ValueError，并清除上一周期轨迹，避免继续执行过期计划。
        空轨迹始终表示全刹车，不要求速度曲线字段。
        """
        self._global_wps = None
        self._speed_profile = None
        self._profile_elapsed_s = 0.0
        self.pp.reset()
        self.pid.soft_reset()  # 保留积分（起步助力），只清微分防踢
        if follow_speed_profile and ego_centric_trajectory:
            if not math.isfinite(self.control_dt) or self.control_dt <= 0:
                raise ValueError("Speed-profile tracking requires a positive finite control dt")
            profile = []
            for index, wp in enumerate(ego_centric_trajectory):
                try:
                    t, speed = float(wp["t"]), float(wp["optional_v"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid speed-profile waypoint {index}: t and optional_v are required") from exc
                if not math.isfinite(t) or not math.isfinite(speed) or t < 0 or speed < 0:
                    raise ValueError(f"Invalid speed-profile waypoint {index}: time and speed must be finite and nonnegative")
                if profile and t <= profile[-1][0]:
                    raise ValueError("Speed-profile waypoint times must be strictly increasing")
                profile.append((t, speed))
            self._speed_profile = profile
        self._global_wps = coords.trajectory_ego_to_global(
            ego_centric_trajectory, capture_ego_x, capture_ego_y, capture_ego_yaw_rad
        )
        self._target_speed = float(target_speed or 0.0)

    def has_trajectory(self) -> bool:
        return bool(self._global_wps)

    def compute_control(
        self, ego_x: float, ego_y: float, ego_yaw_rad: float, current_speed: float
    ) -> "carla.VehicleControl":
        """根据当前自车位姿/速度算本 tick 控制量。无轨迹则全刹。"""
        if not self._global_wps:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        target_speed = self._target_speed
        if self._speed_profile is not None:
            # This control will be applied during the next simulation tick. Looking
            # one tick ahead lets a trajectory starting at v(0)=0 accelerate.
            self._profile_elapsed_s += self.control_dt
            if self._profile_elapsed_s > self._speed_profile[-1][0] + 1e-9:
                return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            target_speed = self._profile_speed_at(self._profile_elapsed_s)
            if target_speed <= 1e-6:
                # Hold a planned stop even if no forward pursuit point remains or
                # the PID integral retained from the previous plan is positive.
                return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        steer_rad = self.pp.compute_steer(
            ego_x, ego_y, ego_yaw_rad, current_speed, self._global_wps
        )
        if not self.pp.target_valid:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        throttle, brake = self.pid.compute(
            target_speed - current_speed, dt=self.control_dt
        )
        steer_norm = _clamp(
            (steer_rad / self.max_steer_rad) * self.steer_sign, -1.0, 1.0
        )
        return carla.VehicleControl(throttle=throttle, steer=steer_norm, brake=brake)

    def _profile_speed_at(self, elapsed_s: float) -> float:
        """Linearly interpolate the validated profile at simulation time."""
        profile = self._speed_profile
        index = bisect_right([sample[0] for sample in profile], elapsed_s)
        if index == 0:
            return profile[0][1]
        if index >= len(profile):
            return profile[-1][1]
        t0, v0 = profile[index - 1]
        t1, v1 = profile[index]
        fraction = (elapsed_s - t0) / (t1 - t0)
        return v0 + fraction * (v1 - v0)
