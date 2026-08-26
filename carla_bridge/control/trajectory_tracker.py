"""轨迹跟踪器：决策轨迹 -> carla.VehicleControl
================================================
把决策的 ego-centric 轨迹在**捕获时刻**一次性转成全局 waypoint（``coords``），
之后每个控制 tick 用 Pure Pursuit(横向) + PID(纵向) 算 ``carla.VehicleControl``。

目标速度取决策的 ``target_speed``（后续可细化为按 waypoint 的 ``optional_v``）。
方向盘左正(项目约定)经 ``steer_sign`` 转成 CARLA 归一化 steer ∈ [-1,1]：
CARLA ``VehicleControl.steer`` 正值=右转，与项目左正相反，故 ``steer_sign=-1``。
"""
from __future__ import annotations

from typing import List, Optional

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

    def set_trajectory(
        self,
        ego_centric_trajectory: List[dict],
        capture_ego_x: float,
        capture_ego_y: float,
        capture_ego_yaw_rad: float,
        target_speed: float,
    ) -> None:
        """设定本周期决策轨迹（ego-centric）+ 捕获位姿 + 目标速度。"""
        self._global_wps = coords.trajectory_ego_to_global(
            ego_centric_trajectory, capture_ego_x, capture_ego_y, capture_ego_yaw_rad
        )
        self._target_speed = float(target_speed or 0.0)
        self.pid.soft_reset()  # 保留积分（起步助力），只清微分防踢

    def has_trajectory(self) -> bool:
        return bool(self._global_wps)

    def compute_control(
        self, ego_x: float, ego_y: float, ego_yaw_rad: float, current_speed: float
    ) -> "carla.VehicleControl":
        """根据当前自车位姿/速度算本 tick 控制量。无轨迹则全刹。"""
        if not self._global_wps:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        steer_rad = self.pp.compute_steer(
            ego_x, ego_y, ego_yaw_rad, current_speed, self._global_wps
        )
        throttle, brake = self.pid.compute(
            self._target_speed - current_speed, dt=self.control_dt
        )
        steer_norm = _clamp(
            (steer_rad / self.max_steer_rad) * self.steer_sign, -1.0, 1.0
        )
        return carla.VehicleControl(throttle=throttle, steer=steer_norm, brake=brake)
