"""Scripted NPC 控制器：NOA 路由跟随 + ACC 跟车
================================================
让一辆交通流车辆沿指定路由行驶（NOA 导航），并自适应跟车（ACC：前方有车则减速）。
复用 carla_bridge 的 Pure Pursuit(横向) + PID(纵向)，不依赖 CARLA agents。

ACC 语义：每 tick 扫描前方 ``max_dist`` 内、横向带 ``lateral_band`` 内的车辆（同车道
前方），若有则把目标速度降到前车速度（跟车）；前方无车则按 ``target_speed`` 巡航。
"""
from __future__ import annotations

import math
from typing import List, Optional

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.logging_utils import get_logger
from carla_bridge.state import coords
from carla_bridge.control.pure_pursuit import PurePursuit
from carla_bridge.control.pid import PID

logger = get_logger("carla_scripted")


class ScriptedVehicleController:
    """驱动一辆 NPC 沿全局路由行驶，带 ACC。"""

    def __init__(
        self,
        vehicle,
        waypoints: List[carla.Location],
        target_speed: float,
        controller_cfg: dict,
        acc_enabled: bool = True,
        dt: float = 0.1,
    ):
        self.vehicle = vehicle
        self.target_speed = float(target_speed)
        self.acc_enabled = acc_enabled
        self.dt = dt
        self._global_wps: List[dict] = [{"x": wp.x, "y": wp.y} for wp in waypoints]
        self._end_loc = waypoints[-1] if waypoints else None
        self._finished = False

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
        )
        self.max_steer_rad = controller_cfg.get("max_steer_rad", 0.6)
        # 项目 steer_rad 左正 ↔ CARLA steer 右正，取反（与 trajectory_tracker 一致）
        self.steer_sign = controller_cfg.get("steer_sign", -1)

    def tick(self, world) -> None:
        """每控制 tick 调用：算控制量并 apply。到终点停止。"""
        if self._finished or not self._global_wps:
            return
        tf = self.vehicle.get_transform()
        loc = tf.location
        yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
        v = self.vehicle.get_velocity()
        speed = math.hypot(v.x, v.y)

        # ACC：前方同车道有车 -> 降到前车速度
        target = self.target_speed
        if self.acc_enabled:
            lead_speed = self._lead_ahead_speed(world, loc, yaw)
            if lead_speed is not None:
                target = min(target, lead_speed)

        steer_rad = self.pp.compute_steer(loc.x, loc.y, yaw, speed, self._global_wps)
        throttle, brake = self.pid.compute(target - speed, dt=self.dt)
        steer_norm = max(-1.0, min(1.0, (steer_rad / self.max_steer_rad) * self.steer_sign))
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer_norm, brake=brake)
        )

        # 到终点
        if self._end_loc is not None and loc.distance(self._end_loc) < 5.0:
            self._finished = True
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            logger.info("scripted 车辆到达终点，停止")

    def _lead_ahead_speed(
        self, world, loc, yaw, max_dist: float = 25.0, lateral_band: float = 4.0
    ) -> Optional[float]:
        """前方 max_dist 内、横向带 lateral_band 内有车 -> 返回其速度；否则 None。"""
        try:
            actors = world.get_actors()
        except Exception:
            return None
        for actor in actors:
            if actor.id == self.vehicle.id:
                continue
            if not getattr(actor, "type_id", "").startswith("vehicle."):
                continue
            try:
                a_loc = actor.get_location()
            except Exception:
                continue
            fwd, left = coords.global_to_ego(a_loc.x, a_loc.y, loc.x, loc.y, yaw)
            if 0 < fwd < max_dist and abs(left) < lateral_band:
                v = actor.get_velocity()
                return math.hypot(v.x, v.y)
        return None

    def destroy(self) -> None:
        try:
            self.vehicle.destroy()
        except Exception:
            pass
