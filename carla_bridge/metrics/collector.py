"""闭环指标采集
==============
逐 tick 采集闭环安全/舒适指标，run 结束产出 summary dict 供报告。

* 碰撞：由 ``sensor.other.collision`` 触发 -> :meth:`on_collision`。
* 路线完成度：来自 ``RoutePlanner.progress_fraction``。
* 闯红灯：自车近距红灯且仍前进（带冷却，避免单次停车重复计数）。
* 逆行：自车朝向与所在车道方向反向（连续 3 tick 判一次）。
* 超速：速度超过限速的 tick 数。
* 舒适度：纵向最大加速度 / 最大 jerk。

原 ADE/FDE 不适用闭环（未来由自车决策产生），故用本指标集。
"""
from __future__ import annotations

import math
from typing import List, Optional

import carla  # 须在 mulmem_carla(3.9) 运行

from carla_bridge.state import coords


class MetricsCollector:
    """闭环指标采集器。"""

    def __init__(
        self,
        world,
        route,
        speed_limit_mps: float = 13.9,   # ~50 km/h
        dt_s: float = 0.1,               # = fixed_delta_seconds（10Hz）
    ):
        self.world = world
        self.route = route
        self.speed_limit = speed_limit_mps
        self.dt = dt_s

        self.collision_count = 0
        self.red_light_violations = 0
        self.wrong_way_violations = 0
        self.speeding_ticks = 0
        self.total_ticks = 0

        self._speeds: List[float] = []
        self._accels: List[float] = []
        self._max_speed = 0.0
        self._max_accel = 0.0

        self._wrong_way_streak = 0
        self._last_rl_time_s = -1e9  # 闯红灯冷却（sim 秒，2s）

    # ------------------------------------------------------------------

    def on_collision(self) -> None:
        self.collision_count += 1

    def on_tick(self, ego_vehicle, elapsed_s: float, ego_state: dict) -> None:
        self.total_ticks += 1
        speed = float(ego_state.get("speed", 0.0))
        accel = float(ego_state.get("acceleration", 0.0))
        self._speeds.append(speed)
        self._accels.append(accel)
        self._max_speed = max(self._max_speed, speed)
        self._max_accel = max(self._max_accel, abs(accel))

        if speed > self.speed_limit:
            self.speeding_ticks += 1

        self._check_red_light(ego_vehicle, elapsed_s)
        self._check_wrong_way(ego_vehicle)

    # ------------------------------------------------------------------

    def _check_red_light(self, ego_vehicle, elapsed_s: float = None) -> None:
        """闯红灯判定：红灯 + trigger volume 在 ego 前向带内 + 仍在移动。

        用 trigger volume（灯的停止线判定盒）而非灯杆位置——交叉方向的灯离 ego
        也可能 <8m，按灯杆距离会误报。冷却基于 sim 秒（2s），与 tick 频率解耦。
        """
        if elapsed_s is not None and elapsed_s - self._last_rl_time_s < 2.0:
            return
        tf = ego_vehicle.get_transform()
        try:
            lights = self.world.get_actors().filter("traffic.traffic_light*")
        except Exception:
            return
        v = ego_vehicle.get_velocity()
        speed = (v.x * v.x + v.y * v.y) ** 0.5
        if speed <= 0.5:
            return
        ego_yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
        for tl in lights:
            try:
                if tl.state != carla.TrafficLightState.Red:
                    continue
                # trigger volume 中心（停止线判定盒）转全局再转 ego 前向带
                trig = tl.get_transform().transform(tl.get_trigger_volume().location)
                fwd, left = coords.global_to_ego(
                    trig.x, trig.y, tf.location.x, tf.location.y, ego_yaw
                )
                if 0.0 < fwd < 8.0 and abs(left) < 2.5:
                    self.red_light_violations += 1
                    self._last_rl_time_s = elapsed_s if elapsed_s is not None else 0.0
                    return
            except Exception:
                continue

    def _check_wrong_way(self, ego_vehicle) -> None:
        try:
            wp = self.world.get_map().get_waypoint(
                ego_vehicle.get_location(), project_to_road=True
            )
            if wp is None:
                return
            wp_yaw = coords.carla_yaw_deg_to_rad(wp.transform.rotation.yaw)
            ego_yaw = coords.carla_yaw_deg_to_rad(
                ego_vehicle.get_transform().rotation.yaw
            )
            diff = ego_yaw - wp_yaw
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff < -math.pi:
                diff += 2 * math.pi
            if abs(diff) > math.pi * 0.75:  # 反向行驶
                self._wrong_way_streak += 1
                if self._wrong_way_streak >= 3:
                    self.wrong_way_violations += 1
                    self._wrong_way_streak = 0
            else:
                self._wrong_way_streak = 0
        except Exception:
            pass

    # ------------------------------------------------------------------

    def route_completion(self) -> float:
        return self.route.progress_fraction() if self.route else 0.0

    def _max_jerk(self) -> float:
        if len(self._accels) < 3:
            return 0.0
        return max(
            abs(self._accels[i] - self._accels[i - 1]) / self.dt
            for i in range(1, len(self._accels))
        )

    def summary(self) -> dict:
        return {
            "total_ticks": self.total_ticks,
            "collision_count": self.collision_count,
            "route_completion": round(self.route_completion(), 4),
            "red_light_violations": self.red_light_violations,
            "wrong_way_violations": self.wrong_way_violations,
            "speeding_ticks": self.speeding_ticks,
            "max_speed_mps": round(self._max_speed, 3),
            "max_accel_mps2": round(self._max_accel, 3),
            "max_jerk_mps3": round(self._max_jerk(), 3),
        }
