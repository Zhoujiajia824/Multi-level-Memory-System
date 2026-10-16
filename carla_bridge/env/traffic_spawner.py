"""交通流生成
============
通过 Traffic Manager 生成自车周围的交通车辆（autopilot），以及行人（AI 控制器
随机走动）。用于场景的自定义交通流。行人生成失败时自动跳过（不同 CARLA 版本
walker API 略有差异），不影响主流程。
"""
from __future__ import annotations

import random
from typing import List

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("carla_traffic")


class TrafficSpawner:
    """交通车辆 + 行人生成器。"""

    def __init__(self, world, client, ego_vehicle, seed: int = 42):
        self.world = world
        self.client = client
        self.ego = ego_vehicle
        self.tm = client.get_trafficmanager(8000)
        try:
            self.tm.set_synchronous_mode(True)
        except Exception:
            pass
        self.rng = random.Random(seed)
        self._vehicles: List = []
        self._walkers: List = []
        self._walker_controllers: List = []

    def spawn_vehicles(self, num: int, radius: float = 80.0) -> int:
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = list(bp_lib.filter("vehicle.*"))
        spawn_points = self.world.get_map().get_spawn_points()
        ego_loc = self.ego.get_location()
        candidates = [p for p in spawn_points if p.location.distance(ego_loc) < radius]
        if not candidates:
            candidates = spawn_points
        self.rng.shuffle(candidates)

        count = 0
        for sp in candidates:
            if count >= num:
                break
            bp = self.rng.choice(vehicle_bps)
            if bp.has_attribute("color"):
                color = self.rng.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            try:
                veh = self.world.spawn_actor(bp, sp)
            except Exception:
                continue
            self._vehicles.append(veh)
            veh.set_autopilot(True)
            try:
                self.tm.vehicle_percentage_speed_difference(veh, self.rng.uniform(-10, 20))
            except Exception:
                pass
            count += 1
        logger.info("生成交通车辆 %d/%d", count, num)
        return count

    def spawn_walkers(self, num: int, radius: float = 60.0) -> int:
        """生成行人（best-effort：API 失败则跳过）。"""
        bp_lib = self.world.get_blueprint_library()
        walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
        if not walker_bps:
            return 0
        try:
            controller_bp = bp_lib.find("controller.ai.walker")
        except Exception:
            logger.warning("无 walker 控制器蓝图，跳过行人生成")
            return 0

        ego_loc = self.ego.get_location()
        count = 0
        for _ in range(num * 4):
            if count >= num:
                break
            loc = carla.Location(
                x=ego_loc.x + self.rng.uniform(-radius, radius),
                y=ego_loc.y + self.rng.uniform(-radius, radius),
                z=ego_loc.z + 1.0,
            )
            wp = self.world.get_map().get_waypoint(loc, project_to_road=True)
            if wp is None:
                continue
            spawn = carla.Transform(wp.transform.location)
            bp = self.rng.choice(walker_bps)
            try:
                walker = self.world.spawn_actor(bp, spawn)
                self._walkers.append(walker)
                controller = self.world.spawn_actor(controller_bp, spawn, attach_to=walker)
                self._walker_controllers.append(controller)
                controller.start()
                target = self._random_street_location(wp.transform.location)
                if target is not None:
                    controller.go_to_location(target)
                controller.set_max_speed(self.rng.uniform(1.0, 1.5))
            except Exception as e:
                logger.debug("行人生成失败: %s", e)
                continue
            count += 1
        logger.info("生成行人 %d/%d", count, num)
        return count

    def _random_street_location(self, near_loc):
        """best-effort 取一个街道随机点作为行人目标。"""
        for _ in range(5):
            try:
                loc = self.world.get_random_location_from_street()
                if loc is not None:
                    return loc
            except Exception:
                return None
        return None

    def destroy(self) -> None:
        for c in self._walker_controllers:
            try:
                c.stop()
            except Exception:
                pass
        for actors in (self._walker_controllers, self._walkers, self._vehicles):
            for a in actors:
                try:
                    a.destroy()
                except Exception:
                    pass
        self._vehicles.clear()
        self._walkers.clear()
        self._walker_controllers.clear()

        try:
            self.tm.set_synchronous_mode(False)
        except Exception as exc:
            logger.warning("Traffic Manager 同步模式恢复失败: %s", exc)
