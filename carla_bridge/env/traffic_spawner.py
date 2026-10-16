"""交通流生成
============
通过 Traffic Manager 生成自车周围的交通车辆（autopilot），以及行人（AI 控制器
随机走动）。用于场景的自定义交通流。行人生成失败时自动跳过（不同 CARLA 版本
walker API 略有差异），不影响主流程。
"""
from __future__ import annotations

import math
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
        self.tm_port = 8000
        self.tm = client.get_trafficmanager(self.tm_port)
        self._tm_synchronous = False
        try:
            self.tm.set_synchronous_mode(True)
            self._tm_synchronous = True
        except Exception:
            pass
        self.rng = random.Random(seed)
        self._vehicles: List = []
        self._walkers: List = []
        self._walker_controllers: List = []
        self.strict = False
        self.reserved_locations = []
        self.spawn_manifest = {"vehicles": [], "walkers": []}
        self.night_lights = False

    def configure_reproducibility(self, seed, reserved_locations):
        self.strict = True
        if not self._tm_synchronous:
            raise RuntimeError("paper experiments require synchronous Traffic Manager")
        self.tm.set_random_device_seed(int(seed))
        self.world.set_pedestrians_seed(int(seed))
        self.reserved_locations = reserved_locations
        self.night_lights = self.world.get_weather().sun_altitude_angle < 0

    def spawn_vehicles(self, num: int, radius: float = 80.0) -> int:
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = list(bp_lib.filter("vehicle.*"))
        if self.strict:
            vehicle_bps = sorted([bp for bp in vehicle_bps if bp.has_attribute("number_of_wheels")
                                  and bp.get_attribute("number_of_wheels").as_int() == 4], key=lambda bp: bp.id)
            if not vehicle_bps:
                raise RuntimeError("no four-wheel traffic vehicle blueprints")
        spawn_points = self.world.get_map().get_spawn_points()
        ego_loc = self.ego.get_location()
        candidates = [p for p in spawn_points if p.location.distance(ego_loc) < radius]
        if self.strict:
            candidates = [p for p in candidates if p.location.distance(ego_loc) > 8 and
                          all(p.location.distance(site) > 8 for site in self.reserved_locations)]
        if not candidates and not self.strict:
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
            veh.set_autopilot(True, self.tm_port)
            if self.night_lights:
                veh.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
            self.spawn_manifest["vehicles"].append({"id": veh.id, "blueprint": bp.id,
                "x": sp.location.x, "y": sp.location.y, "z": sp.location.z, "yaw": sp.rotation.yaw})
            try:
                self.tm.vehicle_percentage_speed_difference(veh, self.rng.uniform(-10, 20))
            except Exception:
                pass
            count += 1
        logger.info("生成交通车辆 %d/%d", count, num)
        return count

    def spawn_walkers(self, num: int, radius: float = 60.0) -> int:
        """场景初始化时生成 AI 行人；开始控制前只推进一次初始化帧。

        只能在闭环开始前调用。运行中的定点行人事件由 WalkerController 驱动，
        不在事件回调中偷偷推进仿真。出生点和目标都来自 CARLA 的人行导航网格。
        """
        if num <= 0:
            return 0
        bp_lib = self.world.get_blueprint_library()
        walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
        if self.strict:
            walker_bps.sort(key=lambda bp: bp.id)
        if not walker_bps:
            return 0
        try:
            controller_bp = bp_lib.find("controller.ai.walker")
        except Exception:
            logger.warning("无 walker 控制器蓝图，跳过行人生成")
            return 0

        ego_loc = self.ego.get_location()
        pending = []
        # 导航采样覆盖整张地图；有界拒绝采样，保留 ego 周围的点。
        for _ in range(max(100, num * 100)):
            if len(pending) >= num:
                break
            loc = self._navigation_location()
            if loc is None or loc.distance(ego_loc) > radius:
                continue
            if self.strict and any(loc.distance(site) <= 8 for site in self.reserved_locations):
                continue
            target = self._navigation_location(away_from=loc)
            if target is None:
                continue
            bp = self.rng.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = controller = None
            try:
                walker = self.world.spawn_actor(bp, carla.Transform(loc))
                self._walkers.append(walker)
                # Attached actors use a relative transform, not the world spawn location.
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                self._walker_controllers.append(controller)
                pending.append((walker, controller, target))
                self.spawn_manifest["walkers"].append({"id": walker.id, "blueprint": bp.id,
                    "start": [loc.x, loc.y, loc.z], "target": [target.x, target.y, target.z]})
            except Exception as e:
                logger.debug("行人生成失败: %s", e)
                self._discard_walker(walker, controller)

        if pending:
            # Official 0.9.15 generate_traffic.py waits for a snapshot before start():
            # without this, AddWalker reads an uninitialized transform (often 0,0,0).
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        count = 0
        for walker, controller, target in pending:
            try:
                controller.start()
                controller.go_to_location(target)
                controller.set_max_speed(self.rng.uniform(1.0, 1.5))
            except Exception as e:
                logger.warning("行人控制初始化失败: %s", e)
                self._discard_walker(walker, controller)
                continue
            count += 1
        logger.info("生成行人 %d/%d", count, num)
        if count < num:
            logger.warning("导航区域或出生空间不足，实际行人数 %d，小于请求数 %d", count, num)
        return count

    def _navigation_location(self, away_from=None):
        """只接受官方人行导航采样；不把道路 waypoint 当作导航网格。"""
        for _ in range(5):
            try:
                loc = self.world.get_random_location_from_navigation()
                if (loc is not None and all(math.isfinite(v) for v in (loc.x, loc.y, loc.z))
                        and (away_from is None or loc.distance(away_from) > 1.0)):
                    return loc
            except Exception as exc:
                logger.warning("行人导航点采样失败: %s", exc)
                return None
        return None

    def _discard_walker(self, walker, controller) -> None:
        """清理半初始化的行人；销毁失败时保留登记供最终清理重试。"""
        if controller is not None:
            try:
                if controller.is_alive:
                    controller.stop()
            except Exception:
                pass
        for actor, registry in ((controller, self._walker_controllers), (walker, self._walkers)):
            if actor is None:
                continue
            try:
                if actor.is_alive and actor.destroy() is False:
                    continue
                registry.remove(actor)
            except Exception as exc:
                logger.warning("行人初始化回滚失败: %s", exc)

    def destroy(self) -> bool:
        """尽力释放本轮资源；失败时返回 False 并保留引用供重试。"""
        # Unregister while actors are alive and TM is still synchronous.
        # Otherwise its background loop can read destroyed vehicles on resume.
        unregister_failed = []
        for vehicle in self._vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.set_autopilot(False, self.tm_port)
            except Exception as exc:
                logger.warning("交通车辆自动驾驶注销失败: %s", exc)
                unregister_failed.append(vehicle)
        for c in self._walker_controllers:
            try:
                if c.is_alive:
                    c.stop()
            except Exception:
                pass
        for actors in (self._walker_controllers, self._walkers, self._vehicles):
            remaining = []
            for a in actors:
                if any(a is failed for failed in unregister_failed):
                    # Preserve the live actor until it can be unregistered. Destroying
                    # it now would leave TM holding the same stale handle as the crash.
                    remaining.append(a)
                    continue
                try:
                    if a.is_alive and a.destroy() is False:
                        remaining.append(a)
                except Exception as exc:
                    logger.warning("交通 actor 清理失败: %s", exc)
                    remaining.append(a)
            actors[:] = remaining

        actors_remain = bool(self._vehicles or self._walkers or self._walker_controllers)
        if actors_remain:
            logger.warning("仍有交通 actor 未完成清理，保留 TM 同步状态以便重试")
        elif self._tm_synchronous:
            try:
                self.tm.set_synchronous_mode(False)
                self._tm_synchronous = False
            except Exception as exc:
                logger.warning("Traffic Manager 同步模式恢复失败: %s", exc)
        return not actors_remain and not self._tm_synchronous
