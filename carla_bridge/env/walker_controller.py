"""行人行为控制：过马路 / 路边行走
================================
在 ego 前方生成行人，控制其过马路（spawn 路边 -> 目标对侧）或沿路边走（spawn 路边 ->
目标前方路边）。定点事件用 CARLA ``WalkerControl`` 沿指定线段行走，到达后停止。
这些人工坐标不保证属于人行导航网格，不能交给 AI 寻路器；随机交通行人仍使用 AI。

坐标系：CARLA 全局系 x 前、y **右**（与项目 ego-centric y=左 相反）。side=right 对应
CARLA +y（右手边）。横向偏移必须沿自车**侧向单位向量**而非全局 y，否则 ego 朝向非
+x 时事件位置随朝向旋转到错误方位（见 :func:`lateral_offset`）。
"""
from __future__ import annotations

import math
import random
from typing import Optional

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("carla_walker")


def lateral_offset(ego_tf, side_sign: float, dist: float):
    """自车侧向偏移向量（CARLA 全局系）。

    Args:
        ego_tf: ego 的 carla.Transform（只用 forward 向量）。
        side_sign: +1=右侧（CARLA +y），-1=左侧。
        dist: 侧向偏移距离（米）。

    Returns:
        ``(dx, dy)``：加到某基准点上的全局偏移。右侧向 = ``(-fwd.y, fwd.x)``
        （fwd 逆时针旋 90° 的右手侧），可单测。
    """
    fwd = ego_tf.get_forward_vector()
    return -fwd.y * side_sign * dist, fwd.x * side_sign * dist


class WalkerController:
    """行人生成 + 行为控制。"""

    def __init__(self, world, seed: int = 42):
        self.world = world
        self.rng = random.Random(seed)
        self._walkers = []
        self._motions = []

    def spawn_crossing(
        self,
        ego_tf,
        ahead_m: float = 20.0,
        side: str = "right",
        cross_dist: float = 8.0,
    ) -> Optional[carla.Actor]:
        """ego 前方 ahead_m、side 侧路边生成行人，目标对侧（过马路）。"""
        fwd = ego_tf.get_forward_vector()
        side_sign = 1.0 if side == "right" else -1.0  # CARLA +y=右
        cx = ego_tf.location.x + fwd.x * ahead_m
        cy = ego_tf.location.y + fwd.y * ahead_m
        sdx, sdy = lateral_offset(ego_tf, side_sign, cross_dist)
        spawn_loc = carla.Location(x=cx + sdx, y=cy + sdy, z=ego_tf.location.z + 1.0)
        tdx, tdy = lateral_offset(ego_tf, -side_sign, cross_dist)
        target_loc = carla.Location(x=cx + tdx, y=cy + tdy, z=ego_tf.location.z + 1.0)
        return self._spawn_walker(spawn_loc, target_loc)

    def spawn_roadside(
        self,
        ego_tf,
        ahead_m: float = 15.0,
        side: str = "right",
        walk_ahead_m: float = 30.0,
    ) -> Optional[carla.Actor]:
        """路边生成行人，沿路向前走 walk_ahead_m。"""
        fwd = ego_tf.get_forward_vector()
        side_sign = 1.0 if side == "right" else -1.0  # CARLA +y=右
        sx = ego_tf.location.x + fwd.x * ahead_m
        sy = ego_tf.location.y + fwd.y * ahead_m
        sdx, sdy = lateral_offset(ego_tf, side_sign, 3.0)
        spawn_loc = carla.Location(x=sx + sdx, y=sy + sdy, z=ego_tf.location.z + 1.0)
        tx = ego_tf.location.x + fwd.x * (ahead_m + walk_ahead_m)
        ty = ego_tf.location.y + fwd.y * (ahead_m + walk_ahead_m)
        tdx, tdy = lateral_offset(ego_tf, side_sign, 3.0)
        target_loc = carla.Location(x=tx + tdx, y=ty + tdy, z=ego_tf.location.z + 1.0)
        return self._spawn_walker(spawn_loc, target_loc)

    def _spawn_walker(self, spawn_loc, target_loc) -> Optional[carla.Actor]:
        bp_lib = self.world.get_blueprint_library()
        walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
        if not walker_bps:
            return None
        bp = self.rng.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        try:
            walker = self.world.spawn_actor(bp, carla.Transform(spawn_loc))
            self._walkers.append(walker)
            # Do not read get_location() before the next normal simulation tick.
            # The scheduler calls tick() before creating this frame's new walkers.
            dx, dy = target_loc.x - spawn_loc.x, target_loc.y - spawn_loc.y
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                return walker
            self._motions.append((walker, target_loc, dx / length, dy / length,
                                  self.rng.uniform(1.0, 1.5)))
            return walker
        except Exception as e:
            logger.debug("行人生成失败: %s", e)
            return None

    def tick(self) -> None:
        """在正常 world.tick 之前控制已完成出生帧的事件行人；不推进世界。"""
        active = []
        for walker, target, ux, uy, speed in self._motions:
            try:
                if not walker.is_alive:
                    continue
                loc = walker.get_location()
                dx, dy = target.x - loc.x, target.y - loc.y
                # Crossing the target's perpendicular plane also counts as arrival,
                # avoiding oscillation if one simulation step passes the endpoint.
                arrived = math.hypot(dx, dy) <= 0.25 or dx * ux + dy * uy <= 0
                walker.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(x=ux, y=uy, z=0.0),
                    speed=0.0 if arrived else speed,
                ))
                if not arrived:
                    active.append((walker, target, ux, uy, speed))
            except Exception as exc:
                logger.warning("事件行人控制失败: %s", exc)
                # Stop a previously moving walker if an API read/control failed.
                try:
                    walker.apply_control(carla.WalkerControl(speed=0.0))
                except Exception:
                    pass
        self._motions = active

    def destroy(self) -> bool:
        self._motions.clear()
        remaining = []
        for w in self._walkers:
            try:
                if w.is_alive and w.destroy() is False:
                    remaining.append(w)
            except Exception as exc:
                logger.warning("事件行人清理失败: %s", exc)
                remaining.append(w)
        self._walkers = remaining
        return not remaining
