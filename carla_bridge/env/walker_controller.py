"""行人行为控制：过马路 / 路边行走
================================
在 ego 前方生成行人，控制其过马路（spawn 路边 -> 目标对侧）或沿路边走（spawn 路边 ->
目标前方路边）。用 CARLA ``controller.ai.walker``（AI 控制器自动避障走动）。

坐标系：CARLA 全局系 x 前、y **右**（与项目 ego-centric y=左 相反）。side=right 对应
CARLA +y（右手边）。横向偏移必须沿自车**侧向单位向量**而非全局 y，否则 ego 朝向非
+x 时事件位置随朝向旋转到错误方位（见 :func:`lateral_offset`）。
"""
from __future__ import annotations

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
        self._controllers = []

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
        try:
            walker = self.world.spawn_actor(bp, carla.Transform(spawn_loc))
            self._walkers.append(walker)
            ctrl_bp = bp_lib.find("controller.ai.walker")
            controller = self.world.spawn_actor(ctrl_bp, carla.Transform(spawn_loc), attach_to=walker)
            self._controllers.append(controller)
            controller.start()
            controller.go_to_location(target_loc)
            controller.set_max_speed(self.rng.uniform(1.0, 1.5))
            return walker
        except Exception as e:
            logger.debug("行人生成失败: %s", e)
            return None

    def destroy(self) -> None:
        for c in self._controllers:
            try:
                c.stop()
            except Exception:
                pass
        for w in self._walkers:
            try:
                w.destroy()
            except Exception:
                pass
        for c in self._controllers:
            try:
                c.destroy()
            except Exception:
                pass
        self._walkers.clear()
        self._controllers.clear()
