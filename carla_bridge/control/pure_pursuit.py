"""Pure Pursuit 横向跟踪
=======================
经典 Pure Pursuit：在全局轨迹上选前瞻点，由自车（近似后轴）到前瞻点的几何关系
算方向盘角（弧度，**左正**，项目约定）。

    steer_rad = atan2(2 * L * sin(alpha), 2 * ld)

* ``L`` = 轴距，``alpha`` = 前瞻点相对自车朝向的夹角，``ld`` = 前瞻距离。
* ``ld = clamp(min + k * speed, min, max)``。

纯数学，不依赖 carla，可在任意环境单测。
"""
from __future__ import annotations

import math
from typing import List

from carla_bridge.state import coords


class PurePursuit:
    """Pure Pursuit 横向控制器。"""

    def __init__(
        self,
        wheelbase_m: float = 2.8,
        lookahead_min: float = 3.0,
        lookahead_max: float = 12.0,
        lookahead_k: float = 0.5,
    ):
        self.L = wheelbase_m
        self.ld_min = lookahead_min
        self.ld_max = lookahead_max
        self.ld_k = lookahead_k

    def compute_steer(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw_rad: float,
        speed: float,
        global_waypoints: List[dict],
    ) -> float:
        """返回方向盘角（弧度，左正）。无轨迹返回 0。"""
        if not global_waypoints:
            return 0.0

        ld = self.ld_min + self.ld_k * max(0.0, speed)
        ld = max(self.ld_min, min(self.ld_max, ld))

        # 选前瞻点：第一个距自车 >= ld 的点
        target = None
        for wp in global_waypoints:
            d = math.hypot(wp["x"] - ego_x, wp["y"] - ego_y)
            if d >= ld:
                target = (wp["x"], wp["y"])
                break
        if target is None:
            target = (global_waypoints[-1]["x"], global_waypoints[-1]["y"])

        fwd, left = coords.global_to_ego(target[0], target[1], ego_x, ego_y, ego_yaw_rad)
        alpha = math.atan2(left, fwd)  # 左正
        ld_eff = max(math.hypot(fwd, left), 1e-3)
        return math.atan2(2.0 * self.L * math.sin(alpha), 2.0 * ld_eff)
