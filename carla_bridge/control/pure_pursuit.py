"""Pure Pursuit 横向跟踪
=======================
经典 Pure Pursuit：在全局轨迹上选前瞻点，由自车（近似后轴）到前瞻点的几何关系
算方向盘角（弧度，**左正**，项目约定）。

    steer_rad = atan2(2 * L * sin(alpha), ld)

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
        self.reset()

    def reset(self):
        self._progress = 0.0
        self.target_valid = False
        self._path = None

    def compute_steer(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw_rad: float,
        speed: float,
        global_waypoints: List[dict],
    ) -> float:
        """返回方向盘角（弧度，左正）。无轨迹返回 0。"""
        self.target_valid = False
        if not global_waypoints:
            return 0.0
        if self._path is not global_waypoints:
            self.reset()
            self._path = global_waypoints
        ld = max(self.ld_min, min(self.ld_max, self.ld_min + self.ld_k * max(0.0, speed)))
        # Project onto the remaining polyline; progress never moves backwards.
        best = None
        for i in range(int(self._progress), len(global_waypoints) - 1):
            a, b = global_waypoints[i:i + 2]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            length2 = dx * dx + dy * dy
            if length2 <= 1e-12:
                continue
            t = ((ego_x - a["x"]) * dx + (ego_y - a["y"]) * dy) / length2
            t = max(max(0.0, self._progress - i), min(1.0, t))
            x, y = a["x"] + t * dx, a["y"] + t * dy
            distance2 = (x - ego_x) ** 2 + (y - ego_y) ** 2
            if best is None or distance2 < best[0]:
                best = (distance2, i + t, x, y)
        if best is None:
            x, y = global_waypoints[-1]["x"], global_waypoints[-1]["y"]
        else:
            _, self._progress, x, y = best
            remaining = ld
            for i in range(int(self._progress) + 1, len(global_waypoints)):
                wp = global_waypoints[i]
                dx, dy = wp["x"] - x, wp["y"] - y
                length = math.hypot(dx, dy)
                if length >= remaining and length > 1e-9:
                    x, y = x + dx * remaining / length, y + dy * remaining / length
                    break
                remaining -= length
                x, y = wp["x"], wp["y"]
        fwd, left = coords.global_to_ego(x, y, ego_x, ego_y, ego_yaw_rad)
        if fwd <= 1e-3:
            return 0.0  # Tracker brakes when no forward target remains.
        self.target_valid = True
        return math.atan(2.0 * self.L * left / max(fwd * fwd + left * left, 1e-6))
