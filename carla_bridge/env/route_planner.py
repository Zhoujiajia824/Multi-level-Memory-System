"""全局路径规划（自实现 A*）+ 导航语义派生
==========================================
自实现 A* over CARLA waypoint 图（``waypoint.next()``），**不依赖 agents 模块**
（agents 的 GlobalRoutePlanner 在某些环境会导致 CARLA 崩溃，改为纯 map API 实现）。

从起点 waypoint 沿 ``next(step)`` 展开，A*（启发式=到终点距离）找最短路径；每个节点
的航向变化派生导航语义（straight/left_turn/right_turn/lane_follow）。``end`` 为空时
回退沿路前进。
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.logging_utils import get_logger
from carla_bridge.state import coords

logger = get_logger("carla_route")

NAV_STRAIGHT = "straight"
NAV_LEFT = "left_turn"
NAV_RIGHT = "right_turn"
NAV_LANE_FOLLOW = "lane_follow"

# A* 搜索上限（防爆栈/超时）
_MAX_EXPANSIONS = 8000
_GOAL_THRESH_FACTOR = 3.0  # 到终点 step×3 内即认为到达
_TURN_THRESH = 0.35  # 弧度，航向变化超过此值判转向


class RoutePlanner:
    """全局路由器：自实现 A* 起点->终点；无终点时回退沿路前进。"""

    def __init__(self, world, ego_vehicle, python_api_dir: Optional[str] = None):
        # python_api_dir 保留参数兼容，但自实现 A* 不再使用
        self.world = world
        self.ego = ego_vehicle
        self.map = world.get_map()
        self._route: List[carla.Location] = []
        self._options: List[str] = []  # 导航语义列表（A* 模式）
        # walk-forward 回退用
        self._turns: List[str] = []
        self._junction_indices: List[int] = []
        self._mode = "none"

    # ------------------------------------------------------------------
    # 规划
    # ------------------------------------------------------------------

    def plan(
        self,
        start_loc: Optional[carla.Location] = None,
        end_loc: Optional[carla.Location] = None,
        sampling_resolution: float = 2.0,
        length_m: float = 200.0,
        turns: Optional[List[str]] = None,
    ) -> List[carla.Location]:
        if end_loc is not None:
            return self._plan_astar(start_loc, end_loc, sampling_resolution)
        return self._plan_walk_forward(length_m, turns or [], sampling_resolution)

    def _plan_astar(self, start_loc, end_loc, step) -> List[carla.Location]:
        if start_loc is None:
            start_loc = self.ego.get_location()
        start_wp = self.map.get_waypoint(start_loc, project_to_road=True)
        end_wp = self.map.get_waypoint(end_loc, project_to_road=True)
        if start_wp is None or end_wp is None:
            logger.warning("A* 起终点投影到路面失败，路由为空")
            self._route, self._options = [], []
            return []
        wps, options = self._astar_search(start_wp, end_wp, step)
        self._route = [wp.transform.location for wp in wps]
        self._options = options
        self._mode = "astar"
        logger.info(
            "A* 全局路由(自实现): %d 个 waypoint, start=(%.1f,%.1f) end=(%.1f,%.1f)",
            len(self._route), start_loc.x, start_loc.y, end_loc.x, end_loc.y,
        )
        return self._route

    def _astar_search(self, start_wp, end_wp, step) -> Tuple[List, List[str]]:
        """A* over waypoint.next() 图。返回 (waypoint 列表, 导航语义列表)。"""
        end_loc = end_wp.transform.location
        goal_thresh = step * _GOAL_THRESH_FACTOR

        def wp_key(wp) -> tuple:
            return (wp.road_id, wp.lane_id, round(wp.s, 1))

        def h(wp) -> float:
            return wp.transform.location.distance(end_loc)

        start_key = wp_key(start_wp)
        # best: key -> (g, wp, parent_key, nav_option)
        best = {start_key: (0.0, start_wp, None, NAV_LANE_FOLLOW)}
        # pq: (f, g, key, wp)
        pq = [(h(start_wp), 0.0, start_key, start_wp)]
        expansions = 0
        found_key = None

        while pq and expansions < _MAX_EXPANSIONS:
            f, g, key, wp = heapq.heappop(pq)
            if g > best.get(key, (1e9,))[0]:
                continue
            expansions += 1
            if wp.transform.location.distance(end_loc) < goal_thresh:
                found_key = key
                break
            try:
                nxts = wp.next(step)
            except Exception:
                nxts = []
            if not nxts:
                continue
            cur_yaw = coords.carla_yaw_deg_to_rad(wp.transform.rotation.yaw)
            for nxt in nxts:
                nkey = wp_key(nxt)
                d = wp.transform.location.distance(nxt.transform.location)
                ng = g + d
                if nkey in best and best[nkey][0] <= ng:
                    continue
                nav = self._infer_nav(cur_yaw, nxt)
                best[nkey] = (ng, nxt, key, nav)
                heapq.heappush(pq, (ng + h(nxt), ng, nkey, nxt))

        if found_key is None:
            logger.warning("A* 未找到路径（ expansions=%d ），路由为空", expansions)
            return [], []

        # 回溯重建路径
        wps, options = [], []
        k = found_key
        while k is not None:
            g, wp, pk, nav = best[k]
            wps.append(wp)
            options.append(nav)
            k = pk
        wps.reverse()
        options.reverse()
        return wps, options

    @staticmethod
    def _infer_nav(cur_yaw_rad: float, nxt_wp) -> str:
        """next 相对 current 的航向变化 -> 导航语义。

        项目 yaw 与 CARLA 同约定（+yaw=右转，见 coords），故 diff>0 为右转。
        """
        n_yaw = coords.carla_yaw_deg_to_rad(nxt_wp.transform.rotation.yaw)
        diff = n_yaw - cur_yaw_rad
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        if diff > _TURN_THRESH:
            return NAV_RIGHT
        if diff < -_TURN_THRESH:
            return NAV_LEFT
        return NAV_LANE_FOLLOW

    # ------------------------------------------------------------------
    # walk-forward 回退
    # ------------------------------------------------------------------

    def _plan_walk_forward(self, length_m, turns, step_m) -> List[carla.Location]:
        self._turns = list(turns)
        start = self.map.get_waypoint(self.ego.get_location(), project_to_road=True)
        if start is None:
            return []
        route: List[carla.Location] = []
        junctions: List[int] = []
        wp = start
        traveled = 0.0
        turn_idx = 0
        while traveled < length_m:
            route.append(wp.transform.location)
            nxt = wp.next(step_m)
            if not nxt:
                break
            if wp.is_junction:
                junctions.append(len(route) - 1)
                choice = self._turns[turn_idx] if turn_idx < len(self._turns) else "straight"
                turn_idx += 1
                wp = self._pick_outlet(wp, nxt, choice)
            else:
                wp = nxt[0]
            traveled += step_m
        self._route = route
        self._junction_indices = junctions
        self._options = []
        self._mode = "walk"
        logger.info("沿路路由(回退): %d 个 waypoint, length=%.0fm", len(route), length_m)
        return route

    def _pick_outlet(self, wp, candidates, choice):
        ego_yaw = coords.carla_yaw_deg_to_rad(wp.transform.rotation.yaw)
        scored = []
        for c in candidates:
            _, left = coords.global_to_ego(
                c.transform.location.x, c.transform.location.y,
                wp.transform.location.x, wp.transform.location.y, ego_yaw,
            )
            scored.append((c, left))
        if not scored:
            return wp
        if choice == "left":
            return max(scored, key=lambda s: s[1])[0]
        if choice == "right":
            return min(scored, key=lambda s: s[1])[0]
        return min(scored, key=lambda s: abs(s[1]))[0]

    # ------------------------------------------------------------------
    # 导航 + 进度
    # ------------------------------------------------------------------

    def nav_instruction(self, lookahead_m: float = 30.0) -> str:
        if not self._route:
            return NAV_LANE_FOLLOW
        ego_loc = self.ego.get_location()
        idx = self._nearest_index(ego_loc)
        if self._mode == "astar" and self._options:
            dist = 0.0
            for j in range(idx, len(self._options)):
                if j > idx:
                    dist += self._route[j - 1].distance(self._route[j])
                if dist > lookahead_m:
                    break
                nav = self._options[j]
                if nav != NAV_LANE_FOLLOW:
                    return nav
            return NAV_LANE_FOLLOW
        for k, ji in enumerate(self._junction_indices):
            if ji <= idx:
                continue
            if self._path_distance(idx, ji) <= lookahead_m:
                turn = self._turns[k] if k < len(self._turns) else "straight"
                if turn == "left":
                    return NAV_LEFT
                if turn == "right":
                    return NAV_RIGHT
                return NAV_STRAIGHT
        return NAV_LANE_FOLLOW

    def center_points_ahead(
        self,
        ego_loc=None,
        n: int = 5,
        step_m: float = 8.0,
    ) -> List[carla.Location]:
        """沿路由取 ego 前方 n 个车道中心点（约每 step_m 一个）。

        CARLA waypoint 天然投影在车道中心线上，路由点即车道中心序列。供决策
        prompt 注入米制"回正锚点"（route_center 伪对象）使用——VLM 无法从
        480×270 图像定量判断横向偏移，但能直接抄写米制坐标。

        无路由或前方无点（接近终点）返回空列表，自然退化为不注入。
        """
        if not self._route:
            return []
        if ego_loc is None:
            ego_loc = self.ego.get_location()
        idx = self._nearest_index(ego_loc)
        # 从最近点向后扫描，跳过 ego 身后/过近的点，按累计弧长取点
        out: List[carla.Location] = []
        acc = 0.0
        next_pick = step_m
        for i in range(idx, len(self._route)):
            if i > idx:
                acc += self._route[i - 1].distance(self._route[i])
            if acc >= next_pick - 1e-6:
                out.append(self._route[i])
                next_pick += step_m
                if len(out) >= n:
                    break
        return out

    def progress_fraction(self) -> float:
        if not self._route:
            return 0.0
        idx = self._nearest_index(self.ego.get_location())
        return idx / max(1, len(self._route) - 1)

    def _nearest_index(self, loc) -> int:
        best, best_d = 0, float("inf")
        for i, p in enumerate(self._route):
            d = p.distance(loc)
            if d < best_d:
                best_d, best = d, i
        return best

    def _path_distance(self, i0: int, i1: int) -> float:
        d = 0.0
        for i in range(i0, i1):
            d += self._route[i].distance(self._route[i + 1])
        return d
