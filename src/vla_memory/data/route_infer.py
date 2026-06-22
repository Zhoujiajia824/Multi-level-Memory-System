"""
导航语义推断器
==============
根据未来 ego trajectory 推断伪导航语义。
使用 yaw 变化、横向位移、未来速度变化、轨迹曲率等规则。

这是 demo 级伪导航语义，不是 nuScenes 官方导航标签。
后续可替换为真实地图导航 / route planner。

判断类别：straight, left_turn, right_turn, lane_follow,
lane_change_left, lane_change_right, slow_or_stop, unknown
"""
from __future__ import annotations

import math
from typing import List, Dict, Optional

from src.vla_memory.schemas.ego_state import EgoState
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("route_infer")

# ===================== 导航语义枚举常量 =====================

NAV_STRAIGHT = "straight"
NAV_LEFT_TURN = "left_turn"
NAV_RIGHT_TURN = "right_turn"
NAV_LANE_FOLLOW = "lane_follow"
NAV_LANE_CHANGE_LEFT = "lane_change_left"
NAV_LANE_CHANGE_RIGHT = "lane_change_right"
NAV_SLOW_OR_STOP = "slow_or_stop"
NAV_UNKNOWN = "unknown"

ALL_NAV_CATEGORIES = [
    NAV_STRAIGHT, NAV_LEFT_TURN, NAV_RIGHT_TURN, NAV_LANE_FOLLOW,
    NAV_LANE_CHANGE_LEFT, NAV_LANE_CHANGE_RIGHT, NAV_SLOW_OR_STOP, NAV_UNKNOWN,
]
"""所有有效的导航语义类别。"""


class RouteInfer:
    """导航语义推断器。

    根据未来 ego trajectory 推断伪导航语义。
    使用 yaw 变化、横向位移、速度变化、轨迹曲率等启发式规则。

    README 必须说明这是 demo 级伪导航语义，不是 nuScenes 官方导航标签。

    Args:
        yaw_threshold: 航向角变化阈值（弧度），超过判定为转弯。默认 0.15。
        lateral_threshold: 横向位移阈值（米），超过判定为变道。默认 2.0。
        speed_stop_threshold: 速度低于此值判定为停车（m/s）。默认 1.0。
        straight_yaw_max: 航向角变化小于此值且横向位移小时判定为直行。默认 0.05。
        straight_lateral_max: 横向位移小于此值时判定为直行（米）。默认 1.0。
    """

    def __init__(
        self,
        yaw_threshold: float = 0.15,
        lateral_threshold: float = 2.0,
        speed_stop_threshold: float = 1.0,
        straight_yaw_max: float = 0.05,
        straight_lateral_max: float = 1.0,
    ):
        self.yaw_threshold = yaw_threshold
        self.lateral_threshold = lateral_threshold
        self.speed_stop_threshold = speed_stop_threshold
        self.straight_yaw_max = straight_yaw_max
        self.straight_lateral_max = straight_lateral_max

    def infer(
        self,
        future_poses: List[Dict],
        current_speed: float = 0.0,
    ) -> str:
        """推断伪导航语义。

        Args:
            future_poses: 未来 ego_pose 列表（每个包含 translation, rotation）。
            current_speed: 当前速度（m/s）。

        Returns:
            导航语义字符串（ALL_NAV_CATEGORIES 之一）。
        """
        if not future_poses:
            return NAV_UNKNOWN

        # 判断减速或停车
        if current_speed < self.speed_stop_threshold:
            return NAV_SLOW_OR_STOP

        # 计算航向角变化
        yaw_change = self._compute_yaw_change(future_poses)

        # 计算横向位移（基于起点方向坐标系）
        lateral = self._compute_lateral_displacement(future_poses)

        # 计算终点速度变化（用于辅助判断）
        future_speed_change = self._compute_speed_change(future_poses)

        # 推断逻辑（优先级从高到低）
        if abs(yaw_change) > self.yaw_threshold:
            return NAV_LEFT_TURN if yaw_change > 0 else NAV_RIGHT_TURN

        if abs(lateral) > self.lateral_threshold:
            return NAV_LANE_CHANGE_LEFT if lateral > 0 else NAV_LANE_CHANGE_RIGHT

        if abs(yaw_change) < self.straight_yaw_max and abs(lateral) < self.straight_lateral_max:
            return NAV_STRAIGHT

        # 有一定横向位移或航向角变化，但不够判定为转弯/变道
        return NAV_LANE_FOLLOW

    @staticmethod
    def _compute_yaw_change(poses: List[Dict]) -> float:
        """计算轨迹总航向角变化量。

        从第一个 pose 到最后一个 pose 的 yaw 差值，归一化到 [-pi, pi]。

        Args:
            poses: ego_pose 列表，每个包含 rotation [w,x,y,z]。

        Returns:
            航向角变化量（弧度）。
        """
        if len(poses) < 2:
            return 0.0
        first_yaw = EgoState.quat_to_yaw(poses[0].get("rotation", [1, 0, 0, 0]))
        last_yaw = EgoState.quat_to_yaw(poses[-1].get("rotation", [1, 0, 0, 0]))
        diff = last_yaw - first_yaw
        # 归一化到 [-pi, pi]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    @staticmethod
    def _compute_lateral_displacement(poses: List[Dict]) -> float:
        """计算轨迹的横向位移。

        基于起点到终点的连线方向，计算垂直于该方向的位移分量。
        正值表示向左偏移。

        Args:
            poses: ego_pose 列表，每个包含 translation [x,y,z]。

        Returns:
            横向位移量（米）。
        """
        if len(poses) < 2:
            return 0.0

        start = poses[0].get("translation", [0, 0, 0])
        end = poses[-1].get("translation", [0, 0, 0])

        # 起点到终点的方向向量
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            return 0.0

        # 前进方向单位向量
        fwd_x = dx / dist
        fwd_y = dy / dist

        # 计算每个中间点相对于前进方向的横向偏移
        max_lateral = 0.0
        for pose in poses:
            t = pose.get("translation", [0, 0, 0])
            rel_x = t[0] - start[0]
            rel_y = t[1] - start[1]
            # 横向分量 = rel_x * (-fwd_y) + rel_y * fwd_x
            lateral = -rel_x * fwd_y + rel_y * fwd_x
            max_lateral = max(max_lateral, abs(lateral))

        # 用最大横向偏移的符号（基于最终位置）
        final_rel_x = end[0] - start[0]
        final_rel_y = end[1] - start[1]
        signed_lateral = -final_rel_x * fwd_y + final_rel_y * fwd_x

        # 返回最大横向偏移，带符号
        return math.copysign(max_lateral, signed_lateral)

    @staticmethod
    def _compute_speed_change(poses: List[Dict]) -> float:
        """计算未来轨迹的速度变化趋势。

        用于辅助判断减速行为。

        Args:
            poses: ego_pose 列表。

        Returns:
            速度变化量（正为加速，负为减速），近似值。
        """
        if len(poses) < 3:
            return 0.0

        # 用位移估计速度
        t0 = poses[0].get("translation", [0, 0, 0])
        t1 = poses[len(poses) // 2].get("translation", [0, 0, 0])
        t2 = poses[-1].get("translation", [0, 0, 0])

        d1 = math.sqrt((t1[0] - t0[0]) ** 2 + (t1[1] - t0[1]) ** 2)
        d2 = math.sqrt((t2[0] - t1[0]) ** 2 + (t2[1] - t1[1]) ** 2)

        return d2 - d1  # 正值加速，负值减速
