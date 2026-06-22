"""
轨迹构建器
===========
构建 ego-centric 坐标系下的历史轨迹和未来真值轨迹。
x 为前向，y 为左向，单位米。
如果历史不足 N 秒，使用实际可获得历史，并记录 history_seconds_actual。
"""
from __future__ import annotations

import math
from typing import List, Dict, Optional

from src.vla_memory.schemas.ego_state import EgoState
from src.vla_memory.schemas.trajectory import Trajectory, TrajectoryPoint
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("trajectory_builder")


class TrajectoryBuilder:
    """轨迹构建器。

    构建历史轨迹和未来真值轨迹，输出 ego-centric 坐标系。
    时间戳单位默认为微秒（nuScenes）。

    Args:
        timestamp_unit: 时间戳单位，'us' 微秒 / 's' 秒。
    """

    def __init__(self, timestamp_unit: str = "us"):
        self.timestamp_divisor = 1e6 if timestamp_unit == "us" else 1.0

    def build_history_trajectory(
        self,
        current_pose: Dict,
        past_poses: List[Dict],
        history_seconds: float = 5.0,
    ) -> Trajectory:
        """构建 ego-centric 坐标系下的最近 N 秒历史轨迹。

        Args:
            current_pose: 当前帧 ego_pose。
            past_poses: 过去的 ego_pose 列表（按时间升序，不含当前帧）。
            history_seconds: 期望的历史时间窗口（秒）。

        Returns:
            Trajectory 实例。history_seconds_actual 字段记录实际覆盖时长。
        """
        current_ts = current_pose["timestamp"]
        current_t = current_pose["translation"]
        current_yaw = EgoState.quat_to_yaw(current_pose.get("rotation", [1, 0, 0, 0]))

        cos_yaw = math.cos(-current_yaw)
        sin_yaw = math.sin(-current_yaw)

        points = []
        actual_max_dt = 0.0

        for pose in past_poses:
            ts = pose.get("timestamp", 0)
            dt = (current_ts - ts) / self.timestamp_divisor

            if dt < 0 or dt > history_seconds:
                continue

            t = pose["translation"]
            dx = t[0] - current_t[0]
            dy = t[1] - current_t[1]
            ego_x = dx * cos_yaw - dy * sin_yaw
            ego_y = dx * sin_yaw + dy * cos_yaw

            points.append(TrajectoryPoint(t=round(-dt, 4), x=round(ego_x, 4), y=round(ego_y, 4)))
            actual_max_dt = max(actual_max_dt, dt)

        # 按时间排序（从远到近）
        points.sort(key=lambda p: p.t)

        if len(points) < len(past_poses) and past_poses:
            logger.debug(
                f"历史轨迹: 请求 {history_seconds}s, "
                f"实际 {actual_max_dt:.2f}s, "
                f"{len(points)} 个点"
            )

        return Trajectory(
            coordinate_system="ego_centric",
            points=points,
            history_seconds_actual=round(actual_max_dt, 4) if points else 0.0,
        )

    def build_future_trajectory(
        self,
        current_pose: Dict,
        future_poses: List[Dict],
        future_seconds: float = 3.0,
    ) -> Trajectory:
        """构建 ego-centric 坐标系下的未来真值轨迹。

        用于评测模块计算 ADE / FDE。

        Args:
            current_pose: 当前帧 ego_pose。
            future_poses: 未来 ego_pose 列表（按时间升序）。
            future_seconds: 未来时间窗口（秒）。

        Returns:
            Trajectory 实例。
        """
        current_ts = current_pose["timestamp"]
        current_t = current_pose["translation"]
        current_yaw = EgoState.quat_to_yaw(current_pose.get("rotation", [1, 0, 0, 0]))

        cos_yaw = math.cos(-current_yaw)
        sin_yaw = math.sin(-current_yaw)

        points = []
        for pose in future_poses:
            ts = pose.get("timestamp", 0)
            dt = (ts - current_ts) / self.timestamp_divisor

            if dt < 0 or dt > future_seconds:
                continue

            t = pose["translation"]
            dx = t[0] - current_t[0]
            dy = t[1] - current_t[1]
            ego_x = dx * cos_yaw - dy * sin_yaw
            ego_y = dx * sin_yaw + dy * cos_yaw

            points.append(TrajectoryPoint(t=round(dt, 4), x=round(ego_x, 4), y=round(ego_y, 4)))

        return Trajectory(
            coordinate_system="ego_centric",
            points=points,
        )
