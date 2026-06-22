"""
自车状态构建器
==============
P3 前：根据 ego_pose 时间序列差分估计速度、加速度。
P3 起：优先从 nuScenes CAN bus 真值（pose.json + vehicle_monitor.json）构建，
        失败时回退到差分。

yaw 从 nuScenes 四元数 [w,x,y,z] 转换；
CAN bus 提供 yaw_rate / steering / throttle / brake 等差分无法得到的字段。
不允许伪造车辆状态。缺失必要 ego_pose 时记录错误。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from src.vla_memory.schemas.ego_state import EgoState
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("ego_state_builder")


class EgoStateBuilder:
    """自车状态构建器。

    支持两种来源：
    1. ``build_from_poses`` —— 从 nuScenes ego_pose 时间序列差分（旧路径）。
    2. ``build_from_can_bus`` —— 从 CAN bus 真值直接读取（P3 新路径）。

    统一入口 ``build`` 优先尝试 CAN bus，失败时回退到差分。

    Args:
        timestamp_unit: 时间戳单位，'us' 表示微秒（nuScenes 默认）。
    """

    def __init__(self, timestamp_unit: str = "us"):
        self.timestamp_divisor = 1e6 if timestamp_unit == "us" else 1.0

    # ------------------------------------------------------------------
    # P3 新增：CAN bus 真值
    # ------------------------------------------------------------------

    def build_from_can_bus(
        self,
        scene_name: str,
        sample_timestamp_us: int,
        can_bus_loader,
    ) -> EgoState:
        """从 CAN bus 真值构建 EgoState。

        Args:
            scene_name: nuScenes scene 名称（如 ``"scene-0061"``）。
            sample_timestamp_us: 目标时间戳（微秒，与 sample.timestamp 同源）。
            can_bus_loader: CanBusLoader 实例。

        Returns:
            EgoState，``source="can_bus"``（或 ``"can_bus_pose_only"`` 当
            vehicle_monitor 缺失时）。

        Raises:
            KeyError: CAN bus 在该时间戳附近没有可用 pose（由 loader 抛出）。
            FileNotFoundError: 该 scene 的 pose.json 不存在。
        """
        sample = can_bus_loader.query_at(scene_name, sample_timestamp_us)

        # 全局位置
        x, y, z = sample.pos

        # yaw 直接用 quaternion 转换（与差分路径同一公式，保持一致性）
        yaw = EgoState.quat_to_yaw(list(sample.orientation_quat))

        # 体坐标系速度 -> 全局坐标系 (绕 z 轴旋转 yaw)
        vbx, vby, _ = sample.vel
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        vx = cos_y * vbx - sin_y * vby
        vy = sin_y * vbx + cos_y * vby
        # 优先用 vehicle_monitor 标定速度；缺失时从 pose.vel 算
        speed = sample.speed_mps if sample.speed_mps is not None else math.sqrt(vx * vx + vy * vy)

        # 加速度同理转到全局系（accel[2] 通常含重力分量，丢弃）
        abx, aby, _ = sample.accel
        ax = cos_y * abx - sin_y * aby
        ay = sin_y * abx + cos_y * aby
        acceleration = math.sqrt(ax * ax + ay * ay)

        # yaw_rate：优先 vehicle_monitor，否则用 pose.rotation_rate[2]
        if sample.yaw_rate is not None:
            yaw_rate = sample.yaw_rate
        else:
            yaw_rate = float(sample.rotation_rate[2])

        source = "can_bus" if sample.speed_mps is not None else "can_bus_pose_only"

        return EgoState(
            timestamp=int(sample.utime_us),
            x=x, y=y, z=z, yaw=yaw,
            vx=vx, vy=vy, speed=max(0.0, speed),
            ax=ax, ay=ay, acceleration=acceleration,
            yaw_rate=yaw_rate,
            steering_angle=sample.steering,
            throttle=sample.throttle,
            brake=sample.brake,
            gear=sample.gear,
            source=source,
        )

    def build(
        self,
        current_pose: Dict,
        *,
        scene_name: Optional[str] = None,
        can_bus_loader=None,
        prev_pose: Optional[Dict] = None,
        prev_prev_pose: Optional[Dict] = None,
    ) -> EgoState:
        """统一入口：CAN bus 优先，差分回退。

        Args:
            current_pose: 当前帧 ego_pose（差分回退所需）。
            scene_name: scene 名称（CAN bus 查询所需）。
            can_bus_loader: CanBusLoader 实例；None 时直接走差分。
            prev_pose / prev_prev_pose: 差分回退所需的相邻帧。

        Returns:
            EgoState。``source`` 字段标识真实来源。
        """
        if can_bus_loader is not None and scene_name:
            ts = current_pose.get("timestamp")
            if ts is not None:
                try:
                    return self.build_from_can_bus(scene_name, int(ts), can_bus_loader)
                except (KeyError, FileNotFoundError) as e:
                    logger.warning(
                        "CAN bus 查询失败 (scene=%s, ts=%s)，回退到 ego_pose 差分: %s",
                        scene_name, ts, e,
                    )
        return self.build_from_poses(current_pose, prev_pose, prev_prev_pose)

    # ------------------------------------------------------------------
    # 旧路径：ego_pose 差分（保留）
    # ------------------------------------------------------------------

    def build_from_poses(
        self,
        current_pose: Dict,
        prev_pose: Optional[Dict] = None,
        prev_prev_pose: Optional[Dict] = None,
    ) -> EgoState:
        """从 1-3 个 ego_pose 构建 EgoState。

        Args:
            current_pose: 当前帧 ego_pose，必须包含 translation, rotation, timestamp。
            prev_pose: 上一帧 ego_pose（可选）。缺失时速度、加速度置零。
            prev_prev_pose: 上上一帧 ego_pose（可选）。缺失时加速度置零。

        Returns:
            EgoState 实例。

        Raises:
            ValueError: current_pose 缺少必需字段。
        """
        # 校验必需字段
        if "translation" not in current_pose:
            raise ValueError("current_pose 缺少 'translation' 字段，无法构建 EgoState")
        if "timestamp" not in current_pose:
            raise ValueError("current_pose 缺少 'timestamp' 字段，无法构建 EgoState")

        translation = current_pose["translation"]
        rotation = current_pose.get("rotation", [1, 0, 0, 0])
        timestamp = current_pose["timestamp"]

        x, y, z = translation[0], translation[1], translation[2]
        yaw = EgoState.quat_to_yaw(rotation)

        vx, vy, speed = 0.0, 0.0, 0.0
        ax, ay, acceleration = 0.0, 0.0, 0.0

        # 速度：相邻两帧差分
        if prev_pose is not None:
            prev_t = prev_pose.get("translation", [0, 0, 0])
            prev_ts = prev_pose.get("timestamp", 0)
            dt = (timestamp - prev_ts) / self.timestamp_divisor
            if dt > 0:
                vx = (x - prev_t[0]) / dt
                vy = (y - prev_t[1]) / dt
                speed = math.sqrt(vx * vx + vy * vy)
            else:
                logger.warning("相邻帧时间差 ≤ 0，速度置零")

        # 加速度：三帧差分
        if prev_prev_pose is not None and prev_pose is not None:
            prev_t = prev_pose.get("translation", [0, 0, 0])
            prev_ts = prev_pose.get("timestamp", 0)
            pp_t = prev_prev_pose.get("translation", [0, 0, 0])
            pp_ts = prev_prev_pose.get("timestamp", 0)
            dt1 = (prev_ts - pp_ts) / self.timestamp_divisor
            dt2 = (timestamp - prev_ts) / self.timestamp_divisor
            if dt1 > 0 and dt2 > 0:
                prev_vx = (prev_t[0] - pp_t[0]) / dt1
                prev_vy = (prev_t[1] - pp_t[1]) / dt1
                ax = (vx - prev_vx) / dt2
                ay = (vy - prev_vy) / dt2
                acceleration = math.sqrt(ax * ax + ay * ay)

        return EgoState(
            timestamp=timestamp,
            x=x, y=y, z=z, yaw=yaw,
            vx=vx, vy=vy, speed=speed,
            ax=ax, ay=ay, acceleration=acceleration,
        )

    def build_from_sample_list(
        self,
        sample_tokens: List[str],
        adapter,  # NuScenesAdapter
        *,
        can_bus_loader=None,
    ) -> List[EgoState]:
        """从样本 token 列表批量构建 EgoState。

        Args:
            sample_tokens: 按时间排序的 sample token 列表。
            adapter: 已加载的 NuScenesAdapter 实例。
            can_bus_loader: 可选的 CanBusLoader；非 None 时优先用 CAN bus 真值。

        Returns:
            EgoState 列表，与输入顺序一一对应。
        """
        results = []
        for i, token in enumerate(sample_tokens):
            current_pose = adapter._get_sample_ego_pose(token)
            prev_pose = adapter._get_sample_ego_pose(sample_tokens[i - 1]) if i >= 1 else None
            prev_prev_pose = adapter._get_sample_ego_pose(sample_tokens[i - 2]) if i >= 2 else None

            scene_name = None
            if can_bus_loader is not None:
                # 通过 adapter 反查 scene_name（"scene-XXXX"）
                try:
                    sample = adapter.nusc.get("sample", token)
                    scene = adapter.nusc.get("scene", sample["scene_token"])
                    scene_name = scene["name"]
                except Exception as e:
                    logger.warning("反查 scene_name 失败 (token=%s): %s", token, e)

            state = self.build(
                current_pose,
                scene_name=scene_name,
                can_bus_loader=can_bus_loader,
                prev_pose=prev_pose,
                prev_prev_pose=prev_prev_pose,
            )
            results.append(state)

        return results
