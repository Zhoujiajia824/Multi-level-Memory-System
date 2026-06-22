"""动力学适配器（预留）
====================
将轨迹转为方向盘转角、油门、刹车控制量。
第一版不实现完整动力学模型，仅提供 stub 接口。
后续可替换为 bicycle model、MPC、PID 或学习型控制器。
"""
from __future__ import annotations

from typing import Dict, Any, List

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("dynamics_adapter")


class DynamicsAdapter:
    """动力学适配器骨架。

    第一版仅提供 stub 接口，不做真实的轨迹到控制量转换。
    后续版本可替换为：
    - Bicycle model（自行车动力学模型）
    - MPC（模型预测控制）
    - PID 控制器
    - 学习型控制器（神经网络）

    Args:
        wheel_base: 轴距（米），默认 2.8 米。
        max_steer: 最大方向盘转角（弧度），默认 0.6。
    """

    def __init__(self, wheel_base: float = 2.8, max_steer: float = 0.6):
        self.wheel_base = wheel_base
        self.max_steer = max_steer

    def trajectory_to_control(
        self,
        trajectory: List[Dict],
        current_speed: float,
    ) -> Dict[str, Any]:
        """将轨迹转换为控制指令（完整实现，后续版本）。

        第一版不实现。请在后续版本中替换为 bicycle model / MPC / PID。

        Args:
            trajectory: ego-centric 轨迹点列表。
            current_speed: 当前速度（m/s）。

        Raises:
            NotImplementedError: 第一版不实现。
        """
        raise NotImplementedError(
            "动力学适配器尚未实现，将在后续版本中支持。"
            "第一版是离线研究 demo，不是实车控制系统。"
        )

    def trajectory_to_control_stub(
        self,
        trajectory: List[Dict],
        current_speed: float,
    ) -> Dict[str, Any]:
        """将轨迹转换为占位控制指令（stub）。

        返回全零的控制量占位字段，仅用于输出格式展示。
        不做任何真实的动力学计算。

        Args:
            trajectory: ego-centric 轨迹点列表。
            current_speed: 当前速度（m/s）。

        Returns:
            占位控制指令字典，包含：
            - steering: 方向盘转角（弧度），始终为 0.0
            - throttle: 油门开度（0-1），始终为 0.0
            - brake: 刹车力度（0-1），始终为 0.0
            - speed_command: 速度指令（m/s），取目标速度
            - is_stub: True，标记为 stub 输出
        """
        # 尝试从轨迹中获取目标速度
        target_speed = current_speed
        if trajectory:
            last_wp = trajectory[-1]
            if isinstance(last_wp, dict) and "optional_v" in last_wp:
                target_speed = last_wp["optional_v"]

        logger.info(
            f"生成 stub 控制指令: "
            f"steering=0.0, throttle=0.0, brake=0.0, "
            f"speed_command={target_speed:.2f}"
        )

        return {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "speed_command": target_speed,
            "is_stub": True,
            "note": "占位控制指令，第一版不做真实动力学计算。"
                    "后续可替换为 bicycle model / MPC / PID / 学习型控制器。",
        }
