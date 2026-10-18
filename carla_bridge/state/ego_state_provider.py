"""CARLA 自车 -> EgoState
========================
从 ego vehicle 直读 transform / velocity / acceleration / control，构造项目
``EgoState``。CARLA 提供当前真值（速度/加速度直读，优于 nuScenes 差分），
``source="carla"``。速度/加速度向量经 ``coords`` 旋转到 ego-centric（前/左）。
"""
from __future__ import annotations

import math
from typing import Optional

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.schemas.ego_state import EgoState
from carla_bridge.control.steering_model import SteeringModel
from carla_bridge.state import coords


class EgoStateProvider:
    """CARLA vehicle -> EgoState。

    Args:
        max_steer_rad: 未提供 steering_model 时，用于换算轮角的静态最大角（弧度）。
        steering_model: 可选的共享转向模型，按当前速度换算归一化 steer。
            ``steering_angle`` 是左正转角估计；绑定 PDM 时为等效自行车转角。
            它不是实测前轮角或方向盘角，也不包含执行器输入迟滞。
    """

    def __init__(self, max_steer_rad: float = 0.6, *,
                 steering_model: Optional[SteeringModel] = None):
        self.max_steer_rad = max_steer_rad
        self.steering_model = steering_model

    def build(
        self,
        ego_vehicle,
        elapsed_sim_s: float,
        yaw_rate: Optional[float] = None,
    ) -> EgoState:
        """构造 EgoState。

        Args:
            ego_vehicle: carla.Vehicle。
            elapsed_sim_s: 仿真累计时间（秒），转微秒作 timestamp。
            yaw_rate: 偏航角速率（rad/s），由上层从历史位姿差分得到；None 则留空。
        """
        tf = ego_vehicle.get_transform()
        loc = tf.location
        yaw_rad = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)

        v = ego_vehicle.get_velocity()
        a = ego_vehicle.get_acceleration()
        ctrl = ego_vehicle.get_control()

        evx, evy = coords.rotate_vector_to_ego(v.x, v.y, yaw_rad)
        speed = math.hypot(evx, evy)
        eax, eay = coords.rotate_vector_to_ego(a.x, a.y, yaw_rad)
        accel = math.hypot(eax, eay)

        # Convert the command using the same speed-dependent approximation as
        # execution. PDM separately reads actual front-wheel feedback; the
        # semantic state remains a clearly documented command estimate.
        steer_rad = None
        if ctrl:
            steer_rad = (-self.steering_model.normalized_to_bicycle(ctrl.steer, speed)
                         if self.steering_model is not None else -float(ctrl.steer) * self.max_steer_rad)

        ts_us = int(elapsed_sim_s * 1e6)
        return EgoState(
            timestamp=ts_us,
            x=loc.x, y=loc.y, z=loc.z, yaw=yaw_rad,
            vx=evx, vy=evy, speed=max(0.0, speed),
            ax=eax, ay=eay, acceleration=accel,
            yaw_rate=yaw_rate,
            steering_angle=steer_rad,
            throttle=float(ctrl.throttle) if ctrl else None,
            brake=float(ctrl.brake) if ctrl else None,
            gear="D",
            source="carla",
        )
