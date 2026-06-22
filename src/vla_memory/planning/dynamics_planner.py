"""动力学规划接口（CARLA 闭环预留壳）
=====================================
P7 预留：不实现具体算法，仅冻结调用接口和数据约定，方便后续接入
CARLA / 自定义仿真器时直接替换实现。

设计目标
--------
把决策模型输出的 ego-centric 轨迹（x 前向, y 左向, 单位米）转换为
一串低层控制指令（steering / throttle / brake / gear），喂给 CARLA
的 ``vehicle.apply_control`` 或类似 API。

推荐内部实现（v0.2+）：
1. Bicycle model（kinematic 或 dynamic）+ wheelbase / max_steer 约束；
   wheelbase 已在 ``config/decision.yaml -> dynamics_adapter.wheel_base`` 配置。
2. 用 Pure Pursuit / Stanley / MPC 做横向跟随，PID 做纵向跟随。
3. 对每个 control step 输出一条 ``ControlCommand``。

字段约定见 ``ControlCommand``。所有量需符合 SI 单位：弧度 / 米/秒 /
0~1 归一化。

接入 CARLA 时的最小 shim 示例
----------------------------
::

    planner = DynamicsPlanner(wheelbase_m=2.8, max_steer_rad=0.6)
    commands = planner.plan(trajectory=decision.trajectory, ego_state=cur, dt=0.05)
    for cmd in commands:
        ctrl = carla.VehicleControl(
            throttle=cmd.throttle,
            brake=cmd.brake,
            steer=cmd.steer_normalized,
        )
        vehicle.apply_control(ctrl)
        world.tick()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ControlCommand:
    """单步低层控制指令（SI 单位）。

    Attributes:
        timestamp_s: 该指令对应的时间（秒），从 plan 调用时刻起算。
        steer_rad: 方向盘转角（弧度），左正右负。
        steer_normalized: 方向盘归一化值 [-1, 1]（CARLA 风格），
            等于 ``steer_rad / max_steer_rad``。
        throttle: 油门 [0, 1]。
        brake: 刹车 [0, 1]。
        target_speed_mps: 该步期望速度（米/秒），用于上层审计。
        gear: 档位（"D" / "R" / "N" / "P"）；CARLA 默认 "D"。
    """
    timestamp_s: float
    steer_rad: float
    steer_normalized: float
    throttle: float
    brake: float
    target_speed_mps: float
    gear: str = "D"


class DynamicsPlanner:
    """轨迹 -> 低层控制量。**v0.1 仅为接口壳，不做实质规划。**

    构造参数预留了 bicycle 模型需要的关键车辆参数。后续实现请确保：

    * 输出长度 = ``ceil(trajectory.horizon_seconds / dt)``。
    * 单步 ``throttle * brake == 0``（不允许同时踩油门和刹车）。
    * ``abs(steer_rad) <= max_steer_rad``，溢出做饱和裁剪并 warning。

    Args:
        wheelbase_m: 前后轴距（米），默认 2.8（小型轿车）。
            生产环境应从 ``config/decision.yaml -> dynamics_adapter.wheel_base`` 取。
        max_steer_rad: 方向盘最大转角（弧度），默认 0.6（约 34°）。
        max_acceleration_mps2: 最大加速度（m/s²），默认 3.0。
        max_deceleration_mps2: 最大减速度（m/s²），默认 -5.0（负号）。
    """

    def __init__(
        self,
        wheelbase_m: float = 2.8,
        max_steer_rad: float = 0.6,
        max_acceleration_mps2: float = 3.0,
        max_deceleration_mps2: float = -5.0,
    ):
        self.wheelbase_m = wheelbase_m
        self.max_steer_rad = max_steer_rad
        self.max_acceleration_mps2 = max_acceleration_mps2
        self.max_deceleration_mps2 = max_deceleration_mps2

    def plan(
        self,
        trajectory: List[dict],
        ego_state: Optional[dict] = None,
        dt: float = 0.05,
    ) -> List[ControlCommand]:
        """从轨迹生成控制指令序列。**未实现** —— 调用即抛 ``NotImplementedError``。

        Args:
            trajectory: ego-centric 轨迹点列表 ``[{"t": ..., "x": ..., "y": ...,
                "optional_v": ...}, ...]``，由 DecisionClient 输出 + parse_decision_output 校验。
            ego_state: 当前自车状态（含速度 / 航向角等），用于初始化跟踪误差。
                通常直接传 ``EgoState.to_dict()`` 的结果即可。
            dt: 控制循环周期（秒），默认 0.05（20 Hz，CARLA 常用）。

        Returns:
            ``List[ControlCommand]``，长度 = ``ceil(轨迹总时长 / dt)``。

        Raises:
            NotImplementedError: v0.1 仅为接口壳；要实现请参考模块 docstring。
        """
        raise NotImplementedError(
            "DynamicsPlanner.plan 是 v0.1 预留接口。\n"
            "本方法应把决策模型输出的 ego-centric 轨迹翻译为低层控制量\n"
            "(steer/throttle/brake)，预期在接入 CARLA 闭环时实现。\n"
            "推荐内部实现：bicycle model + Pure Pursuit/Stanley/MPC。\n"
            "配置项位于 config/decision.yaml -> dynamics_adapter，\n"
            "其中 wheel_base / max_steer_angle / max_acceleration / "
            "max_deceleration 已就位。"
        )

    def update_config(self, **kwargs: Any) -> None:
        """运行时覆盖某些参数（如 CARLA 中临时换车型时）。

        仅修改已存在的属性，未知 key 抛 ``KeyError``，避免拼写错误。
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise KeyError(f"DynamicsPlanner 没有参数 '{k}'")
            setattr(self, k, v)
