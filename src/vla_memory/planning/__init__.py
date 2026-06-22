"""规划接口模块（v0.1 预留壳）
================================
P7 新增。本模块提供两个预留接口，本身不做实质规划——目的是给后续接入
CARLA 闭环 / 多模态轨迹规划留好稳定的入口，避免后期改链路。

  - ``DynamicsPlanner``  —— 轨迹 → 低层控制量（throttle/brake/steer），
    适用于 CARLA 等仿真闭环。预留 bicycle / MPC / PID 等实现位。
  - ``TrajectorySampler`` —— 多模态轨迹候选 → 选定轨迹 + 评分，
    适用于扩散 / 自回归规划器输出多条候选时的下游"挑最优"环节。

具体接入指引见 ``DynamicsPlanner`` / ``TrajectorySampler`` 的 docstring。
"""
from .dynamics_planner import DynamicsPlanner
from .trajectory_sampler import TrajectorySampler

__all__ = ["DynamicsPlanner", "TrajectorySampler"]
