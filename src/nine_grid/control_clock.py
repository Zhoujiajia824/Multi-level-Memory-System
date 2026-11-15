"""重规划时钟与优化采样网格的独立契约；单位均为秒或赫兹。

本模块只使用标准库，CARLA SDK 桥也可复用。降低重规划频率不改变
MPC 时域、预测采样或物理积分精度；执行器消费当前规划的完整控制前缀。
"""
from __future__ import annotations

import math


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' 必须是有限正数')
    return float(value)


def _integer_ratio(value, name):
    if not math.isfinite(value):
        raise ValueError(name + ' 必须为有限值')
    rounded = round(value)
    if rounded < 1 or not math.isclose(value, rounded, rel_tol=0., abs_tol=1e-8):
        raise ValueError(name + ' 必须为正整数')
    return int(rounded)


def execution_prefix_count(config=None):
    """返回一次重规划之间必须执行的 MPC 控制段数，不修改配置。

    例如物理 50Hz、MPC 网格 0.1s、重规划 1Hz 时，每个 MPC 段
    仍含 5 个物理子步；本次控制窗口消费 10 段，而非保持首控制 1s。
    """
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError('控制时钟配置必须为对象')
    clock, mpc = config.get('clock', {}), config.get('mpc', {})
    if not isinstance(clock, dict) or not isinstance(mpc, dict):
        raise ValueError('clock 和 mpc 必须为配置对象')
    physics_hz = _positive(clock.get('physics_hz', 50), 'clock.physics_hz')
    planner_hz = _positive(clock.get('planner_hz', 10), 'clock.planner_hz')
    step_s = _positive(mpc.get('step_s', .1), 'mpc.step_s')
    physics_steps = _integer_ratio(step_s * physics_hz, '每个 MPC 段的物理子步数')
    substeps = mpc.get('physics_substeps', 5)
    if type(substeps) is not int or substeps != physics_steps:
        raise ValueError('MPC 子步数必须等于 MPC 采样间隔乘物理频率')
    prefix = _integer_ratio(1. / (planner_hz * step_s), '重规划周期包含的 MPC 控制段数')
    horizon_steps = mpc.get('horizon_steps', 30)
    if type(horizon_steps) is not int or horizon_steps < 2:
        raise ValueError('MPC 时域步数必须为至少 2 的整数')
    if prefix > horizon_steps:
        raise ValueError('重规划执行窗口不能超过当前 MPC 控制时域')
    return prefix
