"""离散制动释放包络；速度m/s，加速度m/s²，jerk为m/s³，步长s。

模型与MPC一致：每段加速度恒定，下一段最多以jerk*dt提高加速度。
这是非倒退与制动释放的纵向可行条件，不证明道路或交通安全。
"""
from __future__ import annotations

import math


def _inputs(value, dt, jerk):
    value, dt, jerk = float(value), float(dt), float(jerk)
    if not all(math.isfinite(v) for v in (value, dt, jerk)) or dt <= 0 or jerk <= 0:
        raise ValueError('制动释放包络需要有限状态、正步长和正jerk上限')
    return value, dt, jerk


def stopping_speed_required(acceleration, dt, jerk):
    """从本段负加速度开始，以最大正jerk逐段释放到0所需的最小速度。"""
    acceleration, dt, jerk = _inputs(acceleration, dt, jerk)
    b = max(0., -acceleration)
    q = jerk * dt
    n = max(1, int(math.ceil(b / q)))
    return dt * (n * b - q * n * (n - 1) / 2)


def brake_acceleration_limit(speed, dt, jerk):
    """返回非负b_max；若本段a>=-b_max，最快释放制动可保持未来速度非负。

    可行带还必须与前一真实加速度的jerk区间相交。初态区间为空时，
    不能通过本函数声称同时满足jerk与非倒退。
    """
    speed, dt, jerk = _inputs(speed, dt, jerk)
    if speed < 0:
        raise ValueError('制动释放包络不接受负速度')
    q = jerk * dt
    ratio = speed / (q * dt)
    n = max(1, int(math.ceil((math.sqrt(1 + 8 * ratio) - 1) / 2)))
    return (speed / dt + q * n * (n - 1) / 2) / n
