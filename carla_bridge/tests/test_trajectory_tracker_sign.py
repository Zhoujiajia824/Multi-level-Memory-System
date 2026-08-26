"""T5: 转向符号端到端（钉死 Bug 1：左偏轨迹 → CARLA steer<0 即左转）。

需 import carla 包（构造 VehicleControl 不需要服务器）。
事实链：项目 steer_rad 左正；CARLA VehicleControl.steer 正=右转；
故 steer_sign=-1 时左偏轨迹的最终 steer_norm < 0（CARLA 负=左）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import carla  # noqa: F401  仅验证包可用

from carla_bridge.control.trajectory_tracker import TrajectoryTracker


def _cfg():
    return {
        "wheelbase_m": 2.8,
        "max_steer_rad": 0.6,
        "lookahead_min_m": 3.0,
        "lookahead_max_m": 12.0,
        "lookahead_k": 0.5,
        "pid": {"kp": 1.6, "ki": 0.05, "kd": 0.05,
                "max_throttle": 0.75, "max_brake": 0.8, "deadband_mps": 0.15},
        "steer_sign": -1,
    }


def test_left_trajectory_gives_carla_left_steer():
    """ego 朝 +x，决策轨迹向左弯（ego 系 y=左 为正）→ CARLA steer<0（左）。"""
    tracker = TrajectoryTracker(_cfg(), control_dt_s=0.1)
    left_traj = [{"t": i * 0.1, "x": float(i), "y": 0.5 * i} for i in range(1, 26)]
    tracker.set_trajectory(left_traj, 0.0, 0.0, 0.0, 5.0)
    ctrl = tracker.compute_control(0.0, 0.0, 0.0, 5.0)
    assert ctrl.steer < 0, (
        f"左偏轨迹必须下发 CARLA 左转（steer<0），got {ctrl.steer}——steer_sign 镜像回归！"
    )


def test_right_trajectory_gives_carla_right_steer():
    tracker = TrajectoryTracker(_cfg(), control_dt_s=0.1)
    right_traj = [{"t": i * 0.1, "x": float(i), "y": -0.5 * i} for i in range(1, 26)]
    tracker.set_trajectory(right_traj, 0.0, 0.0, 0.0, 5.0)
    ctrl = tracker.compute_control(0.0, 0.0, 0.0, 5.0)
    assert ctrl.steer > 0, f"右偏轨迹应 CARLA 右转（steer>0），got {ctrl.steer}"


def test_default_config_is_negated():
    """无 steer_sign 键时默认 -1（防配置漂移回归）。"""
    cfg = _cfg()
    del cfg["steer_sign"]
    tracker = TrajectoryTracker(cfg, control_dt_s=0.1)
    assert tracker.steer_sign == -1


def test_no_trajectory_full_brake():
    tracker = TrajectoryTracker(_cfg(), control_dt_s=0.1)
    tracker.set_trajectory([], 0.0, 0.0, 0.0, 5.0)
    ctrl = tracker.compute_control(0.0, 0.0, 0.0, 5.0)
    assert ctrl.brake == 1.0 and ctrl.steer == 0.0


if __name__ == "__main__":
    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL T5 PASSED")
