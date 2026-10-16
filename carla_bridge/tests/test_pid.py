"""纵向 PID：饱和、反向误差、重规划和死区后的反馈连续性。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carla_bridge.control.pid import PID


def test_positive_error_gives_throttle():
    pid = PID(kp=1.6, ki=0.05, kd=0.0, deadband_mps=0.15)
    throttle, brake = pid.compute(2.0, dt=0.1)  # 需要加速 2 m/s
    assert throttle > 0 and brake == 0.0
    # kp=1.6 时油门应明显大于旧 kp=1.0 的 0.2
    assert throttle > 0.3, f"2m/s 误差油门应>0.3（kp=1.6），got {throttle}"


def test_negative_error_gives_brake():
    pid = PID(kp=1.6, deadband_mps=0.15)
    throttle, brake = pid.compute(-2.0, dt=0.1)
    assert throttle == 0.0 and brake > 0


def test_deadband_coasts():
    """误差在死区内 → (0,0) 滑行，不抖动。"""
    pid = PID(kp=1.6, deadband_mps=0.15)
    for err in (0.1, 0.0, -0.1):
        throttle, brake = pid.compute(err, dt=0.1)
        assert throttle == 0.0 and brake == 0.0, f"死区内应滑行，err={err}"


def test_soft_reset_keeps_integral():
    """soft_reset 清微分项，但保留未饱和输出时的积分。"""
    pid = PID(kp=1.0, ki=0.1, kd=0.05, deadband_mps=0.15)
    for _ in range(20):
        pid.compute(0.2, dt=0.1)  # 非饱和输出才积累积分
    i_before = pid._i
    assert i_before > 0
    pid.soft_reset()
    assert pid._i == i_before, "soft_reset 不应清积分"
    assert pid._prev_err is None, "soft_reset 应清微分基准"
    pid.reset()
    assert pid._i == 0.0, "硬 reset 清积分"


def test_integral_anti_windup():
    """积分被钳制在 ±5。"""
    pid = PID(kp=0.1, ki=0.5, deadband_mps=0.0)
    for _ in range(500):
        pid.compute(3.0, dt=0.1)
    assert pid._i <= 5.0 + 1e-9


def test_deadband_disabled_legacy():
    """deadband=0 时行为退化为经典 PID（不滑行）。"""
    pid = PID(kp=1.0, ki=0.0, kd=0.0, deadband_mps=0.0)
    t, b = pid.compute(0.05, dt=0.1)
    assert t == 0.05 and b == 0.0


def test_saturated_acceleration_does_not_store_throttle_for_later():
    pid = PID(kp=1.6, ki=0.5, kd=0.0, deadband_mps=0.0)
    for _ in range(100):
        throttle, brake = pid.compute(3.0, dt=0.1)
        assert throttle == pid.max_throttle and brake == 0.0
    assert pid._i == 0.0
    throttle, brake = pid.compute(-0.2, dt=0.1)
    assert throttle == 0.0 and brake > 0.0


def test_braking_releases_integral_from_unsaturated_acceleration():
    pid = PID(kp=0.1, ki=0.5, kd=0.0, deadband_mps=0.0)
    for _ in range(5):
        pid.compute(1.0, dt=0.1)
    assert pid._i > 0.0
    throttle, brake = pid.compute(-0.2, dt=0.1)
    assert throttle == 0.0 and brake > 0.0
    assert pid._i < 0.0


def test_measured_speed_derivative_ignores_target_step():
    pid = PID(kp=0.2, ki=0.0, kd=1.0, deadband_mps=0.0)
    pid.compute(0.0, dt=0.1, current_speed=1.0)
    throttle, brake = pid.compute(0.2, dt=0.1, current_speed=1.0)
    assert throttle == pytest.approx(0.04)
    assert brake == 0.0


def test_deadband_updates_measurement_instead_of_leaving_stale_derivative():
    pid = PID(kp=1.0, ki=0.0, kd=0.1, deadband_mps=0.15)
    pid.compute(1.0, dt=0.1, current_speed=1.0)
    assert pid.compute(0.05, dt=0.1, current_speed=1.5) == (0.0, 0.0)
    throttle, brake = pid.compute(0.4, dt=0.1, current_speed=1.6)
    assert throttle == pytest.approx(0.3)
    assert brake == 0.0


def test_planned_stop_still_brakes_inside_coasting_deadband():
    pid = PID(kp=1.0, ki=0.0, kd=0.0, deadband_mps=0.15)
    throttle, brake = pid.compute(-0.1, current_speed=0.1, allow_coast=False)
    assert throttle == 0.0 and brake == pytest.approx(0.1)


@pytest.mark.parametrize("error,dt,current_speed", [
    (float("nan"), 0.1, 0.0), (0.0, 0.0, 0.0),
    (0.0, float("inf"), 0.0), (0.0, 0.1, float("nan")),
])
def test_invalid_feedback_is_not_converted_into_actuator_command(error, dt, current_speed):
    with pytest.raises(ValueError):
        PID().compute(error, dt, current_speed=current_speed)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
