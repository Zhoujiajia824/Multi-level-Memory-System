"""T7: 长尾事件侧向几何 lateral_offset（钉死 Bug 3：side=right 真在右手边）。

需 import carla 包（walker_controller 顶层 import）。stub transform。
右手边判定：fwd × offset 的符号在 CARLA 左手系（x前 y右）下，右手侧向 = (-fwd.y, fwd.x)。
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import carla  # noqa: F401

from carla_bridge.env.walker_controller import lateral_offset
from carla_bridge.state import coords


def _tf(x, y, yaw_deg):
    yaw_rad = coords.carla_yaw_deg_to_rad(yaw_deg)
    fwd = SimpleNamespace(x=math.cos(yaw_rad), y=math.sin(yaw_rad))
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=0.0),
        get_forward_vector=lambda f=fwd: f,
    )


def _right_unit(fwd):
    """CARLA 全局系下 fwd 的右手单位向量：(-fwd.y, fwd.x)。"""
    return -fwd.y, fwd.x


def test_right_offset_at_yaw_0():
    tf = _tf(0, 0, 0)
    dx, dy = lateral_offset(tf, 1.0, 8.0)
    # ego 朝 +x：右手边是 +y
    assert abs(dx - 0.0) < 1e-9 and abs(dy - 8.0) < 1e-9, f"got ({dx},{dy})"


def test_right_offset_at_yaw_90():
    """ego 朝 +y（右转 90° 后）：右手边随车身旋转到 -x。"""
    tf = _tf(0, 0, 90)
    dx, dy = lateral_offset(tf, 1.0, 8.0)
    assert abs(dx + 8.0) < 1e-9 and abs(dy - 0.0) < 1e-9, (
        f"朝 +y 时右手边在 -x，got ({dx},{dy})"
    )


def test_right_offset_at_yaw_180():
    """ego 朝 -x：右手边是 -y。"""
    tf = _tf(0, 0, 180)
    dx, dy = lateral_offset(tf, 1.0, 8.0)
    assert abs(dx - 0.0) < 1e-9 and abs(dy + 8.0) < 1e-9, f"got ({dx},{dy})"


def test_left_is_mirror_of_right():
    for yaw_deg in (0, 37, 90, 143, 180, -45):
        tf = _tf(0, 0, yaw_deg)
        rdx, rdy = lateral_offset(tf, 1.0, 5.0)
        ldx, ldy = lateral_offset(tf, -1.0, 5.0)
        assert abs(rdx + ldx) < 1e-9 and abs(rdy + ldy) < 1e-9, (
            f"左右应镜像 @yaw={yaw_deg}"
        )


def test_offset_always_perpendicular_and_magnitude():
    for yaw_deg in (0, 30, 90, 200, -60):
        tf = _tf(0, 0, yaw_deg)
        dx, dy = lateral_offset(tf, 1.0, 8.0)
        assert abs(math.hypot(dx, dy) - 8.0) < 1e-9, "偏移模长应等于 dist"
        fwd = tf.get_forward_vector()
        # 垂直性：dot(fwd, offset) == 0
        assert abs(fwd.x * dx + fwd.y * dy) < 1e-9, "侧向偏移应垂直于前向"


if __name__ == "__main__":
    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL T7 PASSED")
