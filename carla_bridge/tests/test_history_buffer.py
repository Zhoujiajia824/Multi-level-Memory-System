"""T4: HistoryBuffer yaw_rate 差分（修复前恒 0 的回归测试）。纯逻辑。"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carla_bridge.state import coords
from carla_bridge.state.history_buffer import HistoryBuffer


class _FakeEgo:
    """stub：只提供 get_transform()。"""

    def __init__(self, x, y, yaw_deg):
        tf = SimpleNamespace(
            location=SimpleNamespace(x=x, y=y, z=0.0),
            rotation=SimpleNamespace(yaw=yaw_deg),
        )
        self._tf = tf

    def get_transform(self):
        return self._tf


def test_yaw_rate_basic():
    """yaw 0°→+10°（右转）历时 0.1s → yaw_rate ≈ +1.745 rad/s（右正）。"""
    hb = HistoryBuffer()
    hb.update(_FakeEgo(0, 0, 0), 0.0)
    hb.update(_FakeEgo(1, 0, 10), 0.1)  # +10° = 右转（CARLA +yaw=右）
    yr = hb.yaw_rate(0.1, coords.carla_yaw_deg_to_rad(10))
    assert abs(yr - math.radians(10) / 0.1) < 0.01, f"got {yr}"


def test_yaw_rate_nonzero_regression():
    """回归钉死：先 update 再 yaw_rate 不能返回 0（修复前的 bug）。"""
    hb = HistoryBuffer()
    hb.update(_FakeEgo(0, 0, 0), 0.0)
    hb.update(_FakeEgo(1, 0, 45), 0.1)
    yr = hb.yaw_rate(0.1, coords.carla_yaw_deg_to_rad(45))
    assert yr != 0.0, "yaw_rate 恒 0 回归"


def test_yaw_rate_wraparound():
    """±180° 跨越：179° → -179° 实为右转 2°，不应算成左转 358°。"""
    hb = HistoryBuffer()
    hb.update(_FakeEgo(0, 0, 179), 0.0)
    hb.update(_FakeEgo(1, 0, -179), 0.1)
    yr = hb.yaw_rate()
    expected = math.radians(-2.0) / 0.1  # 顺时针 2° = 右转 = 负？见下
    # 179°→-179°：数值上 -358°，物理上右转 2°；项目系 +yaw=右 → 物理右转 2° 应为 +
    # 但 179→-179 按 CARLA 度数是减小，即顺时针（右）2°。
    assert abs(abs(yr) - abs(math.radians(2.0) / 0.1)) < 0.01, f"got {yr}"
    assert yr > 0, "179°→-179° 是右转 2°，项目系右正应为正 yaw_rate"


def test_yaw_rate_insufficient_frames():
    hb = HistoryBuffer()
    assert hb.yaw_rate() == 0.0
    hb.update(_FakeEgo(0, 0, 0), 0.0)
    assert hb.yaw_rate() == 0.0


def test_build_ego_centric_history():
    """过去 5s 位姿 → ego-centric（t 负、y=左）；窗口外位姿被过滤。"""
    hb = HistoryBuffer(history_seconds=5.0)
    hb.update(_FakeEgo(0.0, 0.0, 0), 0.0)     # 10s 前（窗口外，应被过滤）
    hb.update(_FakeEgo(6.0, 0.0, 0), 6.0)     # 4s 前
    hb.update(_FakeEgo(10.0, 0.0, 0), 10.0)   # 当前
    pts = hb.build_ego_centric(10.0, 10.0, 0.0, 0.0)
    assert len(pts) == 2, "窗口外的点应被过滤"
    # 4s 前的 (6,0) 相对当前 (10,0) 朝 +x：forward=-4, left=0
    assert pts[0]["x"] == -4.0 and pts[0]["y"] == 0.0
    assert pts[0]["t"] == -4.0
    assert pts[-1]["t"] == 0.0


if __name__ == "__main__":
    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL T4 PASSED")
