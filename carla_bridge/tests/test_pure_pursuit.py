"""T2: Pure Pursuit 横向控制（左正 steer_rad）。

纯数学，不依赖 carla 包。waypoints 是 CARLA 全局坐标（+y=右）；PurePursuit 内部
经 coords.global_to_ego 转成 (fwd, left) 再算 alpha=atan2(left, fwd)，故：
  全局点在 ego 左侧 → left>0 → steer_rad>0（项目左正）。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carla_bridge.control.pure_pursuit import PurePursuit


def _straight_line_wps(n=30, step=1.0):
    return [{"x": i * step, "y": 0.0} for i in range(n)]


def test_target_on_left_gives_positive_steer():
    """ego 朝 +x，全局点 (10,-5)：-y=左侧 → steer_rad > 0（左正）。"""
    pp = PurePursuit()
    steer = pp.compute_steer(0.0, 0.0, 0.0, 5.0, [{"x": 10.0, "y": -5.0}])
    assert steer > 0, f"左侧目标应产生左正 steer，got {steer}"


def test_target_on_right_gives_negative_steer():
    """ego 朝 +x，全局点 (10,+5)：+y=右侧 → steer_rad < 0。"""
    pp = PurePursuit()
    steer = pp.compute_steer(0.0, 0.0, 0.0, 5.0, [{"x": 10.0, "y": 5.0}])
    assert steer < 0, f"右侧目标应产生右负 steer，got {steer}"


def test_on_straight_line_steer_near_zero():
    """自车在直线上朝 +x，轨迹沿 x 轴 → steer ≈ 0。"""
    pp = PurePursuit()
    steer = pp.compute_steer(0.0, 0.0, 0.0, 5.0, _straight_line_wps())
    assert abs(steer) < 1e-6


def test_offset_right_of_line_gives_left_correction():
    """自车偏到线右侧（全局 y=+2，+y=右），线为 y=0 → 向左修正 steer>0。"""
    pp = PurePursuit()
    steer = pp.compute_steer(0.0, 2.0, 0.0, 5.0, _straight_line_wps())
    assert steer > 0, f"偏右应左修，got {steer}"


def test_empty_returns_zero():
    pp = PurePursuit()
    assert pp.compute_steer(0, 0, 0, 5, []) == 0.0


def test_lookahead_monotone_with_speed():
    """速度越高前瞻越远，同横向偏移比例下转向更缓。"""
    pp = PurePursuit(lookahead_min=3.0, lookahead_max=12.0, lookahead_k=0.5)
    wps = [{"x": float(i), "y": 0.5} for i in range(1, 31)]
    s_low = pp.compute_steer(0, 0, 0, 0.0, wps)
    s_high = pp.compute_steer(0, 0, 0, 30.0, wps)
    assert abs(s_high) < abs(s_low), (
        f"高速应选更远前瞻点、转向更缓，low={s_low}, high={s_high}"
    )


if __name__ == "__main__":
    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL T2 PASSED")
