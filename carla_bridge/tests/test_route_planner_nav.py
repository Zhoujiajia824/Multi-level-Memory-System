"""T6: nav 语义左右标签（钉死 Bug 2：+yaw 差 = 右转）。

需 import carla 包（route_planner 顶层 import）。用 stub waypoint，不需要服务器。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import carla  # noqa: F401

from carla_bridge.env.route_planner import RoutePlanner


def _wp(yaw_deg):
    """stub carla.Waypoint：只需 transform.rotation.yaw。"""
    return SimpleNamespace(
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=yaw_deg),
        ),
        road_id=0, lane_id=0, s=0.0,
    )


def test_positive_yaw_delta_is_right_turn():
    """yaw 增加（+30°）= 右转 → NAV_RIGHT。"""
    nav = RoutePlanner._infer_nav(0.0, _wp(30))
    assert nav == "right_turn", f"+yaw=右转，got {nav}"


def test_negative_yaw_delta_is_left_turn():
    nav = RoutePlanner._infer_nav(0.0, _wp(-30))
    assert nav == "left_turn", f"-yaw=左转，got {nav}"


def test_small_delta_is_lane_follow():
    for dy in (5, -5):
        nav = RoutePlanner._infer_nav(0.0, _wp(dy))
        assert nav == "lane_follow", f"小变化应 lane_follow，got {nav} (dy={dy})"


def test_wraparound_170_to_minus160():
    """170°→-160°：物理右转 30° → NAV_RIGHT（diff 归一化 +30°）。"""
    from carla_bridge.state import coords
    cur = coords.carla_yaw_deg_to_rad(170)
    nav = RoutePlanner._infer_nav(cur, _wp(-160))
    assert nav == "right_turn", f"170→-160 应右转（+30° wrap），got {nav}"


def test_wraparound_minus170_to_160():
    """-170°→160°：物理左转 30° → NAV_LEFT。"""
    from carla_bridge.state import coords
    cur = coords.carla_yaw_deg_to_rad(-170)
    nav = RoutePlanner._infer_nav(cur, _wp(160))
    assert nav == "left_turn", f"-170→160 应左转，got {nav}"


if __name__ == "__main__":
    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL T6 PASSED")
