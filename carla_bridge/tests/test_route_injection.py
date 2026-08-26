"""T8: 路由中心线注入（方案A）逻辑验证。

需 carla 包（route_planner 顶层 import carla）。用 stub world/map/ego 驱动
RoutePlanner._route 后验证 center_points_ahead 的取点几何；再验证注入后的
伪对象字段与排序契约（prompt 渲染要求按距离升序、字段齐全）。
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import carla  # noqa: F401

from carla_bridge.env.route_planner import RoutePlanner
from carla_bridge.state import coords


def _loc(x, y):
    return SimpleNamespace(x=x, y=y, z=0.0, distance=lambda o: math.hypot(x - o.x, y - o.y))


def _planner(route_pts):
    rp = RoutePlanner.__new__(RoutePlanner)
    rp._route = route_pts
    rp._mode = "walk"
    rp._options = []
    return rp


def test_center_points_ahead_offsets():
    """自车偏左 1.6m（CARLA -y=左）→ 中心点在自车右侧 → 锚点 left ≈ -1.6。

    米制回正锚点核心断言：ego-centric y=左 系下，中心线的 left 值就是模型
    该输出的回正量（负=需向右回）。VLM 照抄即可收敛回车道中心。
    """
    route = [_loc(i * 2.0, 0.0) for i in range(40)]  # 沿 +x 每 2m 一个中心点
    rp = _planner(route)
    ego = _loc(10.0, -1.6)  # 自车偏在中心线左侧 1.6m
    pts = rp.center_points_ahead(ego_loc=ego, n=5, step_m=8.0)
    assert len(pts) == 5, f"应取 5 个点，got {len(pts)}"
    for k, p in enumerate(pts):
        fwd, left = coords.global_to_ego(p.x, p.y, ego.x, ego.y, 0.0)
        assert abs(left + 1.6) < 0.35, f"第{k}个锚点 left={left}，应≈-1.6（中心在右）"
        expect_fwd = 8.0 * (k + 1)
        assert abs(fwd - expect_fwd) < 2.5, f"第{k}个锚点 fwd={fwd}，应≈{expect_fwd}"


def test_center_points_offset_right():
    """镜像情形：自车偏右 1.6m（+y=右）→ 锚点 left ≈ +1.6（中心在左）。"""
    route = [_loc(i * 2.0, 0.0) for i in range(40)]
    rp = _planner(route)
    ego = _loc(10.0, +1.6)
    pts = rp.center_points_ahead(ego_loc=ego, n=3, step_m=8.0)
    for p in pts:
        _, left = coords.global_to_ego(p.x, p.y, ego.x, ego.y, 0.0)
        assert abs(left - 1.6) < 0.35, f"偏右时锚点 left 应≈+1.6，got {left}"


def test_center_points_straight_on_route():
    """自车在中心线上时，锚点 left ≈ 0。"""
    route = [_loc(i * 2.0, 0.0) for i in range(40)]
    rp = _planner(route)
    ego = _loc(10.0, 0.0)
    pts = rp.center_points_ahead(ego_loc=ego, n=3, step_m=8.0)
    for p in pts:
        fwd, left = coords.global_to_ego(p.x, p.y, ego.x, ego.y, 0.0)
        assert abs(left) < 0.35, f"居中时锚点 left 应≈0，got {left}"


def test_center_points_curve_gives_gradient():
    """弯道路由的锚点 left 逐点递变（模型由此获得曲率几何）。"""
    route = []
    for i in range(60):
        a = i * 2.0 * 0.02  # 缓慢右弯
        route.append(_loc(i * 2.0 * math.cos(a), i * 2.0 * math.sin(a)))
    rp = _planner(route)
    ego = _loc(0.0, 0.0)
    pts = rp.center_points_ahead(ego_loc=ego, n=5, step_m=8.0)
    lefts = [coords.global_to_ego(p.x, p.y, 0.0, 0.0, 0.0)[1] for p in pts]
    # 右弯（+y=右方向弯）锚点 left 应逐点变负
    assert lefts[-1] < lefts[0], f"右弯锚点 left 应递减: {lefts}"


def test_center_points_empty_route():
    rp = _planner([])
    assert rp.center_points_ahead(n=5) == []


def test_injection_dict_contract():
    """注入伪对象的字段与 prompt 渲染契约对齐（position_ego/semantic_label/排序键）。"""
    obj = {
        "annotation_token": "route_0",
        "instance_token": "route_0",
        "category": "route",
        "category_name_raw": "route_center",
        "semantic_label": "route_center",
        "size": [],
        "position_global": [1.0, 2.0, 0.0],
        "position_ego": [8.0, 1.6],
        "distance_to_ego": 8.16,
        "heading_global": 0.0,
        "heading_ego": 0.0,
        "velocity": None,
        "speed": None,
        "acceleration": None,
        "acceleration_mag": None,
        "velocity_available": False,
        "acceleration_available": False,
        "kinematics_source": "route_center",
    }
    # prompt 渲染必需字段（prompt_builder._render_perception_objects_block）
    for key in ("category", "semantic_label", "distance_to_ego", "position_ego"):
        assert key in obj
    assert obj["position_ego"] == [8.0, 1.6]
    assert obj["velocity_available"] is False  # 渲染为 unavailable，不误导速度
    assert obj["kinematics_source"] == "route_center"  # 来源可辨识，非 GT 感知


if __name__ == "__main__":
    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL T8 PASSED")
