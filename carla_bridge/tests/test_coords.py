"""T1: coords 坐标变换正确性（钉死手性事实链）。

事实链：CARLA 全局系 x前/y右/z上，+yaw=右转（YAW_SIGN=+1 下项目 yaw 同约定）。
ego-centric：x 前、y 左。

关键判定用例：ego 朝 +y（yaw=90°，右转 90° 后的朝向）时——
  全局 +x 方向的点应落在 ego 左侧（left=+1），因为朝 +y 时左手边是 +x。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carla_bridge.state import coords


def test_yaw_conversion_and_normalization():
    # +90° CARLA yaw = +pi/2 项目 yaw（右转 90°）
    assert abs(coords.carla_yaw_deg_to_rad(90) - math.pi / 2) < 1e-9
    # 归一化到 [-pi, pi]
    assert abs(coords.carla_yaw_deg_to_rad(270) - (-math.pi / 2)) < 1e-9
    assert abs(coords.carla_yaw_deg_to_rad(-270) - (math.pi / 2)) < 1e-9


def test_chirality_key_case():
    """ego 朝 +y（yaw=+90°=右转90°）时，全局 +x 的点在 ego 左手边。"""
    fwd, left = coords.global_to_ego(1.0, 0.0, 0.0, 0.0, math.pi / 2)
    # ego 朝 +y：正前方(+y)是 forward，+x 在左手侧
    assert abs(fwd - 0.0) < 1e-9
    assert abs(left - 1.0) < 1e-9, f"全局 +x 应在 ego 左侧，got left={left}"


def test_forward_and_right():
    """ego 朝 +x（yaw=0）时：+y=右侧（left=-1），+x=前方。"""
    fwd, left = coords.global_to_ego(0.0, 1.0, 0.0, 0.0, 0.0)
    assert abs(fwd - 0.0) < 1e-9
    assert abs(left - (-1.0)) < 1e-9, "CARLA +y=右 → left=-1"


def test_roundtrip():
    """global_to_ego / ego_to_global 互逆（多个朝向）。"""
    for yaw_deg in (0, 30, 90, 135, 180, -90):
        yaw = coords.carla_yaw_deg_to_rad(yaw_deg)
        for fwd, lft in ((5.0, 0.0), (3.0, 2.0), (-1.0, -4.0)):
            gx, gy = coords.ego_to_global(fwd, lft, 10.0, -5.0, yaw)
            f2, l2 = coords.global_to_ego(gx, gy, 10.0, -5.0, yaw)
            assert abs(f2 - fwd) < 1e-9 and abs(l2 - lft) < 1e-9, (
                f"roundtrip 失败 @yaw={yaw_deg}: ({fwd},{lft}) -> ({f2},{l2})"
            )


def test_rotate_vector_to_ego():
    """ego 朝 +y 时全局速度 (0, 5)（朝 +y）应转成 forward=5, left=0。"""
    fwd, left = coords.rotate_vector_to_ego(0.0, 5.0, math.pi / 2)
    assert abs(fwd - 5.0) < 1e-9 and abs(left - 0.0) < 1e-9
    # 全局速度 (5, 0)（朝 +x）在朝 +y 的 ego 系里是纯左向
    fwd, left = coords.rotate_vector_to_ego(5.0, 0.0, math.pi / 2)
    assert abs(left - 5.0) < 1e-9 and abs(fwd - 0.0) < 1e-9


def test_trajectory_transform():
    """轨迹整体转全局后逐点可逆。"""
    traj = [
        {"t": 0.0, "x": 0.0, "y": 0.0, "optional_v": 5.0},
        {"t": 0.1, "x": 1.0, "y": 0.5, "optional_v": 5.0},
        {"t": 0.2, "x": 2.0, "y": 1.0, "optional_v": 4.5},
    ]
    yaw = coords.carla_yaw_deg_to_rad(45)
    gtraj = coords.trajectory_ego_to_global(traj, 3.0, 4.0, yaw)
    assert len(gtraj) == 3 and "optional_v" in gtraj[2]
    f, l = coords.global_to_ego(gtraj[1]["x"], gtraj[1]["y"], 3.0, 4.0, yaw)
    assert abs(f - 1.0) < 1e-9 and abs(l - 0.5) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL T1 PASSED")
