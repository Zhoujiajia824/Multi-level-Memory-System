"""CARLA <-> 项目坐标系变换
===========================
项目记忆 / 决策 / 轨迹统一用 **ego-centric** 坐标系：x 前向、y 左向、单位米。
CARLA(Unreal) 全局坐标系：x 前向、y 右向、z 上向，yaw 单位度。

本模块是 CARLA 侧数据进出记忆系统的统一命门：感知对象位置、历史轨迹、决策
轨迹回控都必须经过这里。**坐标手性（y 右 vs y 左）与 yaw 方向是本项目最大
的坑**，集中在本文件处理，其余模块只调用这里的高层函数。

约定
----
* CARLA yaw：0° = 朝 +x（前），**正方向逆时针（向左）**。若 P1 实测发现左右
  镜像（CARLA 在该版本是顺时针为正），把模块常量 ``YAW_SIGN`` 改成 -1 即可，
  无需改其它代码。
* ego-centric：x 前向、y 左向（与 ``src/vla_memory`` 完全一致）。
* 所有角度在项目内部一律用弧度；只在进出 CARLA API 时用度。
"""
from __future__ import annotations

import math
from typing import Tuple

# CARLA yaw 正方向：+1 = 逆时针(向左)为正（标准数学约定，多数 CARLA 版本如此）。
# 实测若左右镜像，改成 -1。**全项目唯一的手性开关**。
YAW_SIGN: int = 1


def carla_yaw_deg_to_rad(yaw_deg: float) -> float:
    """CARLA yaw(度) -> 项目 yaw(弧度)，考虑 ``YAW_SIGN``，归一化到 [-pi, pi]。"""
    yaw_rad = YAW_SIGN * math.radians(float(yaw_deg))
    while yaw_rad > math.pi:
        yaw_rad -= 2.0 * math.pi
    while yaw_rad < -math.pi:
        yaw_rad += 2.0 * math.pi
    return yaw_rad


def global_to_ego(
    px: float, py: float, ego_x: float, ego_y: float, ego_yaw_rad: float
) -> Tuple[float, float]:
    """CARLA 全局点 -> ego-centric ``(forward, left)``。

    Args:
        px, py: CARLA 全局坐标（y 向右）。
        ego_x, ego_y: 自车全局位置。
        ego_yaw_rad: 自车朝向（弧度，已用 :func:`carla_yaw_deg_to_rad` 转换）。

    Returns:
        ``(forward, left)``：ego-centric 前向/左向分量，米。
    """
    dx = px - ego_x
    dy = py - ego_y  # CARLA y 向右
    cos_y = math.cos(ego_yaw_rad)
    sin_y = math.sin(ego_yaw_rad)
    # 旋转 -yaw 进入 ego 系（x 前, y 右），再把 y 右 -> y 左（取反）
    forward = dx * cos_y + dy * sin_y
    right = -dx * sin_y + dy * cos_y
    left = -right
    return forward, left


def rotate_vector_to_ego(
    vx: float, vy: float, ego_yaw_rad: float
) -> Tuple[float, float]:
    """把 CARLA 全局向量（y 右）旋转到 ego-centric ``(forward, left)``，不平移。

    用于速度/加速度等自由向量（只需旋转，不需减自车位置）。
    """
    cos_y = math.cos(ego_yaw_rad)
    sin_y = math.sin(ego_yaw_rad)
    forward = vx * cos_y + vy * sin_y
    right = -vx * sin_y + vy * cos_y
    return forward, -right


def ego_to_global(
    forward: float, left: float, ego_x: float, ego_y: float, ego_yaw_rad: float
) -> Tuple[float, float]:
    """ego-centric ``(forward, left)`` -> CARLA 全局 ``(gx, gy)``。

    :func:`global_to_ego` 的逆，用于把决策的 ego-centric 轨迹转回全局 waypoint
    供控制器跟踪。
    """
    right = -left
    cos_y = math.cos(ego_yaw_rad)
    sin_y = math.sin(ego_yaw_rad)
    # 旋转 +yaw 回全局系
    dx = forward * cos_y - right * sin_y
    dy = forward * sin_y + right * cos_y
    return ego_x + dx, ego_y + dy


def heading_ego(global_heading_rad: float, ego_yaw_rad: float) -> float:
    """全局朝向角 -> 相对 ego 的朝向角（弧度，归一化 [-pi, pi]）。"""
    diff = global_heading_rad - ego_yaw_rad
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi
    return diff


def trajectory_ego_to_global(
    trajectory: list, ego_x: float, ego_y: float, ego_yaw_rad: float
) -> list:
    """把决策输出的 ego-centric 轨迹整体转成全局 waypoint 列表。

    Args:
        trajectory: ``[{"t":..., "x":forward, "y":left, "optional_v":...}, ...]``，
            即 :class:`DecisionOutput.trajectory` 的格式（x 前、y 左）。
        ego_x, ego_y, ego_yaw_rad: 捕获时刻自车全局位姿。

    Returns:
        ``[{"t":..., "x":gx, "y":gy, "optional_v":...}, ...]``，全局坐标，
        供 Pure Pursuit 在自车移动后继续跟踪。
    """
    out = []
    for wp in trajectory or []:
        fwd = float(wp.get("x", 0.0))
        lft = float(wp.get("y", 0.0))
        gx, gy = ego_to_global(fwd, lft, ego_x, ego_y, ego_yaw_rad)
        item = {"t": wp.get("t"), "x": gx, "y": gy}
        if "optional_v" in wp:
            item["optional_v"] = wp["optional_v"]
        out.append(item)
    return out
