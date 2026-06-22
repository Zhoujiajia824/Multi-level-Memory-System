"""评测指标模块
==============
计算 ADE、FDE、轨迹有效率、行为准确率等评测指标。
支持轨迹插值/重采样。预留 collision_proxy 和 offroad_proxy 接口。
所有函数使用中文 docstring。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("metrics")


def compute_ade(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """计算 Average Displacement Error (ADE)。

    对所有对齐的 waypoint 计算欧氏距离后取平均。
    当两条轨迹长度不同时，只比较前 min(len(pred), len(gt)) 个点。

    Args:
        predicted: 预测轨迹点列表，每个包含 x, y。
        ground_truth: 真值轨迹点列表，每个包含 x, y。

    Returns:
        ADE 值（米）。输入为空时返回 inf。

    Raises:
        ValueError: 轨迹中存在 NaN 或 Inf 值时。
    """
    if not predicted or not ground_truth:
        return float("inf")

    n = min(len(predicted), len(ground_truth))
    if n == 0:
        return float("inf")

    # 检查 NaN / Inf
    _check_finite(predicted[:n], "预测轨迹")
    _check_finite(ground_truth[:n], "真值轨迹")

    total_displacement = 0.0
    for i in range(n):
        px, py = predicted[i].get("x", 0), predicted[i].get("y", 0)
        gx, gy = ground_truth[i].get("x", 0), ground_truth[i].get("y", 0)
        total_displacement += math.sqrt((px - gx) ** 2 + (py - gy) ** 2)

    return total_displacement / n


def compute_fde(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """计算 Final Displacement Error (FDE)。

    取预测轨迹和真值轨迹的最后一个对齐 waypoint 之间的欧氏距离。
    建议在调用前先使用 resample_trajectory 统一点数。

    Args:
        predicted: 预测轨迹点列表。
        ground_truth: 真值轨迹点列表。

    Returns:
        FDE 值（米）。输入为空时返回 inf。

    Raises:
        ValueError: 终点坐标为 NaN 或 Inf 时。
    """
    if not predicted or not ground_truth:
        return float("inf")

    # 对齐到最后一个点（取较短轨迹的最后一个点）
    n = min(len(predicted), len(ground_truth))
    pred_last = predicted[n - 1]
    gt_last = ground_truth[n - 1]

    px, py = pred_last.get("x", 0), pred_last.get("y", 0)
    gx, gy = gt_last.get("x", 0), gt_last.get("y", 0)

    # 检查终点坐标是否有限
    for name, val in [("预测轨迹终点 x", px), ("预测轨迹终点 y", py),
                      ("真值轨迹终点 x", gx), ("真值轨迹终点 y", gy)]:
        if not math.isfinite(val):
            raise ValueError(f"{name} 包含非有限值: {val}")

    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2)


def is_valid_trajectory(
    trajectory: List[Dict],
    min_waypoints: int = 20,
    max_waypoints: int = 30,
    max_step_displacement: float = 5.0,
) -> Tuple[bool, Optional[str]]:
    """判断轨迹是否有效，并返回错误原因。

    检查项：
    1. 必须是 list 类型且不为空。
    2. waypoint 数量在 [min_waypoints, max_waypoints] 范围内。
    3. 每个 waypoint 必须是 dict 且包含 x、y 和 t 字段。
    4. 每个坐标值必须有限（非 NaN、非 Inf）。
    5. 相邻 waypoint 之间的位移不超过 max_step_displacement。

    Args:
        trajectory: 轨迹点列表。
        min_waypoints: 最少 waypoint 数（默认 20）。
        max_waypoints: 最多 waypoint 数（默认 30）。
        max_step_displacement: 单步最大位移（米，默认 5.0）。

    Returns:
        (是否有效, 错误原因) 元组。有效时错误原因为 None。
    """
    if not isinstance(trajectory, list):
        return False, "轨迹不是 list 类型"

    if len(trajectory) == 0:
        return False, "轨迹为空"

    if len(trajectory) < min_waypoints:
        return False, f"waypoint 数量不足: {len(trajectory)} < {min_waypoints}"

    if len(trajectory) > max_waypoints:
        return False, f"waypoint 数量过多: {len(trajectory)} > {max_waypoints}"

    for i, wp in enumerate(trajectory):
        if not isinstance(wp, dict):
            return False, f"第 {i} 个 waypoint 不是 dict 类型"

        # 检查必要字段
        missing_fields = []
        for field in ("x", "y"):
            if field not in wp:
                missing_fields.append(field)
        if missing_fields:
            return False, f"第 {i} 个 waypoint 缺少字段: {missing_fields}"

        # 检查数值有限性
        for field in ("x", "y"):
            val = wp[field]
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                return False, f"第 {i} 个 waypoint 的 {field} 值非有限: {val}"

        # 检查 t 字段（如果存在）
        if "t" in wp:
            t_val = wp["t"]
            if not isinstance(t_val, (int, float)) or not math.isfinite(t_val):
                return False, f"第 {i} 个 waypoint 的 t 值非有限: {t_val}"

    # 检查单步位移
    for i in range(1, len(trajectory)):
        dx = trajectory[i]["x"] - trajectory[i - 1]["x"]
        dy = trajectory[i]["y"] - trajectory[i - 1]["y"]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > max_step_displacement:
            return False, (
                f"第 {i - 1}-{i} 步位移过大: {dist:.2f}m > {max_step_displacement}m"
            )

    return True, None


def compute_valid_trajectory_rate(
    results: List[Dict],
    trajectory_key: str = "trajectory",
    min_waypoints: int = 20,
    max_waypoints: int = 30,
) -> float:
    """计算轨迹有效率。

    遍历评测结果列表，统计有效轨迹的占比。

    Args:
        results: 评测结果列表，每条包含 trajectory 字段。
        trajectory_key: 轨迹数据的键名（默认 "trajectory"）。
        min_waypoints: 最少 waypoint 数。
        max_waypoints: 最多 waypoint 数。

    Returns:
        有效轨迹率（0.0 ~ 1.0）。输入为空时返回 0.0。
    """
    if not results:
        return 0.0

    valid_count = 0
    for r in results:
        traj = r
        # 支持嵌套键访问
        for key in trajectory_key.split("."):
            if isinstance(traj, dict):
                traj = traj.get(key, [])
            else:
                traj = []
                break

        if isinstance(traj, list):
            valid, _ = is_valid_trajectory(
                traj,
                min_waypoints=min_waypoints,
                max_waypoints=max_waypoints,
            )
            if valid:
                valid_count += 1

    return valid_count / len(results)


def resample_trajectory(
    trajectory: List[Dict],
    target_num: int = 25,
    target_times: Optional[List[float]] = None,
) -> List[Dict]:
    """将轨迹重采样到固定 waypoint 数（线性插值）。

    支持两种模式：
    1. 按目标 waypoint 数等间距插值（默认）。
    2. 按目标时间戳列表插值（target_times 参数）。

    使用线性插值，保留 t 和 optional_v 字段。
    起点和终点坐标保持不变。

    注意：不要外推到真值范围之外。如果 target_times 超出原始轨迹时间范围，
    只返回在原始范围内的插值点。

    Args:
        trajectory: 原始轨迹点列表（至少 2 个点）。
        target_num: 目标 waypoint 数（默认 25）。
        target_times: 目标时间戳列表（可选）。若提供，则忽略 target_num。

    Returns:
        重采样后的轨迹点列表。输入不足 2 个点时原样返回。
    """
    if not trajectory or len(trajectory) < 2:
        return trajectory

    n = len(trajectory)

    # 确保轨迹有 t 字段，没有则用索引生成
    times = []
    for i, wp in enumerate(trajectory):
        t_val = wp.get("t")
        if t_val is None:
            t_val = i * 0.1
        times.append(t_val)

    # 模式 2：按目标时间戳插值
    if target_times is not None:
        return _resample_by_times(trajectory, times, target_times)

    # 模式 1：按目标 waypoint 数等间距插值
    if n == target_num:
        return trajectory

    result = []
    for i in range(target_num):
        # 在原始轨迹中的浮点索引
        src_idx = i * (n - 1) / (target_num - 1)
        idx_low = int(src_idx)
        idx_high = min(idx_low + 1, n - 1)
        frac = src_idx - idx_low

        x = trajectory[idx_low]["x"] * (1 - frac) + trajectory[idx_high]["x"] * frac
        y = trajectory[idx_low]["y"] * (1 - frac) + trajectory[idx_high]["y"] * frac

        # t 字段插值
        t = times[idx_low] * (1 - frac) + times[idx_high] * frac

        wp = {"t": round(t, 4), "x": round(x, 4), "y": round(y, 4)}

        # optional_v 字段插值
        v_low = trajectory[idx_low].get("optional_v")
        v_high = trajectory[idx_high].get("optional_v")
        if v_low is not None and v_high is not None:
            wp["optional_v"] = round(v_low * (1 - frac) + v_high * frac, 2)

        result.append(wp)

    return result


def _resample_by_times(
    trajectory: List[Dict],
    times: List[float],
    target_times: List[float],
) -> List[Dict]:
    """按目标时间戳列表进行轨迹插值。

    仅返回在原始轨迹时间范围 [min(times), max(times)] 内的插值点。
    不外推到真值范围之外。

    Args:
        trajectory: 原始轨迹点列表。
        times: 原始轨迹时间戳列表。
        target_times: 目标时间戳列表。

    Returns:
        插值后的轨迹点列表。
    """
    t_min = times[0]
    t_max = times[-1]

    result = []
    for target_t in target_times:
        # 不外推到原始范围之外
        if target_t < t_min or target_t > t_max:
            continue

        # 找到目标时间所在的区间
        idx_low = 0
        for j in range(len(times) - 1):
            if times[j] <= target_t <= times[j + 1]:
                idx_low = j
                break
        idx_high = idx_low + 1

        if idx_high >= len(times):
            # 目标时间等于最后一个时间点
            wp = {"t": round(target_t, 4),
                  "x": round(trajectory[-1]["x"], 4),
                  "y": round(trajectory[-1]["y"], 4)}
        else:
            dt = times[idx_high] - times[idx_low]
            frac = (target_t - times[idx_low]) / dt if dt > 0 else 0.0

            x = trajectory[idx_low]["x"] * (1 - frac) + trajectory[idx_high]["x"] * frac
            y = trajectory[idx_low]["y"] * (1 - frac) + trajectory[idx_high]["y"] * frac

            wp = {"t": round(target_t, 4), "x": round(x, 4), "y": round(y, 4)}

            # optional_v 插值
            v_low = trajectory[idx_low].get("optional_v")
            v_high = trajectory[idx_high].get("optional_v")
            if v_low is not None and v_high is not None:
                wp["optional_v"] = round(v_low * (1 - frac) + v_high * frac, 2)

        result.append(wp)

    return result


def compute_behavior_accuracy(
    predicted_behaviors: List[str],
    ground_truth_behaviors: List[str],
    nav_to_behavior_map: Optional[Dict[str, str]] = None,
) -> Tuple[float, int]:
    """计算行为准确率。

    使用 nav_to_behavior_map 将导航语义伪标签映射为行为枚举后比较。
    如果没有映射表，则直接做字符串比较。
    如果真值标签为 'unknown' 或空字符串，跳过该样本。

    注意：这不是人工标注真值，只是 demo 级伪行为标签。
    README 和 eval_report.md 中必须说明 behavior_accuracy 的限制。

    Args:
        predicted_behaviors: 预测行为列表。
        ground_truth_behaviors: 真值行为列表（可能是导航语义伪标签）。
        nav_to_behavior_map: 导航语义到行为的映射字典。
            例如: {"left_turn": "TURN_LEFT", "straight": "KEEP_LANE"}

    Returns:
        (准确率, 有效样本数) 元组。输入为空时返回 (0.0, 0)。
    """
    if not predicted_behaviors or not ground_truth_behaviors:
        return 0.0, 0

    n = min(len(predicted_behaviors), len(ground_truth_behaviors))
    if n == 0:
        return 0.0, 0

    valid_count = 0
    correct = 0

    for i in range(n):
        pred = predicted_behaviors[i]
        gt = ground_truth_behaviors[i]

        # 跳过真值为 unknown 或空的样本
        if not gt or gt == "unknown" or gt == "UNKNOWN":
            continue

        # 将真值标签映射为行为枚举
        if nav_to_behavior_map and gt in nav_to_behavior_map:
            gt_mapped = nav_to_behavior_map[gt]
        else:
            gt_mapped = gt

        valid_count += 1
        if pred == gt_mapped:
            correct += 1

    if valid_count == 0:
        return 0.0, 0

    return correct / valid_count, valid_count


def _check_finite(trajectory: List[Dict], name: str) -> None:
    """检查轨迹中的坐标值是否有限（非 NaN、非 Inf）。

    Args:
        trajectory: 轨迹点列表。
        name: 轨迹名称（用于错误信息）。

    Raises:
        ValueError: 发现非有限值时。
    """
    for i, wp in enumerate(trajectory):
        for field in ("x", "y"):
            val = wp.get(field)
            if val is not None and not math.isfinite(val):
                raise ValueError(
                    f"{name} 第 {i} 个 waypoint 的 {field} 值非有限: {val}"
                )


def collision_proxy_stub(
    predicted: List[Dict],
    obstacles: Optional[List[Dict]] = None,
) -> Optional[bool]:
    """碰撞检测代理指标（预留接口，第一版不实现）。

    后续可接入 nuScenes bounding box 信息进行碰撞检测。
    需要使用 nuScenes 的 instance / sample_annotation 数据来获取障碍物信息。

    Args:
        predicted: 预测轨迹点列表。
        obstacles: 障碍物列表（含位置和尺寸信息）。

    Returns:
        None（第一版不实现）。
    """
    logger.info("collision_proxy 预留接口，第一版不实现。后续可接入 nuScenes bounding box。")
    return None


def offroad_proxy_stub(
    predicted: List[Dict],
    road_boundary: Optional[List[Dict]] = None,
) -> Optional[bool]:
    """偏离道路检测代理指标（预留接口，第一版不实现）。

    后续可接入 nuScenes 地图信息进行道路边界检测。
    需要使用 nuScenes 的 map 数据（lane / road_block / road_segment）。

    Args:
        predicted: 预测轨迹点列表。
        road_boundary: 道路边界点列表。

    Returns:
        None（第一版不实现）。
    """
    logger.info("offroad_proxy 预留接口，第一版不实现。后续可接入 nuScenes 地图信息。")
    return None


# ============================================================
# P6 新增：L2 per horizon (UniAD / VAD 标准做法)
# ============================================================


def compute_l2_at_horizon(
    predicted: List[Dict],
    ground_truth: List[Dict],
    horizon_seconds: float,
    total_horizon_seconds: float = 3.0,
) -> Optional[float]:
    """计算指定预测时刻的 L2 (欧氏) 误差。

    假设 ``predicted`` 和 ``ground_truth`` 已经重采样到相同的均匀时间网格，
    覆盖 ``[0, total_horizon_seconds]``。函数按比例换算出该时刻对应的
    数组下标，然后取两轨迹该下标点的欧氏距离。

    Args:
        predicted: 预测轨迹（重采样后的均匀网格），每个点含 x, y。
        ground_truth: 真值轨迹（重采样后的均匀网格），每个点含 x, y。
        horizon_seconds: 查询时刻（秒），如 1.0 / 2.0 / 3.0。
        total_horizon_seconds: 重采样轨迹覆盖的总时长（秒），默认 3.0。

    Returns:
        L2 误差（米）。如果该时刻超出轨迹覆盖范围或任一轨迹为空，返回 None。

    Raises:
        ValueError: 该点含 NaN / Inf。
    """
    if not predicted or not ground_truth:
        return None
    if horizon_seconds <= 0 or total_horizon_seconds <= 0:
        return None
    if horizon_seconds > total_horizon_seconds:
        return None

    n = min(len(predicted), len(ground_truth))
    if n < 2:
        return None

    # 重采样后的均匀网格：网格上有 n 个点，覆盖 [0, total_horizon_seconds]，
    # dt = total_horizon_seconds / (n - 1)。
    # idx 0 对应 t=0（当前帧），idx n-1 对应 t=total_horizon_seconds。
    dt = total_horizon_seconds / (n - 1)
    idx = int(round(horizon_seconds / dt))
    if idx <= 0 or idx >= n:
        # 边界守护：idx=0 是当前帧，无意义；超界丢弃
        return None

    px, py = predicted[idx].get("x", 0.0), predicted[idx].get("y", 0.0)
    gx, gy = ground_truth[idx].get("x", 0.0), ground_truth[idx].get("y", 0.0)
    for name, val in (("pred x", px), ("pred y", py), ("gt x", gx), ("gt y", gy)):
        if not math.isfinite(val):
            raise ValueError(f"L2@{horizon_seconds}s {name} 非有限值: {val}")

    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2)


def compute_l2_per_horizon(
    predicted: List[Dict],
    ground_truth: List[Dict],
    horizons_seconds: List[float],
    total_horizon_seconds: float = 3.0,
) -> Dict[str, Optional[float]]:
    """批量计算多个时刻的 L2 误差。

    Args:
        predicted: 预测轨迹（重采样后）。
        ground_truth: 真值轨迹（重采样后）。
        horizons_seconds: 要计算的时刻列表，如 [1.0, 2.0, 3.0]。
        total_horizon_seconds: 重采样轨迹覆盖的总时长。

    Returns:
        ``{"L2_1s": x.xx, "L2_2s": x.xx, "L2_3s": x.xx}`` 形式的字典。
        无法计算的时刻值为 None。
    """
    out: Dict[str, Optional[float]] = {}
    for h in horizons_seconds:
        # 命名：整数秒 -> "L2_3s"，浮点 -> "L2_1.5s"
        label = f"L2_{int(h)}s" if abs(h - int(h)) < 1e-6 else f"L2_{h}s"
        try:
            out[label] = compute_l2_at_horizon(
                predicted, ground_truth, h, total_horizon_seconds,
            )
        except ValueError:
            out[label] = None
    return out
