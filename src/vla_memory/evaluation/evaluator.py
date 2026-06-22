"""评测器模块
============
执行 memory_on 和 memory_off 两种模式的评测。
从 config/evaluation.yaml 读取参数，支持分组统计和行为映射。
包含行为分组、中位数统计、fallback 统计等完整评测功能。

P6 升级：
- 修复行为准确率永远 0 的 bug：默认提供 ``nav_to_behavior_map``，
  比较前统一大写归一化。
- 新增 L2 per horizon（1s / 2s / 3s）评测，结果同时写入 EvalSampleResult
  和 EvalSummary（按 horizon 求均值）。
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.vla_memory.evaluation.metrics import (
    compute_ade,
    compute_behavior_accuracy,
    compute_fde,
    compute_l2_per_horizon,
    is_valid_trajectory,
    resample_trajectory,
)
from src.vla_memory.schemas.evaluation import EvalSampleResult, EvalSummary
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("evaluator")


# ============================================================
# 默认配置（与 config/evaluation.yaml 保持同步）
# ============================================================

# RouteInfer 输出的导航语义 -> DecisionOutput 的行为枚举
_DEFAULT_NAV_TO_BEHAVIOR_MAP: Dict[str, str] = {
    "straight": "KEEP_LANE",
    "left_turn": "TURN_LEFT",
    "right_turn": "TURN_RIGHT",
    "lane_follow": "KEEP_LANE",
    "lane_change_left": "CHANGE_LANE_LEFT",
    "lane_change_right": "CHANGE_LANE_RIGHT",
    "slow_or_stop": "SLOW_DOWN",
    "unknown": "UNKNOWN",
}

_DEFAULT_L2_HORIZONS_S = [1.0, 2.0, 3.0]


def _normalize_behavior(b: Any) -> str:
    """归一化行为字符串：转大写 + 去空白。空/None 返回空串。"""
    if not b:
        return ""
    return str(b).strip().upper()


class Evaluator:
    """评测管理器。

    对决策输出进行评测，支持 memory_on 和 memory_off 两种模式。
    支持从配置字典加载参数。

    评测流程：
    1. 检查轨迹有效性（waypoint 数、字段完整性、单步位移）。
    2. 重采样到统一 waypoint 数。
    3. 计算 ADE / FDE。
    4. 通过 nav_to_behavior_map 映射后比较行为准确率。
    5. 按 scene_id、weather_id、behavior 分组统计。

    Args:
        resample_num: 重采样 waypoint 数。
        min_waypoints: 轨迹有效性最少 waypoint 数。
        max_waypoints: 轨迹有效性最多 waypoint 数。
        max_step_displacement: 单步最大位移（米）。
        nav_to_behavior_map: 导航语义到行为枚举的映射字典。
        prediction_horizon_seconds: 未来轨迹预测时间窗（秒）。
    """

    def __init__(
        self,
        resample_num: int = 25,
        min_waypoints: int = 20,
        max_waypoints: int = 30,
        max_step_displacement: float = 5.0,
        nav_to_behavior_map: Optional[Dict[str, str]] = None,
        prediction_horizon_seconds: float = 3.0,
        normalize_behavior_case: bool = True,
        l2_horizons_seconds: Optional[List[float]] = None,
    ):
        self.resample_num = resample_num
        self.min_waypoints = min_waypoints
        self.max_waypoints = max_waypoints
        self.max_step_displacement = max_step_displacement
        # P6 bug fix：未传 map 时用内置默认 map（而非空字典），
        # 否则 "straight" → "KEEP_LANE" 的映射永远不发生 → 行为准确率恒为 0。
        self.nav_to_behavior_map = (
            nav_to_behavior_map if nav_to_behavior_map is not None
            else dict(_DEFAULT_NAV_TO_BEHAVIOR_MAP)
        )
        self.prediction_horizon_seconds = prediction_horizon_seconds
        self.normalize_behavior_case = normalize_behavior_case
        self.l2_horizons_seconds = (
            list(l2_horizons_seconds) if l2_horizons_seconds
            else list(_DEFAULT_L2_HORIZONS_S)
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Evaluator":
        """从配置字典创建 Evaluator 实例。

        从 evaluation.yaml 的各个子配置中提取参数。

        Args:
            config: 评测配置字典（来自 config/evaluation.yaml）。

        Returns:
            Evaluator 实例。
        """
        displacement_cfg = config.get("displacement_metrics", {}) or {}
        validity_cfg = config.get("validity", {}) or {}
        behavior_cfg = config.get("behavior_accuracy", {}) or {}
        l2_cfg = config.get("l2_per_horizon", {}) or {}

        nav_map = behavior_cfg.get("nav_to_behavior_map")
        # config 给空 dict 时也走默认，避免重蹈"空 map → 永远 0"覆辙
        if not nav_map:
            nav_map = None

        return cls(
            resample_num=displacement_cfg.get("resample_num", 25),
            min_waypoints=validity_cfg.get("min_waypoints", 20),
            max_waypoints=validity_cfg.get("max_waypoints", 30),
            max_step_displacement=validity_cfg.get("max_displacement_per_step", 5.0),
            nav_to_behavior_map=nav_map,
            prediction_horizon_seconds=config.get("prediction_horizon_seconds", 3.0),
            normalize_behavior_case=behavior_cfg.get("normalize_case", True),
            l2_horizons_seconds=l2_cfg.get("horizons_seconds"),
        )

    def evaluate_sample(
        self,
        predicted_trajectory: List[Dict],
        ground_truth_trajectory: List[Dict],
        predicted_behavior: str = "",
        ground_truth_behavior: str = "",
        sample_token: str = "",
        scene_token: str = "",
        mode: str = "memory_on",
        scene_id: str = "",
        weather_id: str = "",
        fallback_used: bool = False,
    ) -> EvalSampleResult:
        """评测单个样本。

        流程：
        1. 检查轨迹有效性（waypoint 数、字段完整性、单步位移）。
        2. 重采样到统一 waypoint 数。
        3. 计算 ADE / FDE。
        4. 通过 nav_to_behavior_map 映射后比较行为准确率。

        如果真值轨迹为空或不足，跳过该样本并记录 warning。
        不允许伪造真值轨迹。

        Args:
            predicted_trajectory: 预测轨迹。
            ground_truth_trajectory: 真值轨迹（来自 nuScenes ego pose 派生）。
            predicted_behavior: 预测行为。
            ground_truth_behavior: 真值行为（可能是导航语义伪标签）。
            sample_token: 样本 token。
            scene_token: 场景 token。
            mode: 评测模式。
            scene_id: 场景类型。
            weather_id: 天气类型。
            fallback_used: 是否使用了规则 fallback。

        Returns:
            单样本评测结果 EvalSampleResult。
        """
        result = EvalSampleResult(
            sample_token=sample_token,
            scene_token=scene_token,
            mode=mode,
            scene_id=scene_id,
            weather_id=weather_id,
            predicted_behavior=predicted_behavior,
            ground_truth_behavior=ground_truth_behavior,
            behavior=predicted_behavior,
            fallback_used=fallback_used,
        )

        # ---- 检查真值轨迹 ----
        if not ground_truth_trajectory:
            result.ade = None
            result.fde = None
            result.is_valid_trajectory = False
            result.valid_error = "真值轨迹为空，无法评测"
            result.error_message = "真值轨迹为空"
            logger.debug(f"样本 {sample_token}: 真值轨迹为空，跳过")
            return result

        # ---- 1. 检查预测轨迹有效性 ----
        valid, valid_error = is_valid_trajectory(
            predicted_trajectory,
            min_waypoints=self.min_waypoints,
            max_waypoints=self.max_waypoints,
            max_step_displacement=self.max_step_displacement,
        )
        result.is_valid_trajectory = valid
        result.valid_error = valid_error

        if not valid or not predicted_trajectory:
            result.ade = None
            result.fde = None
            result.error_message = f"预测轨迹无效: {valid_error}"
            logger.debug(f"样本 {sample_token}: 预测轨迹无效 ({valid_error})")
            return result

        # ---- 2. 重采样 ----
        pred_resampled = resample_trajectory(
            predicted_trajectory, self.resample_num,
        )
        gt_resampled = resample_trajectory(
            ground_truth_trajectory, self.resample_num,
        )

        # 检查重采样后真值是否有效
        if not gt_resampled or len(gt_resampled) < 2:
            result.ade = None
            result.fde = None
            result.error_message = "真值轨迹重采样后点数不足"
            logger.debug(f"样本 {sample_token}: 真值轨迹重采样后点数不足")
            return result

        # ---- 3. 计算 ADE / FDE ----
        result.ade = compute_ade(pred_resampled, gt_resampled)
        result.fde = compute_fde(pred_resampled, gt_resampled)

        # ---- 3.1 P6 新增：L2 per horizon ----
        # 重采样后的两条轨迹覆盖 [0, prediction_horizon_seconds]，
        # compute_l2_per_horizon 内部按比例换算 idx。
        l2_dict = compute_l2_per_horizon(
            pred_resampled, gt_resampled,
            horizons_seconds=self.l2_horizons_seconds,
            total_horizon_seconds=self.prediction_horizon_seconds,
        )
        result.l2_per_horizon = l2_dict

        # ---- 4. 行为准确率（P6 bug fix） ----
        # 关键修复：
        # 1) 即使调用方没传 nav_to_behavior_map，__init__ 也会用 _DEFAULT_NAV_TO_BEHAVIOR_MAP。
        # 2) lookup 用 lower-case 的 ground_truth_behavior，因为 RouteInfer 输出小写。
        # 3) 比较前对 pred 和 mapped gt 统一大写归一化（VLM 偶尔输出小写）。
        if predicted_behavior and ground_truth_behavior:
            if ground_truth_behavior not in ("unknown", "UNKNOWN", ""):
                gt_lookup = str(ground_truth_behavior).strip().lower()
                gt_mapped = self.nav_to_behavior_map.get(gt_lookup, ground_truth_behavior)
                if self.normalize_behavior_case:
                    result.behavior_correct = (
                        _normalize_behavior(predicted_behavior)
                        == _normalize_behavior(gt_mapped)
                    )
                else:
                    result.behavior_correct = predicted_behavior == gt_mapped

        return result

    def aggregate_results(self, results: List[EvalSampleResult]) -> EvalSummary:
        """汇总评测结果。

        包含 ADE/FDE 均值、标准差、中位数、有效轨迹率、行为准确率、
        按 scene_id、weather_id、behavior 分组统计。

        Args:
            results: 所有样本的评测结果列表。

        Returns:
            汇总评测结果 EvalSummary。
        """
        if not results:
            return EvalSummary(mode="unknown")

        mode = results[0].mode

        # 过滤有效结果（ADE 非空且非 inf）
        valid_results = [
            r for r in results
            if r.ade is not None and r.ade != float("inf")
        ]

        # 计算 ADE / FDE 均值、标准差、中位数（过滤 inf）
        ade_values = [r.ade for r in valid_results if r.ade is not None]
        fde_values = [r.fde for r in valid_results if r.fde is not None]

        summary = EvalSummary(
            mode=mode,
            total_samples=len(results),
            valid_samples=len(valid_results),
            valid_trajectory_rate=(
                sum(1 for r in results if r.is_valid_trajectory) / len(results)
                if results else 0.0
            ),
            fallback_count=sum(1 for r in results if r.fallback_used),
        )

        if ade_values:
            summary.ade_mean = statistics.mean(ade_values)
            summary.ade_std = (
                statistics.stdev(ade_values) if len(ade_values) > 1 else 0.0
            )
            summary.ade_median = statistics.median(ade_values)
        if fde_values:
            summary.fde_mean = statistics.mean(fde_values)
            summary.fde_std = (
                statistics.stdev(fde_values) if len(fde_values) > 1 else 0.0
            )
            summary.fde_median = statistics.median(fde_values)

        # ---- P6 新增：L2 per horizon 均值 ----
        # 收集所有有效样本里出现过的 horizon 标签（如 L2_1s/L2_2s/L2_3s），
        # 各自计算均值（忽略 None）。
        l2_aggregate: Dict[str, List[float]] = {}
        for r in valid_results:
            if not r.l2_per_horizon:
                continue
            for k, v in r.l2_per_horizon.items():
                if v is not None and not (isinstance(v, float) and v != v):
                    l2_aggregate.setdefault(k, []).append(float(v))
        if l2_aggregate:
            summary.l2_mean_per_horizon = {
                k: statistics.mean(vs) for k, vs in l2_aggregate.items()
            }

        # 行为准确率
        behavior_results = [
            r for r in results
            if r.behavior_correct is not None and r.ground_truth_behavior not in ("", "unknown", "UNKNOWN")
        ]
        if behavior_results:
            correct = sum(1 for r in behavior_results if r.behavior_correct)
            summary.behavior_accuracy = correct / len(behavior_results)
            summary.behavior_valid_count = len(behavior_results)

        # 分组统计
        self._compute_grouped_stats(results, summary)

        return summary

    def _compute_grouped_stats(
        self,
        results: List[EvalSampleResult],
        summary: EvalSummary,
    ) -> None:
        """计算分组统计（scene_id、weather_id、behavior）。

        每个分组计算：样本数、ADE 均值、FDE 均值、轨迹有效率。

        Args:
            results: 评测结果列表。
            summary: 汇总对象（就地更新）。
        """
        scene_groups: Dict[str, list] = {}
        weather_groups: Dict[str, list] = {}
        behavior_groups: Dict[str, list] = {}

        for r in results:
            if r.scene_id:
                scene_groups.setdefault(r.scene_id, []).append(r)
            if r.weather_id:
                weather_groups.setdefault(r.weather_id, []).append(r)
            if r.behavior:
                behavior_groups.setdefault(r.behavior, []).append(r)

        for sid, group in scene_groups.items():
            summary.scene_grouped[sid] = self._group_stats(group)

        for wid, group in weather_groups.items():
            summary.weather_grouped[wid] = self._group_stats(group)

        for bid, group in behavior_groups.items():
            summary.behavior_grouped[bid] = self._group_stats(group)

    @staticmethod
    def _group_stats(group: List[EvalSampleResult]) -> Dict[str, Any]:
        """计算单个分组的统计信息。

        Args:
            group: 该分组的评测结果列表。

        Returns:
            统计信息字典。
        """
        ade_vals = [
            r.ade for r in group
            if r.ade is not None and r.ade != float("inf")
        ]
        fde_vals = [
            r.fde for r in group
            if r.fde is not None and r.fde != float("inf")
        ]

        return {
            "count": len(group),
            "ade_mean": statistics.mean(ade_vals) if ade_vals else None,
            "fde_mean": statistics.mean(fde_vals) if fde_vals else None,
            "valid_rate": (
                sum(1 for r in group if r.is_valid_trajectory) / len(group)
            ),
        }
