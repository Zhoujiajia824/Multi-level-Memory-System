"""
评测数据模型
============
定义评测结果的统计数据结构。
包含单样本评测结果（EvalSampleResult）和汇总评测结果（EvalSummary）。
支持 memory_on / memory_off 两种模式的对比评测。
"""
from __future__ import annotations

from typing import Optional, Dict

from pydantic import BaseModel, Field, validator


class EvaluationResult(BaseModel):
    """单样本评测结果数据模型（基础版）。

    Attributes:
        sample_id: 样本唯一标识（通常为 sample_token）。
        memory_mode: 评测模式，'memory_on' 或 'memory_off'。
        ade: Average Displacement Error（米）。
        fde: Final Displacement Error（米）。
        behavior_correct: 行为预测是否正确，None 表示无法判断。
        valid_trajectory: 轨迹是否有效。
    """
    sample_id: str = Field(..., description="样本唯一标识")
    memory_mode: str = Field(..., description="评测模式: memory_on / memory_off")
    ade: Optional[float] = Field(None, description="Average Displacement Error（米）")
    fde: Optional[float] = Field(None, description="Final Displacement Error（米）")
    behavior_correct: Optional[bool] = Field(None, description="行为预测是否正确")
    valid_trajectory: bool = Field(True, description="轨迹是否有效")

    @validator("memory_mode")
    def _validate_mode(cls, v):
        """校验评测模式。"""
        if v not in ("memory_on", "memory_off"):
            raise ValueError(f"无效的评测模式: '{v}'，有效值为: memory_on / memory_off")
        return v


class EvalSampleResult(BaseModel):
    """单样本评测详情（用于 JSONL 输出）。

    包含完整的评测信息，用于 detail 输出和后续分析。

    Attributes:
        sample_token: nuScenes sample token。
        scene_token: nuScenes scene token。
        mode: 评测模式。
        ade: ADE 值。
        fde: FDE 值。
        is_valid_trajectory: 轨迹有效性。
        valid_error: 轨迹无效时的错误原因。
        predicted_behavior: 预测行为。
        ground_truth_behavior: 真值行为（来自伪标签）。
        behavior_correct: 行为正确性。
        scene_id: 场景类型。
        weather_id: 天气类型。
        behavior: 预测行为（用于分组统计）。
        fallback_used: 是否使用了规则 fallback。
        error_message: 错误信息。
    """
    sample_token: str = Field("", description="sample token")
    scene_token: str = Field("", description="scene token")
    mode: str = Field(..., description="评测模式")
    ade: Optional[float] = Field(None, description="ADE 值")
    fde: Optional[float] = Field(None, description="FDE 值")
    # P6 新增：每个 horizon 的 L2 误差，如 {"L2_1s": 1.23, "L2_2s": 2.34, "L2_3s": None}
    l2_per_horizon: Optional[Dict[str, Optional[float]]] = Field(
        None, description="L2@horizon 误差字典",
    )
    is_valid_trajectory: bool = Field(True, description="轨迹有效性")
    valid_error: Optional[str] = Field(None, description="轨迹无效时的错误原因")
    predicted_behavior: str = Field("", description="预测行为")
    ground_truth_behavior: str = Field("", description="真值行为（伪标签）")
    behavior_correct: Optional[bool] = Field(None, description="行为正确性")
    scene_id: str = Field("", description="场景类型")
    weather_id: str = Field("", description="天气类型")
    behavior: str = Field("", description="预测行为（用于分组统计）")
    fallback_used: bool = Field(False, description="是否使用了规则 fallback")
    error_message: Optional[str] = Field(None, description="错误信息")


class EvalSummary(BaseModel):
    """评测汇总结果数据模型。

    包含总体指标和分组统计信息。

    Attributes:
        mode: 评测模式。
        total_samples: 总样本数。
        valid_samples: 有效样本数。
        ade_mean: ADE 均值。
        ade_std: ADE 标准差。
        ade_median: ADE 中位数。
        fde_mean: FDE 均值。
        fde_std: FDE 标准差。
        fde_median: FDE 中位数。
        valid_trajectory_rate: 轨迹有效率。
        behavior_accuracy: 行为准确率。
        behavior_valid_count: 行为准确率有效样本数。
        fallback_count: 使用 fallback 的样本数。
        scene_grouped: 按场景分组的统计。
        weather_grouped: 按天气分组的统计。
        behavior_grouped: 按行为分组的统计。
    """
    mode: str = Field(..., description="评测模式")
    total_samples: int = Field(0, description="总样本数")
    valid_samples: int = Field(0, description="有效样本数")
    ade_mean: Optional[float] = Field(None, description="ADE 均值")
    ade_std: Optional[float] = Field(None, description="ADE 标准差")
    ade_median: Optional[float] = Field(None, description="ADE 中位数")
    fde_mean: Optional[float] = Field(None, description="FDE 均值")
    fde_std: Optional[float] = Field(None, description="FDE 标准差")
    fde_median: Optional[float] = Field(None, description="FDE 中位数")
    # P6 新增：L2@horizon 均值字典，如 {"L2_1s": 1.10, "L2_2s": 2.20, "L2_3s": 3.30}
    l2_mean_per_horizon: Optional[Dict[str, float]] = Field(
        None, description="L2@horizon 均值字典",
    )
    valid_trajectory_rate: float = Field(0.0, description="轨迹有效率")
    behavior_accuracy: Optional[float] = Field(None, description="行为准确率")
    behavior_valid_count: int = Field(0, description="行为准确率有效样本数")
    fallback_count: int = Field(0, description="使用 fallback 的样本数")
    scene_grouped: Dict[str, dict] = Field(default_factory=dict, description="场景分组统计")
    weather_grouped: Dict[str, dict] = Field(default_factory=dict, description="天气分组统计")
    behavior_grouped: Dict[str, dict] = Field(default_factory=dict, description="行为分组统计")
