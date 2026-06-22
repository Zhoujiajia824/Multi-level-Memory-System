"""评测模块
==========
包含评测指标、评测器、报告生成器。

核心组件：
- metrics: ADE、FDE、轨迹有效性、行为准确率、轨迹重采样等指标计算。
- evaluator: 评测管理器，支持分组统计和配置加载。
- report_writer: 生成 CSV、JSONL、Markdown 三种评测报告。
"""
from src.vla_memory.evaluation.metrics import (
    compute_ade,
    compute_fde,
    compute_behavior_accuracy,
    compute_valid_trajectory_rate,
    is_valid_trajectory,
    resample_trajectory,
    collision_proxy_stub,
    offroad_proxy_stub,
)
from src.vla_memory.evaluation.evaluator import Evaluator
from src.vla_memory.evaluation.report_writer import ReportWriter
