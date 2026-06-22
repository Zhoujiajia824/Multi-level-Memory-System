"""流水线模块
============
完整 demo 的流水线步骤。R2 重构后：旧的 batch 版 scene/memory/decision
pipeline 已删除，替换为 ``OnlineDrivingLoop`` 逐帧在线循环。
"""
from src.vla_memory.pipeline.prepare_nuscenes import run_prepare_nuscenes
from src.vla_memory.pipeline.online_loop import OnlineDrivingLoop, default_output_path
from src.vla_memory.pipeline.eval_pipeline import run_eval_pipeline
from src.vla_memory.pipeline.full_demo_pipeline import (
    run_full_demo,
    enrich_keyframes_with_state,
)

__all__ = [
    "run_prepare_nuscenes",
    "OnlineDrivingLoop",
    "default_output_path",
    "run_eval_pipeline",
    "run_full_demo",
    "enrich_keyframes_with_state",
]
