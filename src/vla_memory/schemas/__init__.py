"""
数据模型模块
============
定义全链路统一数据结构，包含帧元数据、自车状态、轨迹、
场景理解、记忆记录、决策输出、评测结果等核心数据模型。
所有枚举字段使用严格校验。
"""
from src.vla_memory.schemas.frame import FrameMeta
from src.vla_memory.schemas.ego_state import EgoState
from src.vla_memory.schemas.trajectory import TrajectoryPoint, Trajectory
from src.vla_memory.schemas.scene import (
    SurroundingObject,
    SceneUnderstandingResult,
    VALID_SCENE_IDS,
    VALID_WEATHER_IDS,
    VALID_TRAFFIC_DENSITIES,
    VALID_OBJECT_TYPES,
    VALID_RELATIVE_POSITIONS,
)
from src.vla_memory.schemas.memory import (
    MemoryRecord,
    MidTermMemoryRecord,
    ShortTermMemoryItem,
    LongTermRule,
)
from src.vla_memory.schemas.decision import DecisionOutput, VALID_BEHAVIORS, VALID_RISK_LEVELS
from src.vla_memory.schemas.evaluation import EvaluationResult, EvalSampleResult, EvalSummary

__all__ = [
    # 帧
    "FrameMeta",
    # 自车状态
    "EgoState",
    # 轨迹
    "TrajectoryPoint",
    "Trajectory",
    # 场景理解
    "SurroundingObject",
    "SceneUnderstandingResult",
    "VALID_SCENE_IDS",
    "VALID_WEATHER_IDS",
    "VALID_TRAFFIC_DENSITIES",
    "VALID_OBJECT_TYPES",
    "VALID_RELATIVE_POSITIONS",
    # 记忆
    "MemoryRecord",
    "MidTermMemoryRecord",
    "ShortTermMemoryItem",
    "LongTermRule",
    # 决策
    "DecisionOutput",
    "VALID_BEHAVIORS",
    "VALID_RISK_LEVELS",
    # 评测
    "EvaluationResult",
    "EvalSampleResult",
    "EvalSummary",
]
