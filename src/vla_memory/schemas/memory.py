"""
记忆数据模型
============
定义中期记忆记录 MemoryRecord 和长期规则 LongTermRule 的数据结构。
MemoryRecord 是全链路核心数据结构之一，贯穿场景理解、记忆构建、检索和决策。
"""
from __future__ import annotations

from typing import Optional, List, Any

from pydantic import BaseModel, Field

from src.vla_memory.schemas.scene import VALID_SCENE_IDS, VALID_WEATHER_IDS


class MemoryRecord(BaseModel):
    """中期记忆记录数据模型。

    存储历史关键帧的完整经验信息，用于 FAISS 检索和多维度联合评分。
    每条记录对应一个关键帧从感知到决策的完整闭环。

    Attributes:
        record_id: 记录唯一标识，通常为 sample_token。
        frame_meta: 关键帧元信息字典。
        image_feature_path: DINOv2 图像特征向量文件路径。
        scene_text: 场景描述文本，用于文本相似度检索。
        scene_id: 场景类型标识，必须是 VALID_SCENE_IDS 之一。
        weather_id: 天气类型标识，必须是 VALID_WEATHER_IDS 之一。
        nav_instruction: 导航语义指令。
        ego_state: 自车状态字典。
        history_trajectory: 历史 ego-centric 轨迹点列表。
        decision_reason: VLM 决策原因摘要。
        behavior: 行为决策。
        trajectory: 决策输出的轨迹点列表。
    """
    record_id: str = Field(..., description="记录唯一标识")
    frame_meta: Optional[dict] = Field(None, description="关键帧元信息字典")
    image_feature_path: Optional[str] = Field(None, description="图像特征向量文件路径")
    scene_text: Optional[str] = Field(None, description="场景描述文本")
    scene_id: Optional[str] = Field(None, description="场景类型标识")
    weather_id: Optional[str] = Field(None, description="天气类型标识")
    nav_instruction: Optional[str] = Field(None, description="导航语义指令")
    ego_state: Optional[dict] = Field(None, description="自车状态字典")
    history_trajectory: Optional[List[dict]] = Field(None, description="历史轨迹点列表")
    decision_reason: Optional[str] = Field(None, description="决策原因摘要")
    behavior: Optional[str] = Field(None, description="行为决策")
    trajectory: Optional[List[dict]] = Field(None, description="决策输出轨迹")


class ShortTermMemoryItem(BaseModel):
    """短期记忆项数据模型。

    存储最近 N 个关键帧的摘要数据，使用 deque 滑动窗口管理。

    Attributes:
        frame_id: 帧唯一标识。
        timestamp: 时间戳（微秒）。
        image_path: 图像文件路径。
        image_feature_path: 特征向量文件路径。
        scene_description: 场景描述。
        scene_id: 场景类型。
        weather_id: 天气类型。
        nav_instruction: 导航语义。
        ego_state: 自车状态字典。
        history_trajectory: 历史轨迹。
        scene_understanding_result: 场景理解完整结果。
    """
    frame_id: str = Field(..., description="帧唯一标识")
    timestamp: int = Field(0, description="时间戳（微秒）")
    image_path: str = Field("", description="图像文件路径")
    image_feature_path: Optional[str] = Field(None, description="特征向量文件路径")
    scene_description: Optional[str] = Field(None, description="场景描述")
    scene_id: Optional[str] = Field(None, description="场景类型")
    weather_id: Optional[str] = Field(None, description="天气类型")
    nav_instruction: Optional[str] = Field(None, description="导航语义")
    ego_state: Optional[dict] = Field(None, description="自车状态字典")
    history_trajectory: Optional[list] = Field(None, description="历史轨迹")
    scene_understanding_result: Optional[dict] = Field(None, description="场景理解结果")


class LongTermRule(BaseModel):
    """长期记忆规则数据模型。

    存储在 YAML 文件中的驾驶规则和常识。

    Attributes:
        rule_id: 规则唯一标识。
        scene_id: 适用的场景类型，'all' 表示所有场景。
        weather_id: 适用的天气类型，'all' 表示所有天气。
        title: 规则标题。
        content: 规则内容。
        priority: 优先级，1 最高，5 最低。
    """
    rule_id: str = Field(..., description="规则唯一标识")
    scene_id: str = Field("all", description="适用场景类型，'all' 表示所有")
    weather_id: str = Field("all", description="适用天气类型，'all' 表示所有")
    title: str = Field("", description="规则标题")
    content: str = Field("", description="规则内容")
    priority: int = Field(5, description="优先级（1最高，5最低）")

    @classmethod
    def from_yaml_dict(cls, data: dict) -> "LongTermRule":
        """从 YAML 加载的字典创建规则实例。"""
        return cls(**data)


# ===================== 类型别名 =====================
# 多个模块使用 MidTermMemoryRecord 名称进行 import，
# 实际类名是 MemoryRecord。添加别名保持兼容性。
MidTermMemoryRecord = MemoryRecord
