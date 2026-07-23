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

    # ===================== Phase 1 新增：metadata 字段（37 个）=====================
    # 向后兼容：所有新字段都有默认值；旧 mid_term_meta.json（12 字段）加载时经
    # MidTermMemoryRecord(**data) 的 Pydantic 解析自动补默认，不报错。
    # Phase 1 不做价值门控/事件识别/淘汰，故价值分数=None、标签=空、统计=0、is_active=True。
    # 禁止设 model_config=ConfigDict(extra="forbid")，否则会破坏旧 JSON 加载。

    # ---- 1. 基础状态 ----
    memory_id: str = Field("", description="记忆唯一标识（= record_id，规范字段名）")
    memory_type: str = Field("frame_memory", description="记忆类型（frame_memory / episode / ...）")
    status: str = Field("active", description="记忆状态（active / archived / deleted）")
    version: str = Field("v0.2", description="写入时的 metadata schema 版本")
    created_at: int = Field(0, description="创建时间（帧时间戳，μs；用事件时间便于复现与排序）")
    updated_at: int = Field(0, description="最后更新时间（μs）；Phase 1 无更新，= created_at")

    # ---- 2. 来源 ----
    source_dataset: str = Field("", description="来源数据集/项目名")
    source_version: str = Field("", description="来源数据集版本（如 v1.0-trainval）")
    source_scene_token: str = Field("", description="来源场景 token")
    source_scene_name: str = Field("", description="来源场景名（如 scene-0001）")
    source_sample_token: str = Field("", description="来源 sample token")
    source_frame_id: str = Field("", description="来源帧标识（= sample_token）")
    source_mode: str = Field("", description="写入时的运行模式（memory_on / memory_off）")

    # ---- 3. 视觉输入 ----
    visual_input_type: str = Field("", description="视觉输入类型（single_front / surround_mosaic）")
    image_path: str = Field("", description="主感知图像路径（surround 模式下为 mosaic 路径）")
    feature_path: str = Field("", description="视觉特征文件路径（规范名，与 image_feature_path 同值）")
    feature_dim: int = Field(0, description="视觉特征维度（如 768）")

    # ---- 4. 场景标签（Phase 1 默认/空；后续阶段由事件识别填充，不伪造）----
    event_type: str = Field("frame_memory", description="事件类型（Phase 1 默认 frame_memory；后续由事件识别填）")
    scene_tags: List[str] = Field(default_factory=list, description="场景标签列表（Phase 1 空）")
    risk_tags: List[str] = Field(default_factory=list, description="风险标签列表（Phase 1 空）")

    # ---- 5. 写入价值（Phase 1 无门控 → 标 legacy/default）----
    admission_score: float = Field(1.0, description="准入分数（Phase 1 legacy=1.0，逐帧全存）")
    admission_reasons: List[str] = Field(default_factory=list, description="准入原因（Phase 1 = [legacy_no_gating]）")
    admission_policy_version: str = Field("legacy", description="准入策略版本（Phase 1 = legacy，无门控）")

    # ---- 6. 记忆价值（Phase 1 不计算 → None，绝不伪造分数）----
    memory_value_score: Optional[float] = Field(None, description="综合记忆价值分（后续阶段计算）")
    salience_score: Optional[float] = Field(None, description="显著性分（后续阶段计算）")
    rarity_score: Optional[float] = Field(None, description="稀有度分（后续阶段计算）")
    confidence_score: Optional[float] = Field(None, description="置信度分（后续阶段计算）")
    redundancy_score: Optional[float] = Field(None, description="冗余度分（后续阶段计算）")
    retrieval_utility: Optional[float] = Field(None, description="检索效用分（后续阶段计算）")
    # Phase 3 新增：recency_score（近期性分，淘汰时按 last_retrieved_at/created_at 快照计算）
    recency_score: Optional[float] = Field(None, description="近期性分（淘汰时快照计算，0~1，越近越高）")

    # ---- 7. 使用统计 ----
    hit_count: int = Field(0, description="累计被检索命中次数")
    successful_hit_count: int = Field(0, description="命中且被采纳/成功的次数")
    failed_hit_count: int = Field(0, description="命中但失败的次数")
    last_retrieved_at: Optional[int] = Field(None, description="最近一次被检索命中的时间（μs）")

    # ---- 8. 更新与删除（soft delete）----
    conflict_count: int = Field(0, description="与其他记忆冲突次数")
    superseded_by: Optional[str] = Field(None, description="被哪条记忆取代（record_id）")
    deleted_reason: Optional[str] = Field(None, description="删除原因（淘汰 / 冲突 / 手动）")
    # Phase 3 新增：deleted_at（soft delete 时间戳，μs；None=未删除）
    deleted_at: Optional[int] = Field(None, description="soft delete 时间戳（μs）；None=未删除")
    is_active: bool = Field(True, description="是否有效（False=逻辑删除/soft delete，待 rebuild；Phase 3 淘汰用）")

    # ===================== Phase 5 新增：事件级记忆（event_memory）=====================
    # 连续高价值帧合并为一个事件，只存 start/peak/end 关键帧 + 结构化摘要。
    # frame_memory 记录这些字段留默认空值（向后兼容）。memory_type 区分 event_memory/frame_memory。
    event_id: str = Field("", description="事件唯一标识（event_memory 专属；frame_memory 为空）")
    event_start_sample_token: str = Field("", description="事件起始帧 sample_token")
    event_peak_sample_token: str = Field("", description="事件高潮帧 sample_token（admission 最高）")
    event_end_sample_token: str = Field("", description="事件结束帧 sample_token")
    anchor_sample_token: str = Field("", description="锚点帧（= peak，代表该事件的检索锚）")
    key_sample_tokens: List[str] = Field(default_factory=list, description="关键帧 sample_token 列表（start/peak/end 去重）")
    anchor_image_path: str = Field("", description="锚点帧图像路径（surround 模式下为 mosaic 路径）")
    key_image_paths: List[str] = Field(default_factory=list, description="关键帧图像路径列表")
    ego_summary: str = Field("", description="事件自车状态摘要（速度/加速度趋势，确定性模板生成）")
    perception_summary: str = Field("", description="事件感知摘要（对象/行人/路口计数与距离）")
    decision_summary: str = Field("", description="事件决策摘要（behavior 序列）")
    admission_summary: str = Field("", description="事件准入摘要（事件类型 + peak admission + 帧数）")
    usage: dict = Field(default_factory=dict, description="事件级使用统计（如 {hits, last_retrieved_at}；与帧级 hit_count 互补）")

    # ===================== Phase 6 新增：冲突感知更新 =====================
    # 冲突检测后软更新（降权/标记 deprecated/superseded/增 conflict_count），不物理删除。
    # status 取值：active / low_confidence / deprecated / superseded / inactive / deleted。
    last_conflict_at: Optional[int] = Field(None, description="最近一次冲突时间（μs）")
    conflict_reasons: List[str] = Field(default_factory=list, description="冲突原因列表（policy_conflict/style_variant/...）")
    previous_versions: List[str] = Field(default_factory=list, description="该记忆取代的旧 memory_id 列表（版本链）")
    update_history: List[dict] = Field(default_factory=list, description="更新历史 [{action, conflict_type, reason, at, by_new}]")


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
