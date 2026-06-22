"""
场景理解数据模型
================
定义 VLM 场景理解输出的数据结构 SceneUnderstandingResult。
包含场景描述、自车状态文本、周围物体、车道描述、交通密度、
风险因素、场景类型、天气类型。
所有 scene_id、weather_id、traffic_density 使用严格校验枚举。

P4 起：额外提供按类别分桶的结构化字段（lanes / vehicles / pedestrians /
traffic_lights / intersections），以便决策模型直接取用。
旧字段 surrounding_objects / lane_description 仍保留以保证向后兼容。
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator


# ===================== 有效枚举值常量 =====================

VALID_SCENE_IDS = [
    "intersection", "dead_end", "lane_change", "car_following",
    "obstacle_avoidance", "straight_road", "turning", "merge",
    "crosswalk", "unknown",
]
"""有效的场景类型标识列表。"""

VALID_WEATHER_IDS = ["sunny", "rainy", "snowy", "foggy", "night", "cloudy", "unknown"]
"""有效的天气类型标识列表。"""

VALID_TRAFFIC_DENSITIES = ["low", "medium", "high", "unknown"]
"""有效的交通密度枚举列表。"""

VALID_OBJECT_TYPES = ["vehicle", "pedestrian", "cyclist", "obstacle", "traffic_light", "unknown"]
"""有效的周围物体类型枚举列表。"""

VALID_RELATIVE_POSITIONS = [
    "front", "front_left", "front_right", "left", "right", "rear", "unknown",
]
"""有效的相对位置枚举列表。"""

# ===================== P4 新增：结构化子模型枚举 =====================

VALID_LANE_SIDES = ["ego", "left", "right", "oncoming", "unknown"]
"""车道相对位置：自车所在 / 左侧 / 右侧 / 对向 / 未知。"""

VALID_LANE_TYPES = ["solid", "dashed", "double", "merge", "exit", "unknown"]
"""车道线类型：实线 / 虚线 / 双线 / 汇入 / 出口 / 未知。"""

VALID_TRAFFIC_LIGHT_STATES = ["red", "yellow", "green", "off", "unknown"]
"""信号灯状态。"""

VALID_VEHICLE_MOTIONS = ["stationary", "approaching", "receding", "crossing", "unknown"]
"""他车运动状态相对自车。"""

VALID_PEDESTRIAN_INTENTS = ["crossing", "standing", "walking_along", "unknown"]
"""行人意图。"""

VALID_INTERSECTION_TYPES = ["four_way", "three_way_T", "three_way_Y", "roundabout", "unknown"]
"""路口类型。"""


# ===================== 结构化子模型（P4 新增）=====================


class LaneInfo(BaseModel):
    """单条车道信息。"""
    side: str = Field("unknown", description="车道相对位置")
    type: str = Field("unknown", description="车道线类型")
    color: Optional[str] = Field(None, description="车道线颜色: white / yellow / unknown")
    direction: Optional[str] = Field(None, description="行驶方向: forward / left / right / unknown")

    @validator("side")
    def _validate_side(cls, v):
        return v if v in VALID_LANE_SIDES else "unknown"

    @validator("type")
    def _validate_type(cls, v):
        return v if v in VALID_LANE_TYPES else "unknown"


class VehicleInfo(BaseModel):
    """周围车辆。"""
    relative_position: str = Field("unknown", description="相对自车位置")
    distance_m: Optional[float] = Field(None, description="估计距离（米）")
    type: Optional[str] = Field(None, description="车辆类型: car/truck/bus/motorcycle/unknown")
    motion: Optional[str] = Field(None, description="运动状态")

    @validator("relative_position")
    def _validate_position(cls, v):
        return v if v in VALID_RELATIVE_POSITIONS else "unknown"

    @validator("motion")
    def _validate_motion(cls, v):
        if v is None:
            return None
        return v if v in VALID_VEHICLE_MOTIONS else "unknown"


class PedestrianInfo(BaseModel):
    """周围行人。"""
    relative_position: str = Field("unknown", description="相对自车位置")
    distance_m: Optional[float] = Field(None, description="估计距离（米）")
    intent: Optional[str] = Field(None, description="行人意图")

    @validator("relative_position")
    def _validate_position(cls, v):
        return v if v in VALID_RELATIVE_POSITIONS else "unknown"

    @validator("intent")
    def _validate_intent(cls, v):
        if v is None:
            return None
        return v if v in VALID_PEDESTRIAN_INTENTS else "unknown"


class TrafficLightInfo(BaseModel):
    """信号灯。"""
    state: str = Field("unknown", description="信号灯当前状态")
    relative_position: str = Field("unknown", description="相对自车位置")
    controls_ego_lane: Optional[bool] = Field(None, description="是否管控自车车道")

    @validator("state")
    def _validate_state(cls, v):
        return v if v in VALID_TRAFFIC_LIGHT_STATES else "unknown"

    @validator("relative_position")
    def _validate_position(cls, v):
        return v if v in VALID_RELATIVE_POSITIONS else "unknown"


class IntersectionInfo(BaseModel):
    """路口信息。"""
    present: bool = Field(False, description="视野内是否有路口")
    type: Optional[str] = Field(None, description="路口类型")
    distance_m: Optional[float] = Field(None, description="距路口估计距离（米）")
    has_stop_sign: Optional[bool] = Field(None, description="是否有停止标志")

    @validator("type")
    def _validate_type(cls, v):
        if v is None:
            return None
        return v if v in VALID_INTERSECTION_TYPES else "unknown"


# ===================== 旧 SurroundingObject（保留以兼容） =====================


class SurroundingObject(BaseModel):
    """周围物体数据模型（旧字段，混合所有类型）。

    P4 起推荐使用 vehicles / pedestrians / traffic_lights 等分桶字段；
    本类保留以保证旧代码和旧 prompt 仍可工作。
    """
    type: str = Field("unknown", description="物体类型")
    relative_position: str = Field("unknown", description="相对位置")
    description: str = Field("", description="物体描述")

    @validator("type")
    def _validate_type(cls, v):
        """校验物体类型。"""
        if v not in VALID_OBJECT_TYPES:
            return "unknown"
        return v

    @validator("relative_position")
    def _validate_position(cls, v):
        """校验相对位置。"""
        if v not in VALID_RELATIVE_POSITIONS:
            return "unknown"
        return v


class SceneUnderstandingResult(BaseModel):
    """VLM 场景理解输出数据模型。

    对每个关键帧前视角图片调用 VLM 后输出的结构化结果。
    所有枚举字段（scene_id、weather_id、traffic_density）使用严格校验。

    P4 起额外提供按类别分桶的结构化字段（lanes / vehicles / pedestrians /
    traffic_lights / intersections）。旧字段 surrounding_objects /
    lane_description 仍保留以保证向后兼容；建议下游优先用新字段。
    """
    # ---- 旧/通用字段 ----
    scene_description: str = Field("", description="驾驶场景结构化描述")
    ego_status_text: str = Field("", description="自车状态自然语言描述")
    surrounding_objects: List[SurroundingObject] = Field(
        default_factory=list, description="周围物体列表（兼容旧字段）",
    )
    lane_description: str = Field("", description="车道线、道路边界描述（兼容旧字段）")
    traffic_density: str = Field("unknown", description="交通密度")
    risk_factors: List[str] = Field(default_factory=list, description="潜在风险列表")
    scene_id: str = Field("unknown", description="场景类型标识")
    weather_id: str = Field("unknown", description="天气类型标识")
    raw_response: Optional[str] = Field(None, description="VLM 原始响应文本")

    # ---- P4 新增结构化字段 ----
    lanes: List[LaneInfo] = Field(default_factory=list, description="车道线分类")
    vehicles: List[VehicleInfo] = Field(default_factory=list, description="周围车辆")
    pedestrians: List[PedestrianInfo] = Field(default_factory=list, description="周围行人")
    traffic_lights: List[TrafficLightInfo] = Field(default_factory=list, description="信号灯")
    intersections: IntersectionInfo = Field(
        default_factory=lambda: IntersectionInfo(present=False),
        description="路口信息",
    )

    @validator("scene_id")
    def _validate_scene_id(cls, v):
        """校验场景类型。"""
        if v not in VALID_SCENE_IDS:
            return "unknown"
        return v

    @validator("weather_id")
    def _validate_weather_id(cls, v):
        """校验天气类型。"""
        if v not in VALID_WEATHER_IDS:
            return "unknown"
        return v

    @validator("traffic_density")
    def _validate_traffic_density(cls, v):
        """校验交通密度。"""
        if v not in VALID_TRAFFIC_DENSITIES:
            return "unknown"
        return v
