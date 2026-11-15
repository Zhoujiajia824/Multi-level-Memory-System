"""平台无关的数据契约；距离为米，时间为秒，角度为弧度。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

VERSION = "nine-grid-v1.0"
ANSWER_PROTOCOL_VERSION = "local-goal-frenet-1.0"


@dataclass
class Actor:
    id: str
    kind: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    heading: float
    length: float
    width: float
    lane_id: str | None = None
    visible: bool = True
    history: list[dict] = field(default_factory=list)
    radius: float | None = None
    # CARLA几何中心运动学参数；None保持原highway车体模型，不把车长当真实轴距。
    wheelbase_m: float | None = None
    rear_axle_to_center_m: float | None = None


@dataclass
class Lane:
    id: str
    center: np.ndarray
    polygon: np.ndarray
    width: float
    speed_limit: float
    successors: list[str] = field(default_factory=list)
    left: str | None = None
    right: str | None = None
    priority: int = 0
    boundaries: list[dict] = field(default_factory=list)


@dataclass
class SceneSnapshot:
    frame_id: str
    time: float
    ego: Actor
    actors: list[Actor]
    lanes: list[Lane]
    route: np.ndarray
    route_lane_ids: list[str]
    drivable: Any
    scenario: str
    rules: dict = field(default_factory=dict)
    assets: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    version: str = VERSION
    coordinates: str = "world_xy_m"


@dataclass
class Representations:
    frame_id: str
    time: float
    bev_path: str
    feature_image_path: str
    occ: np.ndarray
    occ_known: np.ndarray
    grid: dict
    bev_feature: np.ndarray | None = None
    version: str = VERSION
    coordinates: str = "bev_ego_x_forward_y_left_m;grid_frenet_m"
    dino_signature: str | None = None
    occupancy_config: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    frame_id: str
    records: list[dict] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)
    snapshot_id: str = ""
    version: str = VERSION
    time: float | None = None
    coordinates: str = "query_bev_ego;memory_coordinates_are_historical"
    diagnostics: dict = field(default_factory=dict)


@dataclass
class ModelAnswer:
    frame_id: str
    time: float
    attention: list[dict]
    local_goal: dict
    explanation: dict
    world_goal: np.ndarray
    goal_reference: dict
    goal_id: str
    raw: str = ""
    source: str = "fast"
    version: str = VERSION
    protocol_version: str = ANSWER_PROTOCOL_VERSION
    coordinates: str = "local_goal:source_reference_delta_s_and_absolute_d_m;world_goal:world_xy_m"
    target_lane_id: str | None = None


@dataclass
class PredictionBundle:
    frame_id: str
    time: float
    times: np.ndarray
    trajectories: dict[str, np.ndarray]
    actors: dict[str, Actor]
    diagnostics: dict = field(default_factory=dict)
    version: str = VERSION
    coordinates: str = "world_xy_m"
    # 每条轨迹为世界坐标 [X,Y,heading,speed]，包含当前状态。


@dataclass
class PlanResult:
    frame_id: str
    time: float
    valid: bool
    states: np.ndarray
    controls: np.ndarray
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)
    version: str = VERSION
    coordinates: str = "states:world_xy_m;controls:acceleration_mps2_steering_rad"
    # states 为世界坐标 [X,Y,psi,v]；controls 为物理 [a,delta]。


@dataclass
class QualityReport:
    frame_id: str
    valid: bool
    score: float | None
    metrics: dict
    precollision: bool
    collision: bool
    reasons: list[str] = field(default_factory=list)
    version: str = VERSION
    time: float | None = None
    coordinates: str = "source_scene_world_xy_m"


@dataclass
class ReflectionTask:
    scene: SceneSnapshot
    representations: Representations
    predictions: PredictionBundle
    answer: ModelAnswer | None
    raw_answer: str
    errors: list[str]
    plan: PlanResult
    baseline: PlanResult
    quality: QualityReport
    retrieval: RetrievalResult
    frame_id: str = field(init=False)
    time: float = field(init=False)
    version: str = field(init=False)
    coordinates: str = field(init=False)

    def __post_init__(self):
        self.frame_id=self.scene.frame_id
        self.time=self.scene.time
        self.version=self.scene.version
        self.coordinates=self.scene.coordinates


@dataclass
class ExperienceRecord:
    id: str
    frame_id: str
    time: float
    version: str
    situation: dict
    answer: dict
    quality: dict
    assets: dict
    features: dict
    status: str = "active"
    revision: int = 1
    coordinates: str = "source_scene_world_xy_and_source_reference_frenet_m"
    answer_protocol_version: str = ANSWER_PROTOCOL_VERSION
    goal_reference: dict = field(default_factory=dict)


def plain(value):
    """JSON 无损保存数值与空值；几何对象单独保存 WKT。"""
    if hasattr(value, "__dataclass_fields__"):
        return plain(asdict(value))
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "wkt"):
        return value.wkt
    return value
