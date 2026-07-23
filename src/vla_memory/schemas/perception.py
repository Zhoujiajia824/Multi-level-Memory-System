"""
Oracle 感知对象数据模型
======================
定义 PerceptionObject 数据结构：基于 nuScenes **ground-truth 3D 标注**投影到
相机图像后得到的结构化感知对象。

重要说明（oracle 来源）
----------------------
PerceptionObject 中的检测框、类别、位置、速度、加速度等信息**全部来自 nuScenes
GT 标注（sample_annotation）的投影与因果差分**，**不是外部检测模型或运动学模型
的预测结果**。字段 ``is_oracle`` 恒为 True 以显式标注。

运动学因果性
-----------
速度 / 加速度严格满足在线因果性：仅使用当前帧与历史帧（沿 annotation 的 ``prev``
链回溯），**绝不读取 ``next``（未来帧）**。当某目标缺少足够历史（首次出现）时，
``velocity`` / ``acceleration`` 置空（None），``velocity_available`` /
``acceleration_available`` 置 False，``kinematics_source`` 标记为不可用原因。
**禁止为追求字段完整而填假速度、假加速度或默认 0。**

坐标系
------
- ``position_global``：全局坐标系 [x, y, z]，米。
- ``position_ego`` / ``velocity`` / ``acceleration``：ego-centric 坐标系，x 前向、
  y 左向（与项目轨迹一致）。其中 velocity/acceleration 是**目标自身**速度/加速度
  旋转到 ego 轴向（非相对 ego 运动），``velocity_frame`` 标注坐标系。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# 速度/加速度运动学来源标记（用于审计与下游感知）
KINEMATICS_SOURCE_AVAILABLE = "annotation_keyframe_diff_2hz"
"""速度+加速度均由 annotation prev 链因果差分得到（2Hz 关键帧）。"""

KINEMATICS_SOURCE_VELOCITY_ONLY = "annotation_keyframe_diff_2hz_velocity_only"
"""仅有速度（2 帧），加速度因缺少更深历史而不可用。"""

KINEMATICS_SOURCE_NO_HISTORY = "unavailable_no_history"
"""目标在本帧首次出现（无 prev），速度/加速度均不可用。"""

KINEMATICS_SOURCE_INVALID_DT = "unavailable_invalid_dt"
"""相邻帧时间差非正，速度/加速度不可用。"""


class PerceptionObject(BaseModel):
    """单个 oracle 感知对象（nuScenes GT 投影）。

    所有字段均派生自 nuScenes sample_annotation 的 ground-truth 标注，
    ``is_oracle`` 恒为 True。
    """

    # ---- 标识 ----
    annotation_token: str = Field(..., description="本帧 sample_annotation 唯一 token")
    instance_token: str = Field(..., description="目标实例 token（跨帧稳定 ID）")

    # ---- 语义 ----
    category: str = Field(
        "unknown",
        description="映射到 VALID_OBJECT_TYPES 的类别（vehicle/pedestrian/cyclist/obstacle/...）",
    )
    category_name_raw: str = Field(
        "", description='nuScenes 原始点分类别名，如 "vehicle.car.parked" / "human.pedestrian.adult"'
    )
    semantic_label: str = Field(
        "", description="语义标签（car/truck/pedestrian/barrier/cone/... 等细粒度语义）"
    )
    attributes: List[str] = Field(
        default_factory=list,
        description='nuScenes 属性名列表，如 ["vehicle.moving", "vehicle.parked"]',
    )

    # ---- 几何 ----
    size: List[float] = Field(
        default_factory=list, description="3D 尺寸 [w, l, h]（米，nuScenes wlh 约定）"
    )
    position_global: List[float] = Field(
        default_factory=list, description="全局坐标系中心 [x, y, z]（米）"
    )
    position_ego: List[float] = Field(
        default_factory=list,
        description="ego-centric 坐标系位置 [x(前向), y(左向)]（米，已舍 z）",
    )
    distance_to_ego: float = Field(0.0, description="到 ego 的平面距离（米）")
    heading_global: float = Field(0.0, description="全局朝向角（弧度，[-pi, pi]）")
    heading_ego: float = Field(0.0, description="相对 ego 的朝向角（弧度）")

    # ---- 多相机投影 ----
    visible_cameras: List[str] = Field(
        default_factory=list, description="该目标可见的相机名列表（投影落在图像内）"
    )
    boxes_2d: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="各相机图像上的 2D 投影框 {cam_name: [x1, y1, x2, y2]}（像素）",
    )

    # ---- 运动学（因果：仅当前+历史）----
    velocity: Optional[List[float]] = Field(
        None,
        description="ego-centric 速度 [vx(前向), vy(左向)]（m/s）；无历史时为 None",
    )
    velocity_frame: str = Field(
        "ego", description="速度坐标系：ego（ego-centric）或 global"
    )
    speed: Optional[float] = Field(None, description="速度大小（m/s）；不可用时 None")
    acceleration: Optional[List[float]] = Field(
        None,
        description="ego-centric 加速度 [ax(前向), ay(左向)]（m/s^2）；无足够历史时 None",
    )
    acceleration_mag: Optional[float] = Field(
        None, description="加速度大小（m/s^2）；不可用时 None"
    )
    velocity_available: bool = Field(
        False, description="速度是否可用（False=缺历史，velocity/speed 为 None）"
    )
    acceleration_available: bool = Field(
        False, description="加速度是否可用（False=缺更深历史，acceleration 为 None）"
    )
    kinematics_source: str = Field(
        KINEMATICS_SOURCE_NO_HISTORY,
        description="运动学来源标记（见模块常量；oracle 差分 / 不可用原因）",
    )

    # ---- 标注质量 / 可见度 ----
    num_lidar_pts: int = Field(0, description="盒内 LiDAR 点数（质量参考）")
    visibility_level: str = Field(
        "", description='nuScenes visibility level，如 "v3"（80-100% 可见）'
    )

    # ---- oracle 显式标注 ----
    is_oracle: bool = Field(
        True,
        description="恒为 True：本对象来自 nuScenes GT 投影，非检测模型预测",
    )

    def to_serializable_dict(self) -> dict:
        """转换为普通 dict（pydantic v1/v2 兼容），用于 jsonl 序列化与 prompt 渲染。"""
        try:
            return self.model_dump()  # pydantic v2
        except AttributeError:  # pragma: no cover
            return self.dict()  # pydantic v1
