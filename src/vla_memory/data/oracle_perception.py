"""
Oracle 感知对象生成（nuScenes GT 投影）
=======================================
基于 nuScenes **ground-truth 3D 标注（sample_annotation）** 投影到 6 个相机图像，
生成结构化的 perception_objects 列表。

⚠️ Oracle 来源声明：本模块产出的检测框、类别、位置、速度、加速度全部来自 nuScenes
GT 标注的投影与因果差分，**不是外部检测模型或运动学模型的预测结果**。每个对象的
``is_oracle`` 字段恒为 True。

运动学因果性红线：速度/加速度仅沿 annotation 的 ``prev`` 链回溯（当前+历史），
**绝不读取 ``next``（未来帧）**；缺历史则置空并标记不可用，禁止填假值/默认 0。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.schemas.ego_state import EgoState
from src.vla_memory.schemas.perception import (
    PerceptionObject,
    KINEMATICS_SOURCE_AVAILABLE,
    KINEMATICS_SOURCE_INVALID_DT,
    KINEMATICS_SOURCE_NO_HISTORY,
    KINEMATICS_SOURCE_VELOCITY_ONLY,
)

logger = get_logger("oracle_perception")

DEFAULT_SURROUND_CAMERAS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
"""六视角环视相机默认顺序（2x3 布局：上排前向三摄、下排后向三摄）。"""


# ----------------------------------------------------------------------
# 工具：类别映射、可见性枚举
# ----------------------------------------------------------------------

def map_category(category_name: str) -> Tuple[str, str]:
    """nuScenes 点分类别 → (VALID_OBJECT_TYPES 主类别, 细粒度语义标签)。

    例：'vehicle.car.parked' → ('vehicle', 'car')；'human.pedestrian.adult' →
    ('pedestrian', 'adult')；'movable_object.barrier' → ('obstacle', 'barrier')。
    """
    parts = (category_name or "").split(".")
    semantic = parts[1] if len(parts) > 1 else (parts[0] if parts else "")
    head = parts[0] if parts else ""
    if head == "vehicle":
        if semantic in ("bicycle", "motorcycle"):
            return "cyclist", semantic
        return "vehicle", semantic or "vehicle"
    if head == "human":
        return "pedestrian", semantic or "pedestrian"
    if head in ("movable_object", "static_object"):
        return "obstacle", semantic
    return "unknown", semantic


def _parse_box_visibility(name: str):
    """字符串 → nuscenes BoxVisibility 枚举。"""
    try:
        from nuscenes.utils.geometry_utils import BoxVisibility
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "oracle_perception 需要 nuscenes-devkit（geometry_utils）。"
            f"请安装: pip install nuscenes-devkit。原因: {e}"
        )
    mapping = {"ALL": BoxVisibility.ALL, "ANY": BoxVisibility.ANY, "NONE": BoxVisibility.NONE}
    return mapping.get((name or "ANY").upper(), BoxVisibility.ANY)


# ----------------------------------------------------------------------
# 因果运动学（仅 prev，绝不 next）
# ----------------------------------------------------------------------

def estimate_kinematics_causal(
    nusc, ann: dict, ego_yaw: float
) -> Tuple[
    Optional[List[float]], Optional[float],            # velocity(ego), speed
    Optional[List[float]], Optional[float],            # acceleration(ego), accel_mag
    str, bool, bool,                                    # source, velocity_available, accel_available
]:
    """沿 annotation 的 prev 链因果差分速度/加速度（绝不触碰 next）。

    时间戳取自父 sample（annotation 自身无 timestamp）。速度需 2 帧、加速度需 3 帧。
    缺历史 → 对应字段 None + 标记不可用。**禁止填假值/默认 0。**

    Returns:
        (velocity_ego, speed, acceleration_ego, accel_mag, source,
         velocity_available, acceleration_available)
        velocity/acceleration 为 ego-centric [前向, 左向]。
    """
    cur_t = ann["translation"]
    cur_ts = nusc.get("sample", ann["sample_token"])["timestamp"]

    prev_token = ann.get("prev", "")
    if not prev_token:
        # 本帧首次出现：无历史，速度/加速度均不可用
        return None, None, None, None, KINEMATICS_SOURCE_NO_HISTORY, False, False

    prev_ann = nusc.get("sample_annotation", prev_token)
    prev_ts = nusc.get("sample", prev_ann["sample_token"])["timestamp"]
    dt = (cur_ts - prev_ts) / 1e6  # 微秒→秒
    if dt <= 0:
        return None, None, None, None, KINEMATICS_SOURCE_INVALID_DT, False, False

    # 全局速度（2 帧差分）
    gvx = (cur_t[0] - prev_ann["translation"][0]) / dt
    gvy = (cur_t[1] - prev_ann["translation"][1]) / dt

    # 全局→ego-centric 2D 旋转（与项目轨迹一致的范式）
    cos_y, sin_y = math.cos(-ego_yaw), math.sin(-ego_yaw)
    evx = gvx * cos_y - gvy * sin_y
    evy = gvx * sin_y + gvy * cos_y
    speed = math.sqrt(evx * evx + evy * evy)

    # 加速度需再回溯一帧
    pp_token = prev_ann.get("prev", "")
    if not pp_token:
        return [evx, evy], speed, None, None, KINEMATICS_SOURCE_VELOCITY_ONLY, True, False
    pp_ann = nusc.get("sample_annotation", pp_token)
    pp_ts = nusc.get("sample", pp_ann["sample_token"])["timestamp"]
    dt1 = (prev_ts - pp_ts) / 1e6
    if dt1 <= 0:
        return [evx, evy], speed, None, None, KINEMATICS_SOURCE_VELOCITY_ONLY, True, False

    pgvx = (prev_ann["translation"][0] - pp_ann["translation"][0]) / dt1
    pgvy = (prev_ann["translation"][1] - pp_ann["translation"][1]) / dt1
    gax = (gvx - pgvx) / dt
    gay = (gvy - pgvy) / dt
    eax = gax * cos_y - gay * sin_y
    eay = gax * sin_y + gay * cos_y
    amag = math.sqrt(eax * eax + eay * eay)
    return [evx, evy], speed, [eax, eay], amag, KINEMATICS_SOURCE_AVAILABLE, True, True


# ----------------------------------------------------------------------
# 主入口：生成 perception_objects
# ----------------------------------------------------------------------

def get_perception_objects(
    nusc,
    sample_token: str,
    ego_pose: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> List[PerceptionObject]:
    """为指定 sample 生成 oracle perception_objects 列表（nuScenes GT 投影）。

    流程：
      1. 遍历 sample 的所有 sample_annotation，算 ego 位置/距离/朝向/因果运动学；
      2. 按 max_distance_m 过滤；
      3. 对 6 相机逐个投影，给每个对象挂 boxes_2d 与 visible_cameras；
      4. 仅保留至少在一个相机可见的对象（喂给相机 mosaic VLM），按距离排序。

    Args:
        nusc: nuscenes.nuscenes.NuScenes 实例。
        sample_token: 当前帧 sample token。
        ego_pose: 当前帧 ego pose dict（含 translation/rotation），来自 adapter._get_sample_ego_pose。
        cfg: oracle 配置 dict（max_distance_m / box_visibility / cameras）。

    Returns:
        PerceptionObject 列表（每个 is_oracle=True）。
    """
    cfg = cfg or {}
    max_dist = float(cfg.get("max_distance_m", 50.0))
    box_vis = _parse_box_visibility(cfg.get("box_visibility", "ANY"))
    cameras = cfg.get("cameras") or DEFAULT_SURROUND_CAMERAS

    sample = nusc.get("sample", sample_token)

    # 当前帧 ego 位姿
    ego_t = ego_pose["translation"]
    ego_yaw = EgoState.quat_to_yaw(ego_pose["rotation"])
    cos_y, sin_y = math.cos(-ego_yaw), math.sin(-ego_yaw)

    # ---- 步骤 1+2：基础信息 + 距离过滤 ----
    objects: Dict[str, PerceptionObject] = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        ann_t = ann["translation"]

        # 全局→ego-centric 位置（2D 旋转）
        dx = ann_t[0] - ego_t[0]
        dy = ann_t[1] - ego_t[1]
        ex = dx * cos_y - dy * sin_y
        ey = dx * sin_y + dy * cos_y
        dist = math.hypot(ex, ey)
        if dist > max_dist:
            continue

        # 类别（优先 annotation 冗余字段，回退 instance 表）
        category_name = ann.get("category_name") or ""
        if not category_name:
            try:
                inst = nusc.get("instance", ann["instance_token"])
                category_name = inst.get("category_name", "")
            except Exception:  # pragma: no cover
                category_name = ""
        category, semantic = map_category(category_name)

        # 属性
        attrs: List[str] = []
        for a_tok in ann.get("attribute_tokens", []):
            try:
                attrs.append(nusc.get("attribute", a_tok)["name"])
            except Exception:  # pragma: no cover
                pass

        # 可见度 level
        vis_level = ""
        try:
            vis_level = nusc.get("visibility", ann.get("visibility_token", ""))["level"]
        except Exception:  # pragma: no cover
            pass

        ann_yaw = EgoState.quat_to_yaw(ann["rotation"])
        vel, speed, acc, amag, ksrc, vok, aok = estimate_kinematics_causal(nusc, ann, ego_yaw)

        objects[ann_token] = PerceptionObject(
            annotation_token=ann_token,
            instance_token=ann.get("instance_token", ""),
            category=category,
            category_name_raw=category_name,
            semantic_label=semantic,
            attributes=attrs,
            size=list(ann.get("size", [])),
            position_global=list(ann_t),
            position_ego=[round(ex, 4), round(ey, 4)],
            distance_to_ego=round(dist, 4),
            heading_global=ann_yaw,
            heading_ego=ann_yaw - ego_yaw,
            velocity=vel,
            speed=speed,
            acceleration=acc,
            acceleration_mag=amag,
            velocity_available=vok,
            acceleration_available=aok,
            kinematics_source=ksrc,
            num_lidar_pts=int(ann.get("num_lidar_pts", 0)),
            visibility_level=vis_level,
        )

    # ---- 步骤 3：6 相机投影，挂 boxes_2d + visible_cameras ----
    try:
        from nuscenes.utils.geometry_utils import view_points
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "oracle_perception 投影需要 nuscenes-devkit（geometry_utils.view_points）。"
            f"请安装: pip install nuscenes-devkit。原因: {e}"
        )

    for cam in cameras:
        sd_token = sample["data"].get(cam)
        if sd_token is None:
            continue
        try:
            _, box_list, intrinsic = nusc.get_sample_data(sd_token, box_vis_level=box_vis)
        except Exception as e:  # pragma: no cover
            logger.warning("get_sample_data 投影失败 cam=%s: %s", cam, e)
            continue
        if not box_list:
            continue
        # corners 已在相机坐标系下；view = K @ [I | 0]
        view = intrinsic.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
        for box in box_list:
            obj = objects.get(box.token)
            if obj is None:
                continue  # 被距离过滤掉
            corners = box.corners()  # (3, 8)
            uv = view_points(corners, view, normalize=True)
            x1, y1 = float(uv[0].min()), float(uv[1].min())
            x2, y2 = float(uv[0].max()), float(uv[1].max())
            obj.boxes_2d[cam] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            if cam not in obj.visible_cameras:
                obj.visible_cameras.append(cam)

    # ---- 步骤 4：仅保留至少在一个相机可见的对象，按距离排序 ----
    visible_objs = [o for o in objects.values() if o.visible_cameras]
    visible_objs.sort(key=lambda o: o.distance_to_ego)
    logger.info(
        "oracle perception: sample=%s 候选=%d 距离过滤后=%d 相机可见=%d",
        sample_token[:8], len(sample["anns"]), len(objects), len(visible_objs),
    )
    return visible_objs
