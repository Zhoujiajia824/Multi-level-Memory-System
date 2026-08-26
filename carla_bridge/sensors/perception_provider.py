"""CARLA 服务端 GT -> PerceptionObject
=====================================
用 ``world.get_actors()`` 拿服务端真值（位置/速度/bbox/类别），映射成项目
``PerceptionObject``（``is_oracle=True``）。零检测误差，等价于 nuScenes oracle。

运动学因果性：速度/加速度用 ``actor.get_velocity()`` / ``get_acceleration()``
直读 CARLA 当前真值（不读未来），首次出现的目标速度即可用（优于 nuScenes prev
链差分）。过滤 ego 自身。位置/速度/加速度一律走 ``coords`` 转 ego-centric
（y=左），与决策轨迹手性一致。
"""
from __future__ import annotations

import math
from typing import List

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.schemas.perception import PerceptionObject
from carla_bridge.state import coords

logger = get_logger("carla_perception")


def _map_category(type_id: str) -> tuple:
    """CARLA type_id -> (项目 category, semantic)。与 VALID_OBJECT_TYPES 对齐。"""
    t = type_id or ""
    if t.startswith("vehicle."):
        if "bicycle" in t or "motorcycle" in t:
            return "cyclist", t.split(".")[-1]
        return "vehicle", (t.split(".")[-1] or "vehicle")
    if t.startswith("walker."):
        return "pedestrian", "pedestrian"
    if t.startswith("traffic.traffic_light"):
        return "traffic_light", t
    # 其它静态/可移动物体当障碍
    return "obstacle", t


_TL_STATE_LABEL = {
    "Red": "traffic_light_red",
    "Yellow": "traffic_light_yellow",
    "Green": "traffic_light_green",
}


class PerceptionProvider:
    """CARLA GT actors -> PerceptionObject 列表。"""

    def __init__(self, world, max_distance_m: float = 50.0):
        self.world = world
        self.max_distance_m = max_distance_m

    def get_objects(
        self, ego_vehicle, ego_x: float, ego_y: float, ego_yaw_rad: float
    ) -> List[dict]:
        """返回当前帧 perception_objects（dict 列表），按距离升序。

        Args:
            ego_vehicle: 自车 actor（用于排除自身）。
            ego_x, ego_y: 自车全局位置（CARLA 系）。
            ego_yaw_rad: 自车朝向（弧度，项目系）。

        Returns:
            ``PerceptionObject.model_dump()`` 列表，与 nuScenes oracle 路径同构。
        """
        ego_id = ego_vehicle.id
        actors = self.world.get_actors()
        objs: List[PerceptionObject] = []

        for actor in actors:
            if actor.id == ego_id:
                continue
            type_id = getattr(actor, "type_id", "")
            if not (
                type_id.startswith("vehicle.")
                or type_id.startswith("walker.")
                or type_id.startswith("traffic.traffic_light")
            ):
                continue

            loc = actor.get_location()
            fwd, left = coords.global_to_ego(loc.x, loc.y, ego_x, ego_y, ego_yaw_rad)
            dist = math.hypot(fwd, left)
            if dist > self.max_distance_m:
                continue
            # 交通灯：只保留 ego 前向带内的灯（后方/交叉远处灯是噪声），并读出红绿
            # 状态写入 semantic_label——决策 prompt 只渲染 semantic_label，塞别处 VLM 看不到
            semantic = None
            if type_id.startswith("traffic.traffic_light"):
                if fwd < -2.0 or abs(left) > 15.0:
                    continue
                try:
                    state_name = str(actor.state).split(".")[-1]
                except Exception:
                    state_name = ""
                semantic = _TL_STATE_LABEL.get(state_name, "traffic_light_unknown")

            category, cat_semantic = _map_category(type_id)
            semantic = semantic or cat_semantic

            # 速度/加速度（车/行人/骑行者直读；交通灯无运动学）
            vel = None
            speed = None
            acc = None
            amag = None
            vok = False
            aok = False
            if category in ("vehicle", "pedestrian", "cyclist"):
                v = actor.get_velocity()
                evx, evy = coords.rotate_vector_to_ego(v.x, v.y, ego_yaw_rad)
                speed = math.hypot(evx, evy)
                vel = [round(evx, 4), round(evy, 4)]
                vok = True
                a = actor.get_acceleration()
                eax, eay = coords.rotate_vector_to_ego(a.x, a.y, ego_yaw_rad)
                amag = math.hypot(eax, eay)
                acc = [round(eax, 4), round(eay, 4)]
                aok = True

            # 朝向
            try:
                heading_rad = coords.carla_yaw_deg_to_rad(
                    actor.get_transform().rotation.yaw
                )
            except Exception:
                heading_rad = 0.0
            heading_ego = coords.heading_ego(heading_rad, ego_yaw_rad)

            # bbox 尺寸 [w, l, h]（extent 是半长）
            size: List[float] = []
            try:
                ext = actor.bounding_box.extent
                size = [round(2 * ext.y, 3), round(2 * ext.x, 3), round(2 * ext.z, 3)]
            except Exception:
                pass

            aid = str(actor.id)
            objs.append(PerceptionObject(
                annotation_token=aid,
                instance_token=aid,
                category=category,
                category_name_raw=type_id,
                semantic_label=semantic,
                size=size,
                position_global=[round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)],
                position_ego=[round(fwd, 4), round(left, 4)],
                distance_to_ego=round(dist, 4),
                heading_global=heading_rad,
                heading_ego=heading_ego,
                velocity=vel,
                speed=speed,
                acceleration=acc,
                acceleration_mag=amag,
                velocity_available=vok,
                acceleration_available=aok,
                kinematics_source="carla_gt_direct",
            ))

        objs.sort(key=lambda o: o.distance_to_ego)
        logger.info(
            "CARLA GT perception: 全部actor=%d, 在%.0fm内可见=%d",
            len(actors), self.max_distance_m, len(objs),
        )
        return [o.to_serializable_dict() for o in objs]
