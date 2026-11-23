"""Pure, display-only adapter for a checksummed offline ``scene.json``.

The caller authenticates the record, evidence and file. This adapter reads no
files and imports no driving engine. It renders archived current XY geometry;
route, observed history and later evidence are never a planned trajectory.
"""
from __future__ import annotations

from copy import deepcopy
import math
from statistics import median

from shapely.errors import ShapelyError

from . import geometry
from .library_roads import build_library_roads


_ROLES = {"left_front", "front", "right_front", "left_side", "ego", "right_side",
          "left_rear", "rear", "right_rear"}


def _point(actor):
    if not isinstance(actor, dict):
        return None
    value = actor.get("position")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        xy = [float(value[0]), float(value[1])]
        return xy if all(math.isfinite(number) for number in xy) else None
    except (TypeError, ValueError):
        return None


def _matches(grid, actors, ego_id):
    """Only decode already assigned slots; never assign an object to a region."""
    result = []
    if not isinstance(grid, dict) or grid.get("available") is False:
        return result
    slots = grid.get("slots")
    if isinstance(slots, dict):
        slots = [{**slot, "role": role} for role, slot in slots.items() if isinstance(slot, dict)]
    elif isinstance(slots, list):
        slots = [slot for row in slots for slot in (row if isinstance(row, list) else [row])]
    else:
        return result
    for slot in slots:
        if not isinstance(slot, dict) or slot.get("role") not in _ROLES:
            continue
        identity = slot.get("id")
        occupied = slot.get("state") == "occupied"
        identity = str(identity) if occupied and identity is not None else None
        actor = actors.get(identity)
        result.append({"role": slot["role"], "state": slot.get("state"), "id": identity,
            "kind": slot.get("kind"), "applicable": slot.get("applicable"),
            "state_valid": slot.get("state_valid"), "actor_available": actor is not None,
            "is_ego": identity == ego_id if identity is not None else False,
            "position": _point(actor), "source": "archived_grid_slot"})
    return result


def _recorded_future(frame, future, road, ground_z):
    unavailable = {"available": False, "kind": "recorded_future_execution",
        "reason": "未提供通过归档校验的未来实驶监督；不以导航参考或历史轨迹补造"}
    empty_ribbon = {"available": False, "polygons": [], "left": [], "right": []}
    if not isinstance(future, dict) or future.get("available") is not True:
        if isinstance(future, dict) and future.get("reason"):
            unavailable["reason"] = future["reason"]
        return [], [], empty_ribbon, unavailable
    if (future.get("sha256_verified") is not True or future.get("evidence_frame_id") != frame["frame_id"] or
            geometry._number(future.get("evidence_time")) is None or
            abs(float(future["evidence_time"]) - frame["time"]) > 1e-5):
        return [], [], empty_ribbon, {**unavailable, "reason": "未来实驶监督未通过所选证据帧/时间绑定"}
    rows = future.get("ego_trajectory")
    if not isinstance(rows, list):
        return [], [], empty_ribbon, unavailable
    samples = []
    for row in rows:
        time_s = geometry._number(row.get("time_s")) if isinstance(row, dict) else None
        position = _point(row)
        samples.append(None if time_s is None or position is None else {"time": time_s, "position": position})
    anchor = next((i for i, row in enumerate(samples) if row is not None and abs(row["time"] - frame["time"]) <= 1e-4), None)
    if anchor is None or math.dist(samples[anchor]["position"], _point(frame["ego"])) > .05:
        return [], [], empty_ribbon, {**unavailable, "reason": "当前证据没有同一时刻/位置的真实监督采样，未插值或平移轨迹"}
    intervals = [b["time"] - a["time"] for a, b in zip(samples, samples[1:])
                 if a is not None and b is not None and b["time"] > a["time"]]
    typical_interval = median(intervals) if intervals else None
    chosen, truncated = [], None
    for row in samples[anchor:]:
        if row is None:
            truncated = "missing_or_invalid_sample"; break
        if chosen:
            delta = row["time"] - chosen[-1]["time"]
            if delta <= 0:
                truncated = "non_monotonic_time"; break
            if typical_interval is not None and delta > typical_interval * 1.5 + 1e-5:
                truncated = "sampling_gap"; break
        chosen.append(row)
    source = str(future.get("source", ""))
    title = "记录的未来人类实驶轨迹（离线监督）" if "human" in source else "记录的未来已执行轨迹（离线监督）"
    provenance = {"available": len(chosen) >= 2, "kind": "recorded_future_execution", "title": title,
        "source": future.get("source_path", "audit/supervision.json"), "actual_source": source,
        "anchor_frame_id": frame["frame_id"], "time": frame["time"],
        "record_anchor_frame_id": future.get("anchor_frame_id"), "record_anchor_time": future.get("anchor_time"),
        "recorded_start_time": samples[0]["time"] if samples and samples[0] else None,
        "recorded_end_time": samples[-1]["time"] if samples and samples[-1] else None,
        "clip_start_time": chosen[0]["time"] if chosen else None,
        "clip_end_time": chosen[-1]["time"] if chosen else None, "point_count": len(chosen),
        "sample_times": [row["time"] for row in chosen], "typical_interval_s": typical_interval,
        "truncated_reason": truncated, "reference_point": "matches_archived_scene_ego_reference",
        "display_only": True, "online_input": False, "sha256_verified": True,
        "sampling_note": "连接真实离散采样供离线审阅；未重算规划、未把后续观测插值成连续驾驶，缺样处停止连接"}
    if len(chosen) < 2:
        provenance["reason"] = "当前证据之后不足两个真实采样点，不绘制未来轨迹或走廊"
        return [], [], empty_ribbon, provenance
    trajectory = [[*row["position"], ground_z] for row in chosen]
    ribbon = geometry._ribbon(frame, trajectory, ground_z, road)
    ribbon.update(source="recorded_future_execution_schematic_ribbon",
        note="记录实驶路径的展示几何带，半宽=存档车宽/2，无额外余量；按原存档路面裁剪，不是规划安全走廊或安全认证")
    return trajectory, [trajectory], ribbon, provenance


def build_library_scene(scene, grid=None, engine_root=None, future=None):
    """Adapt one authenticated archived snapshot to the episode renderer schema.

    ``engine_root`` is accepted for API symmetry but intentionally unused:
    offline browsing never reruns lane matching, a planner or potential theory.
    Missing height is explicitly illustrative 2.5D, and missing region polygons,
    trajectories, active semantic answers or camera calibration remain missing.
    """
    missing = {"available": False, "status": "unavailable", "read_only": True,
               "snapshot_only": True, "frame_id": None, "time": None}
    if not isinstance(scene, dict):
        return {**missing, "reason": "归档scene.json不是场景对象"}
    identity = scene.get("frame_id")
    try:
        time_s = float(scene.get("time"))
    except (TypeError, ValueError):
        time_s = math.nan
    missing.update(frame_id=identity, time=time_s if math.isfinite(time_s) else None)
    if not isinstance(identity, str) or not identity or not math.isfinite(time_s):
        return {**missing, "reason": "归档场景缺少有效frame_id或time"}
    if _point(scene.get("ego")) is None:
        return {**missing, "reason": "归档场景缺少有效自车世界坐标"}
    if not isinstance(scene.get("actors"), list):
        return {**missing, "reason": "归档场景未保存当前actors列表，不能当作对象为空"}

    archived = deepcopy(scene)
    actors = archived["actors"]
    warnings = []
    valid_actors = [row for row in actors if _point(row) is not None and isinstance(row.get("id"), str)]
    if len(valid_actors) != len(actors):
        warnings.append("部分归档对象缺少有效ID/世界坐标；未绘制这些对象，但保留原始actors列表")
    frame = {"available": True, "status": "available", "read_only": True,
        "snapshot_only": True, "frame_id": identity, "time": time_s,
        "ego": archived["ego"], "actors": actors, "scene": archived,
        "map": {}, "rules": archived.get("rules") or {}, "plan": None,
        "model_input": None, "model_output": None, "retrieval": None,
        "cameras": [], "camera_overlays": {}, "warnings": warnings}
    # Reuse pure geometry only. enrich_frame also rebuilds live lane chains,
    # which is inappropriate when the archive contains slot matches but no
    # saved spatial region polygons.
    objects = geometry._objects({**frame, "actors": valid_actors})
    if len(objects) < len(valid_actors):
        warnings.append("部分对象缺少可用的存档长度/宽度；未用默认实体尺寸补造对象框")
    z, ground_quality = geometry._ground_height(frame)
    try:
        render, recorded_road = build_library_roads(archived, z)
        roads = []  # The normalized local render geometry is the display map.
    except (ValueError, TypeError, KeyError, AttributeError, ShapelyError) as exc:
        render = {"road_surface": {"polygons": [], "recorded_surface_only": True}, "markings": []}
        roads, recorded_road = [], None
        warnings.append("归档道路几何不可显示：" + str(exc))

    ego_id = str(archived["ego"].get("id", "ego"))
    actor_map = {str(row.get("id")): row for row in valid_actors}
    actor_map[ego_id] = archived["ego"]
    matches = _matches(grid, actor_map, ego_id)
    role_map = {}
    for match in matches:
        if match["id"] is not None:
            role_map.setdefault(match["id"], []).append(match["role"])
    for obj in objects:
        obj["grid_roles"] = role_map.get(obj["id"], [])
        obj["grid_matched"] = bool(obj["grid_roles"])
        # Slot matching is not evidence of MPC selection or APF participation.
        obj["selected"], obj["potential_source"], obj["attention"] = False, False, False
        obj["potential_source_note"] = "离线scene未保存规划势场源诊断，不能由八向匹配推定"
    if any(match["id"] is not None and not match["actor_available"] for match in matches):
        warnings.append("部分八向匹配ID不在本证据scene的当前对象中；只保留匹配标签，不用张量相对值补造世界对象")
    illustrative = sum(obj["geometry_quality"] == "illustrative_2_5d" for obj in objects)
    if illustrative:
        warnings.append("对象世界高度/z未保存；框高为2.5D示意，不是真实三维标注或相机投影")
    source_diag = archived.get("diagnostics")
    source_diag = source_diag if isinstance(source_diag, dict) else {}
    if source_diag.get("lane_map_complete") is False:
        warnings.append("存档声明局部车道地图不完整；路面与标线仅按已有地图几何展示")
    trajectory, trajectory_segments, ribbon, trajectory_provenance = _recorded_future(frame, future, recorded_road, z)
    if trajectory_provenance.get("truncated_reason"):
        warnings.append("未来实驶记录存在缺样或时间断档，展示线已停止在断档之前")

    capabilities = {
        "road_surface": {"available": bool(render["road_surface"].get("polygons")),
                         "source": render["road_surface"].get("source", "recorded_scene_geometry"),
                         "reason": None if render["road_surface"].get("polygons") else "没有可用的归档路面几何"},
        "road_markings": {"available": bool(render["markings"]), "count": len(render["markings"]),
                          "note": "仅归档中可识别的物理标线；未知/未保存的不补画"},
        "current_objects": {"available": True, "archived_count": len(actors), "rendered_count": len(objects),
                            "measured_3d_count": len(objects) - illustrative, "illustrative_2_5d_count": illustrative},
        "grid_object_matches": {"available": bool(matches), "source": "archived_grid_slot"},
        "region_polygons": {"available": False, "reason": "存档八向匹配未保存空间区域多边形；未重新运行对象匹配"},
        "planned_trajectory": {"available": False, "reason": "离线经验未保存该帧规划轨迹；route与history不作为规划轨迹"},
        "recorded_future_trajectory": trajectory_provenance,
        "potential_field": {"available": False, "reason": "未保存该帧规划势场诊断或实际参与对象"},
        "semantic_goal_attention": {"available": False, "reason": "经验文本不等于该帧实际模型规划输出"},
        "camera_projection": {"available": False, "reason": "scene.json没有本查看器可验证的相机标定与三维几何；不套用CARLA投影"}}
    visualization = {"objects": objects, "roads": roads, "render_geometry": render,
        "regions": [], "region_status": capabilities["region_polygons"], "grid_matches": matches,
        "trajectory": trajectory, "trajectory_segments": trajectory_segments,
        "trajectory_provenance": trajectory_provenance,
        "trajectory_states": [], "corridor": [part["outer"] for part in ribbon["polygons"]], "goal": None,
        "corridor_ribbon": ribbon, "topology": render.get("topology", {}),
        "body_sweep": {"polygons": [], "displayed_as_corridor": False},
        "ground_z_m": z, "ground_projection_quality": ground_quality,
        "coordinates": "world_x_y_z_up_m", "status": "available", "warnings": warnings}
    frame["visualization"] = visualization
    frame["potential"] = {**capabilities["potential_field"], "deferred": False, "values": []}
    frame["library_scene"] = {"source": "archived_scene_json", "snapshot_only": True,
        "read_only": True, "coordinates": archived.get("coordinates"), "capabilities": capabilities,
        "ego_geometry_source": (source_diag.get("field_sources") or {}).get("ego_geometry")
            if isinstance(source_diag.get("field_sources") or {}, dict) else None,
        "notes": ["每个证据是独立存档快照；未来实驶线来自单独校验的离线监督，不作为在线输入",
                  "route为存档导航参考，history为已观测历史；均未作为规划轨迹绘制",
                  "三维地面采用局部平面；未恢复道路坡度、对象俯仰或侧倾",
                  "八向匹配不等于势场源、MPC规划对象或模型关注对象"]}
    return frame
