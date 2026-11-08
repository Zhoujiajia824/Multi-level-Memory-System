"""可见真值到 BEV、面积占据及当前状态九宫格；不使用模型答案构造检索键。"""
from __future__ import annotations

import math
import colorsys
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont
from .paths import filesystem_path
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union

from .contracts import Actor, SceneSnapshot, Representations
from .configuration import (grid_distance_config, grid_metric_signature,
                            grid_observation_config, grid_observation_signature,
                            occupancy_distance_config)

FEATURE_VERSION = "bev-occ-grid-2.0"
GRID_REPRESENTATION_VERSION = "lane-eight-interaction-2.0"
OCC_REPRESENTATION_VERSION = "area-occupancy-evidence-2.0"
CHANNELS = ("crosswalk", "ego", "vehicle", "bicycle", "pedestrian", "static_obstacle")
# NAVSIM官方对象配色；保留本项目真实尺寸和行人圆形表示。
COLORS = {"ego": (222, 112, 97), "vehicle": (105, 156, 219), "bicycle": (227, 140, 71),
          "pedestrian": (176, 122, 161), "static_obstacle": (237, 201, 72)}
BACKGROUND_COLOR = (255, 255, 255)
ROAD_COLOR = (211, 211, 211)
CENTER_LINE_COLOR = (102, 102, 102)
CROSSWALK_COLOR = (176, 122, 161)
CROSSWALK_ALPHA = .3
SLOTS = (("left_front", "front", "right_front"), ("left_side", "ego", "right_side"),
         ("left_rear", "rear", "right_rear"))


def valid_dino_signature(value):
    """真实模型身份与显式测试身份分开命名，空身份不能作为检索证据。"""
    return isinstance(value,str) and bool(re.fullmatch(r"dinov2-base-sha256:[0-9a-f]{64}|test:[a-zA-Z0-9_.-]+",value))


def dino_state_signature(state,config):
    """对实际加载的权重逐张量哈希，不依赖文件路径或检查点封装格式。"""
    # 来源路径与下载提交号不是前向运算定义；相同权重搬迁后须保持身份。
    metadata={'_name_or_path','name_or_path','_commit_hash','architectures','transformers_version'}
    semantic_config={key:value for key,value in config.items() if key not in metadata}
    digest=hashlib.sha256()
    digest.update(json.dumps(semantic_config,sort_keys=True,separators=(',',':'),default=str).encode('utf-8'))
    for name,value in sorted(state.items()):
        if hasattr(value,'detach'):
            import torch
            tensor=value.detach().cpu().contiguous()
            dtype=str(tensor.dtype).removeprefix('torch.')
            shape=list(tensor.shape)
            raw=tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        else:
            array=np.ascontiguousarray(value)
            dtype=str(array.dtype);shape=list(array.shape);raw=array.tobytes()
        header=json.dumps([name,dtype,shape],ensure_ascii=False,separators=(',',':')).encode('utf-8')
        digest.update(len(header).to_bytes(8,'little'));digest.update(header)
        digest.update(len(raw).to_bytes(8,'little'));digest.update(raw)
    return 'dinov2-base-sha256:'+digest.hexdigest()


def _shape(actor: Actor):
    if actor.kind == "pedestrian":
        return Point(actor.position).buffer(actor.radius or actor.width / 2, quad_segs=24)
    c, s = math.cos(actor.heading), math.sin(actor.heading)
    p = np.array([[actor.length/2, actor.width/2], [actor.length/2, -actor.width/2],
                  [-actor.length/2, -actor.width/2], [-actor.length/2, actor.width/2]])
    return Polygon(p @ np.array([[c, s], [-s, c]]) + actor.position)


def _local(points, ego):
    c, s = math.cos(ego.heading), math.sin(ego.heading)
    return (np.asarray(points, dtype=float) - ego.position) @ np.array([[c, -s], [s, c]])


def _pixel(points, ego):
    p = _local(points, ego)
    return np.column_stack((250 - p[:, 1] * 10, 700 - p[:, 0] * 10))


def _reference_data(reference):
    points = np.asarray(reference, dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.all(np.isfinite(points)) or len(points) < 2:
        raise ValueError("统一 Frenet 参考线至少需要两个点")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
    points = points[keep]
    if len(points)<2:
        raise ValueError('统一 Frenet 参考线必须包含非零长度路段')
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    yaw = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    mid = (arc[1:] + arc[:-1]) / 2
    kappa = np.gradient(yaw, mid) if len(yaw) > 1 else np.zeros(1)
    dkappa = np.gradient(kappa, mid) if len(yaw) > 1 else np.zeros(1)
    return points, delta, lengths, arc, yaw, mid, kappa, dkappa


def _project(points, data, anchor_s=None):
    """统一参考线投影；并列解以自车附近的连续弧长支路优先。"""
    ref, delta, lengths, arc, yaw, mid, kap, dkap = data
    values = []
    for point in np.atleast_2d(points):
        t = np.clip(np.sum((point - ref[:-1]) * delta, axis=1) / lengths**2, 0, 1)
        feet = ref[:-1] + t[:, None] * delta
        distances = np.linalg.norm(point - feet, axis=1)
        candidates = np.where(distances <= distances.min() + 1e-5)[0]
        j = candidates[np.argmin(np.abs(arc[candidates] + t[candidates]*lengths[candidates] - anchor_s))] if anchor_s is not None else candidates[0]
        tangent = delta[j] / lengths[j]
        normal = np.array([-tangent[1], tangent[0]])
        longitudinal = arc[j] + t[j] * lengths[j]
        values.append((longitudinal, np.dot(point - feet[j], normal), yaw[j],
                       np.interp(longitudinal, mid, kap), np.interp(longitudinal, mid, dkap)))
    return np.asarray(values)


def _frenet(actor, data, anchor_s=None):
    if any(np.asarray(value).shape != (2,) or not np.all(np.isfinite(value))
           for value in (actor.position, actor.velocity, actor.acceleration)) or not np.isfinite(actor.heading):
        return None
    s, d, angle, k, kp = _project(actor.position, data, anchor_s)[0]
    t = np.array([math.cos(angle), math.sin(angle)])
    n = np.array([-t[1], t[0]])
    denominator = 1-k*d
    if abs(denominator) < .1:
        return None
    sd = float(np.dot(actor.velocity, t) / denominator)
    dd = float(np.dot(actor.velocity, n))
    # 真值世界加速度转换成 Frenet 坐标的二阶导数，包含曲率耦合项。
    sdd = float((np.dot(actor.acceleration, t) + kp*d*sd**2 + 2*k*sd*dd) / denominator)
    ddd = float(np.dot(actor.acceleration, n) - k*denominator*sd**2)
    yaw = float((actor.heading-angle+math.pi) % (2*math.pi)-math.pi)
    return {"position": [float(s), float(d)], "velocity": [sd, dd], "acceleration": [sdd, ddd],
            "heading": yaw, "reference_heading": float(angle)}


def _adjacent_road_evidence(scene, current, lanes, side, settings):
    """只在当前截面确认同向邻道与边界权限；远处虚线不能提供当前位置证据。"""
    unknown = {"known": False, "exists": None, "crossable": None, "reason": "map_unknown"}
    if current is None or scene.diagnostics.get("lane_map_complete") is not True:
        return unknown
    identity = getattr(current, side)
    if identity is None:
        return {"known": True, "exists": False, "crossable": False, "reason": "no_adjacent_lane"}
    target = lanes.get(identity)
    if target is None:
        return dict(unknown, reason="adjacent_lane_geometry_missing")
    try:
        source_data, target_data = _reference_data(current.center), _reference_data(target.center)
        source_s, _, source_heading, _, _ = _project(scene.ego.position, source_data)[0]
        a = np.array([np.interp(source_s, source_data[3], source_data[0][:, i]) for i in (0, 1)])
        target_s, _, target_heading, _, _ = _project(a, target_data)[0]
        if math.cos(target_heading-source_heading) <= 0:
            return {"known": True, "exists": False, "crossable": False, "reason": "no_same_direction_adjacent_lane"}
        b = np.array([np.interp(target_s, target_data[3], target_data[0][:, i]) for i in (0, 1)])
        middle = Point((a+b)/2)
        permissions = []
        for lane in (current, target):
            nearby = [edge for edge in lane.boundaries
                      if len(edge.get("points", [])) >= 2
                      and LineString(edge["points"]).distance(middle) <= settings["boundary_local_tolerance_m"]]
            if not nearby or any(type(edge.get("crossable")) is not bool for edge in nearby):
                return dict(unknown, reason="current_boundary_permission_unknown")
            permissions.extend(edge["crossable"] for edge in nearby)
    except (ValueError, TypeError, IndexError):
        return dict(unknown, reason="current_boundary_geometry_invalid")
    crossable = bool(all(permissions))
    return {"known": True, "exists": True, "crossable": crossable,
            "reason": "crossable_boundary" if crossable else "noncrossable_boundary"}


def _visibility_geometry(scene):
    """保留相机可见域的孔洞和分离区域；新证据损坏时不退回旧的全覆盖多边形。"""
    try:
        if 'visibility_wkt' in scene.diagnostics:
            raw=scene.diagnostics['visibility_wkt']
            if not isinstance(raw,str):
                return None
            geometry=shapely.from_wkt(raw)
        else:
            raw=scene.diagnostics.get('visibility_polygon')
            if raw is None:
                return None
            geometry=Polygon(raw)
        if (geometry.geom_type not in ('Polygon','MultiPolygon') or geometry.is_empty
            or not geometry.is_valid or not np.all(np.isfinite(shapely.get_coordinates(geometry)))):
            return None
        return geometry
    except (ValueError,TypeError,shapely.errors.GEOSException):
        return None


def build_grid(scene: SceneSnapshot, reference=None, current_lane_id=None, reference_anchor_s=None,
               distance_config=None, observation_config=None):
    """每格只选当前可见对象；道路不适用、确认空与遮挡未知分别编码。"""
    observation_config = grid_observation_config(observation_config)
    data = _reference_data(scene.route if reference is None else reference)
    anchor = scene.diagnostics.get("reference_anchor_s") if reference_anchor_s is None and reference is None else reference_anchor_s
    ego_f = _frenet(scene.ego, data, anchor)
    if ego_f is None:
        return {"version": FEATURE_VERSION, "representation_version": GRID_REPRESENTATION_VERSION,
                "valid": False, "reason": "自车 Frenet 投影奇异", "objects": []}
    ego_s = ego_f["position"][0]
    lane_by_id = {lane.id: lane for lane in scene.lanes}
    current = lane_by_id.get(current_lane_id or scene.ego.lane_id)
    roads = {side: _adjacent_road_evidence(scene, current, lane_by_id, side, observation_config)
             for side in ("left", "right")}
    lane_ids = {"center": current.id if current else None, "left": current.left if current else None,
                "right": current.right if current else None}
    lane_shapes = {}
    for role, lane_id in lane_ids.items():
        if role != "center" and (not roads[role]["known"] or not roads[role]["crossable"]):
            lane_shapes[role] = None
            continue
        lane = lane_by_id.get(lane_id)
        if lane is None:
            lane_shapes[role] = None
            continue
        origin_lane = lane
        segments, seen = [], set()
        while lane is not None and lane.id not in seen:
            seen.add(lane.id)
            if len(lane.polygon) >= 3:
                segments.append(Polygon(lane.polygon))
            successors = [lane_by_id[i] for i in lane.successors if i in lane_by_id and i not in seen]
            if not successors:
                break
            # 多后继只为区域几何选择与统一参考线最贴近的一段，不读取周车隐藏路线。
            lane = min(successors, key=lambda a: abs(_project(a.center[min(2, len(a.center)-1)], data, ego_s)[0, 1]))
        # 跨过道路分段接缝后，后方 30m 仍可能位于前驱车道。
        # 每次只接续一条与同一 Frenet 参考最一致的公开前驱，不能将汇入
        # 分叉的全部来向合并成“本车道后方”，也不能读取交通对象的路线。
        local_data = _reference_data(origin_lane.center)
        local_s = _project(scene.ego.position,local_data)[0,0]
        foot = np.array([np.interp(local_s,local_data[3],local_data[0][:,i]) for i in (0,1)])
        expected_d = _project(foot,data,ego_s)[0,1]
        previous = origin_lane
        while True:
            frontier = float(_project(previous.center,data,ego_s)[:,0].min())
            if frontier <= ego_s-30:
                break
            choices = []
            for candidate in scene.lanes:
                if previous.id not in candidate.successors or candidate.id in seen:
                    continue
                candidate_data = _reference_data(candidate.center)
                end = candidate_data[3][-1]
                span = min(end,max(5.,frontier-(ego_s-30)+5))
                sample_s = np.linspace(end-span,end,9)
                sample = np.column_stack([np.interp(sample_s,candidate_data[3],candidate_data[0][:,i]) for i in (0,1)])
                projected = _project(sample,data,ego_s)
                if projected[:,0].min() >= frontier-.01:
                    continue
                candidate_heading = np.interp(sample_s,candidate_data[5],candidate_data[4])
                angle = abs((candidate_heading-projected[:,2]+np.pi)%(2*np.pi)-np.pi)
                if np.mean(angle) >= np.pi/2:
                    continue
                cost = float(np.mean(abs(projected[:,1]-expected_d)/max(candidate.width,.1)+2*angle))
                choices.append((cost,candidate.id,candidate))
            if not choices:
                break
            previous = min(choices,key=lambda item:(item[0],item[1]))[2]
            seen.add(previous.id)
            if len(previous.polygon)>=3:
                segments.append(Polygon(previous.polygon))
        lane_shapes[role] = unary_union(segments) if segments else None
    ego_shape = _shape(scene.ego)
    ego_interval = _project(np.asarray(ego_shape.exterior.coords), data, ego_s)[:, 0]
    candidates = {name: [] for row in SLOTS for name in row if name != "ego"}
    object_data = {}
    invalid_state_ids = []
    for actor in scene.actors:
        if not actor.visible or actor.kind not in ("vehicle", "bicycle", "pedestrian"):
            continue
        f = _frenet(actor, data, ego_s)
        if f is None:
            invalid_state_ids.append(actor.id)
            continue
        ds = f["position"][0] - ego_s
        if ds < -30 or ds > 70:
            continue
        body = _shape(actor)
        interval = _project(np.asarray(body.exterior.coords), data, ego_s)[:, 0]
        overlaps = interval.max() >= ego_interval.min() and interval.min() <= ego_interval.max()
        front = ds >= 0
        longitudinal_gap = max(0., float(interval.min()-ego_interval.max()) if front else float(ego_interval.min()-interval.max()))
        roles = []
        for side, shape in lane_shapes.items():
            if shape is None or body.intersection(shape).area < observation_config["body_lane_area_tolerance_m2"]:
                continue
            if side == "center":
                name = "front" if front else "rear"
            else:
                name = side + ("_side" if overlaps else "_front" if front else "_rear")
            gap = body.distance(ego_shape) if name.endswith("side") else longitudinal_gap
            candidates[name].append((gap, abs(ds), actor.id))
            roles.append(name)
        object_data[actor.id] = {"id": actor.id, "kind": actor.kind, "position": [ds, f["position"][1]-ego_f["position"][1]],
                                "velocity": (np.array(f["velocity"])-ego_f["velocity"]).tolist(),
                                "acceleration": f["acceleration"], "heading": f["heading"],
                                "length": actor.length, "width": actor.width, "slots": [],
                                "state_valid": True, "longitudinal_overlap": bool(overlaps)}
    observed = _visibility_geometry(scene)
    known_geometry = None
    if observed is not None:
        known_geometry = unary_union([observed, ego_shape]+[_shape(a) for a in scene.actors if a.visible])
    lateral_widths = {}
    def slot_coverage(side, name):
        """空状态只依据可见几何与地图，不使用遮挡后的对象数量或状态。"""
        if known_geometry is None or lane_shapes[side] is None:
            return {"coverage": 0., "raw_coverage": 0., "unobserved_area_m2": None,
                    "observation_reason": "visibility_missing"}
        if name == "front":
            start,end = ego_s,ego_s+70
        elif name == "rear":
            start,end = ego_s-30,ego_s
        elif name.endswith("front"):
            start,end = ego_interval.max(),ego_s+70
        elif name.endswith("rear"):
            start,end = ego_s-30,ego_interval.min()
        else:
            start,end = ego_interval.min(),ego_interval.max()
        ref,_,_,arc,_,_,_,_ = data
        if min(arc[-1], end) <= max(arc[0], start):
            return {"coverage": 0., "raw_coverage": 0., "unobserved_area_m2": None,
                    "observation_reason": "reference_region_missing"}
        start,end = max(arc[0],start),min(arc[-1],end)
        # 沿参考线已有顶点构造截面；不以额外稀疏固定采样漏掉弯道。
        ss = np.r_[start, arc[(arc>start)&(arc<end)], end]
        xy = np.column_stack([np.interp(ss,arc,ref[:,i]) for i in (0,1)])
        direction = np.gradient(xy,axis=0)
        direction /= np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
        normal = np.column_stack((-direction[:,1],direction[:,0]))
        if side not in lateral_widths:
            boundaries = np.vstack([np.asarray(part.exterior.coords) for part in _iter_polygons(lane_shapes[side])])
            lateral_extent = float(np.max(np.abs(_project(boundaries,data,ego_s)[:,1])))
            lateral_widths[side] = lateral_extent+lane_by_id[lane_ids[side]].width
        half_width = lateral_widths[side]
        band = Polygon(np.vstack((xy+normal*half_width,(xy-normal*half_width)[::-1]))).buffer(0)
        region = lane_shapes[side].intersection(band)
        area = region.area
        if area <= observation_config["geometry_area_epsilon_m2"]:
            return {"coverage": 0., "raw_coverage": 0., "unobserved_area_m2": None,
                    "observation_reason": "region_geometry_missing"}
        unknown_area = float(region.difference(known_geometry).area)
        known_area = max(0., area-unknown_area)
        return {"coverage": min(1., (known_area+observation_config["empty_unknown_area_tolerance_m2"])/area),
                "raw_coverage": known_area/area, "unobserved_area_m2": unknown_area,
                "region_area_m2": float(area), "known_area_m2": float(known_area),
                "observation_reason": "geometry_visibility"}
    matrices = {key: [[None for _ in range(3)] for _ in range(3)] for key in ("position", "velocity", "acceleration", "heading")}
    slots, selected = {}, {}
    for i, row in enumerate(SLOTS):
        for j, name in enumerate(row):
            if name == "ego":
                for key in matrices:
                    matrices[key][i][j] = [0., 0.] if key in ("position", "velocity") else ego_f[key]
                slots[name] = {"state": "occupied", "id": scene.ego.id, "kind": scene.ego.kind,
                               "applicable": True, "clear": True, "state_valid": True,
                               "coverage": 1., "raw_coverage": 1., "unobserved_area_m2": 0.}
                continue
            side = "left" if name.startswith("left") else "right" if name.startswith("right") else "center"
            road = roads.get(side)
            shared = {"id": None, "kind": None, "state_valid": False, "coverage": 0.,
                      "raw_coverage": 0., "unobserved_area_m2": None}
            if road is not None and not road["known"]:
                slots[name] = dict(shared, state="unknown", applicable=None, clear=False, reason=road["reason"])
            elif road is not None and not road["crossable"]:
                slots[name] = dict(shared, state="not_applicable", applicable=False, clear=True,
                                   reason="no_adjacent_lane" if not road["exists"] else "noncrossable_boundary")
            elif lane_shapes[side] is None:
                slots[name] = dict(shared, state="unknown", applicable=True, clear=False, reason="lane_geometry_missing")
            elif candidates[name]:
                gap, _, actor_id = min(candidates[name])
                slots[name] = dict(slot_coverage(side,name), state="occupied", id=actor_id,
                                   kind=object_data[actor_id]["kind"], clearance=float(gap),
                                   applicable=True, clear=True, state_valid=True)
                selected.setdefault(actor_id, dict(object_data[actor_id], slots=[]))["slots"].append(name)
                for key in matrices:
                    matrices[key][i][j] = object_data[actor_id][key]
            else:
                evidence = slot_coverage(side,name)
                clear = evidence["coverage"] >= observation_config["empty_min_coverage"]
                slots[name] = dict(shared, **evidence, state="empty" if clear else "unknown",
                                   applicable=True, clear=bool(clear), reason="observed_empty" if clear else evidence["observation_reason"])
    distance_config=grid_distance_config(distance_config)
    return {"version": FEATURE_VERSION, "representation_version": GRID_REPRESENTATION_VERSION,
            "valid": not invalid_state_ids, "reason": "可见对象状态无效" if invalid_state_ids else "",
            "invalid_state_ids": sorted(invalid_state_ids), "coordinates": "common_route_frenet", "slots": slots, "roads": roads,
            "distance_config":distance_config,"metric_signature":grid_metric_signature(distance_config),
            "observation_config": observation_config,
            "observation_signature": grid_observation_signature(observation_config),
            "matrices": matrices, "objects": [selected[key] for key in sorted(selected)],
            "ego": dict(ego_f, id=scene.ego.id, world_position=np.asarray(scene.ego.position).tolist(),
                        world_velocity=np.asarray(scene.ego.velocity).tolist(), world_acceleration=np.asarray(scene.ego.acceleration).tolist(),
                        world_heading=scene.ego.heading, length=scene.ego.length, width=scene.ego.width)}


class NineGridBuilder:
    """10 Hz 维护统一参考及车道迟滞；1 Hz 检索只读取已经确认的九宫格。"""
    def __init__(self, config=None):
        settings = (config or {}).get("coordinates",{})
        self.distance_config=grid_distance_config((config or {}).get("grid_distance"))
        self.observation_config=grid_observation_config((config or {}).get("grid_observation"))
        self.occupancy_config=occupancy_distance_config((config or {}).get("occupancy_distance"))
        self.hold = float(settings.get("lane_switch_hold_s",.3))
        self.fraction = float(settings.get("lane_switch_overlap_fraction",.5))
        self.back = float(settings.get("reference_back_m",60))
        self.front = float(settings.get("reference_front_m",150))
        self.lane_id = None
        self.candidate = None
        self.candidate_since = None
        self.reference = None
        self.anchor = None
        self.reference_version = 0
        self.advanced_frame = None
        self.cached_frame = None
        self.cached_grid = None
        self.last_time = None
        self.navigation_revision = None

    def _reference(self,scene,lane_id):
        lanes = {x.id:x for x in scene.lanes}
        if lane_id not in lanes:
            return np.asarray(scene.route).copy()
        lane = lanes[lane_id]
        points = np.asarray(lane.center).copy()
        visited = {lane.id}
        previous = lane
        def join(first,second):
            data = _reference_data(second)
            arc,distance,_,_,_ = _project(first[-1],data)[0]
            # 地图车道多含端部延伸，剔除几何连续后继开头已经经过的区间。
            if abs(distance) < 1 and arc > .1:
                index = int(np.searchsorted(data[3],arc,side="right"))
                second = second[min(index,len(second)-1):]
            return np.vstack((first,second))
        def heading(line,end=False):
            a,b = (line[-2],line[-1]) if end else (line[0],line[1])
            return math.atan2(b[1]-a[1],b[0]-a[0])
        def successor_cost(candidate,source,forward=True):
            end = source.center[-1] if forward else source.center[0]
            start = candidate.center[0] if forward else candidate.center[-1]
            h0 = heading(source.center,forward)
            h1 = heading(candidate.center,not forward)
            delta = abs((h1-h0+math.pi)%(2*math.pi)-math.pi)
            route_distance = np.linalg.norm(np.asarray(scene.route)-candidate.center[len(candidate.center)//2],axis=1).min()
            return (float(np.linalg.norm(end-start)+5*delta+.1*route_distance),candidate.id)
        while _project(scene.ego.position,_reference_data(points))[0,0] < self.back:
            options = [x for x in scene.lanes if previous.id in x.successors and x.id not in visited]
            if not options:
                break
            previous = min(options,key=lambda x:successor_cost(x,previous,False))
            visited.add(previous.id)
            points = join(previous.center,points)
        following = lane
        while True:
            data = _reference_data(points)
            s = _project(scene.ego.position,data)[0,0]
            if data[3][-1]-s >= self.front:
                break
            options = [lanes[k] for k in following.successors if k in lanes and k not in visited]
            if not options:
                break
            following = min(options,key=lambda x:successor_cost(x,following,True))
            visited.add(following.id)
            points = join(points,following.center)
        # 相邻路段采样端部可能重合或有 5m 重叠，不能把重叠弧长倒走计入参考。
        clean = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point-clean[-1])>.01:
                clean.append(point)
        return np.asarray(clean)

    def update(self,scene: SceneSnapshot,build=True):
        """每帧推进参考；只有需要检索表示时才生成实体矩阵。"""
        if self.advanced_frame != scene.frame_id:
            self._advance(scene)
        if not build:
            return None
        if self.cached_frame == scene.frame_id:
            return self.cached_grid
        grid = build_grid(scene,self.reference,self.lane_id,self.anchor,self.distance_config,self.observation_config)
        grid["reference"] = {"lane_id":self.lane_id,"observed_lane_id":scene.ego.lane_id,"revision":self.reference_version,
                             "ego_arc_m":float(self.anchor),"pending_lane_id":self.candidate,
                             "pending_since_s":self.candidate_since}
        self.cached_frame,self.cached_grid = scene.frame_id,grid
        return grid

    def _advance(self,scene: SceneSnapshot):
        if self.last_time is not None and scene.time < self.last_time:
            raise ValueError("九宫格构建器不可跨轮复用，必须重置状态")
        lanes = {lane.id:lane for lane in scene.lanes}
        observed = scene.ego.lane_id
        switch = self.lane_id is None
        if switch:
            self.lane_id = observed
        elif observed != self.lane_id and observed in lanes:
            old = lanes.get(self.lane_id)
            # 纵向驶入公开合法后继不属于换道；横向换道必须持续满足面积及中心条件。
            if old is None or (observed in old.successors and observed not in (old.left,old.right)):
                self.lane_id = observed
                switch = True
            elif observed in (old.left,old.right):
                body = _shape(scene.ego)
                target = Polygon(lanes[observed].polygon)
                eligible = target.covers(Point(scene.ego.position)) and body.intersection(target).area/body.area > self.fraction
                if eligible:
                    if self.candidate != observed:
                        self.candidate,self.candidate_since = observed,scene.time
                    elif scene.time-self.candidate_since >= self.hold-1e-8:
                        self.lane_id = observed
                        switch = True
                else:
                    self.candidate,self.candidate_since = None,None
            else:
                self.candidate,self.candidate_since = None,None
        else:
            self.candidate,self.candidate_since = None,None
        navigation_revision = scene.diagnostics.get("reference_revision")
        navigation_changed = navigation_revision is not None and navigation_revision != self.navigation_revision
        if switch or self.reference is None or navigation_changed:
            self.reference = self._reference(scene,self.lane_id)
            self.anchor = None
            self.reference_version += 1
            self.candidate,self.candidate_since = None,None
        self.navigation_revision = navigation_revision
        data = _reference_data(self.reference)
        self.anchor = _project(scene.ego.position,data,self.anchor)[0,0]
        self.advanced_frame,self.last_time = scene.frame_id,scene.time
        self.cached_frame,self.cached_grid = None,None


def _iter_polygons(geometry):
    if geometry.is_empty:
        return []
    return [geometry] if geometry.geom_type == "Polygon" else [g for g in geometry.geoms if g.geom_type == "Polygon"]


def _crosswalks(scene):
    return [Polygon(p) for p in scene.rules.get("crosswalks", []) if len(p) >= 3]


@lru_cache(maxsize=1)
def _occupancy_cells():
    rows,cols = np.meshgrid(np.arange(200),np.arange(100),indexing="ij")
    return shapely.box(70-(rows+1)*.5,25-(cols+1)*.5,70-rows*.5,25-cols*.5)


def _paint_shape(draw, shape, ego, fill, outline=None, holes_fill=None):
    for part in _iter_polygons(shape):
        draw.polygon([tuple(p) for p in _pixel(np.asarray(part.exterior.coords), ego)], fill=fill, outline=outline)
        if holes_fill is not None:
            for ring in part.interiors:
                draw.polygon([tuple(p) for p in _pixel(np.asarray(ring.coords),ego)],fill=holes_fill)


def _drawing_center(lane):
    # 边线可能已在合法连接口裁开，不能按片段下标求平均生成虚假中心线。
    return np.asarray(lane.center)


class _AntialiasedDraw:
    """以固定逻辑像素绘图，内部超采样，避免改变几何比例和最终画布大小。"""
    def __init__(self,image,scale):
        self.draw=ImageDraw.Draw(image)
        self.scale=scale
        self.font=ImageFont.load_default(size=11*scale)

    def _coordinates(self,points):
        values=np.asarray(points,dtype=float)*self.scale
        return tuple(values.tolist()) if values.ndim==1 else [tuple(point) for point in values]

    def line(self,points,fill,width=1):
        self.draw.line(self._coordinates(points),fill=fill,width=max(1,round(width*self.scale)),joint='curve')

    def polygon(self,points,fill,outline=None):
        self.draw.polygon(self._coordinates(points),fill=fill,outline=outline,width=self.scale)

    def ellipse(self,bounds,fill):
        self.draw.ellipse(self._coordinates(bounds),fill=fill)

    def rectangle(self,bounds,fill):
        self.draw.rectangle(self._coordinates(bounds),fill=fill)

    def textlength(self,text):
        return self.draw.textlength(text,font=self.font)/self.scale

    def text(self,position,text,fill):
        self.draw.text(self._coordinates(position),text,fill=fill,font=self.font)


def _draw_dashed(draw,points,fill,width=2,dash=6,gap=5):
    """虚线按像素弧长连续绘制，避免每段采样点重新开始虚线。"""
    total = 0.
    for first,second in zip(points[:-1],points[1:]):
        direction = second-first
        length = float(np.linalg.norm(direction))
        if length<1e-8:
            continue
        lower,upper = 0.,1.
        for axis,maximum in ((0,500),(1,1000)):
            if abs(direction[axis])<1e-10:
                if not -2<=first[axis]<=maximum+2:
                    upper=-1.
                    break
            else:
                limits=sorted(((-2-first[axis])/direction[axis],(maximum+2-first[axis])/direction[axis]))
                lower=max(lower,limits[0]);upper=min(upper,limits[1])
        position=max(0.,lower)*length
        end=min(1.,upper)*length
        phase=total+position
        while position < end-1e-8:
            cycle = phase%(dash+gap)
            if abs(cycle-dash)<1e-8:
                cycle=float(dash)
            elif abs(cycle-(dash+gap))<1e-8:
                cycle=0.
            step = min(end-position,(dash-cycle) if cycle<dash else (dash+gap-cycle))
            if cycle<dash:
                draw.line([tuple(first+direction*(position/length)),tuple(first+direction*((position+step)/length))],fill=fill,width=width)
            phase += step
            position += step
        total+=length


def _trajectory_color(color,speed):
    """固定0—120km/h速度标尺；类别色相不变，提高饱和度，不混白或压暗。"""
    ratio=float(np.clip(float(speed)*3.6/120,0.,1.))**.65
    hue=colorsys.rgb_to_hsv(*(value/255 for value in color))[0]
    rgb=colorsys.hsv_to_rgb(hue,.65+.35*ratio,1.)
    return tuple(round(value*255) for value in rgb)


def _observed_segments(samples):
    """2Hz真实历史缺失处断开，不能把遮挡前后位置画成已知连续轨迹。"""
    for first,second in zip(samples[:-1],samples[1:]):
        gap=float(second.get('time',math.nan))-float(first.get('time',math.nan))
        if 0.<gap<=.5+1e-6:yield first,second


def _draw_trajectory(draw,samples,ego,color):
    """在真实2Hz观测之间线性插值速度着色，不按时间伪造加减速。"""
    if len(samples)<2:return
    segments=list(_observed_segments(samples))
    for pair in segments:draw.line(_pixel([item['position'] for item in pair],ego),fill=BACKGROUND_COLOR,width=5)
    for pair in segments:
        first,second=_pixel([item['position'] for item in pair],ego)
        speeds=[float(item.get('speed',np.linalg.norm(item.get('velocity',[0.,0.])))) for item in pair]
        count=max(1,int(math.ceil(np.linalg.norm(second-first)/3)))
        for part in range(count):
            lower,upper=part/count,(part+1)/count
            speed=speeds[0]+(speeds[1]-speeds[0])*(lower+upper)/2
            draw.line([first+(second-first)*lower,first+(second-first)*upper],
                      fill=_trajectory_color(color,speed),width=3.5)


def render_bev(scene, labels=True):
    """固定尺寸画布；标签层仅用于 VLM，DINO 使用同一底图的无字版本。"""
    scale=4
    image = Image.new("RGB", (500*scale, 1000*scale), BACKGROUND_COLOR)
    draw = _AntialiasedDraw(image,scale)
    # 显示实际道路地表，规划 on_lane 的端部容差不应绘成突出或虚构的路口。
    surface=scene.assets.get('bev_drivable',scene.drivable)
    if surface is not None:
        _paint_shape(draw,surface,scene.ego,ROAD_COLOR,holes_fill=BACKGROUND_COLOR)
    crosswalks=_crosswalks(scene)
    if crosswalks:
        # NAVSIM人行横道为30%紫色覆盖；先合并再透明合成，交叠不重复加深。
        mask=Image.new('L',image.size,0)
        _paint_shape(_AntialiasedDraw(mask,scale),unary_union(crosswalks),scene.ego,
                     round(255*CROSSWALK_ALPHA),holes_fill=0)
        image.paste(CROSSWALK_COLOR,(0,0),mask)
    for lane in scene.lanes:
        # polygon 仅表示地表，闭环描边会把车道连接段的端面错误画成路障线。
        center = np.asarray(scene.assets.get("physical_lane_centers",{}).get(lane.id,_drawing_center(lane)))
        if len(center) >= 2:
            points = [tuple(p) for p in _pixel(center, scene.ego)]
            _draw_dashed(draw,np.asarray(points),CENTER_LINE_COLOR,width=.65,dash=3,gap=4)
        # 边界与分割线仅保留在地图数据中供规则、风险场和规划使用，不绘入BEV。
        # 连接段使用地图本身的中心线，不按 successor ID 在端点间虚构直线。
    if labels and isinstance(scene.diagnostics.get('goal_reference'),dict):
        # 只进入模型阅读层；无字DINO图及九宫格/占据检索键保持原样。
        from .geometry import sample_polyline, polyline_lengths
        frozen=scene.diagnostics['goal_reference']
        reference=np.asarray(frozen['points'],float);anchor=float(frozen['s_ego_m'])
        stop=min(polyline_lengths(reference)[-1],anchor+70.)
        arc=np.arange(max(0.,anchor-5.),stop+.001,.5)
        if len(arc)>1:
            _draw_dashed(draw,_pixel(sample_polyline(reference,arc),scene.ego),
                         (25,110,75),width=1.,dash=7,gap=4)
        for distance in range(0,71,10):
            if anchor+distance>stop:continue
            point=_pixel([sample_polyline(reference,anchor+distance)],scene.ego)[0]
            if 10<point[0]<440 and 15<point[1]<970:
                draw.ellipse((point[0]-1.5,point[1]-1.5,point[0]+1.5,point[1]+1.5),fill=(25,110,75))
                draw.text((point[0]+4,point[1]-11),f"s+{distance}m",fill=(25,110,75))
    # 已离开可见集合的对象只绘制真实旧观测，不外推车体、方向或位置。
    for actor_id, history in scene.diagnostics.get("historical_only_objects", {}).items():
        if actor_id == scene.ego.id or any(actor.id == actor_id and actor.visible for actor in scene.actors):
            continue
        observed = sorted((h for h in history if "position" in h and
                           scene.time-3-1e-6 <= float(h.get("time", -math.inf)) <= scene.time),
                          key=lambda h: h["time"])
        sampled = []
        for observation in observed:
            if not sampled or observation["time"]-sampled[-1]["time"] >= .5-1e-6:
                sampled.append(observation)
        # 最后一次真实观测必须保留，以便明确旧信息停止的时刻。
        if observed and (not sampled or sampled[-1] is not observed[-1]):
            sampled.append(observed[-1])
        for pair in _observed_segments(sampled):
            _draw_dashed(draw,_pixel([h['position'] for h in pair],scene.ego),
                         (130,130,130),width=2,dash=6,gap=5)
        for sample in sampled:
            point = _pixel([sample['position']],scene.ego)[0]
            draw.ellipse((point[0]-1,point[1]-1,point[0]+1,point[1]+1),fill=(130,130,130))
        if labels and sampled:
            point = _pixel([sampled[-1]["position"]],scene.ego)[0]
            if 0 <= point[0] < 500 and 0 <= point[1] < 1000:
                label = f"past {actor_id} t={sampled[-1]['time']:.1f}s"
                width = int(draw.textlength(label))+4
                position = (min(max(0,point[0]+4),max(0,500-width)),min(max(0,point[1]-12),986))
                draw.text(position,label,fill=(110,110,110))
    actors = [a for a in scene.actors if a.visible] + [scene.ego]
    # 所有历史先画，车体后画，避免另一对象的轨迹衬线擦掉当前可见车体。
    for actor in actors:
        kind = "ego" if actor.id == scene.ego.id else actor.kind
        color = COLORS.get(kind, (80, 80, 80))
        history = [h for h in actor.history if scene.time-3-1e-6 <= float(h.get("time", scene.time)) <= scene.time and "position" in h]
        history = sorted(history, key=lambda h: h.get("time", 0))
        # 按时间抽样，不能通过固定数组下标把非均匀观测当作 2 Hz。
        sampled = []
        for h in history:
            if not sampled or float(h.get("time", 0))-float(sampled[-1].get("time", 0)) >= .5-1e-6:
                sampled.append(h)
        _draw_trajectory(draw,sampled,scene.ego,color)
    label_boxes = []
    for actor in actors:
        kind = "ego" if actor.id == scene.ego.id else actor.kind
        color = COLORS.get(kind, (80, 80, 80))
        shape = _shape(actor)
        _paint_shape(draw, shape, scene.ego, color, (40, 40, 40))
        # 箭头从车体前沿延伸一个车宽；行人使用运动朝向而非伪造意图。
        direction = math.atan2(actor.velocity[1],actor.velocity[0]) if np.linalg.norm(actor.velocity)>.1 else actor.heading
        start = actor.position + np.array([math.cos(direction), math.sin(direction)])*actor.length/2
        tip = start + np.array([math.cos(direction), math.sin(direction)])*actor.width
        p0, p1 = _pixel([start, tip], scene.ego)
        draw.line([tuple(p0), tuple(p1)], fill=color, width=2)
        vector = p1-p0
        norm = np.linalg.norm(vector)
        if norm > 1:
            normal = np.array([-vector[1], vector[0]]) / norm
            base = p1-vector/norm*min(5., norm/2)
            draw.polygon([tuple(p1), tuple(base+normal*3), tuple(base-normal*3)], fill=color)
        if labels:
            anchor = _pixel([actor.position], scene.ego)[0]
            if not (0 <= anchor[0] < 500 and 0 <= anchor[1] < 1000):
                continue
            label = str(actor.id)
            width = max(20, int(draw.textlength(label))+6)
            for offset in ((8,-15),(8,5),(-width-8,-15),(-width-8,5),(8,-32),(8,22)):
                x = min(max(0, anchor[0]+offset[0]), 500-width)
                y = min(max(0, anchor[1]+offset[1]), 986)
                rect = box(x, y, x+width, y+14)
                if all(not rect.intersects(r) for r in label_boxes):
                    label_boxes.append(rect)
                    draw.rectangle((x,y,x+width,y+14), fill=(255,255,255))
                    draw.text((x+2,y), label, fill=(20,20,20))
                    break
    if labels:
        for x in range(-30, 71, 10):
            y = 700-x*10
            if 0 < y < 1000:
                draw.line((0,y,7,y), fill=(45,45,45))
                draw.text((8,y-6), f"x={x}m", fill=(35,35,35))
        for y in range(-20, 21, 10):
            x = 250-y*10
            draw.line((x,992,x,999), fill=(45,45,45))
            draw.text((max(0,x-15),975), f"{y}m", fill=(35,35,35))
    return image.resize((500,1000),Image.Resampling.LANCZOS)


def render_goal_overlay(scene,plan):
    """独立回放图：不回写原模型输入，也不参与图像检索特征。"""
    image=render_bev(scene,labels=True)
    draw=ImageDraw.Draw(image)
    if len(plan.states)>1:
        points=_pixel(np.asarray(plan.states)[:,:2],scene.ego)
        draw.line([tuple(point) for point in points],fill=(0,150,220),width=2)
    state=plan.diagnostics.get('goal_state',{})
    goal=state.get('active_goal')
    if goal is None and state.get('events'):
        goal=state['events'][-1]
    if goal is not None:
        point=_pixel([goal['world_goal']],scene.ego)[0]
        color=(0,130,60) if goal['status']=='active' else (120,120,120)
        if 0<=point[0]<500 and 0<=point[1]<1000:
            x,y=point;draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
            draw.line((x-7,y,x+7,y),fill=color,width=1)
            draw.line((x,y-7,x,y+7),fill=color,width=1)
        draw.text((10,10),f"goal {goal['status']} ds={goal['local_goal']['delta_s_m']:.1f} d={goal['local_goal']['d_m']:.1f}",fill=color)
    return image


def _occupancy_coverage(cells,geometry):
    """完整覆盖/不相交格直接判定，仅边界格执行同一定义的精确面积求交。"""
    shapely.prepare(geometry)
    covered=shapely.covers(geometry,cells)
    partial=~covered & shapely.intersects(geometry,cells)
    values=covered.astype(float)
    values[partial]=shapely.area(shapely.intersection(cells[partial],geometry))/.25
    return values


def build_occupancy(scene):
    """几何相交面积占据，先并集再栅格化，避免同类对象相交造成大于一。"""
    shapes = {key: [] for key in CHANNELS}
    shapes["crosswalk"] = _crosswalks(scene)
    shapes["ego"] = [_shape(scene.ego)]
    for actor in scene.actors:
        if actor.visible and actor.kind in shapes:
            shapes[actor.kind].append(_shape(actor))
    def to_local_polygon(shape):
        return Polygon(_local(np.asarray(shape.exterior.coords), scene.ego),
                       [_local(np.asarray(r.coords), scene.ego) for r in shape.interiors])
    raw = np.zeros((6,200,100), dtype=np.float32)
    for channel, name in enumerate(CHANNELS):
        parts = [to_local_polygon(p) for shape in shapes[name] for p in _iter_polygons(shape)]
        if not parts:
            continue
        union = unary_union(parts).intersection(box(-30,-25,70,25))
        for part in _iter_polygons(union):
            xmin,ymin,xmax,ymax = part.bounds
            row0,row1 = max(0,int(math.floor((70-xmax)/.5))), min(200,int(math.ceil((70-xmin)/.5)))
            col0,col1 = max(0,int(math.floor((25-ymax)/.5))), min(100,int(math.ceil((25-ymin)/.5)))
            cells = _occupancy_cells()[row0:row1,col0:col1]
            raw[channel,row0:row1,col0:col1] += _occupancy_coverage(cells,part)
    raw = np.clip(raw,0,1)
    # 缺少可见几何不是全局可见证据；地图横道也须由平台明确确认可用。
    known = np.zeros_like(raw,dtype=np.float32)
    known[1] = 1
    visible_shape = _visibility_geometry(scene)
    if visible_shape is not None:
        # 完整状态授权意味着部分可见对象的完整实体占据已知。
        visible_shape = unary_union([visible_shape, _shape(scene.ego)] + [_shape(a) for a in scene.actors if a.visible])
        local_visible = unary_union([to_local_polygon(p) for p in _iter_polygons(visible_shape)])
        dynamic_known = _occupancy_coverage(_occupancy_cells(),local_visible).astype(np.float32)
        known[2:] = dynamic_known
    if scene.rules.get("map_available",{}).get("crosswalk") is True:
        known[0] = 1
    available = scene.diagnostics.get("occupancy_channels_available", {})
    for i,name in enumerate(CHANNELS):
        if available.get(name, True) is False:
            known[i] = 0
    # 检索分辨率 2m，已知比例保留为权重，不能把遮挡填成空车道。
    pooled = raw.reshape(6,50,4,25,4).mean(axis=(2,4))
    pooled_known = known.reshape(6,50,4,25,4).mean(axis=(2,4))
    return raw, known, pooled, pooled_known


class DINOExtractor:
    """真实 DINOv2 Base CLS；首次使用加载，显式拉伸到同纵横比 252×504。"""
    def __init__(self, model_name="facebook/dinov2-base", device=None, local_files_only=False):
        # 显式启用特征提取时先载入张量运行库，避免 Windows 图像 codec 抢先载入同名 DLL。
        # 权重仍在第一次 extract 时才读取；关闭记忆功能不创建本实例。
        import torch
        self.model_name = str(model_name)
        self.device = device
        self.local_files_only = local_files_only
        self._model = None
        self.signature = None

    @staticmethod
    def preprocess(image_path):
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((252,504),Image.Resampling.BICUBIC)
        array = np.asarray(image,dtype=np.float32)/255
        array = (array-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        return array.transpose(2,0,1).copy()

    def __call__(self, image_path):
        import torch
        if self._model is None:
            from transformers import AutoModel
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModel.from_pretrained(self.model_name, local_files_only=self.local_files_only).to(self.device).eval()
            if model.config.hidden_size != 768 or model.config.model_type != "dinov2":
                raise ValueError("检索特征版本要求 DINOv2 Base 的 768 维 CLS")
            # 一次加载只计算一次身份；后续帧直接复用，不再次遍历权重。
            self.signature=dino_state_signature(model.state_dict(),model.config.to_dict())
            self._model=model
        tensor = torch.from_numpy(self.preprocess(image_path)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            result = self._model(pixel_values=tensor).last_hidden_state[:,0,:]
            result = torch.nn.functional.normalize(result,dim=-1)
        return result[0].cpu().numpy().astype(np.float32)

    extract = __call__


def build_representations(scene: SceneSnapshot, output_dir, extractor=None, grid_builder=None) -> Representations:
    root = filesystem_path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    # 文件名不直接使用外部 ID，避免斜杠和 Windows 保留字符。
    name = hashlib.sha256(scene.frame_id.encode("utf-8")).hexdigest()[:20]
    bev_path = root/f"{name}_bev.png"
    feature_path = root/f"{name}_feature.png"
    render_bev(scene,labels=False).save(feature_path)
    render_bev(scene,labels=True).save(bev_path)
    raw, raw_known, occ, known = build_occupancy(scene)
    np.savez_compressed(root/f"{name}_occupancy.npz",raw=raw,raw_known=raw_known,occ=occ,known=known)
    feature = np.asarray(extractor(feature_path),dtype=np.float32) if extractor is not None else None
    signature=getattr(extractor,'signature',None) if extractor is not None else None
    if feature is not None and not valid_dino_signature(signature):
        raise ValueError('图像特征提取器没有有效权重身份；测试替身也必须显式使用 test: 身份')
    return Representations(scene.frame_id,scene.time,str(bev_path.resolve()),str(feature_path.resolve()),occ,known,
                           grid_builder.update(scene) if grid_builder is not None else build_grid(scene),feature,
                           dino_signature=signature,
                           occupancy_config=(grid_builder.occupancy_config if grid_builder is not None else occupancy_distance_config()))
