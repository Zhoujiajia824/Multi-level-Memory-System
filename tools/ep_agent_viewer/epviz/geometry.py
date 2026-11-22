"""Read-only, evidence-aware geometry for the independent episode viewer.

Coordinates are the planner's metres: x, y-left, z-up. Camera extrinsics are
CARLA/Unreal (y-right), so they must never be applied directly to planner xyz.
No function opens a simulator connection, modifies an episode or runs planning.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
import sys
import threading
import types
from typing import Any

import numpy as np
from shapely.errors import ShapelyError

_ENGINE_LOCK = threading.RLock()
_ENGINES: dict[str, dict] = {}
_ROLES = ("left_front", "front", "right_front", "left_side", "right_side",
          "left_rear", "rear", "right_rear")
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7))
_ILLUSTRATIVE_HEIGHTS = {"vehicle": 1.5, "bicycle": 1.3, "pedestrian": 1.7,
                         "static_obstacle": 1.0}


def _number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _points(value, minimum=1, dimensions=2):
    try:
        result = np.asarray(value, dtype=float)
        if (result.ndim == 2 and len(result) >= minimum and
                result.shape[1] >= dimensions and np.isfinite(result).all()):
            return result
    except (TypeError, ValueError):
        pass
    return None


def _actor_id(actor):
    return str(actor.get("id", actor.get("entity_id", "")))


def _actor_position(actor):
    value = actor.get("position", actor.get("center"))
    try:
        point = np.asarray(value, float)
        return point[:2] if point.ndim == 1 and len(point) >= 2 and np.isfinite(point).all() else None
    except (TypeError, ValueError):
        return None


def _plan(frame):
    value = frame.get("plan") or {}
    return value.get("plan", value) if isinstance(value, dict) else {}


def _answer(frame):
    diagnostics = _plan(frame).get("diagnostics", {})
    if diagnostics.get("semantic_enabled") is False:
        return {}
    source = frame.get("model_source_frame_id") or (frame.get("plan_record") or {}).get("answer_source")
    if source is not None and source != frame.get("frame_id"):
        value = frame.get("active_model_output") or {}
        if not isinstance(value, dict) or str(value.get("frame_id")) != str(source):
            return {}
    else:
        value = frame.get("model_output") or {}
    if not isinstance(value, dict):
        return {}
    for key in ("answer", "output", "response"):
        if isinstance(value.get(key), dict):
            value = {**value, **value[key]}
    return value


def _attention(frame):
    result = {}
    answer = _answer(frame)
    values = answer.get("attention", answer.get("attention_objects", []))
    if not values and isinstance(answer.get("model_output"), dict):
        values = answer["model_output"].get("attention", [])
    for i, row in enumerate(values or []):
        if isinstance(row, str):
            result[row] = i + 1
        elif isinstance(row, dict):
            identity = str(row.get("id", row.get("entity_id", row.get("object_id", ""))))
            rank = _number(row.get("rank"), i + 1)
            if identity:
                result[identity] = max(1, int(rank))
    return result


def _active_goal(frame):
    diagnostics = _plan(frame).get("diagnostics", {})
    state = diagnostics.get("goal_state") or {}
    if isinstance(state.get("active_goal"), dict):
        return state["active_goal"]
    if isinstance(diagnostics.get("active_goal"), dict):
        return diagnostics["active_goal"]
    if diagnostics.get("active_goal") is False:
        return None
    answer = _answer(frame)
    if answer.get("world_goal") is not None:
        # Do not reuse an old model response as a current active planning goal.
        if answer.get("frame_id") not in (None, frame.get("frame_id")):
            return None
        return answer
    return None


def _raw_actors(frame):
    """Only same-frame raw actor records may contribute measured z/height."""
    state = frame.get("camera_state", frame.get("raw")) or {}
    if not isinstance(state, dict):
        return {}
    state_id = state.get("frame_id")
    if state_id is not None and str(state_id) != str(frame.get("frame_id")):
        return {}
    state_time = _number(state.get("time"))
    if state_time is not None and abs(state_time - float(frame.get("time", 0))) > 1e-5:
        return {}
    records = list(state.get("actors") or [])
    if isinstance(state.get("ego"), dict):
        records.append(state["ego"])
    return {_actor_id(row): row for row in records if isinstance(row, dict)}


def box_corners(center, length, width, height, yaw):
    """Eight genuine or explicitly illustrative box vertices, bottom first."""
    x, y, z = np.asarray(center, float)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rectangle = np.asarray([[-length / 2, -width / 2], [length / 2, -width / 2],
                            [length / 2, width / 2], [-length / 2, width / 2]])
    xy = rectangle @ np.asarray([[c, s], [-s, c]]) + [x, y]
    return np.vstack([np.column_stack([xy, np.full(4, z - height / 2)]),
                      np.column_stack([xy, np.full(4, z + height / 2)])])


def _objects(frame):
    plan = _plan(frame)
    selection = plan.get("diagnostics", {}).get("planning_object_selection", {})
    selected = {str(x) for x in selection.get("selected_ids", [])}
    # Decode the recorded representative IDs; never rematch by angle/position.
    role_map = {}
    assigned = selection.get("region_objects") or {}
    if isinstance(assigned, dict):
        for role in _ROLES:
            ids = assigned.get(role)
            ids = ids if isinstance(ids, list) else [ids] if ids is not None else []
            for identity in ids:
                if isinstance(identity, (str, int)) and not isinstance(identity, bool):
                    role_map.setdefault(str(identity), []).append(role)
    field = plan.get("diagnostics", {}).get("potential_field") or {}
    attention = _attention(frame)
    raw = _raw_actors(frame)
    objects, seen = [], set()
    for record in frame.get("actors") or []:
        if not isinstance(record, dict):
            continue
        identity = _actor_id(record)
        position = _actor_position(record)
        if not identity or identity in seen or position is None:
            continue
        seen.add(identity)
        measured = raw.get(identity, record)
        recorded_position = measured.get("position", measured.get("center", []))
        position_z = recorded_position[2] if isinstance(recorded_position, (list, tuple)) and len(recorded_position) >= 3 else None
        z = _number(measured.get("z", position_z))
        height = _number(measured.get("height_m", measured.get("height")))
        true_3d = z is not None and height is not None and height > 0
        kind = str(record.get("kind", "unknown"))
        if not true_3d:
            height = _ILLUSTRATIVE_HEIGHTS.get(kind, 1.)
            z = height / 2
        length = _number(record.get("length"), 0.)
        width = _number(record.get("width"), 0.)
        yaw = _number(record.get("heading", record.get("yaw")), 0.)
        if length <= 0 or width <= 0:
            continue
        corners = box_corners([*position, z], length, width, height, yaw)
        objects.append({"id": identity, "kind": kind, "center": [*position.tolist(), z],
            "length": length, "width": width, "height": height, "yaw": yaw,
            "footprint": corners[:4, :2].tolist(), "corners": corners.tolist(),
            "grid_roles": role_map.get(identity, []), "grid_matched": identity in role_map,
            "grid_match_source": "archived_planner_region_objects" if identity in role_map else None,
            "selected": identity in selected,
            "potential_source": identity in selected and bool(field),
            "potential_source_note": "规划准入的势场候选源；有效性还依赖当前/预测中心的Frenet投影",
            "source_projection_uncovered_samples": field.get("source_projection_uncovered_samples", {}).get(identity, 0),
            "attention": identity in attention, "attention_rank": attention.get(identity),
            "geometry_quality": "recorded_3d" if true_3d else "illustrative_2_5d",
            "geometry_note": "归档当前帧中心/尺寸/航向的3D框；未归档的俯仰/侧倾不重建" if true_3d else "高度未归档；2.5D示意，不投影为真实3D框"})
    return objects


def _engine(engine_root):
    """Import selected engine under a private namespace, with no bytecode writes.

    This does not install/replace ``nine_grid`` on sys.path and supports reading
    main and CARLA episodes in the same viewer process without mixing modules.
    """
    if not engine_root:
        raise ValueError("未指定匹配的主程序源码目录；车道域和势场重建不可用")
    path = Path(engine_root).expanduser().resolve()
    if (path / "src" / "nine_grid").is_dir():
        path = path / "src" / "nine_grid"
    elif (path / "nine_grid").is_dir():
        path = path / "nine_grid"
    if not (path / "contracts.py").is_file():
        raise ValueError("源码目录不包含 src/nine_grid/contracts.py")
    key = str(path)
    with _ENGINE_LOCK:
        if key not in _ENGINES:
            name = "_epviz_readonly_engine_" + hashlib.sha256(key.encode()).hexdigest()[:12]
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            sys.modules[name] = package
            saved = sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode = True
                _ENGINES[key] = {part: importlib.import_module(name + "." + part)
                    for part in ("contracts", "object_scope", "lane_roles", "lane_potential", "potential_domain")}
            finally:
                sys.dont_write_bytecode = saved
        return _ENGINES[key]


def _shape(value):
    import shapely
    from shapely.geometry import Polygon, shape
    if value is None:
        return None
    if isinstance(value, str):
        return shapely.from_wkt(value)
    if isinstance(value, dict):
        return shape(value)
    return Polygon(value)


def _scene(frame, core):
    """Rehydrate immutable snapshots from their recorded state plus static map."""
    contracts = core["contracts"]
    original = frame.get("scene") or {}
    map_data = frame.get("map") or {}
    if not isinstance(original, dict) or not isinstance(map_data, dict):
        raise ValueError("归档场景/地图不是对象")
    lanes = original.get("lanes") or map_data.get("lanes") or []
    records = []
    for row in lanes:
        polygon = _shape(row.get("polygon_wkt", row.get("polygon")))
        if polygon is None or polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        records.append(contracts.Lane(id=str(row["id"]), center=np.asarray(row["center"], float),
            polygon=np.asarray(polygon.exterior.coords, float), width=float(row["width"]),
            speed_limit=float(row.get("speed_limit", 0)), successors=row.get("successors", []),
            left=row.get("left"), right=row.get("right"), priority=row.get("priority", 0),
            boundaries=row.get("boundaries", [])))
    drivable = _shape(original.get("drivable_wkt", original.get("drivable", map_data.get("drivable"))))
    route = _points(original.get("route", map_data.get("route")), minimum=2)
    if not records or drivable is None or route is None:
        raise ValueError("缺少同帧车道地图、可行驶域或导航路径")
    def actor(row):
        if not isinstance(row, dict) or _actor_position(row) is None:
            raise ValueError("对象当前状态无效")
        names = contracts.Actor.__dataclass_fields__
        values = {key: value for key, value in row.items() if key in names}
        values.setdefault("id", _actor_id(row))
        values.setdefault("kind", "vehicle")
        values.setdefault("heading", 0.)
        for key in ("position", "velocity", "acceleration"):
            values[key] = np.asarray(row.get(key, [0., 0.]), float)[:2]
        return contracts.Actor(**values)
    ego = frame.get("ego") or original.get("ego")
    diagnostics = {**original.get("diagnostics", {})}
    # Rehydration never asserts map completeness absent from original evidence.
    return contracts.SceneSnapshot(frame_id=str(frame.get("frame_id", original.get("frame_id", ""))),
        time=float(frame.get("time", original.get("time", 0))), ego=actor(ego),
        actors=[actor(row) for row in frame.get("actors", original.get("actors", []))],
        lanes=records, route=route[:, :2],
        route_lane_ids=original.get("route_lane_ids", map_data.get("route_lane_ids", [])),
        drivable=drivable, scenario=original.get("scenario", "recorded"),
        rules=frame.get("rules", original.get("rules", {})) or {}, diagnostics=diagnostics)


def _polygons(shape):
    if shape is None or shape.is_empty:
        return []
    if shape.geom_type == "Polygon":
        return [np.asarray(shape.exterior.coords, float).tolist()]
    return [ring for part in getattr(shape, "geoms", []) for ring in _polygons(part)]


def _surfaces(shape, z=0.):
    """Polygon rings with holes retained (a ring exterior alone fills islands)."""
    if shape is None or shape.is_empty:
        return []
    if shape.geom_type == "Polygon":
        xyz = lambda ring: [[float(x), float(y), float(z)] for x, y in ring.coords]
        return [{"outer": xyz(shape.exterior), "holes": [xyz(ring) for ring in shape.interiors]}]
    return [part for child in getattr(shape, "geoms", []) for part in _surfaces(child, z)]


def _lines(shape, z=0.):
    if shape is None or shape.is_empty:
        return []
    if shape.geom_type in ("LineString", "LinearRing"):
        return [[[float(x), float(y), float(z)] for x, y in shape.coords]]
    return [line for child in getattr(shape, "geoms", []) for line in _lines(child, z)]


def _map_lanes(frame):
    return (frame.get("scene") or {}).get("lanes") or (frame.get("map") or {}).get("lanes") or []


def _road_render_geometry(frame, z):
    """Recorded union road surface and physical markings, never lane end caps.

    Small, explicitly topological gaps may receive a display-only stitch. The
    unchanged recorded road remains the clipping domain for trajectory ribbons.
    """
    import shapely
    from shapely.geometry import LineString, Polygon
    scene, map_data = frame.get("scene") or {}, frame.get("map") or {}
    lanes = {str(row.get("id")): row for row in _map_lanes(frame) if isinstance(row, dict)}
    lane_shapes = {}
    for identity, row in lanes.items():
        try:
            shape = _shape(row.get("polygon_wkt", row.get("polygon")))
            if shape is not None and not shape.is_empty and shape.is_valid:
                lane_shapes[identity] = shape
        except (ValueError, TypeError, ShapelyError):
            continue
    recorded = _shape(scene.get("drivable_wkt", scene.get("drivable", map_data.get("drivable"))))
    source = "recorded_drivable_union"
    if recorded is None or recorded.is_empty:
        recorded = shapely.union_all(list(lane_shapes.values()))
        source = "union_of_recorded_lane_polygons"
    if not recorded.is_valid:
        raise ValueError("归档路面多边形无效；不以填洞方式修正")
    connections, rejected = [], []
    display_parts = [recorded]
    max_gap, max_angle = .5, math.radians(15.)
    for identity, row in lanes.items():
        a = _points(row.get("center"), minimum=2)
        if a is None:
            continue
        for successor in row.get("successors", []):
            target = lanes.get(str(successor))
            b = _points(target.get("center"), minimum=2) if target else None
            if b is None:
                continue
            gap = float(np.linalg.norm(b[0, :2] - a[-1, :2]))
            if gap <= 1e-6:
                continue
            u, v = a[-1, :2] - a[-2, :2], b[1, :2] - b[0, :2]
            u /= max(float(np.linalg.norm(u)), 1e-9)
            v /= max(float(np.linalg.norm(v)), 1e-9)
            angle = math.acos(float(np.clip(np.dot(u, v), -1., 1.)))
            direction = b[0, :2] - a[-1, :2]
            forward = float(np.dot(direction, u)) >= -.02 and float(np.dot(direction, v)) >= -.02
            if gap > max_gap or angle > max_angle or not forward:
                rejected.append({"from_lane_id": identity, "to_lane_id": str(successor),
                    "gap_m": gap, "tangent_difference_rad": angle, "reason": "not_a_small_aligned_endpoint_gap"})
                continue
            width = min(_number(row.get("width"), 0.), _number(target.get("width"), 0.))
            if width <= 0:
                continue
            n0, n1 = np.asarray([-u[1], u[0]]), np.asarray([-v[1], v[0]])
            # An endpoint cross-section strip, not a broad hull of both lanes.
            stitch = Polygon([a[-1, :2] + n0 * width / 2, b[0, :2] + n1 * width / 2,
                              b[0, :2] - n1 * width / 2, a[-1, :2] - n0 * width / 2])
            if not stitch.is_valid or stitch.is_empty:
                continue
            addition = stitch.difference(recorded)
            if addition.area <= 1e-8:
                continue
            display_parts.append(stitch)
            connections.append({"from_lane_id": identity, "to_lane_id": str(successor),
                "gap_m": gap, "tangent_difference_rad": angle, "polygons": _surfaces(addition, z),
                "source": "recorded_successor_and_aligned_endpoints", "display_only": True,
                "note": "仅展示补缝，不修改归档路面、规划域或安全判定"})
    rendered = shapely.union_all(display_parts)
    groups, input_edges, absent = {}, 0, 0
    for identity, row in lanes.items():
        for edge in row.get("boundaries", []):
            points = _points(edge.get("points"), minimum=2)
            if points is None:
                continue
            raw_type = str(edge.get("line_type", "")).lower()
            raw_kind = str(edge.get("kind", "")).lower()
            unpainted = raw_type.split(".")[-1] in ("0", "none", "nonetype", "other", "curb", "grass")
            if (unpainted or
                    "virtual" in raw_kind or raw_kind in ("none", "unmarked")):
                absent += 1
                continue
            # Missing line paint evidence is not permission to stroke a lane
            # polygon, its end cap, or its centreline.
            if not raw_type and raw_kind not in ("lane_marking", "road_marking"):
                absent += 1
                continue
            kind = ("broken" if raw_type == "1" or "broken" in raw_type or "striped" in raw_type
                    else "solid" if raw_type == "2" or "solid" in raw_type or "continuous" in raw_type
                    else "unknown")
            color = edge.get("color", edge.get("marking_color"))
            if color is not None:
                name = str(color).lower()
                color = "#e6cb78" if "yellow" in name else "#f0f3f2" if "white" in name else color
            key = (kind, str(color), bool(edge.get("crossable", False)))
            entry = groups.setdefault(key, {"lines": [], "source_ids": [], "source_lane_ids": set(), "color": color})
            line = LineString(points[:, :2])
            if line.length <= 1e-8:
                continue
            entry["lines"].append(line)
            entry["source_ids"].append(str(edge.get("id", "")))
            entry["source_lane_ids"].add(identity)
            input_edges += 1
    markings = []
    for (kind, _, crossable), entry in groups.items():
        # GEOS nodes coincident reversed/part-overlapping edges exactly once.
        merged = shapely.line_merge(shapely.union_all(entry["lines"], grid_size=1e-5))
        for index, line in enumerate(_lines(merged, z)):
            markings.append({"id": "marking-" + str(len(markings)), "kind": kind,
                "crossable": crossable, "color": entry["color"], "points": line,
                "source_boundary_ids": entry["source_ids"], "source_lane_ids": sorted(entry["source_lane_ids"]),
                "source": "deduplicated_recorded_boundary_geometry"})
    return {"road_surface": {"polygons": _surfaces(rendered, z), "source": source,
                "recorded_polygons": _surfaces(recorded, z), "recorded_area_m2": float(recorded.area),
                "display_area_m2": float(rendered.area), "holes_preserved": True},
            "markings": markings, "connections": connections, "rejected_connections": rejected,
            "marking_diagnostics": {"input_edges": input_edges, "ignored_unmarked_or_virtual": absent,
                "rendered_lines": len(markings), "lane_polygon_end_caps_drawn": False},
            "connection_policy": {"maximum_gap_m": max_gap, "maximum_tangent_difference_rad": max_angle,
                "requires_recorded_successor": True, "unrelated_parallel_roads_never_joined": True}}, recorded


def _region_geometry(frame, engine_root):
    """Use actual lane chains and permissions, rather than angular sectors."""
    import shapely
    from shapely.geometry import LineString, Polygon
    core = _engine(engine_root)
    scene = _scene(frame, core)
    diagnostics = scene.diagnostics.get("object_scope") or {}
    settings = diagnostics.get("settings") or (frame.get("configuration") or {}).get("parameters", {}).get("planning_objects", {})
    front, rear = float(settings.get("front_m", 40.)), float(settings.get("rear_m", 20.))
    traces, roads, cache = core["object_scope"].lane_role_geometry(scene, front_m=front, rear_m=rear)
    selection = _plan(frame).get("diagnostics", {}).get("planning_object_selection", {})
    matched = selection.get("region_objects") or {}
    results = []
    for side in ("center", "left", "right"):
        trace = traces.get(side)
        if trace is None or trace.get("shape") is None:
            continue
        points = np.asarray(trace["points"], float)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if arc[-1] <= 0:
            continue
        ego_s = float(trace["ego_s"])
        ego_corners = box_corners([*scene.ego.position, 0.], scene.ego.length, scene.ego.width,
                                 1., scene.ego.heading)[:4, :2]
        ego_interval = [core["lane_roles"].project(points, point)[0] for point in ego_corners]
        # Physical lane bands also avoid painting an entire oversized junction.
        parts = [cache[key][1].intersection(cache[key][2].buffer(cache[key][0].width / 2, cap_style=2))
                 for key in trace["lane_ids"] if key in cache]
        parts += [row["shape"] for row in trace.get("navigation_connections", [])]
        shape = shapely.union_all(parts).intersection(scene.drivable)
        width = max([cache[key][0].width for key in trace["lane_ids"] if key in cache] or [4.])
        intervals = [("front", ego_s, ego_s + front), ("rear", ego_s - rear, ego_s)] if side == "center" else [
            (side + "_front", max(ego_interval), ego_s + front),
            (side + "_side", min(ego_interval), max(ego_interval)),
            (side + "_rear", ego_s - rear, min(ego_interval))]
        for role, start, end in intervals:
            start, end = max(0., start), min(float(arc[-1]), end)
            if end <= start + 1e-8:
                continue
            ss = np.r_[start, arc[(arc > start) & (arc < end)], end]
            xy = np.column_stack([np.interp(ss, arc, points[:, j]) for j in (0, 1)])
            direction = np.gradient(xy, axis=0)
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
            normal = np.column_stack([-direction[:, 1], direction[:, 0]])
            band = Polygon(np.vstack([xy + normal * width, (xy - normal * width)[::-1]])).buffer(0)
            region_shape = shape.intersection(band)
            polygons = _polygons(region_shape)
            scope_ids = [identity for identity, row in diagnostics.get("objects", {}).items()
                         if role in row.get("roles", [])]
            nearest = matched.get(role)
            identities = [] if nearest is None else ([str(nearest)] if not isinstance(nearest, list) else nearest)
            results.append({"name": role, "polygons": polygons, "object_ids": identities,
                "scope_object_ids": scope_ids, "lane_ids": trace["lane_ids"],
                "arc_interval_m": [start - ego_s, end - ego_s],
                "label_position": None if region_shape.is_empty else list(region_shape.representative_point().coords[0]),
                "source": "recorded_map_and_core_lane_chain_reconstruction",
                "note": "沿允许车道链弧长截取；当前帧几何重建，不是角度扇区"})
    return results, {"available": bool(results), "front_m": front, "rear_m": rear, "roads": roads,
                     "assignment_protocol": diagnostics.get("assignment_protocol"),
                     "source": "只读核心车道链重建"}


def _ground_height(frame):
    ego = frame.get("ego") or {}
    measured = _raw_actors(frame).get(_actor_id(ego), ego)
    z = _number(measured.get("z"))
    height = _number(measured.get("height_m", measured.get("height")))
    if z is not None and height is not None and height > 0:
        return z - height / 2, "ego_bottom_local_flat_plane_approximation"
    return 0., "illustrative_flat_plane_no_recorded_elevation"


def _roads(frame, z):
    """Normalize both WKT capture polygons and array/GeoJSON dataset polygons."""
    scene = frame.get("scene") or {}
    map_data = frame.get("map") or {}
    records = scene.get("lanes") or map_data.get("lanes") or []
    result = []
    for lane in records:
        if not isinstance(lane, dict):
            continue
        try:
            shape = _shape(lane.get("polygon_wkt", lane.get("polygon")))
            polygons = [[[x, y, z] for x, y in ring] for ring in _polygons(shape)]
            center = _points(lane.get("center"), minimum=2)
            boundaries = []
            for edge in lane.get("boundaries", []):
                points = _points(edge.get("points"), minimum=2)
                if points is not None:
                    boundaries.append({**edge, "points": [[float(p[0]), float(p[1]), z] for p in points]})
            result.append({"id": str(lane.get("id", "")), "polygons": polygons,
                "center": [] if center is None else [[float(p[0]), float(p[1]), z] for p in center],
                "boundaries": boundaries, "width": lane.get("width"),
                "geometry_quality": "recorded_xy_local_flat_elevation"})
        except (ValueError, TypeError, KeyError, ImportError, ShapelyError):
            continue
    return result


def _trajectory(frame, z):
    states = _points(_plan(frame).get("states"), minimum=1, dimensions=4)
    if states is None:
        return [], [], []
    trajectory = np.column_stack([states[:, :2], np.full(len(states), z)])
    ego = frame.get("ego") or {}
    length, width = _number(ego.get("length"), 0.), _number(ego.get("width"), 0.)
    if length <= 0 or width <= 0:
        return trajectory.tolist(), states.tolist(), []
    # Dense body rectangles plus convex inter-sample hulls form a geometric
    # swept footprint. It is neither a collision check nor a safe-drivable area.
    from shapely.geometry import Polygon
    import shapely
    boxes = [Polygon(box_corners([*state[:2], z], length, width, 1., state[2])[:4, :2]) for state in states]
    parts = boxes + [shapely.union_all([a, b]).convex_hull for a, b in zip(boxes, boxes[1:])]
    corridor = _polygons(shapely.union_all(parts))
    return trajectory.tolist(), states.tolist(), [[[x, y, z] for x, y in ring] for ring in corridor]


def _ribbon(frame, trajectory, z, recorded_road=None, display_margin_m=0.):
    """A continuous trajectory-centred left/right strip, not body convex hulls."""
    import shapely
    from shapely.geometry import LineString
    margin = _number(display_margin_m)
    width = _number((frame.get("ego") or {}).get("width"))
    if margin is None or not 0 <= margin <= 5.:
        raise ValueError("展示走廊附加半宽必须在0至5米之间")
    if width is None or width <= 0 or len(trajectory) < 2:
        return {"available": False, "left": [], "right": [], "polygons": [],
                "reason": "缺少有效自车宽度或两点以上轨迹"}
    xy = np.asarray(trajectory, float)[:, :2]
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-7]]
    if len(xy) < 2:
        return {"available": False, "left": [], "right": [], "polygons": [], "reason": "轨迹只有停车点"}
    line = LineString(xy)
    half_width = width / 2 + margin
    left = line.offset_curve(half_width, quad_segs=12, join_style=1)
    right = line.offset_curve(-half_width, quad_segs=12, join_style=1)
    strip = line.buffer(half_width, quad_segs=12, cap_style=2, join_style=1)
    raw_area = float(strip.area)
    clipped = recorded_road is not None and not recorded_road.is_empty
    if clipped:
        strip = strip.intersection(recorded_road)
    left_parts, right_parts = _lines(left, z), _lines(right, z)
    clipped_left = left.intersection(recorded_road) if clipped else left
    clipped_right = right.intersection(recorded_road) if clipped else right
    return {"available": not strip.is_empty, "left": max(left_parts, key=len, default=[]),
        "right": max(right_parts, key=len, default=[]),
        "left_segments": _lines(clipped_left, z), "right_segments": _lines(clipped_right, z),
        "polygons": _surfaces(strip, z), "unclipped_polygons": _surfaces(line.buffer(half_width,
            quad_segs=12, cap_style=2, join_style=1), z),
        "half_width_m": half_width, "vehicle_width_m": width, "display_margin_m": margin,
        "clipped_to_recorded_road": clipped, "clipped_area_m2": max(0., raw_area - float(strip.area)),
        "valid": bool(strip.is_valid), "trajectory_simple": bool(line.is_simple),
        "source": "continuous_offset_strip_of_recorded_plan_positions",
        "note": "轨迹左右连续带；半宽=车宽/2+明确展示余量，默认0；按原始归档道路裁剪。不是车体扫掠或已通过安全验收的行驶区"}


def _camera_matrix(camera):
    matrix = np.asarray(camera.get("world_to_camera_ue"), float)
    width = _number(camera.get("width"))
    height = _number(camera.get("height"))
    fov = _number(camera.get("fov_deg", camera.get("fov")))
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("缺少有效world_to_camera_ue标定")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4) or not np.allclose(
            matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-3):
        raise ValueError("world_to_camera_ue不是有效刚体外参")
    if width is None or height is None or width <= 0 or height <= 0 or fov is None or not 0 < fov < 180:
        raise ValueError("缺少有效图像尺寸/水平视场角")
    return matrix, width, height, width / (2 * math.tan(math.radians(fov) / 2))


def _camera_point(point, matrix):
    xyz = np.asarray(point, float)
    return (matrix @ np.asarray([xyz[0], -xyz[1], xyz[2], 1.]))[:3]


def _pixel(point, width, height, focal):
    return np.asarray([width / 2 + focal * point[1] / point[0],
                       height / 2 - focal * point[2] / point[0]])


def project_world_point(point, camera, near=0.1, clip_to_image=True):
    """Project planner xyz through archived CARLA UE calibration, or return None."""
    matrix, width, height, focal = _camera_matrix(camera)
    p = _camera_point(point, matrix)
    if p[0] < near or not np.isfinite(p).all():
        return None
    pixel = _pixel(p, width, height, focal)
    if clip_to_image and not (0 <= pixel[0] <= width and 0 <= pixel[1] <= height):
        return None
    return pixel.tolist()


def _clip_image(a, b, width, height, return_parameters=False):
    # Liang-Barsky clipping keeps edges crossing the image even with both
    # endpoints offscreen; unlike endpoint clamping it preserves line geometry.
    d = b - a
    lo, hi = 0., 1.
    for p, q in ((-d[0], a[0]), (d[0], width - a[0]), (-d[1], a[1]), (d[1], height - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            lo = max(lo, r)
        else:
            hi = min(hi, r)
        if lo > hi:
            return None
    points = [(a + lo * d).tolist(), (a + hi * d).tolist()]
    return (points, (lo, hi)) if return_parameters else points


def _project_segment_progress(a, b, camera, near=0.1):
    """Return pixels and original world-segment fractions after both clips.

    Screen-space clipping fractions are perspective distorted; convert them
    back through camera depth before using them for planning-time colours.
    """
    matrix, width, height, focal = _camera_matrix(camera)
    a, b = _camera_point(a, matrix), _camera_point(b, matrix)
    if not np.isfinite(np.r_[a, b]).all() or (a[0] < near and b[0] < near):
        return None
    start, end = 0., 1.
    if a[0] < near:
        start = (near - a[0]) / (b[0] - a[0])
        a = a + (b - a) * start
    elif b[0] < near:
        end = (near - a[0]) / (b[0] - a[0])
        b = a + (b - a) * end
    clipped = _clip_image(_pixel(a, width, height, focal), _pixel(b, width, height, focal),
                          width, height, return_parameters=True)
    if clipped is None:
        return None
    pixels, screen_fractions = clipped
    fractions = []
    for fraction in screen_fractions:
        world_fraction = fraction * a[0] / ((1. - fraction) * b[0] + fraction * a[0])
        fractions.append(float(start + (end - start) * world_fraction))
    return pixels, fractions


def project_segment(a, b, camera, near=0.1):
    """Clip in camera depth before dividing by it, then clip to the image."""
    projected = _project_segment_progress(a, b, camera, near)
    return projected[0] if projected is not None else None


def _project_polygon(vertices, camera, near=0.1):
    """Camera-plane polygon clipping followed by rectangular viewport clipping."""
    matrix, width, height, focal = _camera_matrix(camera)
    points = [_camera_point(p, matrix) for p in vertices]
    if len(points) < 3:
        return []
    clipped = []
    for a, b in zip(points, points[1:] + points[:1]):
        inside_a, inside_b = a[0] >= near, b[0] >= near
        if inside_a:
            clipped.append(a)
        if inside_a != inside_b:
            clipped.append(a + (b - a) * ((near - a[0]) / (b[0] - a[0])))
    if len(clipped) < 3:
        return []
    from shapely.geometry import Polygon, box
    projected = Polygon([_pixel(p, width, height, focal) for p in clipped]).buffer(0)
    return _polygons(projected.intersection(box(0, 0, width, height)))


def _alignment_reason(frame, camera):
    if camera.get("aligned") is False:
        return "图像与当前状态未严格对齐，停止叠加"
    if camera.get("calibrated") is False:
        return "缺少同帧原始状态或相机标定；仅显示原图"
    camera_id = camera.get("frame_id")
    camera_time = _number(camera.get("time"))
    if camera_id is None or str(camera_id) != str(frame.get("frame_id")):
        return "缺少与当前状态相同frame_id的图像证据"
    if camera_time is None or abs(camera_time - float(frame.get("time", 0))) > 1e-5:
        return "图像源时间与当前状态不一致"
    state = frame.get("camera_state", frame.get("raw")) or {}
    raw_frame = state.get("frame", state.get("carla_frame")) if isinstance(state, dict) else None
    camera_frame = camera.get("frame", camera.get("carla_frame"))
    if raw_frame is not None and camera_frame is not None and str(raw_frame) != str(camera_frame):
        return "相机原始帧号与当前观测状态不一致"
    return None


def _camera_overlays(frame, visualization):
    result = {}
    for camera in frame.get("cameras") or []:
        if not isinstance(camera, dict):
            continue
        view = str(camera.get("view", camera.get("name", "camera")))
        overlay = {"valid": False, "reason": None, "frame_id": camera.get("frame_id"),
            "time": camera.get("time"), "width": camera.get("width"), "height": camera.get("height"),
            "boxes": [], "trajectory_segments": [], "trajectory_segments_progress": [],
            "trajectory_samples": [], "trajectory_source_point_count": 0, "corridor_polygons": [],
            "region_polygons": [], "goal": None}
        result[view] = overlay
        reason = _alignment_reason(frame, camera)
        if reason:
            overlay["reason"] = reason
            continue
        try:
            _camera_matrix(camera)
            for obj in visualization["objects"]:
                if obj["geometry_quality"] != "recorded_3d":
                    continue
                corners = obj["corners"]
                segments = [value for a, b in _EDGES if
                            (value := project_segment(corners[a], corners[b], camera)) is not None]
                if segments:
                    overlay["boxes"].append({"id": obj["id"], "kind": obj["kind"], "segments": segments,
                        "attention": obj["attention"], "attention_rank": obj["attention_rank"],
                        "potential_source": obj["potential_source"], "geometry_quality": "recorded_3d",
                        "grid_matched": obj.get("grid_matched", False), "grid_roles": obj.get("grid_roles", [])})
            if visualization["ground_projection_quality"] != "illustrative_flat_plane_no_recorded_elevation":
                trajectory = visualization["trajectory"]
                count = len(trajectory)
                denominator = max(count - 1, 1)
                overlay["trajectory_source_point_count"] = count
                for index, (a, b) in enumerate(zip(trajectory, trajectory[1:])):
                    projected = _project_segment_progress(a, b, camera)
                    if projected is not None:
                        pixels, fractions = projected
                        overlay["trajectory_segments"].append(pixels)
                        overlay["trajectory_segments_progress"].append(
                            [(index + fraction) / denominator for fraction in fractions])
                for index, point in enumerate(trajectory):
                    pixel = project_world_point(point, camera)
                    if pixel is not None:
                        overlay["trajectory_samples"].append(
                            {"point": pixel, "index": index, "progress": index / denominator})
                overlay["corridor_polygons"] = [polygon for ring in visualization["corridor"]
                    for polygon in _project_polygon(ring, camera)]
                z = visualization["ground_z_m"]
                for region in visualization["regions"]:
                    polygons = [polygon for ring in region["polygons"]
                        for polygon in _project_polygon([[x, y, z] for x, y in ring], camera)]
                    overlay["region_polygons"].append({"name": region["name"], "polygons": polygons})
                if visualization["goal"] is not None:
                    overlay["goal"] = project_world_point(visualization["goal"], camera)
            overlay["valid"] = True
            overlay["projection_note"] = "对象框依赖归档三维几何与同帧标定；道路/轨迹按自车底面局部平面近似，坡道投影未验证；未做图像遮挡推断"
        except (ValueError, TypeError, KeyError, ImportError, ShapelyError) as exc:
            overlay["reason"] = str(exc)
    return result


def enrich_frame(frame: dict, engine_root=None, display_margin_m=0.) -> dict:
    """Return a shallow copy with display-only geometry; leave source untouched."""
    result = dict(frame)
    warnings = []
    objects = _objects(frame)
    z, ground_quality = _ground_height(frame)
    trajectory, states, body_sweep = _trajectory(frame, z)
    try:
        render_geometry, recorded_road = _road_render_geometry(frame, z)
    except (ValueError, TypeError, KeyError, ShapelyError) as exc:
        render_geometry, recorded_road = {"road_surface": {"polygons": []}, "markings": [], "warnings": [str(exc)]}, None
        warnings.append("道路展示几何不可重建：" + str(exc))
    ribbon = _ribbon(frame, trajectory, z, recorded_road, display_margin_m)
    corridor = [part["outer"] for part in ribbon["polygons"]]
    goal_record = _active_goal(frame)
    goal = None
    if goal_record is not None:
        xy = _actor_position({"position": goal_record.get("world_goal")})
        if xy is not None:
            goal = [*xy.tolist(), z]
    try:
        regions, region_status = _region_geometry(frame, engine_root)
    except (ImportError, ValueError, TypeError, KeyError, AttributeError, IndexError, ShapelyError) as exc:
        regions, region_status = [], {"available": False, "reason": str(exc)}
        warnings.append("八向车道区域不可重建：" + str(exc))
    if any(obj["geometry_quality"] == "illustrative_2_5d" for obj in objects):
        warnings.append("部分对象缺少三维高度/中心z；只显示2.5D示意，不向相机投影真实3D框")
    visualization = {"objects": objects, "regions": regions, "region_status": region_status,
        "roads": _roads(frame, z),
        "render_geometry": render_geometry, "corridor_ribbon": ribbon,
        "body_sweep": {"polygons": body_sweep, "source": "recorded_body_rectangles_and_adjacent_convex_hulls",
                       "displayed_as_corridor": False, "note": "离散车体扫掠证据；不作为展示行车走廊"},
        "trajectory": trajectory, "trajectory_states": states, "corridor": corridor,
        "corridor_note": "沿实际轨迹的左右连续带，默认半宽为自车车宽/2，按原归档路面裁剪；不代表通过安全验证的可行驶区域",
        "goal": goal, "ground_z_m": z, "ground_projection_quality": ground_quality,
        "coordinates": "world_x_y_left_z_up_m", "status": "available", "warnings": warnings}
    result["visualization"] = visualization
    result["camera_overlays"] = _camera_overlays(frame, visualization)
    result["potential"] = {"available": False, "deferred": True,
        "reason": "势场按需从归档版本和匹配核心公式重建"}
    return result


def build_potential(frame: dict, engine_root=None, resolution=None) -> dict:
    """Reconstruct the current-frame lane field using the selected engine kernel.

    This is a display reconstruction of a spatial slice, not an archived MPC
    future cost tensor. Repulsive sources use their recorded current centres;
    attraction shows the active goal's valley, whose planner cost is sampled at
    the first-arrival event. An absent goal creates no artificial attraction.
    """
    unavailable = {"available": False, "source": "display_reconstruction", "values": []}
    try:
        plan = _plan(frame)
        diagnostics = plan.get("diagnostics") or {}
        evidence = diagnostics.get("potential_field") or {}
        if evidence.get("version") != "lane-frenet-center-ellipse-1.0":
            raise ValueError("归档未提供本版Frenet椭圆势场证据；拒绝用当前公式冒充旧版结果")
        if evidence.get("geometry", {}).get("version") != "lane-frenet-centre-potential-1.0":
            raise ValueError("归档势场坐标几何版本不受支持")
        classes = evidence.get("classes")
        selection = diagnostics.get("planning_object_selection") or {}
        if not isinstance(classes, dict) or "selected_ids" not in selection:
            raise ValueError("缺少归档类别势场参数或实际规划对象清单")
        if any(key not in evidence for key in ("attention_increment", "outer_risk_weight")):
            raise ValueError("缺少归档关注增益或外层risk权重，不能猜测实际势场代价")
        core = _engine(engine_root)
        if core["lane_potential"].VERSION != evidence["geometry"]["version"]:
            raise ValueError("匹配核心势场坐标版本与归档不一致")
        scene = _scene(frame, core)
        domain_meta = evidence.get("domain") or {}
        domain, rebuilt = core["potential_domain"].build_lane_potential_domain(scene,
            front_m=float(domain_meta.get("map_extension_front_m", 120.)),
            rear_m=float(domain_meta.get("map_extension_rear_m", 20.)))
        if rebuilt.get("version") != domain_meta.get("version"):
            raise ValueError("核心车道域版本与归档不匹配：" + str(rebuilt.get("version")) + " / " + str(domain_meta.get("version")))
        if set(rebuilt.get("lane_ids", [])) != set(domain_meta.get("lane_ids", [])):
            raise ValueError("重建车道域与归档车道清单不一致；请提供完整同帧场景和匹配源码")
        def connections(meta):
            return {(row.get("from_lane_id"), row.get("to_lane_id")) for row in meta.get("navigation_connections", [])}
        if connections(rebuilt) != connections(domain_meta):
            raise ValueError("重建导航连接段与归档不一致，不能把不同支路视为同一势场域")
        goal = _active_goal(frame)
        reference = scene.route
        if goal is not None:
            frozen = _points((goal.get("goal_reference") or {}).get("points"), minimum=2)
            if frozen is None:
                raise ValueError("活动引导点缺少冻结参考线，不能替换为另一条导航参考")
            reference = frozen[:, :2]
        geometry = core["lane_potential"].LanePotentialGeometry(reference, domain)
        archived_length = _number(evidence.get("geometry", {}).get("reference_length_m"))
        if archived_length is not None and not math.isclose(geometry.length, archived_length, abs_tol=1e-4, rel_tol=1e-7):
            raise ValueError("重建参考线弧长与归档不一致，不能替换原始势场参考")
        polyline_s = (goal.get("last_s") if goal is not None else scene.diagnostics.get("route_s"))
        if polyline_s is None:
            polyline_s = core["lane_roles"].project(reference, scene.ego.position)[0]
        anchor = float(geometry.chart_arc_from_polyline_s(float(polyline_s)))
        geometry.anchor_s = anchor
        attention = _attention(frame)
        increment = float(evidence["attention_increment"])
        outer_weight = float(evidence["outer_risk_weight"])
        if not np.isfinite([increment, outer_weight]).all() or increment < 0 or outer_weight < 0:
            raise ValueError("归档关注增益/外层risk权重无效")
        actors = {actor.id: actor for actor in scene.actors}
        sources, source_data = [], []
        display_front, display_rear = 70., 30.
        for identity in selection["selected_ids"]:
            identity = str(identity)
            actor = actors.get(identity)
            if actor is None or actor.kind not in classes:
                raise ValueError("实际势场对象缺少当前状态/类别参数：" + identity)
            memberships = selection.get("selected_state", {}).get(identity, {}).get("memberships", {})
            if not memberships:
                raise ValueError("实际规划源缺少车道链投影锚点：" + identity)
            offset = min(memberships.values(), key=lambda row: abs(row["delta_s_m"]))["delta_s_m"]
            source_anchor = float(np.clip(anchor + float(offset), 0., geometry.length))
            source = geometry.project(np.asarray(actor.position).reshape(1, 2), anchors=source_anchor,
                require_domain=False, window_m=min(20., max(10., 5. + .25 * abs(offset))))
            gain = 1. + increment / max(1, attention[identity]) if identity in attention else 1.
            effective = bool(source["coordinate_valid"][0])
            if effective:
                source_data.append((source, classes[actor.kind], gain))
                # Do not confuse the 40/20 object-admission boundary with the
                # support of an admitted source's repulsive field.
                delta = float(source["s"][0] - anchor)
                radius = float(classes[actor.kind]["longitudinal_m"])
                display_front = max(display_front, delta + radius)
                display_rear = max(display_rear, -delta + radius)
            sources.append({"id": identity, "kind": actor.kind, "attention_gain": gain,
                "settings": classes[actor.kind], "position": actor.position.tolist(), "effective": effective,
                "projection_uncovered_future_samples": evidence.get("source_projection_uncovered_samples", {}).get(identity, 0)})
        # The display window is independent of object admission. It only crops
        # the viewport; the actual recorded planner domain remains unchanged.
        from shapely.geometry import LineString
        start, end = max(0., anchor - display_rear), min(geometry.length, anchor + display_front)
        ss = np.linspace(start, end, max(3, int((end - start) / .25) + 1))
        center = geometry.sample(ss, 0.)
        padding = max([float(lane.width) for lane in scene.lanes] or [4.]) * 2 + .5
        window = LineString(center).buffer(padding, cap_style=2).intersection(domain)
        if window.is_empty:
            raise ValueError("实际势场车道域与前70/后30米展示窗口无交集")
        bounds = window.bounds
        lo, hi = np.asarray(bounds[:2]) - .5, np.asarray(bounds[2:]) + .5
        size = np.maximum(hi - lo, 1.)
        # Match the previous scientific plot's default 0.2m raster spacing.
        # Explicit resolution remains useful for inexpensive offline tests.
        longest = max(220, min(400, int(math.ceil(float(size.max()) / .2)) + 1)) if resolution is None else int(resolution)
        longest = max(32, min(400, longest))
        width = max(24, min(400, int(math.ceil(longest * size[0] / size.max()))))
        height = max(24, min(400, int(math.ceil(longest * size[1] / size.max()))))
        if width * height > 120000:
            factor = math.sqrt(120000. / (width * height))
            width, height = int(width * factor), int(height * factor)
        xs, ys = np.linspace(lo[0], hi[0], width), np.linspace(lo[1], hi[1], height)
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        query = geometry.project(points, anchors=anchor)
        valid = query["valid"] & (query["s"] >= start - 1e-7) & (query["s"] <= end + 1e-7)
        repulsion = np.zeros(len(points))
        for source, class_settings, gain in source_data:
            field = geometry.repulsion_from_projection(query, source, class_settings, attention_gain=gain)
            repulsion += np.nan_to_num(field["energy"], nan=0.)
        attraction = np.zeros(len(points))
        goal_xy = None
        if goal is not None:
            settings = (frame.get("configuration") or {}).get("parameters", {}).get("local_goal")
            if not isinstance(settings, dict) or any(key not in settings for key in
                    ("attraction_weight", "longitudinal_scale_m", "lateral_scale_m", "potential_shift")):
                raise ValueError("活动目标缺少归档吸引势场参数")
            settings = {**settings, "shift": settings.get("potential_shift", 6.)}
            goal_xy = goal.get("world_goal")
            field = geometry.attraction(points, goal_xy, settings, anchors=anchor,
                goal_anchor=float(geometry.chart_arc_from_polyline_s(goal.get("s_goal_m", polyline_s))))
            attraction = field["energy"]
            valid &= field["valid"]
        energy = repulsion + attraction
        values = np.where(valid, energy, np.nan).reshape(height, width)
        weighted = np.where(valid, outer_weight * repulsion + attraction, np.nan).reshape(height, width)
        finite = values[np.isfinite(values)]
        json_matrix = lambda value: [[float(x) if math.isfinite(x) else None for x in row] for row in value]
        return {"available": True, "frame_id": frame.get("frame_id"), "time": frame.get("time"),
            "width": width, "height": height, "xs": xs.tolist(), "ys": ys.tolist(),
            "bounds": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
            "values": json_matrix(values), "weighted_values": json_matrix(weighted),
            "attraction_values": json_matrix(np.where(valid, attraction, np.nan).reshape(height, width)),
            "repulsion_values": json_matrix(np.where(valid, repulsion, np.nan).reshape(height, width)),
            "domain_mask": query["domain_valid"].reshape(height, width).tolist(),
            "display_mask": valid.reshape(height, width).tolist(),
            "display_window": {"front_m": display_front, "rear_m": display_rear,
                "s_start_m": start, "s_end_m": end, "object_admission_unchanged": True,
                "planner_domain_unchanged": True},
            "grid_spacing_m": [float(xs[1] - xs[0]), float(ys[1] - ys[0])],
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "theory": evidence["version"], "domain_version": rebuilt["version"],
            "source": "display_reconstruction", "unit": "relative_potential",
            "coordinates": "world_x_y_left_m", "goal": goal_xy, "sources": sources,
            "attraction_active": goal_xy is not None, "empty_source_field": not source_data and goal_xy is None,
            "field_definition": "raw Uatt + sum(Urep); attention gains retained; no outer MPC risk weight",
            "weighted_definition": "Uatt + archived_outer_risk_weight * sum(Urep)",
            "outer_risk_weight": outer_weight, "domain_polygons": _surfaces(domain),
            "colormap": "viridis", "normalization": "linear_shared_negative_valley_positive_hills",
            "note": "当前帧Frenet势场重建；默认原始Uatt+ΣUrep，与plot_lane_potential同定义/线性viridis色系；另保留weighted_values供核对MPC外层risk权重。道路域外未定义(null)，显示前70/后30米且必要时覆盖源的完整作用范围，不扩大原规划域。无活动模型目标不虚构吸引谷。实际MPC吸引势在首次到达事件取样，本图不等同未来预测代价或安全证明"}
    except (ImportError, ValueError, TypeError, KeyError, AttributeError, IndexError, ShapelyError) as exc:
        return {**unavailable, "reason": str(exc)}
