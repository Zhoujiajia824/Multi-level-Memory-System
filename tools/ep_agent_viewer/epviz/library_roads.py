"""Local, evidence-aware map display for immutable offline memory snapshots.

Painted road markings and lane topology are deliberately separate. A numeric
map boundary code with unknown semantics is not evidence of a white solid line.
No polygon is buffered into a replacement lane, and no topology edge adds road.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import shapely
from shapely.errors import ShapelyError
from shapely.geometry import LineString, box

from . import geometry


def _paint(edge, scenario, lane_kind):
    raw = str(edge.get("line_type", "")).lower().split(".")[-1]
    kind = str(edge.get("kind", "")).lower()
    permission = str(edge.get("permission_source", "")).lower()
    source = str(edge.get("source", "")).lower()
    if "unknown_boundary" in permission or "virtual" in kind or kind in ("unmarked", "none"):
        return None, "unknown_or_virtual_boundary"
    if lane_kind == "connector" and source == "nuplan_map_boundary":
        return None, "connector_boundary_not_painted_marking"
    if raw in ("", "0", "none", "nonetype", "unknown", "other", "curb", "grass"):
        return None, "unmarked_or_unknown_type"
    if "solid" in raw and ("dash" in raw or "broken" in raw):
        return None, "mixed_pattern_not_representable_as_one_line"
    if any(value in raw for value in ("dash", "broken", "striped")):
        return "broken", None
    if any(value in raw for value in ("solid", "continuous")):
        return "solid", None
    # These codes have an identified highway-env LineType context. The same
    # integers in nuPlan GPKG are deliberately not interpreted by this rule.
    if source.startswith("highway") or scenario in ("highway", "highway-env", "merge", "roundabout", "intersection", "parking"):
        if raw == "1":
            return "broken", None
        if raw in ("2", "3"):
            return "solid", None
    return None, "unverified_numeric_or_unknown_marking_type"


def build_library_roads(scene, ground_z=0., half_extent_m=80.):
    """Return ``(render_geometry, original_road_shape)`` from recorded geometry.

    The map is clipped only for display. The unchanged original road remains
    available to clip the separately labelled recorded-execution ribbon.
    """
    half_extent_m = float(half_extent_m)
    if not math.isfinite(half_extent_m) or half_extent_m <= 0:
        raise ValueError("地图展示范围必须是正数")
    ego = geometry._actor_position(scene.get("ego") or {})
    if ego is None:
        raise ValueError("缺少自车世界坐标，不能界定局部地图")
    viewport = box(ego[0] - half_extent_m, ego[1] - half_extent_m,
                   ego[0] + half_extent_m, ego[1] + half_extent_m)
    diagnostic = scene.get("diagnostics") or {}
    diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
    lane_meta = diagnostic.get("lane_source_metadata") or {}
    lane_meta = lane_meta if isinstance(lane_meta, dict) else {}
    lanes = {str(row["id"]): row for row in scene.get("lanes", [])
             if isinstance(row, dict) and row.get("id") is not None}
    shapes, invalid_shapes = {}, []
    bbox_count, estimated_widths = 0, []
    for identity, lane in lanes.items():
        try:
            polygon = geometry._shape(lane.get("polygon_wkt", lane.get("polygon")))
            if polygon is not None and not polygon.is_empty and polygon.is_valid:
                shapes[identity] = polygon
                bbox_count += polygon.geom_type == "Polygon" and len(polygon.exterior.coords) == 5 and polygon.equals(polygon.envelope)
            elif polygon is not None:
                invalid_shapes.append(identity)
        except (ValueError, TypeError, ShapelyError):
            invalid_shapes.append(identity)
        width = geometry._number(lane.get("width"))
        if width is not None and width > 8.:
            estimated_widths.append({"lane_id": identity, "archived_width_m": width})
    road = geometry._shape(scene.get("drivable_wkt", scene.get("drivable")))
    source = "recorded_drivable_polygon"
    if road is None or road.is_empty:
        road = shapely.union_all(list(shapes.values()))
        source = "union_of_recorded_lane_polygons"
    if not road.is_valid:
        raise ValueError("原始路面多边形无效，不以凸包或填洞修复")
    local = road.intersection(viewport)

    paint_groups, ignored = {}, Counter()
    input_boundaries = 0
    for identity, lane in lanes.items():
        lane_kind = (lane_meta.get(identity) or {}).get("kind", "unknown")
        for edge in lane.get("boundaries") or []:
            if not isinstance(edge, dict):
                ignored["invalid_boundary"] += 1; continue
            input_boundaries += 1
            kind, reason = _paint(edge, str(scene.get("scenario", "")).lower(), lane_kind)
            if kind is None:
                ignored[reason] += 1; continue
            points = geometry._points(edge.get("points"), minimum=2)
            if points is None:
                ignored["missing_boundary_geometry"] += 1; continue
            line = LineString(points[:, :2]).intersection(viewport)
            if line.is_empty:
                ignored["outside_display_window"] += 1; continue
            color = edge.get("color", edge.get("marking_color"))
            raw_type = str(edge.get("line_type", ""))
            if color is None and any(word in raw_type.lower() for word in ("white", "yellow")):
                color = "yellow" if "yellow" in raw_type.lower() else "white"
            if color is not None:
                name = str(color).lower()
                color = "#e6cb78" if "yellow" in name else "#f0f3f2" if "white" in name else color
            crossable = edge.get("crossable")
            key = (kind, str(color), crossable)
            group = paint_groups.setdefault(key, {"lines": [], "boundary_ids": set(), "lane_ids": set(),
                                                   "line_types": set(), "color": color})
            group["lines"].append(line)
            group["boundary_ids"].add(str(edge.get("id", "")))
            group["lane_ids"].add(identity); group["line_types"].add(raw_type)
    markings = []
    for (kind, _, crossable), group in paint_groups.items():
        merged = shapely.line_merge(shapely.union_all(group["lines"], grid_size=1e-5))
        for path in geometry._lines(merged, ground_z):
            markings.append({"id": "marking-" + str(len(markings)), "kind": kind,
                "points": path, "color": group["color"], "crossable": crossable,
                "line_types": sorted(group["line_types"]),
                "source_boundary_ids": sorted(group["boundary_ids"])[:16],
                "source_boundary_count": len(group["boundary_ids"]),
                "source": "verified_recorded_marking_geometry"})

    edges, connections, unresolved, missing_targets = [], [], [], 0
    for identity, lane in lanes.items():
        center = geometry._points(lane.get("center"), minimum=2)
        if center is None:
            continue
        kind = (lane_meta.get(identity) or {}).get("kind", "unknown")
        clipped = LineString(center[:, :2]).intersection(viewport)
        for part, path in enumerate(geometry._lines(clipped, ground_z)):
            edges.append({"id": identity + ":" + str(part), "lane_id": identity, "kind": kind,
                "path": path, "successors": [str(value) for value in lane.get("successors") or []],
                "source": "recorded_lane_centerline", "display_only": True})
        for successor in lane.get("successors") or []:
            target = lanes.get(str(successor))
            following = geometry._points(target.get("center"), minimum=2) if target else None
            if following is None:
                missing_targets += 1; continue
            gap = float(np.linalg.norm(center[-1, :2] - following[0, :2]))
            if not viewport.covers(LineString([center[-1, :2], following[0, :2]]).envelope):
                continue
            item = {"from_lane_id": identity, "to_lane_id": str(successor), "gap_m": gap}
            if gap <= 1e-5:
                connections.append({**item, "path": [[*center[-1, :2].tolist(), ground_z],
                    [*following[0, :2].tolist(), ground_z]], "status": "coincident_recorded_endpoints"})
            else:
                unresolved.append({**item, "status": "recorded_endpoints_do_not_meet", "drawn": False})
    routes = []
    navigation = geometry._points(scene.get("route"), minimum=2)
    if navigation is not None:
        for path in geometry._lines(LineString(navigation[:, :2]).intersection(viewport), ground_z):
            routes.append({"path": path, "kind": "recorded_navigation_reference",
                           "source": "scene_route_not_execution_or_planned_trajectory", "display_only": True})
    quality = {"lane_map_complete": diagnostic.get("lane_map_complete"),
        "lane_width_source": diagnostic.get("lane_width_source"),
        "drivable_source": diagnostic.get("drivable_source"), "archived_lane_count": len(lanes),
        "axis_aligned_rectangle_lane_count": int(bbox_count), "wide_archived_lane_values": estimated_widths,
        "invalid_lane_polygon_ids": invalid_shapes,
        "note": "只裁剪存档路面；中心拓扑为示意，不代表实漆线。矩形/面积除中心线所得宽度不可视为精确车道边界"}
    return {"road_surface": {"polygons": geometry._surfaces(local, ground_z), "source": source,
            "recorded_area_m2": float(road.area), "display_area_m2": float(local.area),
            "holes_preserved": True, "recorded_surface_only": True},
        "markings": markings, "connections": [], "rejected_connections": unresolved,
        "topology": {"edges": edges, "connections": connections, "routes": routes,
            "unresolved_connections": unresolved, "targets_outside_recorded_local_map": missing_targets,
            "note": "独立中心线拓扑示意；不补绘物理道路、不当作规划轨迹或车道漆线"},
        "display_domain": {"bounds": list(viewport.bounds), "half_extent_m": half_extent_m,
            "source": "display_window_only_not_planning_or_object_matching_scope"},
        "map_quality": quality,
        "marking_diagnostics": {"input_edges": input_boundaries, "rendered_lines": len(markings),
            "ignored_by_reason": dict(ignored), "lane_polygon_end_caps_drawn": False},
        "connection_policy": {"display_stitches_enabled": False, "arbitrary_centerline_buffers": False,
            "note": "中心线后继只用于拓扑标注，不生成连接路面"}}, road
