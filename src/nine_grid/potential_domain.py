"""势场只定义在导航一致的当前车道及可跨越邻道，不把整个路口当作走廊。"""
from __future__ import annotations

import shapely

from .object_scope import lane_role_geometry


def _area_components(geometry):
    """Keep intersection faces; a touching edge cannot authorize driving area."""
    pending = [geometry]
    polygons = []
    discarded = 0
    while pending:
        part = pending.pop()
        if part.is_empty:
            continue
        if part.geom_type == 'Polygon':
            polygons.append(part)
        elif part.geom_type in ('MultiPolygon', 'GeometryCollection'):
            pending.extend(part.geoms)
        else:
            discarded += 1
    return shapely.union_all(polygons), discarded


def build_lane_potential_domain(scene, front_m=120., rear_m=20.):
    """返回世界坐标车道域与证据；量程仅用于延伸地图，不改变对象40/20米准入。"""
    traces, permissions, cache = lane_role_geometry(scene, front_m=front_m, rear_m=rear_m)
    parts = []
    selected = set()
    for trace in traces.values():
        if trace is None:
            continue
        for identity in trace['lane_ids']:
            if identity in selected or identity not in cache:
                continue
            lane, polygon, center = cache[identity]
            # 宽大的交叉口地图面不能授权横向支路。保留真实多边形与车道宽带的交集。
            part = polygon.intersection(center.buffer(float(lane.width)/2, cap_style=2))
            if not part.is_empty:
                parts.append(part)
                selected.add(identity)
    if not parts:
        raise ValueError('势场缺少已确认的当前车道几何，不能使用整个可行驶地图代替')
    domain = shapely.union_all(parts).intersection(scene.drivable)
    if domain.is_empty or not domain.is_valid:
        raise ValueError('导航车道势场域为空或几何无效')
    # Real lane boundaries may only touch other drivable polygons. Shapely then
    # returns area plus isolated lines/points. Preserve exactly the area, without
    # buffering or expanding the permitted lane domain.
    domain, discarded = _area_components(domain)
    if domain.is_empty or not domain.is_valid:
        raise ValueError('导航车道势场域缺少有效二维车道面')
    return domain, {
        'version': 'lane-potential-domain-1.1',
        'lane_ids': sorted(selected),
        'chains': {side: None if trace is None else trace['lane_ids'] for side, trace in traces.items()},
        'adjacent_permissions': permissions,
        'geometry': 'area_only(union(actual_lane_polygon intersect centerline_half_width_band) intersect drivable)',
        'discarded_non_area_components': discarded,
        'outside': 'undefined; rejected_by_domain_constraint',
        'map_extension_front_m': float(front_m), 'map_extension_rear_m': float(rear_m),
        'object_admission_unchanged': True,
    }
