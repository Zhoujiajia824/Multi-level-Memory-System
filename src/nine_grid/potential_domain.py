"""势场只定义在导航一致的当前车道及可跨越邻道，不把整个路口当作走廊。"""
from __future__ import annotations

import shapely

from .object_scope import lane_role_geometry


def polygonal_area(geometry):
    """Keep the exact positive-area components of a geometric intersection.

    Valid lane/drivable intersections may contain detached boundary lines or
    points. These are not drivable areas. Do not buffer, simplify or expand a
    lane to make its type acceptable to the potential chart.
    """
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry
    polygons = []
    def collect(part):
        if part.geom_type == 'Polygon':
            if not part.is_empty and part.area > 0:
                polygons.append(part)
        elif part.geom_type in ('MultiPolygon', 'GeometryCollection'):
            for child in part.geoms:
                collect(child)
    collect(geometry)
    if not polygons:
        raise ValueError('导航车道势场域没有有效面，不能以线或点替代车道')
    return shapely.union_all(polygons)


def build_lane_potential_domain(scene, front_m=120., rear_m=20.):
    """返回世界坐标车道域与证据；量程仅用于延伸地图，不改变对象40/20米准入。"""
    traces, permissions, cache = lane_role_geometry(scene, front_m=front_m, rear_m=rear_m)
    parts = []
    selected = set()
    connections = set()
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
        # 仅加入该角色链实际经过的、已独立复核的导航曲线窄带。
        # 不能并入全部导航支路，不能用整个路口面替代车道域。
        for connection in trace.get('navigation_connections', []):
            pair = (connection['from_lane_id'], connection['to_lane_id'])
            if pair not in connections:
                parts.append(connection['shape'])
                connections.add(pair)
    if not parts:
        raise ValueError('势场缺少已确认的当前车道几何，不能使用整个可行驶地图代替')
    domain = shapely.union_all(parts).intersection(scene.drivable)
    if domain.is_empty or not domain.is_valid:
        raise ValueError('导航车道势场域为空或几何无效')
    intersection_type = domain.geom_type
    domain = polygonal_area(domain)
    if domain.is_empty or not domain.is_valid or domain.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError('导航车道势场面域为空或几何无效')
    return domain, {
        'version': 'lane-potential-domain-1.2',
        'lane_ids': sorted(selected),
        'chains': {side: None if trace is None else trace['lane_ids'] for side, trace in traces.items()},
        'current_lane_association': (traces['center'].get('current_lane_association')
                                     if traces.get('center') is not None else None),
        'adjacent_permissions': permissions,
        'navigation_connections': [{'from_lane_id': pair[0], 'to_lane_id': pair[1]} for pair in sorted(connections)],
        'geometry': 'union(actual_lane_polygon intersect centerline_half_width_band;confirmed_traversed_navigation_connection_band) intersect drivable',
        'intersection_type': intersection_type, 'domain_type': domain.geom_type,
        'area_policy': 'exact_polygonal_components;zero_area_lines_and_points_excluded;no_buffer_or_expansion',
        'outside': 'undefined; rejected_by_domain_constraint',
        'map_extension_front_m': float(front_m), 'map_extension_rear_m': float(rear_m),
        'object_admission_unchanged': True,
    }
