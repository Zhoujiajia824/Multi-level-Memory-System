"""势场只定义在导航一致的当前车道及可跨越邻道，不把整个路口当作走廊。"""
from __future__ import annotations

import shapely

from .object_scope import lane_role_geometry
from .geometry import polygonal_area


def build_lane_potential_domain(scene, front_m=120., rear_m=20.):
    """返回世界坐标车道域与证据；量程仅用于延伸地图，不改变对象40/20米准入。"""
    traces, permissions, cache = lane_role_geometry(scene, front_m=front_m, rear_m=rear_m)
    parts = []
    selected = set()
    connections = set()
    component_evidence = []
    for trace in traces.values():
        if trace is None:
            continue
        for identity in trace['lane_ids']:
            if identity in selected or identity not in cache:
                continue
            lane, polygon, center = cache[identity]
            polygon,_=polygonal_area(polygon,context='public lane source '+str(identity))
            # 宽大的交叉口地图面不能授权横向支路。保留真实多边形与车道宽带的交集。
            part = polygon.intersection(center.buffer(float(lane.width)/2, cap_style=2))
            part,evidence=polygonal_area(part,context='potential lane '+str(identity),require_nonempty=False)
            if evidence['source_type'] not in ('Polygon','MultiPolygon'):
                component_evidence.append(evidence)
            if not part.is_empty:
                parts.append(part)
                selected.add(identity)
        # 仅加入该角色链实际经过的、已独立复核的导航曲线窄带。
        # 不能并入全部导航支路，不能用整个路口面替代车道域。
        for connection in trace.get('navigation_connections', []):
            pair = (connection['from_lane_id'], connection['to_lane_id'])
            if pair not in connections:
                area,evidence=polygonal_area(connection['shape'],
                    context='potential navigation connection '+str(pair),require_nonempty=False)
                if not area.is_empty:
                    parts.append(area)
                if evidence['source_type'] not in ('Polygon','MultiPolygon'):
                    component_evidence.append(evidence)
                connections.add(pair)
    if not parts:
        raise ValueError('势场缺少已确认的当前车道几何，不能使用整个可行驶地图代替')
    domain,overlay_evidence=polygonal_area(shapely.union_all(parts).intersection(scene.drivable),
        context='navigation lane potential domain')
    return domain, {
        'version': 'lane-potential-domain-1.2',
        'polygonal_area':overlay_evidence,'component_overlays':component_evidence,
        'lane_ids': sorted(selected),
        'chains': {side: None if trace is None else trace['lane_ids'] for side, trace in traces.items()},
        'current_lane_association': (traces['center'].get('current_lane_association')
                                     if traces.get('center') is not None else None),
        'adjacent_permissions': permissions,
        'navigation_connections': [{'from_lane_id': pair[0], 'to_lane_id': pair[1]} for pair in sorted(connections)],
        'geometry': 'union(actual_lane_polygon intersect centerline_half_width_band;confirmed_traversed_navigation_connection_band) intersect drivable',
        'outside': 'undefined; rejected_by_domain_constraint',
        'map_extension_front_m': float(front_m), 'map_extension_rear_m': float(rear_m),
        'object_admission_unchanged': True,
    }
