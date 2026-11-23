"""势场只定义在导航一致的当前车道及可跨越邻道，不把整个路口当作走廊。"""
from __future__ import annotations

import shapely

from .object_scope import lane_role_geometry


OVERLAY_GRID_M = 1e-9


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


def _clip_lane_domain(parts, drivable):
    """Intersect valid inputs, retrying only a numerically broken GEOS overlay.

    Coincident lane/map boundaries can produce nested shells or misplaced holes
    even though both inputs are valid. A one-nanometre overlay grid resolves
    that numeric topology; it is not a lane buffer or an invalid-map repair.
    """
    if not drivable.is_valid or drivable.is_empty:
        raise ValueError('势场可行驶地图源几何无效或为空')
    if any(not part.is_valid for part in parts):
        raise ValueError('势场车道源几何无效')
    corridor = shapely.union_all(parts)
    if corridor.is_empty or not corridor.is_valid:
        raise ValueError('势场导航车道源并集无效或为空')
    reason = None
    try:
        domain = shapely.intersection(corridor, drivable)
        if not domain.is_valid:
            reason = shapely.is_valid_reason(domain)
    except shapely.errors.GEOSException as exc:
        reason = type(exc).__name__
    repair = dict(applied=reason is not None, method='valid_input_fixed_precision_overlay',
                  source_valid=dict(corridor=True, drivable=True),
                  grid_size_m=OVERLAY_GRID_M if reason else None,
                  geos_version=shapely.geos_version_string, original_overlay_error=reason)
    if reason is not None:
        domain = shapely.intersection(corridor, drivable, grid_size=OVERLAY_GRID_M)
    if domain.is_empty or not domain.is_valid:
        raise ValueError('导航车道势场域为空或几何无效')
    intersection_type = domain.geom_type
    domain = polygonal_area(domain)
    if domain.is_empty or not domain.is_valid or domain.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError('导航车道势场面域为空或几何无效')
    if reason is not None:
        # Vertex rounding displaces each input vertex by at most half a grid
        # diagonal. The two-sided boundary-strip area bounds this numerical
        # effect; the tolerance follows actual input perimeter, not road width.
        tolerance = 2 * OVERLAY_GRID_M * (corridor.length + drivable.length) + 8 * OVERLAY_GRID_M**2
        excess, verification = {}, {}
        for name, source in (('corridor', corridor), ('drivable', drivable)):
            try:
                difference = shapely.difference(domain, source)
                if not difference.is_valid:
                    raise shapely.errors.GEOSException('Invalid difference overlay')
                verification[name] = 'float64_difference'
            except shapely.errors.GEOSException:
                difference = shapely.difference(domain, source, grid_size=OVERLAY_GRID_M)
                verification[name] = 'same_precision_grid_difference'
            if not difference.is_valid or difference.area > tolerance:
                raise ValueError('势场数值相交超出原车道或可行驶地图范围')
            excess[name] = float(difference.area)
        repair.update(subset_excess_area_m2=excess, subset_area_tolerance_m2=float(tolerance),
                      subset_verification=verification,
                      tolerance_basis='2*grid_m*(corridor_perimeter_m+drivable_perimeter_m)+8*grid_m^2')
    return domain, intersection_type, repair


def build_lane_potential_domain(scene, front_m=120., rear_m=20.):
    """返回世界坐标车道域与证据；量程仅用于延伸地图，不改变对象40/20米准入。"""
    traces, permissions, cache = lane_role_geometry(scene, front_m=front_m, rear_m=rear_m)
    parts = []
    selected = set()
    connections = set()
    for trace in traces.values():
        if trace is None:
            continue
        if trace.get('validated_corridor') is not None:
            # NAVSIM registers this exact road-clipped narrow reference band
            # and retained native suffix, without certifying vehicle motion.
            # Rebuilding whole native lanes here
            # would restore the prefix explicitly replaced by that contract.
            parts.append(trace['validated_corridor'])
            selected.update(trace['lane_ids'])
            connections.update((link['from_lane_id'],link['to_lane_id'])
                               for link in trace.get('navigation_connections', []))
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
    domain, intersection_type, repair = _clip_lane_domain(parts, scene.drivable)
    return domain, {
        'version': 'lane-potential-domain-1.2',
        'lane_ids': sorted(selected),
        'chains': {side: None if trace is None else trace['lane_ids'] for side, trace in traces.items()},
        'current_lane_association': (traces['center'].get('current_lane_association')
                                     if traces.get('center') is not None else None),
        'initialization_recovery': (scene.diagnostics.get('navigation_initialization', {}).get('status') in ('active','reference_only_unverified')
                                    and scene.scenario=='navsim'),
        'initialization_reference_status': (scene.diagnostics.get('navigation_initialization',{}).get('status')
                                             if scene.scenario=='navsim' else None),
        'reference_quality': (scene.diagnostics.get('native_navigation_validation',{}) if scene.scenario=='navsim' else None),
        'domain_registration_is_vehicle_safety_certificate': False,
        'adjacent_permissions': permissions,
        'navigation_connections': [{'from_lane_id': pair[0], 'to_lane_id': pair[1]} for pair in sorted(connections)],
        'geometry': 'union(actual_lane_polygon intersect centerline_half_width_band;confirmed_traversed_navigation_connection_band) intersect drivable',
        'intersection_type': intersection_type, 'domain_type': domain.geom_type,
        'numeric_overlay': repair,
        'area_policy': 'exact_polygonal_components;zero_area_lines_and_points_excluded;no_buffer_or_expansion',
        'outside': 'undefined; rejected_by_domain_constraint',
        'map_extension_front_m': float(front_m), 'map_extension_rear_m': float(rear_m),
        'object_admission_unchanged': True,
    }
