"""CARLA junction movement boundaries are local, not global road barriers.

Keep the raw OpenDRIVE marking and lane-change permission. A complete painted
side boundary of another movement becomes movement-local only after its native
polyline properly crosses a confirmed navigation connector's centerline. No
geometry is removed from the drivable surface or lane-change permission granted.
"""
from __future__ import annotations

from copy import deepcopy
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, unary_union


VERSION = 'carla-junction-boundary-scope-2.0'
_PAINT = {'solid', 'solidsolid', 'solidbroken', 'brokensolid',
          'broken', 'brokenbroken', 'bottsdots'}
_EPSILON_M = 1e-5


def _lines(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return [geometry] if geometry.length > _EPSILON_M else []
    if hasattr(geometry, 'geoms'):
        return [line for part in geometry.geoms for line in _lines(part)]
    return []


def _junction(lane):
    """Require native, agreeing metadata for every sampled connector edge."""
    boundaries = lane.get('boundaries', [])
    if not boundaries:
        return None
    identities = set()
    for edge in boundaries:
        identity = edge.get('junction_id')
        if (edge.get('owner_lane_id') != lane['id'] or edge.get('is_junction') is not True
                or type(identity) is not int or identity < 0):
            return None
        identities.add(identity)
    return next(iter(identities)) if len(identities) == 1 else None


def _points(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Point':
        return [geometry]
    if hasattr(geometry, 'geoms'):
        return [point for part in geometry.geoms for point in _points(part)]
    # Overlapping/collinear lines are not evidence of a transverse crossing.
    return []


def _proper_crossings(line, center):
    result = []
    for point in _points(line.intersection(center)):
        s, t = line.project(point), center.project(point)
        step = min(.25, s/2., (line.length-s)/2., t/2., (center.length-t)/2.)
        if step <= _EPSILON_M:
            continue  # Entry/exit endpoint contact does not prove crossing.
        a, b = center.interpolate(t-step), center.interpolate(t+step)
        tx, ty = b.x-a.x, b.y-a.y
        norm = (tx*tx+ty*ty)**.5
        if norm <= _EPSILON_M:
            continue
        before, after = line.interpolate(s-step), line.interpolate(s+step)
        signed_before = (tx*(before.y-point.y)-ty*(before.x-point.x))/norm
        signed_after = (tx*(after.y-point.y)-ty*(after.x-point.x))/norm
        if ((signed_before > _EPSILON_M and signed_after < -_EPSILON_M)
                or (signed_before < -_EPSILON_M and signed_after > _EPSILON_M)):
            result.append({'point': [point.x, point.y], 'sample_step_m': step,
                           'signed_side_before_m': signed_before,
                           'signed_side_after_m': signed_after})
    return result


def _side(edge, owner):
    """Only native bridge side/index IDs can establish a complete side group."""
    for side in ('left', 'right'):
        prefix = owner+':'+side+':'
        suffix = str(edge.get('id', ''))[len(prefix):]
        if (str(edge.get('id', '')).startswith(prefix) and suffix.isdigit()
                and edge.get('side', side) == side):
            return side
    return None


def _paint(edge):
    name = str(edge.get('line_type', '')).split('.')[-1].lower()
    return (name in _PAINT and edge.get('physical_marking') is True
            and edge.get('map_evidence_known') is True
            and edge.get('kind') not in ('virtual_connection', 'virtual_lane_boundary')
            and len(edge.get('points', [])) >= 2)


def scope_junction_boundaries(mapping):
    """Return an audited copy; absent or ambiguous evidence preserves all edges.

    This CARLA-only adaptation must run before constructing platform-neutral
    Lane objects. It does not claim that a prebuilt town's FBX paint is absent.
    CARLA 0.9.15 itself scopes lane crossing away from junction connectors:
    LibCarla/source/carla/road/element/LaneCrossingCalculator.cpp, lines 65-71.
    """
    result = deepcopy(mapping)
    if result.get('junction_boundary_scope', {}).get('version') == VERSION:
        return result
    lanes = result.get('lanes', [])
    route_ids = set(result.get('route_lane_ids', []))
    route_points = result.get('route', [])
    route = LineString(route_points) if len(route_points) >= 2 else None
    connectors = []
    for lane in lanes:
        junction = _junction(lane)
        if lane['id'] not in route_ids or junction is None or route is None:
            continue
        if len(lane.get('polygon', [])) < 3 or len(lane.get('center', [])) < 2:
            continue
        polygon = Polygon(lane['polygon'])
        center = LineString(lane['center'])
        if (not polygon.is_valid or polygon.is_empty or not center.is_valid
                or center.length <= _EPSILON_M):
            continue
        # Do not infer traversal merely from a neighbouring/unused lane ID.
        interior = polygon.buffer(-_EPSILON_M)
        if interior.is_empty or route.intersection(interior).length <= _EPSILON_M:
            continue
        connectors.append((lane['id'], junction, center))
    modified, groups_evidence = [], []
    for lane in lanes:
        junction = _junction(lane)
        if lane['id'] in route_ids or junction is None:
            continue
        groups = {}
        for edge in lane.get('boundaries', []):
            side = _side(edge, lane['id'])
            if side is not None and _paint(edge):
                groups.setdefault(side, []).append(edge)
        for side, edges in groups.items():
            combined = unary_union([LineString(edge['points']) for edge in edges])
            lines = _lines(combined if combined.geom_type == 'LineString' else linemerge(combined))
            crossings = []
            for target_id, target_junction, center in connectors:
                if target_junction != junction:
                    continue
                for line in lines:
                    crossings.extend(dict(item, navigation_connector_id=target_id)
                                     for item in _proper_crossings(line, center))
            if not crossings:
                continue
            identity = lane['id']+':'+side
            groups_evidence.append({'id': identity, 'owner_lane_id': lane['id'],
                'side': side, 'junction_id': junction, 'raw_boundaries': deepcopy(edges),
                'proper_crossings': crossings,
                'policy': 'complete_other_movement_side_properly_crosses_navigation_centerline',
                'physical_paint_visibility': 'not_inferred_from_OpenDRIVE',
                'lane_change_permissions_unchanged': True})
            for edge in edges:
                edge['boundary_scope_evidence'] = {'version': VERSION, 'group_id': identity,
                                                   'raw_kind': edge.get('kind')}
                edge['kind'] = 'virtual_connection'
                modified.append(edge['id'])
    result['junction_boundary_scope'] = {
        'version': VERSION, 'modified_boundary_ids': modified,
        'navigation_connector_ids': sorted({item[0] for item in connectors}),
        'groups': groups_evidence,
        'proper_crossing_side_tolerance_m': _EPSILON_M,
        'scope': 'same_junction_other_movement_side_with_proven_centerline_crossing',
        'raw_markings_retained': True, 'lane_change_permissions_unchanged': True}
    return result
