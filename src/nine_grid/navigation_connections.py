"""核验自车已确认导航连接；不以较大的全局端点间隙授权支路。"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import shapely
from shapely.geometry import LineString

from .geometry import rectangle_corners


VERSION = 'confirmed-navigation-connection-1.0'
ENDPOINT_TOLERANCE_M = .05  # 原生地图与导航的不同采样产生的弦误差。
MAXIMUM_NATIVE_TRIM_M = 20.  # 平台连接搜索的最大端部裁剪量，不是普通连接阈值。
ROAD_NUMERICAL_TOLERANCE_M = 1e-6
_CACHE_KEY = '_confirmed_navigation_connections'


def _project(points, queries):
    """批量真实欧氏投影，保留端点外残差。"""
    points, queries = np.asarray(points, float), np.atleast_2d(np.asarray(queries, float))
    # 原生圆弧共用角度采样可能重复包含端点；消除零长段不改变道路或弧长。
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError('连接参考至少需要两个二维点')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if len(lengths) == 0 or np.any(lengths <= 1e-8):
        raise ValueError('连接参考包含零长段')
    fraction = np.clip(np.sum((queries[:, None]-points[:-1])*delta, axis=2)/lengths**2, 0., 1.)
    feet = points[:-1]+fraction[:, :, None]*delta
    distances = np.linalg.norm(queries[:, None]-feet, axis=2)
    index = np.argmin(distances, axis=1)
    row = np.arange(len(queries))
    arc = np.r_[0., np.cumsum(lengths)]
    return (arc[index]+fraction[row, index]*lengths[index],
            np.arctan2(delta[index, 1], delta[index, 0]), distances[row, index])


def _fingerprint(scene, rows):
    """缓存随帧、地图、路线、连接内容和车体变化失效；深拷贝不会继承过期授权。"""
    digest = hashlib.sha256()
    digest.update(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    digest.update(str((scene.frame_id, scene.time, scene.coordinates, scene.ego.length, scene.ego.width)).encode('utf-8'))
    digest.update(np.asarray(scene.route, dtype=np.float64).tobytes())
    digest.update(scene.drivable.wkb)
    for lane in scene.lanes:
        digest.update(str((lane.id, lane.width, tuple(lane.successors))).encode('utf-8'))
        digest.update(np.asarray(lane.center, dtype=np.float64).tobytes())
    digest.update(str(tuple(scene.route_lane_ids)).encode('utf-8'))
    return digest.hexdigest()


def _validate(scene, row, lanes, pairs):
    if not isinstance(row, dict) or row.get('version') != VERSION:
        return None, 'unsupported_connection_protocol'
    pair = (row.get('from_lane_id'), row.get('to_lane_id'))
    if not all(isinstance(identity, str) for identity in pair):
        return None, 'malformed_lane_identity'
    if pair not in pairs or pair[0] not in lanes or pair[1] not in lanes:
        return None, 'not_consecutive_ego_navigation_lanes'
    source, target = lanes[pair[0]], lanes[pair[1]]
    if target.id not in source.successors:
        return None, 'not_directed_public_successor'
    if row.get('validated') is not True or type(row.get('outside_samples')) is not int or row['outside_samples'] != 0:
        return None, 'producer_did_not_validate_connection'
    try:
        points = np.asarray(row['points_world'], float)
        headings = np.asarray(row['headings_world'], float)
        dimensions = np.array([row['ego_length_m'], row['ego_width_m']], float)
        maximum, limit = float(row['maximum_steering']), float(row['steering_limit'])
        width = min(float(source.width), float(target.width))
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or
                headings.shape != (len(points),) or not np.isfinite(points).all() or
                not np.isfinite(headings).all() or not np.isfinite(dimensions).all() or
                not math.isfinite(maximum) or not math.isfinite(limit) or
                not math.isfinite(width) or width <= 0 or np.any(dimensions <= 0)):
            return None, 'nonfinite_or_malformed_connection_geometry'
        if not np.allclose(dimensions, [scene.ego.length, scene.ego.width], rtol=0., atol=1e-6):
            return None, 'different_vehicle_dimensions'
        if not 0 < limit < math.pi/2 or maximum < 0 or maximum > limit+1e-8:
            return None, 'steering_limit_exceeded'
        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(chords <= 1e-8) or np.max(chords) > 1.:
            return None, 'connection_sampling_not_dense'
        start_s, start_heading, start_error = _project(source.center, points[0])
        end_s, end_heading, end_error = _project(target.center, points[-1])
        source_length = np.linalg.norm(np.diff(source.center, axis=0), axis=1).sum()
        if (start_error[0] > ENDPOINT_TOLERANCE_M or end_error[0] > ENDPOINT_TOLERANCE_M or
                source_length-start_s[0] > MAXIMUM_NATIVE_TRIM_M+ENDPOINT_TOLERANCE_M or
                end_s[0] > MAXIMUM_NATIVE_TRIM_M+ENDPOINT_TOLERANCE_M):
            return None, 'not_attached_to_native_lane_ends'
        if (math.cos(headings[0]-start_heading[0]) < .5 or
                math.cos(headings[-1]-end_heading[0]) < .5):
            return None, 'connection_heading_opposes_lanes'
        for name, measured in (('from_end_trim_m', source_length-start_s[0]), ('to_start_trim_m', end_s[0])):
            if name in row and (not math.isfinite(float(row[name])) or
                    abs(float(row[name])-measured) > 2*ENDPOINT_TOLERANCE_M):
                return None, 'native_trim_evidence_mismatch'
        route_s, _, route_error = _project(scene.route, points)
        if (np.any(route_error > ENDPOINT_TOLERANCE_M) or np.any(np.diff(route_s) < -ENDPOINT_TOLERANCE_M) or
                abs((route_s[-1]-route_s[0])-chords.sum()) > 2*ENDPOINT_TOLERANCE_M):
            return None, 'connection_not_on_current_frozen_navigation'
        # 独立复算完整车体道路覆盖；不信任生产者outside_samples或点中心在路内。
        bodies = shapely.polygons(np.stack([rectangle_corners(p, h, *dimensions)
                                           for p, h in zip(points, headings)]))
        if not np.all(shapely.covers(scene.drivable.buffer(ROAD_NUMERICAL_TOLERANCE_M), bodies)):
            return None, 'independent_full_body_road_check_failed'
        shape = LineString(points).buffer(width/2, cap_style=2).intersection(scene.drivable)
        if shape.is_empty or not shape.is_valid:
            return None, 'empty_or_invalid_connection_corridor'
        points, headings = points.copy(), headings.copy()
        points.flags.writeable = headings.flags.writeable = False
        return {'from_lane_id': source.id, 'to_lane_id': target.id,
                'points': points, 'headings': headings, 'shape': shape,
                'width_m': width, 'version': VERSION,
                'validation': 'directed_navigation_pair;native_end_attachment;frozen_route;full_body_drivable'}, None
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None, 'malformed_connection_evidence'


def confirmed_navigation_connections(scene):
    """返回经独立几何核验的有序车道对；无证据保持原2米连接规则。

    返回值只提供已确认连接，调用者仍须限定为本次实际走过的角色链，
    并保留物理车体、不可跨线、碰撞及原对象范围的独立约束。
    """
    navigation = scene.diagnostics.get('navigation', {})
    rows = navigation.get('connections', []) if isinstance(navigation, dict) else []
    try:
        fingerprint = _fingerprint(scene, rows)
    except (ValueError, TypeError, AttributeError):
        return {}
    cached = scene.assets.get(_CACHE_KEY)
    if cached is not None and cached.get('fingerprint') == fingerprint:
        return cached['connections']
    lanes = {lane.id: lane for lane in scene.lanes}
    pairs = set(zip(scene.route_lane_ids, scene.route_lane_ids[1:]))
    accepted, rejected = {}, []
    for row in rows if isinstance(rows, list) else []:
        connection, reason = _validate(scene, row, lanes, pairs)
        if connection is None:
            rejected.append({'from_lane_id': row.get('from_lane_id') if isinstance(row, dict) else None,
                             'to_lane_id': row.get('to_lane_id') if isinstance(row, dict) else None,
                             'reason': reason})
        else:
            pair = (connection['from_lane_id'], connection['to_lane_id'])
            if pair in accepted:
                # 重复/歧义证据不凭任意覆盖选一条。
                accepted.pop(pair)
                rejected.append({'from_lane_id': pair[0], 'to_lane_id': pair[1], 'reason': 'ambiguous_duplicate_connection'})
                pairs.discard(pair)
            else:
                accepted[pair] = connection
    scene.assets[_CACHE_KEY] = {'fingerprint': fingerprint, 'connections': accepted, 'rejected': rejected}
    return accepted
