"""道路拓扑限定的当前对象真值；模型和基础规划共享准入边界。

只接受当前车道与当前位置可跨越同向邻道的唯一前后链。量程沿各链弧长，
不是圆形距离。原始快照保留供审计及图像上下文，不被原地修改。
"""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .geometry import actor_shape
from .lane_roles import (ASSIGNMENT_PROTOCOL, trace_lane, lane_owner,
                         belongs_to_trace, project)

VERSION = 'lane-local-object-scope-1.0'
KINDS = ('vehicle', 'bicycle', 'pedestrian', 'static_obstacle')
ROLES = ('left_front', 'front', 'right_front', 'left_side', 'right_side',
         'left_rear', 'rear', 'right_rear')


def scope_settings(scene, config=None):
    previous = scene.diagnostics.get('object_scope', {})
    values = ((config or {}).get('planning_objects') or
              previous.get('settings', {}))
    result = {key: float(values.get(key, default))
              for key, default in (('front_m', 40.), ('rear_m', 20.))}
    for key, maximum in (('front_m', 200.), ('rear_m', 100.)):
        raw = values.get(key, result[key])
        if isinstance(raw, bool) or not np.isfinite(result[key]) or not 0 <= result[key] <= maximum:
            raise ValueError('对象范围 '+key+' 必须是有限合法米值')
    if result['front_m'] <= 0:
        raise ValueError('对象范围 front_m 必须大于零')
    return result


def adjacent_road_evidence(scene, current, lanes, side, tolerance=.3):
    """当前位置两侧共同边界必须有明确可跨越证据，不能用远处虚线授权。"""
    unknown = dict(known=False, exists=None, crossable=None, reason='map_unknown')
    if current is None or scene.diagnostics.get('lane_map_complete') is not True:
        return unknown
    identity = getattr(current, side)
    if identity is None:
        return dict(known=True, exists=False, crossable=False, reason='no_adjacent_lane')
    target = lanes.get(identity)
    if target is None:
        return dict(unknown, reason='adjacent_lane_geometry_missing')
    try:
        _, a, heading, _ = project(current.center, scene.ego.position)
        _, b, other_heading, _ = project(target.center, a)
        if math.cos(other_heading-heading) <= 0:
            return dict(known=True, exists=False, crossable=False, reason='no_same_direction_adjacent_lane')
        lateral = float(np.dot(b-a, [-math.sin(heading), math.cos(heading)]))
        if (side == 'left' and lateral <= 0) or (side == 'right' and lateral >= 0):
            return dict(unknown, reason='adjacent_lane_side_mismatch')
        middle = Point((a+b)/2)
        permissions = []
        for lane in (current, target):
            nearby = [edge for edge in lane.boundaries if len(edge.get('points', [])) >= 2
                      and LineString(edge['points']).distance(middle) <= tolerance]
            if not nearby or any(type(edge.get('crossable')) is not bool for edge in nearby):
                return dict(unknown, reason='current_boundary_permission_unknown')
            permissions.extend(edge['crossable'] for edge in nearby)
    except (ValueError, TypeError, IndexError):
        return dict(unknown, reason='current_boundary_geometry_invalid')
    crossable = bool(all(permissions))
    return dict(known=True, exists=True, crossable=crossable,
                reason='crossable_boundary' if crossable else 'noncrossable_boundary')


def lane_role_geometry(scene, current_lane_id=None, front_m=40., rear_m=20.):
    """返回角色链、邻道权限、地图归属缓存；不读取任何对象隐藏路线。"""
    lanes = {lane.id: lane for lane in scene.lanes}
    # 历史导航身份可保持坐标连续，但不能代替当前物理车道的观测证据。
    current = (lanes.get(current_lane_id or scene.ego.lane_id)
               if scene.ego.lane_id in lanes else None)
    roads = {side: adjacent_road_evidence(scene, current, lanes, side)
             for side in ('left', 'right')}
    traces = {}
    for side in ('center', 'left', 'right'):
        identity = current.id if side == 'center' and current else getattr(current, side, None)
        if side != 'center' and not (roads[side]['known'] and roads[side]['crossable']):
            traces[side] = None
        else:
            traces[side] = trace_lane(scene, identity, back_m=rear_m, front_m=front_m,
                                      center_role=side == 'center')
    cache = {key: (lane, Polygon(lane.polygon), LineString(lane.center))
             for key, lane in lanes.items() if len(lane.polygon) >= 3 and len(lane.center) >= 2}
    return traces, roads, cache


def _interval(body, points):
    return [project(points, xy)[0] for xy in body.exterior.coords]


def select_scope_objects(scene, config=None):
    """当前真值范围内全部有用对象；八向选一在此之后，不能越界补对象。"""
    settings = scope_settings(scene, config)
    traces, roads, cache = lane_role_geometry(scene, **settings)
    ego_body = actor_shape(scene.ego)
    ego_intervals = {side: _interval(ego_body, trace['points'])
                     for side, trace in traces.items() if trace is not None}
    objects, excluded, selected, seen = {}, {}, [], set()
    for actor in sorted(scene.actors, key=lambda a: a.id):
        if actor.id in seen:
            raise ValueError('对象范围输入ID重复：'+actor.id)
        seen.add(actor.id)
        if actor.id == scene.ego.id:
            excluded[actor.id] = 'self_is_not_object'; continue
        if not actor.visible:
            excluded[actor.id] = 'not_currently_visible'; continue
        if actor.kind not in KINDS:
            excluded[actor.id] = 'unsupported_entity_kind'; continue
        body = actor_shape(actor)
        owner = lane_owner(actor, cache)
        memberships = {}
        for side, trace in traces.items():
            belongs = belongs_to_trace(actor, body, owner, trace, cache, 1e-5)
            if not belongs and owner is None and trace is not None:
                # 中心尚在车道外、但真实外廓已经侵入车道的行人/障碍仍算在道上。
                # 额外交集必须位于中心线物理宽度内，宽路口多边形不能单独准入。
                belongs = any(body.intersection(cache[key][1]).intersection(
                    cache[key][2].buffer(cache[key][0].width/2, cap_style=2)).area >= 1e-5
                    for key in trace['lane_ids'] if key in cache)
                if not belongs:
                    # 原生车道端部之间的已验证导航连接同属这条链，不把
                    # 连接上的当前物体误当路外；仍按下方曲线弧长40/20准入。
                    belongs = any(body.intersection(link['shape']).area >= 1e-5
                                  for link in trace.get('navigation_connections', []))
            if not belongs:
                continue
            points = trace['points']
            s, foot, heading, _ = project(points, actor.position)
            end = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            beyond = float(np.dot(np.asarray(actor.position)-foot, [math.cos(heading), math.sin(heading)]))
            if (s <= 1e-7 and beyond < -1e-6) or (s >= end-1e-7 and beyond > 1e-6):
                continue
            ds = float(s-trace['ego_s'])
            if ds < -settings['rear_m']-1e-8 or ds > settings['front_m']+1e-8:
                continue
            intervals = _interval(body, points)
            ego_interval = ego_intervals[side]
            overlap = max(intervals) >= min(ego_interval) and min(intervals) <= max(ego_interval)
            role = ('front' if ds >= 0 else 'rear') if side == 'center' else side+('_side' if overlap else '_front' if ds >= 0 else '_rear')
            gap = max(0., min(intervals)-max(ego_interval) if ds >= 0 else min(ego_interval)-max(intervals))
            memberships[role] = dict(delta_s_m=ds, lane_heading_rad=float(heading), body_interval_m=[min(intervals), max(intervals)],
                longitudinal_gap_m=gap, side_gap_m=float(body.distance(ego_body)),
                body_longitudinal_overlap=bool(overlap), lane_chain=side)
        if not memberships:
            excluded[actor.id] = 'outside_permitted_lane_chain_or_longitudinal_range'
            continue
        selected.append(actor)
        objects[actor.id] = dict(owner_lane_id=owner, roles=sorted(memberships), memberships=memberships)
    diagnostic = dict(version=VERSION, assignment_protocol=ASSIGNMENT_PROTOCOL,
        frame_id=scene.frame_id, time=float(scene.time), settings=settings,
        range_rule='actor_center_arc_distance_on_each_permitted_lane_chain',
        selected_ids=[actor.id for actor in selected], selected_count=len(selected),
        input_count=len(scene.actors), objects=objects, roads=roads,
        chains={side: None if trace is None else dict(lane_ids=trace['lane_ids'],
                    ego_s_m=float(trace['ego_s']), decisions=trace['decisions'])
                for side, trace in traces.items()},
        excluded=[dict(id=key, reason=excluded[key]) for key in sorted(excluded)],
        adaptive_expansion=False,
        limitation='限定车道和固定弧长范围；不声称具备全场景感知或覆盖范围外危险')
    return selected, diagnostic


def scope_scene(scene, config=None):
    actors, diagnostic = select_scope_objects(scene, config)
    diagnostics = {**scene.diagnostics, 'object_scope': diagnostic}
    # 上游曾缓存完整场景路权观测时，必须重建为本次准入范围；规划子集保留
    # 本范围内的冻结路权输入，绝不恢复原始全场景对象。
    if 'planning_scope' not in diagnostics:
        diagnostics.pop('road_rule_observations', None)
    return replace(scene, actors=actors, diagnostics=diagnostics), diagnostic
