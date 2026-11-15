"""轻量 Frenet 八向规划对象筛选；与检索九宫格的表示和观测掩码分开。

本模块在EKF之前缩小规划实体集合，预测、MPC、评分和执行验收使用同一集合。
完整场景仍供图像、模型输入、检索及回放，不被原地删减。筛选没有全场景避碰保证。
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import time
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

from .geometry import (clean_polyline, project_polyline, rectangle_corners,
                       sample_polyline, polyline_lengths, signed_body_distance, wrap_angle, actor_shape)


VERSION = 'frenet-eight-planning-2.0'
DEFAULTS = {
    'version': VERSION,
    'front_m': 30.,
    'rear_m': 10.,
    'lateral_m': 25.,
    'include_attention': True,
    'max_attention_objects': 3,
    'adaptive_front': True,
    'reaction_s': 1.0,
    'braking_mps2': 4.0,
    'stopping_margin_m': 5.0,
    'conflict_horizon_s': 3.0,
    'conflict_margin_m': .5,
}
ROLES = ('left_front', 'front', 'right_front', 'left_side', 'right_side',
         'left_rear', 'rear', 'right_rear')
KINDS = ('vehicle', 'bicycle', 'pedestrian', 'static_obstacle')


def validate_planning_object_config(options=None):
    """开发默认值是可配置起点，未知字段/非法数值不能静默改变筛选含义。"""
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options)-set(DEFAULTS):
        raise ValueError('planning_objects 包含未知字段或不是配置字典')
    result = {**DEFAULTS, **options}
    if result['version'] != VERSION:
        raise ValueError('planning_objects.version 不受支持')
    for key, maximum, allow_zero in (('front_m', 200., False), ('rear_m', 100., True),
                                      ('lateral_m', 100., False)):
        value = result[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not np.isfinite(value) or value > maximum or value < 0 or
                (not allow_zero and value == 0)):
            raise ValueError(f'planning_objects.{key} 必须是有效的米值，范围为 '
                             + ('[0,' if allow_zero else '(0,') + str(maximum)+']')
        result[key] = float(value)
    if type(result['include_attention']) is not bool:
        raise ValueError('planning_objects.include_attention 必须是布尔值')
    if type(result['max_attention_objects']) is not int or not 0 <= result['max_attention_objects'] <= 3:
        raise ValueError('planning_objects.max_attention_objects 必须是0至3的整数')
    if type(result['adaptive_front']) is not bool:
        raise ValueError('planning_objects.adaptive_front 必须是布尔值')
    for key in ('reaction_s','braking_mps2','stopping_margin_m','conflict_horizon_s','conflict_margin_m'):
        value=result[key]
        if type(value) not in (int,float) or not np.isfinite(value) or value<0 or (key in ('braking_mps2','conflict_horizon_s') and value==0):
            raise ValueError('planning_objects.'+key+'必须是有限合法值')
        result[key]=float(value)
    return result


def _body_points(actor):
    if actor.kind == 'pedestrian':
        radius = float(actor.radius or actor.width/2)
        # 真实圆形边界采样仅供 Frenet 区域归属；侧方净距仍用解析圆形几何。
        angles = np.arange(16)*np.pi/8
        return actor.position + radius*np.c_[np.cos(angles), np.sin(angles)]
    return rectangle_corners(actor.position, actor.heading, actor.length, actor.width)


def _selection_reference(scene):
    """导航从出生点开始时，用当前公开车道的真实前缀补齐后方投影支撑。

    不修改全局导航，不用直线外推伪造弯道，不读取任何周车路线。
    """
    reference = clean_polyline(scene.route)
    anchor = scene.diagnostics.get('route_s')
    identity = scene.diagnostics.get('stable_navigation_lane_id', scene.ego.lane_id)
    lane = next((item for item in scene.lanes if item.id == identity), None)
    prefix_length = 0.
    if lane is not None:
        lane_line = clean_polyline(lane.center)
        lane_s, _, _ = project_polyline(reference[0], lane_line, heading=scene.ego.heading)
        foot = sample_polyline(lane_line, lane_s)
        # 只有首导航点确实位于该中心线，才能接入同一条已知道路的前缀。
        if lane_s > 1e-6 and np.linalg.norm(foot-reference[0]) < .01:
            prefix = lane_line[polyline_lengths(lane_line) < lane_s-1e-6]
            if len(prefix):
                prefix_length = float(polyline_lengths(np.vstack((prefix, reference[0])))[-1])
                reference = clean_polyline(np.vstack((prefix, reference)))
                if anchor is not None:
                    anchor = float(anchor)+prefix_length
    return reference, anchor, prefix_length


def _lane_bands(scene, reference, ego_s, ego_d, window):
    """当前位置物理车道截面的横向范围，不以边界可跨越性删除规划对象。

    沿用已确认自车导航参考。缺少当前车道时用自车宽度作中心归格带，明确
    记为几何降级，既不伪造相邻车道，也不将全部可见对象误当不可用。
    """
    lanes = {lane.id: lane for lane in scene.lanes}
    source_id = scene.ego.lane_id if scene.ego.lane_id in lanes else scene.diagnostics.get('stable_navigation_lane_id')
    current = lanes.get(source_id)
    if current is None:
        half = scene.ego.width/2
        return {'center': (ego_d-half, ego_d+half)}, {
            'source_lane_id': source_id, 'source': 'ego_width_geometric_fallback',
            'left_lane_id': None, 'right_lane_id': None}
    source_s, _, source_heading = project_polyline(scene.ego.position, current.center, heading=scene.ego.heading)
    source_foot = sample_polyline(current.center, source_s)
    center_d = project_polyline(source_foot, reference, previous_s=ego_s, window=window)[1]
    bands = {'center': (center_d-current.width/2, center_d+current.width/2)}
    evidence = {'source_lane_id': current.id, 'source': 'physical_lane_cross_section',
                'left_lane_id': None, 'right_lane_id': None}
    for side in ('left', 'right'):
        other = lanes.get(getattr(current, side))
        if other is None:
            continue
        other_s, _, other_heading = project_polyline(source_foot, other.center, heading=source_heading)
        if np.cos(float(wrap_angle(other_heading-source_heading))) <= 0:
            continue
        foot = sample_polyline(other.center, other_s)
        offset = project_polyline(foot, reference, previous_s=ego_s, window=window)[1]
        # 地图的 left/right 字段必须与统一左正右负的几何方向一致。
        if (side == 'left' and offset <= center_d) or (side == 'right' and offset >= center_d):
            continue
        bands[side] = (offset-other.width/2, offset+other.width/2)
        evidence[side+'_lane_id']=other.id
    return bands, evidence


def select_planning_objects(scene, config=None, answer=None):
    """先筛选再预测：八向代表、最多3关注补充、明确的直接冲突例外。"""
    started = time.perf_counter()
    settings = validate_planning_object_config((config or {}).get('planning_objects'))
    existing=scene.diagnostics.get('planning_scope')
    if (isinstance(existing,dict) and existing.get('version')==VERSION and
            existing.get('frame_id')==scene.frame_id and existing.get('time')==scene.time and
            existing.get('settings')==settings and
            existing.get('selected_ids')==sorted(actor.id for actor in scene.actors)):
        # 下游不可从已筛选集合重新计算八格而丢掉直接冲突对象。
        return list(scene.actors),deepcopy(existing)
    reference, anchor, prefix_length = _selection_reference(scene)
    reference_length = float(polyline_lengths(reference)[-1])
    ego_s, ego_d, ego_heading = project_polyline(scene.ego.position, reference,
        previous_s=anchor, heading=scene.ego.heading)
    # 参考线的连续弧长支路搜索窗口足以覆盖裁剪范围与当前实体外廓。
    visible = [actor for actor in scene.actors if actor.visible]
    largest_body = max((np.hypot(a.length, a.width) for a in [scene.ego]+visible), default=0.)
    speed=float(np.linalg.norm(scene.ego.velocity))
    stopping_distance=speed*settings['reaction_s']+speed**2/(2*settings['braking_mps2'])+settings['stopping_margin_m']
    front=max(settings['front_m'],stopping_distance) if settings['adaptive_front'] else settings['front_m']
    rear_scan=settings['rear_m']+max((np.linalg.norm(a.velocity) for a in visible),default=0.)*settings['conflict_horizon_s']
    window = max(front, rear_scan)+largest_body+5.
    bands, road = _lane_bands(scene, reference, ego_s, ego_d, window)
    lane_ids={road.get('source_lane_id'),road.get('left_lane_id'),road.get('right_lane_id')}
    # 本车道在路口/匝道结束后，行驶走廊还包含公开导航的连续后继，
    # 否则几米外的转向段障碍物会因“不是当前道路编号”被错误丢弃。
    lanes_by_id={lane.id:lane for lane in scene.lanes}
    route_segments={str(key).rsplit('|',1)[0] for key in scene.route_lane_ids}
    pending=[key for key in lane_ids if key in lanes_by_id]
    while pending:
        key=pending.pop()
        for successor in lanes_by_id[key].successors:
            if (successor in lanes_by_id and successor not in lane_ids and
                    str(successor).rsplit('|',1)[0] in route_segments):
                lane_ids.add(successor);pending.append(successor)
    lane_polygons=[Polygon(lane.polygon) for lane in scene.lanes if lane.id in lane_ids]
    corridor=unary_union(lane_polygons) if lane_polygons else LineString(reference).buffer(scene.ego.width/2)
    road['candidate_corridor_lane_ids']=sorted(key for key in lane_ids if key in lanes_by_id)
    ego_interval = project_polyline(_body_points(scene.ego), reference, previous_s=ego_s, window=window)[0]
    ego_lo, ego_hi = float(ego_interval.min()), float(ego_interval.max())
    candidates = {name: [] for name in ROLES}
    within, excluded, object_roles, selection_details = {}, {}, {}, {}
    exceptions={}
    seen = set()
    eligible = []
    for actor in sorted(scene.actors, key=lambda item: item.id):
        if actor.id in seen:
            raise ValueError('规划筛选对象ID重复，不能用去重掩盖场景输入错误')
        seen.add(actor.id)
        if actor.id == scene.ego.id:
            excluded[actor.id] = 'self_is_not_obstacle'
            continue
        if not actor.visible:
            excluded[actor.id] = 'not_currently_visible'
            continue
        if actor.kind not in KINDS:
            excluded[actor.id] = 'unsupported_entity_kind'
            continue
        eligible.append(actor)
    # 所有对象使用同一连续支路窗口，可批量投影，避免逐实体反复清理道路折线。
    # 这只是等价数值计算；不会按未来轨迹、类别或关注排名预先隐藏对象。
    ranged = []
    if eligible:
        positions = np.array([actor.position for actor in eligible])
        center_s, center_d, center_h = project_polyline(positions, reference,
            previous_s=ego_s, window=window)
        foot = sample_polyline(reference, center_s)
        tangent = np.c_[np.cos(center_h), np.sin(center_h)]
        beyond = np.sum((positions-foot)*tangent, axis=1)
    for index, actor in enumerate(eligible):
        s, d = center_s[index], center_d[index]
        # 折线投影在首尾会夹紧弧长，不能把实际很远的对象压成近场对象。
        # 公开车道无法补足参考覆盖时明确排除，不用直线外推伪造道路。
        if ((s <= 1e-7 and beyond[index] < -1e-6) or
                (s >= reference_length-1e-7 and beyond[index] > 1e-6)):
            excluded[actor.id] = 'outside_reference_coverage'
            continue
        delta_s, delta_d = float(s-ego_s), float(d-ego_d)
        # 后方快速追近对象只作选集粗筛；这不是替代EKF的运动预测。
        relative_closing=float(actor.velocity@tangent[index])-speed
        rear_exception=(delta_s < -settings['rear_m'] and relative_closing>0 and
                        -delta_s<=settings['rear_m']+relative_closing*settings['conflict_horizon_s'])
        if (delta_s < -settings['rear_m'] and not rear_exception) or delta_s > front:
            excluded[actor.id] = 'outside_longitudinal_center_range'
            continue
        if abs(delta_d) > settings['lateral_m']:
            excluded[actor.id] = 'outside_lateral_center_range'
            continue
        shape=actor_shape(actor)
        drift=np.asarray(actor.velocity,float)*settings['conflict_horizon_s']
        sweep=unary_union([shape,affinity.translate(shape,float(drift[0]),float(drift[1]))]).convex_hull
        on_road=corridor is not None and shape.intersects(corridor)
        enters=(actor.kind!='static_obstacle' and np.linalg.norm(actor.velocity)>.05 and
                corridor is not None and sweep.intersects(corridor.buffer(settings['conflict_margin_m'])))
        if not on_road and not enters:
            excluded[actor.id]='outside_lane_corridor_without_observed_motion_conflict'
            continue
        lateral_speed=abs(float(actor.velocity@np.array([-np.sin(center_h[index]),np.cos(center_h[index])])) )
        closing=speed-float(actor.velocity@tangent[index]) if delta_s>=0 else relative_closing
        net_long=max(0.,abs(delta_s)-(actor.length+scene.ego.length)/2)
        approaching=(closing>0 and net_long<=closing*settings['conflict_horizon_s']+settings['conflict_margin_m'])
        reasons=[]
        if not on_road and enters:reasons.append('observed_motion_enters_lane_corridor')
        if on_road and lateral_speed>.2:reasons.append('observed_lateral_crossing_or_cut_in')
        if rear_exception and on_road:reasons.append('fast_rear_approach')
        if on_road and approaching:reasons.append('short_horizon_longitudinal_conflict')
        if reasons:exceptions[actor.id]=reasons
        ranged.append((actor, float(d), delta_s, delta_d, _body_points(actor),on_road,rear_exception))
    body_s = body_d = np.empty(0)
    if ranged:
        body_s, body_d, _ = project_polyline(np.concatenate([row[4] for row in ranged]),
            reference, previous_s=ego_s, window=window)
    offset = 0
    for actor, d, delta_s, delta_d, points, on_road, rear_exception in ranged:
        projected_s = body_s[offset:offset+len(points)]
        projected_d = body_d[offset:offset+len(points)]
        offset += len(points)
        body_lo, body_hi = float(projected_s.min()), float(projected_s.max())
        lateral_lo, lateral_hi = float(projected_d.min()), float(projected_d.max())
        overlap = body_hi >= ego_lo and body_lo <= ego_hi
        forward = delta_s >= 0.
        longitudinal_gap = max(0., body_lo-ego_hi if forward else ego_lo-body_hi)
        columns = [side for side, (lo, hi) in bands.items()
                   if lateral_hi > lo+1e-8 and lateral_lo < hi-1e-8]
        # 车道外横穿对象作为显式安全例外，不伪装成某个邻道代表。
        if not on_road or rear_exception:columns=[]
        roles = []
        side_gap = None
        for column in columns:
            if column == 'center':
                role = 'front' if forward else 'rear'
            else:
                role = column+('_side' if overlap else '_front' if forward else '_rear')
            if role.endswith('side'):
                if side_gap is None:
                    side_gap = max(0., signed_body_distance(scene.ego, actor))
                gap = side_gap
            else:
                gap = longitudinal_gap
            candidates[role].append((float(gap), abs(delta_s), actor.id))
            roles.append(role)
        within[actor.id] = actor
        object_roles[actor.id] = roles
        selection_details[actor.id] = {'delta_s_m': delta_s, 'delta_d_m': delta_d,
            'body_longitudinal_overlap': bool(overlap), 'longitudinal_gap_m': longitudinal_gap,
            'side_gap_m': side_gap, 'role_source': 'physical_lane_band' if roles else 'direct_conflict_exception'}
    winners = {role: min(rows)[2] if rows else None for role, rows in candidates.items()}
    base_ids = sorted({identity for identity in winners.values() if identity is not None})
    selected_ids = set(base_ids)|set(exceptions)
    exception_added=sorted(set(exceptions)-set(base_ids))
    attention_selected = []
    attention_added = []
    attention_rejected = []
    if answer is not None:
        attention = sorted(answer.attention, key=lambda row: (row['rank'], row['id']))
        seen_attention = set()
        for item in attention:
            identity = item['id']
            if identity in seen_attention:
                continue
            seen_attention.add(identity)
            if identity in selected_ids:
                attention_selected.append(identity)
            elif identity not in within:
                attention_rejected.append({'id': identity, 'reason': excluded.get(identity, 'not_a_current_visible_entity')})
            elif not settings['include_attention']:
                attention_rejected.append({'id': identity, 'reason': 'attention_expansion_disabled'})
            elif len(attention_added) >= settings['max_attention_objects']:
                attention_rejected.append({'id': identity, 'reason': 'attention_expansion_capacity'})
            else:
                selected_ids.add(identity)
                attention_selected.append(identity)
                attention_added.append(identity)
    selected = [within[identity] for identity in sorted(selected_ids)]
    for identity in within:
        if identity not in selected_ids:
            excluded[identity] = 'not_nearest_in_any_eight_role'
    diagnostic = {'version': VERSION, 'settings': settings,'frame_id':scene.frame_id,'time':float(scene.time),
        'reference': 'confirmed_navigation_frenet;s_forward;d_left_positive;meter',
        'public_current_lane_prefix_m': prefix_length,
        'ego_s_m': float(ego_s), 'ego_d_m': float(ego_d), 'road': road,
        'lane_bands_d_m': {key: list(value) for key, value in bands.items()},
        'region_objects': winners, 'base_selected_ids': base_ids,
        'base_selected_count': len(base_ids), 'attention_added_ids': attention_added,
        'effective_front_m':front,'stopping_distance_m':stopping_distance,
        'reference_front_coverage_m':max(0.,reference_length-ego_s),
        'reference_coverage_shortfall_m':max(0.,front-(reference_length-ego_s)),
        'exception_added_ids':exception_added,'exception_added_count':len(exception_added),
        'direct_conflict_reasons':{k:exceptions[k] for k in sorted(exceptions)},
        'conflict_screen':'当前外廓和已观察速度的扫掠粗筛，仅决定纳入EKF/几何集合，不是周车未来预测',
        'attention_added_count': len(attention_added),
        'attention_selected_ids': attention_selected, 'attention_rejected': attention_rejected,
        'selected_ids': sorted(selected_ids), 'selected_count': len(selected),
        'visible_count': len(visible), 'within_range_count': len(within),
        'selected_roles': {key: object_roles[key] for key in sorted(selected_ids)},
        'selected_state': {key: selection_details[key] for key in sorted(selected_ids)},
        'excluded': [{'id': key, 'reason': excluded[key]} for key in sorted(excluded)],
        'selection_seconds': time.perf_counter()-started,
        'limitation': '八向代表与运动冲突粗筛不是全场景安全证明；未读取隐藏对象或未来，覆盖不足明确记录'}
    return selected, diagnostic


def planning_scene(scene, config=None, answer=None):
    """在EKF之前建立局部浅拷贝；原场景仍供BEV、模型、检索与回放。"""
    selected,diagnostic=select_planning_objects(scene,config,answer)
    scoped=replace(scene,actors=selected,diagnostics={**scene.diagnostics,'planning_scope':deepcopy(diagnostic)})
    return scoped,diagnostic


def planning_view(scene, predictions, config=None, answer=None):
    """保留原帧和原始数据，只给本次MPC生成局部浅拷贝视图。"""
    scoped_scene, diagnostic = planning_scene(scene, config, answer)
    selected=scoped_scene.actors
    identities = {actor.id for actor in selected}
    missing=identities-set(predictions.trajectories)
    if missing:
        raise ValueError('规划对象缺少对应预测：'+','.join(sorted(missing)))
    scoped_predictions = replace(predictions,
        actors={key: predictions.actors[key] for key in sorted(identities)},
        trajectories={key: predictions.trajectories[key] for key in sorted(identities)},
        diagnostics={key: value for key, value in predictions.diagnostics.items()
                     if key in identities or key not in predictions.actors})
    return scoped_scene, scoped_predictions, diagnostic
