"""轻量 Frenet 八向规划对象筛选；与检索九宫格的表示和观测掩码分开。

本模块只缩小一次 MPC 内部的实体计算集合。完整可见场景、预测、图像、
模型输入和独立轨迹质量/执行审计继续由调用方保留。筛选没有全场景避碰保证。
"""
from __future__ import annotations

from dataclasses import replace
import time
import numpy as np

from .geometry import (clean_polyline, project_polyline, rectangle_corners,
                       sample_polyline, polyline_lengths, signed_body_distance, wrap_angle)


VERSION = 'frenet-eight-planning-1.0'
DEFAULTS = {
    'version': VERSION,
    'front_m': 30.,
    'rear_m': 10.,
    'lateral_m': 25.,
    'include_attention': True,
    'max_attention_objects': 3,
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
    source_id = scene.diagnostics.get('stable_navigation_lane_id', scene.ego.lane_id)
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
                'left_lane_id': current.left, 'right_lane_id': current.right}
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
    return bands, evidence


def select_planning_objects(scene, config=None, answer=None):
    """八向代表同ID去重；至多补充3个同范围、当前可见的关注实体。"""
    started = time.perf_counter()
    settings = validate_planning_object_config((config or {}).get('planning_objects'))
    reference, anchor, prefix_length = _selection_reference(scene)
    reference_length = float(polyline_lengths(reference)[-1])
    ego_s, ego_d, ego_heading = project_polyline(scene.ego.position, reference,
        previous_s=anchor, heading=scene.ego.heading)
    # 参考线的连续弧长支路搜索窗口足以覆盖裁剪范围与当前实体外廓。
    visible = [actor for actor in scene.actors if actor.visible]
    largest_body = max((np.hypot(a.length, a.width) for a in [scene.ego]+visible), default=0.)
    window = max(settings['front_m'], settings['rear_m'])+largest_body+5.
    bands, road = _lane_bands(scene, reference, ego_s, ego_d, window)
    ego_interval = project_polyline(_body_points(scene.ego), reference, previous_s=ego_s, window=window)[0]
    ego_lo, ego_hi = float(ego_interval.min()), float(ego_interval.max())
    candidates = {name: [] for name in ROLES}
    within, excluded, object_roles, selection_details = {}, {}, {}, {}
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
        if delta_s < -settings['rear_m'] or delta_s > settings['front_m']:
            excluded[actor.id] = 'outside_longitudinal_center_range'
            continue
        if abs(delta_d) > settings['lateral_m']:
            excluded[actor.id] = 'outside_lateral_center_range'
            continue
        ranged.append((actor, float(d), delta_s, delta_d, _body_points(actor)))
    body_s = body_d = np.empty(0)
    if ranged:
        body_s, body_d, _ = project_polyline(np.concatenate([row[4] for row in ranged]),
            reference, previous_s=ego_s, window=window)
    offset = 0
    for actor, d, delta_s, delta_d, points in ranged:
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
        used_fallback = not columns
        if used_fallback:
            # 横向交叉来车、无lane_id行人以及路边障碍物不能因为不在三条
            # 同向车道上就全部漏掉。以同一 Frenet 截面的左右侧方补充归格。
            lo, hi = bands['center']
            columns = ['left' if d >= hi else 'right' if d <= lo else 'center']
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
            'side_gap_m': side_gap, 'role_source': 'frenet_side_fallback' if used_fallback else 'physical_lane_band'}
    winners = {role: min(rows)[2] if rows else None for role, rows in candidates.items()}
    base_ids = sorted({identity for identity in winners.values() if identity is not None})
    selected_ids = set(base_ids)
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
    diagnostic = {'version': VERSION, 'settings': settings,
        'reference': 'confirmed_navigation_frenet;s_forward;d_left_positive;meter',
        'public_current_lane_prefix_m': prefix_length,
        'ego_s_m': float(ego_s), 'ego_d_m': float(ego_d), 'road': road,
        'lane_bands_d_m': {key: list(value) for key, value in bands.items()},
        'region_objects': winners, 'base_selected_ids': base_ids,
        'base_selected_count': len(base_ids), 'attention_added_ids': attention_added,
        'attention_added_count': len(attention_added),
        'attention_selected_ids': attention_selected, 'attention_rejected': attention_rejected,
        'selected_ids': sorted(selected_ids), 'selected_count': len(selected),
        'visible_count': len(visible), 'within_range_count': len(within),
        'selected_roles': {key: object_roles[key] for key in sorted(selected_ids)},
        'selected_state': {key: selection_details[key] for key in sorted(selected_ids)},
        'excluded': [{'id': key, 'reason': excluded[key]} for key in sorted(excluded)],
        'selection_seconds': time.perf_counter()-started,
        'limitation': '局部代表对象策略不保证对被排除对象的全时域避碰；完整观测和独立审计未被删除'}
    return selected, diagnostic


def planning_view(scene, predictions, config=None, answer=None):
    """保留原帧和原始数据，只给本次MPC生成局部浅拷贝视图。"""
    selected, diagnostic = select_planning_objects(scene, config, answer)
    identities = {actor.id for actor in selected}
    scoped_scene = replace(scene, actors=selected)
    scoped_predictions = replace(predictions,
        actors={key: predictions.actors[key] for key in sorted(identities)},
        trajectories={key: predictions.trajectories[key] for key in sorted(identities)},
        diagnostics={key: value for key, value in predictions.diagnostics.items()
                     if key in identities or key not in predictions.actors})
    return scoped_scene, scoped_predictions, diagnostic
