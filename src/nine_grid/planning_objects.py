"""轻量 Frenet 八向规划对象筛选；与检索九宫格的表示和观测掩码分开。

本模块在EKF之前缩小规划实体集合，预测、MPC、评分和执行验收使用同一集合。
先执行与模型一致的车道范围准入，再选八向代表。完整场景仅供图像和审计。
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import time
import hashlib
import json
import numpy as np


VERSION = 'frenet-eight-planning-3.0'
DEFAULTS = {
    'version': VERSION,
    'front_m': 40.,
    'rear_m': 20.,
    'lateral_m': 25.,
    'include_attention': True,
    'max_attention_objects': 3,
    'adaptive_front': False,
    'reaction_s': 1.0,
    'braking_mps2': 4.0,
    'stopping_margin_m': 5.0,
    'conflict_horizon_s': 3.0,
    'conflict_margin_m': .5,
}
ROLES = ('left_front', 'front', 'right_front', 'left_side', 'right_side',
         'left_rear', 'rear', 'right_rear')
KINDS = ('vehicle', 'bicycle', 'pedestrian', 'static_obstacle')
RULE_OBSERVATION_VERSION = 'road-rule-current-observations-1.0'


def road_rule_observations(scene):
    """冻结车道范围内、八向选代表之前的路权对象，不恢复范围外真值。"""
    if scene.diagnostics.get('planning_scope') is None:
        from .object_scope import scope_scene
        scene, _ = scope_scene(scene)
    if not scene.rules.get('road_priority_available'):
        return None
    lane_ids={lane.id for lane in scene.lanes}
    current={actor.id:actor for actor in scene.actors
        if actor.visible and actor.kind in ('vehicle','bicycle') and actor.lane_id in lane_ids}
    def record(actor):
        return dict(id=actor.id,kind=actor.kind,position=np.asarray(actor.position,float).tolist(),
            velocity=np.asarray(actor.velocity,float).tolist(),heading=float(actor.heading),
            length=float(actor.length),width=float(actor.width),lane_id=actor.lane_id)
    existing=scene.diagnostics.get('road_rule_observations')
    if existing is None:
        if scene.diagnostics.get('planning_scope') is not None:
            raise ValueError('已筛选规划场景缺少原帧路权观测快照，不能以局部对象冒充完整规则输入')
        existing=dict(version=RULE_OBSERVATION_VERSION,scene_version=scene.version,
            frame_id=scene.frame_id,time=float(scene.time),coordinates=scene.coordinates,
            actors=[record(current[key]) for key in sorted(current)])
    if (not isinstance(existing,dict) or existing.get('version')!=RULE_OBSERVATION_VERSION
            or existing.get('scene_version')!=scene.version or existing.get('frame_id')!=scene.frame_id
            or existing.get('coordinates')!=scene.coordinates or scene.coordinates!='world_xy_m'
            or not isinstance(existing.get('time'),(int,float))
            or not np.isclose(existing['time'],scene.time,rtol=0.,atol=1e-8)
            or not isinstance(existing.get('actors'),list)):
        raise ValueError('路权当前观测快照的版本、帧、时间或坐标与规划输入不一致')
    seen={}
    fields={'id','kind','position','velocity','heading','length','width','lane_id'}
    for row in existing['actors']:
        if (not isinstance(row,dict) or set(row)!=fields or not isinstance(row['id'],str)
                or not row['id'] or row['id'] in seen or row['kind'] not in ('vehicle','bicycle')
                or row['lane_id'] not in lane_ids):
            raise ValueError('路权当前观测对象身份、类型或公开车道非法')
        position=np.asarray(row['position'],float);velocity=np.asarray(row['velocity'],float)
        numeric=np.asarray([row['heading'],row['length'],row['width']],float)
        if (position.shape!=(2,) or velocity.shape!=(2,) or not np.isfinite(position).all()
                or not np.isfinite(velocity).all() or not np.isfinite(numeric).all()
                or min(numeric[1:])<=0):
            raise ValueError('路权当前观测对象几何或运动数值非法')
        seen[row['id']]=row
    if ((scene.diagnostics.get('planning_scope') is None and set(seen)!=set(current))
            or any(key not in seen or seen[key]!=record(actor) for key,actor in current.items())):
        raise ValueError('路权快照遗漏或不匹配本帧当前观测，禁止静默使用旧对象状态')
    return deepcopy(existing)


def validate_planning_object_config(options=None):
    """开发默认值是可配置起点，未知字段/非法数值不能静默改变筛选含义。"""
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options)-set(DEFAULTS):
        raise ValueError('planning_objects 包含未知字段或不是配置字典')
    result = {**DEFAULTS, **options}
    if result['version'] not in (VERSION, 'frenet-eight-planning-2.0'):
        raise ValueError('planning_objects.version 不受支持')
    result['version'] = VERSION
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


def _selected_input_signature(scene, actors):
    """筛选子集再入规划时验明全部准入几何，防止同帧可变数组绕过筛选。"""
    def entity(actor):
        return dict(id=actor.id, kind=actor.kind, position=np.asarray(actor.position).tolist(),
            velocity=np.asarray(actor.velocity).tolist(), acceleration=np.asarray(actor.acceleration).tolist(),
            heading=float(actor.heading), length=float(actor.length), width=float(actor.width),
            lane_id=actor.lane_id, visible=bool(actor.visible), radius=actor.radius)
    def json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError('规划身份签名包含不支持的值：'+type(value).__name__)
    value=dict(frame_id=scene.frame_id,time=float(scene.time),ego=entity(scene.ego),
        actors=[entity(actor) for actor in sorted(actors,key=lambda item:item.id)],
        lane_map_complete=scene.diagnostics.get('lane_map_complete'),
        lanes=[dict(id=lane.id,center=lane.center,polygon=lane.polygon,width=lane.width,
            successors=lane.successors,left=lane.left,right=lane.right,boundaries=lane.boundaries)
            for lane in sorted(scene.lanes,key=lambda item:item.id)],
        route=scene.route, route_lane_ids=scene.route_lane_ids)
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),
                   allow_nan=False,default=json_value).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def select_planning_objects(scene, config=None, answer=None):
    """只在已准入车道/弧长范围内选八向代表、冲突补充和最多3关注对象。"""
    from .object_scope import scope_scene, scope_settings, ROLES
    started = time.perf_counter()
    options = (config or {}).get('planning_objects')
    existing = scene.diagnostics.get('planning_scope')
    if options is None:
        options = (existing['settings'] if isinstance(existing,dict) and existing.get('version') == VERSION
                   else scope_settings(scene))
    settings = validate_planning_object_config(options)
    if (isinstance(existing, dict) and existing.get('version') == VERSION
            and existing.get('frame_id') == scene.frame_id and existing.get('time') == scene.time
            and existing.get('settings') == settings
            and existing.get('selected_ids') == sorted(a.id for a in scene.actors)):
        if existing.get('selected_input_signature') != _selected_input_signature(scene, scene.actors):
            raise ValueError('已筛选规划场景的当前对象或车道几何发生变化，须从新的完整冻结帧重新筛选')
        # 验明身份后保留原选择中的冲突/关注补充，不从子集重选而缩掉对象。
        return list(scene.actors), deepcopy(existing)
    scoped, scope = scope_scene(scene, config)
    within = {actor.id: actor for actor in scoped.actors}
    excluded = {row['id']: row['reason'] for row in scope['excluded']}
    candidates = {role: [] for role in ROLES}
    exceptions = {}
    speed = float(np.linalg.norm(scene.ego.velocity))
    for identity, actor in within.items():
        row = scope['objects'][identity]
        reasons = []
        for role, evidence in row['memberships'].items():
            gap = evidence['side_gap_m'] if role.endswith('side') else evidence['longitudinal_gap_m']
            candidates[role].append((gap, abs(evidence['delta_s_m']), identity))
            heading = evidence['lane_heading_rad']
            along = np.array([np.cos(heading), np.sin(heading)])
            actor_speed = float(actor.velocity @ along)
            ego_speed = float(scene.ego.velocity @ along)
            closing = ego_speed-actor_speed if evidence['delta_s_m'] >= 0 else actor_speed-ego_speed
            if closing > 0 and evidence['longitudinal_gap_m'] <= closing*settings['conflict_horizon_s']+settings['conflict_margin_m']:
                reasons.append('in_scope_longitudinal_conflict')
            lateral_speed = abs(float(actor.velocity @ np.array([-along[1], along[0]])))
            if actor.kind != 'static_obstacle' and lateral_speed > .2:
                reasons.append('in_scope_observed_lateral_motion')
        if reasons:
            exceptions[identity] = sorted(set(reasons))
    winners = {role: min(rows)[2] if rows else None for role, rows in candidates.items()}
    base_ids = sorted({key for key in winners.values() if key is not None})
    selected_ids = set(base_ids) | set(exceptions)
    exception_added = sorted(set(exceptions)-set(base_ids))
    attention_selected, attention_added, attention_rejected = [], [], []
    if answer is not None:
        seen = set()
        for item in sorted(answer.attention, key=lambda row: (row['rank'], row['id'])):
            identity = item['id']
            if identity in seen:
                continue
            seen.add(identity)
            if identity in selected_ids:
                attention_selected.append(identity)
            elif identity not in within:
                attention_rejected.append(dict(id=identity, reason=excluded.get(identity, 'outside_current_object_scope')))
            elif not settings['include_attention']:
                attention_rejected.append(dict(id=identity, reason='attention_expansion_disabled'))
            elif len(attention_added) >= settings['max_attention_objects']:
                attention_rejected.append(dict(id=identity, reason='attention_expansion_capacity'))
            else:
                selected_ids.add(identity)
                attention_selected.append(identity)
                attention_added.append(identity)
    for identity in within:
        if identity not in selected_ids:
            excluded[identity] = 'not_nearest_in_any_eight_role'
    diagnostic = dict(version=VERSION, settings=settings, frame_id=scene.frame_id, time=float(scene.time),
        reference='each_permitted_lane_chain_arc;s_forward;meter',
        object_scope_version=scope['version'], assignment_protocol=scope['assignment_protocol'],
        road=dict(source_lane_id=scene.ego.lane_id,
                  candidate_corridor_lane_ids=sorted({key for trace in scope['chains'].values() if trace for key in trace['lane_ids']})),
        region_objects=winners, base_selected_ids=base_ids, base_selected_count=len(base_ids),
        effective_front_m=scope['settings']['front_m'], effective_rear_m=scope['settings']['rear_m'],
        stopping_distance_m=speed*settings['reaction_s']+speed**2/(2*settings['braking_mps2'])+settings['stopping_margin_m'],
        adaptive_expansion=False, legacy_adaptive_front_requested=settings['adaptive_front'],
        exception_added_ids=exception_added, exception_added_count=len(exception_added),
        direct_conflict_reasons=exceptions,
        conflict_screen='仅限已准入车道弧长范围的当前速度冲突补充，不扩展对象范围',
        attention_added_ids=attention_added, attention_added_count=len(attention_added),
        attention_selected_ids=attention_selected, attention_rejected=attention_rejected,
        selected_ids=sorted(selected_ids), selected_count=len(selected_ids),
        visible_count=sum(a.visible for a in scene.actors), within_range_count=len(within),
        selected_roles={key:scope['objects'][key]['roles'] for key in sorted(selected_ids)},
        selected_state={key:deepcopy(scope['objects'][key]) for key in sorted(selected_ids)},
        excluded=[dict(id=key,reason=excluded[key]) for key in sorted(excluded)],
        selection_seconds=time.perf_counter()-started,
        limitation='固定车道弧长范围和八向代表不保证覆盖范围外或未纳入对象的危险')
    selected = [within[key] for key in sorted(selected_ids)]
    diagnostic['selected_input_signature'] = _selected_input_signature(scene, selected)
    return selected, diagnostic


def planning_scene(scene, config=None, answer=None):
    """先限定共同车道范围，再缩为EKF集合；原始场景留给图像及审计。"""
    from .object_scope import scope_scene
    existing = scene.diagnostics.get('planning_scope')
    if isinstance(existing,dict) and existing.get('version') == VERSION:
        # 必须在scope删除越界对象之前验证；否则变动对象可能被删掉后绕过签名，
        # 而旧路权快照仍携带它的原状态。
        if (existing.get('frame_id') != scene.frame_id or existing.get('time') != scene.time
                or existing.get('selected_ids') != sorted(actor.id for actor in scene.actors)
                or existing.get('selected_input_signature') != _selected_input_signature(scene, scene.actors)):
            raise ValueError('已筛选规划场景的当前对象或车道几何发生变化，须从新的完整冻结帧重新筛选')
        options = (config or {}).get('planning_objects')
        if options is not None and validate_planning_object_config(options) != existing['settings']:
            raise ValueError('已筛选规划场景不能用新配置扩大或重解释范围，须从完整原帧重新筛选')
    scene, object_scope = scope_scene(scene, config)
    rule_observations=road_rule_observations(scene)
    selected,diagnostic=select_planning_objects(scene,config,answer)
    diagnostics={**scene.diagnostics,'planning_scope':deepcopy(diagnostic)}
    if rule_observations is not None:diagnostics['road_rule_observations']=rule_observations
    scoped=replace(scene,actors=selected,diagnostics=diagnostics)
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
