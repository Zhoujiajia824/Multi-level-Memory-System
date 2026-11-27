"""优化对象为车道范围内全部实体与最多三个模型关注对象的并集。

八向代表只描述检索角色，不再截断势场/碰撞约束。BEV感知与EKF集合
在 perception_scope 中独立构建，独立执行审计可检查更大的感知集合。
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import time
import hashlib
import json
import numpy as np
from .lane_roles import ASSIGNMENT_PROTOCOL


VERSION = 'lane-all-attention-planning-4.0'
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
    """冻结当前感知候选中的公开路权对象，不读取BEV/车道范围外真值。"""
    if scene.diagnostics.get('planning_scope') is None:
        from .perception_scope import perception_scene
        scene, _ = perception_scene(scene)
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
    if result['version'] not in (VERSION, 'frenet-eight-planning-3.0', 'frenet-eight-planning-2.0'):
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


def _attention_signature(answer):
    if answer is None:
        return []
    items = answer.attention
    if not isinstance(items, list) or len(items) > 3:
        raise ValueError('规划关注对象最多三个')
    identities, ranks, rows = set(), set(), []
    for item in items:
        identity, rank = item.get('id'), item.get('rank')
        if (not isinstance(identity, str) or not identity or identity in identities
                or type(rank) is not int or rank not in (1, 2, 3) or rank in ranks):
            raise ValueError('规划关注对象ID和rank必须唯一且合法')
        identities.add(identity); ranks.add(rank)
        rows.append(dict(id=identity, rank=rank))
    return sorted(rows, key=lambda row: (row['rank'], row['id']))


def select_planning_objects(scene, config=None, answer=None):
    """八向范围内全部对象 + 感知候选中最多三个关注对象，按ID去重。"""
    from .object_scope import scope_settings, ROLES
    from .perception_scope import perception_scene
    started = time.perf_counter()
    options = (config or {}).get('planning_objects')
    existing = scene.diagnostics.get('planning_scope')
    if isinstance(existing, dict) and (existing.get('assignment_protocol') != ASSIGNMENT_PROTOCOL
                                     or existing.get('version') != VERSION):
        raise ValueError('已筛选规划场景的对象归属协议已改变，须从完整冻结帧重新筛选')
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
        if answer is not None and _attention_signature(answer) != existing.get('attention_signature'):
            raise ValueError('已筛选规划场景不能重解释关注对象，须从完整冻结帧重新筛选')
        # 验明身份后保留原选择；子集不能冒充完整感知候选。
        return list(scene.actors), deepcopy(existing)
    perceived, perception = perception_scene(scene, config)
    scope = perception['lane_scope']
    available = {actor.id: actor for actor in perceived.actors}
    within = {key: available[key] for key in perception['lane_scope_ids']}
    excluded = {row['id']: row['reason'] for row in perception['excluded']}
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
    selected_ids = set(within)
    exception_added = sorted(set(exceptions)-set(base_ids))
    attention_selected, attention_added, attention_rejected = [], [], []
    if answer is not None:
        _attention_signature(answer)  # 包括已在范围内的关注也不能绕过三个上限。
        seen = set()
        for item in sorted(answer.attention, key=lambda row: (row['rank'], row['id'])):
            identity = item['id']
            if identity in seen:
                continue
            seen.add(identity)
            if identity in selected_ids:
                attention_selected.append(identity)
            elif identity not in available:
                attention_rejected.append(dict(id=identity, reason=excluded.get(identity, 'outside_current_perception_scope')))
            elif not settings['include_attention']:
                attention_rejected.append(dict(id=identity, reason='attention_expansion_disabled'))
            elif len(attention_added) >= settings['max_attention_objects']:
                attention_rejected.append(dict(id=identity, reason='attention_expansion_capacity'))
            else:
                selected_ids.add(identity)
                attention_selected.append(identity)
                attention_added.append(identity)
    for identity in available:
        if identity not in selected_ids:
            excluded[identity] = 'bev_prediction_only_not_model_attention'
    diagnostic = dict(version=VERSION, settings=settings, frame_id=scene.frame_id, time=float(scene.time),
        reference='each_permitted_lane_chain_arc;s_forward;meter',
        object_scope_version=scope['version'], assignment_protocol=scope['assignment_protocol'],
        road=dict(source_lane_id=scene.ego.lane_id,
                  candidate_corridor_lane_ids=sorted({key for trace in scope['chains'].values() if trace for key in trace['lane_ids']})),
        region_objects=winners, base_selected_ids=base_ids, base_selected_count=len(base_ids),
        lane_scope_ids=sorted(within), lane_scope_count=len(within),
        perception_ids=perception['selected_ids'], perception_count=perception['selected_count'],
        prediction_only_ids=sorted(set(available)-selected_ids),
        optimization_policy='all_lane_scope_objects_union_model_attention;representatives_for_retrieval_only',
        attention_signature=_attention_signature(answer),
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
        selected_roles={key:scope['objects'].get(key, {}).get('roles', []) for key in sorted(selected_ids)},
        selected_state={key:deepcopy(perception['objects'][key]) for key in sorted(selected_ids)},
        excluded=[dict(id=key,reason=excluded[key]) for key in sorted(excluded)],
        selection_seconds=time.perf_counter()-started,
        limitation='BEV内未关注且不在车道范围的对象只预测；不声称全部感知对象均参与势场优化')
    selected = [available[key] for key in sorted(selected_ids)]
    diagnostic['selected_input_signature'] = _selected_input_signature(scene, selected)
    return selected, diagnostic


def planning_scene(scene, config=None, answer=None):
    """仅建立优化局部视图，不截断上游完整感知预测集合。"""
    from .perception_scope import perception_scene
    existing = scene.diagnostics.get('planning_scope')
    if isinstance(existing, dict) and (existing.get('assignment_protocol') != ASSIGNMENT_PROTOCOL
                                     or existing.get('version') != VERSION):
        raise ValueError('已筛选规划场景的对象归属协议已改变，须从完整冻结帧重新筛选')
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
        selected, diagnostic = select_planning_objects(scene, config, answer)
        return replace(scene, actors=selected), diagnostic
    scene, perception = perception_scene(scene, config)
    rule_observations=road_rule_observations(scene)
    selected,diagnostic=select_planning_objects(scene,config,answer)
    diagnostics={**scene.diagnostics,'planning_scope':deepcopy(diagnostic)}
    if rule_observations is not None:diagnostics['road_rule_observations']=rule_observations
    scoped=replace(scene,actors=selected,diagnostics=diagnostics)
    return scoped,diagnostic


def planning_view(scene, predictions, config=None, answer=None):
    """保留原帧和原始数据，只给本次MPC生成局部浅拷贝视图。"""
    scoped_scene, diagnostic = planning_scene(scene, config, answer)
    navsim = scene.scenario == 'navsim' and isinstance((config or {}).get('navsim_execution'), dict)
    if navsim and scene.diagnostics.get('planning_scope') is None:
        additions, evidence = _navsim_predicted_conflicts(scene, predictions, config, diagnostic['selected_ids'])
        diagnostic = deepcopy(diagnostic)
        diagnostic['predicted_conflict_expansion'] = evidence
        if additions:
            perceived = scoped_scene.diagnostics['perception_scope']
            available = {actor.id: actor for actor in scene.actors}
            identities = set(diagnostic['selected_ids']) | set(additions)
            selected = [available[key] for key in sorted(identities)]
            diagnostic.update(selected_ids=sorted(identities), selected_count=len(identities),
                prediction_only_ids=sorted(set(diagnostic['perception_ids'])-identities),
                optimization_policy='lane_scope_union_attention_union_predicted_reachable_corridor_conflicts',
                exception_added_ids=sorted(set(diagnostic['exception_added_ids']) | set(additions)),
                excluded=[row for row in diagnostic['excluded'] if row['id'] not in additions],
                conflict_screen='NAVSIM: same-time predicted body sweep intersects public reachable lane corridor',
                limitation='Only observed causal predictions; conservative 2D admission, not a collision certificate')
            diagnostic['exception_added_count'] = len(diagnostic['exception_added_ids'])
            for key in additions:
                diagnostic['direct_conflict_reasons'][key] = ['predicted_reachable_corridor_conflict']
                diagnostic['selected_roles'][key] = []
                diagnostic['selected_state'][key] = deepcopy(perceived['objects'][key])
            scoped_scene = replace(scoped_scene, actors=selected)
        diagnostic['selected_input_signature'] = _selected_input_signature(scoped_scene, scoped_scene.actors)
        diagnostic['selected_prediction_signature'] = _prediction_signature(predictions, diagnostic['selected_ids'])
        scoped_scene = replace(scoped_scene, diagnostics={**scoped_scene.diagnostics,
            'planning_scope': deepcopy(diagnostic)})
    elif navsim and diagnostic.get('selected_prediction_signature') is not None:
        if diagnostic['selected_prediction_signature'] != _prediction_signature(predictions, diagnostic['selected_ids']):
            raise ValueError('已筛选NAVSIM规划视图的预测已变化，须从完整原帧重新筛选')
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


def _prediction_signature(predictions, identities):
    value = dict(times=np.asarray(predictions.times).tolist(), trajectories={
        key: np.asarray(predictions.trajectories[key]).tolist() for key in sorted(identities)})
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def _navsim_predicted_conflicts(scene, predictions, config, selected_ids):
    """补入已感知、已预测且可能在本次时域占用合法走廊的实体。

    只扩大下游优化集合。模型输入和八向量程不变，不读取评分未来。
    加减速包络故意放松jerk，形成保守准入粗筛；没有安全验收或制动决定。
    当前二维输入无法判别立交高度，不能将本筛选解释为三维安全证明。
    """
    from shapely.geometry import LineString
    from shapely.ops import unary_union, substring
    from .geometry import actor_shape, wrap_angle
    from .object_scope import lane_role_geometry
    from .lane_roles import project

    started = time.perf_counter()
    p = config['mpc']
    horizon = float(p['horizon_steps']*p['step_s'])
    times = np.asarray(predictions.times, float)
    if (predictions.frame_id != scene.frame_id or predictions.coordinates != scene.coordinates
            or not np.isclose(predictions.time, scene.time, rtol=0., atol=1e-8)
            or times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or abs(times[0]) > 1e-8 or np.any(np.diff(times) <= 0)
            or times[-1] < horizon-1e-8):
        raise ValueError('NAVSIM冲突纳入必须使用同帧完整时域的当前观测预测')
    allowed = set(scene.diagnostics.get('perception_scope', {}).get('selected_ids', []))
    # A raw snapshot is also accepted by planning_view; its newly frozen
    # perception selection is recomputed without restoring hidden objects.
    if scene.diagnostics.get('perception_scope') is None:
        from .perception_scope import select_perception_objects
        _, perception = select_perception_objects(scene, config)
        allowed = set(perception['selected_ids'])
    available = {actor.id: actor for actor in scene.actors
        if actor.visible and actor.id in allowed and actor.kind in KINDS}
    candidates = sorted(set(available)-set(selected_ids))
    evidence = dict(version='navsim-predicted-reachable-corridor-1.0', horizon_s=horizon,
        candidate_count=len(candidates), added_ids=[], added_count=0, conflicts={},
        source='current_perception_and_existing_causal_prediction_only',
        model_input_and_eight_regions_unchanged=True,
        limitation='Piecewise predicted sweeps and relaxed reachability are admission only; no future truth or height discrimination')
    if not candidates:
        evidence['selection_seconds'] = time.perf_counter()-started
        return {}, evidence
    speed = float(np.linalg.norm(scene.ego.velocity))
    accel = max(0., float(p['acceleration_max_mps2']))
    brake = max(1e-6, -float(p['acceleration_min_mps2']))
    lateral_accel = float(p['lateral_acceleration_abs_max_mps2'])
    margin = float(config.get('planning_objects', {}).get('conflict_margin_m', .5))
    maximum_speed = max(speed, float(p['maximum_speed_mps']))

    def upper_reach(t):
        accelerating = min(t, max(0., (maximum_speed-speed)/accel)) if accel > 0 else 0.
        return speed*accelerating+.5*accel*accelerating**2+maximum_speed*(t-accelerating) if accel > 0 else speed*t

    front = upper_reach(horizon)+scene.ego.length/2+margin
    geometry_scene = replace(scene, diagnostics=deepcopy(scene.diagnostics))
    traces, _, cache = lane_role_geometry(geometry_scene, front_m=max(40., front), rear_m=20.)
    center = traces.get('center')
    if center is None:
        evidence.update(status='no_public_current_lane_corridor', selection_seconds=time.perf_counter()-started)
        return {}, evidence
    line = LineString(center['points'])
    ego_s = float(center['ego_s'])
    _, _, heading, offcenter = project(center['points'], scene.ego.position)
    lateral_speed = abs(float(scene.ego.velocity @ np.array([-np.sin(heading), np.cos(heading)])))
    shapes = []
    for trace in traces.values():
        if trace is None:
            continue
        for key in trace['lane_ids']:
            if key not in cache:
                continue
            lane, polygon, native = cache[key]
            if np.isfinite(lane.width) and lane.width > 0:
                shapes.append(polygon.intersection(native.buffer(float(lane.width)/2, cap_style=2)))
        shapes.extend(link['shape'] for link in trace.get('navigation_connections', []))
    domain = unary_union(shapes).intersection(scene.drivable)
    if domain.is_empty:
        evidence.update(status='empty_public_current_lane_corridor', selection_seconds=time.perf_counter()-started)
        return {}, evidence
    corridors = []
    for index, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
        if t0 >= horizon-1e-8:
            break
        t1 = min(float(t1), horizon)
        braking_t = min(float(t0), speed/brake)
        lower = speed*braking_t-.5*brake*braking_t**2
        a = max(0., ego_s+lower-scene.ego.length/2-margin)
        b = min(line.length, ego_s+upper_reach(t1)+scene.ego.length/2+margin)
        if a >= b:
            continue
        lateral = offcenter+scene.ego.width/2+lateral_speed*t1+.5*lateral_accel*t1*t1+margin
        shape = substring(line, a, b).buffer(lateral, cap_style=2).intersection(domain)
        if not shape.is_empty:
            corridors.append((index, float(t0), t1, shape, np.asarray(shape.bounds)))
    additions = {}
    for identity in candidates:
        actor = available[identity]
        if identity not in predictions.actors or identity not in predictions.trajectories:
            raise ValueError('当前感知冲突候选缺少既有预测：'+identity)
        trajectory = np.asarray(predictions.trajectories[identity], float)
        if (trajectory.shape != (len(times), 4) or not np.isfinite(trajectory).all()
                or not np.allclose(trajectory[0, :2], actor.position, rtol=0., atol=1e-8)):
            raise ValueError('当前感知冲突候选的预测轨迹或首点无效：'+identity)
        radius = float(np.hypot(actor.length, actor.width)/2 if actor.kind != 'pedestrian'
                       else actor.radius or actor.width/2)
        for index, t0, t1, corridor, bounds in corridors:
            first = trajectory[index]
            fraction = (t1-times[index])/(times[index+1]-times[index])
            last = first+(trajectory[index+1]-first)*fraction
            last[2] = first[2]+fraction*float(wrap_angle(trajectory[index+1, 2]-first[2]))
            low = np.minimum(first[:2], last[:2])-radius
            high = np.maximum(first[:2], last[:2])+radius
            if np.any(high < bounds[:2]) or np.any(low > bounds[2:]):
                continue
            a = actor_shape(actor, first[:2], first[2])
            b = actor_shape(actor, last[:2], last[2])
            swept = a.union(b).convex_hull
            # Bound the bulge of a rotating corner between the two endpoint
            # boxes. This is geometry-derived, not another safety-distance knob.
            rotation = abs(float(wrap_angle(last[2]-first[2])))
            if actor.kind != 'pedestrian' and rotation > 1e-8:
                swept = swept.buffer(radius*(1-np.cos(rotation/2)))
            if swept.intersects(corridor):
                additions[identity] = dict(reason='predicted_reachable_corridor_conflict',
                    time_interval_s=[t0, t1], current_lane_id=actor.lane_id)
                break
    evidence.update(status='complete', front_reach_m=front, acceleration_max_mps2=accel,
        braking_mps2=brake, lateral_acceleration_max_mps2=lateral_accel, margin_m=margin,
        reference_front_coverage_m=max(0., line.length-ego_s),
        reference_coverage_shortfall_m=max(0., front-(line.length-ego_s)),
        added_count=len(additions), added_ids=sorted(additions), conflicts=deepcopy(additions),
        lane_ids=sorted({key for trace in traces.values() if trace for key in trace['lane_ids']}),
        selection_seconds=time.perf_counter()-started)
    return additions, evidence
