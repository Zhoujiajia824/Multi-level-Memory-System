"""原帧 Frenet 目标、一次纵向到达代价及不可逆的目标生命周期。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np
from shapely.geometry import Point, Polygon, LineString

from .geometry import (clean_polyline, polyline_lengths, project_polyline,
                       sample_polyline, rectangle_corners, wrap_angle)
from .quintic_reference import quintic_offsets, initial_lateral_state, spatial_samples


DEFAULTS = {
    'minimum_delta_s_m': 2., 'maximum_delta_s_m': 70.,
    'maximum_lateral_slope': .35, 'passed_tolerance_m': .001,
    'projection_window_m': 120., 'attraction_weight': 3.,
    'longitudinal_scale_m': 30., 'lateral_scale_m': 3.5,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def build_goal_reference(scene, config=None, request_version=0):
    """冻结已确认自车导航中心线。与九宫格参考分开命名，不改变检索语义。"""
    from .local_reference import LocalReferenceManager
    parameters = {**DEFAULTS, **(config or {}).get('local_goal', {})}
    reference = clean_polyline(scene.route).copy()
    if len(reference) < 2 or not np.isfinite(reference).all():
        raise ValueError('目标参考线缺失或无效')
    s, d, _ = project_polyline(scene.ego.position, reference,
        previous_s=scene.diagnostics.get('route_s'), heading=scene.ego.heading)
    source_lane = scene.diagnostics.get('stable_navigation_lane_id', scene.ego.lane_id)
    reported_lane=source_lane;confirmation='平台已确认车道'
    reference_valid=True;reference_rejection=None
    confirmed=next((lane for lane in scene.lanes if lane.id==source_lane),None)
    if confirmed is not None and source_lane not in scene.route_lane_ids:
        # 环道连接重叠区的原生lane_index可能短暂落入另一公开分支。
        # 仅在非平行邻道重叠时按自车已知导航确认；不能掩盖尚未更新的平行换道参考。
        parallel=any(lane.id in scene.route_lane_ids and
            (lane.id in (confirmed.left,confirmed.right) or confirmed.id in (lane.left,lane.right))
            for lane in scene.lanes)
        choices=[]
        if not parallel:
            for lane in scene.lanes:
                if lane.id not in scene.route_lane_ids or not Polygon(lane.polygon).covers(Point(scene.ego.position)):
                    continue
                _,offset,tangent=project_polyline(scene.ego.position,lane.center,heading=scene.ego.heading)
                angle=abs(float(wrap_angle(scene.ego.heading-tangent)))
                if angle<np.pi/3:
                    choices.append((angle+abs(offset)/lane.width,lane.id,lane))
        if choices:
            confirmed=min(choices,key=lambda value:value[:2])[2];source_lane=confirmed.id
            confirmation='连接重叠区按公开自车导航、实际位置及朝向确认；未读取周车路线'
    if confirmed is not None:
        if source_lane not in scene.route_lane_ids and any(
                lane.id in scene.route_lane_ids and
                (lane.id in (confirmed.left, confirmed.right) or confirmed.id in (lane.left, lane.right))
                for lane in scene.lanes):
            # 换道已确认后，模型坐标参考随实际行驶车道更新；全局出口导航不变。
            # 不能因全局路线保留原车道ID就继续把目标解释成原中心线的偏移。
            manager = LocalReferenceManager(config or {})
            reference = manager._continuation(scene, confirmed, {lane.id: lane for lane in scene.lanes})
            s, d, _ = project_polyline(scene.ego.position, reference, heading=scene.ego.heading)
            confirmation = '已确认邻道中心，沿原公开导航分支延伸'
        foot=sample_polyline(confirmed.center,project_polyline(scene.ego.position,confirmed.center,
            heading=scene.ego.heading)[0])
        # 导航包含道路内的平滑连接段，不能要求每个点与原始折线中心完全重合。
        # 此项只检查坐标参考身份；不修改道路、车体和实体碰撞验收容差。
        deviation_limit=max(.1,(confirmed.width-scene.ego.width)/2+.1)
        if LineString(reference).distance(Point(foot))>deviation_limit:
            reference_valid=False
            reference_rejection='平台已确认车道与导航中心参考不一致；仅保留导航诊断，不接纳本帧语义目标'
    else:
        reference_valid=False
        reference_rejection='当前已确认车道缺少公开几何；不接纳本帧语义目标'
    version = scene.diagnostics.get('reference_revision', 0)
    identity = _digest({'points': reference.tolist(), 'lanes': scene.route_lane_ids,
                        'stable_lane': source_lane, 'revision': version})
    return {'frame_id': scene.frame_id, 'time': float(scene.time),
        'reference_id': identity, 'reference_version': version,
        'request_version': int(request_version), 'points': reference.tolist(),
        's_ego_m': float(s), 'd_ego_m': float(d), 'source_lane_id': source_lane,
        'reported_lane_id':reported_lane,'lane_confirmation':confirmation,
        'semantic_reference_valid':reference_valid,'reference_rejection':reference_rejection,
        'source_pose': [*scene.ego.position.tolist(), float(scene.ego.heading)],
        'vehicle_constraints': {'steering_abs_max_rad':float((config or {}).get('mpc',{}).get('steering_abs_max_rad',.6)),
            'length_m':float(scene.ego.length),'width_m':float(scene.ego.width),
            'wheelbase_m':scene.ego.wheelbase_m,'rear_axle_to_center_m':scene.ego.rear_axle_to_center_m},
        'limits': parameters, 'route_lane_ids': list(scene.route_lane_ids),
        'local_reference_parameters': deepcopy(LocalReferenceManager(config or {}).p),
        'definition': ('已确认行驶车道中心线，沿已知自车导航连续延伸；s沿道路向前，d左正右负，米'
                       if reference_valid else '仅供诊断的已知导航参考，当前车道身份核验未通过；不允许输出可执行目标')}


def _source_lane(scene, reference):
    lanes = {lane.id: lane for lane in scene.lanes}
    lane = lanes.get(reference['source_lane_id'])
    if lane is None:
        lane = lanes.get(scene.ego.lane_id)
    return lane, lanes


def _candidate_lanes(scene, reference):
    from .local_reference import LocalReferenceManager
    if reference.get('semantic_reference_valid',True) is not True:
        return []
    source, lanes = _source_lane(scene, reference)
    if source is None:
        return []
    if 'local_reference_parameters' not in reference:
        raise ValueError('原帧目标参考缺少局部参考参数，不能用当前默认值重解释历史目标')
    manager = LocalReferenceManager({'local_reference': reference['local_reference_parameters']})
    result = [(source, None)]
    for key in (source.left, source.right):
        if key in lanes:
            reason = manager._legal(scene, source, lanes[key], lanes)
            if (reason is None and key == source.right and not manager.p['allow_right_overtake']
                    and not manager._navigation_change_needed(scene, source, lanes[key], lanes)):
                traffic = manager._traffic(scene, None, source, check_future=False)
                front = traffic['front']
                actor = next((obj for obj in scene.actors if front and obj.id == front['id']), None)
                if (actor is not None and actor.kind in ('vehicle', 'bicycle')
                        and front['speed_mps'] + manager.p['minimum_gain_mps']
                        < float(scene.rules.get('desired_speed', np.linalg.norm(scene.ego.velocity)))):
                    reason = '禁止为绕过本车道慢前车主动从右侧超车；此右换道非导航必需'
            result.append((lanes[key], reason))
    return result


def goal_reference_context(scene, reference):
    """只外发解释目标所需的参考摘要，不发送完整地图或内部控制缓存。"""
    line = np.asarray(reference['points'], float)
    total = polyline_lengths(line)[-1]
    candidates = []
    for lane, reason in _candidate_lanes(scene, reference):
        along = project_polyline(scene.ego.position, lane.center)[0]
        point = sample_polyline(lane.center, along)
        _, lateral, _ = project_polyline(point, line, previous_s=reference['s_ego_m'])
        candidates.append({'lane_id': lane.id, 'center_d_m': float(lateral),
            'width_m': float(lane.width), 'goal_candidate_legal': reason is None,
            'rejection': reason})
    samples = []
    for distance in (0., 10., 20., 40., 70.):
        arc = reference['s_ego_m']+distance
        if arc <= total:
            point = sample_polyline(line, arc)
            delta = point-scene.ego.position
            c, s = np.cos(scene.ego.heading), np.sin(scene.ego.heading)
            samples.append({'delta_s_m': distance,
                'reference_bev_xy_m': [float(c*delta[0]+s*delta[1]), float(-s*delta[0]+c*delta[1])]})
    return {'frame_id': reference['frame_id'], 'time_s': reference['time'],
        'semantic_reference_valid':reference.get('semantic_reference_valid',True),
        'reference_rejection':reference.get('reference_rejection'),
        'reference_id': reference['reference_id'], 'reference_version': reference['reference_version'],
        'request_version': reference['request_version'], 'definition': reference['definition'],
        'source_s_ego_m': reference['s_ego_m'], 'source_d_ego_m': reference['d_ego_m'],
        'delta_s_range_m': [reference['limits']['minimum_delta_s_m'],
            min(reference['limits']['maximum_delta_s_m'], float(total-reference['s_ego_m']))],
        'candidate_lanes': candidates, 'reference_distance_labels': samples,
        'navigation': {'known_ego_route_lane_ids': reference['route_lane_ids'],
            'other_actor_hidden_routes_available': False},
        'rules': {key: scene.rules.get(key) for key in ('desired_speed', 'speed_limit', 'must_wait',
            'wait_reason', 'traffic_lights', 'crosswalks') if key in scene.rules},
        'coordinate_warning': 'BEV横纵刻度是自车笛卡尔米；delta_s是沿本参考线的道路弧长，两者不能直接混用。'}


def goal_seed_path(scene, goal):
    """目标导出的五次空间参考，同时提供初值；不规定速度或到达时间。

    原输入帧冻结参考解释目标，始终经过原世界目标，不吸附。目标之后沿
    该处合法车道平稳归中，仅提供必要道路几何。真实通过后由GoalManager
    释放，不能把本函数结果锁成下一帧仍有效的目标。
    """
    reference = np.asarray(goal['goal_reference']['points'], float)
    s0, d0, heading = project_polyline(scene.ego.position, reference,
        previous_s=goal.get('last_s', goal['goal_reference']['s_ego_m']), heading=scene.ego.heading)
    end = min(polyline_lengths(reference)[-1], max(goal['s_goal_m']+20., s0+120.))
    arc = spatial_samples(max(0.,s0-5.),end,.5,anchors=(s0,goal['s_goal_m']))
    base = sample_polyline(reference, arc)
    headings = project_polyline(base, reference, previous_s=s0, window=160.)[2]
    normal = np.c_[-np.sin(headings), np.cos(headings)]
    span = float(goal['s_goal_m']-s0)
    if span<=1e-6:
        raise ValueError('已纵向通过的目标不能重新生成活动参考')
    from .local_reference import LocalReferenceManager
    manager=LocalReferenceManager({'local_reference':goal['goal_reference']['local_reference_parameters']})
    lanes={lane.id:lane for lane in scene.lanes}
    target=lanes.get(goal.get('target_lane_id'))
    center_d=np.full(len(arc),float(goal['local_goal']['d_m']))
    if target is not None:
        target_path=manager._continuation(scene,target,lanes)
        target_s=project_polyline(base,target_path,heading=heading)[0]
        target_points=sample_polyline(target_path,target_s)
        center_d=np.sum((target_points-base)*normal,axis=1)
    first=np.gradient(center_d,arc);second=np.gradient(first,arc)
    sg=float(goal['s_goal_m']);dg=float(goal['local_goal']['d_m'])
    finish=(dg,float(np.interp(sg,arc,first)),float(np.interp(sg,arc,second)))
    start=initial_lateral_state(scene,reference,s0,d0,heading)
    offset=quintic_offsets(arc-s0,span,start,finish)
    before=arc<s0
    offset[before]=start[0]+(arc[before]-s0)*start[1]+.5*(arc[before]-s0)**2*start[2]
    # 非中心目标可以是换道中间位置；后续回中不能跳变，也不改目标本身。
    after=arc>sg
    tail=min(end-sg,max(20.,float(np.linalg.norm(scene.ego.velocity))*2.))
    if after.any() and tail>1e-6:
        final_s=sg+tail
        final=(float(np.interp(final_s,arc,center_d)),float(np.interp(final_s,arc,first)),
               float(np.interp(final_s,arc,second)))
        offset[after]=quintic_offsets(arc[after]-sg,tail,finish,final)
        finished=arc>=final_s
        offset[finished]=center_d[finished]
    return clean_polyline(base+normal*offset[:, None])


def validate_local_goal(scene, reference, value):
    """严格原帧解释，不夹紧、不吸附、不按新帧重新加 delta_s。"""
    if not isinstance(reference, dict) or reference.get('frame_id') != scene.frame_id or reference.get('time') != scene.time:
        raise ValueError('目标参考与原模型输入帧不一致')
    if reference.get('semantic_reference_valid',True) is not True:
        raise ValueError(reference.get('reference_rejection') or '本帧语义目标参考无效')
    if not isinstance(value, dict) or set(value) != {'delta_s_m', 'd_m'}:
        raise ValueError('local_goal 必须且仅包含 delta_s_m 和 d_m')
    import math
    try:
        finite=all(type(v) in (int,float) and math.isfinite(v) for v in value.values())
    except OverflowError:
        finite=False
    if not finite:
        raise ValueError('局部目标必须是有限数值，不接受布尔值')
    limits = reference['limits']; delta = float(value['delta_s_m']); d = float(value['d_m'])
    if not limits['minimum_delta_s_m'] <= delta <= limits['maximum_delta_s_m']:
        raise ValueError('目标纵向距离超出配置范围')
    line = np.asarray(reference['points'], float); arc = polyline_lengths(line)
    sg = float(reference['s_ego_m']+delta)
    if sg > arc[-1]-1e-6 or sg < 0:
        raise ValueError('目标超出原帧参考线覆盖，不允许端点夹紧')
    point = sample_polyline(line, sg)
    heading = project_polyline(point, line, previous_s=sg, window=5.)[2]
    world = point+d*np.array([-np.sin(heading), np.cos(heading)])
    if scene.drivable is None or not scene.drivable.covers(Point(world)):
        raise ValueError('目标不在真实可行驶区域')
    if abs(d-reference['d_ego_m']) > delta*limits['maximum_lateral_slope']:
        raise ValueError('目标横向位移与前向距离不满足几何可达性')
    candidates = _candidate_lanes(scene, reference)
    target = None;rejected_reasons=[]
    from .local_reference import LocalReferenceManager
    manager = LocalReferenceManager({'local_reference': reference['local_reference_parameters']})
    _, lanes = _source_lane(scene, reference)
    best = float('inf')
    for lane, reason in candidates:
        if reason is not None:
            if Polygon(lane.polygon).covers(Point(world)):
                rejected_reasons.append(reason)
            continue
        # 当前车道的已核查导航含连接段；邻道只能沿相容公开后继。
        path = line if lane.id == reference['source_lane_id'] else manager._continuation(scene, lane, lanes)
        _, lateral, tangent = project_polyline(world, path, heading=heading)
        if abs(lateral) > lane.width/2-scene.ego.width/2+1e-6:
            continue
        corners = rectangle_corners(world, tangent, scene.ego.length, scene.ego.width)
        if not scene.drivable.covers(Polygon(corners)):
            continue
        if abs(lateral) < best:
            best = abs(lateral); target = lane
    if target is None:
        if rejected_reasons:
            raise ValueError('目标被拒绝：'+'；'.join(dict.fromkeys(rejected_reasons)))
        raise ValueError('目标车体不属于合法且导航相容的车道走廊')
    payload = {'local_goal': dict(value), 'goal_reference': deepcopy(reference),
        'world_goal': world, 'target_lane_id': target.id, 's_goal_m': sg}
    seed = goal_seed_path(scene, payload)
    seed_s=project_polyline(seed,line,previous_s=reference['s_ego_m'],window=160.)[0]
    # 只检查原帧当前中心开始的未来连接，不能把初值查询用的身后段算作未来折弯。
    before=(seed_s>=reference['s_ego_m']-1e-6)&(seed_s<=sg+.1)
    if not manager._road_ok(scene, seed[before]):
        raise ValueError('目标几何连接不满足真实车体道路或不可跨越边界')
    section=seed[before]
    headings=np.unwrap(np.arctan2(np.diff(section[:,1]),np.diff(section[:,0])))
    lengths=np.linalg.norm(np.diff(section,axis=0),axis=1)
    if len(headings)>1:
        curvature=np.abs(np.diff(headings))/np.maximum((lengths[:-1]+lengths[1:])/2,1e-9)
        # 仅排除方向盘极限下也无法通过的几何弯曲；不指定通过速度或到达时间。
        from .realtime_solver import bicycle_geometry
        constraints=reference['vehicle_constraints']
        wheelbase,rear=bicycle_geometry(constraints['length_m'],constraints.get('wheelbase_m'),
                                       constraints.get('rear_axle_to_center_m'))
        maximum=np.sin(np.arctan(rear/wheelbase*np.tan(constraints['steering_abs_max_rad'])))/rear
        if np.any(curvature>maximum+1e-6):
            raise ValueError('目标连接曲率超过车辆转弯几何能力')
    payload['goal_id'] = _digest({'frame_id': scene.frame_id, 'source_time': scene.time,
        'reference_id': reference['reference_id'], 'request_version': reference['request_version'], 'local_goal': value})
    return payload


def _project_sequence(positions, reference, initial_s):
    """折线活动段的值和弧长导数，连续搜索抑制环道重投影跳段。"""
    reference = clean_polyline(reference)
    segments = np.diff(reference, axis=0); lengths = np.linalg.norm(segments, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]; units = segments/lengths[:, None]
    aligned = np.sum(units[:-1]*units[1:], axis=1) > 1.-1e-12
    values=[]; gradients=[]; previous=initial_s
    for position in np.asarray(positions):
        allowed = np.ones(len(segments), bool) if previous is None else ((arc[:-1]<=previous+120.) & (arc[1:]>=previous-120.))
        indices = np.flatnonzero(allowed)
        fractions = np.sum((position-reference[indices])*segments[indices], axis=1)/lengths[indices]**2
        clipped = np.clip(fractions, 0., 1.)
        error = position-reference[indices]-clipped[:, None]*segments[indices]
        distance = np.sum(error**2, axis=1)
        choices = np.flatnonzero(distance <= distance.min()+1e-12)
        j = choices[0] if previous is None else choices[np.argmin(abs(arc[indices[choices]]+clipped[choices]*lengths[indices[choices]]-previous))]
        index = indices[j]; fraction = fractions[j]
        s = arc[index]+clipped[j]*lengths[index]
        interior = 0.<fraction<1.
        interior |= fraction==0. and index>0 and aligned[index-1]
        interior |= fraction==1. and index<len(aligned) and aligned[index]
        values.append(s); gradients.append(units[index] if interior else np.zeros(2));previous=s
    return np.asarray(values), np.asarray(gradients)


def first_arrival_residual(positions, reference, goal_s, world_goal, sqrt_weight, initial_s=None):
    """首次纵向截面线性插值；返回二维残差、全部位置Jacobian和离散事件证据。"""
    positions=np.asarray(positions,float); reference=np.asarray(reference,float)
    goal=np.asarray(world_goal,float); root=np.asarray(sqrt_weight,float)
    ss, ds = _project_sequence(positions, reference, initial_s)
    jac=np.zeros((2,len(positions),2))
    crossings=np.flatnonzero((ss[:-1]<goal_s) & (ss[1:]>=goal_s))
    if len(crossings):
        k=int(crossings[0]); span=ss[k+1]-ss[k]
        alpha=(goal_s-ss[k])/span
        point=(1-alpha)*positions[k]+alpha*positions[k+1]
        slope=(positions[k+1]-positions[k])/span
        jac[:,k]=(1-alpha)*(np.eye(2)-np.outer(slope,ds[k]))
        jac[:,k+1]=alpha*(np.eye(2)-np.outer(slope,ds[k+1]))
        mode='first_crossing'; fraction=float(alpha)
    else:
        k=len(positions)-1;point=positions[-1];jac[:,k]=np.eye(2)
        mode='terminal_approach';fraction=None
    residual=root@(point-goal)
    jac=np.einsum('ij,jnk->ink',root,jac)
    return residual,jac,{'mode':mode,'interval_index':k,'alpha':fraction,
        'evaluation_point':point.tolist(),'s_samples':ss.tolist(),
        'goal_s_m':float(goal_s),'discretization':'首次弧长截面相邻0.1秒状态线性插值；非连续动力学精确事件'}


class GoalManager:
    """同一时刻至多一个目标；已替换或已通过的目标永不重新激活。"""
    def __init__(self,config):
        self.p={**DEFAULTS,**config.get('local_goal',{})}
        self.active=None;self.records={};self.latest_order=None;self.revision=0;self.events=[];self.pending_events=[]

    def export_state(self):
        return deepcopy({'active':self.active,'records':self.records,'latest_order':self.latest_order,
                         'revision':self.revision,'pending_events':self.pending_events})

    def import_state(self,state):
        self.active=deepcopy(state['active']);self.records=deepcopy(state['records'])
        self.latest_order=tuple(state['latest_order']) if state['latest_order'] is not None else None
        self.revision=int(state['revision']);self.events=[];self.pending_events=deepcopy(state.get('pending_events',[]))

    def _end(self,record,status,reason,scene=None):
        was_active=self.active is record
        record.update(status=status,invalidation_reason=reason,ended_time=None if scene is None else float(scene.time))
        self.records[record['goal_id']]={'status':status,'source_frame_id':record['source_frame_id'],
            'source_time':record['source_time'],'reason':reason}
        self.events.append(deepcopy(record))
        if was_active:
            self.revision+=1;self.active=None

    def clear(self,reason='新答案无效或通信失败'):
        if self.active is not None:
            self.events=[]
            self._end(self.active,'invalid',reason)
            self.pending_events.extend(deepcopy(self.events))

    def update(self,scene,answer=None):
        self.events=self.pending_events;self.pending_events=[]
        if answer is not None and answer.goal_id not in self.records:
            reference=answer.goal_reference
            order=(float(answer.time),int(reference['request_version']))
            record={'goal_id':answer.goal_id,'source_frame_id':answer.frame_id,'source_time':float(answer.time),
                'goal_reference':deepcopy(reference),'world_goal':np.asarray(answer.world_goal,float).tolist(),
                'local_goal':deepcopy(answer.local_goal),'s_goal_m':float(reference['s_ego_m']+answer.local_goal['delta_s_m']),
                'explanation':deepcopy(answer.explanation),'received_time':float(scene.time),
                'status':'active','last_s':float(reference['s_ego_m']), 'replaced_before_pass':False,
                'target_lane_id':getattr(answer,'target_lane_id',None)}
            self.records[answer.goal_id]={'status':'active','source_frame_id':answer.frame_id,'source_time':float(answer.time)}
            if self.latest_order is not None and order<=self.latest_order:
                self._end(record,'invalid','乱序旧帧或旧请求版本不能覆盖已接纳的新目标',scene)
            else:
                self.latest_order=order
                if self.active is not None:
                    self.active['replaced_before_pass']=True
                    self._end(self.active,'replaced','接纳新的有效目标',scene)
                self.active=record;self.revision+=1
                self.events.append(deepcopy(record))
        if self.active is not None:
            record=self.active;ref=np.asarray(record['goal_reference']['points'],float)
            s,d,_=project_polyline(scene.ego.position,ref,previous_s=record['last_s'],
                heading=scene.ego.heading,window=self.p['projection_window_m'])
            record['last_s']=float(s)
            if s>=record['s_goal_m']-self.p['passed_tolerance_m']:
                error=float(d-record['local_goal']['d_m'])
                record.update(lateral_error_at_pass_m=error,
                    target_lateral_reached=abs(error)<=.35,actual_lane_at_pass=scene.ego.lane_id,
                    ego_center_s_at_pass_m=float(s))
                target=next((lane for lane in scene.lanes if lane.id==record['target_lane_id']),None)
                record['target_lane_reached']=(False if target is None else
                    Polygon(target.polygon).covers(Point(scene.ego.position)) and abs(error)<=.35)
                self._end(record,'passed','实际自车中心已纵向到达原参考线目标截面',scene)
        return self.active

    def diagnostics(self):
        return {'revision':self.revision,'active_goal':deepcopy(self.active),
                'events':deepcopy(self.events),'latest_source_order':self.latest_order}
