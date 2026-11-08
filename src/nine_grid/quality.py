"""PDM 结构的项目适配评分；未来由预测提供，绝不使用真实未来。"""
from __future__ import annotations
import numpy as np
from shapely import prepare
from .contracts import QualityReport
from .geometry import actor_shape, project_polyline, signed_body_distance,rectangle_corners
from .realtime_solver import bicycle_geometry


def _route_projection(scene,states):
    """逐状态连续投影；每步窗口按实际位移展开，防止环道重复路段跳段。"""
    previous=scene.diagnostics.get('route_s')
    values=[]
    for index,state in enumerate(states):
        travel=0. if index==0 else np.linalg.norm(state[:2]-states[index-1,:2])
        value=project_polyline(state[:2],scene.route,previous_s=previous,heading=state[2],window=max(10.,3*travel+5))
        previous=value[0]
        values.append(value)
    return np.asarray(values).T


def _progress(scene,states):
    s,_,_=_route_projection(scene,states)
    return float(max(0.,s[-1]-s[0]))


def _collision_checker(ego,predictions):
    """仅在一次审计内复用同一预测时刻的朝向/实体；不跨原帧缓存。"""
    times=predictions.times
    headings={key:np.unwrap(trajectory[:,2]) for key,trajectory in predictions.trajectories.items()}
    cache={key:{} for key in predictions.trajectories}
    def intersects(position,heading,ego_shape,identity,t):
        if t>times[-1]+1e-7:raise ValueError('预测时域不足')
        actor=predictions.actors[identity]
        if t not in cache[identity]:
            trajectory=predictions.trajectories[identity]
            p=np.array([np.interp(t,times,trajectory[:,j]) for j in (0,1)])
            yaw=np.interp(t,times,headings[identity])
            shape=None if ego.kind=='pedestrian' or actor.kind=='pedestrian' else actor_shape(actor,p,yaw)
            cache[identity][t]=(p,yaw,shape)
        p,yaw,shape=cache[identity][t]
        if shape is None:return signed_body_distance(ego,actor,position,heading,p,yaw)<=0.
        return ego_shape.intersects(shape)
    return intersects


def evaluate(scene, plan, baseline, predictions, config=None):
    cfg = (config or {}).get('quality', {})
    rule_config=(config or {}).get('rules',{})
    tolerance=(config or {}).get('mpc',{}).get('body_constraint_tolerance_m',.001)
    metrics, reasons = {}, []
    wheelbase,rear=bicycle_geometry(scene.ego.length,scene.ego.wheelbase_m,scene.ego.rear_axle_to_center_m)
    axle_ratio=rear/wheelbase
    if not plan.valid or len(plan.states) < 3:
        return QualityReport(scene.frame_id, False, None, metrics, False, False, ['当前规划无有效解'],time=scene.time)
    def invalid(reason):
        return QualityReport(scene.frame_id,False,None,{},False,False,[reason],time=scene.time)
    for item in (plan,baseline,predictions):
        if item.frame_id!=scene.frame_id or item.version!=scene.version or not np.isclose(item.time,scene.time,rtol=0,atol=1e-8):
            return invalid('评分输入的原帧、时间或版本不一致')
    for item in (plan,baseline):
        if item.valid:
            states_check=np.asarray(item.states);controls_check=np.asarray(item.controls)
            if states_check.ndim!=2 or states_check.shape[1]!=4 or controls_check.shape!=(len(states_check)-1,2) or not np.isfinite(states_check).all() or not np.isfinite(controls_check).all():
                return invalid('规划状态或控制维度错误、数值非有限')
    expected={actor.id for actor in scene.actors if actor.visible}
    if set(predictions.trajectories)!=expected or set(predictions.actors)!=expected:
        return invalid('可见对象与预测集合不完整或包含不可见对象')
    prediction_times=np.asarray(predictions.times)
    if prediction_times.ndim!=1 or len(prediction_times)<2 or not np.isfinite(prediction_times).all() or abs(prediction_times[0])>1e-8 or np.any(np.diff(prediction_times)<=0):
        return invalid('预测时刻序列无效')
    for trajectory in predictions.trajectories.values():
        values=np.asarray(trajectory)
        if values.shape!=(len(prediction_times),4) or not np.isfinite(values).all():
            return invalid('对象预测维度错误或数值非有限')
    states = plan.states
    dt = (config or {}).get('mpc', {}).get('step_s', .1)
    times = np.arange(len(states))*dt
    collision, onroad, direction_ok, rule_ok, ttc_ok = False, True, True, True, True
    required_horizon = times[-1]+max(cfg.get('ttc_offsets_s',[0,.3,.6,.9]))
    if predictions.trajectories and predictions.times[-1]+1e-7 < required_horizon:
        return QualityReport(scene.frame_id,False,None,{},False,False,['TTC 所需预测尾段缺失'],time=scene.time)
    for identity,diagnostic in predictions.diagnostics.items():
        if isinstance(diagnostic,dict) and diagnostic.get('valid') is False:
            reasons.append(f'对象 {identity} 的地图运动预测无效')
    accepted_road=None if scene.drivable is None else scene.drivable.buffer(tolerance)
    if accepted_road is not None:prepare(accepted_road)
    intersects=_collision_checker(scene.ego,predictions)
    ego_headings=np.unwrap(states[:,2])
    # 轨迹的实体审计采用 0.02s 插值采样，单独记录，不充当执行否决器。
    for t in np.arange(0,times[-1]+.001,.02):
        p = np.array([np.interp(t,times,states[:,j]) for j in (0,1)])
        h = np.interp(t,times,ego_headings)
        shape = actor_shape(scene.ego,p,h)
        if scene.drivable is None:
            reasons.append('缺少可行驶区域')
            break
        onroad &= bool(accepted_road.covers(shape))
        for actor_id in predictions.trajectories:
            if intersects(p,h,shape,actor_id,t):
                collision=True
    route_s,_,route_h = _route_projection(scene,states)
    angle=(states[:,2]-route_h+np.pi)%(2*np.pi)-np.pi
    direction_ok=bool(np.all((states[:,3] < rule_config.get('wrong_way_min_speed_mps',.5)) | (abs(angle)<=rule_config.get('wrong_way_heading_limit_rad',np.pi/2))))
    rules={**scene.rules,**plan.diagnostics.get('rules',{})}
    for key in ('stop_s','must_wait','wait_reason'):
        if key in plan.diagnostics:
            rules[key]=plan.diagnostics[key]
    stop_s=rules.get('stop_s')
    if stop_s is not None and rules.get('must_wait',False):
        front_positions=[project_polyline(corner,scene.route,previous_s=station,
                         heading=state[2],window=max(10.,scene.ego.length*2))[0]
                         for station,state in zip(route_s,states)
                         for corner in rectangle_corners(state[:2],state[2],scene.ego.length,scene.ego.width)]
        rule_ok=bool(max(front_positions)<=stop_s+tolerance)
    limits=scene.rules.get('speed_limit',33.3333333333)
    rule_ok &= bool(np.max(states[:,3]) <= float(limits)+.1)
    # TTC 对每个候选状态短时直推，区别于沿候选曲线相交检查。
    if predictions.times[-1]+1e-7>=required_horizon:
        for k,t in enumerate(times):
            if states[k,3] < cfg.get('stopped_speed_mps',.005):
                continue
            steering=plan.controls[min(k,len(plan.controls)-1),1]
            chi=states[k,2]+np.arctan(axle_ratio*np.tan(steering))
            for tau in cfg.get('ttc_offsets_s',[0,.3,.6,.9]):
                p=states[k,:2]+tau*states[k,3]*np.array([np.cos(chi),np.sin(chi)])
                ego=actor_shape(scene.ego,p,states[k,2])
                for actor_id in predictions.trajectories:
                    if intersects(p,states[k,2],ego,actor_id,t+tau):
                        ttc_ok=False
    steering=np.r_[plan.controls[:,1],plan.controls[-1,1]] if len(plan.controls)==len(states)-1 else np.zeros(len(states))
    beta=np.arctan(axle_ratio*np.tan(steering))
    velocity=states[:,3,None]*np.column_stack([np.cos(states[:,2]+beta),np.sin(states[:,2]+beta)])
    acceleration=np.gradient(velocity,dt,axis=0)
    acc=acceleration[:,0]*np.cos(states[:,2])+acceleration[:,1]*np.sin(states[:,2])
    yaw_rate=np.gradient(np.unwrap(states[:,2]),dt)
    lateral=-acceleration[:,0]*np.sin(states[:,2])+acceleration[:,1]*np.cos(states[:,2])
    jerk_long=np.gradient(acc,dt)
    jerk_mag=np.gradient(np.hypot(acc,lateral),dt)
    yaw_acc=np.gradient(yaw_rate,dt)
    c=cfg.get('comfort',{})
    comfort=bool(np.min(acc)>=c.get('long_accel_min_mps2',-4.05) and np.max(acc)<=c.get('long_accel_max_mps2',2.4)
        and np.max(abs(lateral))<=c.get('lateral_accel_abs_max_mps2',4.89)
        and np.max(abs(jerk_long))<=c.get('long_jerk_abs_max_mps3',4.13)
        and np.max(abs(jerk_mag))<=c.get('jerk_magnitude_derivative_abs_max_mps3',8.37)
        and np.max(abs(yaw_rate))<=c.get('yaw_rate_abs_max_radps',.95)
        and np.max(abs(yaw_acc))<=c.get('yaw_accel_abs_max_radps2',1.93))
    progress=float(max(0.,route_s[-1]-route_s[0]))
    reference=None
    progress_score=None
    if not baseline.valid:
        reasons.append('无语义物理参照规划失败')
    else:
        reference=_progress(scene,baseline.states)
        if rules.get('must_wait',False) and rules.get('wait_reason') and rule_ok:
            progress_score=1.
        else:
            progress_score=float(np.clip(progress/max(reference,cfg.get('progress_floor_m',.1)),0,1))
    # 可执行不再等于安全；软化后的实线/名义间隙等违规仍必须阻止危险经验入库。
    planner_safety=plan.diagnostics.get('safety_feasible',True) is not False
    gates={'collision_free':not collision,'on_road':onroad,'correct_direction':direction_ok,'rules':rule_ok,
           'planner_safety':planner_safety}
    metrics.update(gates=gates,progress_m=progress,reference_progress_m=reference,progress=progress_score,ttc=float(ttc_ok),comfort=float(comfort))
    valid=not reasons
    weights=cfg.get('weights',dict(progress=5,ttc=5,comfort=2))
    score=(float(all(gates.values()))*sum(weights[k]*metrics[k] for k in weights)/sum(weights.values())) if valid else None
    return QualityReport(scene.frame_id,valid,score,metrics,not ttc_ok,collision,list(dict.fromkeys(reasons)),time=scene.time)


def admissible(answer,plan,quality,threshold=.7):
    return bool(answer is not None and answer.frame_id==plan.frame_id==quality.frame_id
        and answer.version==plan.version==quality.version and quality.time is not None
        and np.isclose(answer.time,plan.time,rtol=0,atol=1e-8) and np.isclose(answer.time,quality.time,rtol=0,atol=1e-8)
        and plan.valid and plan.diagnostics.get('safety_feasible',True) is not False
        and quality.valid and quality.score is not None and np.isfinite(quality.score)
        and quality.score>=threshold and not quality.collision and not quality.precollision)


def reflection_reasons(quality,invalid_answer=False,threshold=.7):
    reasons=[]
    if invalid_answer:
        reasons.append('模型答案非法')
    if quality.valid and quality.score is not None and quality.score<threshold:
        reasons.append('轨迹质量低')
    # 聚合分不可计算也不能掩盖已检出的实体相交；TTC 保留独立原始字段。
    if quality.precollision or quality.collision:
        reasons.append('预碰撞')
    return reasons


def execution_audit(scene,plan,predictions,config=None):
    """对将执行的 0.1 秒独立采样审计，结果不改写或否决候选控制。"""
    from .planner import propagate
    if not plan.valid:
        return {'available':False,'alarm':False,'reason':'无可执行候选'}
    cfg=(config or {}).get('mpc',{})
    dt=cfg.get('step_s',.1)
    substeps=cfg.get('physics_substeps',5)
    state=np.array([*scene.ego.position,scene.ego.heading,np.linalg.norm(scene.ego.velocity)])
    collisions=[];offroad=False
    accepted_road=scene.drivable.buffer(cfg.get('body_constraint_tolerance_m',.001))
    prepare(accepted_road)
    intersects=_collision_checker(scene.ego,predictions)
    for i in range(1,substeps+1):
        t=dt*i/substeps
        state=propagate(state,plan.controls[0],scene.ego.length,dt/substeps,1,
                        scene.ego.wheelbase_m,scene.ego.rear_axle_to_center_m)
        ego_shape=actor_shape(scene.ego,state[:2],state[2])
        offroad|=not accepted_road.covers(ego_shape)
        for identity in predictions.trajectories:
            if intersects(state[:2],state[2],ego_shape,identity,t):
                collisions.append(identity)
    return {'available':True,'alarm':bool(collisions or offroad),'collision_ids':sorted(set(collisions)),
            'offroad':offroad,'executed_policy':'报警仍执行当前有效候选','sample_step_s':dt/substeps}
