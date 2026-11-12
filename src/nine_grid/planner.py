"""控制边界保持硬约束，安全冲突采用软代价；始终保留本帧可执行轨迹。"""
from __future__ import annotations

import time
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry import LineString
from .contracts import PlanResult
from .planning_objects import planning_view
from .local_reference import LocalReferenceManager, LocalReferenceResult
from .goals import GoalManager, goal_seed_path, first_arrival_residual
from .realtime_solver import Dynamics, QuadraticSolver, rectangle_gap, circle_gap, rectangle_distance
from .geometry import (clean_polyline, ego_to_world, world_to_ego,
                       rectangle_corners, polyline_lengths, project_polyline, sample_polyline, wrap_angle)


def propagate(state, control, length=5., dt=.1, substeps=5):
    """与 highway Vehicle.step 同序的显式 Euler，含质心侧偏角。"""
    state=np.asarray(state,float).copy();a,delta=np.asarray(control,float)
    beta=np.arctan(.5*np.tan(delta));step=dt/substeps
    for _ in range(substeps):
        x,y,psi,v=state
        state=np.array([x+v*np.cos(psi+beta)*step,y+v*np.sin(psi+beta)*step,
                        psi+2*v/length*np.sin(beta)*step,v+a*step])
    return state


def body_vertices(states,length,width):
    states=np.asarray(states)
    offsets=np.array([[length/2,width/2],[length/2,-width/2],[-length/2,-width/2],[-length/2,width/2]])
    c,s=np.cos(states[:,2]),np.sin(states[:,2])
    rotated=np.stack((offsets[None,:,0]*c[:,None]-offsets[None,:,1]*s[:,None],
                      offsets[None,:,0]*s[:,None]+offsets[None,:,1]*c[:,None]),axis=-1)
    return states[:,:2,None].transpose(0,2,1)+rotated


def _path_projection(points,reference):
    """折线精确最近点及活动线段；夹在端点时弧长导数为零。"""
    segments=np.diff(reference,axis=0);lengths=np.linalg.norm(segments,axis=1)
    fractions=np.einsum('nsi,si->ns',np.asarray(points)[:,None]-reference[:-1],segments)/lengths**2
    clipped=np.clip(fractions,0.,1.)
    residual=np.asarray(points)[:,None]-reference[:-1]-clipped[:,:,None]*segments
    chosen=np.argmin(np.sum(residual**2,axis=2),axis=1);row=np.arange(len(points))
    tangent=segments[chosen]/lengths[chosen,None]
    arc=np.r_[0.,np.cumsum(lengths)]
    s=arc[chosen]+clipped[row,chosen]*lengths[chosen]
    interior=(fractions[row,chosen]>0.)&(fractions[row,chosen]<1.)
    # 内部共线采样点没有真实折角；选中段的端点不代表参考路径已截断。
    if len(segments)>1:
        units=segments/lengths[:,None]
        aligned=np.sum(units[:-1]*units[1:],axis=1)>1.-1e-12
        before=np.r_[False,aligned][chosen];after=np.r_[aligned,False][chosen]
        interior|=((fractions[row,chosen]==0.)&before)|((fractions[row,chosen]==1.)&after)
    return s,residual[row,chosen],tangent,interior


def _interpolate_with_slope(s,arc,values):
    """数值和导数使用同一分段线性插值，不能以插值后的中心差分代替。"""
    indices=np.clip(np.searchsorted(arc,s,side='right')-1,0,len(arc)-2)
    slopes=np.diff(values)/np.diff(arc)
    derivative=np.where((s>arc[0])&(s<arc[-1]),slopes[indices],0.)
    return np.interp(s,arc,values),derivative


def _stop_speed_with_slope(s,stop_local,length,braking):
    """停车参考的饱和值与同分支导数；零点采用已截断侧导数。"""
    speed=np.sqrt(np.maximum(0.,2*braking*(stop_local-s-length/2)))
    derivative=np.divide(-braking,speed,out=np.zeros_like(speed),where=speed>0.)
    return speed,derivative


def _project_executable_controls(controls,speed,dt,min_a,max_a,max_delta):
    """只投影执行必需的控制箱和非负速度，不混入舒适度或地图约束。"""
    result=np.asarray(controls,float).copy()
    if result.ndim!=2 or result.shape[1]!=2 or not np.isfinite(result).all():
        raise ValueError('MPC控制初值必须是有限的(N,2)数组')
    for index in range(len(result)):
        lower=max(min_a,-max(0.,speed)/dt)
        result[index,0]=np.clip(result[index,0],lower,max_a)
        result[index,1]=np.clip(result[index,1],-max_delta,max_delta)
        speed=max(0.,speed+result[index,0]*dt)
    return result


def _validate_prediction_input(scene, predictions, horizon):
    """拒绝错帧、漏对象和不足时域；不得把接口错误当作空交通或静止尾段。"""
    if (predictions.frame_id != scene.frame_id or predictions.version != scene.version
            or predictions.coordinates != 'world_xy_m' or scene.coordinates != 'world_xy_m'
            or not np.isclose(predictions.time, scene.time, rtol=0., atol=1e-8)):
        raise ValueError('规划预测输入的原帧、时间、版本或世界坐标不一致')
    visible = [actor for actor in scene.actors if actor.visible]
    expected = {actor.id for actor in visible}
    if len(expected) != len(visible) or set(predictions.actors) != expected or set(predictions.trajectories) != expected:
        raise ValueError('规划预测对象集合必须完整对应当前可见对象，不能遗漏或引入不可见对象')
    times = np.asarray(predictions.times, float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or abs(times[0]) > 1e-8 or np.any(np.diff(times) <= 0)):
        raise ValueError('规划预测时刻必须从零开始、严格递增且数值有限')
    if expected and times[-1] < horizon-1e-8:
        raise ValueError('对象预测未覆盖MPC完整时域，不允许静默重复末端位置')
    for actor in visible:
        predicted = predictions.actors[actor.id]
        if (predicted.id != actor.id or not predicted.visible or predicted.kind != actor.kind
                or predicted.length != actor.length or predicted.width != actor.width
                or predicted.radius != actor.radius
                or not np.allclose(predicted.position, actor.position, rtol=0., atol=1e-8)
                or not np.allclose(predicted.velocity, actor.velocity, rtol=0., atol=1e-8)
                or abs(float(wrap_angle(predicted.heading-actor.heading))) > 1e-8):
            raise ValueError(f'对象 {actor.id} 的预测实体描述与当前观测不一致')
        trajectory = np.asarray(predictions.trajectories[actor.id], float)
        if (trajectory.shape != (len(times), 4) or not np.isfinite(trajectory).all()
                or np.any(trajectory[:, 3] < -1e-8)):
            raise ValueError(f'对象 {actor.id} 的预测维度或运动数值非法')


class Planner:
    def __init__(self,config:dict):
        self.config=config
        self.p=config.get("mpc",{})
        self.n=int(self.p.get("horizon_steps",30));self.dt=float(self.p.get("step_s",.1))
        self.substeps=int(self.p.get("physics_substeps",5))
        self.previous=None
        self.previous_time=None
        self.last_diagnostics={}
        self.rule_evidence=[]
        self.references=LocalReferenceManager(config)
        self.goals=GoalManager(config)
        self.previous_reference_revision=None

    def restore_reference_state(self,state):
        """后台仅恢复原帧选择前的几何意图，不共享前台求解器或控制。"""
        self.references.import_state(state['local_reference'])
        self.goals.import_state(state['goals'])

    def export_reference_state(self):
        return {'local_reference':self.references.export_state(),'goals':self.goals.export_state()}

    def clear_semantic_guidance(self):
        """新请求失败清除整个目标和全部派生偏好，仍由同一个MPC求解。"""
        self.goals.clear()
        self.references.reset_intent()
        self.previous=None

    def _rules(self,scene,predictions):
        """公共道路优先级和可见预测确定让行，不能用自身停车解释应该等待。"""
        stop_s=scene.rules.get("stop_s")
        ego_s=project_polyline(scene.ego.position,scene.route,scene.diagnostics.get("route_s"),scene.ego.heading)[0]
        lanes={lane.id:lane for lane in scene.lanes}
        ego_lane=lanes.get(scene.ego.lane_id)
        reason=scene.rules.get("wait_reason")
        rules=self.config.get("rules",{})
        stop_margin=float(rules.get("stop_margin_m",1.))
        self.rule_evidence=[]
        if scene.scenario=="intersection" and ego_lane and scene.rules.get("road_priority_available"):
            route=scene.route;route_shape=LineString(route)
            for identity,actor in predictions.actors.items():
                other_lane=lanes.get(actor.lane_id)
                audit={"actor_id":identity,"source":"visible_public_lane_geometry","active":False}
                self.rule_evidence.append(audit)
                if not actor.visible or not other_lane:
                    audit["reason"]="对象不可见或缺少公开车道";continue
                # 已在自车同一公开路段的对象属于跟驰；不能仅凭当前朝向
                # 相似删除汇入冲突，也不能把对向车当作同向跟驰。
                if actor.lane_id in scene.route_lane_ids:
                    audit["reason"]="已在自车同一公开路段，按跟驰处理";continue
                line=other_lane.center.copy();lane=other_lane;visited={lane.id};path_ids=[lane.id]
                inferred=list(predictions.diagnostics.get(identity,{}).get("successors",[]))
                actor_s=project_polyline(actor.position,line,heading=actor.heading)[0]
                while polyline_lengths(line)[-1]-actor_s<55.:
                    options=[lanes[key] for key in lane.successors if key in lanes and key not in visited]
                    if not options:break
                    next_id=inferred.pop(0) if inferred else None
                    # 优先复用本帧预测器已经推断的公开合法后继。停车时
                    # 预测可能没有后继，按已观察运动朝向选最一致者；这只
                    # 定义公开路权通路，不生成额外恒速轨迹或读取隐藏路线。
                    candidates={candidate.id:candidate for candidate in options}
                    if next_id in candidates:
                        lane=candidates[next_id]
                    else:
                        low_speed=float(self.config.get("prediction",{}).get("low_speed_mps",.3))
                        motion=actor.heading if np.linalg.norm(actor.velocity)<low_speed else np.arctan2(actor.velocity[1],actor.velocity[0])
                        def alignment(candidate):
                            direction=candidate.center[-1]-candidate.center[0]
                            return abs(float(wrap_angle(np.arctan2(direction[1],direction[0])-motion))),candidate.id
                        lane=min(options,key=alignment)
                    visited.add(lane.id)
                    path_ids.append(lane.id)
                    line=clean_polyline(np.vstack((line,lane.center)))
                # 首次入口由完整公共通路的空间几何决定，不受短时预测
                # 刹停或更远处共同出口的最近点影响。
                width=other_lane.width
                corridor=LineString(line).buffer(width/2,cap_style="flat")
                intersection=route_shape.intersection(corridor)
                coordinates=shapely.get_coordinates(intersection)
                audit["public_lane_ids"]=path_ids
                if not len(coordinates):
                    audit["reason"]="公开通路与自车导航没有交汇";continue
                entry_s=float(np.min(project_polyline(coordinates,route)[0]))
                audit["entry_s"]=entry_s
                if entry_s>ego_s+55:
                    audit["reason"]="首次入口在自车局部规则范围之外";continue
                if entry_s<ego_s+scene.ego.length/2-.001:
                    audit["reason"]="自车已通过首次入口，不重新选择远端重合处";continue
                entry=sample_polyline(route,entry_s)
                # 只保留对象车尾至其前方55m通路。完整通路只用于固定
                # 首次入口，已经驶过的道路不会继续触发让行。
                arc=polyline_lengths(line)
                actor_s=project_polyline(actor.position,line,heading=actor.heading)[0]
                rear_s=float(np.min(project_polyline(rectangle_corners(actor.position,actor.heading,actor.length,actor.width),line)[0]))
                end_s=min(arc[-1],actor_s+55.)
                if end_s<=rear_s+1e-6:
                    audit["reason"]="对象前方公共通路不足";continue
                forward=clean_polyline(np.vstack((sample_polyline(line,rear_s),line[(arc>rear_s)&(arc<end_s)],sample_polyline(line,end_s))))
                if not LineString(forward).buffer(width/2,cap_style="flat").buffer(1e-8).covers(shapely.Point(entry)):
                    audit["reason"]="首次入口不在对象车尾及前方55m的当前通路中";continue
                priorities=[candidate.priority for candidate in scene.lanes if candidate.id in scene.route_lane_ids
                            and Polygon(candidate.polygon).buffer(1e-8).covers(shapely.Point(entry))]
                ego_priority=min(priorities,default=ego_lane.priority)
                audit.update(ego_priority=ego_priority,other_priority=other_lane.priority)
                if other_lane.priority<=ego_priority:
                    audit["reason"]="对象不具有更高的公开路权";continue
                proposed=entry_s-stop_margin
                audit.update(active=True,stop_s=proposed,reason="可见高优先级对象尚未清空首次冲突入口")
                if stop_s is None or proposed<stop_s:
                    stop_s=proposed;reason=f"向可见高优先级交通对象 {identity} 让行（公开通路首次入口）"
        return stop_s,reason

    def solve(self,scene,predictions,answer=None):
        started=time.perf_counter();p=self.p;n=self.n;dt=self.dt
        _validate_prediction_input(scene,predictions,n*dt)
        # 完整输入必须先验明；局部MPC视图不污染模型、检索、反思和独立审计。
        scene,predictions,object_selection=planning_view(scene,predictions,self.config,answer)
        cold_start=not getattr(self,'_has_solved',False);self._has_solved=True
        tolerance=float(p.get("body_constraint_tolerance_m",.001))
        braking=float(self.config.get("rules",{}).get("comfort_braking_mps2",3.))
        origin=scene.ego.position.copy();angle=scene.ego.heading
        speed=float(np.linalg.norm(scene.ego.velocity));initial=np.array([0.,0.,0.,speed])
        build_seconds=0.
        if not hasattr(self,'dynamics') or self.dynamics_length!=scene.ego.length:
            self.dynamics=Dynamics(n,dt,self.substeps,scene.ego.length);self.dynamics_length=scene.ego.length
            self.qp=QuadraticSolver(2*n,int(p.get('qp_max_iterations',600)),float(p.get('qp_tolerance',1e-6)))
            build_seconds=self.dynamics.build_seconds
        ego_s=project_polyline(origin,scene.route,scene.diagnostics.get("route_s"),angle)[0]
        full_arc=polyline_lengths(scene.route)
        start_arc=max(0.,ego_s-20);end_arc=min(full_arc[-1],ego_s+max(120,speed*4))
        global_reference=clean_polyline(world_to_ego(sample_polyline(scene.route,np.linspace(start_arc,end_arc,max(2,int((end_arc-start_arc)/.5)+1))),origin,angle))
        global_arc=polyline_lengths(global_reference)
        reference_state_before=self.export_reference_state()
        goal_revision_before=self.goals.revision
        active_goal=self.goals.update(scene,answer)
        goal_changed=self.goals.revision!=goal_revision_before
        new_goal_active=goal_changed and active_goal is not None
        if goal_changed:
            # 旧目标的几何意图始终清除；新有效目标允许继承数值控制初值，
            # 后续必须按本帧状态、新目标及当前风险重新积分、计价与验算。
            self.references.reset_intent()
            if not new_goal_active:
                self.previous=None
        if active_goal is not None:
            # 原参考仅用于坐标、必要导航速度与进度。目标只生成初值，
            # 不把其几何再变成第二份contour/heading代价或永久换道承诺。
            selection=LocalReferenceResult(np.asarray(scene.route,float).copy(),
                int(self.references.state['revision'])+self.goals.revision,
                {'phase':'goal_active','selected_source':'local_goal','maneuver_active':False,
                 'selection_wall_s':0.,'goal_geometry_role':'initialization_only'})
            seed_world=goal_seed_path(scene,active_goal)
        else:
            selection=self.references.select(scene,predictions)
            selection.revision+=self.goals.revision
            seed_world=selection.world_path
        local_arc=polyline_lengths(selection.world_path)
        selected_s=project_polyline(origin,selection.world_path,heading=angle)[0]
        local_begin=max(0.,selected_s-20.)
        local_end=min(local_arc[-1],selected_s+max(120.,speed*4))
        reference=clean_polyline(world_to_ego(sample_polyline(selection.world_path,np.linspace(local_begin,local_end,max(2,int((local_end-local_begin)/.5)+1))),origin,angle))
        local_s0=project_polyline([0.,0.],reference)[0];reference_arc=polyline_lengths(reference)
        tangents=np.unwrap(np.arctan2(np.gradient(reference[:,1]),np.gradient(reference[:,0])))
        signed_curvature=np.gradient(tangents,np.maximum.accumulate(reference_arc+np.arange(len(reference_arc))*1e-9))
        curve_acceleration=float(self.config.get('rules',{}).get('curve_reference_lateral_acceleration_mps2',3.5))
        speed_profile=np.minimum(float(scene.rules.get('desired_speed',25.)),np.sqrt(curve_acceleration/np.maximum(np.abs(signed_curvature),.001)))
        for k in range(len(speed_profile)-2,-1,-1):
            speed_profile[k]=min(speed_profile[k],np.sqrt(speed_profile[k+1]**2+2*braking*(reference_arc[k+1]-reference_arc[k])))
        stop_s,wait_reason=self._rules(scene,predictions)
        # 规则的弧长始终属于全局导航；不能把换道路径的弧长当成停车线坐标。
        stop_local=None if stop_s is None else stop_s-start_arc
        previous_control=np.asarray(scene.diagnostics.get('previous_control',[0.,0.]),float)
        horizon=n*dt;risk_cfg=self.config.get('risk',{})
        line_extent=float(risk_cfg.get('classes',{}).get('noncrossable_line',{}).get('influence_m',1.5))
        reach=speed*horizon+.5*max(0.,float(p.get('acceleration_max_mps2',3.)))*horizon**2+np.hypot(scene.ego.length,scene.ego.width)/2+line_extent+.01
        local_world=scene.drivable.intersection(shapely.box(origin[0]-reach,origin[1]-reach,origin[0]+reach,origin[1]+reach))
        if local_world.is_empty:
            # 已离开局部地图时仍向真实道路计算软距离，不能构造空边界后返回空轨迹。
            local_world=scene.drivable
        if local_world.is_empty:
            raise ValueError('MPC缺少可用道路几何，不能计算规划代价')
        road=shapely.transform(local_world,lambda xy:world_to_ego(xy,origin,angle));boundary=road.boundary
        # 精确地图用于最终完整矩形复核；不以简化多边形扩大道路。
        accepted_road=road.buffer(tolerance)
        # 只准备同一几何的查询索引；不改变坐标、边界或容差。
        shapely.prepare(road);shapely.prepare(accepted_road)
        physical_lines={}
        for lane in scene.lanes:
            for line in lane.boundaries:
                if line.get('crossable',True):continue
                geom=LineString(world_to_ego(line['points'],origin,angle)).intersection(shapely.box(-reach,-reach,reach,reach))
                if geom.is_empty:continue
                item=physical_lines.setdefault(geom.normalize().wkb_hex,{'geometry':geom,'ids':[]});item['ids'].append(line['id'])
        if active_goal is None:
            # 无目标时严格沿用原有裁剪、等弧采样的导航初值，不能从原始非均匀地图点重算曲率。
            seed_reference=reference;seed_arc=reference_arc
            seed_tangents=tangents;seed_curvature=signed_curvature
        else:
            seed_reference=clean_polyline(world_to_ego(seed_world,origin,angle))
            seed_arc=polyline_lengths(seed_reference)
            seed_tangents=np.unwrap(np.arctan2(np.gradient(seed_reference[:,1]),np.gradient(seed_reference[:,0])))
            seed_curvature=np.gradient(seed_tangents,np.maximum.accumulate(seed_arc+np.arange(len(seed_arc))*1e-9))
        goal_context=None
        if active_goal is not None:
            frozen=np.asarray(active_goal['goal_reference']['points'],float)
            goal_ref=world_to_ego(frozen,origin,angle)
            goal_position=world_to_ego(np.asarray(active_goal['world_goal']),origin,angle)
            goal_heading=project_polyline(sample_polyline(frozen,active_goal['s_goal_m']),frozen,
                previous_s=active_goal['s_goal_m'],window=5.)[2]-angle
            rotation=np.array([[np.cos(goal_heading),np.sin(goal_heading)],[-np.sin(goal_heading),np.cos(goal_heading)]])
            sqrt_weight=np.diag([1/self.goals.p['longitudinal_scale_m'],1/self.goals.p['lateral_scale_m']])@rotation
            goal_context=(goal_ref,goal_position,sqrt_weight)
        targets=[]
        for identity,trajectory in predictions.trajectories.items():
            actor=predictions.actors.get(identity)
            if actor is None or not actor.visible:continue
            trajectory=np.asarray(trajectory).copy();trajectory[:,2]=np.unwrap(trajectory[:,2])
            sample=np.stack([np.interp(np.arange(n+1)*dt,predictions.times,trajectory[:,j]) for j in range(4)],axis=1)
            sample[:,:2]=world_to_ego(sample[:,:2],origin,angle);sample[:,2]-=angle
            targets.append((actor,sample,body_vertices(sample[1:],actor.length,actor.width)))
        attention={str(a.get('id',a.get('entity_id',''))):1/max(1,int(a.get('rank',i+1))) for i,a in enumerate(answer.attention)} if answer else {}
        clearance=float(risk_cfg.get('body_clearance_margin_m',.05))
        max_a=float(p.get('acceleration_max_mps2',3));min_a=float(p.get('acceleration_min_mps2',-6))
        max_delta=float(p.get('steering_abs_max_rad',.6));max_jerk=float(p.get('jerk_abs_max_mps3',8));max_rate=float(p.get('steering_rate_abs_max_radps',.8));max_lat=float(p.get('lateral_acceleration_abs_max_mps2',4.89))
        max_speed=min(float(p.get('maximum_speed_mps',33.3333333)),float(scene.rules.get('speed_limit',33.3333333)))
        weights=p.get('weights',{});norm=p.get('normalization',{})
        penalties={'actor_clearance':2000.,'road_clearance':500.,'noncrossable_line_clearance':100.,
                   'yield_stop_line':100.,'speed_limit':10.,'lateral_acceleration':5.}
        penalties.update(p.get('safety_penalties',{}))
        scales={key:float(norm.get(key,default)) for key,default in [('position_m',3.5),('heading_rad',.5),('speed_mps',10),('acceleration_mps2',3),('steering_rad',.6),('jerk_mps3',8),('steering_rate_radps',.8),('progress_m',30)]}
        difference=np.eye(n)-np.eye(n,k=-1);accel_selector=np.eye(2*n)[::2];steer_selector=np.eye(2*n)[1::2]
        def seed(scale=1.):
            controls=[];state=initial.copy();last=previous_control.copy()
            for _ in range(n):
                s,d,_=project_polyline(state[:2],seed_reference);kappa=float(np.interp(s,seed_arc,seed_curvature))
                feed=np.arctan(scene.ego.length*kappa/np.sqrt(max(.01,1-(scene.ego.length*kappa/2)**2)))
                heading=float(np.interp(s,seed_arc,seed_tangents))-np.arctan(.5*np.tan(feed))
                steer=feed+1.2*float(wrap_angle(heading-state[2]))-.3*np.arctan(d/max(state[3],2.))
                navigation_s=s if active_goal is None else project_polyline(state[:2],reference)[0]
                desired=min(max_speed,float(scene.rules.get('desired_speed',25.))*scale,np.sqrt(max_lat*scene.ego.length/max(2*abs(np.sin(np.arctan(.5*np.tan(steer)))),.001))*.9,float(np.interp(navigation_s,reference_arc,speed_profile)))
                if stop_local is not None:
                    global_s=project_polyline(state[:2],global_reference)[0]
                    desired=min(desired,np.sqrt(max(0.,2*braking*(stop_local-global_s-scene.ego.length/2))))
                control=np.clip([np.clip((desired-state[3])*1.2,last[0]-max_jerk*dt,last[0]+max_jerk*dt),np.clip(steer,last[1]-max_rate*dt,last[1]+max_rate*dt)],[min_a,-max_delta],[max_a,max_delta])
                control[0]=max(control[0],-max(0.,state[3])/dt)
                controls.append(control);state=propagate(state,control,scene.ego.length,dt,self.substeps);last=control
            return np.asarray(controls)
        age=None if self.previous_time is None else scene.time-self.previous_time
        reference_changed=self.previous_reference_revision is not None and self.previous_reference_revision!=selection.revision
        # 控制量不依赖目标参考坐标。仅有效新目标接管时放开revision检查，
        # 保留数值初值；其他几何切换、通过/失效仍沿用原清理约定。
        cached=None
        if self.previous is not None:
            try:
                value=np.asarray(self.previous,float)
                if value.shape==(n,2) and np.isfinite(value).all():cached=value
            except (TypeError,ValueError):
                pass
        if cached is None:
            warm_rejection='无有效有限控制缓存'
        elif age is None or not np.isfinite(age) or not -.000001<=age<n*dt:
            warm_rejection='控制缓存时间缺失、超前或已超过预测时域'
        elif reference_changed and not new_goal_active:
            warm_rejection='基础局部参考变化，重新生成初值'
        else:
            warm_rejection=None
        warm=warm_rejection is None
        shifted_steps=min(n-1,max(0,int(round(age/dt)))) if warm else 0
        controls=(np.vstack((cached[shifted_steps:],np.repeat(cached[-1:],shifted_steps,axis=0)))
                  if warm else seed())
        project=lambda value:_project_executable_controls(value,speed,dt,min_a,max_a,max_delta)
        controls=project(controls)
        initial_body=shapely.polygons(body_vertices(initial[None,:],scene.ego.length,scene.ego.width))[0]
        initial_safety={'on_road':bool(shapely.covers(accepted_road,initial_body)),
                        'speed_limit':bool(speed<=max_speed+1e-8)}
        for actor,sample,_ in targets:
            if actor.kind=='pedestrian':
                gap,_=circle_gap(initial[None,:],sample[:1,:2],actor.radius or actor.width/2,scene.ego.length,scene.ego.width)
            else:
                gap,_=rectangle_gap(initial[None,:],sample[:1],scene.ego.length,scene.ego.width,actor.length,actor.width)
            initial_safety['actor_clearance:'+actor.id]=bool(gap[0]>=clearance-tolerance)
        initial_safety['noncrossable_line']=all(initial_body.distance(item['geometry'])>=clearance-tolerance for item in physical_lines.values())
        if stop_local is not None:
            initial_safety['yield_stop_line']=bool(np.max(project_polyline(np.asarray(initial_body.exterior.coords)[:4],global_reference)[0])<=stop_local+tolerance)
        labels=['road_clearance','nonnegative_speed','speed_limit','jerk','steering_rate','lateral_acceleration']
        labels.extend('actor_clearance:'+actor.id for actor,_,_ in targets)
        if physical_lines:labels.append('noncrossable_line_clearance')
        if stop_local is not None:labels.append('yield_stop_line')
        evaluations={'values':0,'linearizations':0,'value_seconds':0.,'linearization_seconds':0.,'geometry_seconds':0.}

        def terms(controls):
            value_started=time.perf_counter();evaluations['values']+=1
            # 线搜索只需真实原值；只有进入下一次QP的名义点才计算控制导数。
            states=self.dynamics.rollout(controls,initial);current=states[1:]
            geometry_started=time.perf_counter()
            s,path_residual,tangent,path_interior=_path_projection(current[:,:2],reference)
            heading,heading_derivative=_interpolate_with_slope(s,reference_arc,tangents)
            vertices=body_vertices(current,scene.ego.length,scene.ego.width);polygons=shapely.polygons(vertices)
            probes=np.concatenate((vertices,(vertices+np.roll(vertices,-1,axis=1))/2),axis=1)
            points=shapely.points(probes.reshape(-1,2));closest=shapely.get_coordinates(shapely.get_point(shapely.shortest_line(boundary,points),0))
            offsets=probes.reshape(-1,2)-closest;distances=np.linalg.norm(offsets,axis=1)
            signs=np.where(shapely.covers(road,points),1.,-1.);road_distance=(distances*signs).reshape(n,8)
            normals=(signs[:,None]*offsets/np.maximum(distances[:,None],1e-10)).reshape(n,8,2)
            arms=probes-current[:,:2,None].transpose(0,2,1)
            road_partial=np.zeros((n,8,4));road_partial[:,:,:2]=normals;road_partial[:,:,2]=-normals[:,:,0]*arms[:,:,1]+normals[:,:,1]*arms[:,:,0]
            # QP只保留非负速度及控制箱；几何、规则和舒适阈值不再产生不可行约束行。
            rows=[lambda c:c['sens'][:,3]]
            lower=[-current[:,3]];upper=[np.full(n,np.inf)]
            road_jac=lambda c:np.einsum('npi,nij->npj',road_partial,c['sens']).reshape(-1,2*n)
            # 凹角、孔洞和车道连接缺口可能完全落在八个探针之间。
            # 在当前名义矩形的真实越界部分加入内侧见证点，不能只在最终拒绝却不给QP修正方向。
            uncovered=np.flatnonzero(~shapely.covers(road,polygons))
            outside=shapely.difference(polygons[uncovered],road)
            areas=shapely.area(outside)
            extra_road_minimum=np.inf
            for local_index in np.flatnonzero(areas>1e-10):
                witness=shapely.get_coordinates(shapely.point_on_surface(outside[local_index]))[0]
                nearest=shapely.get_coordinates(shapely.get_point(shapely.shortest_line(boundary,shapely.Point(witness)),0))[0]
                vector=witness-nearest;depth=float(np.linalg.norm(vector))
                signed=-depth if depth>1e-10 else -np.sqrt(areas[local_index])
                extra_road_minimum=min(extra_road_minimum,signed)
            delta=difference@controls;delta[0]-=previous_control
            beta=np.arctan(.5*np.tan(controls[:,1]));bp=.5/(np.cos(controls[:,1])**2*(1+.25*np.tan(controls[:,1])**2))
            lateral=states[:-1,3]**2*2/scene.ego.length*np.sin(beta)
            lateral_factor=states[:-1,3,None]*4/scene.ego.length*np.sin(beta[:,None])
            lateral_direct=states[:-1,3,None]**2*2/scene.ego.length*np.cos(beta[:,None])*bp[:,None]*steer_selector
            lateral_jac=lambda c:lateral_factor*c['all'][:-1,3]+lateral_direct
            minima=[min(float(road_distance.min()),extra_road_minimum),float(current[:,3].min()),float((max_speed-current[:,3]).min()),float((max_jerk*dt-np.abs(delta[:,0])).min()),float((max_rate*dt-np.abs(delta[:,1])).min()),float((max_lat-np.abs(lateral)).min())]
            evaluations['geometry_seconds']+=time.perf_counter()-geometry_started
            residuals=[];jacobians=[];components={}
            def residual(name,value,jac,weight,scale=1.):
                factor=np.sqrt(float(weight)/n)/scale
                residuals.append(np.asarray(value)*factor);jacobians.append((jac,factor))
                components[name]=components.get(name,0.)+float(weight)*float(np.sum(np.asarray(value)**2))/n/scale**2
            def penalty(name,margin,jac,scale=1.):
                # 正部平方在零点一阶连续；穿透时保留几何导数，不使用截断APF的零梯度。
                margin=np.asarray(margin);active=margin<0.
                derivative=lambda c:(jac(c) if callable(jac) else np.asarray(jac))*active[:,None]
                residual('penalty_'+name,np.minimum(margin,0.),derivative,penalties[name],scale)
            penalty('road_clearance',road_distance.ravel()+tolerance,road_jac)
            if len(uncovered):
                # 完整车体越界面积覆盖孔洞和凹角。不能把随形状重选的见证点
                # 当成固定车体点求导，否则软目标的数值和Jacobian不一致。
                def outside_area_jac(c):
                    partial=np.zeros((len(uncovered),4))
                    steps=p.get('geometry_difference_steps',{})
                    for axis,epsilon in enumerate((steps.get('position_m',1e-5),steps.get('position_m',1e-5),steps.get('heading_rad',1e-5))):
                        plus=current[uncovered].copy();minus=plus.copy()
                        plus[:,axis]+=epsilon;minus[:,axis]-=epsilon
                        positive=shapely.area(shapely.difference(shapely.polygons(body_vertices(plus,scene.ego.length,scene.ego.width)),road))
                        negative=shapely.area(shapely.difference(shapely.polygons(body_vertices(minus,scene.ego.length,scene.ego.width)),road))
                        partial[:,axis]=(positive-negative)/(2*epsilon*scene.ego.width)
                    return np.einsum('ni,nij->nj',partial,c['sens'][uncovered])
                residual('penalty_road_clearance',areas/scene.ego.width,outside_area_jac,penalties['road_clearance'])
            penalty('speed_limit',max_speed-current[:,3],lambda c:-c['sens'][:,3],scales['speed_mps'])
            penalty('lateral_acceleration',max_lat-np.abs(lateral),
                    lambda c:-np.sign(lateral)[:,None]*lateral_jac(c),max_lat)
            target,target_derivative=_interpolate_with_slope(s,reference_arc,speed_profile)
            # 限速已改软约束，速度参考也必须相容，不能仍主动追求高于限速的速度。
            target_derivative=np.where(target>max_speed,0.,target_derivative)
            target=np.minimum(target,max_speed)
            target_direction=target_derivative[:,None]*tangent*path_interior[:,None]
            if stop_local is not None:
                global_s,_,global_tangent,global_interior=_path_projection(current[:,:2],global_reference)
                stop_speed,stop_derivative=_stop_speed_with_slope(global_s,stop_local,scene.ego.length,braking)
                stopped=stop_speed<target;target=np.minimum(target,stop_speed)
                target_direction=np.where(stopped[:,None],stop_derivative[:,None]*global_tangent*global_interior[:,None],target_direction)
            # 线段内部等于原横向距离平方；折线端点保留完整位移，避免
            # 最近线段切换时仅取垂距造成目标跳变和错误的数值梯度。
            path_projectors=np.eye(2)[None,:,:]-tangent[:,:,None]*tangent[:,None,:]
            path_projectors[~path_interior]=np.eye(2)
            path_jac=lambda c:np.einsum('nij,njk->nik',path_projectors,c['sens'][:,:2]).reshape(2*n,2*n)
            if active_goal is None:
                residual('contour',path_residual.ravel(),path_jac,weights.get('contour',3),scales['position_m'])
            heading_error=wrap_angle(current[:,2]+beta-heading)
            heading_curvature=heading_derivative[:,None]
            heading_jac=lambda c:c['sens'][:,2]+bp[:,None]*steer_selector-heading_curvature*c['sj']
            if active_goal is None:
                residual('heading',heading_error,heading_jac,weights.get('heading',1),scales['heading_rad'])
            goal_evidence=None
            if goal_context is not None:
                goal_ref,goal_position,sqrt_weight=goal_context
                goal_residual,goal_jacobian,goal_evidence=first_arrival_residual(states[:,:2],goal_ref,
                    active_goal['s_goal_m'],goal_position,sqrt_weight,active_goal['last_s'])
                goal_jac=lambda c:np.einsum('rni,nij->rj',goal_jacobian,c['all'][:,:2])
                # 一次事件势能0.5*w*||R(p_hit-g)||²，不随时域节点数稀释。
                residual('goal_attraction',goal_residual,goal_jac,.5*self.goals.p['attraction_weight']*n)
            residual('speed',current[:,3]-target,lambda c:c['sens'][:,3]-np.einsum('ni,nij->nj',target_direction,c['sens'][:,:2]),weights.get('speed',1),scales['speed_mps'])
            for index,key,scale,selector in [(0,'acceleration','acceleration_mps2',accel_selector),(1,'steering','steering_rad',steer_selector)]:
                residual('control',controls[:,index],selector,weights.get(key,.1 if index==0 else .2),scales[scale])
            residual('control',delta[:,0]/dt,difference@accel_selector/dt,weights.get('jerk',.2),scales['jerk_mps3'])
            residual('control',delta[:,1]/dt,difference@steer_selector/dt,weights.get('steering_rate',.2),scales['steering_rate_radps'])
            floor=float(risk_cfg.get('clearance_floor_m',.25));increment=float(risk_cfg.get('attention_increment',1.))
            def risk(name,distance,partial,settings,rank_weight):
                extent=float(settings.get('influence_m',12.));eta=float(settings.get('eta',1.))*(1+increment*rank_weight)
                value=np.maximum(1/(distance+floor)-1/(extent+floor),0.)
                derivative=np.where(distance<extent,-1/(distance+floor)**2,0.)
                jac=lambda c:np.einsum('ni,nij->nj',partial() if callable(partial) else partial,c['sens'])*derivative[:,None]
                residual(name,value,jac,weights.get('risk',2)*eta/2)
            geometry_started=time.perf_counter()
            for actor,sample,other_vertices in targets:
                class_risk=risk_cfg.get('classes',{}).get(actor.kind,{'eta':1.,'influence_m':12.})
                if actor.kind=='pedestrian':
                    gap,partial=circle_gap(current,sample[1:,:2],actor.radius or actor.width/2,scene.ego.length,scene.ego.width)
                    distance=np.maximum(gap,0.);distance_partial=partial*(gap>0)[:,None]
                else:
                    gap,partial=rectangle_gap(current,sample[1:],scene.ego.length,scene.ego.width,actor.length,actor.width)
                    # SAT间隙是欧氏净距下界；下界已在势场范围外时，
                    # 其势场严格为零，不必重复求32组顶点—边见证点。
                    relevant=gap<float(class_risk.get('influence_m',12.))
                    distance=np.maximum(gap,0.);distance_partial=np.zeros((n,4))
                    if np.any(relevant):
                        distance[relevant],distance_partial[relevant]=rectangle_distance(
                            current[relevant],vertices[relevant],other_vertices[relevant],gap[relevant])
                penalty('actor_clearance',gap-clearance+tolerance,
                        lambda c,partial=partial:np.einsum('ni,nij->nj',partial,c['sens']))
                minima.append(float((gap-clearance).min()))
                risk('actor_risk',distance,distance_partial,class_risk,attention.get(actor.id,0.))
            line_minimum=np.inf
            for item in physical_lines.values():
                distance=shapely.distance(polygons,item['geometry'])
                contact=distance<=1e-8;signed=distance.copy()
                contact_partial=None
                if np.any(contact):
                    center_nearest=shapely.get_coordinates(shapely.get_point(shapely.shortest_line(item['geometry'],shapely.points(current[:,:2])),0))
                    offset=current[:,:2]-center_nearest;center_distance=np.linalg.norm(offset,axis=1);normal_line=offset/np.maximum(center_distance[:,None],1e-10)
                    c,si=np.cos(current[:,2]),np.sin(current[:,2]);along=normal_line[:,0]*c+normal_line[:,1]*si;across=-normal_line[:,0]*si+normal_line[:,1]*c
                    support=scene.ego.length/2*np.abs(along)+scene.ego.width/2*np.abs(across)
                    signed[contact]=np.minimum(center_distance[contact]-support[contact],0.)
                    contact_partial=np.zeros((n,4));contact_partial[contact,:2]=normal_line[contact]
                    contact_partial[contact,2]=(-scene.ego.length/2*np.sign(along)*across+scene.ego.width/2*np.sign(across)*along)[contact]
                # 候选验收只需真实值；见证点导数留到该候选真正进入下一次QP。
                cache={}
                def line_partial(geometry=item['geometry'],distance=distance,contact=contact,
                                 contact_partial=contact_partial,cache=cache):
                    if 'partial' not in cache:
                        witnesses=shapely.shortest_line(geometry,polygons)
                        remote=shapely.get_coordinates(shapely.get_point(witnesses,0));own=shapely.get_coordinates(shapely.get_point(witnesses,1))
                        normal_line=(own-remote)/np.maximum(distance[:,None],1e-10)
                        partial=np.zeros((n,4));partial[:,:2]=normal_line
                        arm=own-current[:,:2];partial[:,2]=-normal_line[:,0]*arm[:,1]+normal_line[:,1]*arm[:,0]
                        if contact_partial is not None:partial[contact]=contact_partial[contact]
                        cache['partial']=partial
                    return cache['partial']
                penalty('noncrossable_line_clearance',signed-clearance+tolerance,
                        lambda c,partial=line_partial:np.einsum('ni,nij->nj',partial(),c['sens']))
                line_minimum=min(line_minimum,float((signed-clearance).min()))
                risk_partial=lambda partial=line_partial,contact=contact:np.where(contact[:,None],0.,partial())
                risk('line_risk',distance,risk_partial,risk_cfg.get('classes',{}).get('noncrossable_line',{'eta':.03,'influence_m':1.5}),max([attention.get(identity,0.) for identity in item['ids']],default=0.))
            if physical_lines:minima.append(line_minimum)
            if stop_local is not None:
                # 将完整矩形四角投影到同一导航参考线，转弯/偏航时不能用固定L/2代替车头。
                corner_s,_,corner_heading=project_polyline(vertices.reshape(-1,2),global_reference)
                corner_s=corner_s.reshape(n,4);corner_heading=corner_heading.reshape(n,4)
                front=np.argmax(corner_s,axis=1);row=np.arange(n)
                front_heading=corner_heading[row,front]
                front_tangent=np.stack((np.cos(front_heading),np.sin(front_heading)),axis=1)
                front_arm=vertices[row,front]-current[:,:2]
                front_partial=np.c_[front_tangent,-front_tangent[:,0]*front_arm[:,1]+front_tangent[:,1]*front_arm[:,0],np.zeros(n)]
                front_s=corner_s[row,front]
                front_partial[(front_s<=1e-9)|(front_s>=global_arc[-1]-1e-9)]=0.
                gap=stop_local-front_s
                penalty('yield_stop_line',gap+tolerance,lambda c:-np.einsum('ni,nij->nj',front_partial,c['sens']))
                minima.append(float(gap.min()))
            evaluations['geometry_seconds']+=time.perf_counter()-geometry_started
            # 活动目标时不施加原中心线横向/航向吸引；规则速度和进度不变。
            progress=-float(weights.get('progress',1))*(s[-1]-local_s0)/scales['progress_m'];components['progress']=progress
            residual_vector=np.concatenate(residuals)
            objective=float(residual_vector@residual_vector+progress)
            for key in ('actor_risk','line_risk','goal_attraction','contour','heading'):components.setdefault(key,0.)
            metrics={name:value for name,value in zip(labels,minima)}
            control_margin=float(min(np.min(controls[:,0]-min_a),np.min(max_a-controls[:,0]),
                                     np.min(max_delta-np.abs(controls[:,1]))))
            metrics['control_box']=control_margin
            valid=(np.isfinite(states).all() and np.isfinite(controls).all() and minima[1]>=-1e-8
                   and control_margin>=-1e-8 and np.isfinite(objective))
            safety_feasible=(all(initial_safety.values()) and minima[0]>=-tolerance and minima[2]>=-1e-8
                             and all(value>=-tolerance for value in minima[6:])
                             and bool(np.all(shapely.covers(accepted_road,polygons))))
            result=dict(states=states,controls=controls.copy(),objective=objective,components=components,
                lower=np.concatenate(lower),upper=np.concatenate(upper),minima=metrics,valid=bool(valid),polygons=polygons,
                safety_feasible=bool(safety_feasible),
                violation=max(0.,-min(minima),-control_margin),goal_evidence=goal_evidence)
            def linearize():
                linearization_started=time.perf_counter();evaluations['linearizations']+=1
                sensitivity=self.dynamics.sensitivity(controls,initial)
                sens=sensitivity[1:]
                sj=np.einsum('ni,nij->nj',tangent,sens[:,:2])
                sj[~path_interior]=0.
                context={'all':sensitivity,'sens':sens,'sj':sj}
                jacobian=np.vstack([(jac(context) if callable(jac) else np.asarray(jac))*factor for jac,factor in jacobians])
                gradient_extra=-float(weights.get('progress',1))*sj[-1]/scales['progress_m']
                value=dict(hessian=2*jacobian.T@jacobian+np.eye(2*n)*float(p.get('sqp_regularization',1e-4)),
                           gradient=2*jacobian.T@residual_vector+gradient_extra,
                           matrix=np.vstack([row(context) if callable(row) else row for row in rows]))
                evaluations['linearization_seconds']+=time.perf_counter()-linearization_started
                return value
            result['linearize']=linearize
            evaluations['value_seconds']+=time.perf_counter()-value_started
            return result

        solver_started=time.perf_counter();budget=float(p.get('cold_start_budget_s',.5) if cold_start else p.get('solver_wall_budget_s',.1));deadline=started+budget
        qp_before=(self.qp.cache_builds,self.qp.cache_hits,self.qp.setup_seconds)
        statuses=[];last=None;initial_diagnostics=[];qp_seconds=0.;iterations=0
        nominal=terms(controls)
        nominal_source='shifted_previous' if warm else 'navigation_seed'
        def record_initial(value,source):
            initial_diagnostics.append({'source':source,'minimum':min(value['minima'].values()),
                'minimum_by_group':value['minima'],'cost':value['objective'],'cost_terms':value['components']})
        record_initial(nominal,nominal_source)
        initial_reselection={'evaluated':False,'selected':False}
        if not nominal['safety_feasible']:
            initial_reselection.update(evaluated=True,original_violation=nominal['violation'])
            alternatives=[(nominal_source,nominal)]
            if warm:
                navigation=terms(project(seed()));record_initial(navigation,'navigation_seed')
                alternatives.append(('navigation_seed',navigation))
            # 风险较高时比较同一MPC的减速初值；每条均在本帧重新积分、验收并计价。
            if not any(value['safety_feasible'] for _,value in alternatives):
                slower=terms(project(seed(0.)));record_initial(slower,'braking_navigation_seed')
                alternatives.append(('braking_navigation_seed',slower))
            feasible=[item for item in alternatives if item[1]['valid']]
            if not feasible:
                raise ValueError('MPC输入或初值包含非法数值，无法形成有界控制轨迹')
            chosen=min(feasible,key=lambda item:item[1]['objective'])
            initial_reselection.update(selected=chosen[0]!=nominal_source,selected_source=chosen[0],
                selected_violation=chosen[1]['violation'],selected_executable=chosen[1]['valid'],
                selected_safety_feasible=chosen[1]['safety_feasible'])
            nominal_source,nominal=chosen;controls=nominal['controls']
        initial_guess_source=nominal_source
        if not nominal['valid']:
            raise ValueError('MPC当前帧初值未通过动力学与控制边界验收')
        # 优化未改进、未收敛或超时都不会丢掉当前帧已经重新验收的控制轨迹。
        nominal['initialization_source']=nominal_source
        nominal['candidate_source']='validated_current_initialization'
        accepted=nominal
        trust=np.tile([float(p.get('sqp_trust_acceleration_mps2',1.5)),float(p.get('sqp_trust_steering_rad',.15))],n)
        last_iteration_seconds=0.;iteration_history=[];termination='iteration_limit'
        recovery_speed=float(p.get('warm_start_recovery_speed_mps',.3))
        recovery={'eligible':bool(warm and stop_local is None and speed<recovery_speed),
                  'speed_threshold_mps':recovery_speed,'evaluated':False,'selected':False}
        for iteration in range(int(p.get('solver_max_iterations',4))):
            # 至少尝试一次QP；随后预算到期即返回已验收的本帧最佳候选。
            if iteration and time.perf_counter()>=deadline:
                termination='wall_budget';break
            if accepted is not None and iteration and time.perf_counter()+last_iteration_seconds*1.3+.002>=deadline:
                termination='predicted_wall_budget';break
            iteration_started=time.perf_counter()
            if 'linearize' in nominal:nominal.update(nominal.pop('linearize')())
            low=np.maximum(np.tile([min_a,-max_delta],n)-controls.ravel(),-trust)
            high=np.minimum(np.tile([max_a,max_delta],n)-controls.ravel(),trust)
            try:
                step,success,status,spent=self.qp.solve(nominal['hessian'],nominal['gradient'],nominal['matrix'],nominal['lower'],nominal['upper'],low,high)
            except (RuntimeError,np.linalg.LinAlgError) as exc:
                statuses.append(f'QP数值异常：{type(exc).__name__}: {exc}')
                termination='qp_failure';break
            statuses.append(status);qp_seconds+=spent;iterations+=1
            if not success or not np.all(np.isfinite(step)):
                termination='qp_failure';break
            candidates=[]
            evaluation_seconds=0.
            for search in range(int(p.get('sqp_line_search_steps',3))):
                if search and time.perf_counter()+evaluation_seconds*1.2+.001>=deadline:break
                evaluation_started=time.perf_counter()
                proposed=controls+(step.reshape(n,2)*(.5**search))
                proposed=project(proposed)
                candidate=terms(proposed)
                candidate['initialization_source']=nominal_source
                candidate['candidate_source']='optimized_qp_line_search'
                exact=candidate['valid']
                if exact and candidate['objective']<=accepted['objective']:accepted=candidate
                candidates.append(candidate)
                evaluation_seconds=time.perf_counter()-evaluation_started
                if exact and candidate['objective']<=nominal['objective']+1e-8:break
            if not candidates:
                termination='control_projection_infeasible';break
            next_nominal=min([accepted,*[value for value in candidates if value['valid']]],key=lambda value:value['objective'])
            improvement=abs(next_nominal['objective']-nominal['objective'])
            step_norm=float(np.linalg.norm(step,np.inf))
            iteration_history.append({'iteration':iteration+1,'nominal_cost':nominal['objective'],
                'candidate_cost':next_nominal['objective'],'absolute_cost_change':improvement,
                'qp_step_inf':step_norm,'gradient_inf':float(np.linalg.norm(nominal['gradient'],np.inf)),
                'candidate_valid':next_nominal['valid'],'candidate_violation':next_nominal['violation'],
                'qp_residuals':dict(getattr(self.qp,'last_diagnostics',{}))})
            controls=next_nominal['controls'];nominal=next_nominal;last=nominal
            last_iteration_seconds=time.perf_counter()-iteration_started
            # 低速暖解可能把加速持续推迟到时域尾部。先保留本帧已优化、
            # 已验收的候选，再比较同一目标下的导航初值，且共享原预算。
            if iteration==0 and recovery['eligible'] and accepted is not None:
                if iteration+1>=int(p.get('solver_max_iterations',4)):
                    recovery['reason']='iteration_limit'
                elif time.perf_counter()+last_iteration_seconds*1.3+.002>=deadline:
                    recovery['reason']='wall_budget'
                else:
                    alternative=terms(project(seed()));recovery['evaluated']=True
                    exact=alternative['valid']
                    recovery.update(navigation_seed_valid=bool(exact),navigation_seed_cost=alternative['objective'],
                                    warm_candidate_cost=accepted['objective'])
                    if exact and alternative['objective']<accepted['objective']-1e-8:
                        controls=alternative['controls'];nominal=alternative
                        nominal_source='navigation_seed_recovery';recovery['selected']=True
                        alternative['initialization_source']=nominal_source
                        alternative['candidate_source']='validated_current_initialization'
                        accepted=alternative
                        continue
                    recovery['reason']='not_a_better_feasible_initialization'
            # 目标变化很小也可能只是停在低速平台，不能据此宣布收敛。
            # 小优化步才允许提前结束；QP残差同时保存以便核查原因。
            if step_norm<1e-5:
                termination='small_qp_step';break
        elapsed=time.perf_counter()-started
        diagnostics={'solver_seconds':time.perf_counter()-solver_started,'planning_seconds':elapsed,'solver_status':statuses,
            'planning_object_selection':object_selection,
            'solver_backend':'casadi_osqp_realtime_sqp','symbolic_build_seconds':build_seconds,'qp_seconds':qp_seconds,'sqp_iterations':iterations,
            'optimizer_termination':termination,'iteration_history':iteration_history,
            'warm_start_recovery':recovery,
            'initial_guess_source':initial_guess_source,'infeasible_initial_reselection':initial_reselection,
            'qp_cache_builds':self.qp.cache_builds-qp_before[0],'qp_cache_hits':self.qp.cache_hits-qp_before[1],
            'qp_setup_seconds':self.qp.setup_seconds-qp_before[2],
            'value_evaluations':evaluations['values'],'linearization_evaluations':evaluations['linearizations'],
            'value_evaluation_seconds':evaluations['value_seconds'],'linearization_seconds':evaluations['linearization_seconds'],
            'geometry_evaluation_seconds':evaluations['geometry_seconds'],
            'cold_start':cold_start,
            'local_reference':selection.diagnostics,'reference_state_before':reference_state_before,
            'goal_state':self.goals.diagnostics(),'goal_events':self.goals.events,
            'goal_state_before':reference_state_before['goals'],
            'local_reference_revision':selection.revision,'reference_changed':reference_changed,
            'warm_start':warm,'warm_shift_steps':shifted_steps,
            'warm_start_across_goal_update':bool(warm and new_goal_active),
            'warm_start_rejection_reason':warm_rejection,
            'deadline_missed':elapsed>budget,'deadline_budget_seconds':budget,'must_wait':stop_s is not None,'wait_reason':wait_reason,'stop_s':stop_s,
            'rule_evidence':self.rule_evidence,'initial_constraints':initial_diagnostics,
            'map_rejections':[{'actor_id':identity,'reason':info.get('reason'),
                               'trajectory_source':info.get('trajectory_source'),'kinematic_valid':info.get('kinematic_valid')}
                              for identity,info in predictions.diagnostics.items() if info.get('valid') is False],
            'last_optimizer_constraints':None if last is None else {'minimum_by_group':last['minima'],'cost':last['objective']}}
        states=accepted['states'].copy();controls=accepted['controls'].copy()
        states[:,:2]=ego_to_world(states[:,:2],origin,angle);states[:,2]+=angle;self.previous=controls.copy();self.previous_time=scene.time
        self.previous_reference_revision=selection.revision
        diagnostics.update(objective=accepted['objective'],cost_terms=accepted['components'],candidate_source=accepted['candidate_source'],
            accepted_initialization_source=accepted['initialization_source'],
            minimum_constraint=min(accepted['minima'].values()),constraint_minima=accepted['minima'],semantic_enabled=answer is not None,
            active_goal=active_goal is not None,goal_evaluation=accepted.get('goal_evidence'),
            constraint_policy='bounded_controls_soft_safety_v1',trajectory_available=True,
            safety_feasible=accepted['safety_feasible'],initial_safety_checks=initial_safety,
            hard_constraint_groups=['dynamics','acceleration','steering','nonnegative_speed'],
            soft_constraint_violations={key:max(0.,-value) for key,value in accepted['minima'].items()
                                        if key not in ('nonnegative_speed','control_box')},
            geometry_progress='closest_point_eliminated_free_progress')
        # 首控制差分锚定真实已执行控制；数值热启动不等于硬连续性保证。
        diagnostics['control_transition']={
            'previous_executed':previous_control.tolist(),
            'first_planned':controls[0].tolist(),
            'acceleration_change_mps2':float(controls[0,0]-previous_control[0]),
            'steering_change_rad':float(controls[0,1]-previous_control[1]),
        }
        result=PlanResult(scene.frame_id,scene.time,True,states,controls,diagnostics=diagnostics)
        diagnostics['planning_seconds']=time.perf_counter()-started
        diagnostics['deadline_missed']=diagnostics['planning_seconds']>budget
        return result
