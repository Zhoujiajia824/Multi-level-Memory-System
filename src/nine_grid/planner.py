"""参考走廊跟踪MPC：正常轨迹安全优先；无安全解时单列最小风险制动。"""
from __future__ import annotations

import time
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry import LineString
from .contracts import Actor, PlanResult
from .planning_objects import planning_view, road_rule_observations
from .local_reference import (LocalReferenceManager, LocalReferenceResult,
                              rear_covered_reference, terminal_covered_reference)
from .goals import GoalManager, goal_seed_path, first_arrival_residual
from .lane_potential import LanePotentialGeometry
from .potential_domain import build_lane_potential_domain
from .navigation_connections import confirmed_navigation_connections
from .realtime_solver import Dynamics, QuadraticSolver, rectangle_gap, circle_gap, bicycle_geometry
from .stop_control import brake_acceleration_limit
from .geometry import (clean_polyline, ego_to_world, world_to_ego,
                       rectangle_corners, polyline_lengths, project_polyline, sample_polyline, wrap_angle)


def propagate(state, control, length=5., dt=.1, substeps=5,
              wheelbase_m=None,rear_axle_to_center_m=None):
    """几何中心自行车模型显式Euler；未提供轴距时保持highway原公式。"""
    state=np.asarray(state,float).copy();a,delta=np.asarray(control,float)
    wheelbase,rear=bicycle_geometry(length,wheelbase_m,rear_axle_to_center_m)
    beta=np.arctan(rear/wheelbase*np.tan(delta));step=dt/substeps
    for _ in range(substeps):
        x,y,psi,v=state
        state=np.array([x+v*np.cos(psi+beta)*step,y+v*np.sin(psi+beta)*step,
                        psi+v/rear*np.sin(beta)*step,v+a*step])
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


def _reference_window(reference,start,end,anchors=()):
    """Sample a window without shortcutting its original polyline vertices.

    Returned stations retain their ordered branch identity, including an exact
    task endpoint; nearest-XY projection is not used to relocate a loop's end.
    """
    reference=clean_polyline(reference);arc=polyline_lengths(reference)
    start=float(np.clip(start,0.,arc[-1]));end=float(np.clip(end,start,arc[-1]))
    fixed=np.unique(np.r_[start,end,arc[(arc>=start)&(arc<=end)],
        [s for s in anchors if start<=s<=end]])
    grid=np.linspace(start,end,max(2,int(np.ceil((end-start)/.5))+1))
    grid=grid[np.min(np.abs(grid[:,None]-fixed[None,:]),axis=1)>1e-7]
    stations=np.sort(np.r_[fixed,grid])
    return sample_polyline(reference,stations),stations


def _stop_speed_with_slope(s,stop_local,length,braking):
    """停车参考的饱和值与同分支导数；零点采用已截断侧导数。"""
    speed=np.sqrt(np.maximum(0.,2*braking*(stop_local-s-length/2)))
    derivative=np.divide(-braking,speed,out=np.zeros_like(speed),where=speed>0.)
    return speed,derivative


def _following_speed_with_slope(net_gap, lead_speed, minimum_gap, headway, braking):
    """净车距的时距速度包络；同速平衡车距为minimum_gap+headway*v。"""
    gap=np.asarray(net_gap,float)-minimum_gap
    linear=np.maximum(0.,gap/headway)
    stopping=np.sqrt(np.maximum(0.,np.asarray(lead_speed,float)**2+2*braking*gap))
    use_linear=linear<=stopping
    desired=np.minimum(linear,stopping)
    # 对自车沿参考弧长s求导：净车距随s增加而缩短。
    derivative=np.where(use_linear,np.where(linear>0.,-1/headway,0.),
        np.divide(-braking,stopping,out=np.zeros_like(stopping),where=stopping>1e-6))
    return desired,np.where(np.isfinite(gap),derivative,0.)


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
        """当前观测和公开道路优先级确定让行，不能用自身停车解释应该等待。"""
        stop_s=scene.rules.get("stop_s")
        ego_s=project_polyline(scene.ego.position,scene.route,scene.diagnostics.get("route_s"),scene.ego.heading)[0]
        lanes={lane.id:lane for lane in scene.lanes}
        ego_lane=lanes.get(scene.ego.lane_id)
        reason=scene.rules.get("wait_reason")
        rules=self.config.get("rules",{})
        stop_margin=float(rules.get("stop_margin_m",1.))
        self.rule_evidence=[]
        observations=road_rule_observations(scene)
        rule_actors=[] if observations is None else [Actor(
            id=row['id'],kind=row['kind'],position=np.asarray(row['position'],float),
            velocity=np.asarray(row['velocity'],float),acceleration=np.zeros(2),
            heading=row['heading'],length=row['length'],width=row['width'],lane_id=row['lane_id'])
            for row in observations['actors']]
        self.rule_observation_scope={'frame_id':scene.frame_id,'time':float(scene.time),
            'coordinates':scene.coordinates,'source':'current_visible_public_lane_observations_before_mpc_selection',
            'rule_actor_ids':[actor.id for actor in rule_actors],'rule_actor_count':len(rule_actors),
            'mpc_actor_ids':sorted(predictions.actors),'mpc_actor_count':len(predictions.actors),
            'extra_ekf_predictions':0,'public_path_range_m':55.,
            'path_selection_source':'observed_motion_public_successor_geometry'}
        if ego_lane and scene.rules.get("road_priority_available"):
            route=scene.route;route_shape=LineString(route)
            for actor in rule_actors:
                identity=actor.id
                other_lane=lanes.get(actor.lane_id)
                audit={"actor_id":identity,"source":"visible_public_lane_geometry","active":False,
                    "in_mpc_prediction_set":identity in predictions.actors,
                    "path_selection_source":"observed_motion_public_successor_geometry"}
                self.rule_evidence.append(audit)
                if not actor.visible or not other_lane:
                    audit["reason"]="对象不可见或缺少公开车道";continue
                # 已在自车同一公开路段的对象属于跟驰；不能仅凭当前朝向
                # 相似删除汇入冲突，也不能把对向车当作同向跟驰。
                if actor.lane_id in scene.route_lane_ids:
                    audit["reason"]="已在自车同一公开路段，按跟驰处理";continue
                line=other_lane.center.copy();lane=other_lane;visited={lane.id};path_ids=[lane.id]
                actor_s=project_polyline(actor.position,line,heading=actor.heading)[0]
                while polyline_lengths(line)[-1]-actor_s<55.:
                    options=[lanes[key] for key in lane.successors if key in lanes and key not in visited]
                    if not options:break
                    # 规则入口只依赖当前观测和公开几何，不继承EKF短时
                    # 后继假设；否则微小滤波变化会反复撤销/重建停车入口。
                    # EKF动态轨迹仍完整用于实体避碰；这里不另做运动预测。
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

    def emergency_brake(self,scene,predictions,reason,answer=None):
        """执行前重审失败时，从当前物理状态单独生成制动，不借用旧控制。"""
        braking_planner=Planner(self.config)
        # 保留本帧关注带来的实体约束，但制动实例不激活模型目标。
        scoped, scoped_predictions, _ = planning_view(scene,predictions,self.config,answer)
        result=braking_planner.solve(scoped,scoped_predictions,None,_brake_only=True)
        result.diagnostics['emergency_request_reason']=str(reason)
        result.diagnostics['emergency_generation']='fresh_current_state_braking'
        return result

    def solve(self,scene,predictions,answer=None,*,_brake_only=False):
        started=time.perf_counter();p=self.p;n=self.n;dt=self.dt
        _validate_prediction_input(scene,predictions,n*dt)
        # 完整输入必须先验明；局部MPC视图不污染模型、检索、反思和独立审计。
        scene,predictions,object_selection=planning_view(scene,predictions,self.config,answer)
        wheelbase,rear=bicycle_geometry(scene.ego.length,scene.ego.wheelbase_m,scene.ego.rear_axle_to_center_m)
        axle_ratio=rear/wheelbase
        cold_start=not getattr(self,'_has_solved',False);self._has_solved=True
        tolerance=float(p.get("body_constraint_tolerance_m",.001))
        speed_tolerance=float(p.get('speed_numerical_tolerance_mps',1e-4))
        if not np.isfinite(speed_tolerance) or speed_tolerance<0.:
            raise ValueError('速度数值容差必须是有限非负m/s')
        braking=float(self.config.get("rules",{}).get("comfort_braking_mps2",3.))
        origin=scene.ego.position.copy();angle=scene.ego.heading
        speed=float(np.linalg.norm(scene.ego.velocity));initial=np.array([0.,0.,0.,speed])
        build_seconds=0.
        dynamics_key=(n,dt,self.substeps,wheelbase,rear)
        if not hasattr(self,'dynamics') or self.dynamics_key!=dynamics_key:
            self.dynamics=Dynamics(n,dt,self.substeps,scene.ego.length,wheelbase,rear);self.dynamics_key=dynamics_key
            self.qp=QuadraticSolver(2*n,int(p.get('qp_max_iterations',600)),float(p.get('qp_tolerance',1e-6)))
            build_seconds=self.dynamics.build_seconds
        ego_s=project_polyline(origin,scene.route,scene.diagnostics.get("route_s"),angle)[0]
        full_arc=polyline_lengths(scene.route)
        terminal_enabled=scene.scenario=='fail2drive'
        support_front=max(20.,speed*4.)
        navigation_support,terminal_support=terminal_covered_reference(scene,scene.route,forward_m=support_front)
        support_arc=polyline_lengths(navigation_support)
        start_arc=max(0.,ego_s-20);end_arc=min(support_arc[-1],ego_s+max(120,speed*4))
        global_world,global_stations=_reference_window(navigation_support,start_arc,end_arc,
                                                      anchors=(float(full_arc[-1]),))
        global_reference=world_to_ego(global_world,origin,angle)
        global_arc=polyline_lengths(global_reference)
        terminal_local=float(full_arc[-1]-start_arc) if terminal_enabled else None
        terminal_indices=np.flatnonzero(global_stations==full_arc[-1])
        if terminal_enabled and len(terminal_indices):
            terminal_local=float(global_arc[int(terminal_indices[0])])
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
        if _brake_only:
            lane_table={lane.id:lane for lane in scene.lanes}
            own_lane=self.references._lane(scene,lane_table)
            seed_world=scene.route if own_lane is None else self.references._continuation(scene,own_lane,lane_table)
            selection=LocalReferenceResult(seed_world,0,{'phase':'brake_only','selected_source':'navigation_consistent_current_lane','maneuver_active':False})
        elif active_goal is not None:
            try:
                seed_world=goal_seed_path(scene,active_goal)
            except (ValueError,FloatingPointError) as error:
                self.goals._end(active_goal,'invalid','当前物理状态无法连接原目标：'+str(error),scene)
                self.references.reset_intent()
                result=self.emergency_brake(scene,predictions,'目标几何连接失效：'+str(error))
                result.diagnostics['goal_state']=self.goals.diagnostics()
                result.diagnostics['goal_events']=self.goals.events
                return result
            selection=LocalReferenceResult(seed_world,
                int(self.references.state['revision'])+self.goals.revision,
                {'phase':'goal_active','selected_source':'local_goal','maneuver_active':False,
                 'selection_wall_s':0.,'goal_geometry_role':'quintic_reference_before_first_crossing'})
        else:
            selection=self.references.select(scene,predictions)
            selection.revision+=self.goals.revision
            seed_world=selection.world_path
        selection.world_path,control_support=terminal_covered_reference(
            scene,selection.world_path,forward_m=support_front)
        seed_world,seed_support=terminal_covered_reference(scene,seed_world,forward_m=support_front)
        local_arc=polyline_lengths(selection.world_path)
        selected_s=project_polyline(origin,selection.world_path,heading=angle)[0]
        local_begin=max(0.,selected_s-20.)
        local_end=min(local_arc[-1],selected_s+max(120.,speed*4))
        reference_world,_=_reference_window(selection.world_path,local_begin,local_end)
        reference=world_to_ego(reference_world,origin,angle)
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
        reach=speed*horizon+.5*max(0.,float(p.get('acceleration_max_mps2',3.)))*horizon**2+np.hypot(scene.ego.length,scene.ego.width)/2+.01
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
                if line.get('crossable',True) or line.get('kind') in ('virtual_connection','virtual_lane_boundary'):continue
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
            sqrt_weight=np.diag([1/self.goals.p['longitudinal_scale_m'],1/self.goals.p['lateral_scale_m']])
            goal_context=(goal_ref,goal_position,sqrt_weight)
        domain_world,domain_evidence=build_lane_potential_domain(scene,front_m=max(120.,reach),rear_m=20.)
        # 换道五次参考是跟踪目标；势场坐标始终沿导航车道，不随避让轨迹摆动。
        potential_world=(np.asarray(active_goal['goal_reference']['points'],float)
                         if active_goal is not None else scene.route)
        potential_old_s=(active_goal['last_s'] if active_goal is not None else
            project_polyline(origin,scene.route,scene.diagnostics.get('route_s'),angle)[0])
        potential_world,potential_forward_support=terminal_covered_reference(
            scene,potential_world,forward_m=support_front)
        potential_world,potential_reference_evidence=rear_covered_reference(
            scene,potential_world,rear_m=20.,anchor_s=potential_old_s)
        potential_reference_evidence['forward_support']=potential_forward_support
        potential_prefix_s=float(potential_reference_evidence['added_length_m'])
        potential_reference=clean_polyline(world_to_ego(potential_world,origin,angle))
        # 有效面在刚体旋转后仍可能因接触边浮点退化而自交。保留世界域，
        # 仅转换查询点和法向，避免修补/扩张/简化授权车道几何。
        potential_geometry=LanePotentialGeometry(potential_reference,domain_world,domain_pose=(origin,angle))
        potential_s0=float(potential_geometry.chart_arc_from_polyline_s(potential_old_s+potential_prefix_s))
        potential_geometry.anchor_s=potential_s0
        targets=[]
        for identity,trajectory in predictions.trajectories.items():
            actor=predictions.actors.get(identity)
            if actor is None or not actor.visible:continue
            trajectory=np.asarray(trajectory).copy();trajectory[:,2]=np.unwrap(trajectory[:,2])
            sample=np.stack([np.interp(np.arange(n+1)*dt,predictions.times,trajectory[:,j]) for j in range(4)],axis=1)
            sample[:,:2]=world_to_ego(sample[:,:2],origin,angle);sample[:,2]-=angle
            targets.append((actor,sample,body_vertices(sample[1:],actor.length,actor.width)))
        source_projections={};source_anchors={}
        for actor,sample,_ in targets:
            matched=object_selection['selected_state'][actor.id].get('memberships',{})
            if matched:
                membership=min(matched.values(),key=lambda row:abs(row['delta_s_m']))
                offset=float(membership['delta_s_m'])
                anchor=float(np.clip(potential_s0+offset,0.,potential_geometry.length))
                source='lane_scope_membership'
            else:
                # 外部关注对象不伪造八向角色，只投影到本次导航图表。
                bev=scene.diagnostics.get('perception_scope',{}).get('settings',{}).get('bev',{})
                window=max(20.,float(bev.get('front_m',70.))+float(bev.get('rear_m',30.)))
                projected=potential_geometry.project(sample[0,:2],anchors=potential_s0,
                    require_domain=False,window_m=window)
                anchor=float(projected['s'][0]);offset=anchor-potential_s0
                source='bev_attention_current_center_on_navigation_chart'
            source_anchors[actor.id]=dict(source=source,anchor_s_m=anchor,
                has_eight_region_membership=bool(matched))
            source_projections[actor.id]=potential_geometry.project_sequence(sample[:,:2],
                anchor,require_domain=False,
                initial_window_m=min(20.,max(10.,5.+.25*abs(offset))))
        source_coverage={key:int((~value['coordinate_valid']).sum()) for key,value in source_projections.items()
                         if not np.all(value['coordinate_valid'])}
        attention_rows=answer.attention if answer is not None else object_selection.get('attention_signature',[])
        attention={str(a.get('id',a.get('entity_id',''))):1/max(1,int(a.get('rank',i+1))) for i,a in enumerate(attention_rows)}
        clearance=float(risk_cfg.get('body_clearance_margin_m',.05))
        increment=float(risk_cfg.get('attention_increment',1.))
        max_a=float(p.get('acceleration_max_mps2',3));min_a=float(p.get('acceleration_min_mps2',-6))
        max_delta=float(p.get('steering_abs_max_rad',.6));max_jerk=float(p.get('jerk_abs_max_mps3',8));max_rate=float(p.get('steering_rate_abs_max_radps',.8));max_lat=float(p.get('lateral_acceleration_abs_max_mps2',4.89))
        max_speed=min(float(p.get('maximum_speed_mps',33.3333333)),float(scene.rules.get('speed_limit',33.3333333)))
        weights=p.get('weights',{});norm=p.get('normalization',{})
        penalties={'actor_clearance':2000.,'road_clearance':500.,'noncrossable_line_clearance':100.,
                   'yield_stop_line':100.,'speed_limit':10.,'lateral_acceleration':5.,'following_headway':20.}
        penalties.update(p.get('safety_penalties',{}))
        scales={key:float(norm.get(key,default)) for key,default in [('position_m',3.5),('heading_rad',.5),('speed_mps',10),('acceleration_mps2',3),('steering_rad',.6),('jerk_mps3',8),('steering_rate_radps',.8),('progress_m',30)]}
        # 同一几何参考上的前方实体包络。只使用已纳入规划的当前/预测信息。
        headway=float(p.get('follow_time_headway_s',1.2));follow_gap=float(p.get('follow_min_gap_m',2.))
        terminal_braking=float(p.get('terminal_braking_mps2',4.))
        if headway<=0 or terminal_braking<=0 or follow_gap<0:
            raise ValueError('跟驰时距、制动减速度必须为正，停车间距不得为负')
        lead_rear=np.full(n+1,np.inf);lead_speed=np.zeros(n+1);lead_ids=[None]*(n+1)
        lead_classification=[]
        stationary_speed=float(self.config.get('prediction',{}).get('low_speed_mps',.3))
        for actor,sample,_ in targets:
            actor_s,offset,actor_tangent,_=_path_projection(sample[:,:2],reference)
            # 跟驰包络只用于当前位于自车前方的对象；后车驶过原帧原点，
            # 不等于驶到了未来自车前方。后车仍完整参与逐时实体碰撞约束。
            if actor_s[0]<=local_s0:
                lead_classification.append({'id':actor.id,'current_front':False,'eligible_steps':[],
                    'excluded_moving_conflict_steps':[]})
                continue
            axis=np.arctan2(actor_tangent[:,1],actor_tangent[:,0])
            phi=sample[:,2]-axis
            along=(actor.radius or actor.width/2) if actor.kind=='pedestrian' else actor.length/2*np.abs(np.cos(phi))+actor.width/2*np.abs(np.sin(phi))
            across=(actor.radius or actor.width/2) if actor.kind=='pedestrian' else actor.length/2*np.abs(np.sin(phi))+actor.width/2*np.abs(np.cos(phi))
            rear_s=actor_s-along
            longitudinal_speed=sample[:,3]*np.cos(phi)
            lateral_speed=sample[:,3]*np.sin(phi)
            stationary=(sample[:,3]<=stationary_speed)|((actor.kind=='static_obstacle')&(actor.motion_model!='free'))
            same_direction=longitudinal_speed>np.abs(lateral_speed)
            # 横穿/对向仍参与逐时实体避碰和路权，让行不等于同向跟驰。
            # 预测切入后同向的阶段、已停在通道内的实体才进入跟车包络。
            spatial=(np.linalg.norm(offset,axis=1)<=across+scene.ego.width/2+clearance)&(actor_s>local_s0)
            eligible=spatial&(stationary|same_direction)
            lead_classification.append({'id':actor.id,'current_front':True,
                'eligible_steps':np.flatnonzero(eligible).tolist(),
                'stationary_blocker_steps':np.flatnonzero(spatial&stationary).tolist(),
                'excluded_moving_conflict_steps':np.flatnonzero(spatial&~(stationary|same_direction)).tolist()})
            mask=eligible&(rear_s<lead_rear)
            lead_rear[mask]=rear_s[mask]
            lead_speed[mask]=np.maximum(0.,sample[mask,3]*np.cos(phi[mask]))
            for index in np.flatnonzero(mask):lead_ids[index]=actor.id
        def following_speed(ss, indices):
            net_gap=lead_rear[indices]-np.asarray(ss)-scene.ego.length/2
            return _following_speed_with_slope(net_gap,lead_speed[indices],follow_gap,headway,terminal_braking)
        # 目标纵向截面之后，仅由预测位置所在的同向合法车道确定基础引导。
        lane_references=[]
        for lane in scene.lanes:
            line=clean_polyline(world_to_ego(lane.center,origin,angle))
            if len(line)>=2:lane_references.append((lane.id,line))
        lane_widths={lane.id:lane.width for lane in scene.lanes}
        def tracking_geometry(current):
            ss,residual,tangent,interior=_path_projection(current[:,:2],reference)
            progress_tangent=tangent.copy();progress_interior=interior.copy()
            hd,hd_d=_interpolate_with_slope(ss,reference_arc,tangents)
            goal_mask=np.ones(len(current),bool)
            post_goal_corridor=np.zeros(len(current))
            if active_goal is not None:
                source_s=project_polyline(current[:,:2],goal_context[0],previous_s=active_goal['last_s'],window=120.)[0]
                goal_mask=source_s<active_goal['s_goal_m']
                for k in np.flatnonzero(~goal_mask):
                    options=[]
                    for identity,line in lane_references:
                        ls,lr,lt,li=_path_projection(current[k:k+1,:2],line)
                        direction=float(np.arctan2(lt[0,1],lt[0,0]))
                        if abs(float(wrap_angle(direction-current[k,2])))>np.pi/2:continue
                        options.append((float(lr[0]@lr[0]),identity,lr[0],lt[0],bool(li[0]),direction))
                    if options:
                        _,identity,residual[k],tangent[k],interior[k],hd[k]=min(options,key=lambda value:(value[0],value[1]))
                        post_goal_corridor[k]=max(0.,(lane_widths[identity]-scene.ego.width)/2)
                        hd_d[k]=0. # 分段直线投影内方向恒定，切段后重新线性化。
            return ss,residual,tangent,interior,hd,hd_d,goal_mask,progress_tangent,progress_interior,post_goal_corridor
        difference=np.eye(n)-np.eye(n,k=-1);accel_selector=np.eye(2*n)[::2];steer_selector=np.eye(2*n)[1::2]
        # 导航连接已按同一自行车模型积分并验收车体朝向。变曲率段不能
        # 再以稳态圆弧公式重造姿态，否则同一中心线也可能产生越线初值。
        # 只给完全相同的纯导航参考使用；模型目标/主动换道有自己的参考。
        seed_connections=[]
        if (active_goal is None and np.array_equal(seed_world,scene.route) and
                np.isclose(wheelbase,scene.ego.length,rtol=0.,atol=1e-9) and
                np.isclose(rear,scene.ego.length/2,rtol=0.,atol=1e-9)):
            for connection in confirmed_navigation_connections(scene).values():
                points=world_to_ego(connection['points'],origin,angle)
                derivatives=np.gradient(points,axis=0)
                seed_connections.append(dict(points=points,arc=polyline_lengths(points),
                    headings=np.unwrap(connection['headings']-angle),
                    tangents=np.unwrap(np.arctan2(derivatives[:,1],derivatives[:,0])),
                    width_m=connection['width_m']))
        def seed(scale=1.):
            controls=[];state=initial.copy();last=previous_control.copy()
            for step_index in range(n):
                s,d,_=project_polyline(state[:2],seed_reference);kappa=float(np.interp(s,seed_arc,seed_curvature))
                feed=np.arctan(wheelbase*kappa/np.sqrt(max(.01,1-(rear*kappa)**2)))
                heading=float(np.interp(s,seed_arc,seed_tangents))-np.arctan(axle_ratio*np.tan(feed))
                connection_matches=[]
                for connection in seed_connections:
                    cs,_,_=project_polyline(state[:2],connection['points'])
                    foot=sample_polyline(connection['points'],cs)
                    if (.05<cs<connection['arc'][-1]-.05 and
                            np.linalg.norm(state[:2]-foot)<=connection['width_m']/2):
                        ch=float(np.interp(cs,connection['arc'],connection['headings']))
                        theta=float(np.interp(cs,connection['arc'],connection['tangents']))
                        if abs(float(wrap_angle(theta-state[2])))<=.7:
                            connection_matches.append((ch,np.arctan(np.tan(theta-ch)/axle_ratio)))
                if len(connection_matches)==1:
                    heading,feed=connection_matches[0]
                steer=feed+1.2*float(wrap_angle(heading-state[2]))-.3*np.arctan(d/max(state[3],2.))
                navigation_s=s if active_goal is None else project_polyline(state[:2],reference)[0]
                desired=min(max_speed,float(scene.rules.get('desired_speed',25.))*scale,np.sqrt(max_lat*2*rear/max(2*abs(np.sin(np.arctan(axle_ratio*np.tan(steer)))),.001))*.9,float(np.interp(navigation_s,reference_arc,speed_profile)))
                if terminal_local is not None:
                    terminal_s=project_polyline(state[:2],global_reference)[0]
                    desired=min(desired,float(_stop_speed_with_slope(terminal_s,terminal_local,0.,braking)[0]))
                follow_speed,_=following_speed(np.array([navigation_s]),np.array([step_index]))
                desired=min(desired,float(follow_speed[0]))
                if stop_local is not None:
                    global_s=project_polyline(state[:2],global_reference)[0]
                    desired=min(desired,np.sqrt(max(0.,2*braking*(stop_local-global_s-scene.ego.length/2))))
                control=np.clip([np.clip((desired-state[3])*1.2,last[0]-max_jerk*dt,last[0]+max_jerk*dt),np.clip(steer,last[1]-max_rate*dt,last[1]+max_rate*dt)],[min_a,-max_delta],[max_a,max_delta])
                control[0]=max(control[0],-max(0.,state[3])/dt)
                controls.append(control);state=propagate(state,control,scene.ego.length,dt,self.substeps,wheelbase,rear);last=control
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
                        'speed_limit':bool(speed<=max_speed+speed_tolerance)}
        initial_safety['potential_domain']=bool(potential_geometry.domain_contains(initial[:2])[0])
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

        def dense_check(controls,dense):
            """按物理子步重积分；周车仅在原预测样本之间插值，不读取真实未来。"""
            sub_dt=dt/self.substeps
            times=np.arange(len(dense))*sub_dt
            bodies=shapely.polygons(body_vertices(dense,scene.ego.length,scene.ego.width))
            inside=shapely.covers(accepted_road,bodies)
            domain_inside=potential_geometry.domain_contains(dense[:,:2])
            line_ok=np.ones(len(dense),bool)
            line_clearance_minimum=np.inf
            for item in physical_lines.values():
                line_distance=shapely.distance(bodies,item['geometry'])
                line_clearance_minimum=min(line_clearance_minimum,float((line_distance-clearance).min()))
                line_ok &= line_distance>=clearance-tolerance
            actor_ok=np.ones(len(dense),bool);contacts=[];minimum=np.inf
            for actor,sample,_ in targets:
                other=np.stack([np.interp(times,np.arange(n+1)*dt,sample[:,j]) for j in range(4)],axis=1)
                if actor.kind=='pedestrian':
                    distance,_=circle_gap(dense,other[:,:2],actor.radius or actor.width/2,scene.ego.length,scene.ego.width)
                else:
                    distance,_=rectangle_gap(dense,other,scene.ego.length,scene.ego.width,actor.length,actor.width)
                minimum=min(minimum,float(np.min(distance-clearance)))
                actor_ok &= distance>=clearance-tolerance
                hit=np.flatnonzero(distance<0.)
                if len(hit):
                    index=int(hit[0]);normal=other[index,:2]-dense[index,:2]
                    normal=normal/max(float(np.linalg.norm(normal)),1e-9)
                    command_index=min(n-1,index//self.substeps)
                    slip=np.arctan(axle_ratio*np.tan(controls[command_index,1]))
                    own_velocity=dense[index,3]*np.array([np.cos(dense[index,2]+slip),np.sin(dense[index,2]+slip)])
                    other_velocity=other[index,3]*np.array([np.cos(other[index,2]),np.sin(other[index,2])])
                    contacts.append({'actor_id':actor.id,'time_s':float(times[index]),
                        'closing_speed_mps':max(0.,float((own_velocity-other_velocity)@normal)),
                        'ego_speed_mps':float(dense[index,3])})
            stop_ok=True
            if stop_local is not None:
                front_s=project_polyline(body_vertices(dense,scene.ego.length,scene.ego.width).reshape(-1,2),global_reference)[0]
                stop_ok=bool(np.max(front_s)<=stop_local+tolerance)
            terminal_overshoot=0.
            if terminal_local is not None:
                terminal_overshoot=max(0.,float(np.max(_path_projection(dense[:,:2],global_reference)[0])-terminal_local))
            terminal_ok=terminal_overshoot<=tolerance
            return {'valid':bool(inside.all() and domain_inside.all() and line_ok.all() and actor_ok.all() and stop_ok and terminal_ok),
                'substep_s':sub_dt,'samples':len(dense),'actor_clearance_min_m':None if not np.isfinite(minimum) else minimum,
                'offroad_samples':int((~inside).sum()),'line_violation_samples':int((~line_ok).sum()),
                'line_clearance_min_m':None if not np.isfinite(line_clearance_minimum) else line_clearance_minimum,
                'outside_potential_domain_samples':int((~domain_inside).sum()),
                'task_terminal_passed':not terminal_ok,'task_terminal_max_overshoot_m':terminal_overshoot,
                'stop_line_passed':not stop_ok,'predicted_contacts':contacts,
                'limitation':'物理子步离散复核，不是连续时间碰撞证明；周车未来来自预测'}

        def terms(controls,is_brake=False):
            value_started=time.perf_counter();evaluations['values']+=1
            # 线搜索只需真实原值；只有进入下一次QP的名义点才计算控制导数。
            dense_states=self.dynamics.dense_rollout(controls,initial)
            states=dense_states[::self.substeps];current=states[1:]
            field_projection=potential_geometry.project_sequence(states[:,:2],potential_s0)
            field_query={key:value[1:] for key,value in field_projection.items()}
            geometry_started=time.perf_counter()
            s,path_residual,tangent,path_interior,heading,heading_derivative,goal_tracking_mask,s_tangent,s_interior,post_goal_corridor=tracking_geometry(current)
            navigation_s,_,navigation_tangent,navigation_interior=_path_projection(current[:,:2],global_reference)
            vertices=body_vertices(current,scene.ego.length,scene.ego.width);polygons=shapely.polygons(vertices)
            probes=np.concatenate((vertices,(vertices+np.roll(vertices,-1,axis=1))/2),axis=1)
            points=shapely.points(probes.reshape(-1,2));closest=shapely.get_coordinates(shapely.get_point(shapely.shortest_line(boundary,points),0))
            offsets=probes.reshape(-1,2)-closest;distances=np.linalg.norm(offsets,axis=1)
            signs=np.where(shapely.covers(road,points),1.,-1.);road_distance=(distances*signs).reshape(n,8)
            normals=(signs[:,None]*offsets/np.maximum(distances[:,None],1e-10)).reshape(n,8,2)
            arms=probes-current[:,:2,None].transpose(0,2,1)
            road_partial=np.zeros((n,8,4));road_partial[:,:,:2]=normals;road_partial[:,:,2]=-normals[:,:,0]*arms[:,:,1]+normals[:,:,1]*arms[:,:,0]
            # 正常模式的物理/道路/碰撞规则均进入当前线性化约束；最终仍做非线性复核。
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
            beta=np.arctan(axle_ratio*np.tan(controls[:,1]));bp=axle_ratio/(np.cos(controls[:,1])**2*(1+axle_ratio**2*np.tan(controls[:,1])**2))
            # 恒定加速度段的速度单调；分别约束首尾，覆盖段内峰值，避免max速度切换的导数歧义。
            lateral_before=states[:-1,3]**2/rear*np.sin(beta)
            lateral_after=current[:,3]**2/rear*np.sin(beta)
            lateral=np.r_[lateral_before,lateral_after]
            lateral_jac=lambda c:np.vstack((
                states[:-1,3,None]*2/rear*np.sin(beta[:,None])*c['all'][:-1,3]
                +states[:-1,3,None]**2/rear*np.cos(beta[:,None])*bp[:,None]*steer_selector,
                current[:,3,None]*2/rear*np.sin(beta[:,None])*c['sens'][:,3]
                +current[:,3,None]**2/rear*np.cos(beta[:,None])*bp[:,None]*steer_selector))
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
                residual('penalty_'+name,np.minimum(margin,0.),derivative,penalties.get(name,100.),scale)
                rows.append(jac);lower.append(-margin);upper.append(np.full(len(margin),np.inf))
            penalty('road_clearance',road_distance.ravel()+tolerance,road_jac)
            domain_inside=field_projection['domain_valid'][1:]
            domain_margin,domain_normal=potential_geometry.domain_clearance(current[:,:2])
            penalty('potential_domain',domain_margin,
                lambda c:np.einsum('ni,nij->nj',domain_normal,c['sens'][:,:2]))
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
            rows.extend([difference@accel_selector,difference@steer_selector])
            lower.extend([-max_jerk*dt-delta[:,0],-max_rate*dt-delta[:,1]])
            upper.extend([max_jerk*dt-delta[:,0],max_rate*dt-delta[:,1]])
            target,target_derivative=_interpolate_with_slope(s,reference_arc,speed_profile)
            # 参考速度与法规硬限速相容；不能主动追求高于限速的速度。
            target_derivative=np.where(target>max_speed,0.,target_derivative)
            target=np.minimum(target,max_speed)
            target_direction=target_derivative[:,None]*s_tangent*s_interior[:,None]
            follow_target,follow_derivative=following_speed(s,np.arange(1,n+1))
            following=follow_target<target
            target=np.minimum(target,follow_target)
            target_direction=np.where(following[:,None],follow_derivative[:,None]*s_tangent*s_interior[:,None],target_direction)
            # 时距是舒适跟驰目标而非初态可行性门槛；近距/切入时允许连续减速恢复。
            # 单独惩罚欠距，避免导航进度收益长期抵消速度跟踪而再次贴近前车。
            headway_margin=lead_rear[1:]-s-scene.ego.length/2-follow_gap-headway*current[:,3]
            residual('penalty_following_headway',np.minimum(headway_margin,0.),
                lambda c:(-c['sj']-headway*c['sens'][:,3])*(headway_margin<0.)[:,None],
                penalties['following_headway'],scales['position_m'])
            if stop_local is not None:
                global_s,_,global_tangent,global_interior=_path_projection(current[:,:2],global_reference)
                stop_speed,stop_derivative=_stop_speed_with_slope(global_s,stop_local,scene.ego.length,braking)
                stopped=stop_speed<target;target=np.minimum(target,stop_speed)
                target_direction=np.where(stopped[:,None],stop_derivative[:,None]*global_tangent*global_interior[:,None],target_direction)
            if terminal_local is not None:
                # The task endpoint refers to the ego centre, not a red-light
                # bumper stop. Geometry after it is support, never a new goal.
                terminal_speed,terminal_derivative=_stop_speed_with_slope(
                    navigation_s,terminal_local,0.,braking)
                ending=terminal_speed<target;target=np.minimum(target,terminal_speed)
                target_direction=np.where(ending[:,None],
                    terminal_derivative[:,None]*navigation_tangent*navigation_interior[:,None],target_direction)
                terminal_s_jac=lambda c:np.einsum('ni,nij->nj',
                    navigation_tangent*navigation_interior[:,None],c['sens'][:,:2])
                terminal_position_margin=terminal_local-navigation_s
                penalty('route_terminal',terminal_position_margin,lambda c:-terminal_s_jac(c))
            # 线段内部等于原横向距离平方；折线端点保留完整位移，避免
            # 最近线段切换时仅取垂距造成目标跳变和错误的数值梯度。
            path_projectors=np.eye(2)[None,:,:]-tangent[:,:,None]*tangent[:,None,:]
            path_projectors[~path_interior]=np.eye(2)
            path_jac=lambda c:np.einsum('nij,njk->nik',path_projectors,c['sens'][:,:2]).reshape(2*n,2*n)
            residual('contour',path_residual.ravel(),path_jac,weights.get('contour',3),scales['position_m'])
            heading_error=wrap_angle(current[:,2]+beta-heading)
            heading_curvature=heading_derivative[:,None]
            heading_jac=lambda c:c['sens'][:,2]+bp[:,None]*steer_selector-heading_curvature*c['sj']
            residual('heading',heading_error,heading_jac,weights.get('heading',1),scales['heading_rad'])
            # 正常巡航不靠车道内蛇形寻找低代价；实际偏心时给出随行程收敛的恢复包络。
            initial_offset=float(np.linalg.norm(_path_projection(initial[None,:2],reference)[1]))
            recovery_length=float(p.get('tracking_recovery_distance_m',15.))
            base_error=float(p.get('tracking_max_error_m',.25))
            recovery=initial_offset*np.exp(-np.maximum(s-local_s0,0.)/max(recovery_length,1e-6))
            allowed=base_error+recovery
            recovery_slope=np.where(s>local_s0,-recovery/max(recovery_length,1e-6),0.)
            # 截面后的非中心目标不能强迫瞬时回中；此处只用实际车道可容纳范围，
            # 中心跟踪代价保留。真实通过后，下周期从实际状态重新生成基础归中参考。
            allowed=np.where(goal_tracking_mask,allowed,post_goal_corridor)
            recovery_slope=np.where(goal_tracking_mask,recovery_slope,0.)
            tracking_margin=allowed**2-np.sum(path_residual**2,axis=1)
            tracking_jac=lambda c:2*allowed[:,None]*recovery_slope[:,None]*c['sj']-2*np.einsum('ni,nij->nj',path_residual,c['sens'][:,:2])
            penalty('tracking_corridor',tracking_margin,tracking_jac)
            goal_evidence=None
            goal_constant=0.
            if goal_context is not None:
                goal_ref,goal_position,sqrt_weight=goal_context
                goal_residual,goal_jacobian,goal_evidence=first_arrival_residual(states[:,:2],potential_reference,
                    active_goal['s_goal_m']+potential_prefix_s,goal_position,sqrt_weight,
                    active_goal['last_s']+potential_prefix_s,
                    frenet_geometry=potential_geometry)
                goal_jac=lambda c:np.einsum('rni,nij->rj',goal_jacobian,c['all'][:,:2])
                # 一次事件势能0.5*w*||D(F(p_hit)-F(g))||²-C，不随节点数稀释。
                residual('goal_attraction',goal_residual,goal_jac,.5*self.goals.p['attraction_weight']*n)
                goal_constant=float(self.goals.p['potential_shift'])
                components['goal_attraction']-=goal_constant
                goal_evidence['potential_shift']=goal_constant
            residual('speed',current[:,3]-target,lambda c:c['sens'][:,3]-np.einsum('ni,nij->nj',target_direction,c['sens'][:,:2]),weights.get('speed',1),scales['speed_mps'])
            for index,key,scale,selector in [(0,'acceleration','acceleration_mps2',accel_selector),(1,'steering','steering_rad',steer_selector)]:
                residual('control',controls[:,index],selector,weights.get(key,.1 if index==0 else .2),scales[scale])
            residual('control',delta[:,0]/dt,difference@accel_selector/dt,weights.get('jerk',.2),scales['jerk_mps3'])
            residual('control',delta[:,1]/dt,difference@steer_selector/dt,weights.get('steering_rate',.2),scales['steering_rate_radps'])
            field_invalid=int((~field_query['valid']).sum())
            geometry_started=time.perf_counter()
            for actor,sample,other_vertices in targets:
                class_risk=risk_cfg.get('classes',{}).get(actor.kind)
                if class_risk is None:
                    raise ValueError('缺少势场对象类别参数：'+actor.kind)
                if actor.kind=='pedestrian':
                    gap,partial=circle_gap(current,sample[1:,:2],actor.radius or actor.width/2,scene.ego.length,scene.ego.width)
                else:
                    gap,partial=rectangle_gap(current,sample[1:],scene.ego.length,scene.ego.width,actor.length,actor.width)
                penalty('actor_clearance',gap-clearance+tolerance,
                        lambda c,partial=partial:np.einsum('ni,nij->nj',partial,c['sens']))
                minima.append(float((gap-clearance).min()))
                source_projection={key:value[1:] for key,value in source_projections[actor.id].items()}
                field=potential_geometry.repulsion_from_projection(field_query,source_projection,class_risk,
                    attention_gain=1.+increment*attention.get(actor.id,0.))
                # 公共势场在域外为NaN，不存在“域外低势能”。SQP的不可行迭代
                # 仅屏蔽这些残差槽；独立车道域约束和子步验收始终拒绝它们。
                value=np.where(field['valid'],field['residual'],0.)
                derivative=np.where(field['valid'][:,None],field['residual_jacobian'],0.)
                residual('actor_risk',value,
                    lambda c,derivative=derivative:np.einsum('ni,nij->nj',derivative,c['sens'][:,:2]),
                    weights.get('risk',2))
            line_minimum=np.inf
            line_states=dense_states[1:]
            line_polygons=shapely.polygons(body_vertices(line_states,scene.ego.length,scene.ego.width)) if physical_lines else None
            line_count=len(line_states)
            node_indices=np.arange(self.substeps-1,line_count,self.substeps)
            intermediate_indices=np.flatnonzero((np.arange(line_count)+1)%self.substeps)
            for item in physical_lines.values():
                distance=shapely.distance(line_polygons,item['geometry'])
                contact=distance<=1e-8;signed=distance.copy()
                contact_partial=None
                if np.any(contact):
                    center_nearest=shapely.get_coordinates(shapely.get_point(shapely.shortest_line(item['geometry'],shapely.points(line_states[:,:2])),0))
                    offset=line_states[:,:2]-center_nearest;center_distance=np.linalg.norm(offset,axis=1);normal_line=offset/np.maximum(center_distance[:,None],1e-10)
                    c,si=np.cos(line_states[:,2]),np.sin(line_states[:,2]);along=normal_line[:,0]*c+normal_line[:,1]*si;across=-normal_line[:,0]*si+normal_line[:,1]*c
                    support=scene.ego.length/2*np.abs(along)+scene.ego.width/2*np.abs(across)
                    signed[contact]=np.minimum(center_distance[contact]-support[contact],0.)
                    contact_partial=np.zeros((line_count,4));contact_partial[contact,:2]=normal_line[contact]
                    contact_partial[contact,2]=(-scene.ego.length/2*np.sign(along)*across+scene.ego.width/2*np.sign(across)*along)[contact]
                # 候选验收只需真实值；见证点导数留到该候选真正进入下一次QP。
                cache={}
                def line_partial(geometry=item['geometry'],distance=distance,contact=contact,
                                 contact_partial=contact_partial,cache=cache):
                    if 'partial' not in cache:
                        witnesses=shapely.shortest_line(geometry,line_polygons)
                        remote=shapely.get_coordinates(shapely.get_point(witnesses,0));own=shapely.get_coordinates(shapely.get_point(witnesses,1))
                        normal_line=(own-remote)/np.maximum(distance[:,None],1e-10)
                        partial=np.zeros((line_count,4));partial[:,:2]=normal_line
                        arm=own-line_states[:,:2];partial[:,2]=-normal_line[:,0]*arm[:,1]+normal_line[:,1]*arm[:,0]
                        if contact_partial is not None:partial[contact]=contact_partial[contact]
                        cache['partial']=partial
                    return cache['partial']
                # 保留控制节点的原代价；段间物理子步也进入同一硬约束，
                # 不能只在最终拒绝它们，却让QP看不到修正方向。
                penalty('noncrossable_line_clearance',signed[node_indices]-clearance,
                        lambda c,partial=line_partial:np.einsum('ni,nij->nj',partial()[node_indices],c['sens']))
                if len(intermediate_indices):
                    rows.append(lambda c,partial=line_partial:np.einsum(
                        'ni,nij->nj',partial()[intermediate_indices],c['dense'][intermediate_indices]))
                    lower.append(clearance-signed[intermediate_indices])
                    upper.append(np.full(len(intermediate_indices),np.inf))
                line_minimum=min(line_minimum,float((signed-clearance).min()))
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
            # 时域末端仍保留停车空间。车体前缘而非中心到障碍物的距离。
            terminal_margins=[]
            if terminal_local is not None:
                terminal_gap=float(terminal_local-navigation_s[-1]-current[-1,3]**2/(2*braking))
                penalty('route_terminal_stopping',np.array([terminal_gap]),
                    lambda c:(-terminal_s_jac(c)[-1]-current[-1,3]/braking*c['sens'][-1,3])[None,:])
                metrics_terminal_position=float(terminal_position_margin.min())
                metrics_terminal_stopping=terminal_gap
            if np.isfinite(lead_rear[-1]):
                gap_end=lead_rear[-1]-s[-1]-scene.ego.length/2-follow_gap
                stopping_difference=max(0.,(current[-1,3]**2-lead_speed[-1]**2)/(2*terminal_braking))
                terminal_gap=gap_end-stopping_difference
                def terminal_follow_jac(c):
                    dv=current[-1,3]/terminal_braking if current[-1,3]>lead_speed[-1] else 0.
                    return (-c['sj'][-1]-dv*c['sens'][-1,3])[None,:]
                penalty('terminal_stopping',np.array([terminal_gap]),terminal_follow_jac)
                terminal_margins.append(terminal_gap)
            if stop_local is not None:
                terminal_gap=float(gap[-1]-current[-1,3]**2/(2*terminal_braking))
                def terminal_line_jac(c):
                    return (-front_partial[-1]@c['sens'][-1]-current[-1,3]/terminal_braking*c['sens'][-1,3])[None,:]
                penalty('terminal_stopping',np.array([terminal_gap]),terminal_line_jac)
                terminal_margins.append(terminal_gap)
            evaluations['geometry_seconds']+=time.perf_counter()-geometry_started
            # 进度属于导航，横向/航向跟踪属于局部参考，坐标用途不混用。
            progress=-float(weights.get('progress',1))*(navigation_s[-1]-(ego_s-start_arc))/scales['progress_m'];components['progress']=progress
            residual_vector=np.concatenate(residuals)
            objective=float(residual_vector@residual_vector+progress-goal_constant)
            for key in ('actor_risk','line_risk','goal_attraction','contour','heading'):components.setdefault(key,0.)
            metrics={name:value for name,value in zip(labels,minima)}
            metrics['tracking_corridor']=float(tracking_margin.min())
            if terminal_local is not None:
                metrics['route_terminal']=metrics_terminal_position
                metrics['route_terminal_stopping']=metrics_terminal_stopping
            metrics['potential_domain']=float(domain_margin.min())
            if terminal_margins:metrics['terminal_stopping']=float(min(terminal_margins))
            control_margin=float(min(np.min(controls[:,0]-min_a),np.min(max_a-controls[:,0]),
                                     np.min(max_delta-np.abs(controls[:,1]))))
            metrics['control_box']=control_margin
            valid=(np.isfinite(states).all() and np.isfinite(controls).all() and minima[1]>=-1e-8
                   and control_margin>=-1e-8 and np.isfinite(objective))
            dense=dense_check(controls,dense_states)
            safety_feasible=(all(initial_safety.values()) and all(value>=-(speed_tolerance if key=='speed_limit' else tolerance) for key,value in metrics.items()
                             if key!='tracking_corridor' or not is_brake)
                             and dense['valid']
                             and bool(domain_inside.all()) and field_invalid==0
                             and (goal_evidence is None or goal_evidence['potential_valid'])
                             and bool(np.all(shapely.covers(accepted_road,polygons))))
            result=dict(states=states,controls=controls.copy(),objective=objective,components=components,
                lower=np.concatenate(lower),upper=np.concatenate(upper),minima=metrics,valid=bool(valid),polygons=polygons,
                following_headway_minimum=(float(np.min(headway_margin)) if np.isfinite(headway_margin).any() else None),
                safety_feasible=bool(safety_feasible),dense_validation=dense,is_brake=bool(is_brake),
                violation=max(0.,-min(metrics.values()),-control_margin),goal_evidence=goal_evidence,
                potential_invalid_samples=field_invalid)
            def linearize():
                linearization_started=time.perf_counter();evaluations['linearizations']+=1
                sensitivity=self.dynamics.sensitivity(controls,initial)
                sens=sensitivity[1:]
                sj=np.einsum('ni,nij->nj',s_tangent,sens[:,:2])
                sj[~s_interior]=0.
                context={'all':sensitivity,'sens':sens,'sj':sj}
                if physical_lines and len(intermediate_indices):
                    context['dense']=self.dynamics.dense_sensitivity(controls,initial)[1:]
                jacobian=np.vstack([(jac(context) if callable(jac) else np.asarray(jac))*factor for jac,factor in jacobians])
                navigation_jac=navigation_tangent[-1]@sens[-1,:2] if navigation_interior[-1] else np.zeros(2*n)
                gradient_extra=-float(weights.get('progress',1))*navigation_jac/scales['progress_m']
                value=dict(hessian=2*jacobian.T@jacobian+np.eye(2*n)*float(p.get('sqp_regularization',1e-4)),
                           gradient=2*jacobian.T@residual_vector+gradient_extra,
                           matrix=np.vstack([row(context) if callable(row) else row for row in rows]))
                evaluations['linearization_seconds']+=time.perf_counter()-linearization_started
                return value
            result['linearize']=linearize
            evaluations['value_seconds']+=time.perf_counter()-value_started
            return result

        solver_started=time.perf_counter();budget=float(p.get('cold_start_budget_s',.5) if cold_start else p.get('solver_wall_budget_s',.1));deadline=started+budget
        def preference(value):
            # 正常安全解永远优于低成本不安全解；不可行名义点只用于寻找可行改进。
            return (not value['safety_feasible'],0. if value['safety_feasible'] else value['violation'],value['objective'])
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
            chosen=min(feasible,key=lambda item:preference(item[1]))
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
        restoration={'eligible':False,'source':None}
        if (not _brake_only and active_goal is None and seed_connections and
                domain_evidence.get('current_lane_association') and initial_reselection['evaluated']):
            # 制动是可执行后备，不必同时成为SQP的线性化点。零速制动点
            # 的横纵耦合灵敏度会退化，难以修正连接上几厘米的跟踪偏差。
            # 仅用已验证导航连接的有限前进初值做原SQP可行性恢复；
            # accepted始终独立保留验收后的后备，未验收初值绝不执行。
            forward=[item for item in alternatives if item[0]=='navigation_seed' and
                     item[1]['valid'] and not item[1]['safety_feasible']]
            if forward and accepted['safety_feasible']:
                nominal_source,nominal=forward[0]
                controls=nominal['controls']
                nominal['initialization_source']=nominal_source
                nominal['candidate_source']='initialization_restoration_only'
                restoration.update(eligible=True,source=nominal_source)
        trust=np.tile([float(p.get('sqp_trust_acceleration_mps2',1.5)),float(p.get('sqp_trust_steering_rad',.15))],n)
        last_iteration_seconds=0.;iteration_history=[];termination='iteration_limit'
        recovery_speed=float(p.get('warm_start_recovery_speed_mps',.3))
        recovery={'eligible':bool(warm and stop_local is None and speed<recovery_speed),
                  'speed_threshold_mps':recovery_speed,'evaluated':False,'selected':False}
        for iteration in range(0 if _brake_only else int(p.get('solver_max_iterations',4))):
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
                if exact and preference(candidate)<preference(accepted):accepted=candidate
                candidates.append(candidate)
                evaluation_seconds=time.perf_counter()-evaluation_started
                if exact and preference(candidate)<=preference(nominal):break
            if not candidates:
                termination='control_projection_infeasible';break
            # 恢复阶段允许继续线性化仍有微小违约的改进点；可执行解
            # 单独放在accepted中。否则每步都会又被静止后备拉回零速。
            pool=([nominal] if restoration['eligible'] and not nominal['safety_feasible'] else [accepted])
            next_nominal=min([*pool,*[value for value in candidates if value['valid']]],key=preference)
            improvement=abs(next_nominal['objective']-nominal['objective'])
            step_norm=float(np.linalg.norm(step,np.inf))
            iteration_history.append({'iteration':iteration+1,'nominal_cost':nominal['objective'],
                'candidate_cost':next_nominal['objective'],'absolute_cost_change':improvement,
                'qp_step_inf':step_norm,'gradient_inf':float(np.linalg.norm(nominal['gradient'],np.inf)),
                'candidate_valid':next_nominal['valid'],'candidate_violation':next_nominal['violation'],
                'candidate_safety_feasible':next_nominal['safety_feasible'],
                'candidate_dense_validation':next_nominal['dense_validation'],
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
                    if exact and preference(alternative)<preference(accepted):
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
        emergency_candidates=[]
        if _brake_only or not accepted['safety_feasible']:
            # 与语义目标无关的本车道制动，重新积分本帧，不复用未经验收的旧控制。
            brake_reference=global_reference
            lane_table={lane.id:lane for lane in scene.lanes}
            own=self.references._lane(scene,lane_table)
            if own is not None:
                # 交叉口几何最近标签可能是直行连接段。制动仍沿已知导航
                # 相容中心延伸，不能突然跟随错误分支，也不继承VLM目标。
                brake_world,_=terminal_covered_reference(scene,
                    self.references._continuation(scene,own,lane_table),forward_m=support_front)
                brake_reference=clean_polyline(world_to_ego(brake_world,origin,angle))
            brake_arc=polyline_lengths(brake_reference)
            brake_tangents=np.unwrap(np.arctan2(np.gradient(brake_reference[:,1]),np.gradient(brake_reference[:,0])))
            brake_curvature=np.gradient(brake_tangents,brake_arc+np.arange(len(brake_arc))*1e-9)
            def brake_controls(deceleration,follow_lane,relax_initial_jerk=False):
                trajectory=[];state=initial.copy();last=previous_control.copy();release_conflicts=[]
                for step_index in range(n):
                    s,offset,_=project_polyline(state[:2],brake_reference,heading=state[2])
                    # 弯中制动仍需曲率前馈和质心侧偏补偿；仅反馈航向误差会
                    # 在已对齐切向时把转角拉回零，使制动轨迹驶向弯道外缘。
                    kappa=float(np.interp(s,brake_arc,brake_curvature))
                    feed=np.arctan(wheelbase*kappa/np.sqrt(max(.01,1-(rear*kappa)**2)))
                    target_heading=float(np.interp(s,brake_arc,brake_tangents))-np.arctan(axle_ratio*np.tan(feed))
                    desired_steer=(feed+1.2*float(wrap_angle(target_heading-state[2]))-.3*np.arctan(offset/max(state[3],2.))) if follow_lane else 0.
                    acceleration=-min(deceleration,1.8*max(0.,state[3]))
                    release_limit=brake_acceleration_limit(max(0.,state[3]),dt,max_jerk)
                    lower=max(min_a,last[0]-max_jerk*dt,-release_limit)
                    upper=min(max_a,last[0]+max_jerk*dt)
                    if relax_initial_jerk and step_index==0:
                        # Emergency-only exception to longitudinal comfort.
                        # All following steps keep the original release/jerk
                        # envelope, and steering continuity is never relaxed.
                        lower=max(min_a,-release_limit);upper=max_a
                    conflict=lower>upper+1e-10
                    # 从可行初态提前释放制动，不等到v/dt末步截断才破坏jerk。
                    # 已无共同可行区间时优先非倒退和控制箱，明确记录初态冲突。
                    if conflict:
                        selected_acceleration=float(np.clip(upper,max(min_a,-release_limit),max_a))
                        release_conflicts.append({'step':step_index,'speed_mps':float(state[3]),
                            'previous_acceleration_mps2':float(last[0]),'release_lower_mps2':-release_limit,
                            'jerk_upper_mps2':float(last[0]+max_jerk*dt),
                            'selected_acceleration_mps2':selected_acceleration,
                            'reason':'当前速度与前次加速度不能同时满足未来非倒退和jerk释放包络'})
                    else:
                        selected_acceleration=float(np.clip(acceleration,lower,max(lower,upper)))
                    command=np.array([selected_acceleration,
                        np.clip(desired_steer,last[1]-max_rate*dt,last[1]+max_rate*dt)])
                    command=np.clip(command,[min_a,-max_delta],[max_a,max_delta])
                    trajectory.append(command);state=propagate(state,command,scene.ego.length,dt,self.substeps,wheelbase,rear);last=command
                return np.asarray(trajectory),release_conflicts
            brake_specs=[('lane_service_brake',min(braking,-min_a),True),
                         ('lane_maximum_brake',-min_a,True),('heading_maximum_brake',-min_a,False)]
            for label,deceleration,follow_lane,relaxed in [(*spec,False) for spec in brake_specs]+[
                    (spec[0]+'_initial_jerk_relaxed',spec[1],spec[2],True) for spec in brake_specs]:
                if relaxed and any(value['safety_feasible'] and value['emergency_braking_eligible']
                                   for _,value in emergency_candidates):
                    continue
                braking_controls,release_conflicts=brake_controls(deceleration,follow_lane,relaxed)
                value=terms(braking_controls,is_brake=True)
                value['braking_release_conflicts']=release_conflicts
                if release_conflicts:value['safety_feasible']=False
                if not value['valid']:continue
                value['emergency_braking_eligible']=bool(np.all(braking_controls[:,0]<=1e-8))
                transition=float(abs(braking_controls[0,0]-previous_control[0]))
                value['emergency_jerk_relaxation']={
                    'active':bool(relaxed and transition>max_jerk*dt+1e-10),
                    'scope':'first_longitudinal_control_transition_only',
                    'previous_acceleration_mps2':float(previous_control[0]),
                    'first_acceleration_mps2':float(braking_controls[0,0]),
                    'normal_delta_limit_mps2':float(max_jerk*dt),
                    'actual_delta_mps2':transition,
                    'later_longitudinal_and_all_steering_limits_retained':True}
                # Even when every physical check passes, a relaxed comfort
                # transition must never become a certified normal solution.
                if value['emergency_jerk_relaxation']['active']:
                    value['safety_feasible']=False
                value['initialization_source']=label;value['candidate_source']='current_state_braking'
                evidence=value['dense_validation'];contacts=evidence['predicted_contacts']
                _,lateral,route_heading=project_polyline(value['states'][:,:2],brake_reference)
                # 用实际动力学的速度积分比较制动行程；不能用沿导航投影，
                # 否则拉直偏离弯道反而会因“少前进”获得更低的停止线亏量。
                travel=float(dt*np.sum(value['states'][:-1,3]+.5*(1.-1./self.substeps)*dt*braking_controls[:,0]))
                beta=np.r_[np.arctan(axle_ratio*np.tan(braking_controls[:,1])),np.arctan(axle_ratio*np.tan(braking_controls[-1,1]))]
                braking_metrics={'distance_m':travel,'terminal_speed_mps':float(value['states'][-1,3]),
                    'navigation_lateral_rms_m':float(np.sqrt(np.mean(lateral**2))),
                    'navigation_direction_rms_rad':float(np.sqrt(np.mean(wrap_angle(route_heading-value['states'][:,2]-beta)**2))),
                    'distance_comparison_resolution_m':.001,'speed_comparison_resolution_mps':.001}
                value['braking_risk_metrics']=braking_metrics
                # 仅有限制动集合内排序：物理危险优先，再比较停车能力，
                # 同级才保持导航；不让毫米以下积分噪声压过转向连续性。
                risk_order=(not value['safety_feasible'],len(contacts)>0,
                    max([row['closing_speed_mps'] for row in contacts],default=0.),
                    -min([row['time_s'] for row in contacts],default=n*dt),
                    evidence['offroad_samples']+evidence['line_violation_samples'],evidence['stop_line_passed'],
                    round(travel,3),round(braking_metrics['terminal_speed_mps'],3),
                    braking_metrics['navigation_lateral_rms_m'],braking_metrics['navigation_direction_rms_rad'],value['violation'])
                emergency_candidates.append((risk_order,value))
            if not emergency_candidates:
                raise RuntimeError('当前状态不能形成有限有界制动控制，拒绝伪造有效轨迹')
            eligible=[item for item in emergency_candidates if item[1]['emergency_braking_eligible']]
            if not eligible:
                raise RuntimeError('Emergency braking candidates contain positive acceleration; refusing hidden throttle')
            accepted=min(eligible,key=lambda item:item[0])[1]
            termination='validated_brake' if accepted['safety_feasible'] else 'emergency_braking_no_certified_solution'
        execution_mode=('safe_brake' if accepted.get('is_brake') else 'normal') if accepted['safety_feasible'] else 'emergency_unverified'
        elapsed=time.perf_counter()-started
        diagnostics={'solver_seconds':time.perf_counter()-solver_started,'planning_seconds':elapsed,'solver_status':statuses,
            'planning_object_selection':object_selection,'execution_mode':execution_mode,
            'potential_field':{'version':'lane-frenet-center-ellipse-1.0','coordinates':'lane_frenet',
                'kernel':'(1-rho)^3*(1+3*rho+6*rho^2) for rho<1; zero otherwise',
                'sampling_point':'ego_center','source_shape':'lane_aligned_center_ellipse',
                'physical_dimensions_used_for_potential':False,'physical_collision_constraints_retained':True,
                'line_treatment':'full_body_geometric_constraints; no_line_APF',
                'domain':domain_evidence,'invalid_samples':accepted['potential_invalid_samples'],
                'geometry':potential_geometry.diagnostics,'source_projection_uncovered_samples':source_coverage,
                'reference_coverage':potential_reference_evidence,
                'source_projection_policy':'outside_confirmed_reference_no_APF;physical_collision_constraints_retained',
                'classes':risk_cfg['classes'],'attention_increment':increment,
                'source_ids':sorted(actor.id for actor,_,_ in targets),
                'attention_gain_by_id':{actor.id:1.+increment*attention.get(actor.id,0.) for actor,_,_ in targets},
                'source_anchors':source_anchors,
                'collision_constraint_ids':sorted(actor.id for actor,_,_ in targets),
                'outer_risk_weight':float(weights.get('risk',2)),
                'goal_event':'legacy_frozen_polyline_first_crossing; energy_on_same_smooth_Frenet_projection'},
            'emergency':execution_mode=='emergency_unverified',
            'braking_release_conflicts':accepted.get('braking_release_conflicts',[]),
            'emergency_jerk_relaxation':accepted.get('emergency_jerk_relaxation',{'active':False}),
            'task_terminal':dict(enabled=terminal_enabled,reference_point='ego_center',
                task_endpoint=scene.route[-1].tolist(),remaining_m=float(full_arc[-1]-ego_s),
                navigation_window_start_s_m=float(start_arc),terminal_chart_s_m=terminal_local,
                terminal_anchor_in_window=bool(len(terminal_indices)),
                braking_mps2=braking,official_arrival_unchanged=True,
                navigation_support=terminal_support,control_support=control_support,seed_support=seed_support),
            'braking_risk_metrics':accepted.get('braking_risk_metrics'),
            'braking_risk_order_fields':['not_safety_feasible','predicted_contact','maximum_contact_closing_speed_mps',
                'negative_first_contact_time_s','offroad_plus_line_violation_samples','stop_line_passed',
                'speed_integral_distance_m_rounded_3','terminal_speed_mps_rounded_3',
                'navigation_lateral_rms_m','navigation_direction_rms_rad','maximum_constraint_deficit'],
            'dense_validation':accepted['dense_validation'],
            'following':{'lead_ids':lead_ids,'minimum_gap_m':follow_gap,'time_headway_s':headway,'terminal_braking_mps2':terminal_braking,
                'version':'bumper-gap-time-headway-1.0','headway_is_soft_target':True,
                'initial_net_gap_m':float(lead_rear[0]-local_s0-scene.ego.length/2) if np.isfinite(lead_rear[0]) else None,
                'desired_initial_net_gap_m':follow_gap+headway*speed if np.isfinite(lead_rear[0]) else None,
                'minimum_predicted_headway_margin_m':accepted['following_headway_minimum'],
                'eligibility_rule':'moving_longitudinal_velocity_gt_abs_lateral_velocity_or_stationary_corridor_blocker',
                'stationary_speed_threshold_mps':stationary_speed,'candidate_classification':lead_classification,
                'entity_collision_target_ids':[actor.id for actor,_,_ in targets]},
            'tracking_policy':{'before_goal_or_no_goal_max_error_m':float(p.get('tracking_max_error_m',.25)),
                'initial_offset_recovery_distance_m':float(p.get('tracking_recovery_distance_m',15.)),
                'after_predicted_goal_crossing':'实际同向车道内车体可容纳范围，保留基础中心/方向代价，不保留目标吸引',
                'after_actual_goal_pass':'GoalManager释放；下一周期从实际状态生成基础参考'},
            'emergency_candidates':[{'source':value['initialization_source'],'safety_feasible':value['safety_feasible'],
                'risk_order':list(order),'dense_validation':value['dense_validation'],
                'braking_risk_metrics':value.get('braking_risk_metrics'),
                'emergency_braking_eligible':value.get('emergency_braking_eligible'),
                'emergency_jerk_relaxation':value.get('emergency_jerk_relaxation'),
                'braking_release_conflicts':value.get('braking_release_conflicts',[])} for order,value in emergency_candidates],
            'vehicle_dynamics':{'reference_point':'geometric_center','wheelbase_m':wheelbase,
                'rear_axle_to_center_m':rear,'geometry_source':'platform' if scene.ego.wheelbase_m is not None else 'highway_legacy'},
            'solver_backend':'casadi_osqp_realtime_sqp','symbolic_build_seconds':build_seconds,'qp_seconds':qp_seconds,'sqp_iterations':iterations,
            'optimizer_termination':termination,'iteration_history':iteration_history,
            'warm_start_recovery':recovery,
            'initial_guess_source':initial_guess_source,'infeasible_initial_reselection':initial_reselection,
            'navigation_initialization_restoration':restoration,
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
            'rule_evidence':self.rule_evidence,'rule_observation_scope':self.rule_observation_scope,
            'initial_constraints':initial_diagnostics,
            'map_rejections':[{'actor_id':identity,'reason':info.get('reason'),
                               'trajectory_source':info.get('trajectory_source'),'kinematic_valid':info.get('kinematic_valid')}
                              for identity,info in predictions.diagnostics.items()
                              if identity in predictions.actors and isinstance(info,dict) and info.get('valid') is False],
            'last_optimizer_constraints':None if last is None else {'minimum_by_group':last['minima'],'cost':last['objective']}}
        states=accepted['states'].copy();controls=accepted['controls'].copy()
        states[:,:2]=ego_to_world(states[:,:2],origin,angle);states[:,2]+=angle;self.previous=controls.copy();self.previous_time=scene.time
        self.previous_reference_revision=selection.revision
        diagnostics.update(objective=accepted['objective'],cost_terms=accepted['components'],candidate_source=accepted['candidate_source'],
            accepted_initialization_source=accepted['initialization_source'],
            minimum_constraint=min(accepted['minima'].values()),constraint_minima=accepted['minima'],semantic_enabled=answer is not None,
            active_goal=active_goal is not None,goal_evaluation=accepted.get('goal_evidence'),
            constraint_policy='safety_first_tracking_v2',trajectory_available=True,
            safety_feasible=accepted['safety_feasible'],initial_safety_checks=initial_safety,
            speed_numerical_tolerance_mps=speed_tolerance,
            hard_constraint_groups=['dynamics','acceleration','steering','nonnegative_speed','jerk','steering_rate',
                'road','potential_domain','noncrossable_line','actor_clearance','speed_limit','lateral_acceleration','stop_line','terminal_stopping','tracking_corridor']+
                (['route_terminal','route_terminal_stopping'] if terminal_enabled else []),
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
