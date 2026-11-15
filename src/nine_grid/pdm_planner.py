"""PDM-Closed流程的EP-Driver适配：几何×IDM提案、LQR跟踪、实车体评分。

独立实现，不调用MPC.solve/SQP/QP；不是tuplan_garage原版结果复现。
出处、单位及与原版的差异见本目录PDM_ADAPTATION.md。
"""
from __future__ import annotations
from copy import deepcopy
from time import perf_counter
import numpy as np
from shapely.geometry import LineString
from shapely import prepare

from nine_grid.contracts import PlanResult
from nine_grid.geometry import (clean_polyline, polyline_lengths, project_polyline,
    sample_polyline, actor_shape, signed_body_distance, rectangle_corners, wrap_angle)
from nine_grid.local_reference import LocalReferenceManager
from nine_grid.goals import GoalManager, goal_seed_path, first_arrival_residual
from nine_grid.planning_objects import planning_view
from nine_grid.quality import evaluate, _progress, _route_projection
from nine_grid.realtime_solver import bicycle_geometry
from .pdm_primitives import propagate, _validate_prediction_input, public_rules
from .stop_control import brake_acceleration_limit, stopping_speed_required


VERSION = 'ep-pdm-closed-adaptation-1.0'
DEFAULTS = {
    'speed_fractions': [.65, .85, 1.], 'idm_headway_s': 1.2,
    'idm_min_gap_m': 2., 'idm_accel_mps2': 2., 'idm_comfort_braking_mps2': 3.,
    'lqr_horizon_steps': 10, 'q_longitudinal': 10., 'r_longitudinal': 1.,
    'q_lateral': [1., 10., .1], 'r_lateral': 1., 'lqr_min_speed_mps': .5,
    'reference_sample_m': .5, 'proposal_risk_weight': .05, 'proposal_goal_weight': .15,
    'score_progress_weight': 5., 'score_ttc_weight': 5., 'score_comfort_weight': 2.,
    'emergency_braking_fractions': [.5, 1.],
}


def lqr_gain(a, b, q, r, horizon):
    """有限时域Riccati递推：小矩阵线性代数，不调用轨迹优化求解器。"""
    p = q.copy()
    for _ in range(horizon):
        k = np.linalg.solve(r+b.T@p@b, b.T@p@a)
        p = q+a.T@p@(a-b@k)
        p = .5*(p+p.T)
    return k


def idm_acceleration(speed, desired_speed, gap, lead_speed, settings):
    a=float(settings['idm_accel_mps2']); b=float(settings['idm_comfort_braking_mps2'])
    if desired_speed <= 1e-6:
        return -b if speed > 1e-6 else 0.
    interaction=0.
    if np.isfinite(gap):
        wanted=settings['idm_min_gap_m']+max(0., speed*settings['idm_headway_s']
            +speed*(speed-lead_speed)/(2*np.sqrt(a*b)))
        interaction=(wanted/max(float(gap),.05))**2
    return float(a*(1-(max(0.,speed)/desired_speed)**4-interaction))


class Planner:
    """与运行器一致的Planner接口；控制为[a,delta]，状态为几何中心[x,y,psi,v]。"""
    _rules = public_rules

    def __init__(self, config):
        self.config=deepcopy(config); self.p=self.config.get('mpc',{})
        self.settings={**DEFAULTS,**self.config.get('pdm',{})}
        if set(self.settings)-set(DEFAULTS):
            raise ValueError('pdm包含未实现参数')
        for key,value in self.settings.items():
            array=np.asarray(value,dtype=float)
            if not np.isfinite(array).all() or np.any(array<=0):
                raise ValueError('pdm参数必须为有限正值：'+key)
        if max(self.settings['speed_fractions'])>1:
            raise ValueError('PDM速度策略不能超过规则期望速度')
        if (type(self.settings['lqr_horizon_steps']) is not int or
                len(self.settings['q_lateral'])!=3):
            raise ValueError('LQR时域为整数，横向Q须有三个状态权重')
        self.n=int(self.p.get('horizon_steps',30)); self.dt=float(self.p.get('step_s',.1))
        self.substeps=int(self.p.get('physics_substeps',5))
        if self.n!=30 or abs(self.dt-.1)>1e-9 or self.substeps!=5:
            raise ValueError('比较后端保持3s时域、0.1s控制和0.02s物理积分')
        self.references=LocalReferenceManager(config); self.goals=GoalManager(config)
        self.previous=None; self.previous_time=None; self.last_diagnostics={}; self.rule_evidence=[]

    def export_reference_state(self):
        return {'local_reference':self.references.export_state(),'goals':self.goals.export_state()}

    def restore_reference_state(self,state):
        self.references.import_state(state['local_reference']);self.goals.import_state(state['goals'])

    def clear_semantic_guidance(self):
        self.goals.clear();self.references.reset_intent();self.previous=None

    def _geometries(self,scene,predictions,goal):
        lanes={lane.id:lane for lane in scene.lanes}
        lane=self.references._lane(scene,lanes)
        before=self.references.export_state()
        if goal is not None:
            path=goal_seed_path(scene,goal)
            return [('local_goal',path,before,{'phase':'goal_active','selected_source':'local_goal'})]
        selected=self.references.select(scene,predictions)
        proposed=[(selected.diagnostics.get('selected_source','navigation'),selected.world_path,
                   self.references.export_state(),selected.diagnostics)]
        # 换道已进入保持阶段时，不额外生成突然返回原中心的提案。
        if lane is not None and before['phase'] not in ('initiate','hold'):
            keep=self.references._path(scene,lane,lane,lanes)
            if not (keep.shape==selected.world_path.shape and np.allclose(keep,selected.world_path)):
                keep_state=deepcopy(before)
                keep_state.update(phase='keep',path=None,source_lane_id=lane.id,target_lane_id=lane.id,
                                  selected_source='navigation')
                proposed.append(('keep',keep,keep_state,{'phase':'keep','selected_source':'navigation'}))
        return proposed

    def _path_data(self,scene,path):
        path=clean_polyline(path);arc=polyline_lengths(path)
        s=project_polyline(scene.ego.position,path,heading=scene.ego.heading)[0]
        end=min(arc[-1],s+max(100.,np.linalg.norm(scene.ego.velocity)*4))
        start=max(0.,s-10.)
        path=sample_polyline(path,np.linspace(start,end,max(3,int((end-start)/self.settings['reference_sample_m'])+1)))
        arc=polyline_lengths(path)
        chi=np.unwrap(np.arctan2(np.gradient(path[:,1]),np.gradient(path[:,0])))
        curvature=np.gradient(chi,arc)
        wheelbase,rear=bicycle_geometry(scene.ego.length,scene.ego.wheelbase_m,scene.ego.rear_axle_to_center_m)
        # 后轴无侧滑条件：dpsi/ds_center=sin(chi-psi)/lr。
        # 常曲率才有beta=asin(lr*kappa)；五次路径的变曲率不能沿用该近似。
        # RK4沿弧长积分，构造后轴切向与车体朝向一致的参考。
        psi=np.empty(len(path));psi[0]=chi[0]
        for index,step in enumerate(np.diff(arc)):
            start=psi[index];c0=chi[index];c1=chi[index+1];cm=.5*(c0+c1)
            k1=np.sin(c0-start)/rear
            k2=np.sin(cm-(start+.5*step*k1))/rear
            k3=np.sin(cm-(start+.5*step*k2))/rear
            k4=np.sin(c1-(start+step*k3))/rear
            psi[index+1]=start+step*(k1+2*k2+2*k3+k4)/6
        beta=wrap_angle(chi-psi)
        if np.any(abs(beta)>=np.pi/2-.05):
            raise ValueError('参考路径发生无法以向前行驶跟踪的方向突变')
        rear_path=path-rear*np.column_stack((np.cos(psi),np.sin(psi)))
        rear_path=clean_polyline(rear_path)
        if len(rear_path)!=len(path):
            raise ValueError('后轴参考退化，不能静默改变参考采样')
        rear_arc=polyline_lengths(rear_path)
        rear_curvature=np.tan(beta)/rear
        return {'path':path,'arc':arc,'chi':chi,'psi':psi,'curvature':curvature,
                'rear_path':rear_path,'rear_arc':rear_arc,'rear_curvature':rear_curvature,
                'wheelbase':wheelbase,'rear':rear}

    def _prediction_samples(self,predictions):
        result={}
        for identity,trajectory in predictions.trajectories.items():
            values=np.asarray(trajectory,float)
            result[identity]=(values,np.unwrap(values[:,2]))
        return result

    @staticmethod
    def _object_at(predictions,samples,identity,t):
        values,headings=samples[identity]
        if t>predictions.times[-1]+1e-7:
            raise ValueError('PDM候选检查不能重复预测尾端掩盖时域不足')
        return np.array([np.interp(t,predictions.times,values[:,0]),
                         np.interp(t,predictions.times,values[:,1]),
                         np.interp(t,predictions.times,headings),
                         np.interp(t,predictions.times,values[:,3])])

    def _idm_reference(self,scene,predictions,data,fraction,stop_s,samples):
        path=data['path'];s=project_polyline(scene.ego.position,path,heading=scene.ego.heading)[0]
        speed=float(np.linalg.norm(scene.ego.velocity));progress=[s];speeds=[speed];accels=[]
        desired=min(float(scene.rules.get('desired_speed',25.)),float(scene.rules.get('speed_limit',30.)),
                    float(self.p.get('maximum_speed_mps',33.3333)))*fraction
        amax=float(self.p.get('acceleration_max_mps2',3.));amin=float(self.p.get('acceleration_min_mps2',-6.))
        lateral=float(self.config.get('rules',{}).get('curve_reference_lateral_acceleration_mps2',3.5))
        for k in range(self.n):
            center=sample_polyline(path,s)
            heading=float(np.interp(s,data['arc'],data['psi']))
            gap,lead_speed=float('inf'),0.
            for identity,actor in predictions.actors.items():
                other=self._object_at(predictions,samples,identity,k*self.dt)
                os,od,oh=project_polyline(other[:2],path,previous_s=s,window=max(50.,speed*4),heading=other[2])
                if os<s-scene.ego.length/2:
                    continue
                angle=float(wrap_angle(other[2]-oh))
                half_l=actor.radius if actor.kind=='pedestrian' else abs(np.cos(angle))*actor.length/2+abs(np.sin(angle))*actor.width/2
                half_w=actor.radius if actor.kind=='pedestrian' else abs(np.sin(angle))*actor.length/2+abs(np.cos(angle))*actor.width/2
                if abs(od)>scene.ego.width/2+half_w+.05:
                    continue
                candidate=os-s-half_l-scene.ego.length/2
                if candidate<gap:
                    gap=candidate;lead_speed=max(0.,other[3]*np.cos(angle))
            if stop_s is not None:
                route_s=project_polyline(center,scene.route,heading=heading)[0]
                rule_gap=stop_s-route_s-scene.ego.length/2
                if rule_gap<gap:gap,lead_speed=rule_gap,0.
            curvature=abs(float(np.interp(s,data['arc'],data['curvature'])))
            target=min(desired,np.sqrt(lateral/max(curvature,.001)))
            # 临近折线尾端减速；目标点本身不是路径尾端或停车目标。
            target=min(target,np.sqrt(2*self.settings['idm_comfort_braking_mps2']*max(0.,data['arc'][-1]-s-scene.ego.length/2)))
            a=float(np.clip(idm_acceleration(speed,target,gap,lead_speed,self.settings),max(amin,-speed/self.dt),amax))
            s=min(data['arc'][-1],s+speed*self.dt+.5*a*self.dt**2)
            speed=max(0.,speed+a*self.dt)
            progress.append(s);speeds.append(speed);accels.append(a)
        return np.asarray(progress),np.asarray(speeds),np.asarray(accels)

    def _track(self,scene,data,progress,speeds,accels,braking=None):
        """小角度后轴LQR产生加速度和转角速度；实际输出按同一中心模型积分。"""
        state=np.array([*scene.ego.position,scene.ego.heading,np.linalg.norm(scene.ego.velocity)],float)
        states=[state.copy()];dense=[state.copy()];controls=[]
        previous=np.asarray(scene.diagnostics.get('previous_control',[0.,0.]),float).copy()
        amin=float(self.p.get('acceleration_min_mps2',-6.));amax=float(self.p.get('acceleration_max_mps2',3.))
        max_delta=float(self.p.get('steering_abs_max_rad',.6));max_rate=float(self.p.get('steering_rate_abs_max_radps',.8))
        max_jerk=float(self.p.get('jerk_abs_max_mps3',8.));horizon=int(self.settings['lqr_horizon_steps'])
        q=np.diag(self.settings['q_lateral']);r=np.array([[self.settings['r_lateral']]])
        kv=float(lqr_gain(np.ones((1,1)),np.array([[self.dt]]),
            np.array([[self.settings['q_longitudinal']]]),np.array([[self.settings['r_longitudinal']]]),horizon)[0,0])
        previous_rear=None
        for k in range(self.n):
            rear_position=state[:2]-data['rear']*np.array([np.cos(state[2]),np.sin(state[2])])
            station,lateral,heading=project_polyline(rear_position,data['rear_path'],previous_s=previous_rear,heading=state[2])
            previous_rear=station
            reference_heading=float(np.interp(station,data['rear_arc'],data['psi']))
            curvature=float(np.interp(station,data['rear_arc'],data['rear_curvature']))
            delta_ff=float(np.arctan(data['wheelbase']*curvature))
            v=max(self.settings['lqr_min_speed_mps'],state[3])
            a=np.array([[1.,v*self.dt,0.],[0.,1.,v*self.dt/data['wheelbase']],[0.,0.,1.]])
            b=np.array([[0.],[0.],[self.dt]])
            gain=lqr_gain(a,b,q,r,horizon)
            error=np.array([lateral,float(wrap_angle(state[2]-reference_heading)),previous[1]-delta_ff])
            rate=float(np.clip(-(gain@error)[0],-max_rate,max_rate))
            delta=float(np.clip(previous[1]+self.dt*rate,-max_delta,max_delta))
            if braking is None:
                acceleration=float(accels[k]+kv*(speeds[k]-state[3]))
            else:
                # 制动与MPC使用相同jerk约束；紧急标记不隐含放松控制边界。
                acceleration=-abs(braking)
            acceleration=float(np.clip(acceleration,previous[0]-max_jerk*self.dt,previous[0]+max_jerk*self.dt))
            stopping_bound=-brake_acceleration_limit(max(0.,state[3]),self.dt,max_jerk)
            acceleration=float(np.clip(acceleration,max(amin,stopping_bound),amax))
            control=np.array([acceleration,delta]);controls.append(control)
            for _ in range(self.substeps):
                state=propagate(state,control,scene.ego.length,self.dt/self.substeps,1,
                                scene.ego.wheelbase_m,scene.ego.rear_axle_to_center_m)
                dense.append(state.copy())
            states.append(state.copy());previous=control
        return np.asarray(states),np.asarray(controls),np.asarray(dense)

    def _validate(self,scene,predictions,states,controls,dense,data,stop_s,samples):
        tolerance=float(self.p.get('body_constraint_tolerance_m',.001))
        clearance=float(self.config.get('risk',{}).get('body_clearance_margin_m',.05))
        if scene.drivable is None or scene.drivable.is_empty:
            raise ValueError('PDM缺少可行驶区域，不伪造道路')
        road=scene.drivable.buffer(tolerance);prepare(road)
        lines={}
        for lane in scene.lanes:
            for b in lane.boundaries:
                if b.get('crossable',True) or b.get('kind') in ('virtual_lane_boundary','virtual_connection'):
                    continue
                line=LineString(b['points']);lines[line.normalize().wkb_hex]=line
        offroad=line_crossing=0;contacts=[];minimum=float('inf');stop_ok=True
        for index,state in enumerate(dense):
            t=index*self.dt/self.substeps
            shape=actor_shape(scene.ego,state[:2],state[2])
            offroad+=not road.covers(shape)
            line_crossing+=any(shape.intersects(line) for line in lines.values())
            for identity,actor in predictions.actors.items():
                other=self._object_at(predictions,samples,identity,t)
                gap=float(signed_body_distance(scene.ego,actor,state[:2],state[2],other[:2],other[2]))
                minimum=min(minimum,gap)
                if gap<clearance-tolerance:
                    if not any(row['actor_id']==identity for row in contacts):
                        contacts.append({'actor_id':identity,'time_s':t,'ego_speed_mps':float(state[3]),
                                         'clearance_m':gap})
            if stop_s is not None:
                front=max(project_polyline(rectangle_corners(state[:2],state[2],scene.ego.length,scene.ego.width),scene.route)[0])
                stop_ok &= front<=stop_s+tolerance
        stations,_,headings=_route_projection(scene,states)
        direction=bool(np.all((states[:,3]<.5)|(abs(wrap_angle(states[:,2]-headings))<=np.pi/2)))
        speed_limit=min(float(scene.rules.get('speed_limit',33.3333)),float(self.p.get('maximum_speed_mps',33.3333)))
        speed_tolerance=float(self.p.get('speed_numerical_tolerance_mps',1e-4))
        # 速度使用独立m/s数值容差；不能借车体米制容差或0.1m/s余量放松限速。
        # dense包含真实初态和全部物理子步，规划期望速度本身仍不加该容差。
        speed_ok=bool(np.max(dense[:,3])<=speed_limit+speed_tolerance)
        center_error=np.abs(project_polyline(states[:,:2],data['path'])[1])
        initial_error=center_error[0]
        travel=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1))]
        bound=float(self.p.get('tracking_max_error_m',.25))+initial_error*np.exp(-travel/float(self.p.get('tracking_recovery_distance_m',15.)))
        tracking=bool(np.all(center_error<=bound+1e-6))
        beta=np.arctan(data['rear']/data['wheelbase']*np.tan(controls[:,1]))
        lat=np.maximum(states[:-1,3],states[1:,3])**2*np.sin(beta)/data['rear']
        lateral_ok=bool(np.max(abs(lat))<=float(self.p.get('lateral_acceleration_abs_max_mps2',4.89))+.01)
        previous=np.asarray(scene.diagnostics.get('previous_control',[0.,0.]),float)
        change=np.diff(np.vstack((previous,controls)),axis=0)
        actuator_ok=bool(np.min(states[:,3])>=-1e-7
            and np.min(controls[:,0])>=float(self.p.get('acceleration_min_mps2',-6.))-1e-7
            and np.max(controls[:,0])<=float(self.p.get('acceleration_max_mps2',3.))+1e-7
            and np.max(abs(controls[:,1]))<=float(self.p.get('steering_abs_max_rad',.6))+1e-7
            and np.max(abs(change[:,1]))<=float(self.p.get('steering_rate_abs_max_radps',.8))*self.dt+1e-7)
        jerk_ok=bool(np.max(abs(change[:,0]))<=float(self.p.get('jerk_abs_max_mps3',8.))*self.dt+1e-7)
        control_ok=actuator_ok and jerk_ok
        jerk=float(self.p.get('jerk_abs_max_mps3',8.))
        release_required=stopping_speed_required(min(0.,previous[0]+jerk*self.dt),self.dt,jerk)
        # 时域末端仍须有减速余地；只比较当前候选走廊及同一预测尾帧。
        final=states[-1];station=project_polyline(final[:2],data['path'],heading=final[2])[0]
        braking=float(self.p.get('terminal_braking_mps2',4.));terminal_margin=float('inf')
        for identity,actor in predictions.actors.items():
            other=self._object_at(predictions,samples,identity,self.n*self.dt)
            os,od,oh=project_polyline(other[:2],data['path'],previous_s=station,heading=other[2])
            difference=float(wrap_angle(other[2]-oh))
            width=actor.radius if actor.kind=='pedestrian' else .5*(abs(np.sin(difference))*actor.length+abs(np.cos(difference))*actor.width)
            length=actor.radius if actor.kind=='pedestrian' else .5*(abs(np.cos(difference))*actor.length+abs(np.sin(difference))*actor.width)
            if os>=station and abs(od)<=scene.ego.width/2+width+.05:
                gap=os-station-length-scene.ego.length/2
                lead_speed=max(0.,other[3]*np.cos(difference))
                margin=gap-float(self.p.get('follow_min_gap_m',2.))-max(0.,final[3]**2-lead_speed**2)/(2*braking)
                terminal_margin=min(terminal_margin,margin)
        if stop_s is not None:
            terminal_margin=min(terminal_margin,stop_s-stations[-1]-scene.ego.length/2-final[3]**2/(2*braking))
        terminal_ok=terminal_margin>=-tolerance
        safe=not(offroad or line_crossing or contacts) and stop_ok and direction and speed_ok and tracking and lateral_ok and terminal_ok and control_ok
        return {'valid':bool(safe),'offroad_samples':int(offroad),'line_violation_samples':int(line_crossing),
                'predicted_contacts':contacts,'actor_clearance_min_m':None if not np.isfinite(minimum) else minimum,
                'stop_line_passed':not bool(stop_ok),'correct_direction':direction,'speed_limit':speed_ok,
                'effective_speed_limit_mps':speed_limit,'speed_numerical_tolerance_mps':speed_tolerance,
                'tracking_corridor':tracking,'max_reference_error_m':float(max(center_error)),
                'terminal_stopping_margin_m':None if not np.isfinite(terminal_margin) else float(terminal_margin),
                'lateral_acceleration':lateral_ok,'control_bounds':control_ok,
                'actuator_bounds':actuator_ok,'jerk_bound':jerk_ok,
                'initial_release_conflict':bool(release_required>states[0,3]+1e-7),
                'maximum_jerk_mps3':float(np.max(abs(change[:,0]))/self.dt),'sample_step_s':self.dt/self.substeps,
                'limitation':'0.02秒离散物理复核及已观测对象预测，不是连续时间安全证明'}

    def _risk_goal(self,scene,predictions,plan,answer,goal,samples):
        risk=self.config.get('risk',{});floor=float(risk.get('clearance_floor_m',.25))
        attention={str(a.get('id',a.get('entity_id',''))):1/max(1,int(a.get('rank',i+1)))
                   for i,a in enumerate(answer.attention)} if answer else {}
        total=0.
        for identity,actor in predictions.actors.items():
            cfg=risk.get('classes',{}).get(actor.kind,{'eta':1.,'influence_m':12.})
            eta=float(cfg['eta'])*(1+float(risk.get('attention_increment',1.))*attention.get(identity,0.))
            values=[]
            for k,state in enumerate(plan.states[1:],1):
                other=self._object_at(predictions,samples,identity,k*self.dt)
                gap=max(0.,float(signed_body_distance(scene.ego,actor,state[:2],state[2],other[:2],other[2])))
                values.append(.5*eta*max(1/(gap+floor)-1/(float(cfg['influence_m'])+floor),0.)**2)
            total+=float(np.mean(values))
        # 不可跨越线同样有1.5m基础排斥势；物理验收另行阻止跨线。
        lines={}
        for lane in scene.lanes:
            for boundary in lane.boundaries:
                if not boundary.get('crossable',True) and boundary.get('kind') not in ('virtual_connection','virtual_lane_boundary'):
                    geom=LineString(boundary['points']);lines[geom.normalize().wkb_hex]=(geom,boundary.get('id'))
        lc=risk.get('classes',{}).get('noncrossable_line',{'eta':.03,'influence_m':1.5})
        for line,identity in lines.values():
            eta=float(lc['eta'])*(1+float(risk.get('attention_increment',1.))*attention.get(identity,0.))
            total+=float(np.mean([.5*eta*max(1/(actor_shape(scene.ego,s[:2],s[2]).distance(line)+floor)
                -1/(float(lc['influence_m'])+floor),0.)**2 for s in plan.states[1:]]))
        attraction=0.;event=None
        if goal is not None:
            reference=np.asarray(goal['goal_reference']['points'],float)
            heading=project_polyline(sample_polyline(reference,goal['s_goal_m']),reference,goal['s_goal_m'],window=5.)[2]
            rotation=np.array([[np.cos(heading),np.sin(heading)],[-np.sin(heading),np.cos(heading)]])
            root=np.diag([1/self.goals.p['longitudinal_scale_m'],1/self.goals.p['lateral_scale_m']])@rotation
            residual,_,event=first_arrival_residual(plan.states[:,:2],reference,goal['s_goal_m'],goal['world_goal'],root,goal['last_s'])
            attraction=.5*float(self.goals.p['attraction_weight'])*(residual@residual)
        return total,float(attraction),event

    def emergency_brake(self,scene,predictions,reason):
        result=self.solve(scene,predictions,None,_brake_only=True)
        result.diagnostics['emergency_request_reason']=str(reason)
        return result

    def solve(self,scene,predictions,answer=None,*,_brake_only=False):
        started=perf_counter();cold=self.previous is None
        timing={key:0. for key in ('reference_s','idm_s','lqr_rollout_s','geometry_validation_s','risk_goal_s','scoring_s')}
        _validate_prediction_input(scene,predictions,self.n*self.dt)
        scene,predictions,scope=planning_view(scene,predictions,self.config,answer)
        before=self.export_reference_state();revision=self.goals.revision
        goal=self.goals.update(scene,answer)
        if self.goals.revision!=revision:self.references.reset_intent()
        stop_s,reason=self._rules(scene,predictions)
        samples=self._prediction_samples(predictions);candidates=[];rejected=[]
        if not _brake_only:
            stage_started=perf_counter()
            try:geometries=self._geometries(scene,predictions,goal)
            except ValueError as error:
                rejected.append({'geometry':'local_goal' if goal else 'navigation','reason':str(error)})
                if goal is not None:
                    self.goals._end(goal,'invalid','当前物理状态无法连接原目标：'+str(error),scene)
                    self.references.reset_intent();goal=None
                geometries=[]
            timing['reference_s']+=perf_counter()-stage_started
            for name,path,reference_state,reference_diagnostic in geometries:
                try:data=self._path_data(scene,path)
                except ValueError as error:
                    rejected.append({'geometry':name,'reason':str(error)});continue
                for fraction in self.settings['speed_fractions']:
                    stage_started=perf_counter()
                    progress,speeds,accels=self._idm_reference(scene,predictions,data,float(fraction),stop_s,samples)
                    timing['idm_s']+=perf_counter()-stage_started;stage_started=perf_counter()
                    states,controls,dense=self._track(scene,data,progress,speeds,accels)
                    timing['lqr_rollout_s']+=perf_counter()-stage_started;stage_started=perf_counter()
                    validation=self._validate(scene,predictions,states,controls,dense,data,stop_s,samples)
                    timing['geometry_validation_s']+=perf_counter()-stage_started
                    diagnostic={'safety_feasible':validation['valid'],'execution_mode':'normal','emergency':False,
                        'stop_s':stop_s,'must_wait':stop_s is not None,'wait_reason':reason,
                        'dense_validation':validation,'local_reference':reference_diagnostic}
                    plan=PlanResult(scene.frame_id,scene.time,True,states,controls,diagnostics=diagnostic)
                    stage_started=perf_counter()
                    risk,attraction,event=self._risk_goal(scene,predictions,plan,answer,goal,samples)
                    timing['risk_goal_s']+=perf_counter()-stage_started
                    candidates.append({'id':name+':'+str(fraction),'plan':plan,'reference_state':reference_state,
                        'progress_m':_progress(scene,states),'risk':risk,'attraction':attraction,'goal_event':event,
                        'safe':validation['valid']})
        safe=[candidate for candidate in candidates if candidate['safe']]
        if safe:
            baseline=max(safe,key=lambda c:c['progress_m'])['plan']
            for candidate in safe:
                stage_started=perf_counter()
                report=evaluate(scene,candidate['plan'],baseline,predictions,self.config)
                timing['scoring_s']+=perf_counter()-stage_started
                candidate['quality_valid']=report.valid
                candidate['quality_metrics']={key:report.metrics.get(key) for key in
                    ('gates','progress_m','reference_progress_m','progress','ttc','comfort')}
                if not report.valid:
                    candidate['safe']=False;candidate['quality_reasons']=report.reasons;continue
                if not all(report.metrics['gates'].values()):
                    candidate['safe']=False
                    candidate['quality_reasons']=['独立评分门槛未通过']
                    continue
                metrics=report.metrics;settings=self.settings
                weight=np.array([settings['score_progress_weight'],settings['score_ttc_weight'],settings['score_comfort_weight']])
                base=float(weight@np.array([metrics['progress'],metrics['ttc'],metrics['comfort']])/weight.sum())
                candidate['score']=base-settings['proposal_risk_weight']*candidate['risk']-settings['proposal_goal_weight']*candidate['attraction']
            safe=[candidate for candidate in safe if candidate['safe']]
        if safe:
            chosen=sorted(safe,key=lambda c:(-c['score'],c['id']))[0]
            result=chosen['plan'];self.references.import_state(chosen['reference_state'])
            result.diagnostics.update(pdm_score=chosen['score'],goal_attraction=chosen['attraction'],
                apf_risk=chosen['risk'],goal_event=chosen['goal_event'],selected_proposal=chosen['id'])
        else:
            # 新鲜当前状态生成制动提案，不调用MPC、不使用上一轮控制兜底。
            self.references.import_state(before['local_reference'])
            lanes={l.id:l for l in scene.lanes};lane=self.references._lane(scene,lanes)
            # 制动不重新生成横向回归五次曲线，沿实际车道的合法导航延伸跟踪。
            path=scene.route if lane is None else self.references._continuation(scene,lane,lanes)
            brake_reference_error=None
            try:
                data=self._path_data(scene,path)
            except ValueError as error:
                # 地图路径退化时仍构造物理有界制动；不会伪装成安全解。
                brake_reference_error=str(error)
                heading=scene.ego.heading
                direction=np.array([np.cos(heading),np.sin(heading)])
                path=scene.ego.position+np.array([-10.,max(100.,np.linalg.norm(scene.ego.velocity)*4)])[:,None]*direction
                data=self._path_data(scene,path)
            brakes=[]
            for fraction in self.settings['emergency_braking_fractions']:
                stage_started=perf_counter()
                states,controls,dense=self._track(scene,data,None,None,None,
                    braking=abs(float(self.p.get('acceleration_min_mps2',-6.)))*fraction)
                timing['lqr_rollout_s']+=perf_counter()-stage_started;stage_started=perf_counter()
                validation=self._validate(scene,predictions,states,controls,dense,data,stop_s,samples)
                if brake_reference_error is not None:
                    validation['valid']=False
                    validation['reference_unverified']=brake_reference_error
                timing['geometry_validation_s']+=perf_counter()-stage_started
                if not validation['actuator_bounds']:
                    # 与MPC相同：物理不可执行与安全未证实不能混为一谈。
                    rejected.append({'geometry':'brake:'+str(fraction),
                                     'reason':'初始速度/前一控制不支持满足全部控制边界的制动'})
                    continue
                contacts=validation['predicted_contacts']
                # 无安全制动时先减少预测接触，再降接触速度/越界数量，明确非安全保证。
                order=(not validation['valid'],bool(contacts),len(contacts),
                       sum(row['ego_speed_mps'] for row in contacts),
                       validation['offroad_samples']+validation['line_violation_samples'],
                       -fraction)
                brakes.append((order,states,controls,validation))
            if not brakes:
                raise RuntimeError('当前状态不能形成满足控制边界的PDM制动序列，不伪造物理可执行轨迹')
            _,states,controls,validation=min(brakes,key=lambda row:row[0])
            mode='safe_brake' if validation['valid'] else 'emergency_unverified'
            result=PlanResult(scene.frame_id,scene.time,True,states,controls,reason=mode,
                diagnostics={'execution_mode':mode,'emergency':not validation['valid'],
                    'safety_feasible':validation['valid'],'dense_validation':validation,
                    'stop_s':stop_s,'must_wait':stop_s is not None,'wait_reason':reason,
                    'emergency_generation':'fresh_current_state_lqr_braking',
                    'emergency_comfort_jerk_relaxed':not validation['jerk_bound'],
                    'initial_release_conflict':validation['initial_release_conflict'],
                    'local_reference':{'phase':'brake_only'},
                    'brake_candidates':[{'risk_order':list(row[0]),'validation':row[3]} for row in brakes]})
        result.diagnostics.update(planner_backend=VERSION,source_pipeline='geometry_x_IDM_then_LQR_bicycle_then_score',
            solver_status='PDM_candidate_selection_no_MPC_optimization',cold_start=cold,
            wall_s=perf_counter()-started,pdm_timing=timing,planning_scope=scope,reference_state_before=before,
            goal_state=self.goals.diagnostics(),goal_events=self.goals.events,rule_evidence=self.rule_evidence,
            proposal_count=len(candidates),geometries_rejected=rejected,
            proposals=[{k:v for k,v in c.items() if k not in ('plan','reference_state')}
                       |{'validation':c['plan'].diagnostics['dense_validation']} for c in candidates],
            actual_horizon_s=self.n*self.dt,control_step_s=self.dt)
        self.previous=result.controls.copy();self.previous_time=scene.time
        self.last_diagnostics=result.diagnostics
        return result
