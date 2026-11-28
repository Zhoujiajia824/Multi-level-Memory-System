"""Causal execution-aware refinement of one MPC reference, not a score oracle.

The SDK callback previews only our controller/vehicle. All scene, rule, APF and
traffic costs below use the same observed snapshot and causal predictions as
MPC. No official metric cache, future annotations or scorer is accessible here.
"""
from __future__ import annotations
import time
import numpy as np
import shapely
from scipy.signal import savgol_filter
from .protocol import EXECUTION_TTC_OFFSETS_S

VERSION = 'navsim-execution-aware-refinement-1'
TTC_OFFSETS_S = np.asarray(EXECUTION_TTC_OFFSETS_S)
DEFAULTS = dict(enabled=False, iterations=4, wall_budget_s=3.,
    position_difference_m=.01, heading_difference_rad=.002,
    reference_regularization=.05, comfort_weight=20.)


def validate_config(options=None):
    if options is None:options={}
    if not isinstance(options,dict) or set(options)-set(DEFAULTS):
        raise ValueError('Unknown NAVSIM execution-refinement options')
    result={**DEFAULTS,**options}
    if type(result['enabled']) is not bool or type(result['iterations']) is not int or not 1<=result['iterations']<=8:
        raise ValueError('Execution refinement requires boolean enabled and 1..8 iterations')
    for key in set(DEFAULTS)-{'enabled','iterations'}:
        value=result[key]
        if type(value) not in (int,float) or not np.isfinite(value) or value<0 or (key not in ('reference_regularization','comfort_weight') and value==0):
            raise ValueError('Invalid execution-refinement option: '+key)
    if result['wall_budget_s']>60.:
        raise ValueError('Execution-refinement wall budget must be at most 60 seconds')
    return result


def expected_centers(native_states,scene):
    """Native rear-axle local states -> snapshot-frame geometric centres."""
    from nine_grid.geometry import ego_to_world, wrap_angle
    native=np.asarray(native_states,float);rear=scene.ego.rear_axle_to_center_m
    origin=np.asarray(scene.ego.position)-rear*np.array([np.cos(scene.ego.heading),np.sin(scene.ego.heading)])
    xy=native[...,:2]+rear*np.stack((np.cos(native[...,2]),np.sin(native[...,2])),axis=-1)
    xy=ego_to_world(xy.reshape(-1,2),origin,scene.ego.heading).reshape(xy.shape)
    heading=wrap_angle(native[...,2]+scene.ego.heading)
    speed=np.hypot(native[...,3],native[...,4]+rear*native[...,9])
    return np.concatenate((xy,heading[...,None],speed[...,None]),axis=-1)


def comfort_values(native_states,*,benchmark_profile='v1_navtest',rear_axle_to_center_m=1.461,observed_history=None):
    """Six public comfort signals of predicted self-motion, before rounding.

    This matches the pinned SDK's published filtering, including all 41 times.
    Signals derive only from our own controller prediction, not observations
    of what will happen in the dataset or a benchmark score.
    """
    states=np.asarray(native_states,float)
    if states.ndim!=3 or states.shape[1:]!=(41,11) or not np.isfinite(states).all():
        raise ValueError('Comfort signals require finite native Nx41x11 predicted states')
    from .profiles import get_profile
    v1=get_profile(benchmark_profile)['is_v1']
    if observed_history is not None:
        if v1:raise ValueError('NAVSIM v1 comfort does not use history padding')
        prefix=np.asarray(observed_history,float)
        if prefix.ndim!=2 or prefix.shape[1]!=11 or not len(prefix) or not np.isfinite(prefix).all():
            raise ValueError('Comfort observed-history prefix is invalid')
        states=np.concatenate((np.broadcast_to(prefix,(len(states),)+prefix.shape),states),axis=1)
    # Preserve the pinned v2 public metric implementation literally. Its
    # acceleration helper adds BOTH displacement terms to longitudinal x;
    # this is not the rigid-body centre-acceleration transform used for our
    # physical constraints. We do not fix or relabel the official formula.
    ax=states[...,5] if v1 else states[...,5]+rear_axle_to_center_m*(states[...,9]**2+states[...,10])
    ay=states[...,6]
    ax8=savgol_filter(ax,8,2,axis=1)
    magnitude8=savgol_filter(np.hypot(states[...,5],states[...,6]),8,2,axis=1)
    yaw=np.unwrap(states[...,2],axis=1)
    return dict(longitudinal_acceleration=savgol_filter(ax,41 if v1 else 8,2,axis=1),
        lateral_acceleration=savgol_filter(ay,41 if v1 else 8,2,axis=1),
        longitudinal_jerk=savgol_filter(ax8,41 if v1 else 15,2,deriv=1,delta=.1,axis=1),
        magnitude_jerk=savgol_filter(magnitude8,41 if v1 else 15,2,deriv=1,delta=.1,axis=1),
        yaw_rate=savgol_filter(yaw,5,2,deriv=1,delta=.1,axis=1),
        yaw_acceleration=savgol_filter(yaw,5,3,deriv=2,delta=.1,axis=1))


class _Costs:
    def __init__(self,scene,plan,predictions,config,options,answer=None,benchmark_profile=None):
        from types import SimpleNamespace
        from nine_grid.geometry import clean_polyline
        from nine_grid.planner import _path_projection
        from nine_grid.planning_objects import planning_view
        from nine_grid.lane_potential import LanePotentialGeometry
        from nine_grid.potential_domain import build_lane_potential_domain
        self.scene,self.plan,self.config,self.options=scene,plan,config,options
        self.benchmark_profile=benchmark_profile
        from .profiles import get_profile
        self.history=None
        if not get_profile(benchmark_profile)['is_v1']:
            from .native_history import build_native_history
            self.history=build_native_history(scene)
        if not plan.diagnostics.get('semantic_enabled',False):answer=None
        if answer is None:
            attention=(plan.diagnostics.get('planning_object_selection',{}).get('attention_signature',[])
                       if plan.diagnostics.get('semantic_enabled',False) else [])
            answer=SimpleNamespace(attention=attention) if attention else None
        elif answer.frame_id!=scene.frame_id or not np.isclose(answer.time,scene.time,rtol=0.,atol=1e-8):
            raise ValueError('Execution refinement model semantics belong to another source frame')
        scoped,self.predictions,self.selection=planning_view(scene,predictions,config,answer)
        self.actors=scoped.actors;self.p=config['mpc'];self.dt=.1;self.n=40
        self.nominal=np.asarray(plan.states,float)
        if self.nominal.shape!=(41,4):raise ValueError('Execution refinement requires complete forty-step MPC states')
        try:self.reference=clean_polyline(self.nominal[:,:2])
        except ValueError:self.reference=clean_polyline(scene.route)
        # A stopped nominal plan still has a valid public geometric reference.
        if np.linalg.norm(self.reference[-1]-self.reference[0])<.05:self.reference=clean_polyline(scene.route)
        self.navigation=clean_polyline(scene.route)
        self.nav_s0=_path_projection(self.nominal[:1,:2],self.navigation)[0][0]
        self.penalties={'actor_clearance':2000.,'road_clearance':500.,'noncrossable_line_clearance':100.,
            'yield_stop_line':100.,'speed_limit':10.,'lateral_acceleration':5.,'following_headway':20.,
            **self.p.get('safety_penalties',{})}
        self.weights=self.p.get('weights',{});self.norm=self.p.get('normalization',{})
        self.reserve=config.get('navsim_execution',{})
        self.reserve_schedule=plan.diagnostics.get('tracking_reserve_recovery')
        if self.reserve_schedule is not None:
            schedule=self.reserve_schedule
            if (len(schedule.get('times_s',[]))!=41
                    or not np.allclose(schedule['times_s'],np.arange(41)*.1,rtol=0.,atol=1e-8)
                    or len(schedule.get('road_total_hard_margin_m',[]))!=41):
                raise ValueError('MPC tracking reserve schedule has a different execution horizon')
        reach=max(120.,float(np.max(self.nominal[:,3]))*4+30.)
        from shapely.geometry import box, LineString
        e=scene.ego.position
        self.road=scene.drivable.intersection(box(e[0]-reach,e[1]-reach,e[0]+reach,e[1]+reach))
        if self.road.is_empty:self.road=scene.drivable
        shapely.prepare(self.road);self.boundary=self.road.boundary
        self.lines=[]
        for lane in scene.lanes:
            for row in lane.boundaries:
                if not row.get('crossable',True) and row.get('kind') not in ('virtual_connection','virtual_lane_boundary'):
                    line=LineString(row['points'])
                    if not line.is_empty:self.lines.append(line)
        must_wait=plan.diagnostics.get('must_wait',scene.rules.get('must_wait',False))
        self.stop=plan.diagnostics.get('stop_s',scene.rules.get('stop_s')) if must_wait else None
        self.samples={}
        for actor in self.actors:
            tr=np.asarray(self.predictions.trajectories[actor.id],float).copy();tr[:,2]=np.unwrap(tr[:,2])
            self.samples[actor.id]=np.column_stack([np.interp(np.arange(41)*.1,self.predictions.times,tr[:,k]) for k in range(4)])[1:]
        self.goal=(plan.diagnostics.get('goal_state',{}).get('active_goal')
                   if plan.diagnostics.get('active_goal',False) and plan.diagnostics.get('semantic_enabled',False) else None)
        if self.goal is not None and (self.goal['source_frame_id']!=scene.frame_id
                or not np.isclose(self.goal['source_time'],scene.time,rtol=0.,atol=1e-8)):
            raise ValueError('Execution-refinement goal belongs to another source frame')
        domain,_=build_lane_potential_domain(scoped,front_m=reach,rear_m=20.)
        chart=(np.asarray(self.goal['goal_reference']['points']) if self.goal is not None else self.navigation)
        self.field=LanePotentialGeometry(chart,domain)
        initial=self.field.project(self.nominal[:1,:2],require_domain=False)
        self.field.anchor_s=float(initial['s'][0])
        self.nominal_projection=self.field.project_sequence(self.nominal[:,:2],self.field.anchor_s)
        self.source_projection={}
        for actor in self.actors:
            sample=self.samples[actor.id]
            source=self.field.project(sample[:1,:2],anchors=self.field.anchor_s,require_domain=False)
            self.source_projection[actor.id]=self.field.project_sequence(sample[:,:2],float(source['s'][0]),require_domain=False)
        self.attention={row['id']:1/max(1,int(row.get('rank',i+1)))
            for i,row in enumerate(self.selection.get('attention_signature',[]))}
        # TTC is a short constant-heading motion hazard, distinct from a
        # collision along the curved submitted trajectory. It uses the frozen
        # perception prediction set without expanding model input or APF scope.
        perceived=set(scene.diagnostics['perception_scope']['selected_ids'])
        current={actor.id:actor for actor in scene.actors if actor.visible and actor.id in perceived}
        if set(current)!=perceived:
            raise ValueError('Execution TTC requires the complete frozen current perception set')
        query_times=(np.arange(41)[:,None]*self.dt+TTC_OFFSETS_S[None,:]).ravel()
        prediction_times=np.asarray(predictions.times,float)
        if (prediction_times.ndim!=1 or len(prediction_times)<2
                or not np.isfinite(prediction_times).all() or np.any(np.diff(prediction_times)<=0)
                or prediction_times[0]>1e-8 or prediction_times[-1]<float(query_times.max())-1e-8):
            raise ValueError('Causal execution TTC requires existing predictions through 4.9 seconds')
        peak=max(float(self.p['maximum_speed_mps']),float(np.linalg.norm(scene.ego.velocity))
            +max(0.,float(self.p['acceleration_max_mps2']))*4.)
        reach=peak*float(query_times.max())+np.hypot(scene.ego.length,scene.ego.width)/2
        lower=scene.ego.position-reach;upper=scene.ego.position+reach
        self.ttc_actors=[];self.ttc_samples={};excluded=[]
        self.physical_actors=[current[key] for key in sorted(current)];self.physical_samples={}
        for identity,actor in sorted(current.items()):
            if identity not in predictions.trajectories:
                raise ValueError('Perceived TTC actor lacks an existing causal prediction: '+identity)
            trajectory=np.asarray(predictions.trajectories[identity],float).copy()
            if trajectory.shape!=(len(prediction_times),4) or not np.isfinite(trajectory).all():
                raise ValueError('Perceived TTC actor has an invalid causal prediction: '+identity)
            trajectory[:,2]=np.unwrap(trajectory[:,2])
            sample=np.column_stack([np.interp(query_times,prediction_times,trajectory[:,k]) for k in range(4)])
            self.physical_samples[identity]=sample[::len(TTC_OFFSETS_S)].copy()
            radius=(actor.radius or actor.width/2) if actor.kind=='pedestrian' else np.hypot(actor.length,actor.width)/2
            actor_lower=sample[:,:2].min(axis=0)-radius;actor_upper=sample[:,:2].max(axis=0)+radius
            if np.any(actor_upper<lower) or np.any(actor_lower>upper):
                excluded.append(identity);continue
            self.ttc_actors.append(actor);self.ttc_samples[identity]=sample
        self.ttc_provenance=dict(source='current_perception_and_existing_causal_prediction_only',
            current_perception_ids=sorted(perceived),actor_ids=[actor.id for actor in self.ttc_actors],
            bbox_excluded_ids=excluded,projection_offsets_s=TTC_OFFSETS_S.tolist(),
            causal_prediction_end_s=float(prediction_times[-1]),maximum_query_s=float(query_times.max()),
            tail_extrapolation=False,model_and_apf_scope_unchanged=True,
            future_observations_used=False,scorer_or_metric_cache_used=False,
            limitation='Constant-heading hazard proxy with causal predicted entities; not an official TTC score or future collision guarantee')
        self.physical_recovery=None

    def physical_observations(self,states,source='initial_nominal_native_body_and_current_perception_causal_predictions'):
        """Actual candidate body versus causal entities/map, without risk reserves."""
        from nine_grid.planner import body_vertices
        from nine_grid.realtime_solver import circle_gap, rectangle_gap
        from nine_grid.tracking_reserve import signed_body_road_clearance
        tolerance=float(self.p.get('body_constraint_tolerance_m',.001))
        ego=self.scene.ego;states=np.asarray(states,float)
        polygons=shapely.polygons(body_vertices(states,ego.length,ego.width))
        covered=shapely.covers(self.road,polygons);offroad=np.flatnonzero(~covered)
        area=np.zeros(len(states))
        if len(offroad):area[offroad]=shapely.area(shapely.difference(polygons[offroad],self.road))
        buffered_offroad=np.flatnonzero(~shapely.covers(self.road.buffer(tolerance),polygons))
        road_gap=signed_body_road_clearance(polygons,self.road,self.boundary)
        road=dict(violation=bool(len(offroad)),actual_unbuffered_road=True,
            first_time_s=float(offroad[0]*self.dt) if len(offroad) else None,
            minimum_signed_clearance_m=float(np.min(road_gap)),
            minimum_time_s=float(np.argmin(road_gap)*self.dt),violation_samples=int(len(offroad)),
            maximum_outside_area_m2=float(np.max(area)),total_outside_area_m2=float(np.sum(area)),
            outside_samples=[dict(time_s=float(index*self.dt),area_m2=float(area[index])) for index in offroad],
            signed_gap_convention='inside: Euclidean body-boundary distance; outside: negative sqrt(area), a merit proxy, not penetration depth',
            buffered_legacy_diagnostic=dict(numerical_tolerance_m=tolerance,violation=bool(len(buffered_offroad)),
                violation_samples=int(len(buffered_offroad)),first_time_s=float(buffered_offroad[0]*self.dt) if len(buffered_offroad) else None))
        contacts=[]
        for actor in self.physical_actors:
            sample=self.physical_samples[actor.id]
            if actor.kind=='pedestrian':
                gap,_=circle_gap(states,sample[:,:2],actor.radius or actor.width/2,ego.length,ego.width)
            else:gap,_=rectangle_gap(states,sample,ego.length,ego.width,actor.length,actor.width)
            hits=np.flatnonzero(gap < -tolerance)
            if len(hits):
                contacts.append(dict(actor_id=actor.id,kind=actor.kind,first_time_s=float(hits[0]*self.dt),
                    minimum_gap_m=float(np.min(gap)),minimum_time_s=float(np.argmin(gap)*self.dt),
                    violation_samples=int(len(hits))))
        return dict(active=bool(road['violation'] or contacts),source_frame_id=self.scene.frame_id,
            source_time=float(self.scene.time),source_scene_version=self.scene.version,
            source=source,
            safety_feasible_flag_used=False,future_observations_used=False,scorer_or_metric_cache_used=False,
            road=road,contacts=contacts,physical_contact_threshold_m=-tolerance,
            evaluated_actor_ids=[actor.id for actor in self.physical_actors])

    def evaluate(self,native):
        from nine_grid.geometry import wrap_angle
        from nine_grid.planner import body_vertices, _path_projection
        from nine_grid.realtime_solver import rectangle_gap, circle_gap
        from nine_grid.tracking_reserve import signed_body_road_clearance
        from .execution_evidence import native_center_kinematics
        from nine_grid.goals import first_arrival_residual, DEFAULTS as GOAL_DEFAULTS
        native=np.asarray(native,float);batch=len(native);states=expected_centers(native,self.scene)
        cur=states[:,1:];flat=cur.reshape(-1,4);ego=self.scene.ego
        native_cur=native[:,1:];residual=[];constraints=[];components={};constraint_groups={};physical=[];physical_groups={}
        def physical_row(name,values):
            start=sum(value.shape[1] for value in physical)
            physical.append(values);physical_groups[name]=[start,start+values.shape[1]]
        def add(name,values,weight,scale=1.):
            values=np.asarray(values,float).reshape(batch,-1)
            block=values*np.sqrt(float(weight)/self.n)/float(scale)
            residual.append(block);components[name]=np.sum(block*block,axis=1)
        def soft(name,margin,scale=1.,weight=None):
            add(name,np.minimum(np.asarray(margin),0.),self.penalties.get(name,100.) if weight is None else weight,scale)
        def hard(name,margin,scale=1.,residual_weight=2000.,enforce=True):
            raw=np.asarray(margin).reshape(batch,-1)/float(scale)
            if name.startswith('physical_perceived_actor:') or name.startswith('line:') or name in ('potential_domain','stop_line'):
                physical_row(name,np.minimum(raw+float(self.p.get('body_constraint_tolerance_m',.001))/float(scale),0.))
            elif name in ('acceleration_lower','acceleration_upper','steering','steering_rate','jerk','lateral_acceleration','frenet_chart_coverage'):
                physical_row(name,np.minimum(raw+1e-8/float(scale),0.))
            elif name in ('speed_limit','nonnegative_speed'):
                physical_row(name,np.minimum(raw+float(self.p.get('speed_numerical_tolerance_mps',1e-4))/float(scale),0.))
            if enforce:
                start=sum(value.shape[1] for value in constraints)
                constraints.append(raw);constraint_groups[name]=[start,start+raw.shape[1]]
            if residual_weight:add('constraint_'+name,np.minimum(raw,0.),residual_weight)
        ss,off,tan,interior=_path_projection(flat[:,:2],self.reference)
        add('contour',off,self.weights.get('contour',3.),self.norm.get('position_m',3.5))
        # This stage compensates execution of the already chosen MPC intent;
        # a nearest-path cost alone could silently reschedule that intent.
        nominal_xy=np.tile(self.nominal[1:,:2],(batch,1))
        execution_scale=max(float(self.p.get('tracking_max_error_m',.25)),.05)
        add('execution_tracking',flat[:,:2]-nominal_xy,self.weights.get('contour',3.),execution_scale)
        add('execution_heading',wrap_angle(flat[:,2]-np.tile(self.nominal[1:,2],batch)),
            self.weights.get('heading',1.),self.norm.get('heading_rad',.5))
        tangent_heading=np.arctan2(tan[:,1],tan[:,0])
        centre_beta=np.arctan2(native_cur[...,4]+ego.rear_axle_to_center_m*native_cur[...,9],native_cur[...,3])
        add('heading',wrap_angle(flat[:,2]+centre_beta.ravel()-tangent_heading),self.weights.get('heading',1.),self.norm.get('heading_rad',.5))
        nominal_speed=np.tile(self.nominal[1:,3],batch)
        add('speed',flat[:,3]-nominal_speed,self.weights.get('speed',1.),self.norm.get('speed_mps',10.))
        ns,_,nt,_=_path_projection(flat[:,:2],self.navigation);ns=ns.reshape(batch,self.n)
        # Equivalent nonnegative residual for the existing negative linear
        # progress term; its constant zero shift has no effect on optimization.
        maximum=float(self.p['maximum_speed_mps'])*4+float(self.p['acceleration_max_mps2'])*8+ego.length
        progress=np.maximum(0.,maximum-(ns[:,-1]-self.nav_s0))
        add('progress',np.sqrt(progress[:,None]*self.n),self.weights.get('progress',1.),np.sqrt(self.norm.get('progress_m',30.)))
        vertices=body_vertices(flat,ego.length,ego.width);polygons=shapely.polygons(vertices)
        first_nominal=self.physical_recovery is None
        if first_nominal:
            self.physical_recovery=self.physical_observations(states[0])
        probes=np.concatenate((vertices,(vertices+np.roll(vertices,-1,axis=1))/2),axis=1)
        points=shapely.points(probes.reshape(-1,2));distance=shapely.distance(self.boundary,points)
        distance=np.where(shapely.covers(self.road,points),distance,-distance).reshape(batch,self.n,8)
        soft('road_clearance',distance-self.reserve.get('road_tracking_margin_m',.5))
        road_hard=(np.asarray(self.reserve_schedule['road_total_hard_margin_m'],float)[1:]
                   if self.reserve_schedule is not None else np.zeros(self.n))
        if not np.isfinite(road_hard).all() or np.any(road_hard<0):
            raise ValueError('MPC road reserve schedule is invalid')
        hard('road',distance-road_hard[None,:,None],self.norm.get('position_m',3.5))
        # The same full-polygon clearance used by the MPC also covers road
        # concavities and holes whose nearest point lies between its probes.
        full_body_clearance=signed_body_road_clearance(polygons,self.road,self.boundary).reshape(batch,self.n)
        hard('road_full_body',full_body_clearance-road_hard[None,:],self.norm.get('position_m',3.5))
        covered=shapely.covers(self.road,polygons);area=np.zeros(len(polygons))
        if not covered.all():area[~covered]=shapely.area(shapely.difference(polygons[~covered],self.road))
        add('road_body_outside',area.reshape(batch,-1)/ego.width,self.penalties['road_clearance'])
        hard('body_inside_road',-area.reshape(batch,-1)/ego.width,self.norm.get('position_m',3.5))
        initial_polygons=shapely.polygons(body_vertices(states[:,0],ego.length,ego.width))
        initial_covered=shapely.covers(self.road,initial_polygons);initial_area=np.zeros(batch)
        if not initial_covered.all():
            initial_area[~initial_covered]=shapely.area(shapely.difference(initial_polygons[~initial_covered],self.road))
        self.unbuffered_road_covered=np.column_stack((initial_covered,covered.reshape(batch,self.n)))
        physical_area=np.column_stack((initial_area,area.reshape(batch,self.n)))
        # Match the existing complete-footprint signed penetration convention.
        # An area residual would make its squared restoration objective vanish
        # at a higher order while an actual footprint remains outside.
        physical_row('road_footprint',-np.sqrt(physical_area)/self.norm.get('position_m',3.5))
        anchors=np.tile(self.nominal_projection['s'][1:],batch)
        query=self.field.project(flat[:,:2],anchors=anchors,window_m=10.)
        hard('frenet_chart_coverage',np.where(query['coordinate_valid'],1.,-1.))
        domain_gap,_=self.field.domain_clearance(flat[:,:2]);soft('potential_domain',domain_gap)
        hard('potential_domain',domain_gap,self.norm.get('position_m',3.5))
        body_margin=float(self.config.get('risk',{}).get('body_clearance_margin_m',.05))
        tracking=float(self.reserve.get('actor_tracking_margin_m',.75))
        headway=np.full((batch,self.n),np.inf)
        for actor in self.actors:
            sample=np.tile(self.samples[actor.id],(batch,1))
            if actor.kind=='pedestrian':gap,_=circle_gap(flat,sample[:,:2],actor.radius or actor.width/2,ego.length,ego.width)
            else:gap,_=rectangle_gap(flat,sample,ego.length,ego.width,actor.length,actor.width)
            soft('actor_clearance:'+actor.id,gap-body_margin-tracking,weight=self.penalties['actor_clearance'])
            actor_hard=body_margin
            if self.reserve_schedule is not None:
                margins=self.reserve_schedule['actor_total_hard_margin_m'].get(actor.id)
                if margins is None or len(margins)!=41:
                    raise ValueError('MPC tracking reserve lacks a selected actor: '+actor.id)
                margins=np.asarray(margins,float)
                if not np.isfinite(margins).all() or np.any(margins<body_margin-1e-10):
                    raise ValueError('MPC actor reserve schedule weakens physical body clearance')
                actor_hard=np.tile(margins[1:],batch)
            hard('actor:'+actor.id,gap-actor_hard,self.norm.get('position_m',3.5))
            other={key:np.tile(value,(batch,)+(1,)*(value.ndim-1)) for key,value in self.source_projection[actor.id].items()}
            spec=self.config['risk']['classes'][actor.kind]
            field=self.field.repulsion_from_projection(query,other,spec,
                attention_gain=1.+self.config['risk'].get('attention_increment',1.)*self.attention.get(actor.id,0.))
            add('risk:'+actor.id,np.where(field['valid'],field['residual'],0.),self.weights.get('risk',2.))
            # Current-lane following is an additional comfort preference. Real
            # body-distance costs above are always retained, also for side actors.
            # The MPC has already classified current leads along its recovery
            # corridor. Refinement preserves that classification instead of
            # treating rear/crossing/side objects as new fixed-centre leaders.
            if actor.id in self.plan.diagnostics.get('following',{}).get('lead_ids',[]):
                actor_s,_,actor_tangent,_=_path_projection(sample[:,:2],self.navigation)
                leads=self.plan.diagnostics['following']['lead_ids']
                if len(leads)!=41:raise ValueError('Frozen MPC following classification has the wrong horizon')
                eligible=np.tile(np.asarray(leads[1:],object)==actor.id,batch)
                ahead=(actor_s>ns.ravel()) & eligible
                actor_angle=sample[:,2]-np.arctan2(actor_tangent[:,1],actor_tangent[:,0])
                own_angle=flat[:,2]-np.arctan2(nt[:,1],nt[:,0])
                supports=(actor.length*np.abs(np.cos(actor_angle))+actor.width*np.abs(np.sin(actor_angle))
                    +ego.length*np.abs(np.cos(own_angle))+ego.width*np.abs(np.sin(own_angle)))/2
                following=actor_s-ns.ravel()-supports-self.p.get('follow_min_gap_m',2.)-self.p.get('follow_time_headway_s',1.2)*flat[:,3]
                headway=np.minimum(headway,np.where(ahead,following,np.inf).reshape(batch,self.n))
        projected=np.repeat(states[:,:,None,:],len(TTC_OFFSETS_S),axis=2)
        heading_vector=np.stack((np.cos(states[...,2]),np.sin(states[...,2])),axis=-1)
        projected[...,:2]+=states[:,:,None,3,None]*TTC_OFFSETS_S[None,None,:,None]*heading_vector[:,:,None,:]
        ttc_flat=projected.reshape(-1,4)
        for actor in self.ttc_actors:
            sample=np.tile(self.ttc_samples[actor.id],(batch,1))
            if actor.kind=='pedestrian':
                gap,_=circle_gap(ttc_flat,sample[:,:2],actor.radius or actor.width/2,ego.length,ego.width)
            else:
                gap,_=rectangle_gap(ttc_flat,sample,ego.length,ego.width,actor.length,actor.width)
            # Preserve each nominal risk row when it already lacks the same
            # declared execution reserve, instead of allowing a near miss to
            # collapse to contact during reference compensation.
            margin=(gap-body_margin-tracking).reshape(batch,-1)
            name='causal_ttc:'+actor.id
            hard(name,margin,self.norm.get('position_m',3.5),residual_weight=0.)
            soft(name,margin,scale=self.norm.get('position_m',3.5),
                weight=self.penalties['actor_clearance']/len(TTC_OFFSETS_S))
        if self.physical_actors:
            # Auxiliary heading projections cannot prevent recovery from an
            # already predicted real contact. Same-time perceived bodies keep
            # their physical clearance floor independently of those proxies.
            physical_flat=states.reshape(-1,4)
            for actor in self.physical_actors:
                sample=np.tile(self.physical_samples[actor.id],(batch,1))
                if actor.kind=='pedestrian':
                    gap,_=circle_gap(physical_flat,sample[:,:2],actor.radius or actor.width/2,ego.length,ego.width)
                else:gap,_=rectangle_gap(physical_flat,sample,ego.length,ego.width,actor.length,actor.width)
                hard('physical_perceived_actor:'+actor.id,(gap-body_margin).reshape(batch,-1),
                    self.norm.get('position_m',3.5),residual_weight=0.)
        soft('following_headway',headway,scale=self.norm.get('position_m',3.5))
        for i,line in enumerate(self.lines):
            separation=shapely.distance(polygons,line).reshape(batch,self.n)
            crosses=shapely.intersects(polygons,line).reshape(batch,self.n)
            soft('line:'+str(i),np.where(crosses,-.05,separation)-body_margin,weight=self.penalties['noncrossable_line_clearance'])
            hard('line:'+str(i),np.where(crosses,-.05,separation)-body_margin,self.norm.get('position_m',3.5))
        if self.stop is not None:
            corner_s=_path_projection(vertices.reshape(-1,2),self.navigation)[0].reshape(batch,self.n,4)
            soft('yield_stop_line',float(self.stop)-np.max(corner_s,axis=2))
            hard('stop_line',float(self.stop)-np.max(corner_s,axis=2),self.norm.get('position_m',3.5))
        ax=native_cur[...,5];steer=native_cur[...,7];rate=native_cur[...,8]
        speed=native_cur[...,3]
        lat=native_center_kinematics(native_cur,ego.rear_axle_to_center_m)['lateral_acceleration']
        jerk=np.diff(native[...,5],axis=1)/self.dt
        hard('acceleration_lower',ax-float(self.p['acceleration_min_mps2']),self.norm.get('acceleration_mps2',3.))
        hard('acceleration_upper',float(self.p['acceleration_max_mps2'])-ax,self.norm.get('acceleration_mps2',3.))
        hard('nonnegative_speed',speed,self.norm.get('speed_mps',10.))
        hard('steering',float(self.p['steering_abs_max_rad'])-np.abs(steer),self.norm.get('steering_rad',.6))
        hard('steering_rate',float(self.p['steering_rate_abs_max_radps'])-np.abs(rate),self.norm.get('steering_rate_radps',.8))
        hard('jerk',float(self.p['jerk_abs_max_mps3'])-np.abs(jerk),self.norm.get('jerk_mps3',8.))
        hard('lateral_acceleration',float(self.p['lateral_acceleration_abs_max_mps2'])-np.abs(lat),float(self.p['lateral_acceleration_abs_max_mps2']))
        speed_limit=min(float(self.p['maximum_speed_mps']),float(self.scene.rules.get('speed_limit',self.p['maximum_speed_mps'])))
        center_speed=cur[...,3]
        upper=self.plan.diagnostics.get('initial_speed_recovery')
        speed_upper=np.full(self.n,speed_limit)
        if upper is not None:
            speed_upper=np.asarray(upper['hard_upper_speed_mps'],float)
            if speed_upper.shape!=(self.n,) or not np.isfinite(speed_upper).all() or np.any(speed_upper<speed_limit-1e-8):
                raise ValueError('MPC initial-speed recovery envelope is invalid or has a different horizon')
        soft('speed_limit',speed_limit-center_speed,scale=self.norm.get('speed_mps',10.))
        hard('speed_limit',speed_upper[None,:]-center_speed,self.norm.get('speed_mps',10.))
        add('acceleration',ax,self.weights.get('acceleration',.1),self.norm.get('acceleration_mps2',3.))
        add('steering',steer,self.weights.get('steering',.2),self.norm.get('steering_rad',.6))
        add('jerk',jerk,self.weights.get('jerk',.2),self.norm.get('jerk_mps3',8.))
        add('steering_rate',rate,self.weights.get('steering_rate',1.),self.norm.get('steering_rate_radps',.8))
        c=self.config['quality']['comfort'];cw=self.options['comfort_weight']
        signals=comfort_values(native,benchmark_profile=self.benchmark_profile,
            rear_axle_to_center_m=ego.rear_axle_to_center_m,
            observed_history=self.history['states'] if self.history is not None else None)
        comfort_ax=signals['longitudinal_acceleration']
        soft('comfort_acceleration_lower',comfort_ax-.95*c['long_accel_min_mps2'],weight=cw)
        soft('comfort_acceleration_upper',.95*c['long_accel_max_mps2']-comfort_ax,weight=cw)
        hard('comfort_acceleration_lower',comfort_ax-.95*c['long_accel_min_mps2'],abs(c['long_accel_min_mps2']))
        hard('comfort_acceleration_upper',.95*c['long_accel_max_mps2']-comfort_ax,c['long_accel_max_mps2'])
        for name,value,limit in [('longitudinal_jerk',signals['longitudinal_jerk'],c['long_jerk_abs_max_mps3']),
                ('magnitude_jerk',signals['magnitude_jerk'],c['jerk_magnitude_derivative_abs_max_mps3']),
                ('lateral_acceleration',signals['lateral_acceleration'],c['lateral_accel_abs_max_mps2']),
                ('yaw_rate',signals['yaw_rate'],c['yaw_rate_abs_max_radps']),('yaw_acceleration',signals['yaw_acceleration'],c['yaw_accel_abs_max_radps2'])]:
            soft('comfort_'+name,.95*limit-np.abs(value),scale=limit,weight=cw)
            hard('comfort_'+name,.95*limit-np.abs(value),limit)
        if self.goal is not None:
            parameters={**GOAL_DEFAULTS,**self.config.get('local_goal',{})}
            sqrt_weight=np.diag([1/parameters['longitudinal_scale_m'],1/parameters['lateral_scale_m']])
            goals=[];goal_events=[]
            for path in states:
                value,_,event=first_arrival_residual(path[:,:2],np.asarray(self.goal['goal_reference']['points']),
                    self.goal['s_goal_m'],np.asarray(self.goal['world_goal']),sqrt_weight,self.goal['last_s'],frenet_geometry=self.field)
                goals.append(value);goal_events.append(event)
            add('goal_attraction',goals,.5*parameters['attraction_weight']*self.n)
            hard('goal_chart_domain',np.asarray([1. if event['potential_valid'] else -1. for event in goal_events])[:,None])
            self.goal_events=goal_events
        result=np.concatenate(residual,axis=1)
        if not np.isfinite(result).all():raise ValueError('Execution-refinement cost contains nonfinite values')
        self.constraint_groups=constraint_groups
        self.physical_residual=np.concatenate(physical,axis=1)
        self.physical_groups=physical_groups
        if first_nominal:
            merit=float(self.physical_residual[0]@self.physical_residual[0])
            self.physical_recovery.update(contact_or_offroad_observed=self.physical_recovery['active'],
                active=bool(not _physical_feasible(merit,self.unbuffered_road_covered[0])),initial_true_hard_merit=merit,
                true_hard_violation_groups={key:dict(minimum_normalized=float(np.min(self.physical_residual[0,a:b])),
                    squared_normalized_violation=float(self.physical_residual[0,a:b]@self.physical_residual[0,a:b]))
                    for key,(a,b) in physical_groups.items() if np.any(self.physical_residual[0,a:b]<0.)},
                trigger='complete_same_source_true_hard_residual; excludes reserves comfort and heading-projection TTC',
                speed_hard_source='same_MPC_center_speed_initial_recovery_envelope_or_current_map_limit')
        return result,np.concatenate(constraints,axis=1),components


def _physical_feasible(physical_merit,unbuffered_road_covered):
    return bool(physical_merit<=1e-12 and np.all(unbuffered_road_covered))


def _execution_incumbent_rank(physical_merit,objective,physically_feasible=None):
    """A physically feasible incumbent always precedes any infeasible one."""
    feasible=physical_merit<=1e-12 if physically_feasible is None else bool(physically_feasible)
    return ((0,float(objective)) if feasible else
            (1,float(physical_merit),float(objective)))


def _solve_same_reference_qp(hessian,gradient,matrix,lower,trust,deadline=np.inf):
    """Bounded feasible-start fallback for the unchanged convex reference QP.

    Positive scalar objective normalization and variable/row scaling preserve
    the same Hessian, gradient, linear rows and trust box. Every returned
    direction is checked in original units; solver success alone is unused.
    """
    from scipy.optimize import LinearConstraint, minimize
    diagnostic=dict(accepted=False,algorithm='same_QP_feasible_zero_start_SLSQP',
        objective_unchanged=True,linear_rows_unchanged=True,box_unchanged=True,
        maximum_iterations=200,strict_original_primal_limit=1e-8)
    if np.any(lower>0.) or time.perf_counter()>=deadline:
        diagnostic['reason']='zero_not_feasible_or_remaining_budget_exhausted'
        return None,diagnostic
    scaled=matrix*trust[None,:]
    active=-np.sum(np.abs(scaled),axis=1)<lower-1e-10
    row_scale=np.maximum(np.linalg.norm(scaled[active],axis=1),1e-8)
    rows=scaled[active]/row_scale[:,None];bounds=lower[active]/row_scale
    H=hessian*trust[:,None]*trust[None,:];g=gradient*trust
    objective_scale=max(float(np.max(np.abs(H))),float(np.max(np.abs(g))),1.)
    best=None;best_change=0.;iterations=0
    class BudgetExpired(Exception):pass
    def consider(z):
        nonlocal best,best_change
        candidate=np.asarray(z)*trust
        if not np.isfinite(candidate).all():return
        primal=float(max(np.max(lower-matrix@candidate,initial=0.),
            np.max(np.abs(candidate)-trust,initial=0.)))
        change=float(gradient@candidate+.5*candidate@hessian@candidate)
        if primal<=1e-8 and change<best_change-1e-10:best=candidate.copy();best_change=change
    def callback(z):
        nonlocal iterations
        iterations+=1;consider(z)
        if time.perf_counter()>=deadline:raise BudgetExpired()
    try:
        result=minimize(lambda z:(.5*z@H@z+g@z)/objective_scale,np.zeros(len(trust)),
            jac=lambda z:(H@z+g)/objective_scale,bounds=[(-1.,1.)]*len(trust),
            constraints=[LinearConstraint(rows,bounds,np.full(len(bounds),np.inf))] if len(bounds) else [],
            method='SLSQP',callback=callback,options=dict(maxiter=200,ftol=1e-12,disp=False))
        consider(result.x)
        diagnostic.update(solver_reported_success=bool(result.success),solver_message=str(result.message),
            iterations=int(result.nit),budget_exhausted=False)
    except BudgetExpired:
        diagnostic.update(solver_reported_success=False,solver_message='remaining_refinement_budget',
            iterations=iterations,budget_exhausted=True)
    if best is not None:
        primal=float(max(np.max(lower-matrix@best,initial=0.),np.max(np.abs(best)-trust,initial=0.)))
        diagnostic.update(accepted=True,strict_original_primal=primal,predicted_objective_change=best_change,
            candidate_is_feasible_descent_not_a_trajectory_safety_certificate=True)
    return best,diagnostic


def _feasible_reference_direction(hessian,gradient,matrix,lower,trust,raw_step):
    """Recover a feasible descent direction along the known-feasible zero ray.

    The rejected QP iterate is never called optimal or accepted as solved.
    Its ray is intersected analytically with the unchanged linear rows and
    trust box, then checked again before nonlinear backtracking can use it.
    """
    diagnostic=dict(accepted=False,not_an_optimal_QP_solution=True)
    raw_step=np.asarray(raw_step,float)
    if not np.isfinite(raw_step).all() or np.any(lower>0.):
        diagnostic['reason']='zero_not_feasible_or_nonfinite_raw_iterate'
        return raw_step,diagnostic
    product=matrix@raw_step
    if not np.isfinite(product).all():
        diagnostic['reason']='nonfinite_linear_product'
        return raw_step,diagnostic
    ratios=[1.];below=product<lower
    if below.any():ratios.extend((lower[below]/product[below]).tolist())
    outside=np.abs(raw_step)>trust
    if outside.any():ratios.extend((trust[outside]/np.abs(raw_step[outside])).tolist())
    factor=max(0.,min(ratios))*.99;candidate=raw_step*factor
    primal=float(max(np.max(lower-matrix@candidate,initial=0.),
        np.max(np.abs(candidate)-trust,initial=0.)))
    change=float(gradient@candidate+.5*candidate@hessian@candidate)
    diagnostic.update(factor=factor,strict_linear_primal=primal,predicted_objective_change=change,
        accepted=bool(primal<=1e-8 and change<-1e-10))
    return candidate,diagnostic


def refine_execution(scene,plan,predictions,config,preview_callback,answer=None,benchmark_profile=None):
    """Optimize reference poses against predicted actual execution.

    Returns a reference and its separate predicted execution. No candidate is
    substituted by a second braking planner, and no official score is queried.
    """
    from .protocol import trajectory_from_centers
    options=validate_config(config.get('navsim_execution_refinement'))
    nominal=trajectory_from_centers(plan.states,.1,scene.ego.rear_axle_to_center_m).astype(float)
    diagnostic=dict(version=VERSION,enabled=options['enabled'],frame_id=scene.frame_id,
        scorer_or_metric_cache_used=False,future_observations_used=False,
        nominal_mpc_replaced_by_brake=False,postsolve_safety_veto=False,
        expected_execution_is_safety_certificate=False,options=options)
    if not options['enabled']:
        diagnostic.update(status='disabled',selected='nominal_reference')
        return dict(submitted_poses=nominal,expected_states=None,diagnostics=diagnostic)
    if (scene.scenario!='navsim' or not plan.valid or plan.frame_id!=scene.frame_id
            or predictions.frame_id!=scene.frame_id
            or plan.version!=scene.version or predictions.version!=scene.version
            or not np.isclose(plan.time,scene.time,rtol=0.,atol=1e-8)
            or not np.isclose(predictions.time,scene.time,rtol=0.,atol=1e-8)):
        raise ValueError('Execution refinement requires one valid same-frame NAVSIM plan and predictions')
    from nine_grid.planner import _validate_prediction_input
    _validate_prediction_input(scene,predictions,4.)
    if benchmark_profile is None:raise ValueError('Execution refinement requires an explicit frozen benchmark profile')
    from .profiles import get_profile
    get_profile(benchmark_profile)
    if preview_callback is None:raise ValueError('Execution refinement requires its current-frame SDK preview callback')
    begin=time.perf_counter();deadline=begin+options['wall_budget_s'];calls=0;proposals=0
    costs=_Costs(scene,plan,predictions,config,options,answer,benchmark_profile)
    from nine_grid.realtime_solver import QuadraticSolver
    qp=QuadraticSolver(24,maximum_iterations=1200,tolerance=1e-5,require_feasible_iterate=True)
    def preview(poses):
        nonlocal calls,proposals
        value=np.asarray(preview_callback(np.asarray(poses,float)),float);calls+=1;proposals+=len(poses)
        if value.shape!=(len(poses),41,11) or not np.isfinite(value).all():
            raise ValueError('SDK preview returned an invalid or truncated state batch')
        if not np.allclose(value[:,0,:3],0.,rtol=0.,atol=1e-8):
            raise ValueError('SDK preview does not start at the source rear axle')
        return value
    q=nominal.copy();expected=preview(q[None])[0];base_r,base_constraints,base_components=costs.evaluate(expected[None])
    base_score=float(base_r[0]@base_r[0]);best_score=base_score
    base_merit=float(costs.physical_residual[0]@costs.physical_residual[0]);current_merit=base_merit
    initial_road_covered=costs.unbuffered_road_covered[0].copy();current_road_covered=initial_road_covered.copy()
    recovering=bool(costs.physical_recovery['active'])
    auxiliary_mask=np.zeros(base_constraints.shape[1],bool)
    for key,(a,b) in costs.constraint_groups.items():
        if key in ('road','road_full_body') or key.startswith(('comfort_','causal_ttc:','actor:')):auxiliary_mask[a:b]=True
    physical_incumbent=(q.copy(),expected.copy(),best_score) if _physical_feasible(base_merit,current_road_covered) else None
    baseline_limits=np.minimum(base_constraints[0],0.)
    penalty=np.tile([1.,1.,3.],8)*options['reference_regularization'];history=[];termination='iteration_limit'
    scales=np.tile([options['position_difference_m']]*2+[options['heading_difference_rad']],8)
    for iteration in range(options['iterations']):
        if time.perf_counter()>=deadline:termination='wall_budget';break
        batch=np.repeat(q[None],49,axis=0)
        for k in range(24):
            batch[1+2*k].flat[k]+=scales[k];batch[2+2*k].flat[k]-=scales[k]
        values=preview(batch);residual,constraints,_=costs.evaluate(values)
        restoring=recovering and not _physical_feasible(current_merit,current_road_covered)
        working_residual=costs.physical_residual.copy() if restoring else residual
        r=working_residual[0]
        J=np.column_stack([(working_residual[1+2*k]-working_residual[2+2*k])/(2*scales[k]) for k in range(24)])
        qp_penalty=np.zeros_like(penalty) if restoring else penalty
        prior=(q-nominal).ravel();H=2*(J.T@J+np.diag(qp_penalty))+np.eye(24)*1e-4
        gradient=2*(J.T@r+qp_penalty*prior)
        enforced=~auxiliary_mask if restoring else np.ones(len(baseline_limits),bool)
        full_matrix=np.column_stack([(constraints[1+2*k]-constraints[2+2*k])/(2*scales[k]) for k in range(24)])
        matrix=full_matrix[enforced]
        # Zero remains feasible even if the observed nominal has unavoidable
        # violations: retain its per-row deficit, do not loosen another row.
        lower=(np.minimum(constraints[0],baseline_limits)-constraints[0]-1e-7)[enforced]
        trust=np.tile([.75,.75,.15],8)
        step,success,qp_status,_=qp.solve(H,gradient,matrix,lower,
            np.full(len(lower),np.inf),-trust,trust)
        raw_qp_diagnostic=dict(qp.last_diagnostics);direction_recovery=None;same_qp_recovery=None
        if not success:
            recovered,same_qp_recovery=_solve_same_reference_qp(H,gradient,matrix,lower,trust,deadline)
            if same_qp_recovery['accepted']:
                step=recovered;success=True;qp_status='same_QP_feasible_start_SLSQP'
        if not success:
            step,direction_recovery=_feasible_reference_direction(H,gradient,matrix,lower,trust,step)
            direction_recovery.update(raw_qp_status=qp_status,raw_qp_diagnostic=raw_qp_diagnostic)
            success=direction_recovery['accepted']
            if success:qp_status='feasible_zero_anchor_direction:'+qp_status
        if not success:
            termination='reference_qp_'+qp_status;break
        step=step.reshape(8,3)
        search_steps=[step];search_names=['physical_feasibility' if restoring else 'original_objective']
        auxiliary_direction=None
        if restoring and time.perf_counter()<deadline:
            original_J=np.column_stack([(residual[1+2*k]-residual[2+2*k])/(2*scales[k]) for k in range(24)])
            original_H=2*(original_J.T@original_J+np.diag(penalty))+np.eye(24)*1e-4
            original_gradient=2*(original_J.T@residual[0]+penalty*prior)
            original_step,original_ok,original_status,_=qp.solve(original_H,original_gradient,matrix,
                lower,np.full(len(lower),np.inf),-trust,trust)
            auxiliary_direction=dict(qp_eligible=bool(original_ok),qp_status=original_status,
                qp_diagnostic=dict(qp.last_diagnostics),same_preview_and_constraints=True)
            if original_ok:
                search_steps.append(original_step.reshape(8,3));search_names.append('original_objective')
        if time.perf_counter()>=deadline:termination='wall_budget_after_linearization';break
        base_fractions=(1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,0.)
        fractions=base_fractions*len(search_steps)
        trial=np.array([q+f*direction for direction in search_steps for f in base_fractions]);actual=preview(trial)
        rr,mm,cc=costs.evaluate(actual)
        physical_merits=np.einsum('ij,ij->i',costs.physical_residual,costs.physical_residual)
        trial_road_covered=costs.unbuffered_road_covered.copy()
        physical_feasible=(physical_merits<=1e-12) & np.all(trial_road_covered,axis=1)
        scores=np.einsum('ij,ij->i',rr,rr)+np.sum(penalty*(trial-nominal).reshape(len(fractions),24)**2,axis=1)
        # Nonlinear optimizer filter keeps every constraint row separate: a
        # pre-existing collision or initial steering deficit cannot excuse a
        # new violation elsewhere. It is not an independent audit/brake gate.
        eligible=np.all(mm[:,enforced]>=baseline_limits[None,enforced]-1e-6,axis=1)
        # A covered source footprint is an exact geometric invariant. The
        # normalized SQP tolerance must not authorize a newly outside sliver.
        eligible &= np.all(trial_road_covered[:,initial_road_covered],axis=1)
        if physical_incumbent is not None:eligible &= physical_feasible
        phase_scores=physical_merits if restoring else scores
        phase_before=current_merit if restoring else best_score
        # A nonlinear constraint can curve away from an otherwise feasible
        # SQP direction. A same-Jacobian minimum-norm correction repairs that
        # local linearization error before the identical nonlinear filter.
        # The correction uses the original trust box and never changes a
        # physical boundary, objective weight or nominal per-row deficit.
        corrections=[]
        for index in np.argsort(phase_scores)[:3]:
            if eligible[index] or phase_scores[index]>=phase_before or time.perf_counter()>=deadline:continue
            offset=(trial[index]-q).ravel()
            correction,okay,status,_=qp.solve(2*np.diag(np.tile([1.,1.,3.],8)),np.zeros(24),
                matrix,baseline_limits[enforced]-mm[index,enforced]-1e-7,np.full(len(lower),np.inf),
                -trust-offset,trust-offset)
            evidence=dict(fraction=fractions[index],qp_eligible=bool(okay),qp_status=status,
                qp_diagnostic=dict(qp.last_diagnostics),correction_max=float(np.max(np.abs(correction))))
            corrections.append(evidence)
            if not okay:continue
            corrected=trial[index]+correction.reshape(8,3)
            corrected_actual=preview(corrected[None]);cr,cm,_=costs.evaluate(corrected_actual)
            corrected_merit=float(costs.physical_residual[0]@costs.physical_residual[0])
            corrected_road_covered=costs.unbuffered_road_covered[0].copy()
            corrected_feasible=_physical_feasible(corrected_merit,corrected_road_covered)
            corrected_score=float(cr[0]@cr[0]+np.sum(penalty*(corrected-nominal).ravel()**2))
            corrected_eligible=bool(np.all(cm[0,enforced]>=baseline_limits[enforced]-1e-6)
                and np.all(corrected_road_covered[initial_road_covered])
                and (physical_incumbent is None or corrected_feasible))
            evidence.update(nonlinear_eligible=corrected_eligible,objective=corrected_score)
            improvement=(_execution_incumbent_rank(corrected_merit,corrected_score,corrected_feasible)<_execution_incumbent_rank(current_merit,best_score,_physical_feasible(current_merit,current_road_covered))
                if recovering else corrected_score<best_score)
            if corrected_eligible and improvement:
                trial[index]=corrected;actual[index]=corrected_actual[0]
                mm[index]=cm[0];scores[index]=corrected_score;physical_merits[index]=corrected_merit;eligible[index]=True
                trial_road_covered[index]=corrected_road_covered;physical_feasible[index]=corrected_feasible
                break
        candidates=np.flatnonzero(eligible)
        chosen=(min(candidates,key=lambda index:_execution_incumbent_rank(physical_merits[index],scores[index],physical_feasible[index])
            if recovering else (0,float(scores[index]))) if len(candidates) else len(fractions)-1)
        changed=bool(eligible[chosen] and (_execution_incumbent_rank(physical_merits[chosen],scores[chosen],physical_feasible[chosen])<_execution_incumbent_rank(current_merit,best_score,_physical_feasible(current_merit,current_road_covered))
            if recovering else scores[chosen]<best_score-1e-9))
        history.append(dict(iteration=iteration+1,cost_before=best_score,
            candidate_cost=float(scores[chosen]) if np.isfinite(scores[chosen]) else None,
            phase='physical_feasibility_restoration' if restoring else 'original_objective',
            physical_merit_before=current_merit,candidate_physical_merit=float(physical_merits[chosen]),
            accepted=bool(changed),step_fraction=fractions[chosen],
            constraint_violation=float(max(0.,-np.min(mm[chosen]))),qp_status=qp_status,
            raw_qp_diagnostic=raw_qp_diagnostic,feasible_direction_recovery=direction_recovery,
            same_qp_feasible_start_recovery=same_qp_recovery,second_order_corrections=corrections,
            search_directions=search_names,selected_direction=search_names[chosen//len(base_fractions)],
            auxiliary_original_objective_direction=auxiliary_direction))
        if not changed:termination='no_objective_improvement';break
        accepted_step=trial[chosen]-q
        q=trial[chosen];expected=actual[chosen];best_score=float(scores[chosen]);current_merit=float(physical_merits[chosen])
        current_road_covered=trial_road_covered[chosen].copy()
        if physical_feasible[chosen]:
            if recovering and physical_incumbent is None:
                baseline_limits[auxiliary_mask]=np.minimum(mm[chosen,auxiliary_mask],0.)
            if physical_incumbent is None or best_score<physical_incumbent[2]:
                physical_incumbent=(q.copy(),expected.copy(),best_score)
        if np.max(np.abs(accepted_step))<1e-5:termination='small_reference_step';break
    if physical_incumbent is not None:q,expected,best_score=physical_incumbent
    _,final_constraints,final_components=costs.evaluate(expected[None])
    final_physical=costs.physical_observations(expected_centers(expected[None],scene)[0],
        source='final_expected_native_body_and_current_perception_causal_predictions')
    diagnostic.update(status='complete',selected='refined_reference' if not np.array_equal(q,nominal) else 'nominal_reference',
        nominal_objective=base_score,final_objective=best_score,
        nominal_constraint_violation=float(max(0.,-np.min(base_constraints))),
        expected_constraint_violation=float(max(0.,-np.min(final_constraints))),
        constraint_filter='true_hard_rows_retained; auxiliary_rows_restore_after_phase_I; physical_feasible_incumbent_cannot_regress; no_score_or_brake_veto',
        constraint_groups={key:dict(nominal_min=float(np.min(base_constraints[0,a:b])),
            final_min=float(np.min(final_constraints[0,a:b]))) for key,(a,b) in costs.constraint_groups.items()},
        objective_components={key:dict(nominal=float(base_components[key][0]),final=float(final_components[key][0])) for key in base_components},
        optimization_actor_ids=sorted(costs.predictions.actors),optimization_scope=costs.selection,
        model_attention_signature=costs.selection.get('attention_signature',[]),
        model_goal_id=costs.goal.get('goal_id') if costs.goal is not None else None,
        expected_goal_event=costs.goal_events[0] if costs.goal is not None else None,
        causal_ttc=costs.ttc_provenance,
        physical_recovery=costs.physical_recovery,
        physical_recovery_policy='causal_true_hard_phase_I; tracking_reserves_comfort_and_heading_TTC_remain_costs_during_restoration; auxiliary_hard_restored_after_feasibility; feasible_incumbent_never_replaced_by_unsafe',
        phase_I_auxiliary_hard_groups=[key for key,(a,b) in costs.constraint_groups.items() if np.all(auxiliary_mask[a:b])],
        phase_I_hard_release_applied=any(row['phase']=='physical_feasibility_restoration' for row in history),
        tracking_reserve_policy=dict(physical_floor_preserved=True,declared_MPC_margin_schedule_preserved=True,
            original_reserve_soft_costs_retained=True,hard_during_phase_I=False,hard_restored_after_physical_feasibility=True),
        initial_physical_merit=base_merit,final_physical_merit=float(costs.physical_residual[0]@costs.physical_residual[0]),
        final_true_hard_violation_groups={key:dict(minimum_normalized=float(np.min(costs.physical_residual[0,a:b])),
            squared_normalized_violation=float(costs.physical_residual[0,a:b]@costs.physical_residual[0,a:b]))
            for key,(a,b) in costs.physical_groups.items() if np.any(costs.physical_residual[0,a:b]<0.)},
        has_physical_feasible_incumbent=physical_incumbent is not None,
        road_feasibility_contract='actual_unbuffered_full_body_covers; no_new_outside_at_initially_covered_frames',
        strict_no_new_road=dict(initially_covered_frames=int(np.sum(initial_road_covered)),
            final_new_outside_frames=np.flatnonzero(initial_road_covered & ~costs.unbuffered_road_covered[0]).tolist()),
        final_physical_observations=final_physical,
        physical_recovery_incomplete=bool(recovering and not _physical_feasible(float(costs.physical_residual[0]@costs.physical_residual[0]),costs.unbuffered_road_covered[0])),
        expected_physical_violation=bool(not _physical_feasible(float(costs.physical_residual[0]@costs.physical_residual[0]),costs.unbuffered_road_covered[0])),
        preview_calls=calls,preview_proposals=proposals,iteration_history=history,termination=termination,
        elapsed_s=time.perf_counter()-begin,separate_wall_budget_s=options['wall_budget_s'],
        wall_budget_exceeded=time.perf_counter()>deadline)
    diagnostic.update(benchmark_profile=benchmark_profile,comfort_signal_history_used=costs.history is not None,
        comfort_history_provenance=costs.history['provenance'] if costs.history is not None else None,
        comfort_proxy_is_official_HC=False,last_reference_qp=dict(qp.last_diagnostics),
        mpc_recoverable_tracking_reserve_preserved=costs.reserve_schedule is not None)
    diagnostic['objective_components']['reference_regularization']=dict(nominal=0.,
        final=float(np.sum(penalty*(q-nominal).ravel()**2)))
    return dict(submitted_poses=q.astype(np.float32),expected_states=expected,diagnostics=diagnostic)
