"""Scoped, read-only official-manager observations for offline demo collection.

The hook runs after the original manager tick and only controls its loop flag.
It never ticks CARLA, changes a scenario or applies a substitute control.
"""
from copy import deepcopy
import math
import traceback
import weakref

from .demo_segment import SegmentTracker, STOP_METHOD, lane_body_sample
from .demo_behavior import _hull
from .route import lane_key, project, xy


def _status(node):
    value=getattr(node,'status',None)
    return getattr(value,'name',str(value))


def _walk(node):
    yield node
    for child in getattr(node,'children',()):yield from _walk(child)


def _route_lane_section(waypoint):
    """Keep the map basis used to replay the observed vehicle's cross-section."""
    transform=waypoint.transform
    axis=transform.get_forward_vector();norm=math.hypot(axis.x,axis.y)
    width=float(waypoint.lane_width)
    if not math.isfinite(width) or width<=0 or not math.isfinite(norm) or norm<1e-6:
        raise ValueError('Invalid original route lane cross-section')
    sides=[]
    for side in ('left','right'):
        neighbor=getattr(waypoint,'get_'+side+'_lane')()
        if (neighbor and str(neighbor.lane_type).split('.')[-1].lower()=='driving'
                and waypoint.lane_id*neighbor.lane_id<0):
            sides.append(side)
    return dict(lane_id=lane_key(waypoint),width_m=width,opposing_sides=sides,
                center_xy_carla_m=[transform.location.x,transform.location.y],
                forward_xy_carla=[axis.x/norm,axis.y/norm])


def _json_value(value):
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,dict):return {str(k):_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_value(v) for v in value]
    return str(value)


def read_criteria(route_scenario):
    """Capture the actual upstream criteria; no new driving-rule controller."""
    rows=[];failure=None
    benign={'NORMAL_DRIVING','ROUTE_COMPLETION','ROUTE_COMPLETED','MIN_SPEED_INFRACTION'}
    for criterion in route_scenario.get_criteria():
        events=[]
        for event in getattr(criterion,'events',()):
            kind=event.get_type();kind=getattr(kind,'name',str(kind))
            events.append(dict(type=kind,frame=int(event.get_frame()),message=event.get_message(),
                               dictionary=_json_value(event.get_dict())))
            if kind not in benign:failure='official_criterion_event:'+kind
        row=dict(name=criterion.name,criterion_type=type(criterion).__name__,
                 test_status=str(getattr(criterion,'test_status','')),actual_value=_json_value(getattr(criterion,'actual_value',None)),
                 events=events)
        if row['test_status']=='FAILURE' and row['criterion_type'] not in ('MinimumSpeedRouteTest','RouteCompletionTest'):
            failure='official_criterion_failure:'+row['criterion_type']
        rows.append(row)
    return rows,failure


class SegmentRuntime:
    def __init__(self,agent):
        self.agent=weakref.ref(agent);self.tracker=SegmentTracker(agent.config['official_demo']['segment'])
        self.previous_s=0.;self.previous_leaders={};self.gate=None;self.termination=None
        self.stop=None;self.tick_count=0;self.last_frame=None
        self.stop_sign_ids={}

    def _target_stop(self,manager,scenario,snapshot):
        """Match the route stop criterion to the sign selected by ObscuredStopSign."""
        criterion=next((c for c in manager.scenario.get_criteria()
                        if type(c).__name__=='RunningStopTest'),None)
        sign=getattr(criterion,'_target_stop_sign',None)
        if sign is None:return None,False
        if scenario not in self.stop_sign_ids:
            # Pinned ObscuredStopSign selects the first sign along the trigger
            # lane, before the junction, whose trigger centre is within 1.5 m.
            trigger=scenario.config.trigger_points[0].location
            signs=[]
            for actor in criterion._list_stop_signs:
                frozen=snapshot.find(actor.id)
                if frozen is None:continue
                center=frozen.get_transform().transform(actor.trigger_volume.location)
                if center.distance(trigger)<=scenario.max_stop_dist:signs.append((actor.id,center))
            wp=scenario._ego_wp
            while wp and not wp.is_junction and wp.transform.location.distance(trigger)<=scenario.max_stop_dist:
                match=next((key for key,center in signs if wp.transform.location.distance_2d(center)<=1.5),None)
                if match is not None:
                    self.stop_sign_ids[scenario]=match
                    break
                following=wp.next(1.)
                wp=following[0] if following else None
        if sign.id!=self.stop_sign_ids.get(scenario):return None,False
        return sign,criterion._stop_completed is True

    def _gate(self,route):
        if self.gate is not None:return self.gate
        spec=self.tracker.spec;gates=[]
        for target in spec['paired_gen_scenarios']:
            point=target['trigger_xyz_carla']
            foot=project([point[0],-point[1]],route.points,elevations=route.elevations,elevation=point[2])
            if foot['distance_m']>6.:raise ValueError('Paired event trigger is not on the unchanged official route')
            station=foot['s_m']+target['distance_m']
            if station>route.length+3.:raise ValueError('Paired interaction lies beyond the original route')
            gates.append(min(route.length,station))
        self.gate=max(gates)
        return self.gate

    def _targets(self,manager,snapshot,ego_pose):
        expected=self.tracker.spec['expected_scenarios'];result=[]
        for index,item in enumerate(expected):
            matches=[s for s in manager.scenario.list_scenarios if s.config.type==item['type']
                and math.dist([getattr(s.config.trigger_points[0].location,k) for k in ('x','y','z')],
                              item['trigger_xyz_carla'])<1.]
            if len(matches)>1:raise ValueError('Ambiguous instantiated official target scenario')
            row=dict(target_key=str(index)+':'+item['type'],type=item['type'],name=item['name'],
                     instantiated=bool(matches),active=False,ended=False,observed_near=False,
                     vulnerable_motion_observed=False,actors_observed=[],main_status='NOT_INSTANTIATED',timeout=False)
            # The pinned expert brakes at 40 m from a full blockage.
            near_limit=40. if item['type']=='RoadBlocked' else self.tracker.spec['target_near_distance_m']
            row['observed_near_limit_m']=near_limit
            trailing_actor=None
            if matches:
                scenario=matches[0];children=list(scenario.behavior_tree.children)
                if row['target_key'] in self.tracker.bypasses and getattr(scenario,'_construction_wp',None) is not None:
                    # Use the same physical neighbor as the pinned upstream constructor,
                    # which explicitly handles mislabeled TwoWays XML at same-direction sites.
                    home=scenario._construction_wp
                    side='left' if scenario._direction=='right' else 'right'
                    other=getattr(home,'get_'+side+'_lane')()
                    if other is None or str(other.lane_type).split('.')[-1].lower()!='driving':
                        raise ValueError('Official construction site has no adjacent driving lane')
                    row['bypass_site']=dict(side=side,home_lane_id=lane_key(home),neighbor_lane_id=lane_key(other),
                                            opposing=home.lane_id*other.lane_id<0)
                    # Same trailing obstacle reference the official construction behavior
                    # registers with the expert; retain its current footprint, never its spawn pose.
                    if item['type']=='ConstructionObstacleTwoWays':
                        trailing_actor=scenario.other_actors[-2]
                    else:
                        trailing_actor=scenario.last
                # BasicScenario wraps actual behavior between a trigger and a
                # reset-blackboard/WaitForever tail. The root stays RUNNING.
                trigger=children[0] if children else None
                main=children[1] if len(children)>1 else None
                row['main_status']=_status(main)
                row['trigger_status']=_status(trigger)
                row['active']=_status(trigger)=='SUCCESS' and row['main_status'] in ('RUNNING','SUCCESS')
                row['ended']=row['active'] and row['main_status']=='SUCCESS'
                row['timeout']=any(getattr(node,'_scenario_timeout',False) is True for node in _walk(scenario.behavior_tree))
                actors=list(getattr(scenario,'other_actors',()) or ())
                # CustomObstacle/CustomObstacleTwoWays retain their real props
                # here without adding them to BasicScenario.other_actors. Read
                # only actor references; the paired transforms are future spawn
                # poses, never evidence of a currently observed physical body.
                actors.extend(item[0] for item in (getattr(scenario,'_construction_transforms',()) or ()) if item)
                registry={'RoadBlocked':'obstacle_transforms','ImageOnObject':'props',
                          'ObscuredStopSign':'props','PedestrianCrowd':'pedestrians'}.get(item['type'])
                if registry:
                    actors.extend(entry[0] for entry in getattr(scenario,registry))
                sign=None
                if item['type']=='ObscuredStopSign' and row['active']:
                    sign,completed=self._target_stop(manager,scenario,snapshot)
                    row['stop_sign_completed']=completed
                    if sign is not None:actors.append(sign)
                # ConstructionObstaclePedestrian registers the crossing walker
                # as `adversary`, not in BasicScenario.other_actors.
                adversary=getattr(scenario,'adversary',None)
                if adversary is not None:actors.append(adversary)
                seen=set()
                for actor in actors:
                    if actor is None or actor.id in seen:continue
                    seen.add(actor.id)
                    frozen=snapshot.find(actor.id)
                    if frozen is None:continue
                    pose=frozen.get_transform();distance=math.dist([pose.location.x,pose.location.y],[ego_pose.location.x,ego_pose.location.y])
                    if abs(pose.location.z-ego_pose.location.z)>5. or distance>100.:continue
                    velocity=frozen.get_velocity();speed=math.hypot(velocity.x,velocity.y)
                    data=dict(id=int(actor.id),type_id=str(actor.type_id),distance_m=distance,
                              position_xyz_carla_m=[pose.location.x,pose.location.y,pose.location.z],speed_mps=speed)
                    if trailing_actor is not None and row['active'] and distance<=near_limit:
                        vertices=actor.bounding_box.get_world_vertices(pose)
                        try:
                            data['footprint_xy_carla_m']=_hull([(v.x,v.y) for v in vertices])
                        except ValueError:
                            # A zero-area prop cannot establish that the lane was obstructed.
                            data['footprint_xy_carla_m']=[]
                        if actor.id==trailing_actor.id:row['trailing_obstacle_id']=int(actor.id)
                    row['actors_observed'].append(data)
                    if distance<=near_limit:
                        if item['type']=='RoadBlocked':
                            front=ego_pose.get_forward_vector()
                            if ((pose.location.x-ego_pose.location.x)*front.x
                                    +(pose.location.y-ego_pose.location.y)*front.y)>0.:
                                row['blockage_ahead']=True
                        # For an obscured sign require the physical occluder to
                        # have been seen as well as the criterion's real stop.
                        if sign is not None and actor.id==sign.id and scenario.prop_name!='none':continue
                        row['observed_near']=True
                        if actor.type_id.startswith(('walker.','animal.')) and speed>.2:
                            row['vulnerable_motion_observed']=True
            result.append(row)
        return result

    def observe(self,manager):
        agent=self.agent()
        if agent is None:return
        snapshot=agent.world.get_snapshot();frame=int(snapshot.frame);stamp=float(snapshot.timestamp.elapsed_seconds)
        if frame==self.last_frame:return
        if agent.last_frame!=frame:raise ValueError('After-tick segment observation is not bound to the Agent source frame')
        ego=snapshot.find(agent.ego.id)
        if ego is None:raise ValueError('Segment ego absent from official snapshot')
        pose=ego.get_transform()
        source=getattr(agent,'_segment_source_ego',None)
        if source is None or source[0]!=frame:raise ValueError('Segment lacks same-frame observed execution ego')
        observed=source[1];speed=math.hypot(*observed['velocity'][:2]);position=observed['position']
        route=agent.extractor.route;heading=float(observed['heading'])
        projected=project(position,route.points,self.previous_s,25.,heading,
                          elevations=route.elevations,elevation=pose.location.z)
        valid=(projected['distance_m']<=6. and projected['heading_error_rad']<=math.pi/2.
               and projected.get('elevation_error_m',0.)<=3. and projected['s_m']>=self.previous_s-2.)
        if valid:self.previous_s=projected['s_m']
        current_lane=observed['lane_id'];i=projected['segment_index']
        nearby_lanes=[lane_key(wp) for wp in route.waypoints[max(0,i-1):min(len(route.waypoints),i+3)]]
        targets=self._targets(manager,snapshot,pose)
        criteria,failure=read_criteria(manager.scenario)
        if any(t['timeout'] for t in targets):failure='official_target_scenario_timeout'
        leaders={};lead_braking=False;front=pose.get_forward_vector();right=pose.get_right_vector()
        for actor in agent.world.get_actors().filter('vehicle.*'):
            if actor.id==agent.ego.id:continue
            state=snapshot.find(actor.id)
            if state is None:continue
            location=state.get_transform().location;dx,dy=location.x-pose.location.x,location.y-pose.location.y
            if abs(location.z-pose.location.z)>3. or not 0.<dx*front.x+dy*front.y<60. or abs(dx*right.x+dy*right.y)>2.5:continue
            other_pose=state.get_transform()
            other_lane=lane_key(agent.world.get_map().get_waypoint(location))
            direction=other_pose.get_forward_vector()
            if other_lane!=current_lane or direction.x*front.x+direction.y*front.y<.8:continue
            motion=state.get_velocity();accel=state.get_acceleration();other_speed=math.hypot(motion.x,motion.y)
            tangent=(motion.x*accel.x+motion.y*accel.y)/max(other_speed,.1)
            prior=self.previous_leaders.get(actor.id)
            braking=(tangent < -1. or prior is not None and prior-other_speed>.05)
            leaders[actor.id]=other_speed
            if braking and other_speed>.2:lead_braking=True
        self.previous_leaders=leaders
        if self.tracker.spec['family']=='HardBrake':
            for target in targets:
                if target['active'] and leaders:target['observed_near']=True
        gate=self._gate(route)
        row=dict(frame=frame,time=stamp,ego_s_m=projected['s_m'],ego_lane_id=current_lane,
                 ego_position_xyz_carla_m=[position[0],-position[1],pose.location.z],
                 ego_reference='observed_execution_bounding_box_center_xy',
                 speed_mps=speed,projection_valid=valid,projection_distance_m=projected['distance_m'],
                 heading_error_rad=projected['heading_error_rad'],route_lane_match=current_lane in nearby_lanes,
                 target_gate_s_m=gate,in_target_approach=gate-20.<=projected['s_m']<=gate+5.,
                 past_target_gate=projected['s_m']>=min(route.length,gate+self.tracker.spec['pass_margin_m']),
                 target_scenarios=targets,lead_braking_observed=lead_braking and any(t['active'] for t in targets),
                 observed_leading_vehicles=[dict(id=k,speed_mps=v) for k,v in leaders.items()],
                 criteria=criteria,failure_reason=failure)
        if self.tracker.bypasses and valid and failure is None:
            section=_route_lane_section(route.waypoints[i])
            row['route_lane_section']=section
            row['ego_lane_body']=lane_body_sample(section,position,observed['heading'],observed['length'],observed['width'])
        self.tracker.observe(row);self.tick_count+=1;self.last_frame=frame
        agent.archive.append('segment_ticks',row)
        if self.tracker.status!='incomplete' and manager._running:
            self.stop=dict(requested=True,confirmed=True,method=STOP_METHOD,frame=frame,time=stamp)
            manager._running=False
        elif not manager._running and self.stop is None:
            self.termination='official_route_ended_naturally'
        self.write()

    def write(self):
        agent=self.agent()
        if agent is None or agent.archive is None:return
        report=self.tracker.report()
        if self.stop is not None:report['control_stop']=deepcopy(self.stop)
        if self.termination is not None:report['termination']=self.termination
        agent.archive.json('segment_capture.json',report)


def install_manager_hook(agent):
    from leaderboard.scenarios.scenario_manager import ScenarioManager
    original=ScenarioManager._tick_scenario
    if getattr(original,'_ep_demo_segment_hook',False):raise ValueError('Another collection segment hook is still active')
    owner=weakref.ref(agent)
    def after_tick(manager,*args,**kwargs):
        result=original(manager,*args,**kwargs)
        current=owner()
        if (current is not None and current._segment_runtime is not None
                and getattr(getattr(manager,'_agent_wrapper',None),'_agent',None) is current):
            try:
                current._segment_runtime.observe(manager)
            except Exception as error:
                # Preserve an actual runtime/SDK failure as a failed capture;
                # only the explicit completed/budget handshake is a normal stop.
                snapshot=current.world.get_snapshot()
                failure=dict(frame=int(snapshot.frame),time=float(snapshot.timestamp.elapsed_seconds),
                    type=type(error).__name__,error=str(error),source='segment_after_official_tick',
                    classification='segment_observation_failure',traceback=traceback.format_exc())
                current._demo_errors.append(failure)
                try:current.archive.json('capture_failure.json',failure)
                except Exception:pass  # Retain the original runtime/SDK error.
                raise
        return result
    after_tick._ep_demo_segment_hook=True
    ScenarioManager._tick_scenario=after_tick
    agent._segment_hook=(ScenarioManager,original,after_tick)


def restore_manager_hook(agent):
    value=getattr(agent,'_segment_hook',None)
    if value is None:return
    manager,original,installed=value
    if manager._tick_scenario is installed:manager._tick_scenario=original
    agent._segment_hook=None
