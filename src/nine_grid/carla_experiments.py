"""Unified physical scenario director; it never ticks the world or controls ego.

The director owns scripted intent. Only public_rule_payload may be passed to the
ego planner; the actor routes and event state are evaluation-only evidence.
"""
import copy
import math

try:
    from .carla_experiment_spec import validate_settings, integer, settings_from_manifest, COORDINATE_CONTRACT
    from .carla_experiment_routes import (inspect_instance, RouteGeometry, adjacent,
        distance, wrap, lane_id)
except ImportError:
    from carla_experiment_spec import validate_settings, integer, settings_from_manifest, COORDINATE_CONTRACT
    from carla_experiment_routes import (inspect_instance, RouteGeometry, adjacent,
        distance, wrap, lane_id)


class EventState:
    """Integer-tick transition logic, independently testable without CARLA."""
    def __init__(self,source):
        self.source=source;self.id=source['role'];self.state='waiting'
        self.trigger_tick=None;self.action_start_tick=None;self.completed_tick=None
        self.transitions=[]

    def transition(self,state,tick,**evidence):
        if self.state==state:return
        item={'event':'scenario_event_transition','id':self.id,'from':self.state,
              'state':state,'tick':tick,'evidence':evidence}
        self.transitions.append(item);self.state=state
        if state=='active':self.trigger_tick=tick;self.action_start_tick=tick+1
        if state=='completed':self.completed_tick=tick

    def consider(self,tick,ego_s,gap=None):
        integer(tick,'event tick')
        if self.state in ('active','completed','missed'):return
        trigger=self.source.get('trigger',{})
        lower,upper=trigger.get('ego_s_m',[0.,float('inf')])
        if ego_s>upper:
            self.transition('missed',tick,reason='ego_passed_trigger_window',ego_s_m=ego_s);return
        if ego_s<lower:return
        self.transition('armed',tick,ego_s_m=ego_s)
        if tick<trigger.get('earliest_tick',0):return
        bounds=trigger.get('gap_m')
        if bounds is not None and (gap is None or not bounds[0]<=gap<=bounds[1]):return
        self.transition('active',tick,ego_s_m=ego_s,gap_m=gap)

    def payload(self):
        return {'id':self.id,'state':self.state,'required':self.source.get('required',False),
            'behavior':self.source['behavior'],'trigger_tick':self.trigger_tick,
            'action_start_tick':self.action_start_tick,'completed_tick':self.completed_tick}


class SignalProgram:
    """One phase per verified incoming approach, rather than per lamp head.

    Only stop-waypoint road/section, direction and tangent establish an approach.
    Different or unknown approaches remain exclusive: lamp orientation does not
    establish compatible movements. Every treatment receives this same program.
    """
    def __init__(self,world,carla,config):
        self.carla=carla;self.config=config;self.groups=[];self.saved=[];self.states={}
        self.last_signature=None;self.phase_evidence=[]
        lights=list(world.get_actors().filter('traffic.traffic_light*'))
        seen=set()
        def key(light):
            t=light.get_transform()
            return (round(t.location.x,3),round(t.location.y,3),round(t.location.z,3),round(t.rotation.yaw,3))
        for light in sorted(lights,key=key):
            if light.id in seen:continue
            group=sorted(list(light.get_group_traffic_lights()) or [light],key=key)
            for member in group:
                if member.id not in seen:
                    self.saved.append((member,member.get_state(),bool(member.is_frozen())))
                    seen.add(member.id)
            phases=[];approaches=[]
            for member in group:
                incoming=self._incoming_approach(member)
                index=next((i for i,previous in enumerate(approaches)
                    if incoming is not None and previous is not None
                    and incoming[0]==previous[0]
                    and all(abs(wrap(a-b))<math.radians(20.)
                        for a,b in zip(incoming[1],previous[1]))),None)
                if index is None:
                    phases.append([member]);approaches.append(incoming)
                else:phases[index].append(member)
            self.groups.append(phases)
            self.phase_evidence.append({'phases':[{
                'light_ids':[str(member.id) for member in phase],
                'incoming_sections':None if approach is None else [list(key) for key in approach[0]],
                'source':'stop_waypoint_incoming_road_section_direction' if approach is not None else 'unknown_single_light',
                } for phase,approach in zip(phases,approaches)],
                'different_approaches':'exclusive_no_unverified_opposing_flow_merge'})
        try:
            for light,_,_ in self.saved:light.freeze(True)
            self.apply(0)
        except BaseException:
            self.close();raise

    @staticmethod
    def _incoming_approach(light):
        """Do not guess an approach from a pole transform or a missing map link."""
        try:points=list(light.get_stop_waypoints())
        except (AttributeError,RuntimeError):return None
        if not points:return None
        by_section={}
        for point in points:
            if not all(hasattr(point,key) for key in ('road_id','section_id','lane_id','transform')):
                return None
            if point.lane_id==0:return None
            key=(int(point.road_id),int(point.section_id),1 if point.lane_id>0 else -1)
            angle=math.radians(float(point.transform.rotation.yaw))
            if not math.isfinite(angle):return None
            by_section.setdefault(key,[]).append(angle)
        keys=tuple(sorted(by_section));headings=[]
        for key in keys:
            angles=by_section[key]
            if any(abs(wrap(angle-angles[0]))>=math.radians(20.) for angle in angles):return None
            headings.append(math.atan2(sum(math.sin(a) for a in angles),sum(math.cos(a) for a in angles)))
        return keys,tuple(headings)

    def apply(self,tick):
        green=self.config.get('green_ticks',500);yellow=self.config.get('yellow_ticks',150)
        red=self.config.get('all_red_ticks',50);slot=green+yellow+red
        signature=[]
        for group in self.groups:
            phase=(tick+self.config.get('offset_ticks',0))%(slot*len(group))
            active=phase//slot;within=phase%slot
            for index,lights in enumerate(group):
                name='Red'
                if index==active:name='Green' if within<green else ('Yellow' if within<green+yellow else 'Red')
                for light in lights:
                    if self.states.get(light.id)!=name:
                        light.set_state(getattr(self.carla.TrafficLightState,name));self.states[light.id]=name
                    signature.append(name)
        changed=tuple(signature)!=self.last_signature;self.last_signature=tuple(signature)
        return changed

    def close(self):
        errors=[]
        for light,state,frozen in self.saved:
            try:
                if light.is_alive:light.set_state(state);light.freeze(frozen)
            except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
        return errors


class Experiment:
    def __init__(self,world,client,settings,carla):
        validate_settings(settings)
        frozen=settings_from_manifest(settings['instance_manifest'])
        if any(settings.get(key)!=value for key,value in frozen.items()):
            raise ValueError('runtime scenario settings differ from their frozen instance')
        self.world,self.client,self.settings,self.carla=world,client,copy.deepcopy(settings),carla
        self.name=settings['scenario_name'];self.manifest=copy.deepcopy(settings['instance_manifest'])
        self.actors=[];self.events=[];self.controllers=[];self._rows=[]
        self._pending_initial_velocities=[];self.ego=None;self._closed=False
        self._signals=None;self._weather=None;self._stops=[];self._errors=[];self._ego_stops={};self._stop_regions={}
        self._tick=-1;self._frame=None;self._ego_hint=None;self._ego_s=0.
        self._ego_projection={'valid':False,'reason':'not_observed'};self._ego_projection_position=None
        self._state={};self.physics_hz=50
        try:self._setup()
        except BaseException:
            self.close();raise

    def _spawn(self,name,wp,role,speed=0.,offset=0.):
        blueprint=self.world.get_blueprint_library().find(name)
        if blueprint.has_attribute('role_name'):blueprint.set_attribute('role_name','hero' if role=='ego' else 'ep_driver_'+role)
        if blueprint.has_attribute('is_invincible'):blueprint.set_attribute('is_invincible','false')
        pose=wp.transform;right=pose.get_right_vector()
        loc=self.carla.Location(x=pose.location.x-offset*right.x,y=pose.location.y-offset*right.y,z=pose.location.z+.3)
        actor=self.world.try_spawn_actor(blueprint,self.carla.Transform(loc,pose.rotation))
        if actor is None:raise RuntimeError('scenario actor spawn failed: '+role)
        self.actors.append(actor)
        if speed>0:
            f=pose.get_forward_vector();self._pending_initial_velocities.append((actor,speed*f.x,speed*f.y,speed*f.z))
        self.events.append({'event':'actor_created','role':role,'actor_id':str(actor.id),'blueprint':name})
        return actor

    def _setup(self):
        self.road_map=self.world.get_map()
        self.forward_route,placements,self.selection_report=inspect_instance(self.road_map,self.settings,self.carla)
        self.route_waypoints=list(self.forward_route);self.geometry=RouteGeometry(self.forward_route)
        self.selected_spawn_index=-1
        preset=getattr(self.carla.WeatherParameters,self.settings.get('weather','ClearNoon'),None)
        if preset is None:raise ValueError('unknown CARLA weather preset')
        self._weather=self.world.get_weather();self.world.set_weather(preset)
        self._signals=SignalProgram(self.world,self.carla,self.settings.get('signals',{}))
        self.ego=self._spawn(self.settings['ego_blueprint'],self.forward_route[0],'ego')
        for source,plan in zip(self.settings['actors'],placements):
            speed=source['speed_mps'] if source['type']=='vehicle' and source['behavior']!='parked' else 0.
            actor=self._spawn(source['blueprint'],plan['waypoint'],source['role'],speed,source['placement'].get('offset_m',0.))
            row={'source':source,'actor':actor,'plan':plan,'geometry':RouteGeometry(plan['route']) if plan['route'] else None,
                'hint':None,'event':EventState(source),'stop_tick':None,'resume_tick':None,'wheelbase':None,
                'target_speed':speed,'transition_count':0,'violation_ticks':0,'stop_signs':{},'initial_s':self.geometry.project(self._center(actor))[0],
                'tracking':{},'last_speed':None,'maximum_speed_error_mps':0.}
            self._rows.append(row)
            if source['type']=='vehicle' and source['behavior']=='parked':actor.apply_control(self.carla.VehicleControl(brake=1.,hand_brake=True))
            if source['type']=='pedestrian':actor.apply_control(self.carla.WalkerControl(speed=0.))
        for light,_,_ in self._signals.saved:
            for wp in light.get_stop_waypoints():self._stops.append(('signal',light,wp))
        for sign in self.world.get_actors().filter('traffic.stop*'):
            points=sign.get_stop_waypoints() if hasattr(sign,'get_stop_waypoints') else []
            source='map_stop_sign_waypoint'
            hold_region=None
            if not points:
                box=sign.trigger_volume;local=box.location
                pose=sign.get_transform()
                zone=pose.transform(self.carla.Location(x=local.x,y=local.y,z=local.z))
                yaw=math.radians(pose.rotation.yaw+box.rotation.yaw)
                hold_region={'x':float(zone.x),'y':float(zone.y),'z':float(zone.z),
                    'extent_x':abs(float(box.extent.x)),'extent_y':abs(float(box.extent.y)),
                    'yaw':yaw}
                self._stop_regions[str(sign.id)]=hold_region
                wp=self.road_map.get_waypoint(zone,project_to_road=True,lane_type=self.carla.LaneType.Driving)
                points=[] if wp is None else [wp]
                source='map_stop_sign_trigger_hold_zone'
            for wp in points:
                self._stops.append(('stop',sign,wp))
                s,d,_,error=self.geometry.project(wp.transform.location)
                if error<2. and any(lane_id(w)==lane_id(wp) for w in self.forward_route):
                    self._ego_stops[str(sign.id)]={'id':str(sign.id),'position':[float(wp.transform.location.x),-float(wp.transform.location.y)],
                        'lane_id':lane_id(wp),'must_stop':True,'source':source,'route_s_m':s,'hold_start_tick':None,'completed_tick':None,
                        'hold_region_carla':hold_region}
        self.events.append({'event':'instance_created','instance_hash':self.settings['instance_hash'],
            'map':self.road_map.name,'signal_policy':'map_incoming_approach_exclusive_physical_ticks',
            'signal_phases':copy.deepcopy(self._signals.phase_evidence),
            'route_task':self.settings['route'].get('task',{}),'server_acceptance':'pending'})

    @staticmethod
    def _speed(actor):
        v=actor.get_velocity();return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)

    def _center(self,actor):
        """Use the same world bounding-box center as planner and evaluator."""
        p=actor.bounding_box.location
        return actor.get_transform().transform(self.carla.Location(x=p.x,y=p.y,z=p.z))

    @staticmethod
    def _half_along_route(actor,geometry,index):
        a=geometry.points[index].transform.location
        b=geometry.points[min(index+1,len(geometry.points)-1)].transform.location
        route_heading=math.atan2(b.y-a.y,b.x-a.x)
        box=actor.bounding_box
        relative=math.radians(actor.get_transform().rotation.yaw+box.rotation.yaw)-route_heading
        return abs(math.cos(relative))*abs(box.extent.x)+abs(math.sin(relative))*abs(box.extent.y)

    def _gap(self,row):
        a=row['actor'];s,_,index,_=self.geometry.project(self._center(a))
        return (s-self._ego_s-self._half_along_route(a,self.geometry,index)
                -self._half_along_route(self.ego,self.geometry,self._ego_hint))

    def _in_stop_region(self,actor,region):
        box=actor.bounding_box;pose=actor.get_transform()
        front=pose.transform(self.carla.Location(x=box.location.x+abs(box.extent.x),y=box.location.y,z=box.location.z))
        dx,dy=front.x-region['x'],front.y-region['y'];c,s=math.cos(region['yaw']),math.sin(region['yaw'])
        return (abs(dx*c+dy*s)<=region['extent_x'] and abs(-dx*s+dy*c)<=region['extent_y']
                and abs(front.z-region['z'])<3.)

    def _flush_events(self,row):
        events=row['event'].transitions
        for item in events[row['transition_count']:]:
            event=copy.deepcopy(item);event['frame']=None if self._frame is None else self._frame+(event['tick']-self._tick)
            event['time']=event['tick']/float(self.physics_hz);self.events.append(event)
        row['transition_count']=len(events)

    def _safe_speed(self,row,request,tick):
        actor=row['actor'];pose=actor.get_transform();loc=self._center(actor)
        limit=float(actor.get_speed_limit())/3.6
        if not math.isfinite(limit) or limit<=0:raise RuntimeError('NPC map speed limit is unavailable: '+row['source']['role'])
        request=min(request,limit)
        geometry=row['geometry'];s,_,hint,_=geometry.project(loc,row['hint']);row['hint']=hint
        request=min(request,math.sqrt(4.*max(0.,geometry.length-s-3.)))
        f=pose.get_forward_vector();r=pose.get_right_vector();half=abs(actor.bounding_box.extent.x)
        # Current physical objects, including ego. NPC scripts get no privileged
        # path predictions of ego and use the same policy under all treatments.
        for other in self.actors:
            if other.id==actor.id or not other.is_alive:continue
            p=self._center(other);dx,dy=p.x-loc.x,p.y-loc.y
            longitudinal=dx*f.x+dy*f.y;lateral=abs(dx*r.x+dy*r.y)
            if abs(p.z-loc.z)>1.5 or longitudinal<=0:continue
            box=other.bounding_box;t=other.get_transform();of=t.get_forward_vector();orr=t.get_right_vector()
            extent_long=abs(box.extent.x*(of.x*f.x+of.y*f.y))+abs(box.extent.y*(orr.x*f.x+orr.y*f.y))
            extent_lat=abs(box.extent.x*(of.x*r.x+of.y*r.y))+abs(box.extent.y*(orr.x*r.x+orr.y*r.y))
            if lateral>abs(actor.bounding_box.extent.y)+extent_lat+.3:continue
            gap=longitudinal-half-extent_long;v=other.get_velocity();lead=max(0.,v.x*f.x+v.y*f.y)
            request=min(request,max(0.,lead+(gap-2.)/1.2),math.sqrt(6.*max(0.,gap-1.5)))
        route_lane_ids={lane_id(w) for w in geometry.points[max(0,hint-1):hint+30]}
        for kind,sign,wp in self._stops:
            if lane_id(wp) not in route_lane_ids:continue
            target_s,_,_,error=geometry.project(wp.transform.location,hint,80)
            distance_to_line=target_s-s-half
            if error>2. or not -1.<=distance_to_line<=50.:continue
            stop=False
            if kind=='signal':stop=self._signals.states.get(sign.id,str(sign.get_state()).split('.')[-1]) in ('Red','Yellow')
            else:
                key=str(sign.id);hold=row['stop_signs'].setdefault(key,{'start':None,'completed':False})
                region=self._stop_regions.get(key)
                inside=self._in_stop_region(actor,region) if region else 0.<=distance_to_line<=5.
                holding=inside and self._speed(actor)<.15
                if not hold['completed']:
                    if not holding:hold['start']=None
                    elif hold['start'] is None:hold['start']=tick
                    elif tick-hold['start']>=self.physics_hz:hold['completed']=True
                stop=not hold['completed']
            if stop:request=min(request,math.sqrt(4.*max(0.,distance_to_line-1.)))
        current=self.road_map.get_waypoint(loc)
        for rule in self.settings.get('public_rules',[]):
            if current is None or lane_id(current) not in rule['controlled_lane_ids']:continue
            stop_point=self.geometry.at(rule['stop_route_s_m']).transform.location
            ahead=(stop_point.x-loc.x)*f.x+(stop_point.y-loc.y)*f.y-half
            if not -1.<=ahead<=50.:continue
            lo,hi=rule['conflict_route_s_m'];centre=self.geometry.at((lo+hi)/2.).transform.location
            for other in self.actors:
                if other.id==actor.id or not other.type_id.startswith('vehicle.'):continue
                ow=self.road_map.get_waypoint(other.get_location())
                if ow and lane_id(ow) in rule['priority_lane_ids'] and distance(self._center(other),centre)<40.:
                    request=min(request,math.sqrt(4.*max(0.,ahead-1.)));break
        return request

    def _follow(self,row,tick,dt):
        actor=row['actor'];source=row['source'];event=row['event'];geometry=row['geometry']
        loc=actor.get_location();speed=self._speed(actor)
        # The steering controller retains the SDK actor/control origin. Progress,
        # event conditions and gaps use XY arc at the actual box center.
        center=self._center(actor)
        s,_,hint,error=geometry.project(center,row['hint']);row['hint']=hint
        row['tracking'].update(control_source_tick=tick,route_error_m=error)
        if error>max(8.,float(geometry.points[hint].lane_width)*2.):
            row['violation_ticks']+=1
            if row['violation_ticks']>=self.physics_hz:raise RuntimeError('NPC left its frozen route: '+source['role'])
        else:row['violation_ticks']=0
        lookahead=max(3.,min(10.,2.+speed*.6));target=geometry.at(min(geometry.length,s+lookahead)).transform.location
        tx,ty=target.x,target.y;request=source['speed_mps']
        if source['behavior']=='cut_in' and event.state in ('active','completed'):
            elapsed=max(0,tick-event.action_start_tick+2);fraction=min(1.,elapsed/float(source.get('duration_ticks',200)))
            blend=10.*fraction**3-15.*fraction**4+6.*fraction**5
            ego_s,_,_,_=self.geometry.project(center)
            destination=self.geometry.at(min(self.geometry.length,ego_s+lookahead)).transform.location
            direction=self.carla.LaneChange.Right if source['placement']['lane']=='left' else self.carla.LaneChange.Left
            if fraction<1. and not int(geometry.points[hint].lane_change)&int(direction):
                raise RuntimeError('cut-in crossed outside its legal corridor')
            tx=(1.-blend)*tx+blend*destination.x;ty=(1.-blend)*ty+blend*destination.y
            if event.state=='completed':
                row['geometry']=self.geometry;geometry=self.geometry;row['hint']=None
        if source['behavior'] in ('stop_resume','hard_brake') and event.state=='active':
            request=0.
        request=self._safe_speed(row,request,tick)
        if row['wheelbase'] is None:
            try:from .carla_bridge import wheel_geometry
            except ImportError:from carla_bridge import wheel_geometry
            row['wheelbase']=wheel_geometry(actor.get_physics_control().wheels,actor.get_transform().get_inverse_matrix(),actor.bounding_box.location.x)[0]
        alpha=wrap(math.atan2(ty-loc.y,tx-loc.x)-math.radians(actor.get_transform().rotation.yaw))
        steering=math.atan2(2.*row['wheelbase']*math.sin(alpha),max(1.,math.hypot(tx-loc.x,ty-loc.y)))
        steering=max(-.6,min(.6,steering))
        if abs(steering)>.02:request=min(request,math.sqrt(2.5*row['wheelbase']/abs(math.tan(steering))))
        decel=source.get('deceleration_mps2',4.) if event.state=='active' else 4.
        # A physical speed ramp, not set_target_velocity during driving.
        previous=row['target_speed']
        requested=max(0.,min(previous+2.*dt,max(previous-decel*dt,request)))
        actor.apply_ackermann_control(self.carla.VehicleAckermannControl(steer=steering,
            steer_speed=.8,speed=requested,acceleration=abs(requested-previous)/dt))
        row['target_speed']=requested

    def before_tick_tick(self,tick,dt):
        integer(tick,'physical tick')
        if self._closed:raise RuntimeError('scenario already closed')
        if tick!=self._tick:raise RuntimeError('scenario event source tick is not the latest frozen state')
        if not math.isfinite(dt) or abs(dt-1./self.physics_hz)>1e-7:raise ValueError('scenario physical dt mismatch')
        if self._signals.apply(tick+1):
            self.events.append({'event':'signal_phase_changed','condition_tick':tick,'effective_tick':tick+1})
        try:
            for row in self._rows:
                actor=row['actor'];source=row['source'];event=row['event']
                if not actor.is_alive:raise RuntimeError('scenario actor disappeared: '+source['role'])
                if source['behavior'] not in ('follow','parked') and self._ego_projection.get('valid',False):
                    event.consider(tick,self._ego_s,self._gap(row))
                self._flush_events(row)
                if source['type']=='pedestrian':
                    end=row['plan']['end'].transform.location;loc=actor.get_location()
                    dx,dy=end.x-loc.x,end.y-loc.y;norm=max(1e-9,math.hypot(dx,dy))
                    speed=source['speed_mps'] if event.state=='active' else 0.
                    actor.apply_control(self.carla.WalkerControl(direction=self.carla.Vector3D(x=dx/norm,y=dy/norm,z=0.),speed=speed))
                elif source['type']=='vehicle' and source['behavior']!='parked':self._follow(row,tick,dt)
        except Exception as exc:
            self._errors.append(str(exc));self._cache_state();raise

    def before_tick(self,elapsed_s,dt):
        tick=int(round(elapsed_s*self.physics_hz))
        if abs(elapsed_s-tick/self.physics_hz)>1e-6:raise ValueError('scenario time is off physical grid')
        return self.before_tick_tick(tick,dt)

    def after_tick(self,frame,tick,snapshot=None):
        integer(frame,'CARLA frame');integer(tick,'physical tick')
        if tick!=self._tick+1:raise RuntimeError('scenario after_tick must receive every physical state exactly once')
        if self._frame is not None and frame!=self._frame+1:raise RuntimeError('external world tick detected by scenario director')
        self._tick,self._frame=tick,frame
        center=self._center(self.ego)
        candidate_s,_,candidate_hint,error=self.geometry.project(center,0 if self._ego_hint is None else self._ego_hint)
        a=self.geometry.points[candidate_hint].transform.location
        b=self.geometry.points[min(candidate_hint+1,len(self.geometry.points)-1)].transform.location
        heading_error=abs(wrap(math.radians(self.ego.get_transform().rotation.yaw)-math.atan2(b.y-a.y,b.x-a.x)))
        width=float(getattr(self.geometry.points[candidate_hint],'lane_width',3.5))
        maximum_error=max(3.,min(6.,1.5*width))
        previous_position=getattr(self,'_ego_projection_position',None)
        travel=0. if previous_position is None else math.hypot(center.x-previous_position[0],center.y-previous_position[1])
        continuous=(previous_position is None or abs(candidate_s-self._ego_s)<=max(2.,travel*1.5+1.))
        valid=error<=maximum_error and heading_error<=math.pi/3 and continuous
        previous_valid=getattr(self,'_ego_projection',{}).get('valid')
        self._ego_projection={'valid':valid,'euclidean_error_m':error,'maximum_error_m':maximum_error,
            'heading_error_rad':heading_error,'continuous':continuous,'candidate_s_m':candidate_s,
            'reason':None if valid else 'outside_local_route_projection_gate',
            'policy':'last_valid_anchor_local_window_no_global_relocalization'}
        if valid:
            self._ego_s,self._ego_hint=candidate_s,candidate_hint
            self._ego_projection_position=(center.x,center.y)
        if previous_valid is not None and previous_valid!=valid:
            self.events.append({'event':'scenario_route_projection_recovered' if valid else 'scenario_route_projection_invalid',
                'tick':tick,'frame':frame,'last_valid_route_s_m':self._ego_s,
                'projection':copy.deepcopy(self._ego_projection)})
        for item in (self._ego_stops.values() if valid else []):
            distance_to_stop=item['route_s_m']-self._ego_s-self._half_along_route(self.ego,self.geometry,self._ego_hint)
            region=item.get('hold_region_carla')
            if region:
                in_zone=self._in_stop_region(self.ego,region)
            else:in_zone=0.<=distance_to_stop<=5.
            holding=in_zone and self._speed(self.ego)<.15
            if item['must_stop']:
                if not holding:item['hold_start_tick']=None
                elif item['hold_start_tick'] is None:item['hold_start_tick']=tick
                elif tick-item['hold_start_tick']>=self.physics_hz:
                    item['must_stop']=False;item['completed_tick']=tick
        for row in self._rows:
            event=row['event'];source=row['source'];actor=row['actor'];kind=source['behavior']
            if tick==0 and source['type']=='vehicle':row['target_speed']=self._speed(actor)
            measured=self._speed(actor)
            acceleration=None if row['last_speed'] is None else (measured-row['last_speed'])*self.physics_hz
            speed_error=measured-row['target_speed']
            row['maximum_speed_error_mps']=max(row['maximum_speed_error_mps'],abs(speed_error))
            row['tracking'].update(observation_tick=tick,measured_speed_mps=measured,
                commanded_speed_mps=row['target_speed'],speed_error_mps=speed_error,
                measured_acceleration_mps2=acceleration,maximum_speed_error_mps=row['maximum_speed_error_mps'])
            row['last_speed']=measured
            if kind in ('follow','parked'):
                if self._ego_s>=source.get('completion_s_m',row['initial_s']+10.):event.transition('completed',tick,ego_passed_initial_region=True)
            elif event.state=='active':
                if kind=='cross' and distance(actor.get_location(),row['plan']['end'].transform.location)<.6:
                    event.transition('completed',tick,actual_endpoint_reached=True)
                elif kind=='cut_in':
                    _,lateral,_,_=self.geometry.project(self._center(actor))
                    wp=self.road_map.get_waypoint(self._center(actor))
                    heading=abs(wrap(math.radians(actor.get_transform().rotation.yaw-wp.transform.rotation.yaw))) if wp else math.pi
                    if tick>=event.action_start_tick+source.get('duration_ticks',200)-1 and abs(lateral)<.35 and heading<.12:
                        event.transition('completed',tick,actual_lateral_error_m=lateral)
                elif kind in ('stop_resume','hard_brake'):
                    if self._speed(actor)<.15 and row['stop_tick'] is None:row['stop_tick']=tick
                    if row['stop_tick'] is not None and tick-row['stop_tick']>=source.get('hold_ticks',150):
                        # Mark action completion after a real stop/hold. Following
                        # resumes next tick; arrival still requires the whole route.
                        event.transition('completed',tick,actual_stop_tick=row['stop_tick'])
            self._flush_events(row)
        self._cache_state()

    def _cache_state(self):
        self._state={'version':'carla-scenario-events-2.0','instance_hash':self.settings['instance_hash'],
            'coordinate_contract':copy.deepcopy(COORDINATE_CONTRACT),
            'tick':self._tick,'frame':self._frame,'ego_s_m':self._ego_s,'route_length_m':self.geometry.length,
            'ego_route_projection':copy.deepcopy(self._ego_projection),
            'scenario_valid':not self._errors,'errors':list(self._errors),
            'events':[r['event'].payload() for r in self._rows],
            'actor_tracking':{r['source']['role']:copy.deepcopy(r['tracking']) for r in self._rows},
            'actor_roles':{'ego':str(self.ego.id),**{r['source']['role']:str(r['actor'].id) for r in self._rows}}}

    def evaluation_state(self):
        return copy.deepcopy(self._state)

    def public_rule_payload(self):
        return {'version':'carla-public-road-rules-1.0','frame':self._frame,'physical_tick':self._tick,
            'coordinate_contract':copy.deepcopy(COORDINATE_CONTRACT),
            'rules':copy.deepcopy(self.settings.get('public_rules',[])),
            'stop_obligations':copy.deepcopy(list(self._ego_stops.values())),
            'source':'published_current_map_rules;no_actor_intent_or_future_trajectories'}

    def close(self):
        if self._closed:return
        self._closed=True;errors=[]
        for actor in reversed(self.actors):
            try:
                if actor.is_alive and actor.destroy() is False:errors.append('actor destroy returned False: '+str(actor.id))
            except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
        if self._signals is not None:errors.extend(self._signals.close())
        if self._weather is not None:
            try:self.world.set_weather(self._weather)
            except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
        if errors:self.events.append({'event':'scenario_cleanup_error','errors':errors})


def create_experiment(world,client,settings,carla):
    return Experiment(world,client,settings,carla)
