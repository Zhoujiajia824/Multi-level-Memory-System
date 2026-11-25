"""Cautious speed envelopes for the *offline demonstration expert only*.

These are explicit collection quality settings, not a change to the paper
planner or to official traffic rules/scoring. Inputs are current actor states
and the expert's current ego reference. No route ID, event clock or future NPC
script is consulted. Upstream pedestrian/vehicle/light/stop checks still run
after this upper bound and may reduce it further.
"""
import math


SCHEMA = 'cautious-official-demonstration-1.0'
SETTINGS = dict(normal_deceleration_mps2=2.,reaction_s=.8,lateral_acceleration_mps2=1.5,
    maneuver_speed_mps=2.5,vulnerable_speed_mps=2.,vulnerable_notice_m=20.,
    static_notice_m=30.,stop_margin_m=2.,vulnerable_margin_m=1.,
    following_gap_m=3.,following_headway_s=1.8,following_response_s=2.,
    forecast_s=3.,vertical_margin_m=.25,lookahead_m=100.,sample_m=.5,
    recovery_lateral_m=.35,recovery_heading_rad=.15)


def _finite(values):
    return all(math.isfinite(float(v)) for v in values)


def _sample_route(points, ppm, horizon):
    step=max(1,int(round(ppm*SETTINGS['sample_m'])))
    count=min(len(points),int(math.ceil((horizon+5.)*ppm))+2)
    chosen=list(range(0,count,step))
    if count and (not chosen or chosen[-1]!=count-1):chosen.append(count-1)
    path=[];arc=[];indices=[];distance=0.
    for index in chosen:
        p=[float(v) for v in points[index][:3]]
        if len(p)<3 or not _finite(p):raise ValueError('Demonstration reference must have finite XYZ points')
        if path:
            ds=math.hypot(p[0]-path[-1][0],p[1]-path[-1][1])
            if ds<1e-5:continue
            distance+=ds
        path.append(p);arc.append(distance);indices.append(index)
        if distance>=horizon:break
    if len(path)<2:raise ValueError('Demonstration expert has insufficient actual route geometry')
    return path,arc,indices


def _project(point,path,arc,stop=None,start=0.):
    best=None
    for i,(a,b) in enumerate(zip(path,path[1:])):
        if stop is not None and arc[i]>stop:break
        if arc[i+1]<start:continue
        dx,dy=b[0]-a[0],b[1]-a[1];ds=arc[i+1]-arc[i]
        u=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(ds*ds)))
        x,y=a[0]+u*dx,a[1]+u*dy
        residual=math.hypot(point[0]-x,point[1]-y)
        if best is None or residual<best[0]:
            best=(residual,arc[i]+u*ds,(dx/ds,dy/ds),a[2]+u*(b[2]-a[2]))
    return best


def _projection_failure(message,evidence):
    error=ValueError(message)
    error.evidence=dict(schema='demonstration-reference-projection-1.0',**evidence)
    raise error


def _angle(first,second):
    delta=first-second
    return abs(math.atan2(math.sin(delta),math.cos(delta)))


def _qualified_projection(point,path,arc,limits,heading,evidence):
    """Project within a causal route interval; reject another level/direction."""
    start,stop=limits;rows=[]
    for index,(a,b) in enumerate(zip(path,path[1:])):
        if arc[index]>stop:break
        if arc[index+1]<start:continue
        dx,dy=b[0]-a[0],b[1]-a[1];ds=arc[index+1]-arc[index]
        u=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(ds*ds)))
        station=arc[index]+u*ds
        if station<start-1e-6 or station>stop+1e-6:continue
        residual=math.hypot(point[0]-a[0]-u*dx,point[1]-a[1]-u*dy)
        height=a[2]+u*(b[2]-a[2])
        error=_angle(heading,math.atan2(dy,dx))
        if abs(point[2]-height)<=3. and error<=math.pi/3.:
            rows.append((residual,station,(dx/ds,dy/ds),height,error))
    if not rows:
        _projection_failure('No same-direction same-level demonstration reference in the verified local interval',
            dict(evidence,ego_position_xyz_m=list(point),ego_heading_rad=heading,search_station_m=list(limits)))
    rows.sort(key=lambda row:row[0]);best=rows[0]
    # Adjacent segments represent one projection. A near-tied, far-away station
    # indicates a loop/parallel branch; never silently jump to the other pass.
    alternatives=[r for r in rows if r[0]<=best[0]+.25 and abs(r[1]-best[1])>8.]
    if alternatives:
        _projection_failure('Ambiguous demonstration reference across separate route stations',
            dict(evidence,ego_position_xyz_m=list(point),projection_candidates=[list(best[:2]),list(alternatives[0][:2])]))
    return best[:4]


def _at(value,x,y):
    """Linear coordinate mapping only; does not add or alter route geometry."""
    if value<=x[0]:return float(y[0])
    for i in range(len(x)-1):
        if value<=x[i+1]:
            u=(value-x[i])/(x[i+1]-x[i])
            return float(y[i]+u*(y[i+1]-y[i]))
    return float(y[-1])


def _support(body,tangent,normal=False):
    axis=(-tangent[1],tangent[0]) if normal else tangent
    c,s=math.cos(body['heading']),math.sin(body['heading'])
    return abs(axis[0]*c+axis[1]*s)*body['length']/2.+abs(-axis[0]*s+axis[1]*c)*body['width']/2.


def _height_interval(body):
    # Production records always supply measured eight-vertex extrema. The
    # symmetric fallback is only for explicit planar/offline call contracts.
    values=(body.get('z_min',body['position'][2]-.5),body.get('z_max',body['position'][2]+.5))
    if not _finite(values) or values[0]>values[1]:raise ValueError('Invalid measured entity height interval')
    return values


def _same_height(first,second,shift=0.):
    a,b=_height_interval(first);c,d=_height_interval(second)
    return min(b,d+shift)+SETTINGS['vertical_margin_m']>=max(a,c+shift)


def speed_envelope(ego,path,arc,bodies,target_speed,map_limit,offsets=(),*,
                   ego_search_range=(0.,8.),reference_context=None):
    """Pure current-frame computation, also used for recorded-source replay.

    Coordinates here are CARLA world XYZ, not the paper adapter's flipped Y.
    Entity sizes affect clearance only; this does not construct APF fields.
    """
    values=[target_speed,map_limit,ego['speed'],ego['length'],ego['width'],*ego['position']]
    if not _finite(values) or target_speed<0 or map_limit<=0:raise ValueError('Invalid demonstration speed/ego measurement')
    context=dict(reference_context or {})
    if reference_context is None:
        ep=_project(ego['position'],path,arc,stop=ego_search_range[1],start=ego_search_range[0])
    else:
        ep=_qualified_projection(ego['position'],path,arc,ego_search_range,ego['heading'],context)
    if ep is None or ep[0]>4.:
        _projection_failure('Demonstration ego is too far from its current reference',dict(context,
            ego_position_xyz_m=list(ego['position']),ego_heading_rad=ego.get('heading'),
            search_station_m=list(ego_search_range),reference_first_xyz_m=list(path[0]),
            reference_last_xyz_m=list(path[-1]),projection_residual_m=ep[0] if ep else None,
            projection_station_m=ep[1] if ep else None,
            projection_heading_rad=math.atan2(ep[2][1],ep[2][0]) if ep else None,
            projection_height_m=ep[3] if ep else None))
    limits=[dict(reason='current_map_vehicle_limit',speed_mps=float(map_limit),distance_m=0.)]
    settings=SETTINGS
    entity_horizon=min(settings['lookahead_m'],max(40.,ego['speed']**2/(2.*settings['normal_deceleration_mps2'])+
        ego['speed']*settings['reaction_s']+ego['length']+10.))
    def add(reason,speed,distance,actor_id=None):
        free=max(0.,distance-settings['stop_margin_m']-ego['length']/2.-settings['reaction_s']*ego['speed'])
        allowed=math.sqrt(max(0.,speed*speed+2.*settings['normal_deceleration_mps2']*free))
        limits.append(dict(reason=reason,speed_mps=allowed,zone_speed_mps=speed,
            distance_m=distance,actor_id=actor_id))
    heading_error=None
    if ego.get('heading') is not None:
        heading=float(ego['heading'])
        if not math.isfinite(heading):raise ValueError('Invalid frozen demonstration heading')
        angle=heading-math.atan2(ep[2][1],ep[2][0])
        heading_error=abs(math.atan2(math.sin(angle),math.cos(angle)))
    # A reference returning to its original lane does not mean the physical
    # vehicle has returned. Retain cautious tracking until current body pose is
    # aligned, then release without a stale scenario latch or arbitrary timer.
    if ep[0]>settings['recovery_lateral_m'] or (heading_error is not None and heading_error>settings['recovery_heading_rad']):
        add('physical_reference_alignment_pending',settings['maneuver_speed_mps'],0.)
    # Apply curvature limits ahead, before reaching the bend/transition.
    for i in range(1,len(path)-1):
        u=(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1])
        v=(path[i+1][0]-path[i][0],path[i+1][1]-path[i][1])
        length=(arc[i+1]-arc[i-1])/2.
        angle=abs(math.atan2(u[0]*v[1]-u[1]*v[0],u[0]*v[0]+u[1]*v[1]))
        curvature=angle/length
        ahead=arc[i]-ep[1]
        if ahead>=-ego['length']/2. and curvature>.005:
            add('reference_curvature',math.sqrt(settings['lateral_acceleration_mps2']/curvature),max(0.,ahead))
    # Persist across a popped private scenario entry until the actual ego
    # reference has returned to the unchanged original route.
    for station,offset in offsets:
        ahead=station-ep[1]
        if offset>.2 and ahead>=-ego['length']/2.:
            add('active_reference_offset',settings['maneuver_speed_mps'],max(0.,ahead))
            break
    for body in bodies:
        if not _finite([*body['position'],*body['velocity'],body['length'],body['width'],body['heading']]):
            raise ValueError('Nonfinite demonstration entity state')
        p=_project(body['position'],path,arc)
        if not _same_height(ego,body):continue
        along,side=_support(body,p[2]),_support(body,p[2],True)
        distance=p[1]-ep[1]-along
        if distance < -ego['length']/2.-settings['stop_margin_m']:continue
        if distance>entity_horizon:continue
        lateral=p[0]-side-ego['width']/2.
        kind=body['kind'];identifier=body['id']
        vulnerable=kind in ('pedestrian','bicycle','animal')
        if vulnerable and distance<=settings['vulnerable_notice_m'] and lateral<4.:
            add('near_vulnerable_road_user',settings['vulnerable_speed_mps'],max(0.,distance),identifier)
        if kind=='obstacle' and distance<=settings['static_notice_m'] and lateral<1.:
            add('approaching_physical_obstacle',settings['maneuver_speed_mps'],max(0.,distance),identifier)
        if vulnerable and lateral<12.:
            conflicts=[]
            travel=math.hypot(*body['velocity'][:2])*settings['forecast_s']+6.
            for k in range(13):
                seconds=k*settings['forecast_s']/12.
                future=[body['position'][j]+seconds*body['velocity'][j] for j in range(3)]
                projection=_project(future,path,arc,stop=p[1]+travel,start=max(0.,p[1]-travel))
                gap=projection[1]-ep[1]-_support(body,projection[2])
                margin=projection[0]-_support(body,projection[2],True)-ego['width']/2.
                if (gap>=-ego['length']/2. and _same_height(ego,body,body['velocity'][2]*seconds)
                        and margin<settings['vulnerable_margin_m']):conflicts.append(max(0.,gap))
            if conflicts:add('vulnerable_path_conflict',0.,min(conflicts),identifier)
        elif lateral<.3:
            forward=sum(body['velocity'][j]*p[2][j] for j in range(2))
            sideways=abs(body['velocity'][0]*p[2][1]-body['velocity'][1]*p[2][0])
            if kind=='vehicle' and forward>.5 and forward>sideways:
                gap=max(0.,distance-ego['length']/2.)
                desired=settings['following_gap_m']+settings['following_headway_s']*ego['speed']
                allowed=max(0.,forward+(gap-desired)/settings['following_response_s'])
                limits.append(dict(reason='following_headway',speed_mps=allowed,distance_m=distance,actor_id=identifier))
            else:add('occupied_current_reference',0.,max(0.,distance),identifier)
    chosen=min([float(target_speed),*[row['speed_mps'] for row in limits]])
    limits.sort(key=lambda row:row['speed_mps'])
    return chosen,dict(schema=SCHEMA,target_before_mps=float(target_speed),target_after_mps=chosen,
        map_vehicle_limit_mps=float(map_limit),ego_speed_xy_mps=float(ego['speed']),
        binding_reasons=[row for row in limits if row['speed_mps']<=chosen+1e-6][:12],
        relevant_limits=limits[:16],settings=dict(settings),current_entity_count=len(bodies),
        actual_reference_residual_m=ep[0],actual_reference_heading_error_rad=heading_error,
        ego_reference_station_m=ep[1],reference_projection=context,
        source='current_frozen_world_and_current_ego_reference',
        future_NPC_script_read=False,scenario_id_used=False,paper_planner_modified=False,
        hazard_checks_follow=True,limitation='causal_constant_velocity_envelope;not_a_collision_free_proof')


class CautiousDemonstrationPolicy:
    """Read current frozen geometry; return a speed upper bound, never controls."""
    def __init__(self):self._projection_state=None

    def _reference(self,pilot,route_np,ego,snapshot):
        planner=pilot._waypoint_planner;cursor=int(planner.route_index)
        ppm=float(pilot.config.points_per_meter)
        if not math.isfinite(ppm) or ppm<=0:raise ValueError('Invalid expert reference sampling density')
        full=getattr(planner,'route_points',None)
        if full is None:
            # A complete zero-cursor explicit route needs no history. The real
            # upstream planner always exposes route_points; a missing nonzero
            # prefix cannot be silently reconstructed from the current slice.
            if cursor!=0:raise ValueError('Nonzero expert cursor has no complete reference geometry')
            full=route_np
        original=planner.original_route_points
        if cursor<0 or cursor>=len(full)-1 or len(original)!=len(full):
            raise ValueError('Expert reference cursor/original arrays are inconsistent')
        frame=int(snapshot.frame);stamp=float(snapshot.timestamp.elapsed_seconds)
        previous=self._projection_state
        if previous is not None and previous['pilot']!=id(pilot):previous=None
        first=previous is None
        if first:
            history=min(60.,max(15.,float(getattr(pilot.config,'extra_route_length',50.))+8.))
            begin=max(0,cursor-int(math.ceil(history*ppm)))
            end=min(len(full),cursor+int(math.ceil((SETTINGS['lookahead_m']+8.)*ppm))+2)
        else:
            delta=stamp-previous['time']
            if frame<previous['frame'] or delta<0:
                _projection_failure('Demonstration projection source moved backwards',dict(source_frame=frame,source_time=stamp))
            movement=math.dist(ego['position'][:2],previous['position'][:2])
            allowance=max(2.,2.*max(ego['speed'],previous['speed'])*max(delta,.05)+.5)
            if movement>allowance or (frame==previous['frame'] and math.dist(ego['position'],previous['position'])>.01):
                _projection_failure('Demonstration physical source jumped outside its causal projection neighborhood',
                    dict(source_frame=frame,source_time=stamp,observed_movement_m=movement,allowed_movement_m=allowance,
                        previous_position_xyz_m=previous['position'],ego_position_xyz_m=list(ego['position'])))
            anchor=previous['index']
            begin=max(0,int(math.floor(anchor-15.*ppm)))
            end=min(len(full),int(math.ceil(anchor+(SETTINGS['lookahead_m']+allowance+3.)*ppm))+2)
        selected=full[begin:end]
        horizon=(end-begin)/ppm+2.
        path,arc,indices=_sample_route(selected,ppm,horizon)
        absolute=[begin+index for index in indices]
        context=dict(source_frame=frame,source_time=stamp,upstream_cursor=cursor,window_begin_index=begin,
            window_end_index=end,window_first_xyz_m=list(path[0]),window_last_xyz_m=list(path[-1]),
            initialization_prefix_window=first,geometry_source='same_upstream_private_route_points',
            upstream_cursor_changed=False,fabricated_connection=False,global_relocalization=False)
        if first and cursor>0:
            published=getattr(pilot,'org_dense_route_world_coord',None)
            if not published:
                _projection_failure('Expanded startup projection lacks the published original route start',context)
            start=published[0][0].location
            seed=[float(getattr(start,k)) for k in ('x','y','z')]
            following=next((row[0].location for row in published[1:] if math.hypot(row[0].location.x-seed[0],
                row[0].location.y-seed[1])>.2),None)
            if following is None:
                _projection_failure('Published original route has no initial direction',context)
            heading=math.atan2(following.y-seed[1],following.x-seed[0])
            if (math.dist(seed[:2],ego['position'][:2])>4. or abs(seed[2]-ego['position'][2])>3.
                    or _angle(ego['heading'],heading)>math.pi/3.):
                _projection_failure('Startup source is not at the published original route start/direction/level',
                    dict(context,published_start_xyz_m=seed,published_heading_rad=heading,
                        ego_position_xyz_m=list(ego['position']),ego_heading_rad=ego['heading']))
            original_path=[list(map(float,original[index][:3])) for index in absolute]
            original_arc=[0.]
            for a,b in zip(original_path,original_path[1:]):original_arc.append(original_arc[-1]+math.dist(a[:2],b[:2]))
            # Anchor the broad startup window to the *published start*, not to
            # whichever parallel road happens to be nearest the current car.
            seed_stop=_at(min(end-1,cursor+8.*ppm),absolute,original_arc)
            seed_projection=_qualified_projection(seed,original_path,original_arc,
                (0.,seed_stop),heading,context)
            if seed_projection[0]>.5:
                _projection_failure('Published start is absent from the bounded true reference prefix',
                    dict(context,published_start_xyz_m=seed,published_start_residual_m=seed_projection[0]))
            anchor=_at(seed_projection[1],original_arc,absolute)
            low,high=max(begin,anchor-2.*ppm),min(end-1,anchor+8.*ppm)
            context.update(published_start_xyz_m=seed,published_start_index=anchor,published_heading_rad=heading)
        elif first:
            anchor=0.;low,high=0.,min(end-1,8.*ppm)
        else:
            low=max(begin,anchor-2.*ppm);high=min(end-1,anchor+(allowance+3.)*ppm)
        search=(_at(low,absolute,arc),_at(high,absolute,arc))
        context.update(search_index_range=[low,high],search_station_m=list(search))
        return selected,path,arc,indices,absolute,search,context

    def limit(self,pilot,route_np,target_speed,ego_speed):
        snapshot=pilot._world.get_snapshot();hero=pilot._vehicle
        frozen=snapshot.find(hero.id)
        if frozen is None:raise ValueError('Demonstration ego missing from current frozen source')
        pose=frozen.get_transform();velocity=frozen.get_velocity()
        ego=dict(position=[float(getattr(pose.location,k)) for k in ('x','y','z')],
            speed=math.hypot(velocity.x,velocity.y),length=2.*hero.bounding_box.extent.x,width=2.*hero.bounding_box.extent.y,
            heading=math.radians(pose.rotation.yaw))
        ego_vertices=hero.bounding_box.get_world_vertices(pose)
        ego.update(z_min=min(float(v.z) for v in ego_vertices),z_max=max(float(v.z) for v in ego_vertices))
        selected,path,arc,indices,absolute,search,context=self._reference(pilot,route_np,ego,snapshot)
        bodies=[]
        for actor in pilot._world.get_actors():
            if actor.id==hero.id:continue
            identity=actor.type_id
            if not identity.startswith(('vehicle.','walker.','animal.','static.prop.')):continue
            item=snapshot.find(actor.id)
            if item is None:continue
            transform=item.get_transform();p=transform.location
            if math.hypot(p.x-pose.location.x,p.y-pose.location.y)>SETTINGS['lookahead_m']+10.:continue
            box=actor.bounding_box
            if min(box.extent.x,box.extent.y)<=0 or box.extent.z<.015:continue
            # Use the real bbox's local centre transformed by this snapshot,
            # including its own yaw. No live transform or old scene is read.
            matrix=transform.get_matrix();local=box.location
            center=[sum(matrix[i][j]*[local.x,local.y,local.z,1.][j] for j in range(4)) for i in range(3)]
            vertices=box.get_world_vertices(transform)
            v=item.get_velocity()
            wheels=getattr(actor,'attributes',{}).get('number_of_wheels')
            kind=('pedestrian' if identity.startswith('walker.pedestrian.') else
                'animal' if identity.startswith(('walker.animal.','animal.')) else
                'bicycle' if identity in ('vehicle.bh.crossbike','vehicle.diamondback.century','vehicle.gazelle.omafiets') or wheels=='2' else
                'vehicle' if identity.startswith('vehicle.') else 'obstacle')
            bodies.append(dict(id=str(actor.id),kind=kind,position=center,
                velocity=[float(v.x),float(v.y),float(v.z)],length=2.*box.extent.x,width=2.*box.extent.y,
                z_min=min(float(v.z) for v in vertices),z_max=max(float(v.z) for v in vertices),
                heading=math.radians(transform.rotation.yaw+box.rotation.yaw)))
        original=pilot._waypoint_planner.original_route_points;offsets=[]
        for station,index,global_index in zip(arc,indices,absolute):
            offsets.append((station,math.hypot(selected[index][0]-original[global_index][0],selected[index][1]-original[global_index][1])))
        result,evidence=speed_envelope(ego,path,arc,bodies,float(target_speed),float(hero.get_speed_limit())/3.6,offsets,
            ego_search_range=search,reference_context=context)
        physical_index=_at(evidence['ego_reference_station_m'],arc,absolute)
        self._projection_state=dict(pilot=id(pilot),frame=int(snapshot.frame),time=float(snapshot.timestamp.elapsed_seconds),
            index=physical_index,position=list(ego['position']),speed=ego['speed'])
        evidence['reference_projection']['actual_projection_index']=physical_index
        evidence.update(source_frame=int(snapshot.frame),source_time=float(snapshot.timestamp.elapsed_seconds),
            expert_speed_xyz_mps=float(ego_speed),speed_semantics='limits_use_frozen_horizontal_speed')
        return result,evidence
