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


def speed_envelope(ego,path,arc,bodies,target_speed,map_limit,offsets=()):
    """Pure current-frame computation, also used for recorded-source replay.

    Coordinates here are CARLA world XYZ, not the paper adapter's flipped Y.
    Entity sizes affect clearance only; this does not construct APF fields.
    """
    values=[target_speed,map_limit,ego['speed'],ego['length'],ego['width'],*ego['position']]
    if not _finite(values) or target_speed<0 or map_limit<=0:raise ValueError('Invalid demonstration speed/ego measurement')
    ep=_project(ego['position'],path,arc,stop=8.)
    if ep[0]>4.:raise ValueError('Demonstration ego is too far from its current reference')
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
        source='current_frozen_world_and_current_ego_reference',
        future_NPC_script_read=False,scenario_id_used=False,paper_planner_modified=False,
        hazard_checks_follow=True,limitation='causal_constant_velocity_envelope;not_a_collision_free_proof')


class CautiousDemonstrationPolicy:
    """Read current frozen geometry; return a speed upper bound, never controls."""
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
        ppm=float(pilot.config.points_per_meter)
        path,arc,indices=_sample_route(route_np,ppm,SETTINGS['lookahead_m'])
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
        planner=pilot._waypoint_planner;offsets=[]
        original=planner.original_route_points;start=planner.route_index
        for station,index in zip(arc,indices):
            if start+index>=len(original):break
            offsets.append((station,math.hypot(route_np[index][0]-original[start+index][0],route_np[index][1]-original[start+index][1])))
        result,evidence=speed_envelope(ego,path,arc,bodies,float(target_speed),float(hero.get_speed_limit())/3.6,offsets)
        evidence.update(source_frame=int(snapshot.frame),source_time=float(snapshot.timestamp.elapsed_seconds),
            expert_speed_xyz_mps=float(ego_speed),speed_semantics='limits_use_frozen_horizontal_speed')
        return result,evidence
