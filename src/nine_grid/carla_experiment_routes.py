"""Decode and audit frozen public routes. No route search is performed at runtime."""
import hashlib
import math


def distance(a,b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)


def wrap(angle):
    return (angle+math.pi)%(2*math.pi)-math.pi


def lane_id(wp):
    return '{}|{}|{}'.format(wp.road_id,wp.section_id,wp.lane_id)


def map_hash(road_map):
    return hashlib.sha256(road_map.to_opendrive().encode('utf-8')).hexdigest()


def decode_route(road_map,spec,verified_map_hash=None):
    if (verified_map_hash or map_hash(road_map)) != spec['opendrive_sha256']:
        raise ValueError('locked route OpenDRIVE checksum does not match loaded map')
    route=[]
    for point in spec['points']:
        wp=road_map.get_waypoint_xodr(point['road_id'],point['lane_id'],point['s_m'])
        if wp is None or str(wp.lane_type)!='Driving':
            raise ValueError('locked point is not a real driving lane: '+str(point))
        if 'section_id' in point and wp.section_id!=point['section_id']:
            raise ValueError('locked route lane section mismatch')
        if all(k in point for k in ('x_m','y_m','z_m')):
            p=wp.transform.location
            if math.sqrt((p.x-point['x_m'])**2+(p.y-point['y_m'])**2+(p.z-point['z_m'])**2)>.10:
                raise ValueError('locked route geometry changed')
        if route:
            old=route[-1];gap=distance(old.transform.location,wp.transform.location)
            if not .01<gap<=4.5: raise ValueError('locked route points must be contiguous, <=4.5m apart')
            if abs(wp.transform.location.z-old.transform.location.z)>1.5:
                raise ValueError('locked route crosses road levels')
            # A waypoint must be reachable along the public directed topology, not
            # merely spatially close on an adjacent road or the opposite direction.
            # CARLA next(d) advances OpenDRIVE reference s, whose metric differs
            # from an offset lane's Euclidean chord (especially at tight bends).
            options=old.next(float(spec.get('sampling_step_m',2.)))
            if not any(lane_id(x)==lane_id(wp) and distance(x.transform.location,wp.transform.location)<.8
                       for x in options):
                raise ValueError('locked route contains an unconnected/reversed step')
        route.append(wp)
    return route


class RouteGeometry:
    def __init__(self,waypoints):
        if len(waypoints)<2: raise ValueError('route needs two points')
        self.points=list(waypoints);self.arcs=[0.]
        for a,b in zip(waypoints,waypoints[1:]):
            pa,pb=a.transform.location,b.transform.location
            self.arcs.append(self.arcs[-1]+math.hypot(pb.x-pa.x,pb.y-pa.y))
        self.length=self.arcs[-1]

    def at(self,s):
        if not math.isfinite(s) or s<0 or s>self.length+.001:
            raise ValueError('placement outside frozen route')
        import bisect
        return self.points[min(len(self.points)-1,bisect.bisect_left(self.arcs,s))]

    def project(self,location,hint=None,forward_segments=12):
        # s and lateral d are strictly XY Frenet coordinates, shared with the
        # planner/evaluator. Height only disambiguates candidate road levels;
        # it never changes the segment fraction, arc length or returned XY error.
        lo=0 if hint is None else max(0,hint-5)
        hi=len(self.points)-1 if hint is None else min(len(self.points)-1,hint+forward_segments)
        best=None
        for i in range(lo,hi):
            a=self.points[i].transform.location;b=self.points[i+1].transform.location
            dx,dy,dz=b.x-a.x,b.y-a.y,b.z-a.z;length2=dx*dx+dy*dy
            if length2<1e-10: continue
            q=max(0.,min(1.,((location.x-a.x)*dx+(location.y-a.y)*dy)/length2))
            ex,ey,ez=location.x-a.x-q*dx,location.y-a.y-q*dy,location.z-a.z-q*dz
            # XY distance is authoritative, including next to a grade vertex.
            # Adding z to this cost would shift a raised bounding-box center's
            # projection onto the wrong slope segment. Height only breaks exact
            # planar ties, as at an overlapping road level.
            candidate=(ex*ex+ey*ey,ez*ez,i)
            if best is None or candidate<best[0]:
                best=(candidate,self.arcs[i]+q*(self.arcs[i+1]-self.arcs[i]),
                    # CARLA world y points right; planner world y is negated.
                    (dy*ex-dx*ey)/max(1e-8,math.hypot(dx,dy)),i,math.hypot(ex,ey))
        if best is None: raise ValueError('route projection failed')
        return best[1],best[2],best[3],best[4]


def adjacent(wp,side):
    result=wp.get_left_lane() if side=='left' else wp.get_right_lane()
    if (result is None or str(result.lane_type)!='Driving'
            or abs(wrap(math.radians(result.transform.rotation.yaw-wp.transform.rotation.yaw)))>.4
            or abs(result.transform.location.z-wp.transform.location.z)>1.5):
        raise ValueError('missing same-direction adjacent lane: '+side)
    return result


def sidewalk(wp,side):
    start=wp.transform.rotation.yaw;seen=set()
    for _ in range(20):
        if wp is None or lane_id(wp) in seen: break
        seen.add(lane_id(wp))
        if str(wp.lane_type)=='Sidewalk': return wp
        same=math.cos(math.radians(wp.transform.rotation.yaw-start))>=0
        wp=wp.get_left_lane() if ((side=='left')==same) else wp.get_right_lane()
    raise ValueError('crossing requires real sidewalk on both sides')


def placement_plan(road_map,ego_route,row,carla,verified_map_hash=None):
    place=row['placement'];geometry=RouteGeometry(ego_route)
    if 'route' in place:
        route=decode_route(road_map,place['route'],verified_map_hash)
        return {'waypoint':route[0],'route':route,'end':None,'base':route[0]}
    base=geometry.at(place['s_m']);lane=place['lane']
    index=geometry.points.index(base)
    if row['type']=='pedestrian':
        if not lane.startswith('sidewalk_'):raise ValueError('pedestrian must start on sidewalk')
        side=lane.split('_')[1];start=sidewalk(base,side)
        end=sidewalk(base,'right' if side=='left' else 'left')
        if not 2.<distance(start.transform.location,end.transform.location)<35.:
            raise ValueError('invalid pedestrian crossing length')
        return {'waypoint':start,'route':None,'end':end,'base':base}
    if lane=='current':route=ego_route[index:]
    else:
        route=[]
        for point in ego_route[index:]:
            try:other=adjacent(point,lane)
            except ValueError:break
            if route and distance(route[-1].transform.location,other.transform.location)>4.5:break
            route.append(other)
    if len(route)<2:raise ValueError('actor has no declared continuous route')
    start=route[0]
    if abs(place.get('offset_m',0.))>float(start.lane_width)*.5:
        raise ValueError('actor centre lies outside requested lane')
    if row['behavior']=='cut_in':
        crossing=carla.LaneChange.Right if lane=='left' else carla.LaneChange.Left
        if not int(start.lane_change)&int(crossing):raise ValueError('cut-in initial corridor forbids crossing')
        # Full future actor trajectory is not an input to ego planning.
        required=row['speed_mps']*(row.get('duration_ticks',200)/50.+1.)+10.
        if RouteGeometry(route).length<required:raise ValueError('cut-in adjacent corridor too short')
    return {'waypoint':start,'route':route,'end':None,'base':base}


def inspect_instance(road_map,settings,carla):
    fingerprint=map_hash(road_map)
    route=decode_route(road_map,settings['route'],fingerprint)
    placements=[placement_plan(road_map,route,row,carla,fingerprint) for row in settings['actors']]
    geometry=RouteGeometry(route)
    return route,placements,{'version':'carla-locked-route-2.0','success':True,
        'map':road_map.name,'opendrive_sha256':fingerprint,'instance_hash':settings['instance_hash'],
        'route_length_m':geometry.length,'interaction_length_m':0.,'selected_spawn_index':-1,
        'candidates':[{'spawn_index':-1,'reason':'accepted'}],'rejection_counts':{},
        'selected_route':settings['route']['points'],'task':settings['route'].get('task',{}),
        'coordinate_contract':settings['route']['coordinate_contract'],
        'scope':'locked public geometry and actor placement; server physics acceptance is separate'}
