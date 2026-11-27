"""Read-only official-route and public-map conversion (SDK process, stdlib only).

All output coordinates use x=CARLA.x, y=-CARLA.y and heading=-CARLA.yaw.
This module never generates a replacement task, advances the world or decides
official success. Projection failure remains visible instead of jumping a lap.
"""
import math
import io
import struct
import xml.etree.ElementTree as ET

FINISH_GATE_SCHEMA = 'ep-driver-f2d-finish-gate-1.0'


def _finish_gate(transform, endpoint_xy, terminal_lane_id):
    """Retain the public official finish plane; never infer an orientation."""
    result = dict(schema=FINISH_GATE_SCHEMA, valid=False,
        source='official_set_global_plan_dense_last_transform',
        reference_point='carla_actor_origin',
        terminal_lane_id=terminal_lane_id, official_endpoint_unchanged=True,
        official_completion_external=True)
    try:
        point, rotation = transform.location, transform.rotation
        direction = transform.get_forward_vector()
        position = [float(point.x), float(point.y), float(point.z)]
        forward = [float(direction.x), float(direction.y), float(direction.z)]
        angles = dict(yaw=float(rotation.yaw), pitch=float(rotation.pitch), roll=float(rotation.roll))
    except (AttributeError, TypeError, ValueError):
        result['reason'] = 'legacy_navigation_without_complete_transform'
        return result
    if (not all(math.isfinite(v) for v in position+forward+list(angles.values()))
            or abs(math.sqrt(sum(v*v for v in forward))-1.) > 1e-5):
        result['reason'] = 'invalid_public_finish_transform'
        return result
    if math.dist([position[0], -position[1]], endpoint_xy) > 1e-6:
        result['reason'] = 'official_finish_not_the_retained_route_endpoint'
        return result
    result.update(valid=True, position_xyz_carla=position, forward_xyz_carla=forward,
        orientation_deg=angles, position_xy=[position[0], -position[1]],
        reason='complete_unmodified_official_finish_plane')
    return result


def lane_key(waypoint):
    return None if waypoint is None else '{}|{}|{}'.format(
        waypoint.road_id, waypoint.section_id, waypoint.lane_id)


def wrap(angle):
    return (angle + math.pi) % (2. * math.pi) - math.pi


def xy(location):
    return [float(location.x), -float(location.y)]


def _nearest_boundary_crossable(point, edges):
    """Use the local marking, including restrictive transitions at a shared vertex."""
    nearest = math.inf
    allowed = False
    for edge in edges:
        points = edge['points']
        if any(math.dist(a,b) > 1e-8 for a,b in zip(points,points[1:])):
            distance = project(point, points)['distance_m']
        elif points and not edge['crossable']:
            # Distinct OpenDRIVE stations can have identical CARLA XY. Keep a
            # restrictive point, but never grant lane access from a point alone.
            distance = min(math.dist(point, p) for p in points)
        else:
            continue
        if distance < nearest:
            nearest, allowed = distance, edge['crossable']
        elif distance == nearest:
            allowed = allowed and edge['crossable']
    return allowed


def project(point, points, previous_s=None, window=None, heading=None, minimum_s=None, maximum_s=None,
            segment_indices=None, elevations=None, elevation=None):
    """Return real Euclidean residual, not only signed lateral distance."""
    if (elevations is None) != (elevation is None):
        raise ValueError('Projection elevation requires both route heights and the public query height')
    if elevations is not None and (len(elevations) != len(points)
            or not all(math.isfinite(v) for v in elevations) or not math.isfinite(elevation)):
        raise ValueError('Projection heights must be finite and match the route')
    arc, candidates = 0., []
    for index, (a, b) in enumerate(zip(points, points[1:])):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length <= 1e-8:
            continue
        end = arc + length
        if segment_indices is not None and index not in segment_indices:
            arc = end
            continue
        if previous_s is not None and window is not None and (end < previous_s-window or arc > previous_s+window):
            arc = end
            continue
        if (minimum_s is not None and end < minimum_s) or (maximum_s is not None and arc > maximum_s):
            arc = end
            continue
        fraction = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(length*length)))
        if previous_s is not None and window is not None:
            fraction=max(fraction,max(0.,(previous_s-window-arc)/length))
            fraction=min(fraction,min(1.,(previous_s+window-arc)/length))
        if minimum_s is not None:
            fraction=max(fraction,max(0.,(minimum_s-arc)/length))
        if maximum_s is not None:
            fraction=min(fraction,min(1.,(maximum_s-arc)/length))
        foot = [a[0]+fraction*dx, a[1]+fraction*dy]
        tangent = math.atan2(dy, dx)
        distance = math.hypot(point[0]-foot[0], point[1]-foot[1])
        mismatch = abs(wrap(tangent-heading)) if heading is not None else 0.
        height_error = (abs(elevation-(elevations[index]+fraction*(elevations[index+1]-elevations[index])))
                        if elevations is not None else 0.)
        # Heading is a gate, not a weight that could conceal a distant foot.
        candidates.append((distance, mismatch, abs(arc+fraction*length-(previous_s or 0.)),
                           index, arc+fraction*length, foot, tangent,
                           ((point[0]-foot[0])*(-dy)+(point[1]-foot[1])*dx)/length,
                           height_error, math.hypot(distance,height_error)))
        arc = end
    if not candidates:
        raise ValueError('Official route has no projectable segment in the current progress window')
    aligned = [row for row in candidates if row[1] <= math.pi/2.]
    row = min(aligned or candidates, key=lambda item: (round(item[9], 8), item[2]))
    result = dict(s_m=row[4], position=row[5], heading_rad=row[6], lateral_m=row[7],
                  distance_m=row[0], heading_error_rad=row[1], segment_index=row[3])
    if elevations is not None:
        result['elevation_error_m'] = row[8]
    return result


def point_in_polygon(point, polygon):
    inside = False
    x, y = point
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        cross = (x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0])
        if abs(cross) < 1e-8 and min(a[0],b[0])-1e-8 <= x <= max(a[0],b[0])+1e-8 and min(a[1],b[1])-1e-8 <= y <= max(a[1],b[1])+1e-8:
            return True
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
    return inside


def polygons_intersect(a, b):
    if any(point_in_polygon(p, b) for p in a) or any(point_in_polygon(p, a) for p in b):
        return True
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    for p, q in zip(a, a[1:]+a[:1]):
        for r, s in zip(b, b[1:]+b[:1]):
            if orient(p,q,r)*orient(p,q,s) < 0 and orient(r,s,p)*orient(r,s,q) < 0:
                return True
    return False


def _driving(wp):
    return wp is not None and str(wp.lane_type).split('.')[-1].lower() == 'driving'


def _permission(value, side):
    name = str(value).split('.')[-1].lower()
    if name in ('0','1','2','3'):
        return bool(int(name)&(2 if side=='left' else 1))
    return name == 'both' or side in name


def boundary_semantics(marking_type, waypoint_permission, marking_permission, is_junction):
    """Committed CARLA semantics: an unmarked junction edge is not solid paint."""
    name=str(marking_type).split('.')[-1].lower()
    virtual=bool(is_junction) and name in ('none','nonetype')
    known=name in ('none','nonetype','solid','solidsolid','solidbroken','brokensolid',
                   'broken','brokenbroken','bottsdots','grass','curb')
    crossable=bool(waypoint_permission and marking_permission)
    if name in ('solid','solidsolid','curb','grass') or not known:
        crossable=False
    if virtual:
        crossable=True
    return dict(crossable=crossable,kind='virtual_lane_boundary' if virtual else 'lane_boundary',
        physical_marking=name not in ('none','nonetype'),map_evidence_known=known,
        permission_source='junction_unmarked_virtual' if virtual else 'current_waypoint_and_marking')


def _distinct_lane_samples(waypoints):
    """Collapse SDK aliases of an exactly identical XY centre, in lane order.

    Dense navigation and regular map sampling can give distinct s values whose
    public float positions coincide. Evaluating their microscopic yaw change
    twice makes the left/right ring backtrack and self-intersect. This is not a
    geometric simplifier: no separated centre is moved, merged or discarded.
    A genuine turn, width/height change or marking transition is not an alias.
    """
    result,removed=[],[]
    def semantics(wp):
        return (str(wp.lane_change),bool(wp.is_junction),
                str(wp.left_lane_marking.type),str(wp.left_lane_marking.lane_change),
                str(wp.right_lane_marking.type),str(wp.right_lane_marking.lane_change))
    for waypoint in waypoints:
        if result:
            previous=result[-1]
            a,b=previous.transform.location,waypoint.transform.location
            if a.x==b.x and a.y==b.y:
                evidence=dict(lane_id=lane_key(waypoint),position_xy=xy(a),
                    kept_s_m=float(previous.s),removed_s_m=float(waypoint.s),
                    elevation_difference_m=abs(a.z-b.z),
                    heading_difference_rad=abs(wrap(math.radians(
                        previous.transform.rotation.yaw-waypoint.transform.rotation.yaw))),
                    width_difference_m=abs(previous.lane_width-waypoint.lane_width))
                # Bounds only identify numerical aliases at an exact position;
                # they never authorize a spatial tolerance or enlarge a road.
                if (abs(previous.s-waypoint.s)>1e-3 or evidence['elevation_difference_m']>1e-3
                        or evidence['heading_difference_rad']>1e-5 or evidence['width_difference_m']>1e-5
                        or semantics(previous)!=semantics(waypoint)):
                    error=ValueError('Ambiguous coincident public lane samples: '+str(evidence))
                    error.evidence=evidence
                    raise error
                removed.append(evidence)
                continue
        result.append(waypoint)
    return result,removed


def _shared_lane_edges(road_map, groups, same_level):
    """Use one public cross-section boundary for directly adjacent lanes.

    Independent float32 lane centres and differing longitudinal sample grids
    otherwise create microscopic cracks inside one continuous driving road.
    Outer edges, shoulders and separate roads are never modified here.
    """
    query=getattr(road_map,'get_waypoint_xodr',None)
    edges,audit={},[]
    if query is None:
        return edges,audit
    seen=set()
    for key,samples in groups.items():
        middle=sorted(samples.values(),key=lambda wp:wp.s)[len(samples)//2]
        for side,opposite in (('left','right'),('right','left')):
            other=getattr(middle,'get_'+side+'_lane')()
            other_key=lane_key(other)
            if (not _driving(other) or other_key not in groups or
                    (other.road_id,other.section_id)!=(middle.road_id,middle.section_id) or
                    math.cos(math.radians(other.transform.rotation.yaw-middle.transform.rotation.yaw))<.9 or
                    lane_key(getattr(other,'get_'+opposite+'_lane')())!=key):
                continue
            identity=tuple(sorted((key,other_key)))
            if identity in seen:
                continue
            seen.add(identity)
            peer=groups[other_key]
            low=max(min(wp.s for wp in samples.values()),min(wp.s for wp in peer.values()))
            high=min(max(wp.s for wp in samples.values()),max(wp.s for wp in peer.values()))
            # Rounded dictionary keys deduplicate samples; they are not valid
            # public s queries when rounding would step beyond a road exit.
            stations={}
            for wp in list(samples.values())+list(peer.values()):
                if low<=wp.s<=high:
                    stations.setdefault(round(wp.s,6),float(wp.s))
            stations=sorted(stations.values())
            shared=[]; separation=0.
            for station in stations:
                a=query(middle.road_id,middle.lane_id,station)
                b=query(other.road_id,other.lane_id,station)
                if (lane_key(a)!=key or lane_key(b)!=other_key or
                        not same_level(a) or not same_level(b) or
                        lane_key(getattr(a,'get_'+side+'_lane')())!=other_key or
                        lane_key(getattr(b,'get_'+opposite+'_lane')())!=key or
                        min(a.lane_width,b.lane_width)<=0):
                    shared=[];break
                pa,pb=xy(a.transform.location),xy(b.transform.location)
                def raw_edge(wp,edge):
                    p=xy(wp.transform.location);yaw=math.radians(wp.transform.rotation.yaw)
                    sign=1. if edge=='left' else -1.
                    return [p[0]+sign*math.sin(yaw)*wp.lane_width/2.,p[1]+sign*math.cos(yaw)*wp.lane_width/2.]
                gap=math.dist(raw_edge(a,side),raw_edge(b,opposite))
                def ulp32(value):
                    value=abs(float(value))
                    return math.ldexp(1.,-149 if value==0. else max(-149,math.frexp(value)[1]-24))
                # This bound identifies float32 cross-section disagreement;
                # it is not a point-in-road tolerance or a polygon buffer.
                precision=4.*(sum(math.hypot(ulp32(p[0]),ulp32(p[1])) for p in (pa,pb))+
                    sum(ulp32(wp.lane_width)+wp.lane_width*math.radians(ulp32(wp.transform.rotation.yaw)) for wp in (a,b)))
                if gap>precision:
                    shared=[];break
                fraction=a.lane_width/(a.lane_width+b.lane_width)
                point=[pa[i]+fraction*(pb[i]-pa[i]) for i in (0,1)]
                shared.append((station,point,a,b));separation=max(separation,gap)
            if len(shared)<2:
                continue
            edges[(key,side)]=[(s,p,a) for s,p,a,b in shared]
            edges[(other_key,opposite)]=[(s,p,b) for s,p,a,b in shared]
            audit.append(dict(lane_ids=list(identity),sample_count=len(shared),
                maximum_original_boundary_gap_m=separation,
                source='same_road_section_direct_adjacent_public_s_half_width_interpolation'))
    return edges,audit


def _public_section_bounds(road_map, required_roads):
    """Public OpenDRIVE section limits; no native lane traversal is needed."""
    export=getattr(road_map,'to_opendrive',None)
    if export is None:
        return {}
    result={}
    for _,road in ET.iterparse(io.StringIO(export()),events=('end',)):
        if road.tag!='road':
            continue
        identity=int(road.attrib['id']);end=float(road.attrib['length'])
        if identity in required_roads:
            sections=road.findall('lanes/laneSection')
            for index,section in enumerate(sections):
                start=float(section.attrib['s'])
                stop=float(sections[index+1].attrib['s']) if index+1<len(sections) else end
                result[(identity,index)]=(start,stop)
        road.clear()
    return result


def _interior_public_station(boundary, increasing):
    """An inward float32 s remains inside after the SDK's argument narrowing."""
    packed=struct.pack('>f',boundary)
    value=struct.unpack('>f',packed)[0]
    bits=struct.unpack('>I',packed)[0]
    if (increasing and value<=boundary) or (not increasing and value>=boundary):
        bits+=1 if increasing else -1
    return float(struct.unpack('>f',struct.pack('>I',bits))[0])


class RouteAdapter:
    """Frozen official world-coordinate route; public map has no scenario oracle."""
    VERSION = 'official-route-residual-1.0'

    def __init__(self, road_map, global_plan_world_coord, config=None):
        self.map, self.config = road_map, dict(config or {})
        self.points, self.elevations, self.waypoints, self.lane_ids = [], [], [], []
        last_transform = None
        for entry in global_plan_world_coord:
            transform = entry[0] if isinstance(entry, (tuple, list)) else entry
            last_transform = transform
            location = getattr(transform, 'location', transform)
            if not all(hasattr(location, key) for key in ('x', 'y', 'z')):
                raise ValueError('Expected the official world-coordinate plan, not the GPS plan')
            point = xy(location)
            if self.points and math.dist(point, self.points[-1]) < .05:
                continue
            self.points.append(point)
            self.elevations.append(float(location.z))
            wp = road_map.get_waypoint(location)
            self.waypoints.append(wp)
            key = lane_key(wp)
            if key is not None and (not self.lane_ids or self.lane_ids[-1] != key):
                self.lane_ids.append(key)
        if len(self.points) < 2:
            raise ValueError('Official world-coordinate plan must contain two distinct points')
        self.finish_gate = _finish_gate(last_transform, self.points[-1], lane_key(self.waypoints[-1]))
        self.maximum_gap=max(math.dist(a,b) for a,b in zip(self.points,self.points[1:]))
        if self.maximum_gap>float(self.config.get('maximum_official_route_gap_m',8.)):
            raise ValueError('Official route is sparse; retain the evaluator dense plan before base-agent downsampling')
        self.length = sum(math.dist(a,b) for a,b in zip(self.points,self.points[1:]))
        self.lanes = self._build_map()
        self.drivable_polygons = [lane['polygon'] for lane in self.lanes]
        self.previous_s, self.previous_position, self.previous_time = None, None, None
        self.previous_frame, self.last_projection = None, None

    def height_at(self, point, z_hint=None):
        # XY alone cannot distinguish the same route's bridge/underpass visits.
        # A public map/query height selects the branch; it never changes it.
        result = project(point, self.points, elevations=self.elevations if z_hint is not None else None,
                         elevation=z_hint)
        i = result['segment_index']
        length = math.dist(self.points[i], self.points[i+1])
        fraction = math.dist(self.points[i], result['position'])/length
        return self.elevations[i]+fraction*(self.elevations[i+1]-self.elevations[i])

    def _build_map(self):
        resolution = float(self.config.get('map_resolution_m', 2.))
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError('map_resolution_m must be positive')
        points = list(self.map.generate_waypoints(resolution)) + self.waypoints
        for a, b in self.map.get_topology():
            points.extend((a,b))
        margin = float(self.config.get('map_corridor_margin_m', 100.))
        lowx, highx = min(p[0] for p in self.points)-margin, max(p[0] for p in self.points)+margin
        lowy, highy = min(p[1] for p in self.points)-margin, max(p[1] for p in self.points)+margin
        groups = {}
        def retain(wp):
            if not _driving(wp):
                return
            location = wp.transform.location
            p = xy(location)
            if not (lowx <= p[0] <= highx and lowy <= p[1] <= highy):
                return
            # Preserve the existing single-layer 2D driving domain. Retaining
            # both overlapping heights would let the worker union different
            # floors into one traversable surface; that needs a separate 3D
            # map contract, not merely a less selective initialization filter.
            if abs(location.z-self.height_at(p)) > float(self.config.get('same_level_tolerance_m', 1.5)):
                return
            groups.setdefault(lane_key(wp), {})[round(wp.s, 6)] = wp
        for wp in points:
            retain(wp)
        # The SDK topology links a lane entry to its successor's entry. Its
        # pair need not include the current lane's own exit, and uniform map
        # samples stop short of that exit. Retain public lane-end geometry so
        # the resulting road does not acquire a false gap before a junction.
        # No point is extrapolated, widened, or joined to a different lane.
        lane_ends = []
        section_bounds=_public_section_bounds(self.map,{next(iter(r.values())).road_id for r in groups.values()})
        query=getattr(self.map,'get_waypoint_xodr',None)
        for key, samples in list(groups.items()):
            end = sorted(samples.values(), key=lambda wp: wp.s,
                         reverse=next(iter(samples.values())).lane_id > 0)[-1]
            bounds=section_bounds.get((end.road_id,end.section_id))
            if bounds is None or query is None:
                continue
            start,stop=bounds
            # The Python SDK narrows s to float32. A double infinitesimally
            # inside may round beyond the section and return no waypoint.
            # Choose the adjacent representable interior s, never extrapolate.
            station=_interior_public_station(start,True) if end.lane_id>0 else _interior_public_station(stop,False)
            wp=query(end.road_id,end.lane_id,station)
            if lane_key(wp)==key:
                retained = round(wp.s, 6) not in samples
                retain(wp)
                if retained and round(wp.s, 6) in samples:
                    lane_ends.append(dict(lane_id=key, s_m=float(wp.s),
                                          position_xy=xy(wp.transform.location)))
        lanes = []
        self.lane_elevations = {}
        self.lane_junctions = {}
        shared_edges,shared_audit=_shared_lane_edges(self.map,groups,lambda wp:
            abs(wp.transform.location.z-self.height_at(xy(wp.transform.location)))<=
            float(self.config.get('same_level_tolerance_m',1.5)))
        self.map_sample_audit=dict(policy='exact_xy_adjacent_numerical_alias_only;first_sample_retained',
            no_spatial_merging=True,official_navigation_unchanged=True,removed_samples=[],
            lane_end_samples=lane_ends, lane_end_source='public_opendrive_section_limit_get_waypoint_xodr',
            shared_internal_boundaries=shared_audit)
        for key in sorted(groups):
            waypoints = sorted(groups[key].values(), key=lambda w:w.s, reverse=next(iter(groups[key].values())).lane_id > 0)
            waypoints,removed=_distinct_lane_samples(waypoints)
            self.map_sample_audit['removed_samples'].extend(removed)
            if len(waypoints) < 2:
                continue
            center, left, right, boundaries = [], [], [], []
            for wp in waypoints:
                p, yaw, width = wp.transform.location, math.radians(wp.transform.rotation.yaw), wp.lane_width
                center.append(xy(p))
                left.append([p.x+math.sin(yaw)*width/2., -p.y+math.cos(yaw)*width/2.])
                right.append([p.x-math.sin(yaw)*width/2., -p.y-math.cos(yaw)*width/2.])
            edges={}
            for side,coordinates in (('left',left),('right',right)):
                records=[(float(wp.s),point,wp) for wp,point in zip(waypoints,coordinates)]
                shared=shared_edges.get((key,side))
                if shared:
                    low,high=shared[0][0],shared[-1][0]
                    records=[r for r in records if not low<=r[0]<=high]+shared
                    records.sort(key=lambda row:row[0],reverse=waypoints[0].lane_id>0)
                edges[side]=[r[1] for r in records]
                for i,(_,_,wp) in enumerate(records[:-1]):
                    marking = getattr(wp, side+'_lane_marking')
                    name = str(marking.type).split('.')[-1]
                    semantics=boundary_semantics(marking.type,_permission(wp.lane_change,side),
                                                 _permission(marking.lane_change,side),wp.is_junction)
                    boundaries.append(dict(id='{}:{}:{}'.format(key,side,i), points=edges[side][i:i+2],
                        line_type=name, **semantics, source='public_waypoint_marking_and_lane_change',
                        is_junction=bool(wp.is_junction)))
            middle = waypoints[len(waypoints)//2]
            if self.config.get('teaching_junction_overlap', False) and middle.is_junction:
                self.lane_junctions[key] = getattr(middle, 'junction_id', None)
            adjacent = {}
            for side in ('left','right'):
                other = getattr(middle,'get_'+side+'_lane')()
                same = _driving(other) and math.cos(math.radians(other.transform.rotation.yaw-middle.transform.rotation.yaw)) > .5
                adjacent[side] = lane_key(other) if same and lane_key(other) in groups else None
            successors = sorted(set(lane_key(wp) for wp in waypoints[-1].next(max(2.,resolution*2)) if lane_key(wp) in groups and lane_key(wp) != key))
            lanes.append(dict(id=key,center=center,polygon=edges['left']+edges['right'][::-1],width=float(middle.lane_width),
                speed_limit=float(self.config.get('map_speed_cap_mps',30.)),successors=successors,
                left=adjacent['left'],right=adjacent['right'],priority=0,boundaries=boundaries))
            self.lane_elevations[key] = [float(wp.transform.location.z) for wp in waypoints]
        missing = set(self.lane_ids)-{lane['id'] for lane in lanes}
        if missing:
            layered = {lane_key(wp) for wp in self.waypoints if lane_key(wp) in missing and
                       abs(wp.transform.location.z-self.height_at(xy(wp.transform.location))) >
                       float(self.config.get('same_level_tolerance_m',1.5)) and
                       abs(wp.transform.location.z-self.height_at(xy(wp.transform.location),wp.transform.location.z)) <=
                       float(self.config.get('same_level_tolerance_m',1.5))}
            if layered:
                raise ValueError('unsupported_multilevel_2d_navigation: official route height is ambiguous near XY; '
                                 'different floors must not be unioned into one driving domain: '+
                                 ','.join(sorted(layered)))
            raise ValueError('Public map conversion is missing official route lanes: '+','.join(sorted(missing)))
        return lanes

    def _junction_overlap(self, point, projection, observed):
        """Resolve a local connector label ambiguity using the already observed trajectory."""
        last = self.last_projection or {}
        junction = getattr(self, 'lane_junctions', {}).get(observed)
        if junction is None or junction < 0 or not last.get('valid') or projection['heading_error_rad'] > math.radians(10):
            return None
        lanes = {lane['id']: lane for lane in self.lanes}
        reported = lanes[observed]
        i = projection['segment_index']
        for index in (i, i+1):
            intended = lane_key(self.waypoints[index])
            if self.lane_junctions.get(intended) != junction:
                continue
            planned = lanes[intended]
            width = min(planned['width'], reported['width'])
            tracking_tolerance = min(.3, width / 10.)
            if (math.dist(point, self.previous_position) > width or projection['distance_m'] > tracking_tolerance
                    or not all(point_in_polygon(point, lane['polygon']) for lane in (planned, reported))):
                continue
            p = project(point, planned['center'])
            r = project(point, reported['center'])
            if (p['distance_m'] > tracking_tolerance
                    or abs(self.lane_height_at(intended, point)-self.lane_height_at(observed, point)) > .25):
                continue
            # Check the route occurrence here, not a lane elsewhere along a loop.
            first = end = index
            while first > 0 and lane_key(self.waypoints[first-1]) == intended:
                first -= 1
            while end+1 < len(self.waypoints) and lane_key(self.waypoints[end+1]) == intended:
                end += 1
            incoming = lane_key(self.waypoints[first-1]) if first else None
            outgoing = lane_key(self.waypoints[end+1]) if end+1 < len(self.waypoints) else None
            previous_lane = (last.get('junction_overlap') or {}).get('route_lane_id', last.get('observed_lane_id'))
            turn = abs(wrap(p['heading_rad']-r['heading_rad']))
            incoming_successors = lanes.get(incoming, {}).get('successors', ())
            if (previous_lane in (incoming, intended) and intended in incoming_successors
                    and observed in incoming_successors and turn <= math.radians(20)
                    and max(p['s_m'],r['s_m']) <= width and math.dist(planned['center'][0],reported['center'][0]) <= .1):
                kind = 'common_entry'
            elif previous_lane != intended:
                continue
            elif (outgoing in planned['successors'] and outgoing in reported['successors'] and turn <= math.radians(20)
                    and max(math.dist(point,lane['center'][-1]) for lane in (planned,reported)) <= width
                    and math.dist(planned['center'][-1],reported['center'][-1]) <= .1):
                kind = 'common_exit'
            elif math.radians(45) <= turn <= math.radians(135):
                # Across the crossing connector, physical heading follows the
                # intended path. A vehicle taking the other path fails the
                # heading/center checks, even inside the overlapping polygons.
                kind = 'transverse_crossing'
            else:
                continue
            return dict(kind=kind,junction_id=junction,route_lane_id=intended,observed_lane_id=observed,
                        route_center_distance_m=p['distance_m'],connector_heading_difference_rad=turn)
        return None

    def update(self, ego_position, ego_heading, frame, timestamp, speed, current_lane_id=None):
        if self.previous_frame == frame:
            return dict(self.last_projection)
        if self.previous_frame is not None and frame < self.previous_frame:
            raise ValueError('Official route frame moved backwards')
        elapsed = max(0.,timestamp-self.previous_time) if self.previous_time is not None else 0.
        travel = math.dist(ego_position,self.previous_position) if self.previous_position is not None else 0.
        reachable = max(2.,abs(speed)*elapsed+3.*elapsed*elapsed+1.)
        window = 25. if self.previous_s is None else max(6.,2.*min(travel,reachable)+2.)
        result = project(ego_position,self.points,0. if self.previous_s is None else self.previous_s,window,ego_heading)
        result.update(valid=True,reason='accepted',version=self.VERSION)
        teaching_overlap = self.config.get('teaching_junction_overlap', False)
        if teaching_overlap:
            result['observed_lane_id'] = current_lane_id
        maximum = float(self.config.get('route_projection_maximum_distance_m', 6.))
        if result['distance_m'] > maximum:
            result.update(valid=False,reason='distance_exceeded')
        elif result['heading_error_rad'] > math.pi/2.:
            result.update(valid=False,reason='direction_mismatch')
        elif self.previous_position is not None and travel > reachable:
            result.update(valid=False,reason='observed_displacement_exceeds_reachable_distance')
        elif self.previous_s is not None and result['s_m'] < self.previous_s-2.:
            result.update(valid=False,reason='backward_progress')
        if current_lane_id is not None and current_lane_id not in self.lane_ids:
            legal_adjacent = False
            for lane in self.lanes:
                if lane['id'] not in self.lane_ids:
                    continue
                side = 'left' if lane['left']==current_lane_id else 'right' if lane['right']==current_lane_id else None
                if side is None:
                    continue
                edges=[edge for edge in lane['boundaries'] if ':'+side+':' in edge['id']]
                legal_adjacent |= _nearest_boundary_crossable(ego_position,edges)
            if not legal_adjacent:
                overlap = self._junction_overlap(ego_position, result, current_lane_id) if teaching_overlap and result['valid'] else None
                if overlap:
                    result.update(reason='overlapping_junction_connectors', junction_overlap=overlap)
                elif not teaching_overlap or result['valid']:
                    result.update(valid=False,reason='current_lane_not_navigation_or_legal_adjacent')
        if result['valid']:
            self.previous_s = result['s_m']
        result['trusted_progress_m'] = self.previous_s
        result['route_length_m'] = self.length
        result['official_completion_is_external'] = True
        self.previous_position, self.previous_time, self.previous_frame = list(ego_position),timestamp,frame
        self.last_projection = dict(result)
        return result

    def lane_height_at(self, lane_id, point, z_hint=None):
        lane = next(item for item in self.lanes if item['id'] == lane_id)
        heights = self.lane_elevations[lane_id]
        result = project(point,lane['center'],elevations=heights if z_hint is not None else None,
                         elevation=z_hint)
        i = result['segment_index']
        fraction = math.dist(lane['center'][i],result['position'])/math.dist(lane['center'][i],lane['center'][i+1])
        return heights[i]+fraction*(heights[i+1]-heights[i])

    def stop_projection(self, position, lane_id, current_lane_id, elevation_m=None):
        """Associate an upcoming public stop with its actual controlled lane."""
        if self.previous_s is None or (lane_id not in self.lane_ids and lane_id != current_lane_id):
            return None
        segments = None
        if lane_id != current_lane_id:
            # Membership anywhere in the route is insufficient: a parallel
            # lane used only later must not impose its current stop on this
            # lane. A dense transition may have one endpoint in either lane.
            segments = {i for i in range(len(self.points)-1)
                        if lane_id in (lane_key(self.waypoints[i]),lane_key(self.waypoints[i+1]))}
            if not segments:
                return None
        try:
            candidates = project(position,self.points,self.previous_s,None,
                minimum_s=max(0.,self.previous_s-.1),segment_indices=segments,
                elevations=self.elevations if elevation_m is not None else None,elevation=elevation_m)
        except ValueError:
            return None
        if (elevation_m is not None and candidates['elevation_error_m'] >
                float(self.config.get('same_level_tolerance_m',1.5))):
            return None
        lane = next((item for item in self.lanes if item['id'] == lane_id),None)
        if lane is None:
            return None
        lane_projection = project(position,lane['center'])
        if abs(wrap(lane_projection['heading_rad']-candidates['heading_rad'])) > .7:
            return None
        if candidates['distance_m'] > max(3.,lane['width']*1.5) or candidates['s_m'] < self.previous_s-.1:
            return None
        dx,dy=position[0]-candidates['position'][0],position[1]-candidates['position'][1]
        if abs(dx*math.cos(candidates['heading_rad'])+dy*math.sin(candidates['heading_rad'])) > .5:
            # A behind/remote stop clipped to an endpoint is not a stop line at
            # that endpoint, even when its signed lateral residual is zero.
            return None
        return float(candidates['s_m'])
