"""Read-only official-route and public-map conversion (SDK process, stdlib only).

All output coordinates use x=CARLA.x, y=-CARLA.y and heading=-CARLA.yaw.
This module never generates a replacement task, advances the world or decides
official success. Projection failure remains visible instead of jumping a lap.
"""
import math


def lane_key(waypoint):
    return None if waypoint is None else '{}|{}|{}'.format(
        waypoint.road_id, waypoint.section_id, waypoint.lane_id)


def wrap(angle):
    return (angle + math.pi) % (2. * math.pi) - math.pi


def xy(location):
    return [float(location.x), -float(location.y)]


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


class RouteAdapter:
    """Frozen official world-coordinate route; public map has no scenario oracle."""
    VERSION = 'official-route-residual-1.0'

    def __init__(self, road_map, global_plan_world_coord, config=None):
        self.map, self.config = road_map, dict(config or {})
        self.points, self.elevations, self.waypoints, self.lane_ids = [], [], [], []
        for entry in global_plan_world_coord:
            transform = entry[0] if isinstance(entry, (tuple, list)) else entry
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
        for wp in points:
            if not _driving(wp):
                continue
            location = wp.transform.location
            p = xy(location)
            if not (lowx <= p[0] <= highx and lowy <= p[1] <= highy):
                continue
            # Preserve the existing single-layer 2D driving domain. Retaining
            # both overlapping heights would let the worker union different
            # floors into one traversable surface; that needs a separate 3D
            # map contract, not merely a less selective initialization filter.
            if abs(location.z-self.height_at(p)) > float(self.config.get('same_level_tolerance_m', 1.5)):
                continue
            groups.setdefault(lane_key(wp), {})[round(wp.s, 6)] = wp
        lanes = []
        self.lane_elevations = {}
        for key in sorted(groups):
            waypoints = sorted(groups[key].values(), key=lambda w:w.s, reverse=next(iter(groups[key].values())).lane_id > 0)
            if len(waypoints) < 2:
                continue
            center, left, right, boundaries = [], [], [], []
            for wp in waypoints:
                p, yaw, width = wp.transform.location, math.radians(wp.transform.rotation.yaw), wp.lane_width
                center.append(xy(p))
                left.append([p.x+math.sin(yaw)*width/2., -p.y+math.cos(yaw)*width/2.])
                right.append([p.x-math.sin(yaw)*width/2., -p.y-math.cos(yaw)*width/2.])
            for side, coordinates in (('left',left),('right',right)):
                for i, wp in enumerate(waypoints[:-1]):
                    marking = getattr(wp, side+'_lane_marking')
                    name = str(marking.type).split('.')[-1]
                    semantics=boundary_semantics(marking.type,_permission(wp.lane_change,side),
                                                 _permission(marking.lane_change,side),wp.is_junction)
                    boundaries.append(dict(id='{}:{}:{}'.format(key,side,i), points=coordinates[i:i+2],
                        line_type=name, **semantics, source='public_waypoint_marking_and_lane_change',
                        is_junction=bool(wp.is_junction)))
            middle = waypoints[len(waypoints)//2]
            adjacent = {}
            for side in ('left','right'):
                other = getattr(middle,'get_'+side+'_lane')()
                same = _driving(other) and math.cos(math.radians(other.transform.rotation.yaw-middle.transform.rotation.yaw)) > .5
                adjacent[side] = lane_key(other) if same and lane_key(other) in groups else None
            successors = sorted(set(lane_key(wp) for wp in waypoints[-1].next(max(2.,resolution*2)) if lane_key(wp) in groups and lane_key(wp) != key))
            lanes.append(dict(id=key,center=center,polygon=left+right[::-1],width=float(middle.lane_width),
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
                if edges:
                    nearest=min(edges,key=lambda edge:project(ego_position,edge['points'])['distance_m'])
                    legal_adjacent |= nearest['crossable']
            if not legal_adjacent:
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
