"""Owned actors and causal, auditable events for independent demonstrations.

The environment may schedule its own events. The teacher never receives this
object, role labels, event timers, actor plans or release decisions. Completion
here describes the observed interaction, not successful/official driving.
"""
import math

from .demo_routes import station_waypoint, crossing_drive_path, _stop_controls_route
from .route import xy, project, point_in_polygon, wrap

WALKABLE_SURFACES = frozenset(('Roads','RoadLines','Sidewalks','Ground','Terrain'))


def _hull(points):
    """CARLA boxes return eight 3D vertices, not an ordered 2D polygon."""
    values = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(values) < 3 or any(not math.isfinite(v) for p in values for v in p):
        raise ValueError('Actor has no finite planar physical body')
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    sides = []
    for ordered in (values, list(reversed(values))):
        side = []
        for point in ordered:
            while len(side) > 1 and cross(side[-2], side[-1], point) <= 0.:
                side.pop()
            side.append(point)
        sides.extend(side[:-1])
    if len(sides) < 3:
        raise ValueError('Actor footprint is degenerate')
    return sides


def _body(actor):
    return _hull([xy(v) for v in actor.bounding_box.get_world_vertices(actor.get_transform())])


def _point_segment(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    norm = dx*dx+dy*dy
    t = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/norm)) if norm else 0.
    return math.hypot(point[0]-a[0]-t*dx, point[1]-a[1]-t*dy)


def _intersection(a, b, c, d):
    dx, dy, ex, ey = b[0]-a[0], b[1]-a[1], d[0]-c[0], d[1]-c[1]
    determinant = dx*ey-dy*ex
    if abs(determinant) < 1e-10:
        return None
    px, py = c[0]-a[0], c[1]-a[1]
    t, u = (px*ey-py*ex)/determinant, (px*dy-py*dx)/determinant
    return (a[0]+t*dx, a[1]+t*dy) if -1e-8 <= t <= 1.+1e-8 and -1e-8 <= u <= 1.+1e-8 else None


def _segment_distance(a, b, c, d):
    if _intersection(a, b, c, d) is not None:
        return 0.
    return min(_point_segment(p, x, y) for p, x, y in ((a,c,d),(b,c,d),(c,a,b),(d,a,b)))


def _polygon_distance(a, b):
    if point_in_polygon(a[0], b) or point_in_polygon(b[0], a):
        return 0.
    return min(_segment_distance(p,q,r,s) for p,q in zip(a,a[1:]+a[:1]) for r,s in zip(b,b[1:]+b[:1]))


def _path_body_distance(path, polygon):
    if any(point_in_polygon(p, polygon) for p in path):
        return 0.
    return min(_segment_distance(a,b,c,d) for a,b in zip(path,path[1:])
               for c,d in zip(polygon,polygon[1:]+polygon[:1]))


def _at(path, station):
    arc = 0.
    for a, b in zip(path, path[1:]):
        length = math.dist(a, b)
        if arc+length >= station:
            t = max(0., min(1., (station-arc)/length))
            return [a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])], math.atan2(b[1]-a[1], b[0]-a[0])
        arc += length
    return list(path[-1]), math.atan2(path[-1][1]-path[-2][1], path[-1][0]-path[-2][0])


def _slice_interval(polygon, center, heading):
    """Actual body intersection with one common road-normal cross-section."""
    c, s = math.cos(heading), math.sin(heading)
    local = [((p[0]-center[0])*c+(p[1]-center[1])*s,
              -(p[0]-center[0])*s+(p[1]-center[1])*c) for p in polygon]
    values = []
    for a, b in zip(local, local[1:]+local[:1]):
        if abs(a[0]) <= 1e-7:
            values.append(a[1])
        if a[0]*b[0] < 0.:
            t = -a[0]/(b[0]-a[0])
            values.append(a[1]+t*(b[1]-a[1]))
    return (min(values), max(values)) if values else None


def _adjacent_lanes(wp, *, right):
    """Traverse one geographic side even when the adjacent lane faces the opposite way."""
    cursor = wp
    yaw = wp.transform.rotation.yaw
    visited = {(wp.road_id, wp.section_id, wp.lane_id)}
    while True:
        same_direction = math.cos(math.radians(cursor.transform.rotation.yaw-yaw)) >= 0.
        cursor = cursor.get_right_lane() if right == same_direction else cursor.get_left_lane()
        if cursor is None:
            return
        key = (cursor.road_id, cursor.section_id, cursor.lane_id)
        if key in visited:
            return
        visited.add(key)
        yield cursor


class NoWalkableCrossingSurface(RuntimeError):
    """Terrain ruled out the roadside starts for this occluded crossing."""


def _walker_starts(wp, lateral, crossing, *, probe_edges=False):
    """Same route station. Keep crossing starts on the right and road walkers in their lane."""
    origin = wp.transform.location
    if not crossing:
        return [(lateral, origin.z, str(wp.lane_type).split('.')[-1])]
    yaw = math.radians(wp.transform.rotation.yaw)
    bands = []
    outer_edge = -wp.lane_width/2.
    # A roadside actor may need to start just beyond a whole neighboring
    # travel lane (1075). Bound the displacement to one lane plus kerb clearance.
    max_side_shift = wp.lane_width+.4
    for neighbor in _adjacent_lanes(wp, right=True):
        p = neighbor.transform.location
        center = (p.x-origin.x)*math.sin(yaw)-(p.y-origin.y)*math.cos(yaw)
        half = neighbor.lane_width/2.
        kind = str(neighbor.lane_type).split('.')[-1]
        if center < 0. and kind in ('Driving', 'Parking', 'Shoulder', 'Sidewalk', 'Biking', 'Curb'):
            outer_edge = min(outer_edge, center-half)
        if kind in ('Sidewalk', 'Shoulder') and half > .4 and center < 0.:
            # A centre on a kerb/outer edge can pass a point-on-lane check while
            # the pedestrian capsule is already inside adjacent static geometry.
            inside = min(center+half-.4, max(center-half+.4, lateral))
            offsets = [inside, center]
            if probe_edges:
                offsets.extend((center+half-.4, center-half+.4))
            for offset in offsets:
                if abs(offset-lateral) <= max_side_shift and offset < -wp.lane_width/2.:
                    bands.append((offset, p.z, kind))
    bands.sort(key=lambda item: abs(item[0]-lateral))
    # Unmapped starts need a real surface beyond the outer edge. An adjacent
    # driving lane or the edge of a narrow shoulder is not a roadside fallback.
    verge = outer_edge-.45
    if lateral <= verge:
        bands.insert(0, (lateral, origin.z, None))
    for offset in (min(lateral-.5, verge), lateral+.5, verge):
        if offset <= verge and abs(offset-lateral) <= max_side_shift:
            bands.append((offset, origin.z, None))
    poses = []
    for candidate in bands:
        if all(abs(candidate[0]-saved[0]) > .05 for saved in poses):
            poses.append(candidate)
    return poses


class DemoEvents:
    def __init__(self, world, waypoints, spec, carla):
        self.world, self.waypoints, self.spec, self.carla = world, waypoints, spec, carla
        self.route = [xy(wp.transform.location) for wp in waypoints]
        self.teaching_path = self.route
        self.ego_width = 2.
        self.actors, self.rows = [], []
        self.activated_at = self.completion_progress_m = None
        self.completed, self.active = False, False
        self.stationary_s = 0.
        self.stop_observed, self.stop_rule_seen = False, False
        self.stop_stationary_s = self.stop_max_stationary_s = 0.
        self.stop_encounters = {}
        self.last_time = None
        self.geometry_audit = None
        self.walker_placements = []

    def _cross_branch(self, desired_station):
        result = crossing_drive_path(self.waypoints, desired_station)
        if result is None:
            raise RuntimeError('No same-level driving branch with a real perpendicular route crossing and exit clearance')
        return result

    def _crossing_left_limit(self, waypoint):
        origin, yaw = waypoint.transform.location, math.radians(waypoint.transform.rotation.yaw)
        limit = waypoint.lane_width/2.
        for cursor in _adjacent_lanes(waypoint, right=False):
            if str(cursor.lane_type).split('.')[-1] != 'Driving':
                break
            p = cursor.transform.location
            left = (p.x-origin.x)*math.sin(yaw)-(p.y-origin.y)*math.cos(yaw)
            limit = max(limit,left+cursor.lane_width/2.)
        return limit

    def _try_walker_station(self, blueprint, wp, lateral, rotation, config, station, occluded):
        origin = wp.transform.location
        yaw = math.radians(wp.transform.rotation.yaw)
        crossing = config['role'] == 'crossing_actor'
        attempts = []
        self.walker_placements.append(dict(id=config['id'], blueprint=blueprint.id,
            station_m=station, designed_station_m=config['s_offset_m'], requested_lateral_m=lateral, attempts=attempts))
        for offset, road_z, mapped in _walker_starts(wp, lateral, crossing, probe_edges=occluded):
            x, y = origin.x+offset*math.sin(yaw), origin.y-offset*math.cos(yaw)
            location = self.carla.Location(x=x, y=y, z=road_z+1.8)
            row = dict(lateral_m=offset, mapped_surface=mapped, xy_carla_m=[x,y], spawns=[])
            attempts.append(row)
            hit = self.world.ground_projection(location, 3.6)
            # A ray without a same-level, global-XY hit cannot establish a
            # surface outside mapped lanes (e.g. a large-map tile origin shift).
            if hit is not None and math.hypot(hit.location.x-x, hit.location.y-y) <= .15:
                label = str(hit.label).split('.')[-1]
                row['ray_label'] = label
                row['ray_z_m'] = hit.location.z
                # In Town13 the upper ray can start inside a vegetation mesh.
                # Resolve the walking surface independently beneath that point.
                if (occluded and mapped in ('Shoulder','Sidewalk') and label == 'Vegetation'
                        and abs(hit.location.z-location.z) <= .1):
                    low = self.carla.Location(x=x, y=y, z=road_z+.45)
                    lower = self.world.ground_projection(low, .9)
                    if (lower is None or not math.hypot(lower.location.x-x,lower.location.y-y) <= .15
                            or not -.45 <= lower.location.z-road_z <= .46):
                        row['rejected'] = 'near-ground sidewalk probe unresolved'
                        continue
                    label = str(lower.label).split('.')[-1]
                    hit = lower
                    row.update(lower_ray_label=label, lower_ray_z_m=hit.location.z)
                    if label in WALKABLE_SURFACES and hit.location.z >= low.z-.02:
                        row['rejected'] = 'near-ground hit is at the ray origin'
                        continue
                if not math.isfinite(hit.location.z) or abs(hit.location.z-road_z) > 1.8:
                    row['rejected'] = 'surface at different elevation'
                    continue
                if label not in WALKABLE_SURFACES:
                    row['rejected'] = 'surface is not walkable'
                    continue
                ground_z = hit.location.z
                row['height_source'] = 'ground_projection'
            elif mapped is not None and not occluded:
                ground_z = road_z
                row['height_source'] = 'mapped_surface'
            else:
                row['rejected'] = 'no measured roadside surface'
                continue
            row['ground_z_m'] = ground_z
            # Spawn transforms refer to the walker origin. Keep the same XY
            # and try a bounded capsule clearance if the engine reports overlap.
            for lift in (.9, 1.15):
                pos = self.carla.Location(x=x, y=y, z=ground_z+lift)
                attempt = dict(z_m=pos.z)
                row['spawns'].append(attempt)
                try:
                    actor = self.world.spawn_actor(blueprint, self.carla.Transform(pos, rotation))
                except RuntimeError as exc:
                    attempt['error'] = str(exc)
                    if 'collision' not in str(exc).lower():
                        raise
                    continue
                attempt['actor_id'] = actor.id
                return actor
        return None

    def spawn(self):
        library = self.world.get_blueprint_library()
        actors = self.spec['actor_specs']
        occluded = self.spec.get('family') == 'PedestrianOtherBlocker'
        shift = 0.
        walker = None
        if occluded:
            # Find the actual walker start first, then place its paired occluder
            # at the corresponding station on this same short route.
            actors = sorted(actors, key=lambda a: a['role'] != 'crossing_actor')
        for config in actors:
            blueprint = None
            for name in config['blueprint_candidates']:
                values = sorted(library.filter(name), key=lambda item:item.id)
                if values:
                    blueprint = values[0]
                    break
            if blueprint is None:
                raise RuntimeError('Missing required standard asset for '+config['id'])
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible','false')
            station = float(config['s_offset_m'])+shift
            wp = original = station_waypoint(self.waypoints,station)
            for _ in range(abs(config['lane_offset'])):
                wp = wp.get_left_lane() if config['lane_offset'] > 0 else wp.get_right_lane()
                if wp is None or str(wp.lane_type).split('.')[-1] != 'Driving':
                    raise RuntimeError('Required physical actor driving lane does not exist: '+config['id'])
            pose, lateral, role = wp.transform,float(config['lateral_offset_m']),config['role']
            cross_path, branch_station, interaction_station = None,None,station
            stop_sign_id = None
            if role == 'roadside_crowd':
                sidewalk, visited = original.get_right_lane(),set()
                while sidewalk and str(sidewalk.lane_type).split('.')[-1] != 'Sidewalk':
                    identity = (sidewalk.road_id,sidewalk.section_id,sidewalk.lane_id)
                    if identity in visited:
                        sidewalk = None
                        break
                    visited.add(identity)
                    sidewalk = sidewalk.get_right_lane()
                if sidewalk is None:
                    raise RuntimeError('Crowd requires a real right sidewalk')
                wp,pose,lateral = sidewalk,sidewalk.transform,0.
            if role == 'stop_sign_visual_occluder':
                signs = [sign for sign in self.world.get_actors().filter('traffic.stop')
                         if _stop_controls_route(sign,self.waypoints,target_station=config['s_offset_m'])]
                if not signs:
                    raise RuntimeError('No real stop controlling this development route')
                sign = min(signs,key=lambda actor:min(actor.get_location().distance(w.transform.location) for w in self.waypoints))
                stop_sign_id = sign.id
                sign_pose = sign.get_transform()
                nearest = min(self.waypoints,key=lambda item:item.transform.location.distance(sign_pose.location))
                lane = nearest.transform.location
                dx,dy = lane.x-sign_pose.location.x,lane.y-sign_pose.location.y
                norm = math.hypot(dx,dy)
                if norm < 2.:
                    raise RuntimeError('No safe roadside placement for the stop-sign occluder')
                pose = self.carla.Transform(self.carla.Location(x=sign_pose.location.x+.7*dx/norm,
                    y=sign_pose.location.y+.7*dy/norm,z=sign_pose.location.z),
                    self.carla.Rotation(yaw=sign_pose.rotation.yaw,pitch=sign_pose.rotation.pitch,roll=sign_pose.rotation.roll))
                lateral = 0.
                interaction_station = project(xy(nearest.transform.location),self.route)['s_m']
            if role == 'cross_traffic_vehicle':
                cross_path,interaction_station,branch_station = self._cross_branch(config['s_offset_m'])
                pose,lateral = cross_path[0].transform,0.
            yaw = math.radians(pose.rotation.yaw)
            if occluded and role == 'roadside_occluder':
                # Put the static prop between the approaching vehicle and the
                # measured pedestrian start, even for a kerb-side alternative.
                trigger = next(t['distance_m'] for t in self.spec['trigger_specs'] if 'distance_m' in t)
                observer,_ = _at(self.route, interaction_station-trigger)
                center = xy(pose.location)
                target = xy(walker.get_location())
                tangent = (math.cos(yaw), -math.sin(yaw))
                ray = (target[0]-observer[0],target[1]-observer[1])
                along = ray[0]*tangent[0]+ray[1]*tangent[1]
                along_center = (center[0]-observer[0])*tangent[0]+(center[1]-observer[1])*tangent[1]
                if not 0. < along_center < along:
                    raise RuntimeError('Roadside blocker must be ahead of the viewpoint and before the pedestrian')
                t = along_center/along
                lateral = (observer[0]+t*ray[0]-center[0])*math.sin(yaw)+(observer[1]+t*ray[1]-center[1])*math.cos(yaw)
            location = self.carla.Location(x=pose.location.x+lateral*math.sin(yaw),
                y=pose.location.y-lateral*math.cos(yaw),z=pose.location.z+.35)
            # A public cross branch supplies the actual actor heading. Applying
            # the catalog's generic +90 offset again would point it off-road.
            rotation = self.carla.Rotation(yaw=pose.rotation.yaw+(0. if cross_path else config['yaw_offset_deg']))
            if config['actor_type'] == 'walker':
                actor = None
                for delta in ((0., 2., -2., 4., -4., 6., -6., 8., -8.) if occluded else (0.,)):
                    if occluded:
                        shift = delta
                        interaction_station = float(config['s_offset_m'])+shift
                        wp = original = station_waypoint(self.waypoints,interaction_station)
                    if role == 'crossing_actor':
                        rotation = self.carla.Rotation(yaw=wp.transform.rotation.yaw-90.)
                    actor = self._try_walker_station(blueprint, wp, lateral, rotation, config, interaction_station, occluded)
                    if actor is not None:
                        walker = actor
                        break
                if actor is None:
                    message = 'Required walker spawn failed: %s (%s); see capture_failure.json walker_placements' % (
                        config['id'], blueprint.id)
                    attempted = [a for record in self.walker_placements for a in record['attempts']]
                    mapped = [a for a in attempted if a['mapped_surface'] in ('Sidewalk','Shoulder')]
                    if (occluded and mapped and not any(a['spawns'] for a in attempted)
                            and all(a.get('rejected') == 'surface is not walkable' or
                                    (a.get('ray_label') == 'Vegetation' and a.get('rejected') in
                                     ('near-ground sidewalk probe unresolved','near-ground hit is at the ray origin'))
                                    for a in mapped)):
                        raise NoWalkableCrossingSurface(message)
                    raise RuntimeError(message)
            else:
                actor = self.world.try_spawn_actor(blueprint,self.carla.Transform(location,rotation))
            if actor is None:
                raise RuntimeError('Required actor spawn failed: '+config['id'])
            self.actors.append(actor)
            if (self.spec.get('family')=='Animals' and self.spec.get('map')=='Town13'
                    and role in ('crossing_actor','slow_lead')
                    and actor.type_id!='walker.animal.1009'):
                raise RuntimeError('Town13 Animals requires the actual 1075 walker.animal.1009 asset; no pedestrian substitute')
            if config['actor_type'] in ('obstacle','visual_proxy'):
                actor.set_simulate_physics(False)
            # Make full-wall props' physical long axis road-normal. This is
            # initialization-only, audited below, never an online teleport.
            if role == 'full_blocker' and config['actor_type'] == 'obstacle':
                body = _body(actor)
                _,heading = _at(self.route,interaction_station)
                tangent = [p[0]*math.cos(heading)+p[1]*math.sin(heading) for p in body]
                normal = [-p[0]*math.sin(heading)+p[1]*math.cos(heading) for p in body]
                if max(tangent)-min(tangent) > max(normal)-min(normal):
                    rotated = self.carla.Rotation(yaw=rotation.yaw+90.)
                    actor.set_transform(self.carla.Transform(location,rotated))
            if occluded:
                interaction_station = project(xy(actor.get_location()),self.route)['s_m']
            row = dict(config=config,actor=actor,original_waypoint=original,cross_path=cross_path,
                       start=actor.get_location(),interaction_station_m=interaction_station,
                       branch_crossing_s_m=branch_station,finished=False,actions=set(),
                       stop_sign_id=stop_sign_id,
                       crossing_left_limit_m=self._crossing_left_limit(original) if role == 'crossing_actor' else None,
                       prior_motion_observed=False,brake_observed=False,brake_stopped_at=None,resume_origin=None)
            self.rows.append(row)
            if config['actor_type'] == 'vehicle':
                if role == 'braking_lead':
                    self._clear_brake_light(actor)
                    self._drive_vehicle(row,self.waypoints,float(config['speed_mps']))
                else:
                    actor.apply_control(self.carla.VehicleControl(brake=1.))
        if occluded:
            order = {config['id']: i for i,config in enumerate(self.spec['actor_specs'])}
            self.rows.sort(key=lambda row:order[row['config']['id']])
        return [dict(id=row['config']['id'],actor_id=row['actor'].id,type_id=row['actor'].type_id,
                     config=row['config'],actual_position_xy=xy(row['actor'].get_location()),
                     actual_yaw_deg=row['actor'].get_transform().rotation.yaw,
                     interaction_station_m=row['interaction_station_m'],
                     branch_crossing_s_m=row['branch_crossing_s_m'],
                     stop_sign_id=row['stop_sign_id'],
                     crossing_left_limit_m=row['crossing_left_limit_m'],
                     control_interface=('CARLA.WalkerControl' if row['config']['role'] in ('crossing_actor','slow_lead')
                                        else 'CARLA.VehicleControl' if row['config']['actor_type']=='vehicle' else 'static_prop'),
                     animal_asset_reference=(dict(route_id=1075,type_id='walker.animal.1009',parameter='walker',
                         scope='asset_only;independent_site_and_events')
                         if self.spec.get('family')=='Animals' and row['actor'].type_id=='walker.animal.1009' else None),
                     initial_light_state=(int(row['actor'].get_light_state()) if row['config']['role']=='braking_lead' else None))
                for row in self.rows]

    def validate_bodies(self, teaching_path, ego_width, ego_length=5.):
        """Return measured-footprint prechecks, never a physical safety proof."""
        if not all(math.isfinite(v) and v > 0. for v in (ego_width,ego_length)):
            raise ValueError('Measured ego dimensions must be positive')
        if (len(teaching_path) < 2 or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in teaching_path)
                or any(math.dist(a,b)<1e-5 for a,b in zip(teaching_path,teaching_path[1:]))):
            raise ValueError('Teaching path requires distinct consecutive finite XY points')
        self.teaching_path,self.ego_width = teaching_path,float(ego_width)
        audit = dict(schema='independent-demo-body-precheck-1.0',actors=[],full_lane_blockage=None,
                     passing_clearance=None,ego_width_m=ego_width,ego_length_m=ego_length,
                     limitation='Planar initialization precheck only; actual physical contact and compliance remain independently audited')
        self.geometry_audit = audit
        for row in self.rows:
            audit['actors'].append(dict(id=row['config']['id'],actor_id=row['actor'].id,footprint_xy=_body(row['actor'])))
        if 'full_lane_blockage' in self.spec['required_capabilities']:
            rows = [r for r in self.rows if r['config']['role'] == 'full_blocker']
            if not rows:
                raise RuntimeError('Full-blockage design has no owned full blockers')
            station = sum(r['interaction_station_m'] for r in rows)/len(rows)
            center,heading = _at(self.route,station)
            intervals = [_slice_interval(_body(r['actor']),center,heading) for r in rows]
            intervals = [value for value in intervals if value is not None]
            width,cursor = rows[0]['original_waypoint'].lane_width,-rows[0]['original_waypoint'].lane_width/2.
            for left,right in sorted(intervals):
                if left > cursor+.05:
                    raise RuntimeError('Measured full blockage leaves a lateral gap in the common road cross-section')
                cursor = max(cursor,right)
            if cursor < width/2.-.05:
                raise RuntimeError('Measured actors do not cover the full current lane at a common station')
            audit['full_lane_blockage'] = dict(station_m=station,intervals_d_m=intervals,lane_width_m=width,covered=True)
        fixed_roles = {'blocking_vehicle','blocking_obstacle','passable_edge_obstacle','roadside_occluder',
                       'adjacent_construction','roadside_visual_distractor','stop_sign_visual_occluder','roadside_crowd'}
        measured = []
        for row in self.rows:
            if row['config']['role'] not in fixed_roles:
                continue
            polygon,minimum = _body(row['actor']),float('inf')
            for a,b in zip(teaching_path,teaching_path[1:]):
                length = math.dist(a,b)
                count = max(1,int(math.ceil(length/.5)))
                heading = math.atan2(b[1]-a[1],b[0]-a[0])
                c,s = math.cos(heading),math.sin(heading)
                for i in range(count+1):
                    t = i/count
                    center = [a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]
                    # The half-sample extension guards the gap between body
                    # samples; width remains the measured physical ego width.
                    footprint = [(center[0]+u*c-v*s,center[1]+u*s+v*c)
                        for u,v in ((ego_length/2.+.25,ego_width/2.),(ego_length/2.+.25,-ego_width/2.),
                                    (-ego_length/2.-.25,-ego_width/2.),(-ego_length/2.-.25,ego_width/2.))]
                    minimum = min(minimum,_polygon_distance(footprint,polygon))
            measured.append(dict(id=row['config']['id'],minimum_body_clearance_m=minimum,required_m=.2))
            if minimum < .2-1e-6:
                raise RuntimeError('Authored teaching path lacks measured body clearance: '+row['config']['id'])
        audit['passing_clearance'] = measured
        if self.spec.get('family') == 'PedestrianOtherBlocker':
            blocker = next(r for r in self.rows if r['config']['role'] == 'roadside_occluder')
            walker = next(r for r in self.rows if r['config']['role'] == 'crossing_actor')
            trigger = next(t['distance_m'] for t in self.spec['trigger_specs'] if 'distance_m' in t)
            viewpoint,_ = _at(self.route, blocker['interaction_station_m']-trigger)
            target = xy(walker['actor'].get_location())
            body = _body(blocker['actor'])
            sight = _path_body_distance([viewpoint,target],body)
            separation = _polygon_distance(_body(walker['actor']),body)
            audit['roadside_occlusion'] = dict(approach_viewpoint_xy=viewpoint, pedestrian_xy=target,
                sightline_body_gap_m=sight, initial_body_separation_m=separation,
                evidence='planar_approach_sightline_intersects_measured_blocker_footprint')
            if sight > 1e-5 or separation <= 0.:
                raise RuntimeError('Measured blocker must occlude the crossing pedestrian with separate physical bodies')
        self.geometry_audit = audit
        return audit

    def _observed(self, row, scene):
        identity = 'carla_'+str(row['actor'].id)
        return next((a for a in scene.get('actors',[]) if a.get('id') == identity),None)

    def _drive_vehicle(self, row, path, speed):
        actor = row['actor']
        point = xy(actor.get_location())
        points = [xy(wp.transform.location) for wp in path]
        progress = project(point,points)['s_m']
        target,_ = _at(points,progress+max(4.,1.2*speed))
        current_yaw = -math.radians(actor.get_transform().rotation.yaw)
        dx,dy = target[0]-point[0],target[1]-point[1]
        left = -dx*math.sin(current_yaw)+dy*math.cos(current_yaw)
        # NPC actuator approximation is explicit; actual response is observed.
        wheel = math.atan2(2.*2.7*left,max(.1,dx*dx+dy*dy))
        steer = max(-1.,min(1.,-wheel/math.radians(35.)))
        actor.apply_control(self.carla.VehicleControl(steer=steer))
        target_wp = min(path,key=lambda wp:math.dist(point,xy(wp.transform.location)))
        forward = target_wp.transform.get_forward_vector()
        actor.set_target_velocity(self.carla.Vector3D(x=speed*forward.x,y=speed*forward.y,z=speed*forward.z))

    def _clear_brake_light(self,actor):
        """Only the owned demonstration leader's Brake bit is suppressed."""
        current=int(actor.get_light_state())
        brake=int(self.carla.VehicleLightState.Brake)
        actor.set_light_state(self.carla.VehicleLightState(current & ~brake))

    def _observe_brake_light(self,row,scene,observed_speed):
        """Read real actor state before commanding this tick, never infer it."""
        actor=row['actor'];frame=scene['diagnostics'].get('carla_frame')
        if type(frame) is not int:
            raise ValueError('HardBrake light observation requires its actual source frame')
        bits=int(actor.get_light_state());brake_bit=int(self.carla.VehicleLightState.Brake)
        control=actor.get_control();applied_brake=float(getattr(control,'brake',0.))
        if not math.isfinite(applied_brake) or not 0.<=applied_brake<=1.:
            raise ValueError('Invalid actual leader brake pedal during light observation')
        lit=bool(bits & brake_bit)
        if applied_brake>0. and lit:
            raise RuntimeError('HardBrake leader has an observed lit Brake lamp while actually braking')
        if applied_brake>0. and not lit:
            row['brake_without_lights_observed']=True
        return dict(phase='interact',event='lead_brake_light_observed',actor_id=actor.id,
            source_frame=frame,source_time=float(scene['time']),observed_light_state=bits,brake_light_bit=brake_bit,
            brake_light_on=lit,observed_applied_brake=applied_brake,observed_speed_mps=observed_speed,
            measurement='actual_Actor.get_light_state_and_Vehicle.get_control_in_paused_source_frame',
            completed=False)

    def update(self, scene, dt):
        stamp,projection = float(scene['time']),scene['diagnostics']['route_projection']
        if not math.isfinite(dt) or dt <= 0. or not math.isfinite(stamp):
            raise ValueError('Event observation time must be finite and dt positive')
        if self.last_time is not None and stamp <= self.last_time:
            raise ValueError('Event observations must move forward in time')
        if self.last_time is not None and abs(stamp-self.last_time-dt) > max(1e-5,dt*.01):
            raise ValueError('Event dt must match contiguous observed simulation time')
        if projection.get('valid') is False:
            raise RuntimeError('Cannot advance demonstration events from an invalid route projection')
        progress = projection.get('s_m',scene['diagnostics'].get('route_s'))
        if progress is None or not math.isfinite(progress):
            raise ValueError('Event trigger requires a finite current observed route station')
        self.last_time = stamp
        ego_speed = math.hypot(*scene['ego']['velocity'])
        distance = next((event['distance_m'] for event in self.spec['trigger_specs'] if 'distance_m' in event),20.)
        first_station = min(row['interaction_station_m'] for row in self.rows)
        kind,events = self.spec['interaction_kind'],[]
        obligations = scene.get('rules',{}).get('stop_sign_obligations',[])
        if kind == 'stop_sign':
            targets = {row.get('stop_sign_id') for row in self.rows if row.get('stop_sign_id') is not None}
            if targets:
                obligations = [row for row in obligations if row.get('id') in targets]
        if kind == 'hard_brake':
            lead = next(row for row in self.rows if row['config']['role'] == 'braking_lead')
            observed = self._observed(lead,scene)
            current_station = project(observed['position'] if observed else xy(lead['actor'].get_location()),self.route)['s_m']
            trigger = 0. <= current_station-progress <= distance
        else:
            trigger = bool(obligations) if kind == 'stop_sign' else progress >= first_station-distance
        if not self.active and trigger:
            self.active,self.activated_at = True,stamp
            events.append(dict(phase='interact',event='activated',completed=False,observed_progress_m=progress,
                               trigger_station_m=first_station,trigger_source='current_observation'))
        for row in self.rows:
            if row['config']['role'] == 'braking_lead' and not self.active:
                observed = self._observed(row,scene)
                row['prior_motion_observed'] |= observed is not None and math.hypot(*observed['velocity']) > 1.
                self._drive_vehicle(row,self.waypoints,float(row['config']['speed_mps']))
        if not self.active:
            return events
        elapsed = stamp-self.activated_at
        if ego_speed < .1:
            self.stationary_s += dt
        if obligations:
            self.stop_rule_seen = True
        if ego_speed < .1 and any(row.get('affected_now') for row in obligations):
            self.stop_stationary_s += dt
            self.stop_max_stationary_s = max(self.stop_max_stationary_s,self.stop_stationary_s)
        else:
            self.stop_stationary_s = 0.
        for obligation in obligations:
            identity = str(obligation['id'])
            self.stop_encounters[identity] = dict(obligation)
            if obligation.get('status') == 'fulfilled' and obligation.get('completed_frame') is not None:
                self.stop_observed = True
        moving_complete = []
        for row in self.rows:
            a,config = row['actor'],row['config']
            if not a.is_alive:
                if kind in ('blocked_wait','slow_lead'):
                    continue
                raise RuntimeError('Required demonstration actor disappeared before completion: '+config['id'])
            role,speed = config['role'],float(config['speed_mps'])
            yaw = math.radians(row['original_waypoint'].transform.rotation.yaw)
            observed = self._observed(row,scene)
            if kind == 'blocked_wait' and elapsed >= self.spec['wait_s']:
                if not a.destroy():
                    raise RuntimeError('Failed to remove owned blockage actor')
                events.append(dict(phase='interact',event='owned_blockage_removed',actor_id=a.id,completed=False))
            elif role == 'slow_lead':
                a.apply_control(self.carla.WalkerControl(direction=self.carla.Vector3D(x=math.cos(yaw),y=math.sin(yaw)),speed=speed))
                if elapsed >= self.spec['wait_s']:
                    if not a.destroy():
                        raise RuntimeError('Failed to remove owned slow-walking actor')
                    events.append(dict(phase='interact',event='owned_slow_actor_removed',actor_id=a.id,completed=False))
            elif role == 'crossing_actor':
                origin = row['original_waypoint'].transform.location
                location = a.get_location()
                lateral = (location.x-origin.x)*math.sin(yaw)-(location.y-origin.y)*math.cos(yaw)
                body = _body(a)
                # Hull is in public XY (CARLA Y negated). The entire body must
                # leave the driving lanes, while the opposite shoulder can be narrow.
                trailing_edge = min((x-origin.x)*math.sin(yaw)+(y+origin.y)*math.cos(yaw) for x,y in body)
                clearance = min(_path_body_distance(path,body) for path in (self.route,self.teaching_path))
                complete = trailing_edge >= row['crossing_left_limit_m']+.1 and clearance > self.ego_width/2.+.2
                a.apply_control(self.carla.WalkerControl(direction=self.carla.Vector3D(x=math.sin(yaw),y=-math.cos(yaw)),speed=0. if complete else speed))
                moving_complete.append(complete)
                if complete and not row['finished']:
                    row['finished'] = True
                    events.append(dict(phase='interact',event='crossing_body_observed_clear',actor_id=a.id,
                                       lateral_m=lateral,body_trailing_lateral_m=trailing_edge,
                                       path_body_clearance_m=clearance,completed=False))
            elif role == 'braking_lead':
                observed_speed = math.hypot(*observed['velocity']) if observed is not None else math.hypot(a.get_velocity().x,a.get_velocity().y)
                row['prior_motion_observed'] |= not row['brake_observed'] and observed_speed > 1.
                if row['brake_observed'] and observed_speed >= .1 and row['resume_origin'] is None:
                    row['brake_stopped_at'] = None
                if row['prior_motion_observed'] and observed_speed < .1 and row['brake_stopped_at'] is None:
                    row['brake_stopped_at'] = stamp
                    row['brake_observed'] = True
                    events.append(dict(phase='interact',event='lead_stop_observed',actor_id=a.id,
                                       observed_speed_mps=observed_speed,completed=False))
                hold_complete = (row['brake_stopped_at'] is not None and
                                 stamp-row['brake_stopped_at']+1e-8 >= self.spec['wait_s'])
                events.append(self._observe_brake_light(row,scene,observed_speed))
                if not hold_complete:
                    self._clear_brake_light(a)
                    a.apply_control(self.carla.VehicleControl(brake=1.))
                    self._clear_brake_light(a)
                    if 'brake' not in row['actions']:
                        events.append(dict(phase='interact',event='lead_brake_commanded',actor_id=a.id,observed_speed_mps=observed_speed,completed=False))
                        row['actions'].add('brake')
                else:
                    if row['resume_origin'] is None:
                        row['resume_origin'] = xy(a.get_location())
                        events.append(dict(phase='interact',event='lead_resume_commanded',actor_id=a.id,completed=False))
                    self._drive_vehicle(row,self.waypoints,speed)
                    moved = math.dist(xy(a.get_location()),row['resume_origin'])
                    moving_complete.append(row['brake_observed'] and row.get('brake_without_lights_observed') is True
                                           and observed_speed > 1. and moved > 2.)
            elif role == 'cross_traffic_vehicle':
                path = row['cross_path']
                branch = project(xy(a.get_location()),[xy(wp.transform.location) for wp in path])
                clearance = min(_path_body_distance(p,_body(a)) for p in (self.route,self.teaching_path))
                complete = branch['s_m'] >= row['branch_crossing_s_m']+8. and clearance > self.ego_width/2.+.5
                if complete:
                    a.apply_control(self.carla.VehicleControl(brake=1.))
                else:
                    self._drive_vehicle(row,path,speed)
                moving_complete.append(complete)
        if kind in ('blocked_wait','slow_lead'):
            ready = elapsed >= self.spec['wait_s'] and all(not row['actor'].is_alive for row in self.rows)
        elif kind == 'stop_sign':
            fulfilled = [value for value in self.stop_encounters.values() if value.get('status') == 'fulfilled']
            ready = bool(fulfilled) and self.stop_observed and all(
                value.get('cleared') or progress > value['stop_s']+4. for value in fulfilled)
            if ready and self.stop_max_stationary_s+1e-7 < self.spec['wait_s']:
                raise RuntimeError('Real stop was released without the configured observed stationary hold')
        elif kind in ('crossing','construction_crossing','cross_traffic','hard_brake'):
            ready = bool(moving_complete) and all(moving_complete)
        else:
            ready = progress >= max(row['interaction_station_m'] for row in self.rows)+5.
        if ready and not self.completed:
            if kind == 'blocked_wait' and self.stationary_s < 3.:
                raise RuntimeError('Blockage disappeared without a genuinely observed wait')
            self.completed,self.completion_progress_m = True,progress
            events.append(dict(phase='interact',event='interaction_completed',completed=True,
                               observed_progress_m=progress,observed_stationary_s=self.stationary_s,
                               official_success=False))
        return events

    def close(self):
        errors = []
        for actor in reversed(self.actors):
            try:
                if actor.is_alive and not actor.destroy():
                    errors.append('destroy returned false: '+str(actor.id))
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('Demonstration actor cleanup failed: '+str(errors))
