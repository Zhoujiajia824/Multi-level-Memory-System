"""Owned actors and causal, auditable events for independent demonstrations.

The environment may schedule its own events. The teacher never receives this
object, role labels, event timers, actor plans or release decisions. Completion
here describes the observed interaction, not successful/official driving.
"""
import json
import math

from .demo_routes import (station_waypoint, crossing_drive_path, oncoming_drive_path, crossing_lights,
                          right_walking_exit, _stop_controls_route, _bypass_stations)
from .route import xy, project, point_in_polygon, wrap, lane_key

WALKABLE_SURFACES = frozenset(('Roads','RoadLines','Sidewalks','Ground','Terrain'))


def requires_borrow_proof(spec):
    return (spec.get('interaction_reconstruction',{}).get('schema')=='fail2drive-authored-interaction-2.0'
            and spec.get('teacher_policy',{}).get('allow_opposite_lane') is True)


def verify_borrow_capture(directory, spec, execution=None, events=None):
    """Replay the authored two-car interaction against frozen public decisions and physical bodies."""
    if not requires_borrow_proof(spec):return None
    def read_json(name):
        with (directory/name).open(encoding='utf-8') as handle:return json.load(handle)
    def read_rows(name):
        with (directory/name).open(encoding='utf-8') as handle:return [json.loads(line) for line in handle]
    actors=read_json('demonstration_actors.json')
    route=read_json('demonstration_design.json')['route_xy']
    geometry=read_json('development_site_geometry.json')
    layout={a['id']:a for a in actors}
    designed=[a for a in spec['actor_specs'] if a['role']=='oncoming_vehicle']
    ids={}
    for a in designed:
        spawned=layout[a['id']]
        if spawned['config']!=a or not spawned['type_id'].startswith('vehicle.'):
            raise ValueError('Opposing borrow has no matching spawned vehicle for '+a['id'])
        ids['second' if a.get('release_on_borrow') else 'first']=spawned['actor_id']
    if len(designed)!=2 or len(ids)!=2 or ids['first']==ids['second']:
        raise ValueError('Opposing borrow requires two distinct authored oncoming vehicles')
    first,second=ids['first'],ids['second']
    first_id,second_id='carla_'+str(first),'carla_'+str(second)
    blockers={layout[a['id']]['actor_id'] for a in spec['actor_specs'] if a['role'] in ('blocking_vehicle','blocking_obstacle')}
    footprints={a['actor_id']:a['footprint_xy'] for a in geometry['actors']}
    if not blockers or not blockers.issubset(footprints):raise ValueError('Missing spawned blocking footprints')
    blocker_tail=max(project(point,route)['s_m'] for i in blockers for point in footprints[i])
    if execution is None:execution=read_rows('execution.jsonl')
    if events is None:events=read_rows('event_evidence.jsonl')
    by_frame={row['frame']:row for row in execution}
    npc_events={}
    for event in events:
        if event.get('event') not in ('oncoming_body_passed_waiting_ego','oncoming_released','scripted_npc_observation'):continue
        frame=event['frame']
        if event.get('source_frame')!=frame or frame not in by_frame or abs(event['time']-by_frame[frame]['time'])>1e-5:
            raise ValueError('Opposing borrow event is not bound to its execution frame')
        npc_events.setdefault(frame,[]).append(event)
    stages={}
    for row in execution:
        frame=row['frame'];observed=row['observed_ego'];b=row.get('teacher',{}).get('borrow') or {}
        scene_events=npc_events.get(frame,[])
        actual={a['actor_id']:a for e in scene_events if e['event']=='scripted_npc_observation' for a in e['actors']}
        seen={a['id']:a for a in b.get('observed_oncoming',())}
        if b.get('center_in_opposing_lane') is True and 'first_passed' not in stages:
            raise ValueError('Ego entered the opposing lane before a prior frame proved the first vehicle had passed')
        front_wait=first_id in b.get('denied_ids',()) and first_id in seen
        if ('wait' not in stages and front_wait and first in actual and math.hypot(*observed['velocity'])<=.15
                and b.get('body_on_home_lane') is True):
            stages['wait']=frame
        if 'wait' not in stages:continue
        if ('first_passed' not in stages and first in actual and frame>stages['wait']
                and any(e.get('event')=='oncoming_body_passed_waiting_ego' and e.get('actor_id')==first
                    and e['npc_body_s_m'][1]+.25<b['ego_rear_s_m'] for e in scene_events)):
            stages['first_passed']=frame
        if 'first_passed' not in stages or frame<=stages['first_passed']:continue
        if (b.get('center_in_opposing_lane') is True and b.get('ego_lane_id')==observed.get('lane_id')
                and b.get('entered_opposing_lane') is True):
            stages.setdefault('entered_opposing',frame)
        if 'entered_opposing' not in stages:continue
        if second in actual and any(e['event']=='oncoming_released' and e['actor_id']==second and
                                    e['reason']=='ego_observed_in_borrowed_lane' for e in scene_events):
            stages.setdefault('second_released',frame)
        if ('second_released' in stages and frame>stages['second_released'] and second in actual and second_id in seen
                and seen[second_id]['closing_mps']>.25 and b.get('early_return_trigger_id')==second_id
                and b.get('return_path')=='early' and b.get('early_return_s_m',math.inf)<b.get('nominal_return_s_m',0.)
                and b.get('return_clearance')=='sampled_road_and_observed_motion_clear'
                and b.get('obstacle_cleared') is True and b.get('ego_rear_s_m',-math.inf)>blocker_tail):
            stages.setdefault('clear_and_select_early',frame)
        if ('clear_and_select_early' in stages and frame>stages['clear_and_select_early'] and
                b.get('stage')=='returned' and b.get('return_path')=='early' and b.get('home_corners')==4
                and b.get('body_on_home_lane') is True and b.get('ego_lane_id')==observed.get('lane_id')
                and b.get('ego_rear_s_m',-math.inf)>blocker_tail and abs(b.get('measured_lateral_m',math.inf))<.35):
            stages.setdefault('returned_home',frame)
    required=('wait','first_passed','entered_opposing','second_released','clear_and_select_early','returned_home')
    missing=[k for k in required if k not in stages]
    if missing:raise ValueError('Missing observed opposing-borrow phases: '+', '.join(missing))
    return dict(schema='authored-opposing-borrow-proof-1.0',actor_ids=ids,blocking_actor_ids=sorted(blockers),
                measured_last_blocker_s_m=blocker_tail,phase_frames=stages)


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


def _body(actor, state=None):
    points=[xy(v) for v in actor.bounding_box.get_world_vertices((state or actor).get_transform())]
    try:return _hull(points)
    except ValueError:
        if not actor.type_id.startswith('static.') or not all(math.isfinite(v) for p in points for v in p):raise
        # Some upright official image assets have a genuine line footprint.
        # Keep its measured span for clearance without giving it artificial thickness.
        ends=max(((a,b) for a in points for b in points),key=lambda pair:math.dist(*pair))
        if math.dist(*ends)<1e-5:raise
        return list(ends)


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
        self.released_actor_ids = set()
        self.activated_at = self.completion_progress_m = None
        self.completed, self.active = False, False
        self.stationary_s = 0.
        self.stop_observed, self.stop_rule_seen = False, False
        self.stop_stationary_s = self.stop_max_stationary_s = 0.
        self.stop_encounters = {}
        self.last_time = None
        self.geometry_audit = None
        self.walker_placements = []
        self.manual=spec.get('interaction_reconstruction',{}).get('schema')=='fail2drive-authored-interaction-2.0'
        self.signal_program=None
        self.traffic_lights=[]
        self.ego_moved=False
        self.geometry_by_path={}
        self.ego_width,self.ego_length=2.,5.

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
            role = config['role']
            cross_path, branch_station, interaction_station = None,None,station
            if role=='oncoming_vehicle':
                # NPC traffic may start well upstream of the short clip's end.
                # Resolve its full physical path before consulting either lane.
                cross_path=oncoming_drive_path(self.waypoints,station)
                if not cross_path:raise RuntimeError('Missing continuous real opposing lane for '+config['id'])
                wp=cross_path[0]
                original=wp.get_left_lane()
                if (config['lane_offset']!=1 or original is None or
                        str(original.lane_type).split('.')[-1]!='Driving' or wp.lane_id*original.lane_id>=0):
                    raise RuntimeError('Oncoming spawn lacks its adjoining original travel lane: '+config['id'])
                lateral=0.
            else:
                wp = original = station_waypoint(self.waypoints,station)
                for _ in range(abs(config['lane_offset'])):
                    wp = wp.get_left_lane() if config['lane_offset'] > 0 else wp.get_right_lane()
                    if wp is None or str(wp.lane_type).split('.')[-1] != 'Driving':
                        raise RuntimeError('Required physical actor driving lane does not exist: '+config['id'])
                lateral=float(config['lateral_offset_m'])
            pose=wp.transform
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
            if role=='construction_advance_warning':
                # Keep the roadside advance sign beyond the entire bank of
                # contiguous travel lanes, including any same-direction neighbour.
                edge=-original.lane_width/2.
                for neighbour in _adjacent_lanes(original,right=True):
                    kind=str(neighbour.lane_type).split('.')[-1]
                    if kind!='Driving':break
                    n=neighbour.transform.location
                    d=(n.x-pose.location.x)*math.sin(yaw)-(n.y-pose.location.y)*math.cos(yaw)
                    edge=min(edge,d-neighbour.lane_width/2.)
                lateral=min(lateral,edge-1.5)
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
            if self.manual and config['actor_type']!='walker':
                if (not cross_path and role!='stop_sign_visual_occluder' and wp is original):
                    center,heading=_at(self.route,station)
                    location.x=center[0]-lateral*math.sin(heading)
                    location.y=-center[1]-lateral*math.cos(heading)
                location.z=pose.location.z+2.
                hit=self.world.ground_projection(location,4.)
                ground=pose.location.z
                if (hit is not None and math.hypot(hit.location.x-location.x,hit.location.y-location.y)<.15
                        and str(hit.label).split('.')[-1] in WALKABLE_SURFACES
                        and abs(hit.location.z-ground)<1.):
                    ground=hit.location.z
                elif role=='construction_advance_warning':
                    raise RuntimeError('No measured roadside ground for the construction advance warning')
                location.z=ground+float(config.get('z_offset_m',.3 if config['actor_type']=='vehicle' else .03))
            # A public cross branch supplies the actual actor heading. Applying
            # the catalog's generic +90 offset again would point it off-road.
            rotation = self.carla.Rotation(yaw=pose.rotation.yaw+(0. if cross_path else config['yaw_offset_deg']))
            if self.manual:
                rotation.pitch=pose.rotation.pitch+float(config.get('pitch_offset_deg',0.))
                rotation.roll=pose.rotation.roll+float(config.get('roll_offset_deg',0.))
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
            if role == 'full_blocker' and config['actor_type'] == 'obstacle' and not any(config.get(k) for k in ('pitch_offset_deg','roll_offset_deg')):
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
                    self._set_brake_light(actor,False)
                    self._drive_vehicle(row,self.waypoints,float(config['speed_mps']))
                else:
                    actor.apply_control(self.carla.VehicleControl(brake=1.))
        if occluded:
            order = {config['id']: i for i,config in enumerate(self.spec['actor_specs'])}
            self.rows.sort(key=lambda row:order[row['config']['id']])
        if self.manual:
            self._plan_walkers()
            self._configure_crossing_signals()
        return [dict(id=row['config']['id'],actor_id=row['actor'].id,type_id=row['actor'].type_id,
                     config=row['config'],actual_position_xy=xy(row['actor'].get_location()),
                     actual_yaw_deg=row['actor'].get_transform().rotation.yaw,
                     interaction_station_m=row['interaction_station_m'],
                     branch_crossing_s_m=row['branch_crossing_s_m'],
                     stop_sign_id=row['stop_sign_id'],
                     authored_motion=(self._motion_design(row) if self.manual else None),
                     crossing_left_limit_m=row['crossing_left_limit_m'],
                     control_interface=('CARLA.WalkerControl' if row['config']['role'] in ('crossing_actor','slow_lead')
                                        else 'CARLA.VehicleControl' if row['config']['actor_type']=='vehicle' else 'static_prop'),
                     animal_asset_reference=(dict(route_id=1075,type_id='walker.animal.1009',parameter='walker',
                         scope='asset_only;independent_site_and_events')
                         if self.spec.get('family')=='Animals' and row['actor'].type_id=='walker.animal.1009' else None),
                     initial_light_state=(int(row['actor'].get_light_state()) if row['config']['role']=='braking_lead' else None))
                for row in self.rows]

    def _plan_walkers(self):
        for row in self.rows:
            config=row['config']
            if config['role'] not in ('crossing_actor','slow_lead'):continue
            start=row['start'];station=row['interaction_station_m'];wp=row['original_waypoint']
            points=[[start.x,start.y,start.z]]
            if config['role']=='slow_lead':
                station+=float(config['speed_mps'])*float(self.spec['wait_s'])
                wp=station_waypoint(self.waypoints,station)
                # Follow the configured lane offset until the group starts to leave it.
                center=wp.transform.location;right=wp.transform.get_right_vector()
                lateral=config['lateral_offset_m']
                points.append([center.x-right.x*lateral,center.y-right.y*lateral,center.z])
                exit_lane=right_walking_exit(wp)
                if exit_lane is None:raise RuntimeError('Slow pedestrian has no adjoining mapped shoulder/sidewalk exit')
                p=exit_lane.transform.location
                points.append([p.x,p.y,p.z])
            else:
                center=wp.transform.location;right=wp.transform.get_right_vector()
                edge=row['crossing_left_limit_m']+1.25
                points.append([center.x-edge*right.x,center.y-edge*right.y,center.z])
            row['walker_path']=points
            row['walker_index']=1

    def _motion_design(self,row):
        config=row['config'];path=row.get('cross_path')
        if config['role']=='braking_lead':path=self.waypoints
        start=row['start']
        coords=row.get('walker_path') or ([[p.transform.location.x,p.transform.location.y,p.transform.location.z]
                  for p in path] if path else [[start.x,start.y,start.z]])
        return dict(path_xyz_carla_m=coords,target_speed_mps=config['speed_mps'],
                    trigger_specs=self.spec['trigger_specs'],role=config['role'],
                    release_time_s=config.get('release_time_s'),release_on_borrow=config.get('release_on_borrow',False),
                    traffic_signals=self.signal_program if config['role']=='cross_traffic_vehicle' else None)

    def _configure_crossing_signals(self):
        crossing=next((r for r in self.rows if r['config']['role']=='cross_traffic_vehicle'),None)
        if crossing is None:return
        lights=list(self.world.get_actors().filter('traffic.traffic_light'))
        main,cross=crossing_lights(lights,self.waypoints,crossing['cross_path'])
        if not main and not cross:
            self.signal_program=dict(mode='yield_to_cross_traffic',controlled_light_ids=[])
            return
        if not main or not cross or {a.id for a in main}&{a.id for a in cross}:
            raise RuntimeError('Cannot map distinct physical signals to both crossing approaches')
        frozen={light.is_frozen() for light in lights}
        if len(frozen)!=1:raise RuntimeError('Cannot restore mixed global traffic-light freeze settings')
        members={member.id:member for a in main+cross for member in a.get_group_traffic_lights()}
        members.update({a.id:a for a in main+cross})
        self.traffic_lights=[(a,a.get_state()) for a in members.values()]
        self.signal_program=dict(mode='ego_green_crossing_red',ego_green_ids=[a.id for a in main],
            crossing_red_ids=[a.id for a in cross],previously_frozen=frozen.pop())
        self.world.freeze_all_traffic_lights(True)
        for light,_ in self.traffic_lights:
            light.set_state(self.carla.TrafficLightState.Green if light.id in self.signal_program['ego_green_ids']
                            else self.carla.TrafficLightState.Red)

    def validate_bodies(self, teaching_path, ego_width, ego_length=5., *, path_name='normal'):
        """Return measured-footprint prechecks, never a physical safety proof."""
        if not all(math.isfinite(v) and v > 0. for v in (ego_width,ego_length)):
            raise ValueError('Measured ego dimensions must be positive')
        if (len(teaching_path) < 2 or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in teaching_path)
                or any(math.dist(a,b)<1e-5 for a,b in zip(teaching_path,teaching_path[1:]))):
            raise ValueError('Teaching path requires distinct consecutive finite XY points')
        if path_name=='normal':self.teaching_path=teaching_path
        self.ego_width,self.ego_length=float(ego_width),float(ego_length)
        audit = dict(schema='independent-demo-body-precheck-1.0',actors=[],full_lane_blockage=None,
                     passing_clearance=None,ego_width_m=ego_width,ego_length_m=ego_length,
                     limitation='Planar initialization precheck only; actual physical contact and compliance remain independently audited')
        if path_name=='normal':self.geometry_audit=audit
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
                       'adjacent_construction','roadside_visual_distractor','stop_sign_visual_occluder','roadside_crowd',
                       'construction_advance_warning'}
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
        self.geometry_by_path[path_name]=dict(passing_clearance=measured)
        if self.geometry_audit is not None:self.geometry_audit['path_checks']=dict(self.geometry_by_path)
        return audit

    def _observed(self, row, scene):
        identity = 'carla_'+str(row['actor'].id)
        return next((a for a in scene.get('actors',[]) if a.get('id') == identity),None)

    def _drive_vehicle(self, row, path, speed, state=None):
        actor = row['actor']
        state=state or actor
        pose=state.get_transform();point=xy(pose.location)
        points = [xy(wp.transform.location) for wp in path]
        progress = project(point,points)['s_m']
        target,_ = _at(points,progress+max(4.,1.2*speed))
        current_yaw = -math.radians(pose.rotation.yaw)
        dx,dy = target[0]-point[0],target[1]-point[1]
        left = -dx*math.sin(current_yaw)+dy*math.cos(current_yaw)
        # NPC actuator approximation is explicit; actual response is observed.
        wheel = math.atan2(2.*2.7*left,max(.1,dx*dx+dy*dy))
        steer = max(-1.,min(1.,-wheel/math.radians(35.)))
        if self.manual:
            velocity=state.get_velocity()
            measured=math.hypot(velocity.x,velocity.y)
            error=speed-measured
            control=self.carla.VehicleControl(steer=steer,throttle=min(.65,max(0.,.09+.32*error)),
                brake=min(.8,max(0.,-.40*(error+.2))))
            if control.brake>0.:control.throttle=0.
            actor.apply_control(control)
            return
        actor.apply_control(self.carla.VehicleControl(steer=steer))
        target_wp = min(path,key=lambda wp:math.dist(point,xy(wp.transform.location)))
        forward = target_wp.transform.get_forward_vector()
        actor.set_target_velocity(self.carla.Vector3D(x=speed*forward.x,y=speed*forward.y,z=speed*forward.z))

    def _set_brake_light(self,actor,on):
        current=int(actor.get_light_state())
        brake=int(self.carla.VehicleLightState.Brake)
        actor.set_light_state(self.carla.VehicleLightState(current|brake if on else current & ~brake))

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
        wanted=row['config'].get('brake_lights_on',False)
        if applied_brake>0. and lit!=wanted:
            raise RuntimeError('HardBrake leader actual Brake lamp differs from the scripted braking condition')
        if applied_brake>0.:
            row['required_brake_light_observed']=True
            if not lit:row['brake_without_lights_observed']=True
        return dict(phase='interact',event='lead_brake_light_observed',actor_id=actor.id,
            source_frame=frame,source_time=float(scene['time']),observed_light_state=bits,brake_light_bit=brake_bit,
            brake_light_on=lit,observed_applied_brake=applied_brake,observed_speed_mps=observed_speed,
            measurement='actual_Actor.get_light_state_and_Vehicle.get_control_in_paused_source_frame',
            completed=False)

    def _oncoming(self,row,scene,state,progress,ego_speed):
        actor,config=row['actor'],row['config']
        second=config.get('release_on_borrow') is True
        ego=scene['ego'];station=project(ego['position'],self.route)
        wp=station_waypoint(self.waypoints,progress)
        adjacent=wp.get_left_lane()
        ego_in_opposing=(station['lateral_m']>wp.lane_width/2. and adjacent is not None and
                         wp.lane_id*adjacent.lane_id<0 and ego.get('lane_id')==lane_key(adjacent))
        first=next(r for r in self.rows if r['config']['role']=='oncoming_vehicle' and not r['config'].get('release_on_borrow'))
        signal=('ego_observed_in_borrowed_lane' if second else 'ego_wait_observed')
        release=(first.get('passed_ego') and ego_in_opposing) if second else (self.ego_moved and ego_speed<.15)
        frame=scene['diagnostics']['carla_frame'];events=[]
        if release and not row.get('released') and scene['time']-self.activated_at>=config.get('release_time_s',0.):
            row['released']=True
            events.append(dict(event='oncoming_released',reason=signal))
        body=_body(actor,state)
        xy_center=xy(state.get_transform().location)
        samples=[project(p,self.route)['s_m'] for p in body]
        velocity=state.get_velocity();speed=math.hypot(velocity.x,velocity.y)
        if row.get('released'):
            if speed>.5:row['observed_moving']=True
            if (row.get('observed_moving') and not row.get('passed_ego')
                    and max(samples)<progress-self.ego_length/2.-.3):
                row['passed_ego']=True
                events.append(dict(event='oncoming_body_passed_waiting_ego' if not second else 'second_oncoming_body_passed_ego'))
            destination=xy(row['cross_path'][-1].transform.location)
            if math.dist(xy_center,destination)>2.:
                self._drive_vehicle(row,row['cross_path'],config['speed_mps'],state)
            else:
                actor.apply_control(self.carla.VehicleControl(brake=1.))
        else:
            actor.apply_control(self.carla.VehicleControl(brake=1.))
        for e in events:e.update(phase='interact',completed=False,actor_id=actor.id,source_frame=frame,
            source_time=scene['time'],observed_ego_speed_mps=ego_speed,observed_npc_speed_mps=speed,
            ego_route_s_m=progress,npc_body_s_m=[min(samples),max(samples)])
        return events

    def _observed_crossing(self,row,clearance,progress,scene,events):
        """Record actual occupation in front of the ego before accepting a clear path."""
        if not self.manual:return True
        if clearance<=self.ego_width/2. and not row.get('conflict_observed'):
            if progress+self.ego_length/2.>row['interaction_station_m']+.25:
                raise RuntimeError('Crossing actor entered the ego path only after the ego front reached the conflict point')
            row['conflict_observed']=True
            events.append(dict(event='crossing_actor_occupied_driving_path',actor_id=row['actor'].id,
                source_frame=scene['diagnostics']['carla_frame'],source_time=scene['time'],phase='interact',
                ego_route_s_m=progress,crossing_s_m=row['interaction_station_m'],path_body_clearance_m=clearance,completed=False))
        return bool(row.get('conflict_observed'))

    def update(self, scene, dt, *, snapshot=None):
        stamp,projection = float(scene['time']),scene['diagnostics']['route_projection']
        states={}
        if self.manual:
            frame=scene['diagnostics'].get('carla_frame')
            if snapshot is None or snapshot.frame!=frame:raise ValueError('Scripted actor observations require the same official world frame')
            for row in self.rows:
                actor=row['actor'];state=snapshot.find(actor.id)
                if state is None and actor.is_alive:raise RuntimeError('Owned actor absent from current frozen frame: '+row['config']['id'])
                if state is not None:states[actor.id]=state
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
        critical=[r for r in self.rows if r['config']['role'] not in ('oncoming_vehicle','construction_advance_warning')]
        first_station = min(row['interaction_station_m'] for row in critical)
        kind,events = self.spec['interaction_kind'],[]
        if self.manual and kind in ('crossing','construction_crossing'):
            first_station=min(r['interaction_station_m'] for r in critical if r['config']['role']=='crossing_actor')
        self.ego_moved |= ego_speed>.5
        if self.manual:
            observations=[]
            for row in self.rows:
                actor=row['actor'];state=states.get(actor.id);cfg=row['config']
                if state is None or cfg['actor_type'] not in ('vehicle','walker'):continue
                p=state.get_transform();v=state.get_velocity();ctrl=actor.get_control()
                previous_control={k:getattr(ctrl,k) for k in ('steer','throttle','brake','speed') if hasattr(ctrl,k)}
                if hasattr(ctrl,'direction'):previous_control['direction_xyz']=[ctrl.direction.x,ctrl.direction.y,ctrl.direction.z]
                observations.append(dict(actor_id=actor.id,id=cfg['id'],xyz_carla_m=[p.location.x,p.location.y,p.location.z],
                       yaw_deg=p.rotation.yaw,velocity_xyz_mps=[v.x,v.y,v.z],observed_control=previous_control))
            signals={str(light.id):str(light.get_state()).split('.')[-1] for light,_ in self.traffic_lights}
            if self.traffic_lights and any(signals[str(i)]!=desired for ids,desired in
                    ((self.signal_program['ego_green_ids'],'Green'),(self.signal_program['crossing_red_ids'],'Red'))
                    for i in ids):raise RuntimeError('Observed traffic signal differs from scripted crossing phase')
            events.append(dict(phase='interact' if self.active else 'approach',event='scripted_npc_observation',
                source_frame=snapshot.frame,source_time=stamp,actors=observations,signal_states=signals,completed=False))
        obligations = scene.get('rules',{}).get('stop_sign_obligations',[])
        if kind == 'stop_sign':
            targets = {row.get('stop_sign_id') for row in self.rows if row.get('stop_sign_id') is not None}
            if targets:
                obligations = [row for row in obligations if row.get('id') in targets]
        if kind == 'hard_brake':
            lead = next(row for row in self.rows if row['config']['role'] == 'braking_lead')
            observed = self._observed(lead,scene)
            measured=states.get(lead['actor'].id) or lead['actor']
            current_station = project(observed['position'] if observed else xy(measured.get_transform().location),self.route)['s_m']
            trigger = 0. <= current_station-progress <= distance
        else:
            trigger = bool(obligations) if kind == 'stop_sign' else progress >= first_station-distance
            if any(r['config']['role']=='oncoming_vehicle' for r in self.rows):
                trigger=trigger or progress>=max(0.,_bypass_stations(self.spec)[0]-12.)
        if not self.active and trigger:
            self.active,self.activated_at = True,stamp
            events.append(dict(phase='interact',event='activated',completed=False,observed_progress_m=progress,
                               trigger_station_m=first_station,trigger_source='current_observation'))
        for row in self.rows:
            if row['config']['role'] == 'braking_lead' and not self.active:
                observed = self._observed(row,scene)
                row['prior_motion_observed'] |= observed is not None and math.hypot(*observed['velocity']) > 1.
                self._drive_vehicle(row,self.waypoints,float(row['config']['speed_mps']),states.get(row['actor'].id))
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
            current=states.get(a.id) or a
            yaw = math.radians(row['original_waypoint'].transform.rotation.yaw)
            observed = self._observed(row,scene)
            if role=='oncoming_vehicle':
                events.extend(self._oncoming(row,scene,current,progress,ego_speed))
                continue
            if kind == 'blocked_wait' and elapsed >= self.spec['wait_s']:
                if not a.destroy():
                    raise RuntimeError('Failed to remove owned blockage actor')
                self.released_actor_ids.add(a.id)
                events.append(dict(phase='interact',event='owned_blockage_removed',actor_id=a.id,completed=False))
            elif role=='slow_lead' and self.manual:
                path=row['walker_path'];location=current.get_transform().location
                destination=path[row['walker_index']]
                dx,dy=destination[0]-location.x,destination[1]-location.y
                if math.hypot(dx,dy)<.5 and row['walker_index']<len(path)-1:
                    row['walker_index']+=1;destination=path[row['walker_index']]
                    dx,dy=destination[0]-location.x,destination[1]-location.y
                gap=_path_body_distance(self.route,_body(a,current))
                complete=row['walker_index']==len(path)-1 and math.hypot(dx,dy)<.55 and gap>self.ego_width/2.+.25
                norm=max(.01,math.hypot(dx,dy))
                a.apply_control(self.carla.WalkerControl(direction=self.carla.Vector3D(x=dx/norm,y=dy/norm),speed=0. if complete else speed))
                moving_complete.append(complete)
                if complete and not row['finished']:
                    row['finished']=True;events.append(dict(event='walker_exited_driving_path',actor_id=a.id,
                        phase='interact',completed=False,path_body_clearance_m=gap,source_frame=snapshot.frame))
            elif role == 'slow_lead':
                a.apply_control(self.carla.WalkerControl(direction=self.carla.Vector3D(x=math.cos(yaw),y=math.sin(yaw)),speed=speed))
                if elapsed >= self.spec['wait_s']:
                    if not a.destroy():
                        raise RuntimeError('Failed to remove owned slow-walking actor')
                    self.released_actor_ids.add(a.id)
                    events.append(dict(phase='interact',event='owned_slow_actor_removed',actor_id=a.id,completed=False))
            elif role == 'crossing_actor':
                origin = row['original_waypoint'].transform.location
                location = current.get_transform().location
                lateral = (location.x-origin.x)*math.sin(yaw)-(location.y-origin.y)*math.cos(yaw)
                body = _body(a,current)
                # Hull is in public XY (CARLA Y negated). The entire body must
                # leave the driving lanes, while the opposite shoulder can be narrow.
                trailing_edge = min((x-origin.x)*math.sin(yaw)+(y+origin.y)*math.cos(yaw) for x,y in body)
                clearance = min(_path_body_distance(path,body) for path in (self.route,self.teaching_path))
                occupied=self._observed_crossing(row,clearance,progress,scene,events)
                complete = occupied and trailing_edge >= row['crossing_left_limit_m']+.1 and clearance > self.ego_width/2.+.2
                a.apply_control(self.carla.WalkerControl(direction=self.carla.Vector3D(x=math.sin(yaw),y=-math.cos(yaw)),speed=0. if complete else speed))
                moving_complete.append(complete)
                if complete and not row['finished']:
                    row['finished'] = True
                    events.append(dict(phase='interact',event='crossing_body_observed_clear',actor_id=a.id,
                                       lateral_m=lateral,body_trailing_lateral_m=trailing_edge,
                                       path_body_clearance_m=clearance,completed=False))
            elif role == 'braking_lead':
                velocity=current.get_velocity()
                observed_speed = math.hypot(*observed['velocity']) if observed is not None else math.hypot(velocity.x,velocity.y)
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
                if row['resume_origin'] is None:events.append(self._observe_brake_light(row,scene,observed_speed))
                if not hold_complete:
                    self._set_brake_light(a,config.get('brake_lights_on',False))
                    a.apply_control(self.carla.VehicleControl(brake=1.))
                    self._set_brake_light(a,config.get('brake_lights_on',False))
                    if 'brake' not in row['actions']:
                        events.append(dict(phase='interact',event='lead_brake_commanded',actor_id=a.id,observed_speed_mps=observed_speed,completed=False))
                        row['actions'].add('brake')
                else:
                    if row['resume_origin'] is None:
                        row['resume_origin'] = xy(current.get_transform().location)
                        events.append(dict(phase='interact',event='lead_resume_commanded',actor_id=a.id,completed=False))
                    self._drive_vehicle(row,self.waypoints,speed,current)
                    self._set_brake_light(a,False)
                    moved = math.dist(xy(current.get_transform().location),row['resume_origin'])
                    moving_complete.append(row['brake_observed'] and row.get('required_brake_light_observed') is True
                                           and observed_speed > 1. and moved > 2.)
            elif role == 'cross_traffic_vehicle':
                path = row['cross_path']
                branch = project(xy(current.get_transform().location),[xy(wp.transform.location) for wp in path])
                clearance = min(_path_body_distance(p,_body(a,current)) for p in (self.route,self.teaching_path))
                occupied=self._observed_crossing(row,clearance,progress,scene,events)
                complete = occupied and branch['s_m'] >= row['branch_crossing_s_m']+8. and clearance > self.ego_width/2.+.5
                if complete:
                    a.apply_control(self.carla.VehicleControl(brake=1.))
                    if self.manual and not row['finished']:
                        row['finished']=True
                        events.append(dict(event='cross_traffic_body_cleared',actor_id=a.id,phase='interact',
                                source_frame=snapshot.frame,source_time=stamp,path_body_clearance_m=clearance,completed=False))
                else:
                    self._drive_vehicle(row,path,speed,current)
                moving_complete.append(complete)
        if kind=='slow_lead' and self.manual:ready=bool(moving_complete) and all(moving_complete)
        elif kind in ('blocked_wait','slow_lead'):
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
            ready = progress >= max(row['interaction_station_m'] for row in critical)+5.
        oncoming=[r for r in self.rows if r['config']['role']=='oncoming_vehicle']
        if oncoming:
            ready=(ready and all(r.get('observed_moving') if r['config'].get('release_on_borrow')
                   else r.get('passed_ego') for r in oncoming) and abs(project(scene['ego']['position'],self.route)['lateral_m'])<.6)
        if ready and not self.completed:
            if kind == 'blocked_wait' and self.stationary_s < 3.:
                raise RuntimeError('Blockage disappeared without a genuinely observed wait')
            self.completed,self.completion_progress_m = True,progress
            events.append(dict(phase='interact',event='interaction_completed',completed=True,
                               observed_progress_m=progress,observed_stationary_s=self.stationary_s,
                               official_success=False))
        return events

    def close(self, *, destroy_actors=True):
        errors = []
        if self.traffic_lights:
            for light,state in self.traffic_lights:
                try:light.set_state(state)
                except RuntimeError as exc:errors.append(str(exc))
            try:self.world.freeze_all_traffic_lights(self.signal_program['previously_frozen'])
            except RuntimeError as exc:errors.append(str(exc))
        for actor in (reversed(self.actors) if destroy_actors else ()):
            try:
                if actor.is_alive and not actor.destroy():
                    errors.append('destroy returned false: '+str(actor.id))
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('Demonstration actor cleanup failed: '+str(errors))
