"""Authored short-demo routes, with optional frozen collection-location hints.

This module is SDK duck-typed and has no CARLA import. A failed site search is
an error, never a reason to relax crossing permissions or move a test finish.
"""
import math
import random

from .route import lane_key, xy, wrap, project


class NoSuitableSite(RuntimeError):
    """The verified input is valid, but this road cannot host the interaction."""


def chain(start, length, step=1.):
    points, arc = [start], 0.
    locations = [start.transform.location]
    while arc < length:
        choices = list(points[-1].next(step))
        if not choices:
            return None
        previous = points[-1]
        nxt = min(choices, key=lambda wp: (abs(wrap(math.radians(
            wp.transform.rotation.yaw-previous.transform.rotation.yaw))), lane_key(wp)))
        location = nxt.transform.location
        gap = locations[-1].distance(location)
        if not .05 < gap <= step*2.1:
            return None
        if any(location.distance(past) < .3 for past in locations[:-8]):
            return None
        arc += gap
        points.append(nxt)
        locations.append(location)
    return points


def left_neighbor(wp):
    other = wp.get_left_lane()
    if (other is None or str(other.lane_type).split('.')[-1] != 'Driving'
            or math.cos(math.radians(other.transform.rotation.yaw-wp.transform.rotation.yaw)) < .9):
        return None
    marking = str(wp.left_lane_marking.type).split('.')[-1].lower()
    permission = str(wp.lane_change).split('.')[-1].lower()
    if marking not in ('broken', 'brokenbroken') or permission not in ('left', 'both'):
        return None
    return other


def _return_allowed(adjacent):
    return (str(adjacent.right_lane_marking.type).split('.')[-1].lower() in ('broken', 'brokenbroken')
            and str(adjacent.lane_change).split('.')[-1].lower() in ('right', 'both'))


def _driving_neighbor(wp, side, same_direction=True):
    other = getattr(wp, 'get_' + side + '_lane')()
    if other is None or str(other.lane_type).split('.')[-1] != 'Driving':
        return None
    alignment = math.cos(math.radians(other.transform.rotation.yaw-wp.transform.rotation.yaw))
    if same_direction and alignment < .9 or not same_direction and alignment > -.9:
        return None
    return other


def _right_sidewalk(wp):
    """Traverse public right-side lanes without inventing a sidewalk."""
    seen = {lane_key(wp)}
    for _ in range(12):
        wp = wp.get_right_lane()
        if wp is None or lane_key(wp) in seen:
            return None
        seen.add(lane_key(wp))
        if str(wp.lane_type).split('.')[-1] == 'Sidewalk':
            return wp
    return None


def _stations(waypoints):
    result = [0.]
    for a, b in zip(waypoints, waypoints[1:]):
        result.append(result[-1]+a.transform.location.distance(b.transform.location))
    return result


def _stop_controls_route(sign, waypoints, target_station=None, tolerance_m=4.):
    """Use actual stop trigger/lane intersection, not sign-proximity alone."""
    box = getattr(sign, 'trigger_volume', None)
    if box is None:
        return False
    matrix = sign.get_transform().get_matrix()
    local = [box.location.x, box.location.y, box.location.z, 1.]
    center = [sum(matrix[i][j]*local[j] for j in range(4)) for i in range(3)]
    ex, ey = 1.2*max(.5, abs(box.extent.x)), 1.2*max(.5, abs(box.extent.y))
    return any((target_station is None or abs(station-target_station) <= tolerance_m)
               and abs(wp.transform.location.x-center[0]) < ex
               and abs(wp.transform.location.y-center[1]) < ey
               and abs(wp.transform.location.z-center[2]) <= abs(box.extent.z)+1.5
               for wp, station in zip(waypoints, _stations(waypoints)))


def _crossing_fraction(a, b, c, d):
    ax, ay = b[0]-a[0], b[1]-a[1]
    bx, by = d[0]-c[0], d[1]-c[1]
    determinant = ax*by-ay*bx
    if abs(determinant) < 1e-8:
        return None
    dx, dy = c[0]-a[0], c[1]-a[1]
    t = (dx*by-dy*bx)/determinant
    u = (dx*ay-dy*ax)/determinant
    return t if 0 <= t <= 1 and 0 <= u <= 1 else None


def _segments_cross(a, b, c, d):
    return _crossing_fraction(a, b, c, d) is not None


def _conflicting_branch(waypoints, target_station=None, tolerance_m=4.):
    """Require a public perpendicular driving branch crossing this route."""
    junctions = {}
    for wp in waypoints:
        if wp.is_junction:
            junction = wp.get_junction()
            if junction is not None:
                junctions[junction.id] = (junction, wp.lane_type)
    route = [xy(wp.transform.location) for wp in waypoints]
    stations = _stations(waypoints)
    for junction, lane_type in junctions.values():
        for entry, exit in junction.get_waypoints(lane_type):
            c, d = xy(entry.transform.location), xy(exit.transform.location)
            chord = math.atan2(d[1]-c[1], d[0]-c[0])
            for index, (a, b) in enumerate(zip(route, route[1:])):
                heading = math.atan2(b[1]-a[1], b[0]-a[0])
                fraction = _crossing_fraction(a, b, c, d)
                station = (stations[index]+fraction*(stations[index+1]-stations[index])
                           if fraction is not None else None)
                if (fraction is not None and abs(math.cos(wrap(chord-heading))) < .7
                        and (target_station is None or abs(station-target_station) <= tolerance_m)):
                    return True
    return False


def crossing_drive_path(waypoints, target_station):
    """The same physical vehicle path is checked in site search and at spawn."""
    stations = _stations(waypoints)
    route = [xy(wp.transform.location) for wp in waypoints]
    segments = [(i, a, b) for i, (a, b) in enumerate(zip(route, route[1:]))
                if stations[i] <= target_station+4. and stations[i+1] >= target_station-4.]
    choices, seen = [], set()
    for wp in waypoints:
        if not wp.is_junction:
            continue
        junction = wp.get_junction()
        if junction is None or junction.id in seen:
            continue
        seen.add(junction.id)
        for entry, _ in junction.get_waypoints(wp.lane_type):
            ahead = chain(entry, 35.)
            if not ahead:
                continue
            prefix, cursor = [], entry
            for _ in range(10):
                options = list(cursor.previous(1.))
                if not options:
                    break
                cursor = min(options, key=lambda item: abs(wrap(math.radians(
                    item.transform.rotation.yaw-cursor.transform.rotation.yaw))))
                prefix.append(cursor)
            path = list(reversed(prefix))+ahead
            points = [xy(item.transform.location) for item in path]
            branch_arc = _stations(path)
            for k, (a, b) in enumerate(zip(points, points[1:])):
                for i, c, d in segments:
                    t = _crossing_fraction(c, d, a, b)
                    if t is None:
                        continue
                    u = _crossing_fraction(a, b, c, d)
                    if u is None:
                        continue
                    ego_heading = math.atan2(d[1]-c[1], d[0]-c[0])
                    branch_heading = math.atan2(b[1]-a[1], b[0]-a[0])
                    if abs(math.cos(branch_heading-ego_heading)) > .5:
                        continue
                    ego_z = (1.-t)*waypoints[i].transform.location.z+t*waypoints[i+1].transform.location.z
                    cross_z = (1.-u)*path[k].transform.location.z+u*path[k+1].transform.location.z
                    station = stations[i]+t*(stations[i+1]-stations[i])
                    before = branch_arc[k]+u*(branch_arc[k+1]-branch_arc[k])
                    if (abs(ego_z-cross_z) <= 1.5 and abs(station-target_station) <= 4.
                            and before >= 6. and branch_arc[-1]-before >= 10.):
                        choices.append((abs(station-target_station),station,before,path))
    if not choices:
        return None
    _, station, before, path = min(choices, key=lambda choice:choice[:3])
    return path, station, before


def _start_clear(start, actors):
    """Conservative site feasibility, before any new actors are spawned."""
    origin = start.transform.location
    for actor in actors:
        if not actor.type_id.startswith(('vehicle.', 'walker.pedestrian.', 'static.prop.')):
            continue
        location = actor.get_location()
        if abs(location.z-origin.z) > 3.:
            continue
        box = getattr(actor, 'bounding_box', None)
        radius = math.hypot(box.extent.x, box.extent.y) if box is not None else 1.
        offset = getattr(box, 'location', None)
        if offset is not None:
            radius += math.hypot(offset.x, offset.y)
        if math.hypot(location.x-origin.x, location.y-origin.y) < radius+3.5:
            return False
    return True


def _anchor_candidates(road_map,locations,source,hazard_station):
    """Stay on the actual collection lane; never search a remote substitute site."""
    if not locations:raise RuntimeError('Map has no public spawn-point location type for anchor lookup')
    try:
        point=type(locations[0].location)(**dict(zip(('x','y','z'),source['trigger_xyz_carla_m'])))
        origin=road_map.get_waypoint(point)
    except (TypeError,AttributeError) as error:
        raise ValueError('Cannot resolve the collection anchor through the public map API') from error
    if (origin is None or str(origin.lane_type).split('.')[-1]!='Driving'
            or math.hypot(origin.transform.location.x-point.x,origin.transform.location.y-point.y)>1.25
            or abs(origin.transform.location.z-point.z)>1.5
            or abs(wrap(math.radians(origin.transform.rotation.yaw-source['trigger_yaw_deg'])))>.25):
        raise NoSuitableSite('Original collection anchor has no same-position, same-direction, same-level driving lane')
    distance=float(source['event_offset_m'])
    reference=[origin] if distance==0. else list(origin.next(distance))
    if len(reference)!=1:raise NoSuitableSite('Original collection anchor has no unambiguous connected forward interaction segment')
    event=reference[0]
    starts=list(event.previous(float(hazard_station)))
    return starts,event


def _same_anchor_station(waypoints,station,event):
    actual=station_waypoint(waypoints,station)
    a,b=actual.transform,event.transform
    return (lane_key(actual)==lane_key(event)
            and math.hypot(a.location.x-b.location.x,a.location.y-b.location.y)<=1.5
            and abs(a.location.z-b.location.z)<=1.5
            and abs(wrap(math.radians(a.rotation.yaw-b.rotation.yaw)))<=.25)


def _spawn_candidates(road_map, poses):
    for pose in poses:
        origin = road_map.get_waypoint(pose.location)
        if origin is None:
            continue
        yield origin
        for offset in (5., 10., 15., 20., 25.):
            choices = list(origin.next(offset))
            if choices:
                yield min(choices, key=lambda wp: (abs(wrap(math.radians(
                    wp.transform.rotation.yaw-origin.transform.rotation.yaw))), lane_key(wp)))


def select_route(world, spec, *, near_xyz=None):
    road_map = world.get_map()
    if road_map.name.replace('\\', '/').split('/')[-1] != spec['map']:
        raise ValueError('actual development map differs from the selected catalog design')
    length = float(spec['approach_m'])+float(spec['interaction_m'])+float(spec['recovery_m'])+8.
    locations = list(road_map.get_spawn_points())
    if not locations or near_xyz is not None:
        locations.extend(wp.transform for wp in road_map.generate_waypoints(25.))
    random.Random(spec['seed']).shuffle(locations)
    if near_xyz is not None:
        locations.sort(key=lambda pose: math.dist(
            (pose.location.x, pose.location.y, pose.location.z), near_xyz))
    bypass = spec['interaction_kind'] in ('parked_bypass', 'construction_bypass', 'construction_crossing')
    capabilities = set(spec.get('required_capabilities', ()))
    allow_junction = spec['interaction_kind'] in ('cross_traffic', 'stop_sign')

    def usable_section(wp):
        if wp.is_junction and not allow_junction:
            return False
        if bypass or 'crossable_left_boundary' in capabilities:
            if left_neighbor(wp) is None:
                return False
        elif 'same_direction_left_lane' in capabilities and _driving_neighbor(wp, 'left') is None:
            return False
        if 'same_direction_right_lane' in capabilities and _driving_neighbor(wp, 'right') is None:
            return False
        if 'opposite_left_lane' in capabilities and _driving_neighbor(wp, 'left', same_direction=False) is None:
            return False
        return 'right_sidewalk' not in capabilities or _right_sidewalk(wp) is not None

    actors = list(world.get_actors())
    signs = [actor for actor in actors if actor.type_id.startswith('traffic.stop')]
    actor_stations = [float(actor['s_offset_m']) for actor in spec['actor_specs']]
    hazard_station = min(actor_stations)
    if bypass:
        return_start = max(float(actor['s_offset_m']) for actor in spec['actor_specs']
                           if actor.get('lane_offset', 0) == 0) + 10.
    # The complete return ends at last actor +25m in teaching_path. Screening
    # only the approach would return an unusable first candidate and prevent
    # searching later sites that really support the whole maneuver.
    end_station = max(float(spec['approach_m'])+float(spec['interaction_m']),
                      max(actor_stations)+25. if bypass else 0.)
    from .demo_catalog import validate_route_source
    source=validate_route_source(spec)
    event=None
    if source is not None:
        candidates,event=_anchor_candidates(road_map,locations,source,hazard_station)
    else:
        candidates = _spawn_candidates(road_map, locations)
    for start in candidates:
        if start is None or start.is_junction:
            continue
        if event is None and not allow_junction:
            # A unique sample well inside the examined section can reject a
            # clearly unsuitable road before tracing its entire recovery tail.
            ahead = list(start.next(12.))
            if len(ahead) == 1 and not usable_section(ahead[0]):
                continue
        waypoints = chain(start, length)
        if not waypoints:
            continue
        if event is not None and not _same_anchor_station(waypoints,hazard_station,event):
            continue
        stations = _stations(waypoints)
        if end_station > stations[-1]-2.:
            continue
        active = [wp for wp, station in zip(waypoints, stations) if 8. <= station <= end_station+1.]
        if any(abs(wrap(math.radians(b.transform.rotation.yaw-a.transform.rotation.yaw))) > .12
               for a, b in zip(waypoints, waypoints[1:])):
            continue
        if 'straight_route' in capabilities and any(abs(wrap(math.radians(
                wp.transform.rotation.yaw-waypoints[0].transform.rotation.yaw))) > .12
                for wp in active):
            continue
        if not all(usable_section(wp) for wp in active):
            continue
        if bypass and any(not _return_allowed(left_neighbor(wp))
                          for wp, s in zip(waypoints, stations)
                          if return_start < s < return_start+15.):
            continue
        if 'conflicting_junction' in capabilities:
            if not _conflicting_branch(waypoints, target_station=hazard_station):
                continue
            if crossing_drive_path(waypoints, hazard_station) is None:
                continue
        if 'real_stop_sign' in capabilities and not any(_stop_controls_route(
                sign, waypoints, target_station=hazard_station) for sign in signs):
            continue
        # full_lane_blockage and verified_passing_clearance need measured actor
        # bounds after spawning. They cannot be certified by topology selection.
        if _start_clear(start, actors):
            return waypoints
    if source is not None:
        raise NoSuitableSite('No legal short demonstration fits collection route %04d on its anchored lane; no substitute road was selected'%source['official_route_id'])
    raise NoSuitableSite('No legal development site satisfies '+spec['id']+'; try another explicit map/seed')


def select_demonstration_site(world, spec, *, allow_similar_road=False):
    if type(allow_similar_road) is not bool:
        raise ValueError('allow_similar_road must be boolean')
    source = spec.get('route_source')
    mode, reason = spec['route_policy'], None
    try:
        waypoints = select_route(world, spec)
    except NoSuitableSite as original_error:
        if not allow_similar_road or source is None:
            raise
        reason = str(original_error)
        local_spec = dict(spec, route_policy='independent_topology_selection')
        local_spec.pop('route_source')
        waypoints = select_route(world, local_spec, near_xyz=source['trigger_xyz_carla_m'])
        mode = 'same_map_similar_road'
    station = min(float(actor['s_offset_m']) for actor in spec['actor_specs'])
    event = station_waypoint(waypoints, station)
    position = event.transform.location
    return waypoints, dict(
        mode=mode, allow_similar_road=allow_similar_road, anchor_rejection=reason,
        designed_interaction_station_m=station, site_lane=lane_key(event),
        site_lane_center_xyz_carla_m=[position.x, position.y, position.z],
        site_lane_yaw_deg=event.transform.rotation.yaw)


def teaching_path(waypoints, spec):
    """Full smooth lane change and return, legal site verified before use."""
    route = [xy(wp.transform.location) for wp in waypoints]
    if spec['interaction_kind'] not in ('parked_bypass', 'construction_bypass', 'construction_crossing'):
        return route
    stations = [float(actor['s_offset_m']) for actor in spec['actor_specs']
                if actor.get('lane_offset', 0) == 0]
    first, last = min(stations), max(stations)
    result, arc = [], 0.
    for i, wp in enumerate(waypoints):
        if i:
            arc += math.dist(route[i], route[i-1])
        change_start, change_end = max(1., first-20.), first-8.
        return_start, return_end = last+10., last+25.
        def smooth(t):
            t = max(0., min(1., t))
            return 10*t**3-15*t**4+6*t**5
        amount = smooth((arc-change_start)/(change_end-change_start))*(1-smooth((arc-return_start)/(return_end-return_start)))
        if amount > 1e-9:
            adjacent = left_neighbor(wp)
            if wp.is_junction or adjacent is None or (arc > return_start and not _return_allowed(adjacent)):
                raise RuntimeError('Teaching path would cross an unverified lane boundary')
            target = xy(adjacent.transform.location)
            result.append([route[i][j]+amount*(target[j]-route[i][j]) for j in (0, 1)])
        else:
            result.append(route[i])
    return result


def station_waypoint(waypoints, station):
    arc = 0.
    for a, b in zip(waypoints, waypoints[1:]):
        arc += a.transform.location.distance(b.transform.location)
        if arc >= station:
            return b
    raise ValueError('Requested actor station lies outside development route')


def observed_progress(position, route, heading=None):
    value = project(position, route, heading=heading)
    if value['distance_m'] > 5.:
        raise ValueError('Development route projection residual exceeds 5m')
    return value
