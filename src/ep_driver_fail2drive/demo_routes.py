"""Independent development routes; no official test XML or scenario coordinates.

This module is SDK duck-typed and has no CARLA import. A failed site search is
an error, never a reason to relax crossing permissions or move a test finish.
"""
import math
import random

from .route import lane_key, xy, wrap, project


def chain(start, length, step=1.):
    points, arc = [start], 0.
    while arc < length:
        choices = list(points[-1].next(step))
        if not choices:
            return None
        previous = points[-1]
        nxt = min(choices, key=lambda wp: (abs(wrap(math.radians(
            wp.transform.rotation.yaw-previous.transform.rotation.yaw))), lane_key(wp)))
        gap = previous.transform.location.distance(nxt.transform.location)
        if not .05 < gap <= step*2.1:
            return None
        if any(nxt.transform.location.distance(wp.transform.location) < .3 for wp in points[:-8]):
            return None
        arc += gap
        points.append(nxt)
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


def select_route(world, spec):
    road_map = world.get_map()
    if 'Town13' in road_map.name:
        raise ValueError('demonstrations must use independent development geography, not official Town13')
    if road_map.name.replace('\\', '/').split('/')[-1] != spec['map']:
        raise ValueError('actual development map differs from the selected catalog design')
    length = float(spec['approach_m'])+float(spec['interaction_m'])+float(spec['recovery_m'])+8.
    locations = list(road_map.get_spawn_points())
    random.Random(spec['seed']).shuffle(locations)
    bypass = spec['interaction_kind'] in ('parked_bypass', 'construction_bypass', 'construction_crossing')
    capabilities = set(spec.get('required_capabilities', ()))
    actors = list(world.get_actors())
    signs = [actor for actor in actors if actor.type_id.startswith('traffic.stop')]
    actor_stations = [float(actor['s_offset_m']) for actor in spec['actor_specs']]
    hazard_station = min(actor_stations)
    # The complete return ends at last actor +25m in teaching_path. Screening
    # only the approach would return an unusable first candidate and prevent
    # searching later sites that really support the whole maneuver.
    end_station = max(float(spec['approach_m'])+float(spec['interaction_m']),
                      max(actor_stations)+25. if bypass else 0.)
    candidates = []
    for pose in locations:
        origin = road_map.get_waypoint(pose.location)
        if origin is None:
            continue
        for offset in (0., 5., 10., 15., 20., 25.):
            choices = [origin] if offset == 0 else list(origin.next(offset))
            if choices:
                candidates.append(min(choices, key=lambda wp: (abs(wrap(math.radians(
                    wp.transform.rotation.yaw-origin.transform.rotation.yaw))), lane_key(wp))))
    for start in candidates:
        if start is None or start.is_junction:
            continue
        if not _start_clear(start, actors):
            continue
        waypoints = chain(start, length)
        if not waypoints:
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
        if bypass and any(wp.is_junction or left_neighbor(wp) is None for wp in active):
            continue
        if 'same_direction_left_lane' in capabilities and any(
                _driving_neighbor(wp, 'left') is None for wp in active):
            continue
        if 'crossable_left_boundary' in capabilities and any(left_neighbor(wp) is None for wp in active):
            continue
        if 'same_direction_right_lane' in capabilities and any(
                _driving_neighbor(wp, 'right') is None for wp in active):
            continue
        if 'opposite_left_lane' in capabilities and any(
                _driving_neighbor(wp, 'left', same_direction=False) is None for wp in active):
            continue
        if 'right_sidewalk' in capabilities and any(_right_sidewalk(wp) is None for wp in active):
            continue
        if 'conflicting_junction' in capabilities and not _conflicting_branch(
                waypoints, target_station=hazard_station):
            continue
        if 'real_stop_sign' in capabilities and not any(_stop_controls_route(
                sign, waypoints, target_station=hazard_station) for sign in signs):
            continue
        # full_lane_blockage and verified_passing_clearance need measured actor
        # bounds after spawning. They cannot be certified by topology selection.
        if not bypass and spec['interaction_kind'] not in ('cross_traffic', 'stop_sign'):
            if any(wp.is_junction for wp in active):
                continue
        return waypoints
    raise RuntimeError('No legal development site satisfies '+spec['id']+'; try another explicit map/seed')


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
            if wp.is_junction or adjacent is None:
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
