"""Causal scripted teacher for independent, offline CARLA demonstrations only.

This module has no CARLA, numerical-worker, model, scenario-instance or event
timer dependency. It follows an authored reference using current observed
actors and public traffic rules. It is neither the paper planner nor a safety
certificate. The collector must independently check collisions, road/rule
compliance, full interaction completion and final arrival before admission.

Coordinates are metres in the public state's right-handed XY frame; steering
is a physical wheel angle, positive left. ``bypass_reference`` is an optional
complete teaching path with the same start and finish as ``route_xy``. Only
the teacher sees that path; the ordinary navigation remains ``route_xy``.
"""
from __future__ import annotations

import math
from bisect import bisect_left

from .route import point_in_polygon, polygons_intersect


SCHEMA = "ep-driver-independent-demo-teacher-1.0"

DEFAULT_SETTINGS = {
    "cruise_speed_mps": 5.0,
    "acceleration_min_mps2": -5.0,
    "acceleration_max_mps2": 2.0,
    "jerk_max_mps3": 6.0,
    "steering_max_rad": .6,
    "steering_rate_max_radps": .7,
    "speed_gain_per_s": 1.2,
    "lookahead_min_m": 3.0,
    "lookahead_time_s": .7,
    "lookahead_max_m": 9.0,
    "lateral_acceleration_max_mps2": 1.8,
    "service_deceleration_mps2": 2.5,
    "stop_speed_gain_per_s": .3,
    "stop_margin_m": 2.5,
    "following_distance_m": 5.0,
    "following_time_s": 1.6,
    "following_gain_per_s": .7,
    "corridor_margin_m": .45,
    "actor_front_m": 45.0,
    "actor_prediction_horizon_s": 3.0,
    "actor_prediction_step_s": .5,
    "crossing_arrival_tolerance_s": 1.0,
    "path_sample_step_m": .6,
    "clear_confirmation_s": .5,
    "stopped_speed_mps": .15,
    "finish_tolerance_m": .8,
    "projection_maximum_distance_m": 6.0,
    "projection_initial_window_m": 12.0,
    "projection_window_m": 12.0,
    "projection_backward_tolerance_m": 1.0,
    "projection_heading_limit_rad": 2.0,
    "maximum_step_s": 1.0,
}


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _point(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + " must be an XY pair")
    return (_number(value[0], name + "[0]"), _number(value[1], name + "[1]"))


def _clamp(value, low, high):
    return min(high, max(low, value))


def _wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


class _Path:
    def __init__(self, points, name):
        if not isinstance(points, (list, tuple)) or len(points) < 2:
            raise ValueError(name + " requires at least two XY points")
        self.points = [_point(p, name) for p in points]
        self.segments, self.length = [], 0.
        for a, b in zip(self.points, self.points[1:]):
            distance = math.dist(a, b)
            if distance < 1e-5:
                raise ValueError(name + " contains coincident consecutive points")
            self.segments.append((self.length, self.length + distance, a, b,
                                  math.atan2(b[1] - a[1], b[0] - a[0])))
            self.length += distance
        self.ends = [r[1] for r in self.segments]

    def at(self, station):
        station = _clamp(station, 0., self.length)
        segment = self.segments[bisect_left(self.ends, station)]
        start, end, a, b, heading = segment
        ratio = (station - start) / (end - start)
        return ((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])), heading)

    def project(self, point, heading=None, lower=0., upper=None):
        upper = self.length if upper is None else min(self.length, upper)
        lower = max(0., lower)
        choices = []
        for start, end, a, b, direction in self.segments:
            if end < lower or start > upper:
                continue
            length = end - start
            fraction = ((point[0] - a[0]) * (b[0] - a[0]) +
                        (point[1] - a[1]) * (b[1] - a[1])) / (length * length)
            fraction = _clamp(fraction, max(0., (lower - start) / length),
                              min(1., (upper - start) / length))
            station = start + fraction * length
            closest = (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))
            residual = math.dist(point, closest)
            error = abs(_wrap(direction - heading)) if heading is not None else 0.
            lateral = (-(point[0] - closest[0]) * math.sin(direction) +
                       (point[1] - closest[1]) * math.cos(direction))
            choices.append((residual + .15 * error, residual, station, error, lateral, direction))
        if not choices:
            raise ValueError("no teaching-path segment within the causal projection window")
        _, residual, station, error, lateral, direction = min(choices)
        return {"s_m": station, "distance_m": residual, "heading_error_rad": error,
                "lateral_m": lateral, "heading_rad": direction}


def _segment_distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = dx * dx + dy * dy
    fraction = _clamp(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / norm, 0., 1.) if norm else 0.
    return math.dist(point, (a[0] + fraction * dx, a[1] + fraction * dy))


def _body_distance(point, polygon):
    inside = False
    previous = polygon[-1]
    distance = float("inf")
    for current in polygon:
        distance = min(distance, _segment_distance(point, previous, current))
        if (previous[1] > point[1]) != (current[1] > point[1]):
            crossing = previous[0] + (point[1] - previous[1]) * (current[0] - previous[0]) / (current[1] - previous[1])
            if point[0] < crossing:
                inside = not inside
        previous = current
    return 0. if inside else distance


def _actor_body(actor):
    position = _point(actor.get("position"), "actor.position")
    velocity = _point(actor.get("velocity", [0., 0.]), "actor.velocity")
    heading = _number(actor.get("heading", 0.), "actor.heading")
    length = _number(actor.get("length", 1.), "actor.length")
    width = _number(actor.get("width", 1.), "actor.width")
    if length <= 0. or width <= 0.:
        raise ValueError("actor body dimensions must be positive")
    if actor.get("polygon") is not None:
        polygon = [_point(p, "actor.polygon") for p in actor["polygon"]]
        if len(polygon) < 3:
            raise ValueError("actor.polygon requires at least three points")
    else:
        c, s = math.cos(heading), math.sin(heading)
        polygon = [(position[0] + u * c - v * s, position[1] + u * s + v * c)
                   for u, v in ((length / 2., width / 2.), (length / 2., -width / 2.),
                                (-length / 2., -width / 2.), (-length / 2., width / 2.))]
    return position, velocity, polygon


def _body_box(center, yaw, half_length, half_width):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(center[0] + u*c-v*s, center[1]+u*s+v*c)
            for u, v in ((half_length, half_width), (half_length, -half_width),
                         (-half_length, -half_width), (-half_length, half_width))]


def _swept_polygon(polygon,velocity,t_min,t_max):
    """Convex hull of a body translated over a bounded arrival-time interval."""
    vertices=sorted(set((x+velocity[0]*t,y+velocity[1]*t) for t in (t_min,t_max) for x,y in polygon))
    def chain(points):
        result=[]
        for p in points:
            while len(result)>=2:
                a,b=result[-2:]
                if (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])>0.:
                    break
                result.pop()
            result.append(p)
        return result[:-1]
    return chain(vertices)+chain(reversed(vertices))


def _bypass_span(home, other):
    """Map a complete surveyed reference to stations of departure and return."""
    offset = []
    for index in range(math.ceil(other.length)+1):
        distance = min(float(index), other.length)
        center, yaw = other.at(distance)
        q = home.project(center, yaw)
        if abs(q['lateral_m']) > .20:
            offset.append((distance, q['s_m'], q['lateral_m']))
    if not offset:
        raise ValueError('Opposing bypass reference never leaves the home lane')
    return dict(entry=offset[0][1], exit=offset[-1][1], path_exit=offset[-1][0],
                offset=max((v[2] for v in offset),key=abs))


class _OpposingBorrow:
    """Select among surveyed paths from observed bodies and their current velocities."""
    def __init__(self, home, normal, early):
        self.home, self.normal, self.early = home, normal, early
        self.span, self.early_span = _bypass_span(home,normal), _bypass_span(home,early)
        if (self.span['offset'] <= 0 or self.early_span['offset'] <= 0
                or self.early_span['exit'] >= self.span['exit']
                or abs(self.span['entry'] - self.early_span['entry']) > 1.5):
            raise ValueError('Early return must share the entry and reach the home lane sooner')
        self.committed = self.shortened = self.returned = False
        self.clear_seconds = 0.
        self.blocker_tail = None
        self.denied_ids = set()
        self.entered = False
        self.peak_lateral = 0.

    def _footprint(self,scene,nav,length,width):
        ego=scene['ego']
        corners=_body_box(ego['position'],ego['heading'],length/2.,width/2.)
        route_ids=set(scene.get('route_lane_ids',[]))
        home=[lane for lane in scene.get('lanes',[]) if lane['id'] in route_ids]
        home_corners=sum(any(point_in_polygon(corner,lane['polygon']) for lane in home) for corner in corners)
        ego_lane=ego.get('lane_id')
        parts=str(ego_lane).split('|')
        opposite=False
        if len(parts)==3:
            for lane in home:
                key=lane['id'].split('|')
                if len(key)==3 and key[:2]==parts[:2] and int(key[2])*int(parts[2])<0:
                    opposite=True
                    break
        dx,dy=math.cos(nav['heading_rad']),math.sin(nav['heading_rad'])
        half_s=max(abs((x-ego['position'][0])*dx+(y-ego['position'][1])*dy) for x,y in corners)
        return dict(ego_lane_id=ego_lane,center_in_opposing_lane=opposite,
                    home_corners=home_corners,body_on_home_lane=bool(home) and home_corners==len(corners),
                    measured_lateral_m=nav['lateral_m'],ego_rear_s_m=nav['s_m']-half_s)

    def _traffic(self, scene, nav, ego_length, ego_width):
        s, oncoming, blockers, bodies = nav['s_m'], [], [], []
        actors=scene.get('actors',[])
        if not isinstance(actors,list) or any(not isinstance(actor,dict) for actor in actors):
            raise ValueError('scene.actors must be a list of current observations')
        for actor in actors:
            if actor.get('visible') is False:
                continue
            center, velocity, polygon = _actor_body(actor)
            extent = max(math.dist(center,p) for p in polygon)
            if math.dist(center,scene['ego']['position']) > 145.+extent:
                continue
            q = self.home.project(center)
            heading = q['heading_rad']
            c, sn = math.cos(heading), math.sin(heading)
            closest = self.home.at(q['s_m'])[0]
            # Preserve signed longitudinal distance for visible actors just beyond a path endpoint.
            along = [q['s_m'] + (x-closest[0])*c + (y-closest[1])*sn for x,y in polygon]
            side = [-(x-closest[0])*sn + (y-closest[1])*c for x,y in polygon]
            lo, hi = min(along), max(along)
            bodies.append((str(actor['id']),center,velocity,polygon,extent))
            kind = actor.get('kind')
            if (kind in ('static_obstacle','vehicle') and math.hypot(*velocity)<.25
                    and lo<=self.span['exit'] and hi>=self.span['entry']
                    and min(side)<ego_width/2.+.20 and max(side)>-ego_width/2.-.20):
                blockers.append((str(actor['id']),hi))
            direction = math.cos(_wrap(actor.get('heading',heading)-heading))
            closing = -(velocity[0]*c+velocity[1]*sn)
            if (kind!='vehicle' or direction>-.5 or hi<s-ego_length/2.-.5
                    or q['lateral_m']<self.span['offset']*.4 or min(side)>self.span['offset']+ego_width/2.+.45):
                continue
            oncoming.append(dict(id=str(actor['id']),front_s_m=lo,rear_s_m=hi,closing_mps=max(0.,closing)))
        return oncoming,blockers,bodies

    def _early_clear(self, scene, nav, bodies, length, width, speed, cruise, acceleration):
        ego = scene['ego']
        projection = self.early.project(ego['position'], ego['heading'],max(0.,nav['s_m']-6.),nav['s_m']+10.)
        if projection['distance_m']>.65 or projection['heading_error_rad']>.30:
            return False,'outside_early_return_tracking_corridor'
        polygons = scene.get('drivable_polygons') or []
        if not polygons:
            return False,'missing_drivable_road'
        start = projection['s_m']
        end = min(self.early.length,self.early_span['path_exit']+length/2.+1.)
        if end<=start:
            return False,'past_early_return'
        # Both whole authored references were already audited at their real site.
        # Recheck the forward footprint and measured actor boxes at the switch.
        count = math.ceil((end-start)/.5)
        ref_origin, ref_yaw=self.early.at(start)
        initial_error=(ego['position'][0]-ref_origin[0],ego['position'][1]-ref_origin[1])
        yaw_error=_wrap(ego['heading']-ref_yaw)
        v_max=max(speed,min(cruise,4.2))
        ramp_t=(v_max-speed)/acceleration
        ramp_s=(v_max+speed)*ramp_t/2.
        for i in range(count+1):
            delta = (end-start)*i/count
            center, yaw = self.early.at(start+delta)
            tracking_error=max(0.,1.-delta/5.)
            center=(center[0]+tracking_error*initial_error[0],center[1]+tracking_error*initial_error[1])
            box = _body_box(center,yaw+tracking_error*yaw_error,length/2.+.2,width/2.+.12)
            if any(not any(point_in_polygon(pt,poly) for poly in polygons) for pt in box):
                return False,'return_footprint_outside_road'
            fastest=((math.sqrt(speed*speed+2.*acceleration*delta)-speed)/acceleration
                     if delta<=ramp_s else ramp_t+(delta-ramp_s)/v_max)
            slowest=max(fastest,delta/max(.5,speed))+.35
            for identity,pos,velocity,polygon,radius in bodies:
                begin=(pos[0]+velocity[0]*fastest,pos[1]+velocity[1]*fastest)
                final=(pos[0]+velocity[0]*slowest,pos[1]+velocity[1]*slowest)
                if _segment_distance(center,begin,final)>math.hypot(length/2.+.2,width/2.+.12)+radius:
                    continue
                if polygons_intersect(box,_swept_polygon(polygon,velocity,fastest,slowest)):
                    return False,'return_conflict_with_'+identity
        return True,'sampled_road_and_observed_motion_clear'

    def select(self, scene, nav, speed, length, width, dt, cruise, acceleration):
        s = nav['s_m']
        if self.returned:
            return self.home, None, dict(stage='returned',entered_opposing_lane=True,
                return_path='early' if self.shortened else 'nominal',**self._footprint(scene,nav,length,width))
        oncoming,blockers,bodies = self._traffic(scene,nav,length,width)
        if blockers:
            far_edge = max(tail for _,tail in blockers)
            self.blocker_tail = far_edge if self.blocker_tail is None else max(self.blocker_tail,far_edge)
        exit_s = self.span['exit']+length/2.+.75
        intended_speed = min(cruise,4.2)
        distance = max(0.,exit_s-s)
        # Conservative time to get the whole body home, including acceleration from a stop.
        ramp = max(0.,intended_speed-speed)
        needed = distance/intended_speed + ramp*ramp/(2.*acceleration*intended_speed) + 1.5
        denied=[]
        for row in oncoming:
            space = row['front_s_m']-exit_s
            closing = row['closing_mps']
            row['arrival_margin_s'] = space/closing-needed if closing>.15 else None
            if space<.5 or (closing>.15 and row['arrival_margin_s']<0):
                denied.append(row['id'])
        evidence=dict(stage='approach',observed_oncoming=oncoming,denied_ids=denied,
            observed_blocker_ids=[b[0] for b in blockers],observed_blocker_rear_limit_s_m=self.blocker_tail,
            needed_clearance_time_s=needed,nominal_return_s_m=self.span['exit'],early_return_s_m=self.early_span['exit'])
        footprint=self._footprint(scene,nav,length,width)
        evidence.update(footprint)
        if self.committed and not self.entered and denied and footprint['body_on_home_lane'] and abs(nav['lateral_m'])<.35:
            self.committed=False
        if not self.committed:
            self.denied_ids.update(denied)
            self.clear_seconds=0. if denied else self.clear_seconds+dt
            if self.blocker_tail is None or self.clear_seconds<.5 or s<self.span['entry']-length-3.:
                hold_distance = max(0.,self.span['entry']-s-length/2.-.4) if denied or self.denied_ids else None
                evidence.update(stage='waiting' if denied and speed<=.15 and hold_distance<=.6 else 'approach',
                                hold_distance_m=hold_distance)
                return self.home, hold_distance, evidence
            self.committed=True
        self.entered |= footprint['center_in_opposing_lane']
        self.peak_lateral=max(self.peak_lateral,nav['lateral_m'])
        passed=self.blocker_tail is not None and footprint['ego_rear_s_m']>self.blocker_tail+.35
        evidence.update(entered_opposing_lane=self.entered,yielded_to_ids=sorted(self.denied_ids),obstacle_cleared=passed)
        if self.entered and passed and footprint['body_on_home_lane'] and abs(nav['lateral_m'])<.35:
            self.returned=True
            evidence.update(stage='returned',return_path='early' if self.shortened else 'nominal')
            return self.home, None, evidence
        evidence['stage']='entering' if not self.entered else (
            'returning' if passed and self.peak_lateral-nav['lateral_m']>.15 else 'borrowed')
        # Once borrowing, an available earlier return reduces exposure to an
        # approaching car even while a narrow nominal time surplus remains.
        approaching=[row for row in oncoming if row['closing_mps']>.25 and row['arrival_margin_s']<=2.]
        if self.entered and not self.shortened and approaching and passed:
            evidence['early_return_margin_threshold_s']=2.
            self.shortened,evidence['return_clearance'] = self._early_clear(scene,nav,bodies,length,width,speed,cruise,acceleration)
            if self.shortened:
                evidence['early_return_trigger_id']=min(approaching,key=lambda row:row['front_s_m'])['id']
        evidence['return_path']='early' if self.shortened else 'nominal'
        return self.early if self.shortened else self.normal, None, evidence


class DemoTeacher:
    """Follow an offline teaching reference; never used by formal evaluation.

    Invalid inputs/projections raise ``ValueError`` so the collector records an
    unsuccessful demonstration. No hidden teleport, automatic success, model
    call, scenario timer or emergency fallback is provided here.
    """
    def __init__(self, route_xy, wheelbase_m, settings=None, bypass_reference=None, *,
                 opposing_borrow=False, early_return_reference=None):
        self.settings = dict(DEFAULT_SETTINGS)
        supplied = dict(settings or {})
        unknown = set(supplied) - set(self.settings)
        if unknown:
            raise ValueError("unknown demonstration teacher settings: " + ", ".join(sorted(unknown)))
        self.settings.update(supplied)
        for name, value in self.settings.items():
            self.settings[name] = _number(value, name)
        p = self.settings
        for name, value in p.items():
            if name != "acceleration_min_mps2" and value <= 0.:
                raise ValueError(name + " must be positive")
        if p["acceleration_min_mps2"] >= 0.:
            raise ValueError("acceleration_min_mps2 must be negative")
        if p["lookahead_max_m"] < p["lookahead_min_m"]:
            raise ValueError("lookahead maximum is smaller than minimum")
        if p["service_deceleration_mps2"] > -p["acceleration_min_mps2"]:
            raise ValueError("service deceleration exceeds the actuator deceleration limit")
        self.wheelbase = _number(wheelbase_m, "wheelbase_m")
        if self.wheelbase <= 0.:
            raise ValueError("wheelbase_m must be positive")
        self.navigation = _Path(route_xy, "route_xy")
        self.path = _Path(bypass_reference, "bypass_reference") if bypass_reference is not None else self.navigation
        if any(math.dist(self.path.points[index], self.navigation.points[index]) > .1 for index in (0, -1)):
            raise ValueError("bypass_reference must retain the original navigation start and finish")
        self.borrow = None
        if opposing_borrow or early_return_reference is not None:
            if not opposing_borrow or early_return_reference is None or bypass_reference is None:
                raise ValueError("Opposing borrow requires both surveyed bypass references")
            early = _Path(early_return_reference,'early_return_reference')
            if any(math.dist(early.points[i],self.navigation.points[i])>.1 for i in (0,-1)):
                raise ValueError("Early return must retain the original navigation start and finish")
            self.borrow = _OpposingBorrow(self.navigation,self.path,early)
            self.path = self.navigation
        self.last_frame, self.last_time, self.last_position = None, None, None
        self.nav_s, self.path_s = None, None
        self.acceleration, self.steering = 0., 0.
        self.waiting = False
        self.clear_elapsed = 0.
        self.observed_interaction = False
        self.bypass_peak = 0.

    def _projection(self, path, position, heading, previous_s, speed, dt):
        p = self.settings
        lower = 0. if previous_s is None else previous_s - p["projection_backward_tolerance_m"]
        upper = (p["projection_initial_window_m"] if previous_s is None else
                 previous_s + max(p["projection_window_m"], speed * dt + 2.))
        result = path.project(position, heading, lower, upper)
        if result["distance_m"] > p["projection_maximum_distance_m"]:
            raise ValueError("teaching-path projection residual exceeds maximum distance")
        if result["heading_error_rad"] > p["projection_heading_limit_rad"]:
            raise ValueError("ego heading is incompatible with the teaching path")
        return result

    def _samples(self, station, ego_length):
        p = self.settings
        start = max(0., station - ego_length / 2.)
        end = min(self.path.length, station + p["actor_front_m"])
        count = max(1, int(math.ceil((end - start) / p["path_sample_step_m"])))
        return [(start + (end - start) * i / count, self.path.at(start + (end - start) * i / count)[0])
                for i in range(count + 1)]

    def _stop_speed(self, distance):
        """Include velocity-loop lag instead of treating sqrt(2bd) as a command.

        The linear tail approaches a finite target conservatively. A raw
        constant-deceleration envelope with a lagging speed servo overshoots
        the line/finish even in an ideal bicycle replay.
        """
        distance = max(0., distance)
        return min(math.sqrt(2. * self.settings["service_deceleration_mps2"] * distance),
                   self.settings["stop_speed_gain_per_s"] * distance)

    def _actors(self, scene, station, speed, ego_length, ego_width):
        p = self.settings
        samples = self._samples(station, ego_length)
        corridor = ego_width / 2. + p["corridor_margin_m"]
        cap, conflicts, must_stop = float("inf"), [], False
        actors = scene.get("actors", [])
        if not isinstance(actors, list):
            raise ValueError("scene.actors must be a list of current observations")
        for actor in actors:
            if not isinstance(actor, dict):
                raise ValueError("each observed actor must be a mapping")
            if actor.get("visible") is False:
                continue
            _, velocity, polygon = _actor_body(actor)
            # A spatial bound only avoids unnecessary polygon tests. Inclusion
            # itself uses the authored path corridor, not a circular range.
            center = _point(actor.get("position"), "actor.position")
            extent = max(math.dist(center, point) for point in polygon)
            motion_range = math.hypot(*velocity) * p["actor_prediction_horizon_s"]
            nearest, near_station = min((math.dist(center, point), s) for s, point in samples)
            # A person or cyclist beside the intended path can enter between
            # observations, even with zero measured speed on the current frame.
            if (actor.get('kind') in ('pedestrian','bicycle') and station-1. <= near_station <= station+14.
                    and nearest-extent < corridor+2.):
                cautious = 2.8 + max(0., near_station-station-4.)*.15
                cap = min(cap,cautious)
                conflicts.append(dict(id=str(actor.get('id','unidentified')),reason='nearby_vulnerable_road_user',
                    speed_cap_mps=cautious,requires_stop=False,evidence='current_actor_body_and_velocity'))
            if nearest > corridor + extent + motion_range:
                continue
            occupied = [(s, point) for s, point in samples if _body_distance(point, polygon) <= corridor]
            if occupied:
                actor_station = occupied[0][0]
                gap = actor_station - station - ego_length / 2.
                direction = self.path.at(actor_station)[1]
                along_speed = velocity[0] * math.cos(direction) + velocity[1] * math.sin(direction)
                desired_gap = p["following_distance_m"] + p["following_time_s"] * speed
                following_cap = max(0., along_speed + p["following_gain_per_s"] * (gap - desired_gap))
                if along_speed <= .25:
                    stop_cap = self._stop_speed(gap - p["stop_margin_m"])
                    actor_cap = min(following_cap, stop_cap)
                else:
                    actor_cap = following_cap
                cap = min(cap, actor_cap)
                stopping = actor_cap <= .05
                must_stop |= stopping
                conflicts.append({"id": str(actor.get("id", "unidentified")), "reason": "observed_path_occupancy",
                                  "gap_m": gap, "observed_along_speed_mps": along_speed,
                                  "speed_cap_mps": actor_cap, "requires_stop": stopping,
                                  "evidence": "current_actor_body_and_velocity"})
                continue
            if math.hypot(*velocity) < .05:
                continue
            steps = max(1, int(math.ceil(p["actor_prediction_horizon_s"] / p["actor_prediction_step_s"])))
            for index in range(1, steps + 1):
                offset = p["actor_prediction_horizon_s"] * index / steps
                shifted = [(x + velocity[0] * offset, y + velocity[1] * offset) for x, y in polygon]
                hits = [s for s, point in samples if _body_distance(point, shifted) <= corridor]
                plausible = [s for s in hits if abs(max(0., s - station) / max(speed, 1.) - offset) <= p["crossing_arrival_tolerance_s"]]
                if not plausible:
                    continue
                gap = min(plausible) - station - ego_length / 2.
                actor_cap = self._stop_speed(gap - p["stop_margin_m"])
                cap = min(cap, actor_cap)
                stopping = actor_cap <= .05
                must_stop |= stopping
                conflicts.append({"id": str(actor.get("id", "unidentified")), "reason": "observed_motion_crosses_path",
                                  "gap_m": gap, "prediction_offset_s": offset, "speed_cap_mps": actor_cap,
                                  "requires_stop": stopping,
                                  "evidence": "constant_current_velocity_extrapolation_not_event_future"})
                break
        return cap, conflicts, must_stop

    def step(self, scene, dt):
        """Return one source-frame bound signed command without changing a world.

        Required input is the plain mapping returned by PublicStateExtractor.
        ``rules.stop_s`` is a station on the original navigation, not the
        teaching detour. Input shape/frame/geometry errors are explicit errors.
        """
        dt = _number(dt, "dt")
        p = self.settings
        if not 0. < dt <= p["maximum_step_s"]:
            raise ValueError("dt must be positive and within maximum_step_s")
        if not isinstance(scene, dict) or not isinstance(scene.get("ego"), dict):
            raise ValueError("scene must contain an ego mapping")
        time = _number(scene.get("time"), "scene.time")
        diagnostics = scene.get("diagnostics", {})
        frame = diagnostics.get("carla_frame", scene.get("frame"))
        if frame is None:
            try:
                frame = int(str(scene.get("frame_id", "")).rsplit(":", 1)[-1])
            except ValueError as exc:
                raise ValueError("scene must identify its physical source frame") from exc
        if type(frame) is not int or frame < 0:
            raise ValueError("scene source frame must be a nonnegative integer")
        if self.last_frame is not None and (frame <= self.last_frame or time <= self.last_time):
            raise ValueError("demonstration source frames and times must increase")
        if self.last_time is not None and abs(time - self.last_time - dt) > max(1e-5, dt * .01):
            raise ValueError("dt does not match elapsed observed simulation time")
        ego = scene["ego"]
        position = _point(ego.get("position"), "ego.position")
        velocity = _point(ego.get("velocity"), "ego.velocity")
        heading = _number(ego.get("heading"), "ego.heading")
        length = _number(ego.get("length"), "ego.length")
        width = _number(ego.get("width"), "ego.width")
        if length <= 0. or width <= 0.:
            raise ValueError("ego dimensions must be positive")
        speed = max(0., velocity[0] * math.cos(heading) + velocity[1] * math.sin(heading))
        if self.last_position is not None:
            reachable = max(1., speed * dt + .5 * p["acceleration_max_mps2"] * dt * dt + .5)
            if math.dist(position, self.last_position) > reachable:
                raise ValueError("observed displacement exceeds the causal teaching projection window")
        nav = self._projection(self.navigation, position, heading, self.nav_s, speed, dt)
        hold_distance = borrow = None
        if self.borrow:
            reference,hold_distance,borrow = self.borrow.select(scene,nav,speed,length,width,dt,
                p['cruise_speed_mps'],p['acceleration_max_mps2'])
            if reference is not self.path:
                self.path = reference
                self.path_s = self.path.project(position,heading,max(0.,nav['s_m']-6.),nav['s_m']+10.)['s_m']
        projection = self._projection(self.path, position, heading, self.path_s, speed, dt)
        station = projection["s_m"]
        lookahead = _clamp(p["lookahead_min_m"] + speed * p["lookahead_time_s"],
                           p["lookahead_min_m"], p["lookahead_max_m"])
        target_point, target_heading = self.path.at(station + lookahead)
        dx, dy = target_point[0] - position[0], target_point[1] - position[1]
        # Ego observations are body centres; pure pursuit is evaluated from the
        # real rear axle when its measured centre offset is available.
        axle_offset = _number(ego.get("rear_axle_to_center_m") or 0., "ego.rear_axle_to_center_m")
        dx += axle_offset * math.cos(heading)
        dy += axle_offset * math.sin(heading)
        lateral = -dx * math.sin(heading) + dy * math.cos(heading)
        distance_squared = max(.01, dx * dx + dy * dy)
        curvature = 2. * lateral / distance_squared
        requested_steering = math.atan(self.wheelbase * curvature)
        requested_steering = _clamp(requested_steering, -p["steering_max_rad"], p["steering_max_rad"])
        steering = _clamp(requested_steering, self.steering - p["steering_rate_max_radps"] * dt,
                           self.steering + p["steering_rate_max_radps"] * dt)
        rules = scene.get("rules", {})
        if not isinstance(rules, dict):
            raise ValueError("scene.rules must be a mapping")
        target_speed = p["cruise_speed_mps"]
        for name in ("desired_speed", "speed_limit"):
            if rules.get(name) is not None:
                value = _number(rules[name], "rules." + name)
                if value < 0.:
                    raise ValueError("public speed constraints cannot be negative")
                target_speed = min(target_speed, value)
        path_curvature = abs(_wrap(target_heading - projection["heading_rad"])) / max(lookahead, .1)
        curvature_bound = max(abs(curvature), path_curvature)
        if curvature_bound > 1e-5:
            target_speed = min(target_speed, math.sqrt(p["lateral_acceleration_max_mps2"] / curvature_bound))
        remaining = max(0., self.path.length - station)
        target_speed = min(target_speed, self._stop_speed(remaining))
        reasons = ["authored_reference_tracking"]
        must_stop = False
        conflicts = []
        if hold_distance is not None:
            hold_cap = self._stop_speed(hold_distance)
            target_speed = min(target_speed,hold_cap)
            must_stop = hold_cap<=.05
            reasons.append('hold_home_lane_until_borrow_clear')
        target_navigation_offset = self.navigation.project(target_point,
            lower=max(0., nav["s_m"] - 2.), upper=nav["s_m"] + lookahead + 15.)["distance_m"]
        if self.path is not self.navigation and max(abs(nav['lateral_m']),target_navigation_offset)>.30:
            target_speed=min(target_speed,4.2)
            reasons.append('limited_speed_on_lateral_maneuver')
        if rules.get("stop_s") is not None:
            stop_station = _number(rules["stop_s"], "rules.stop_s")
            gap = stop_station - nav["s_m"] - length / 2. - p["stop_margin_m"]
            if gap >= -length:
                stop_cap = self._stop_speed(gap)
                target_speed = min(target_speed, stop_cap)
                must_stop |= stop_cap <= .05
                conflicts.append({"id": "public_stop_line", "reason": "current_public_traffic_rule",
                                  "navigation_stop_s_m": stop_station, "front_clearance_m": gap,
                                  "speed_cap_mps": stop_cap, "requires_stop": stop_cap <= .05})
                reasons.append("current_public_stop_line")
        actor_cap, actor_conflicts, actor_stop = self._actors(scene, station, speed, length, width)
        target_speed = min(target_speed, actor_cap)
        conflicts.extend(actor_conflicts)
        must_stop |= actor_stop
        if actor_conflicts:
            reasons.append("current_observed_actor_conflict")
        waiting = self.waiting or (must_stop and speed <= p["stopped_speed_mps"])
        clear_elapsed = 0. if must_stop else self.clear_elapsed + dt
        if waiting and not must_stop and clear_elapsed < p["clear_confirmation_s"]:
            target_speed = 0.
            reasons.append("confirming_current_corridor_clear")
        elif waiting and not must_stop:
            waiting = False
            reasons.append("observed_corridor_clear_resume")
        elif must_stop:
            target_speed = 0.
        finish_arrived = (remaining <= p["finish_tolerance_m"] and
                          math.dist(position, self.path.points[-1]) <= p["finish_tolerance_m"] and
                          speed <= p["stopped_speed_mps"])
        if finish_arrived:
            target_speed = 0.
        desired_acceleration = _clamp(p["speed_gain_per_s"] * (target_speed - speed),
                                      p["acceleration_min_mps2"], p["acceleration_max_mps2"])
        acceleration = _clamp(desired_acceleration, self.acceleration - p["jerk_max_mps3"] * dt,
                              self.acceleration + p["jerk_max_mps3"] * dt)
        interaction = self.observed_interaction or bool(conflicts)
        offset = abs(nav["lateral_m"])
        peak = max(self.bypass_peak, offset)
        if finish_arrived:
            phase, target_speed = "DONE", 0.
            reasons.append("observed_demonstration_finish_not_official_score")
        elif waiting:
            phase = "WAIT"
        elif self.path is not self.navigation and (offset > .35 or target_navigation_offset > .35):
            phase = "RECOVER" if peak > .5 and offset < peak - .2 else "BYPASS"
        elif interaction or peak > .5:
            phase = "RECOVER" if not conflicts else "APPROACH"
        else:
            phase = "APPROACH"
        output = {"schema": SCHEMA, "acceleration_mps2": acceleration, "steering_rad_left": steering,
                  "target_speed_mps": target_speed, "phase": phase, "source_frame": frame,
                  "source_time": time, "reason": ";".join(reasons), "observed_conflicts": conflicts,
                  "projection": {"navigation": nav, "teaching": projection},
                  "tracking_target_xy": list(target_point), "teaching_remaining_m": remaining,
                  "safety_certified": False, "formal_evaluation_control": False}
        if borrow is not None:
            borrow.update(source_frame=frame,source_time=time)
            output['borrow'] = borrow
        self.last_frame, self.last_time, self.last_position = frame, time, position
        self.nav_s = nav["s_m"] if self.nav_s is None else max(self.nav_s, nav["s_m"])
        self.path_s = station if self.path_s is None else max(self.path_s, station)
        self.acceleration, self.steering = acceleration, steering
        self.waiting, self.clear_elapsed = waiting, clear_elapsed
        self.observed_interaction, self.bypass_peak = interaction, peak
        return output
