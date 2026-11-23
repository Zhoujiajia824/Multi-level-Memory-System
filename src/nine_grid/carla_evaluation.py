"""Physical-tick CARLA experiment evaluation, independent of planner diagnostics.

Python 3.7 and standard library only. Positions/headings use the bridge's
world XY convention (CARLA y and yaw negated). This module never ticks or
controls the world. Sensor occurrence frames, not callback times, own events.
"""
import copy
import json
import math
from pathlib import Path


VERSION = 'carla-physical-evaluation-2.2'
DEFAULTS = {
    'arrival_margin_m': 3., 'arrival_lateral_m': 2.5,
    'speed_tolerance_mps': .5, 'overspeed_grace_s': .5,
    'offroad_grace_s': .1, 'wrong_way_grace_s': .3,
    'stall_seconds': 30., 'stall_progress_m': .5,
    'stop_speed_mps': .15, 'stop_hold_s': 1.,
    'body_sample_spacing_m': .35, 'required_event_ids': [],
    'route_projection_max_distance_m': 6.,
    'strict_event_completion': True,
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _norm(vector):
    return math.hypot(float(vector[0]), float(vector[1]))


def _dot(a, b):
    return a[0]*b[0]+a[1]*b[1]


def _sub(a, b):
    return [a[0]-b[0], a[1]-b[1]]


def _lane_id(waypoint):
    return None if waypoint is None else '{}|{}|{}'.format(waypoint.road_id, waypoint.section_id, waypoint.lane_id)


def _native_waypoint_evidence(waypoint):
    transform = waypoint.transform
    return {'lane_id': _lane_id(waypoint),
        'position': [float(transform.location.x), -float(transform.location.y)],
        'z': float(transform.location.z), 'heading_rad': -math.radians(transform.rotation.yaw),
        'width_m': float(waypoint.lane_width), 'lane_type': str(waypoint.lane_type).split('.')[-1]}


def _boundary_topology_evidence(route_waypoints, checkpoints):
    """Raw public map queries; archived once for independent offline replay."""
    result = []
    for gate in checkpoints:
        waypoint = route_waypoints[gate['index']]
        item = {'index': gate['index'], 'route_waypoint': _native_waypoint_evidence(waypoint),
                'distance_m': 3., 'previous': [], 'next': []}
        for direction in ('previous', 'next'):
            method = getattr(waypoint, direction, None)
            if method is not None:
                item[direction] = [_native_waypoint_evidence(other) for other in method(3.)]
        result.append(item)
    return result


def _boundary_links(evidence):
    result = {}
    for item in evidence:
        source = item['route_waypoint']
        result[item['index']] = [other['lane_id'] for direction in ('previous', 'next')
            for other in item[direction]
            if math.cos(other['heading_rad']-source['heading_rad']) > .5
            and abs(other['z']-source['z']) < 1.5 and other['lane_type'] == 'Driving']
    return result


def _boundary_topology(route_waypoints, checkpoints):
    """Public native directed links, used only at a gate's 3m boundary zone."""
    return _boundary_links(_boundary_topology_evidence(route_waypoints, checkpoints))


def _box_corners(actor):
    c, s = math.cos(actor['heading']), math.sin(actor['heading'])
    return [[actor['position'][0]+c*x-s*y, actor['position'][1]+s*x+c*y]
            for x, y in ((-actor['length']/2, -actor['width']/2),
                         (actor['length']/2, -actor['width']/2),
                         (actor['length']/2, actor['width']/2),
                         (-actor['length']/2, actor['width']/2))]


def _convex_intersects(a, b):
    for polygon in (a, b):
        for p, q in zip(polygon, polygon[1:]+polygon[:1]):
            axis = [p[1]-q[1], q[0]-p[0]]
            pa, pb = [_dot(x, axis) for x in a], [_dot(x, axis) for x in b]
            if max(pa) < min(pb)-1e-8 or max(pb) < min(pa)-1e-8:
                return False
    return True


def _sample_rect(ego, spacing, edges_only=True):
    """Actual oriented box perimeter plus center; explicitly sampled geometry."""
    length, width = float(ego['length']), float(ego['width'])
    nx, ny = max(1, int(math.ceil(length/spacing))), max(1, int(math.ceil(width/spacing)))
    c, s = math.cos(ego['heading']), math.sin(ego['heading'])
    points = [list(ego['position'])]
    for ix in range(nx+1):
        for iy in range(ny+1):
            if edges_only and ix not in (0, nx) and iy not in (0, ny):
                continue
            x, y = length*(ix/nx-.5), width*(iy/ny-.5)
            points.append([ego['position'][0]+c*x-s*y, ego['position'][1]+s*x+c*y])
    return points


class EpisodeEvaluator:
    """Consume every completed physical frame, regardless of planner frequency.

    ``road`` supplies measured body_on_road, wrong_way, speed_limit_mps,
    optional expected_wait and coverage flags. Missing measurements stay unknown.
    ``actors`` is the unfiltered physical actor collection, never the MPC subset.
    """
    def __init__(self, route_xy, settings=None, physics_hz=50, directory=None):
        if type(physics_hz) is not int or physics_hz <= 0:
            raise ValueError('evaluation physics_hz must be a positive integer')
        self.hz, self.dt = physics_hz, 1./physics_hz
        self.settings = copy.deepcopy(DEFAULTS)
        self.settings.update(settings or {})
        for key in DEFAULTS:
            if key not in ('required_event_ids', 'strict_event_completion'):
                value = self.settings[key]
                if not _finite(value) or value < 0 or (key == 'body_sample_spacing_m' and value == 0):
                    raise ValueError('invalid evaluation setting: '+key)
        self.route = [[float(p[0]), float(p[1])] for p in route_xy]
        if len(self.route) < 2 or any(not all(_finite(v) for v in p) for p in self.route):
            raise ValueError('evaluation needs a finite locked route')
        self.arc = [0.]
        for a, b in zip(self.route, self.route[1:]):
            self.arc.append(self.arc[-1]+_norm(_sub(b, a)))
        if self.arc[-1] <= 0:
            raise ValueError('evaluation route has zero length')
        end_delta = _sub(self.route[-1], self.route[-2])
        self.goal_heading = math.atan2(end_delta[1], end_delta[0])
        self.projection_metadata = None
        self.directory = Path(directory) if directory is not None else None
        self.tick, self.frame, self.first_frame = None, None, None
        self.timestamp, self.first_timestamp = None, None
        self.start_s = self.progress_s = self.lateral = None
        self.route_projection = {'valid': False, 'reason': 'not_observed', 'distance_m': None}
        self.projection_invalid_ticks = 0
        self.previous = None
        self.actual_distance_m = 0.
        self.arrival, self.stalled = False, False
        self.violations = {'offroad': False, 'wrong_way': False, 'speeding': False}
        self.seconds = {'offroad': 0., 'wrong_way': 0., 'speeding': 0., 'stopped': 0., 'expected_wait': 0.}
        self.max_values = {'speed_mps': 0., 'acceleration_mps2': 0., 'jerk_mps3': 0., 'lateral_acceleration_mps2': 0.}
        self._comfort_squared = {'acceleration_mps2': 0., 'jerk_mps3': 0., 'lateral_acceleration_mps2': 0.}
        self._comfort_samples = {key: 0 for key in self._comfort_squared}
        self._invalid_acceleration_samples = 0
        self.min_gap = self.min_headway = self.min_ttc = None
        self.stall_origin_s = self.stall_origin_tick = None
        self._event_keys, self._pending, self.events = set(), [], []
        self.initialization_events, self.unknown_events = [], []
        self.coverage = {'body_on_road': True, 'wrong_way': True, 'speed_limit': True,
                         'collision_events': False, 'lane_markings': False, 'traffic_lights': False,
                         'stop_signs': False, 'public_priority': False, 'complete_physical_ticks': True}
        self.infrastructure_errors = []
        self.scenario = {'scenario_valid': None, 'events': []}
        self._scenario_event_rows = {}
        self.last_road, self.initial_state = {}, None

    def _write(self, name, value):
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')

    def project(self, position, heading=None, global_search=False, ego=None):
        best = None
        travel = 0. if self.previous is None else _norm(_sub(position, self.previous['position']))
        speed = 0. if self.previous is None else _norm(self.previous.get('velocity', [0., 0.]))
        discontinuity = travel > max(5., 3.*speed*self.dt+1.)
        anchor = 0. if self.progress_s is None else self.progress_s
        low = 0. if global_search else max(0., anchor-5.)
        high = self.arc[-1] if global_search else min(self.arc[-1], anchor+max(5., min(travel, 5.)*2.+2.))
        for index, (a, b) in enumerate(zip(self.route, self.route[1:])):
            if self.arc[index+1] < low or self.arc[index] > high:
                continue
            delta = _sub(b, a)
            length = self.arc[index+1]-self.arc[index]
            if length <= 1e-9:
                continue
            t = max(max(0., (low-self.arc[index])/length),
                    min(min(1., (high-self.arc[index])/length), _dot(_sub(position, a), delta)/(length*length)))
            q = [a[0]+t*delta[0], a[1]+t*delta[1]]
            residual = _sub(position, q)
            distance2 = _dot(residual, residual)
            alignment = 0. if heading is None else .05*(1.-(math.cos(heading)*delta[0]+math.sin(heading)*delta[1])/length)
            result = (distance2+alignment, index, self.arc[index]+t*length,
                      (-delta[1]*residual[0]+delta[0]*residual[1])/length, math.sqrt(distance2))
            if best is None or result[:2] < best[:2]:
                best = result
        if best is None:
            self.route_projection = {'valid': False, 'reason': 'no_local_segment', 'distance_m': None,
                                     'candidate_s_m': None, 'search_s_m': [low, high]}
            return None, None, None
        maximum = self.settings['route_projection_max_distance_m']
        corridor = (self.projection_metadata(best[1], ego, best[2])
                    if callable(self.projection_metadata) and ego is not None else None)
        if corridor is not None and corridor.get('contains_observed_center'):
            maximum = max(maximum, corridor['maximum_distance_m'])
        valid = best[4] <= maximum and not discontinuity
        self.route_projection = {'valid': valid, 'distance_m': best[4], 'candidate_s_m': best[2],
            'reason': ('physical_displacement_discontinuity' if discontinuity else
                       'within_local_route' if valid else 'outside_local_route_distance'),
            'max_distance_m': maximum, 'cross_section': corridor, 'search_s_m': [low, high],
            'relocalization': 'local_only_no_global_jump'}
        return best[2], best[3], best[4]

    def point_at(self, distance):
        distance = max(0., min(float(distance), self.arc[-1]))
        for i in range(len(self.route)-1):
            if distance <= self.arc[i+1] and self.arc[i+1] > self.arc[i]:
                t = (distance-self.arc[i])/(self.arc[i+1]-self.arc[i])
                return [self.route[i][j]+t*(self.route[i+1][j]-self.route[i][j]) for j in (0, 1)]
        return list(self.route[-1])

    def infrastructure_error(self, kind, details=None):
        item = {'kind': str(kind), 'details': details, 'physical_tick': self.tick, 'frame': self.frame}
        if item not in self.infrastructure_errors:
            self.infrastructure_errors.append(item)
            self._write('evaluation_events.jsonl', dict(item, category='infrastructure'))

    def ingest_events(self, events):
        """Idempotent and safe to call again after sensors stop; late events retained."""
        for source in events or ():
            event = copy.deepcopy(source)
            kind, frame = event.get('kind'), event.get('frame')
            if kind not in ('collision', 'lane_invasion', 'red_light', 'stop_sign', 'yield_violation', 'noncrossable_line'):
                continue
            if type(frame) is not int:
                self.infrastructure_error('event_missing_occurrence_frame', {'kind': kind})
                continue
            key = json.dumps([kind, frame, event.get('other_id'), event.get('id'), sorted(event.get('markings', []))], sort_keys=True)
            if key in self._event_keys:
                continue
            self._event_keys.add(key)
            self._pending.append(event)
        self._flush_events()

    def _flush_events(self):
        if self.first_frame is None:
            return
        remaining = []
        for event in self._pending:
            if event['frame'] <= self.first_frame or event.get('phase') == 'initialization':
                self.initialization_events.append(event)
                self._write('evaluation_events.jsonl', dict(event, category='initialization'))
                if event['kind'] == 'collision':
                    # Source frame zero is the end of initialization, before
                    # any tested driving control. Callback latency cannot turn
                    # a collided starting state into an admissible experiment.
                    self.infrastructure_error('initialization_collision', {
                        'event_frame': event['frame'], 'other_id': event.get('other_id')})
                continue
            if event['frame'] > self.frame:
                remaining.append(event)
                continue
            event['physical_tick'] = event['frame']-self.first_frame
            event['late_by_ticks'] = self.frame-event['frame']
            event['violation'] = event['kind'] != 'lane_invasion'
            if event['kind'] == 'lane_invasion':
                markings = [str(m).split('.')[-1] for m in event.get('markings', [])]
                if any(m in ('Solid', 'SolidSolid', 'Curb') for m in markings):
                    event['violation'] = True
                elif any(m in ('SolidBroken', 'BrokenSolid', 'Other', 'NONE', 'None') for m in markings) or not markings:
                    permission = event.get('crossing_permission')
                    if type(permission) is bool:
                        event['violation'] = not permission
                    else:
                        event['violation'] = None
                        self.unknown_events.append(event)
            self.events.append(event)
            self._write('evaluation_events.jsonl', dict(event, category='driving'))
        self._pending = remaining

    def _following(self, ego, actors):
        c, s = math.cos(ego['heading']), math.sin(ego['heading'])
        velocity = ego.get('velocity', [0., 0.])
        own_speed = max(0., velocity[0]*c+velocity[1]*s)
        nearest = None
        for other in actors:
            if other.get('id') == ego.get('id') or other.get('kind') not in ('vehicle', 'bicycle', 'static_obstacle'):
                continue
            if abs(other.get('z', 0.)-ego.get('z', 0.)) > 2.:
                continue
            delta = _sub(other['position'], ego['position'])
            ahead, lateral = delta[0]*c+delta[1]*s, abs(-delta[0]*s+delta[1]*c)
            relative_heading = other['heading']-ego['heading']
            extent_s = abs(math.cos(relative_heading))*other['length']/2+abs(math.sin(relative_heading))*other['width']/2
            extent_d = abs(math.sin(relative_heading))*other['length']/2+abs(math.cos(relative_heading))*other['width']/2
            if ahead <= 0 or lateral > ego['width']/2+extent_d:
                continue
            ov = other.get('velocity', [0., 0.])
            lead_speed = ov[0]*c+ov[1]*s
            if _norm(ov) > .3 and (lead_speed < 0 or lead_speed < abs(-ov[0]*s+ov[1]*c)):
                continue
            gap = ahead-ego['length']/2-extent_s
            row = {'actor_id': other['id'], 'net_gap_m': gap,
                   'time_headway_s': max(0., gap)/own_speed if own_speed > .1 else None,
                   'ttc_s': max(0., gap)/(own_speed-lead_speed) if own_speed-lead_speed > .1 else None}
            if nearest is None or gap < nearest['net_gap_m']:
                nearest = row
        if nearest:
            for key, attr in (('net_gap_m', 'min_gap'), ('time_headway_s', 'min_headway'), ('ttc_s', 'min_ttc')):
                value = nearest[key]
                if value is not None and (getattr(self, attr) is None or value < getattr(self, attr)):
                    setattr(self, attr, value)
        return nearest

    def observe(self, tick, frame, timestamp, ego, actors, road, events=(), scenario_events=None,
                control_execution=None, route_map_evidence=None):
        if type(tick) is not int or tick < 0 or type(frame) is not int or not _finite(timestamp):
            raise ValueError('evaluation identity must contain integer tick/frame and finite timestamp')
        if self.tick is None and tick != 0:
            raise ValueError('evaluation must start at physical tick zero')
        if self.tick is not None and (tick != self.tick+1 or frame != self.frame+1):
            self.coverage['complete_physical_ticks'] = False
            self.infrastructure_error('physical_frame_gap', {'previous_tick': self.tick, 'tick': tick, 'frame': frame})
            raise ValueError('evaluation requires every physical frame exactly once')
        if self.timestamp is not None and abs(timestamp-self.timestamp-self.dt) > 1e-4:
            self.coverage['complete_physical_ticks'] = False
            self.infrastructure_error('physical_clock_step_changed', {
                'previous_timestamp': self.timestamp, 'timestamp': timestamp, 'expected_step_s': self.dt})
            raise ValueError('evaluation requires the declared fixed physical clock')
        if (not all(_finite(v) for v in list(ego['position'])+list(ego.get('velocity', [0., 0.]))+
                    list(ego.get('acceleration', [0., 0.]))+[ego['heading'], ego['length'], ego['width']])
                or min(ego['length'], ego['width']) <= 0):
            raise ValueError('evaluation ego contains nonfinite state')
        acceleration_valid = ego.get('acceleration_sample_valid', True)
        if type(acceleration_valid) is not bool:
            raise ValueError('acceleration sample validity must be boolean')
        if not acceleration_valid:
            if not isinstance(ego.get('acceleration_sample_reason'), str) or not ego['acceleration_sample_reason']:
                raise ValueError('invalid acceleration sample must retain its reason')
            self._invalid_acceleration_samples += 1
        self.tick, self.frame = tick, frame
        self.timestamp = timestamp
        if self.first_frame is None:
            self.first_frame = frame
            self.first_timestamp = timestamp
        progress, lateral, route_distance = self.project(ego['position'], ego['heading'], ego=ego)
        if self.route_projection['valid']:
            self.progress_s, self.lateral = progress, lateral
        else:
            self.projection_invalid_ticks += 1
        if self.first_frame == frame:
            self.initial_state = {'frame': frame, 'ego': copy.deepcopy(ego), 'road': copy.deepcopy(road)}
        if self.start_s is None and self.route_projection['valid']:
            self.start_s = self.progress_s
            self.stall_origin_tick, self.stall_origin_s = tick, self.progress_s
        if self.previous is not None:
            self.actual_distance_m += _norm(_sub(ego['position'], self.previous['position']))
        speed = _norm(ego.get('velocity', [0., 0.]))
        acceleration = list(ego.get('acceleration', [0., 0.]))
        self.max_values['speed_mps'] = max(self.max_values['speed_mps'], speed)
        lateral_acceleration = abs(-math.sin(ego['heading'])*acceleration[0]+math.cos(ego['heading'])*acceleration[1])
        if acceleration_valid:
            self.max_values['acceleration_mps2'] = max(self.max_values['acceleration_mps2'], _norm(acceleration))
            self.max_values['lateral_acceleration_mps2'] = max(self.max_values['lateral_acceleration_mps2'], lateral_acceleration)
        if self.previous is not None and acceleration_valid:
            self._comfort_squared['acceleration_mps2'] += _norm(acceleration)**2
            self._comfort_squared['lateral_acceleration_mps2'] += lateral_acceleration**2
            self._comfort_samples['acceleration_mps2'] += 1
            self._comfort_samples['lateral_acceleration_mps2'] += 1
        # The activation frame contains set_target_velocity's initialization jump.
        # Preserve it in the raw ego record, but do not differentiate through it.
        if (self.previous is not None and acceleration_valid and
                self.previous.get('acceleration_sample_valid', True)):
            jerk = _norm(_sub(acceleration, self.previous.get('acceleration', [0., 0.])))/self.dt
            self.max_values['jerk_mps3'] = max(self.max_values['jerk_mps3'], jerk)
            self._comfort_squared['jerk_mps3'] += jerk**2
            self._comfort_samples['jerk_mps3'] += 1
        for key in ('body_on_road', 'wrong_way'):
            if type(road.get(key)) is not bool:
                self.coverage[key] = False
        limit = road.get('speed_limit_mps')
        if not _finite(limit) or limit <= 0:
            self.coverage['speed_limit'] = False
            limit = None
        for key, value in road.get('coverage', {}).items():
            if key in self.coverage:
                # Unknown once remains unknown; supplied known channels start false until their first frame.
                if tick == 0:
                    self.coverage[key] = bool(value)
                else:
                    self.coverage[key] = self.coverage[key] and bool(value)
        dt = self.dt if tick else 0.
        flags = {'offroad': road.get('body_on_road') is False,
                 'wrong_way': road.get('wrong_way') is True,
                 'speeding': limit is not None and speed > limit+self.settings['speed_tolerance_mps']}
        for key, active in flags.items():
            self.seconds[key] += dt*int(active)
            grace = self.settings[{'offroad': 'offroad_grace_s', 'wrong_way': 'wrong_way_grace_s', 'speeding': 'overspeed_grace_s'}[key]]
            if self.seconds[key] > grace+1e-9:
                self.violations[key] = True
        self.seconds['stopped'] += dt*int(speed < self.settings['stop_speed_mps'])
        expected_wait = road.get('expected_wait', False)
        self.seconds['expected_wait'] += dt*int(bool(expected_wait))
        if (self.route_projection['valid'] and self.stall_origin_s is not None
                and abs(self.progress_s-self.stall_origin_s) >= self.settings['stall_progress_m']) or expected_wait:
            self.stall_origin_s, self.stall_origin_tick = self.progress_s, tick
        self.stalled = (self.route_projection['valid'] and self.stall_origin_tick is not None
            and (tick-self.stall_origin_tick)/self.hz >= self.settings['stall_seconds'])
        endpoint_distance = _norm(_sub(ego['position'], self.route[-1]))
        goal_lane = self.settings.get('arrival_lane_id')
        goal_lane_match = goal_lane is None or ego.get('lane_id') == goal_lane
        goal_heading_error = abs((ego['heading']-self.goal_heading+math.pi)%(2.*math.pi)-math.pi)
        self.arrival = (self.route_projection['valid'] and goal_lane_match and goal_heading_error <= math.pi/3
                        and self.progress_s >= self.arc[-1]-self.settings['arrival_margin_m']
                        and endpoint_distance <= self.settings['arrival_margin_m']
                        and abs(self.lateral) <= self.settings['arrival_lateral_m'])
        self.arrival_evidence = {'goal_lane_id': goal_lane, 'observed_lane_id': ego.get('lane_id'),
            'goal_lane_match': goal_lane_match, 'endpoint_distance_m': endpoint_distance,
            'distance_margin_m': self.settings['arrival_margin_m'],
            'lateral_margin_m': self.settings['arrival_lateral_m'],
            'heading_error_rad': goal_heading_error, 'heading_margin_rad': math.pi/3,
            'route_projection_valid': self.route_projection['valid']}
        if scenario_events is not None:
            self.scenario = copy.deepcopy(scenario_events if isinstance(scenario_events, dict)
                                          else {'scenario_valid': True, 'events': list(scenario_events)})
            scenario_rows = self.scenario.get('events', [])
            if isinstance(scenario_rows, dict):
                scenario_rows = [dict(v, id=k) for k, v in scenario_rows.items()]
            for row in scenario_rows:
                identity = str(row.get('id', row.get('role', 'unknown')))
                if self._scenario_event_rows.get(identity) != row:
                    self._write('scenario_events.jsonl', {
                        'version': VERSION, 'instance_hash': self.scenario.get('instance_hash'),
                        'physical_tick': tick, 'frame': frame, 'timestamp': timestamp,
                        'scenario_event': row})
                    self._scenario_event_rows[identity] = copy.deepcopy(row)
        self.last_road = copy.deepcopy(road)
        self.ingest_events(events)
        following = self._following(ego, actors)
        self.previous = copy.deepcopy(ego)
        map_evidence = route_map_evidence(tick) if callable(route_map_evidence) else route_map_evidence
        self._write('physical_evaluation.jsonl', {'version': VERSION, 'physical_tick': tick, 'frame': frame,
            'instance_hash': self.scenario.get('instance_hash'),
            'coordinate_system': 'carla_world_x_negative_y_m_yaw_left_positive',
            'timestamp': timestamp, 'sim_time_s': tick/self.hz, 'ego': ego, 'actors': actors,
            'route_s_m': self.progress_s if self.route_projection['valid'] else None,
            'route_lateral_m': self.lateral if self.route_projection['valid'] else None,
            'route_projection': copy.deepcopy(self.route_projection), 'last_valid_route_s_m': self.progress_s,
            'route_map_evidence': copy.deepcopy(map_evidence),
            'speed_mps': speed, 'road': road, 'following': following,
            'control_execution': copy.deepcopy(control_execution), 'arrival_evidence': copy.deepcopy(self.arrival_evidence),
            'arrival': self.arrival, 'stalled': self.stalled, 'observed_actor_count': len(actors),
            'actor_scope': 'physical_observation_not_mpc_selection'})
        return self.report()

    def report(self, reason=None):
        self._flush_events()
        collision = any(e['kind'] == 'collision' for e in self.events)
        event_violations = sorted(set(e['kind'] for e in self.events if e.get('violation')))
        violations = [k for k, v in self.violations.items() if v]+event_violations
        rows = self.scenario.get('events', [])
        if isinstance(rows, dict):
            rows = [dict(v, id=k) for k, v in rows.items()]
        event_states = {str(r.get('id', r.get('role', r.get('event', 'unknown')))): r.get('state', r.get('status')) for r in rows}
        required = list(self.settings['required_event_ids'])
        required.extend(str(r.get('id', r.get('role'))) for r in rows
                        if r.get('required') and str(r.get('id', r.get('role'))) not in required)
        missing = [key for key in required if event_states.get(key) not in ('completed', 'complete', 'satisfied', 'safely_avoided')]
        conditions_complete = not missing or not self.settings['strict_event_completion']
        task = self.scenario.get('task_criteria')
        route_checkpoints_complete = (task is None or task.get('route_checkpoints_completed') == task.get('route_checkpoints_total'))
        if task is not None:
            conditions_complete = conditions_complete and task.get('completed') is True
        scenario_valid = self.scenario.get('scenario_valid')
        coverage_complete = all(self.coverage.values()) and not self.unknown_events and not self._pending
        infrastructure_valid = scenario_valid is True and not self.infrastructure_errors and self.coverage['complete_physical_ticks']
        route_measurement_valid = self.projection_invalid_ticks == 0 and self.route_projection['valid']
        valid = infrastructure_valid and coverage_complete and route_measurement_valid
        strict_success = bool(valid and self.arrival and conditions_complete and not violations)
        success_status = ('success' if strict_success else 'failed' if violations or self.stalled or reason in ('timeout', 'duration')
                          else 'projection_invalid' if not self.route_projection['valid'] else
                          'undetermined' if not valid else 'task_incomplete' if self.arrival and not conditions_complete else 'in_progress')
        driving_reason = ('collision' if collision else 'arrival' if self.arrival and conditions_complete else
                          'goal_reached_task_incomplete' if self.arrival else 'route_projection_invalid' if not self.route_projection['valid']
                          else 'stall' if self.stalled else reason or 'running')
        route_remaining = max(1e-9, self.arc[-1]-(self.start_s or 0.))
        contacts, last_contact = 0, {}
        for event in sorted((e for e in self.events if e['kind'] == 'collision'), key=lambda e: e['frame']):
            other = str(event.get('other_id', 'unknown'))
            if other not in last_contact or event['frame']-last_contact[other] > self.hz//2:
                contacts += 1
            last_contact[other] = event['frame']
        return {'version': VERSION, 'physical_hz': self.hz, 'physical_ticks': self.tick,
            'instance_hash': self.scenario.get('instance_hash'),
            'frame': self.frame, 'first_frame': self.first_frame,
            'timestamp': self.timestamp, 'first_timestamp': self.first_timestamp,
            'observations': self.tick+1 if self.tick is not None else 0,
            'simulated_seconds': (self.tick or 0)/self.hz,
            'scenario_valid': scenario_valid, 'evaluation_valid': valid,
            'infrastructure_valid': bool(infrastructure_valid), 'measurement_complete': coverage_complete,
            'route_measurement_valid': route_measurement_valid, 'route_projection': copy.deepcopy(self.route_projection),
            'projection_invalid_ticks': self.projection_invalid_ticks, 'last_valid_route_s_m': self.progress_s,
            'strict_success_status': success_status,
            'strict_success': strict_success, 'arrival': self.arrival,
            'route_completed': self.arrival and route_checkpoints_complete, 'collision': collision,
            'stalled': self.stalled, 'driving_outcome': driving_reason, 'runtime_termination_reason': reason,
            'terminal_reason': driving_reason,
            'progress_m': max(0., (self.progress_s or 0.)-(self.start_s or 0.)) if self.route_projection['valid'] else None,
            'route_completion': (min(1., max(0., ((self.progress_s or 0.)-(self.start_s or 0.))/route_remaining))
                                 if self.route_projection['valid'] else None),
            'arrival_evidence': copy.deepcopy(getattr(self, 'arrival_evidence', None)),
            'distance_travelled_m': self.actual_distance_m, 'violations': sorted(set(violations)),
            'collision_contact_episodes': contacts,
            'collision_contact_episodes_per_km': contacts*1000/self.actual_distance_m if self.actual_distance_m > 0 else None,
            'contact_episode_rule': 'same_other_actor_callbacks_separated_by_more_than_0.5s_are_new_contacts',
            'violation_seconds': dict(self.seconds), 'maximum': dict(self.max_values),
            'rms': {key: math.sqrt(total/self._comfort_samples[key]) if self._comfort_samples[key] else None
                    for key, total in self._comfort_squared.items()},
            'comfort_samples': dict(self._comfort_samples),
            'invalid_acceleration_samples': self._invalid_acceleration_samples,
            'comfort_sampling_policy': 'valid_control_interval_acceleration;jerk_requires_two_valid_samples;raw_measurements_retained',
            'minimum_net_gap_m': self.min_gap, 'minimum_time_headway_s': self.min_headway,
            'minimum_same_direction_ttc_s': self.min_ttc,
            'following_scope': 'current_heading_overlapping_body_corridor; crossing TTC not claimed',
            'event_states': event_states, 'required_events_incomplete': missing,
            'task_criteria': copy.deepcopy(task),
            'scenario_events': copy.deepcopy(rows),
            'events': copy.deepcopy(self.events), 'initialization_events': copy.deepcopy(self.initialization_events),
            'unknown_rule_events': copy.deepcopy(self.unknown_events), 'coverage': dict(self.coverage),
            'pending_event_count': len(self._pending), 'infrastructure_errors': copy.deepcopy(self.infrastructure_errors),
            'scenario_errors': copy.deepcopy(self.scenario.get('errors', [])),
            'geometry_method': self.last_road.get('geometry_method'),
            'limitations': ['50Hz discrete measurement is not a continuous collision proof',
                           'yield metric checks simultaneous conflict-zone occupancy, not all forced-braking cases',
                           'mixed solid/broken markings without crossing-side evidence remain unknown']}

    def save_report(self, reason=None):
        result = self.report(reason)
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory/'evaluation.json'
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(target)
        return result


class CarlaEvaluationProbe:
    """SDK adapter, read only; world must be held by the bridge's single tick owner."""
    def __init__(self, world, carla, ego, route_waypoints, settings=None, directory=None, physics_hz=50):
        self.world, self.carla, self.ego = world, carla, ego
        self.map = world.get_map()
        self.settings = dict(settings or {})
        route = [[p.transform.location.x, -p.transform.location.y] for p in route_waypoints]
        evaluation = dict(self.settings.get('evaluation', self.settings))
        # Arrival belongs to the original fixed destination lane, even when the
        # planner temporarily selects another legal lane for an overtake.
        evaluation['arrival_lane_id'] = _lane_id(route_waypoints[-1])
        self.evaluator = EpisodeEvaluator(route, evaluation, physics_hz, directory)
        try:
            from .carla_task_criteria import TaskCriteria
        except ImportError:
            from carla_task_criteria import TaskCriteria
        self.task_criteria = TaskCriteria(self.settings, route, physics_hz)
        self._route_boundary_evidence = _boundary_topology_evidence(route_waypoints, self.task_criteria.checkpoints)
        self.task_criteria.set_boundary_topology(_boundary_links(self._route_boundary_evidence))
        self._route_waypoints = list(route_waypoints)
        self._projection_corridors = {}
        self._pending_projection_corridors = {}
        self._route_boundary_published = False
        self.evaluator.projection_metadata = self._projection_corridor
        self.task_criteria.projection_metadata = self._projection_corridor
        self._previous_ego = None
        self._previous_waypoint, self._crossing_evidence = None, {}
        self._stops, self._lights, self._stop_dwell, self._stop_passed = [], [], {}, set()
        self._stop_last_stopped, self._omega_history = {}, {}
        self._stop_regions, self._region_inside = [], {}
        self.public_rules = self.settings.get('public_rules', [])
        self._priority_violated = set()
        self._priority_seen = {}
        self._conflict_zones = [dict(id=rule['id'], route_s_m=rule['conflict_route_s_m'],
                                    half_width_m=rule.get('conflict_half_width_m', 2.))
                               for rule in self.public_rules if len(rule.get('conflict_route_s_m', [])) == 2]
        self._conflict_zones.extend(copy.deepcopy(evaluation.get('conflict_zones', [])))
        self._conflict_polygons = {zone['id']: self._zone_polygons(zone) for zone in self._conflict_zones}
        self._zone_presence, self._zone_exits, self._pet = {}, {}, []
        self._pet_keys = set()
        self._coverage = {'collision_events': True, 'lane_markings': True,
                          'traffic_lights': True, 'stop_signs': True, 'public_priority': True}
        for actor in world.get_actors():
            if actor.type_id.startswith('traffic.traffic_light'):
                if not hasattr(actor, 'get_stop_waypoints'):
                    self._coverage['traffic_lights'] = False
                    continue
                self._lights.extend((actor, wp) for wp in actor.get_stop_waypoints())
            elif actor.type_id.startswith('traffic.stop'):
                if hasattr(actor, 'get_stop_waypoints'):
                    self._stops.extend((str(actor.id), wp) for wp in actor.get_stop_waypoints())
                elif hasattr(actor, 'trigger_volume'):
                    box, transform = actor.trigger_volume, actor.get_transform()
                    local = box.location
                    center = transform.transform(carla.Location(x=local.x, y=local.y, z=local.z))
                    if min(box.extent.x, box.extent.y, box.extent.z) <= 0:
                        self._coverage['stop_signs'] = False
                    else:
                        # TrafficSign has no precise stop-waypoint API in 0.9.15.
                        # Evaluate its actual native mandatory-stop region; never
                        # pretend its projected center is a painted stop line.
                        self._stop_regions.append({'id': str(actor.id), 'center': [center.x, -center.y],
                            'z': center.z, 'half': [box.extent.x, box.extent.y, box.extent.z],
                            'heading': -math.radians(transform.rotation.yaw+box.rotation.yaw)})
                else:
                    self._coverage['stop_signs'] = False
        for rule in self.public_rules:
            if rule.get('kind') != 'yield' or not rule.get('priority_lane_ids') or len(rule.get('conflict_route_s_m', [])) != 2:
                self._coverage['public_priority'] = False

    def _projection_corridor(self, index, ego, route_s):
        """Allow lateral movement only in the original road's real cross section.

        Native directed neighbours and their actual widths establish the bounds;
        numeric lane-ID differences never establish geometry or connectivity.
        The local arc window remains authoritative, including on a loop.
        """
        # A segment can straddle a native road/section boundary. Use its
        # connected endpoint matching the actual section, never a global scan.
        section = tuple(str(ego.get('lane_id', '')).split('|')[:2])
        if (tuple(_lane_id(self._route_waypoints[index]).split('|')[:2]) != section
                and index+1 < len(self._route_waypoints)
                and tuple(_lane_id(self._route_waypoints[index+1]).split('|')[:2]) == section):
            index += 1
        if index not in self._projection_corridors:
            waypoint = self._route_waypoints[index]
            origin = waypoint.transform.location
            yaw = -math.radians(waypoint.transform.rotation.yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            lanes = {_lane_id(waypoint): {'offset_m': 0., 'width_m': float(waypoint.lane_width)}}
            native_links = []
            for direction in ('left', 'right'):
                current = waypoint
                for _ in range(16):
                    other = getattr(current, 'get_'+direction+'_lane')()
                    if other is None or _lane_id(other) in lanes:
                        break
                    angle = math.radians(other.transform.rotation.yaw-waypoint.transform.rotation.yaw)
                    location = other.transform.location
                    if (other.road_id != waypoint.road_id or other.section_id != waypoint.section_id
                            or str(other.lane_type).split('.')[-1] != 'Driving'
                            or math.cos(angle) <= .9 or abs(location.z-origin.z) >= 1.5):
                        break
                    dx, dy = location.x-origin.x, -(location.y-origin.y)
                    # Same-s native neighbours must really form a transverse
                    # section; reject a spatially disconnected map connection.
                    if abs(c*dx+s*dy) > 2. or not 0. < float(other.lane_width) < 10.:
                        break
                    offset = -s*dx+c*dy
                    previous = lanes[_lane_id(current)]
                    if abs(offset-previous['offset_m']) > (float(other.lane_width)+previous['width_m'])/2.+.5:
                        break
                    lanes[_lane_id(other)] = {'offset_m': offset, 'width_m': float(other.lane_width)}
                    native_links.append({'from_lane_id': _lane_id(current), 'side': direction,
                                         'to_waypoint': _native_waypoint_evidence(other)})
                    current = other
            lower = min(row['offset_m']-row['width_m']/2. for row in lanes.values())
            upper = max(row['offset_m']+row['width_m']/2. for row in lanes.values())
            self._projection_corridors[index] = {'route_index': index, 'route_lane_id': _lane_id(waypoint), 'lanes': lanes,
                'lateral_interval_m': [lower, upper], 'heading_rad': yaw,
                'maximum_distance_m': max(abs(lower), abs(upper))+.25,
                'source': 'native_same_section_same_direction_driving_neighbour_cross_section'}
            if not hasattr(self, '_pending_projection_corridors'):
                self._pending_projection_corridors = {}
            self._pending_projection_corridors[str(index)] = {
                'route_waypoint': _native_waypoint_evidence(waypoint), 'neighbour_links': native_links}
        corridor = copy.deepcopy(self._projection_corridors[index])
        projected = self.evaluator.point_at(route_s)
        dx, dy = ego['position'][0]-projected[0], ego['position'][1]-projected[1]
        c, s = math.cos(corridor['heading_rad']), math.sin(corridor['heading_rad'])
        lateral, longitudinal = -s*dx+c*dy, c*dx+s*dy
        lane = corridor['lanes'].get(ego.get('lane_id'))
        aligned = math.cos(ego['heading']-corridor['heading_rad']) >= .5
        corridor.update(observed_lane_id=ego.get('lane_id'), lateral_m=lateral,
            longitudinal_residual_m=longitudinal,
            contains_observed_center=bool(lane is not None and aligned and abs(longitudinal) <= 1.
                and abs(lateral-lane['offset_m']) <= lane['width_m']/2.+.25))
        return corridor

    def _pop_route_map_evidence(self, tick):
        """Static native evidence is emitted once, in existing physical records."""
        result = {'version': 'carla-route-map-evidence-1.0',
                  'source': 'native_public_waypoint_topology', 'physical_tick': tick}
        if not self._route_boundary_published:
            result['boundary_gates'] = copy.deepcopy(self._route_boundary_evidence)
            result['route_widths_m'] = [float(waypoint.lane_width) for waypoint in self._route_waypoints]
            self._route_boundary_published = True
        if self._pending_projection_corridors:
            result['corridors'] = copy.deepcopy(self._pending_projection_corridors)
            self._pending_projection_corridors.clear()
        return result if 'boundary_gates' in result or 'corridors' in result else None

    def _state(self, actor, snapshot):
        state = snapshot.find(actor.id)
        if state is None:
            return None
        transform, box = state.get_transform(), actor.bounding_box
        # CARLA Transform.transform mutates Location in place; never pass SDK box metadata.
        local = box.location
        center = transform.transform(self.carla.Location(x=local.x, y=local.y, z=local.z))
        velocity, acceleration = state.get_velocity(), state.get_acceleration()
        omega = math.radians(state.get_angular_velocity().z)
        timestamp = snapshot.timestamp.elapsed_seconds
        past = self._omega_history.get(actor.id)
        alpha = (omega-past[1])/(timestamp-past[0]) if past is not None and timestamp > past[0] else 0.
        self._omega_history[actor.id] = (timestamp, omega)
        offset_x, offset_y = center.x-transform.location.x, center.y-transform.location.y
        heading = -math.radians(transform.rotation.yaw+box.rotation.yaw)
        waypoint = self.map.get_waypoint(center, project_to_road=False, lane_type=self.carla.LaneType.Driving)
        kind = 'pedestrian' if actor.type_id.startswith('walker.') else 'static_obstacle' if actor.type_id.startswith('static.') else 'vehicle'
        return {'id': str(actor.id), 'kind': kind, 'position': [center.x, -center.y], 'z': center.z,
                'actor_origin_carla_xyz': [transform.location.x, transform.location.y, transform.location.z],
                'heading': heading, 'length': 2*abs(box.extent.x), 'width': 2*abs(box.extent.y),
                'velocity': [velocity.x-omega*offset_y, -(velocity.y+omega*offset_x)],
                'acceleration': [acceleration.x-alpha*offset_y-omega*omega*offset_x,
                                 -(acceleration.y+alpha*offset_x-omega*omega*offset_y)],
                'lane_id': _lane_id(waypoint),
                'is_junction': bool(waypoint.is_junction) if waypoint is not None else None,
                'state_source': 'snapshot_origin_to_bbox_center_with_angular_velocity_and_causal_angular_acceleration'}

    def _road(self, ego):
        points = _sample_rect(ego, self.evaluator.settings['body_sample_spacing_m'])
        accepted = []
        for p in points:
            location = self.carla.Location(x=p[0], y=-p[1], z=ego['z'])
            wp = self.map.get_waypoint(location, project_to_road=False, lane_type=self.carla.LaneType.Driving)
            accepted.append(wp is not None and abs(wp.transform.location.z-ego['z']) < 2.)
        center = self.map.get_waypoint(self.carla.Location(x=ego['position'][0], y=-ego['position'][1], z=ego['z']),
                                       project_to_road=False, lane_type=self.carla.LaneType.Driving)
        wrong_way = (center is not None and not center.is_junction
                     and math.cos(ego['heading']+math.radians(center.transform.rotation.yaw)) < 0.)
        return {'body_on_road': all(accepted), 'wrong_way': bool(wrong_way),
                'speed_limit_mps': float(self.ego.get_speed_limit())/3.6, 'coverage': dict(self._coverage),
                'geometry_method': {'name': 'CARLA_Driving_waypoint_body_perimeter_and_center',
                    'spacing_m': self.evaluator.settings['body_sample_spacing_m'], 'samples': len(points),
                    'outside_samples': len(points)-sum(accepted), 'exact_polygon_coverage': False}, 'expected_wait': False}

    def _record_crossing_evidence(self, ego, frame):
        current = self.map.get_waypoint(self.carla.Location(x=ego['position'][0], y=-ego['position'][1], z=ego['z']),
                                       project_to_road=False, lane_type=self.carla.LaneType.Driving)
        wp, previous = self._previous_waypoint, self._previous_ego
        records = []
        if wp is not None and previous is not None and not wp.is_junction:
            yaw = -math.radians(wp.transform.rotation.yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            origin = [wp.transform.location.x, -wp.transform.location.y]
            def lateral(point):
                delta = _sub(point, origin)
                return -s*delta[0]+c*delta[1]
            old_values = [lateral(p) for p in _box_corners(previous)]
            new_values = [lateral(p) for p in _box_corners(ego)]
            motion = lateral(ego['position'])-lateral(previous['position'])
            for side, extent, old_extent, direction in (
                    ('left', max(new_values), max(old_values), 1.),
                    ('right', -min(new_values), -min(old_values), -1.)):
                crossed = extent >= wp.lane_width/2-.03 and (old_extent < wp.lane_width/2 or motion*direction > 1e-5)
                if not crossed:
                    continue
                marking = getattr(wp, side+'_lane_marking')
                flag = int(getattr(self.carla.LaneChange, side.title()))
                records.append({'side': side, 'source_lane_id': _lane_id(wp),
                    'line_type': str(marking.type).split('.')[-1],
                    'waypoint_permission': int(wp.lane_change), 'marking_permission': int(marking.lane_change),
                    'permitted': bool(int(wp.lane_change)&flag) and bool(int(marking.lane_change)&flag),
                    'source': 'previous_physical_waypoint_and_actual_body_lateral_crossing'})
        self._crossing_evidence[frame] = records
        self._previous_waypoint = current

    def _annotate_events(self, events):
        result = []
        for source in events or ():
            event = copy.deepcopy(source)
            if event.get('kind') == 'lane_invasion':
                markings = [str(m).split('.')[-1] for m in event.get('markings', [])]
                candidates = [r for r in self._crossing_evidence.get(event.get('frame'), []) if r['line_type'] in markings]
                if len(candidates) == 1:
                    event['crossing_permission'] = candidates[0]['permitted']
                    event['crossing_evidence'] = candidates[0]
            result.append(event)
        return result

    @staticmethod
    def _front(ego):
        return [ego['position'][0]+math.cos(ego['heading'])*ego['length']/2,
                ego['position'][1]+math.sin(ego['heading'])*ego['length']/2]

    def _line(self, ego, waypoint):
        yaw = -math.radians(waypoint.transform.rotation.yaw)
        forward = [math.cos(yaw), math.sin(yaw)]
        delta = _sub(self._front(ego), [waypoint.transform.location.x, -waypoint.transform.location.y])
        longitudinal, lateral = _dot(delta, forward), abs(-forward[1]*delta[0]+forward[0]*delta[1])
        relevant = (math.cos(ego['heading']-yaw) > .3 and lateral < waypoint.lane_width/2+ego['width']/2
                    and abs(ego['z']-waypoint.transform.location.z) < 2.)
        return longitudinal, relevant

    def _rules(self, ego, actors, frame, road):
        events = []
        light_observations = []
        for light, waypoint in self._lights:
            distance, relevant = self._line(ego, waypoint)
            state = str(light.get_state()).split('.')[-1].lower()
            line_evidence = {'id': str(light.id), 'state': state, 'controlled_lane_id': _lane_id(waypoint),
                'observed_lane_id': ego.get('lane_id'), 'front_signed_distance_m': distance,
                'stop_position': [waypoint.transform.location.x, -waypoint.transform.location.y],
                'stop_heading_rad': -math.radians(waypoint.transform.rotation.yaw),
                'controlled_lane_width_m': waypoint.lane_width, 'relevant': relevant}
            if relevant and abs(distance) <= 35.:
                light_observations.append(line_evidence)
            if relevant and -35. < distance <= 0 and state in ('red', 'yellow'):
                road['expected_wait'] = True
            if self._previous_ego is not None:
                old, old_relevant = self._line(self._previous_ego, waypoint)
                if relevant and old_relevant and old <= 0 < distance and state == 'red':
                    events.append({'kind': 'red_light', 'frame': frame, 'id': str(light.id),
                        'previous_front_signed_distance_m': old, 'crossing_evidence': line_evidence,
                        'source': 'actual_front_crossing_native_stop_waypoint_independent_of_navigation'})
        road['traffic_light_observations'] = light_observations
        for identity, waypoint in self._stops:
            distance, relevant = self._line(ego, waypoint)
            stopped = relevant and -5. <= distance <= .1 and _norm(ego['velocity']) < self.evaluator.settings['stop_speed_mps']
            if stopped:
                self._stop_dwell[identity] = self._stop_dwell.get(identity, 0)+int(self._stop_last_stopped.get(identity, False))
                road['expected_wait'] = True
            elif self._stop_dwell.get(identity, 0)/self.evaluator.hz < self.evaluator.settings['stop_hold_s']:
                self._stop_dwell[identity] = 0
            elif identity not in self._stop_passed and (not relevant or distance < -5.):
                self._stop_dwell[identity] = 0
            if self._previous_ego is not None:
                old, old_relevant = self._line(self._previous_ego, waypoint)
                if relevant and old_relevant and old <= 0 < distance:
                    if self._stop_dwell.get(identity, 0)/self.evaluator.hz < self.evaluator.settings['stop_hold_s']:
                        events.append({'kind': 'stop_sign', 'frame': frame, 'id': identity})
                    self._stop_passed.add(identity)
                elif relevant and distance < -10.:
                    self._stop_passed.discard(identity)
            self._stop_last_stopped[identity] = stopped
        for region in self._stop_regions:
            identity = region['id']
            delta = _sub(self._front(ego), region['center'])
            c, s = math.cos(region['heading']), math.sin(region['heading'])
            inside = (abs(c*delta[0]+s*delta[1]) <= region['half'][0]
                      and abs(-s*delta[0]+c*delta[1]) <= region['half'][1]
                      and abs(ego['z']-region['z']) <= region['half'][2]+1.)
            key = 'region_'+identity
            stopped = inside and _norm(ego['velocity']) < self.evaluator.settings['stop_speed_mps']
            if stopped:
                self._stop_dwell[key] = self._stop_dwell.get(key, 0)+int(self._stop_last_stopped.get(key, False))
                road['expected_wait'] = True
            elif self._stop_dwell.get(key, 0)/self.evaluator.hz < self.evaluator.settings['stop_hold_s']:
                self._stop_dwell[key] = 0
            if not inside and self._region_inside.get(identity, False):
                if self._stop_dwell.get(key, 0)/self.evaluator.hz < self.evaluator.settings['stop_hold_s']:
                    events.append({'kind': 'stop_sign', 'frame': frame, 'id': identity,
                                   'criterion': 'stop_before_exiting_native_trigger_volume'})
                self._stop_dwell[key] = 0
            self._region_inside[identity] = inside
            self._stop_last_stopped[key] = stopped
        for rule in self.public_rules:
            if rule.get('kind') != 'yield' or len(rule.get('conflict_route_s_m', [])) != 2:
                continue
            polygons = self._conflict_polygons.get(rule['id'], [])
            observed = self._priority_seen.setdefault(rule['id'], {})
            for actor in actors:
                if (actor.get('kind') in ('vehicle', 'bicycle', 'pedestrian')
                        and actor.get('lane_id') in rule.get('priority_lane_ids', [])):
                    observed[actor['id']] = frame
            # Preserve already observed priority approach identity through a
            # junction's lane-id transition. No actor's scripted future is read.
            eligible = {identity for identity, seen_frame in observed.items() if frame-seen_frame <= 10*self.evaluator.hz}
            if not any(_convex_intersects(_box_corners(ego), poly) for poly in polygons) or rule['id'] in self._priority_violated:
                continue
            for actor in actors:
                if actor['id'] not in eligible:
                    continue
                # A priority entrant that has already merged into ego's same
                # lane becomes a following relationship, not a crossing breach.
                delta = _sub(actor['position'], ego['position'])
                forward = [math.cos(ego['heading']), math.sin(ego['heading'])]
                if (ego.get('lane_id') and ego.get('lane_id') == actor.get('lane_id')
                        and _dot(delta, forward) > 0
                        and abs(-forward[1]*delta[0]+forward[0]*delta[1]) < (ego['width']+actor['width'])/2
                        and math.cos(actor['heading']-ego['heading']) > .85):
                    continue
                if any(_convex_intersects(_box_corners(actor), poly) for poly in polygons) and abs(actor['z']-ego['z']) < 2.:
                    events.append({'kind': 'yield_violation', 'frame': frame, 'id': rule['id'], 'other_id': actor['id'],
                                   'criterion': 'simultaneous_public_conflict_zone_occupancy'})
                    self._priority_violated.add(rule['id'])
                    break
        return events

    def _zone_polygons(self, zone):
        low, high = zone['route_s_m']
        if not (_finite(low) and _finite(high) and 0 <= low < high <= self.evaluator.arc[-1]):
            raise ValueError('conflict zone interval must lie on locked route')
        width = zone.get('half_width_m', 2.)
        if not _finite(width) or width <= 0:
            raise ValueError('conflict zone half width must be finite positive')
        points = [self.evaluator.point_at(low)]
        points.extend(p for p, distance in zip(self.evaluator.route, self.evaluator.arc) if low < distance < high)
        points.append(self.evaluator.point_at(high))
        polygons = []
        for a, b in zip(points, points[1:]):
            delta = _sub(b, a)
            length = _norm(delta)
            if length <= 1e-9:
                continue
            nx, ny = -delta[1]*width/length, delta[0]*width/length
            polygons.append([[a[0]+nx, a[1]+ny], [b[0]+nx, b[1]+ny],
                             [b[0]-nx, b[1]-ny], [a[0]-nx, a[1]-ny]])
        return polygons

    def _measure_conflict_zones(self, ego, actors, tick, frame):
        """Discrete PET: elapsed time between one actor leaving and another entering.

        A zero denotes simultaneous geometric occupancy, not necessarily contact.
        Only public, explicitly configured conflict regions are evaluated.
        """
        ego_id = str(ego['id'])
        for zone in self._conflict_zones:
            identity = zone['id']
            low, high = zone['route_s_m']
            present = set()
            for actor in [ego]+actors:
                if abs(actor.get('z', 0.)-ego.get('z', 0.)) > 2.:
                    continue
                if any(_convex_intersects(_box_corners(actor), poly) for poly in self._conflict_polygons[identity]):
                    present.add(str(actor['id']))
            before = self._zone_presence.get(identity, set())
            exits = self._zone_exits.setdefault(identity, {})
            for departed in before-present:
                exits[departed] = tick
            for entered in present-before:
                other_ids = (set(exits)|present)-{entered}
                for other in other_ids:
                    if ego_id not in (entered, other):
                        continue
                    separation = 0. if other in present else (tick-exits[other])/self.evaluator.hz
                    if separation > zone.get('pet_window_s', 10.):
                        continue
                    key = (identity, tick, tuple(sorted((entered, other))))
                    if key in self._pet_keys:
                        continue
                    self._pet_keys.add(key)
                    row = {'zone_id': identity, 'physical_tick': tick, 'frame': frame,
                           'entered_actor': entered, 'other_actor': other, 'pet_s': separation,
                           'overlapping_occupancy': other in present}
                    self._pet.append(row)
                    self.evaluator._write('conflict_zone_events.jsonl', row)
            self._zone_presence[identity] = present

    def observe(self, snapshot, tick, events=(), scenario_events=None, control_execution=None,
                acceleration_sample_valid=True, acceleration_sample_reason=None):
        ego = self._state(self.ego, snapshot)
        if ego is None:
            self.evaluator.infrastructure_error('ego_missing_from_world_snapshot')
            raise RuntimeError('physical evaluator cannot find ego in snapshot')
        ego['acceleration_sample_valid'] = acceleration_sample_valid
        ego['acceleration_sample_reason'] = acceleration_sample_reason
        actors = []
        for actor in self.world.get_actors():
            if actor.id != self.ego.id and actor.type_id.startswith(('vehicle.', 'walker.pedestrian.', 'static.prop.')):
                state = self._state(actor, snapshot)
                if state is not None:
                    actors.append(state)
        road = self._road(ego)
        road['stop_sign_geometry'] = 'native_stop_waypoint_if_available_else_native_trigger_volume'
        self._record_crossing_evidence(ego, snapshot.frame)
        rule_events = self._rules(ego, actors, snapshot.frame, road)
        if self.world.get_snapshot().frame != snapshot.frame:
            self.evaluator.infrastructure_error('world_advanced_during_evaluation')
            raise RuntimeError('world changed while reading physical evaluation state')
        scenario = copy.deepcopy(scenario_events if isinstance(scenario_events, dict)
                                 else {'scenario_valid': True, 'events': list(scenario_events or [])})
        scenario['task_criteria'] = self.task_criteria.observe(tick, ego, actors, scenario.get('actor_roles', {}))
        result = self.evaluator.observe(tick, snapshot.frame, snapshot.timestamp.elapsed_seconds,
            ego, actors, road, self._annotate_events(events)+rule_events, scenario,
            control_execution=control_execution, route_map_evidence=self._pop_route_map_evidence)
        self._measure_conflict_zones(ego, actors, tick, snapshot.frame)
        self._previous_ego = ego
        return result

    def ingest_events(self, events):
        self.evaluator.ingest_events(self._annotate_events(events))

    def report(self, reason=None, save=True):
        result = self.evaluator.report(reason)
        result['task_criteria'] = self.task_criteria.report()
        result['conflict_zone_pet'] = {'applicable': bool(self._conflict_zones), 'regions': len(self._conflict_zones),
            'minimum_s': min((row['pet_s'] for row in self._pet), default=None),
            'encounters': len(self._pet), 'resolution_s': self.evaluator.dt,
            'method': '50Hz_actual_yaw_box_intersection_with_public_route_corridor_quads',
            'records': copy.deepcopy(self._pet)}
        if save and self.evaluator.directory is not None:
            target = self.evaluator.directory/'evaluation.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(target)
        return result

    def infrastructure_error(self, kind, details=None):
        self.evaluator.infrastructure_error(kind, details)
