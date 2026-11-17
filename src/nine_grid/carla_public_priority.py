"""Causal public right-of-way screening for the CARLA planner.

This is a current-observation rule adapter, not the independent evaluator.
Public map continuations are conservatively screened without NPC intent/routes.
All coordinates use the bridge's world x, negative-CARLA-y convention.
"""
import math
from collections import deque

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring


VERSION = 'carla-public-priority-screen-1.0'


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label+' must be finite')
    return float(value)


def _coordinates(values):
    return [(_number(float(p[0]), 'x'), _number(float(p[1]), 'y')) for p in values]


def _shape(actor):
    x, y = _coordinates([actor['position']])[0]
    yaw = _number(actor['heading'], 'heading')
    length, width = _number(actor['length'], 'length'), _number(actor['width'], 'width')
    if min(length, width) <= 0:
        raise ValueError('actor footprint must have positive dimensions')
    c, s = math.cos(yaw), math.sin(yaw)
    return Polygon([(x+c*a-s*b, y+s*a+c*b) for a, b in
        ((-length/2, -width/2), (length/2, -width/2), (length/2, width/2), (-length/2, width/2))])


def _first_path_hit(line, start_s, distance, region, body_half_width):
    if distance <= 0 or start_s >= line.length:
        return None
    end = min(line.length, start_s+distance)
    path = substring(line, max(0., start_s), end)
    if path.geom_type != 'LineString':
        return None
    padded = region.buffer(body_half_width)
    traveled = 0.
    points = list(path.coords)
    for a, b in zip(points, points[1:]):
        segment = LineString([a, b])
        if segment.intersects(padded):
            # Lower bound on first contact distance, conservative by at most
            # one public map sampling segment. It is not a predicted trajectory.
            return traveled
        traveled += segment.length
    return None


class PublicPriorityTracker:
    def __init__(self, approach_horizon_s=5., max_approach_distance_m=45.,
                 stationary_speed_mps=.3, lane_width_m=3.5, clearance_m=.35):
        self.horizon = _number(approach_horizon_s, 'approach horizon')
        self.max_distance = _number(max_approach_distance_m, 'approach range')
        self.stationary = _number(stationary_speed_mps, 'stationary threshold')
        self.width = _number(lane_width_m, 'lane width')
        self.clearance = _number(clearance_m, 'clearance')
        if min(self.horizon, self.max_distance, self.width) <= 0 or min(self.stationary, self.clearance) < 0:
            raise ValueError('invalid public priority parameters')
        self._seen = {}
        self._last_time, self._last_frame = None, None
        self._navigation_key, self._navigation = None, None
        self._route_s, self._ego_position = None, None
        self._regions = {}
        self._lane_cache = {}

    def _lane(self, lane_id, lane_paths):
        source = lane_paths.get(lane_id)
        if source is None:
            return None
        center = source.get('center')
        if center is None or len(center) < 2:
            return None
        key = (lane_id, tuple(tuple(float(v) for v in p[:2]) for p in center), tuple(source.get('successors', ())))
        if key not in self._lane_cache:
            self._lane_cache[key] = (LineString(key[1]), key[2])
        return self._lane_cache[key]

    def _public_path_approach(self, actor, lane_paths, region, distance):
        first = self._lane(actor.get('lane_id'), lane_paths)
        if first is None:
            return None, 'current_lane_geometry_unavailable'
        line, _ = first
        position = Point(actor['position'])
        start = line.project(position)
        tangent_a = line.interpolate(max(0., start-.25))
        tangent_b = line.interpolate(min(line.length, start+.25))
        dx, dy = tangent_b.x-tangent_a.x, tangent_b.y-tangent_a.y
        norm = math.hypot(dx, dy)
        vx, vy = actor.get('velocity', (0., 0.))
        if norm <= 1e-9 or (vx*dx+vy*dy)/norm <= self.stationary:
            return None, 'not_moving_along_public_lane'
        queue = deque([(actor.get('lane_id'), start, 0.)])
        visited, incomplete = {}, False
        while queue:
            lane_id, offset, traveled = queue.popleft()
            if lane_id in visited and visited[lane_id] <= traveled:
                continue
            visited[lane_id] = traveled
            if len(visited) > 64:
                return None, 'public_topology_search_limit'
            value = self._lane(lane_id, lane_paths)
            if value is None:
                incomplete = True
                continue
            center, successors = value
            remaining = distance-traveled
            hit = _first_path_hit(center, offset, remaining, region, actor['width']/2+self.clearance)
            if hit is not None:
                return traveled+hit, 'public_lane_and_successor_sweep'
            spent = max(0., center.length-offset)
            if spent >= remaining:
                continue
            for successor in successors:
                next_value = self._lane(successor, lane_paths)
                if next_value is None:
                    incomplete = True
                    continue
                gap = Point(center.coords[-1]).distance(Point(next_value[0].coords[0]))
                if gap > 6.:
                    incomplete = True
                    continue
                if gap > 1e-8:
                    connector = LineString([center.coords[-1], next_value[0].coords[0]])
                    connector_hit = _first_path_hit(connector, 0., remaining-spent, region,
                                                     actor['width']/2+self.clearance)
                    if connector_hit is not None:
                        return traveled+spent+connector_hit, 'public_lane_and_successor_sweep'
                queue.append((successor, 0., traveled+spent+gap))
        return None, 'public_topology_incomplete' if incomplete else 'public_continuations_do_not_reach_conflict'

    def _same_lane_following(self, ego, actor, region):
        if not ego.get('lane_id') or ego.get('lane_id') != actor.get('lane_id'):
            return False
        c, s = math.cos(ego['heading']), math.sin(ego['heading'])
        dx = actor['position'][0]-ego['position'][0]
        dy = actor['position'][1]-ego['position'][1]
        return (dx*c+dy*s > 0 and abs(-dx*s+dy*c) < self.width*.65
                and math.cos(actor['heading']-ego['heading']) > .85)

    def _project_ego(self, ego):
        position = Point(ego['position'])
        navigation = self._navigation
        if self._route_s is None:
            low, high = 0., min(navigation.length, 30.)
        else:
            travel = position.distance(Point(self._ego_position))
            low = max(0., self._route_s-10.)
            high = min(navigation.length, self._route_s+max(20., 2.*travel+10.))
        local = substring(navigation, low, high)
        if local.geom_type != 'LineString':
            raise ValueError('public priority has no local route projection support')
        distance = low+local.project(position)
        self._route_s, self._ego_position = distance, list(ego['position'])
        return distance, local, low

    def update(self, public_payload, ego_dict, actors_list, navigation_xy, time_s, lane_paths=None):
        now = _number(time_s, 'observation time')
        payload = public_payload or {}
        frame = payload.get('frame')
        if frame is not None and (type(frame) is not int or frame < 0):
            raise ValueError('public rule frame must be nonnegative integer')
        if self._last_time is not None and now < self._last_time-1e-8:
            raise ValueError('public priority cannot consume an older observation')
        if frame is not None and self._last_frame is not None and frame < self._last_frame:
            raise ValueError('public priority cannot consume an older frame')
        self._last_time, self._last_frame = now, frame
        coordinates = tuple(_coordinates(navigation_xy))
        if len(coordinates) < 2:
            raise ValueError('public rules need the locked original navigation')
        if coordinates != self._navigation_key:
            if self._navigation_key is not None:
                raise ValueError('public priority locked navigation changed within an episode')
            self._navigation_key, self._navigation = coordinates, LineString(coordinates)
        navigation = self._navigation
        ego = ego_dict
        ego_shape = _shape(ego)
        ego_s, local_navigation, local_offset = self._project_ego(ego)
        actors = {str(actor['id']): actor for actor in actors_list if str(actor['id']) != str(ego['id'])}
        if len(actors) != len([actor for actor in actors_list if str(actor['id']) != str(ego['id'])]):
            raise ValueError('duplicate physical actor identities in priority input')
        lanes = lane_paths or {}
        rows, stops = [], []
        complete = True
        live_rules = set()
        for rule in payload.get('rules', []):
            identity = str(rule.get('id'))
            if identity in live_rules:
                raise ValueError('duplicate public priority rule identity')
            live_rules.add(identity)
            if rule.get('kind') != 'yield':
                complete = False
                rows.append({'rule_id': identity, 'status': 'unsupported_rule_kind'})
                continue
            low, high = (_number(v, 'conflict route distance') for v in rule['conflict_route_s_m'])
            stop_s = _number(rule['stop_route_s_m'], 'stop route distance')
            half_width = _number(rule.get('conflict_half_width_m', self.width/2), 'conflict half width')
            if not (0 <= stop_s <= low < high <= navigation.length+1e-4) or half_width <= 0:
                raise ValueError('public stop/conflict must lie in order on the locked route')
            priority_lanes = set(rule.get('priority_lane_ids', ()))
            controlled_lanes = set(rule.get('controlled_lane_ids', ()))
            if not priority_lanes or not controlled_lanes:
                complete = False
                rows.append({'rule_id': identity, 'status': 'missing_public_lane_priority'})
                continue
            key = (identity, low, high, half_width)
            if key not in self._regions:
                segment = substring(navigation, low, min(high, navigation.length))
                self._regions[key] = segment.buffer(half_width, cap_style=2, join_style=2)
            region = self._regions[key]
            memory = self._seen.setdefault(identity, {})
            # Actor disappearance ends this rule's authority over it. We never
            # move an invisible actor forward from its last known state.
            for actor_id in list(memory):
                if actor_id not in actors:
                    del memory[actor_id]
            for actor_id, actor in actors.items():
                if (actor.get('kind') in ('vehicle', 'bicycle', 'pedestrian')
                        and actor.get('lane_id') in priority_lanes):
                    memory.setdefault(actor_id, {'first_observed_time': now, 'occupied': False})
                    memory[actor_id]['last_priority_lane_time'] = now
            applicable = (ego_s <= high and navigation.distance(Point(ego['position'])) <= 2*self.width
                          and stop_s-ego_s <= max(60., self.max_distance))
            blockers, ignored = [], []
            for actor_id in list(memory):
                actor = actors[actor_id]
                if not applicable:
                    continue
                if abs(float(actor.get('z', 0.))-float(ego.get('z', 0.))) > 2.:
                    ignored.append({'actor_id': actor_id, 'reason': 'different_elevation'})
                    continue
                occupied = region.intersects(_shape(actor))
                if memory[actor_id]['occupied'] and not occupied:
                    del memory[actor_id]
                    ignored.append({'actor_id': actor_id, 'reason': 'observed_clear_of_conflict'})
                    continue
                if self._same_lane_following(ego, actor, region):
                    ignored.append({'actor_id': actor_id, 'reason': 'same_direction_same_lane_lead_for_following'})
                    continue
                memory[actor_id]['occupied'] = occupied
                vx, vy = (_number(v, 'observed velocity') for v in actor.get('velocity', (0., 0.)))
                speed = math.hypot(vx, vy)
                if occupied:
                    blockers.append({'actor_id': actor_id, 'reason': 'current_conflict_occupancy', 'entry_time_lower_bound_s': 0.})
                    continue
                if speed <= self.stationary:
                    ignored.append({'actor_id': actor_id, 'reason': 'stationary_outside_conflict'})
                    continue
                reach = min(self.max_distance, speed*self.horizon+actor['length']/2)
                if Point(actor['position']).distance(region) > reach+actor['length']/2:
                    ignored.append({'actor_id': actor_id, 'reason': 'outside_current_motion_horizon'})
                    continue
                distance, mode = self._public_path_approach(actor, lanes, region, reach) if lanes else (None, 'current_lane_geometry_unavailable')
                if mode == 'current_lane_geometry_unavailable':
                    start = actor['position']
                    end = (start[0]+vx/speed*reach, start[1]+vy/speed*reach)
                    distance = _first_path_hit(LineString([start, end]), 0., reach, region, actor['width']/2+self.clearance)
                    mode = 'current_observed_velocity_sweep_without_public_lane_geometry'
                    complete = False
                elif mode in ('public_topology_search_limit', 'public_topology_incomplete'):
                    complete = False
                if distance is not None:
                    blockers.append({'actor_id': actor_id, 'reason': mode, 'entry_distance_lower_bound_m': distance,
                                     'entry_time_lower_bound_s': distance/max(speed, self.stationary),
                                     'current_speed_mps': speed})
                else:
                    ignored.append({'actor_id': actor_id, 'reason': mode})
            front = Point(ego['position'][0]+math.cos(ego['heading'])*ego['length']/2,
                          ego['position'][1]+math.sin(ego['heading'])*ego['length']/2)
            front_s = local_offset+local_navigation.project(front)
            can_stop_at_line = front_s <= stop_s+.05 and not ego_shape.intersects(region)
            if applicable and blockers and can_stop_at_line:
                point = navigation.interpolate(stop_s)
                stops.append((stop_s, [point.x, point.y], identity))
            rows.append({'rule_id': identity, 'applicable': applicable,
                'status': ('yield_before_public_line' if blockers and applicable and can_stop_at_line else
                           'conflict_already_entered_use_current_obstacle_avoidance' if blockers and applicable else 'clear'),
                'stop_route_s_m': stop_s, 'conflict_route_s_m': [low, high],
                'controlled_lane_ids': sorted(controlled_lanes), 'priority_lane_ids': sorted(priority_lanes),
                'blockers': blockers, 'ignored': ignored, 'current_observed_priority_ids': sorted(memory),
                'stop_line_behind_ego_front': not can_stop_at_line})
        for identity in set(self._seen)-live_rules:
            del self._seen[identity]
        chosen = min(stops, default=None)
        return {'available': complete and bool(payload.get('rules')),
            'stop_position': chosen[1] if chosen else None,
            'evidence': {'version': VERSION, 'frame': frame, 'time_s': now,
                'ego_locked_route_s_m': ego_s, 'chosen_rule_id': chosen[2] if chosen else None,
                'full_observed_actor_count': len(actors), 'rules': rows,
                'settings': {'approach_horizon_s': self.horizon, 'max_approach_distance_m': self.max_distance,
                             'stationary_speed_mps': self.stationary, 'lane_width_m': self.width, 'clearance_m': self.clearance},
                'input_policy': 'current_range_truth_plus_causal_priority_identity_and_public_lane_successors',
                'limitations': ['public branching paths conservatively screened, no NPC future intent',
                                'this screen is not complete traffic negotiation or collision certification',
                                'missing public geometry is explicit; invisible actors are not extrapolated']}}
