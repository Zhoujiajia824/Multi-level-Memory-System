"""PDM-Closed core executed on CARLA observations and road geometry.

This is a simulator adapter, not the original nuPlan benchmark planner. The
official generator, LQR/bicycle simulator, scorer and emergency detector are
imported unchanged from pinned upstream sources. See README.md for deviations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from .dependencies import load_dependencies, validate_dependencies
from .selection import choose_candidate
from .continuity import TrajectoryReference, trajectory_continuity_costs


def _lane_key(wp):
    return "%s:%s:%s" % (wp.road_id, wp.section_id, wp.lane_id)


def _heading(wp):
    return -math.radians(wp.transform.rotation.yaw)


def _xy(wp):
    p = wp.transform.location
    return np.array([p.x, -p.y], dtype=float)


def _same_direction(first, second):
    return math.cos(_heading(first) - _heading(second)) > 0.8


class RoadGeometryError(ValueError):
    """Preserve the failed snapshot even when no proposal can be generated."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _adjacent(wp, side):
    return wp if side == "center" else (wp.get_left_lane() if side == "left" else wp.get_right_lane())


def _connected(first, second):
    """Do not join unrelated lanes merely because they are geometrically close."""
    distance = float(np.linalg.norm(_xy(second) - _xy(first)))
    if distance > 6.0 or not _same_direction(first, second):
        return False
    if _lane_key(first) == _lane_key(second):
        return True
    # next() follows road successors without changing lanes. A small tolerance
    # accommodates the difference between arc distance and waypoint chord.
    return any(_lane_key(wp) == _lane_key(second)
               and float(np.linalg.norm(_xy(wp) - _xy(second))) < 1.0
               for wp in first.next(max(0.05, distance)))


def _lane_corridor(reference, closest, side, progress):
    """Continuous local lane containing ego's station, with a finite endpoint.

    Missing lane segments behind ego do not invalidate the road ahead. At road
    seams verify successors rather than stitching equal left/right indices.
    """
    import carla
    lanes = [_adjacent(base, side) for base in reference]

    def usable(i):
        return (lanes[i] is not None and lanes[i].lane_type == carla.LaneType.Driving
                and _same_direction(reference[i], lanes[i]))

    if not usable(closest):
        return [], {"reason": "no_same_direction_lane_at_ego", "distance_ahead_m": 0.0}
    start, end = closest, closest
    while start > 0 and usable(start - 1) and _connected(lanes[start - 1], lanes[start]):
        start -= 1
    cut = None
    while end + 1 < len(lanes):
        i = end + 1
        reason = ("no_continuous_same_direction_driving_lane" if not usable(i)
                  else "disconnected_lane_successor" if not _connected(lanes[end], lanes[i]) else None)
        if reason:
            cut = {"reason": reason, "reference_index": i,
                   "distance_ahead_m": float(progress[i] - progress[closest]),
                   "last_lane_key": _lane_key(lanes[end])}
            break
        end = i
    return lanes[start:end + 1], cut


def _junction_lane_match(reference, closest, origin, heading, height):
    """Resolve overlapping junction road IDs against the ordered route geometry.

    CARLA's nearest waypoint does not use vehicle heading. At a junction it
    can return a different connection passing through the same position.
    Require a unique containing, aligned lane on the same deck; this is not
    permission to snap an off-route vehicle onto a distant route.
    """
    import carla
    matches = {}
    for side in ("center", "left", "right"):
        for i in range(max(0, closest - 3), min(len(reference) - 1, closest + 3)):
            a, b = _adjacent(reference[i], side), _adjacent(reference[i + 1], side)
            if (a is None or b is None or not (a.is_junction or b.is_junction)
                    or a.lane_type != carla.LaneType.Driving or b.lane_type != carla.LaneType.Driving
                    or not _same_direction(reference[i], a) or not _same_direction(reference[i + 1], b)
                    or not _connected(a, b)):
                continue
            delta = _xy(b) - _xy(a)
            length2 = float(delta @ delta)
            if length2 < 1e-6:
                continue
            u = float((origin - _xy(a)) @ delta / length2)
            if u < 0.0:
                previous = _adjacent(reference[i - 1], side) if i > 0 else None
                if previous is None or not _connected(previous, a):
                    continue
            elif u > 1.0:
                following = _adjacent(reference[i + 2], side) if i + 2 < len(reference) else None
                if following is None or not _connected(b, following):
                    continue
            # At the outer side of a polyline bend, both perpendicular feet
            # can fall beyond their respective segments. The shared internal
            # vertex still belongs to the connected lane; real endpoints do
            # not receive this allowance. Containment/direction limits remain.
            u = float(np.clip(u, 0.0, 1.0))
            offset = float(np.linalg.norm(origin - (_xy(a) + u * delta)))
            route_heading = math.atan2(delta[1], delta[0])
            height_error = abs(height - ((1 - u) * a.transform.location.z + u * b.transform.location.z))
            if (offset > min(a.lane_width, b.lane_width) / 2 - 0.05
                    or math.cos(heading - route_heading) < math.cos(math.radians(30))
                    or height_error > 1.0):
                continue
            evidence = {"offset_m": offset, "height_error_m": height_error,
                        "reference_index": i, "lane_keys": [_lane_key(a), _lane_key(b)]}
            if side not in matches or offset < matches[side]["offset_m"]:
                matches[side] = evidence
    ranked = sorted(matches, key=lambda side: matches[side]["offset_m"])
    if not ranked or (len(ranked) > 1 and matches[ranked[1]]["offset_m"] - matches[ranked[0]]["offset_m"] < 0.25):
        return None, {"method": "junction_geometry_rejected", "matches": matches}
    return ranked[0], {"method": "junction_geometry", "matches": matches}


def _path_states(points, api, first_heading=None):
    points = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("PDM reference path has fewer than two distinct points")
    delta = np.empty_like(points)
    delta[0], delta[-1] = points[1] - points[0], points[-1] - points[-2]
    delta[1:-1] = points[2:] - points[:-2]
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    if first_heading is not None:
        yaw[0] = first_heading
    return [api.StateSE2(float(p[0]), float(p[1]), float(h)) for p, h in zip(points, yaw)]


def _lane_path_points(waypoints):
    """Continuous map tangents for planning, without moving route checkpoints.

    A two-metre polyline changes tangent abruptly at each bend vertex. Feeding
    that chord tangent into pose recovery amplifies sampling error and makes
    repeated replans depend on which side of a vertex ego happens to project.
    Interpolate only between connected map samples, using their actual heading.
    Road safety is still checked against independently constructed lane surfaces.
    """
    from scipy.interpolate import CubicHermiteSpline
    points = np.array([_xy(wp) for wp in waypoints])
    if len(points) < 2:
        raise ValueError("Reference path has fewer than two waypoints")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]
    headings = np.array([_heading(wp) for wp in waypoints])[keep]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Reference path has fewer than two distinct waypoints")
    stations = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    tangents = np.column_stack((np.cos(headings), np.sin(headings)))
    curve = CubicHermiteSpline(stations, points, tangents, axis=0)
    samples = np.unique(np.r_[stations, np.arange(0.0, stations[-1], 0.25)])
    return curve(samples)


def _smooth_path(points, rear, heading, api, transition_m):
    """Start every candidate at the measured rear axle, with a gradual merge."""
    from shapely.geometry import LineString, Point
    line = LineString(points)
    start = line.project(Point(*rear))
    distances = np.arange(0, line.length - start, 1.0)
    if len(distances) < 3:
        raise ValueError("Reference path does not extend sufficiently ahead of ego")
    distances = np.r_[distances, line.length - start]
    nominal = np.array([[line.interpolate(start + d).x, line.interpolate(start + d).y] for d in distances])
    # A receding-horizon path must preserve the measured spatial tangent. A
    # fresh zero-derivative lateral blend at every 0.2s update repeatedly resets
    # the developing lane change and can leave the vehicle in its original lane.
    # Cubic Hermite meets the current pose/tangent and joins the target lane
    # with zero displacement/tangent correction; it needs no semantic history.
    u = np.clip(distances / transition_m, 0, 1)
    blend = 2 * u ** 3 - 3 * u ** 2 + 1
    tangent_blend = u ** 3 - 2 * u ** 2 + u
    ahead = line.interpolate(min(line.length, start + 1e-3))
    tangent = np.array([ahead.x, ahead.y]) - nominal[0]
    tangent /= np.linalg.norm(tangent)
    tangent_delta = np.array([math.cos(heading), math.sin(heading)]) - tangent
    shifted = (nominal + (rear - nominal[0])[None, :] * blend[:, None]
               + transition_m * tangent_delta[None, :] * tangent_blend[:, None])
    shifted[0] = rear
    return api.PDMPath(_path_states(shifted, api, heading))


class _IntersectionMap:
    def __init__(self, carla_map, height, api):
        self.map, self.height, self.api = carla_map, height, api

    def is_in_layer(self, point, layer):
        if layer != self.api.SemanticMapLayer.INTERSECTION:
            return False
        import carla
        wp = self.map.get_waypoint(carla.Location(x=float(point.x), y=float(-point.y), z=self.height),
                                   project_to_road=False, lane_type=carla.LaneType.Driving)
        return bool(wp and wp.is_junction)


class CarlaPDMPlanner:
    """Synchronous snapshot -> official PDM candidate rollouts -> audited choice."""

    def __init__(self, world, ego, route, config=None):
        self.api = load_dependencies()  # Fail before expensive VLM initialization.
        self.world, self.ego, self.route = world, ego, route
        self.map = world.get_map()
        self.config = dict(config or {})
        self.dt = float(self.config.get("sample_interval_s", 0.1))
        # PDMScorer's TTC projection uses fixed 10-Hz offsets in upstream code.
        if abs(self.dt - 0.1) > 1e-8:
            raise ValueError("Pinned upstream PDM scoring requires sample_interval_s=0.1")
        self.horizon = float(self.config.get("horizon_s", 4.0))
        self.max_speed = float(self.config.get("max_speed_mps", 8.0))
        self.map_radius = float(self.config.get("map_radius_m", 100.0))
        self.transition_m = float(self.config.get("lane_change_length_m", 30.0))
        self.lateral_weight = float(self.config.get("lateral_semantic_weight", 0.8))
        self.longitudinal_weight = float(self.config.get("longitudinal_semantic_weight", 0.8))
        self.continuity_weight = float(self.config.get("continuity_weight", 0.15))
        if not math.isfinite(self.continuity_weight) or not 0 <= self.continuity_weight <= 0.5:
            raise ValueError("continuity_weight must be finite and within [0, 0.5]")
        self._previous_plan = None
        self._continuity_reference = None
        self._continuity_timestamp = None
        self._reference_route_source = None
        self._resolved_route_waypoints = None
        self._route_resolution = None
        self._junction_area_cache = {}
        for name, value in (("horizon_s", self.horizon), ("max_speed_mps", self.max_speed),
                            ("map_radius_m", self.map_radius), ("lane_change_length_m", self.transition_m)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        if self.horizon < 3 or self.max_speed > 20 or self.transition_m < 15:
            raise ValueError("PDM requires horizon >=3s, max speed <=20m/s, and lane-change length >=15m")
        self.samples = int(math.ceil(self.horizon / self.dt))
        self.sampling = self.api.TrajectorySampling(num_poses=self.samples, interval_length=self.dt)
        # Upstream's update_rate=2 leaves the freshly reset t=1 leader state at
        # zero, creating a phantom stopped leader on every short replan. Updating
        # at each sample initializes the first prediction and prevents repeated
        # artificial braking in the 0.2s CARLA planning loop.
        self.generator = self.api.PDMGenerator(self.sampling, self.sampling, leading_agent_update_rate=1)
        self.simulator = self.api.PDMSimulator(self.sampling)
        self.scorer = self.api.PDMScorer(self.sampling)
        self.emergency = self.api.PDMEmergencyBrake(self.sampling)
        self.vehicle_parameters = self._vehicle_parameters()
        # Upstream resets only the bicycle's vehicle per rollout. Its LQR
        # otherwise retains the default Pacifica wheelbase and its bicycle
        # keeps a 60-degree steering limit. Bind both official components to
        # this CARLA vehicle so the controller and simulated plant agree.
        self.simulator._tracker = self.api.BatchLQRTracker(
            vehicle=self.vehicle_parameters, discretization_time=self.dt)
        self.simulator._motion_model = self.api.BatchKinematicBicycleModel(
            vehicle=self.vehicle_parameters, max_steering_angle=self.max_steer)
        self.provenance = dict(self.api.provenance)

    def _vehicle_parameters(self):
        box = self.ego.bounding_box
        half_length, width = float(box.extent.x), 2 * float(box.extent.y)
        wheels = self.ego.get_physics_control().wheels
        if len(wheels) < 4:
            raise ValueError("PDM bicycle model requires a four-wheel ego vehicle")
        # CARLA WheelPhysicsControl.position uses centimetres.
        front = np.mean([[w.position.x, w.position.y, w.position.z] for w in wheels[:2]], axis=0)
        rear = np.mean([[w.position.x, w.position.y, w.position.z] for w in wheels[-2:]], axis=0)
        wheelbase = float(np.linalg.norm(front - rear) * 0.01)
        if not 1.0 < wheelbase < 6.0:
            raise ValueError("CARLA returned invalid wheelbase geometry")
        # Geometric center / rear-axle distance is explicitly configurable; the
        # default half-wheelbase approximation is recorded in every plan.
        center_offset = float(self.config.get("rear_axle_to_center_m", wheelbase / 2))
        if not 0 < center_offset < half_length:
            raise ValueError("rear_axle_to_center_m must lie within the ego footprint")
        self.max_steer = math.radians(max(float(w.max_steer_angle) for w in wheels[:2]))
        if not 0 < self.max_steer < math.pi / 2:
            raise ValueError("CARLA returned invalid front-wheel steering limits")
        return self.api.VehicleParameters(width=width, front_length=half_length + center_offset,
                                         rear_length=half_length - center_offset,
                                         cog_position_from_rear_axle=center_offset, wheel_base=wheelbase,
                                         vehicle_name=self.ego.type_id, vehicle_type="carla",
                                         height=2 * float(box.extent.z))

    def _ego_state(self, timestamp_us):
        import carla
        from carla_bridge.state.coords import rotate_vector_to_ego
        tf = self.ego.get_transform()
        box = self.ego.bounding_box
        center = tf.transform(carla.Location(x=box.location.x, y=box.location.y, z=box.location.z))
        yaw = math.radians(tf.rotation.yaw)
        velocity, acceleration = self.ego.get_velocity(), self.ego.get_acceleration()
        v = rotate_vector_to_ego(velocity.x, velocity.y, yaw)
        a = rotate_vector_to_ego(acceleration.x, acceleration.y, yaw)
        if v[0] < -0.1:
            raise ValueError("Reverse driving is unsupported by this PDM adapter")
        state = self.api.EgoState.build_from_center(
            center=self.api.StateSE2(center.x, -center.y, -yaw),
            center_velocity_2d=self.api.StateVector2D(max(0.0, v[0]), v[1]),
            center_acceleration_2d=self.api.StateVector2D(*a),
            tire_steering_angle=-float(self.ego.get_control().steer) * self.max_steer,
            time_point=self.api.TimePoint(timestamp_us), vehicle_parameters=self.vehicle_parameters,
            angular_vel=-math.radians(self.ego.get_angular_velocity().z),
        )
        return state, tf

    def _reference_waypoints(self):
        from .route_waypoints import resolve_route_waypoints
        location = self.ego.get_location()
        raw = self.route._route
        if (self._reference_route_source is not raw or self._resolved_route_waypoints is None
                or len(self._resolved_route_waypoints) != len(raw)):
            retained = getattr(self.route, "_route_waypoints", None)
            if retained is not None and len(retained) == len(raw):
                self._resolved_route_waypoints = list(retained)
                self._route_resolution = getattr(self.route, "_waypoint_resolution", None)
            else:
                self._resolved_route_waypoints, self._route_resolution = resolve_route_waypoints(self.map, raw)
            self._reference_route_source = raw
        self._geometry_diagnostics["reference_resolution"] = self._route_resolution
        index = self.route._nearest_index(location)
        start = max(0, index - 6)
        points, travelled = [], 0.0
        span = max(55.0, self.max_speed * self.horizon + self.transition_m + 10)
        for wp in self._resolved_route_waypoints[start:]:
            if points:
                travelled += float(np.linalg.norm(_xy(wp) - _xy(points[-1])))
            points.append(wp)
            if travelled >= span:
                break
        if len(points) < 2:
            raise ValueError("PDM requires a nonempty reference route")
        if start == 0:
            for _ in range(4):
                previous = points[0].previous(2.0)
                if not previous:
                    break
                wp = min(previous, key=lambda value: abs(math.sin(_heading(value) - _heading(points[0]))))
                points.insert(0, wp)
        # The official IDM treats the end of a path as a stationary leader.
        # Extend the *planning* path along real road so that it does not stop
        # several metres before the fixed evaluation route's final checkpoint.
        while travelled < span:
            options = points[-1].next(2.0)
            if not options:
                break
            chosen = min(options, key=lambda wp: abs(math.atan2(math.sin(_heading(wp) - _heading(points[-1])),
                                                                 math.cos(_heading(wp) - _heading(points[-1])))))
            distance = float(np.linalg.norm(_xy(chosen) - _xy(points[-1])))
            if distance < 0.1:
                break
            travelled += distance
            points.append(chosen)
        return points

    def _road_geometry(self, ego_state):
        """Actual CARLA lane ribbons; legal adjacent lanes are never +/-1m nudges."""
        import carla
        from shapely.geometry import Polygon, LineString, Point
        from shapely.ops import unary_union
        reference = self._reference_waypoints()
        self._reference_for_height = reference
        ego_wp = self.map.get_waypoint(self.ego.get_location(), project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            raise ValueError("Ego is outside driving lanes")
        rear = ego_state.rear_axle.point.array
        origin = np.array([self.ego.get_location().x, -self.ego.get_location().y])
        reference_line = LineString([_xy(wp) for wp in reference])
        projected = reference_line.interpolate(reference_line.project(Point(*origin)))
        base_wp = self.map.get_waypoint(carla.Location(x=projected.x, y=-projected.y,
                                                       z=self.ego.get_location().z), project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
        if base_wp is None:
            raise ValueError("Reference projection is outside driving lanes")
        closest = min(range(len(reference)), key=lambda i: float(np.linalg.norm(_xy(reference[i]) - origin)))
        progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff([_xy(wp) for wp in reference], axis=0), axis=1))]
        current_side = None
        for side, candidate in (("center", base_wp), ("left", base_wp.get_left_lane()),
                                 ("right", base_wp.get_right_lane())):
            if candidate is not None and _lane_key(candidate) == _lane_key(ego_wp):
                current_side = side
        match = {"method": "lane_id"}
        if current_side is None:
            current_side, match = _junction_lane_match(
                reference, closest, origin, -math.radians(self.ego.get_transform().rotation.yaw),
                self.ego.get_location().z)
        self._geometry_diagnostics.update(
            ego_lane_key=_lane_key(ego_wp), reference_lane_key=_lane_key(base_wp),
            current_side=current_side, lane_match=match, closest_reference_index=closest,
            reference_lane_keys=list(dict.fromkeys(_lane_key(wp) for wp in reference)))
        if current_side is None:
            raise ValueError("Ego has left the reference route and its adjacent driving lanes")

        strips = defaultdict(list)
        on_route_keys = set()
        # Include all driving lane ribbons next to the route for the drivable
        # area; only lanes oriented along the route count as on-route direction.
        for i, first in enumerate(reference[:-1]):
            second = reference[i + 1]
            lanes_a, lanes_b = {"center": first}, {"center": second}
            for side in ("left", "right"):
                a, b = first, second
                for lane_index in range(1, 5):
                    a = a.get_left_lane() if side == "left" else a.get_right_lane()
                    b = b.get_left_lane() if side == "left" else b.get_right_lane()
                    if a is None or b is None or a.lane_type != carla.LaneType.Driving or b.lane_type != carla.LaneType.Driving:
                        break
                    lanes_a[side + str(lane_index)], lanes_b[side + str(lane_index)] = a, b
            for lane_name, a in lanes_a.items():
                b = lanes_b[lane_name]
                if not _connected(a, b):
                    continue
                corners = []
                for wp, sign in ((a, 1), (b, 1), (b, -1), (a, -1)):
                    h = _heading(wp)
                    corners.append(_xy(wp) + sign * float(wp.lane_width) / 2 * np.array([-math.sin(h), math.cos(h)]))
                polygon = Polygon(corners)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if polygon.is_empty:
                    continue
                key = _lane_key(a)
                strips[key].append(polygon)
                if _same_direction(first, a):
                    on_route_keys.add(key)
        polygons = {key: unary_union(value).buffer(0.025) for key, value in strips.items()}
        if not polygons:
            raise ValueError("No usable CARLA road polygons")
        lane_dict = {key: SimpleNamespace(polygon=poly) for key, poly in polygons.items() if key in on_route_keys}
        # The route is only one corridor through the intersection. A vehicle's
        # corners can legally pass over other real junction connections while
        # turning. Include their actual surfaces without making them route
        # targets or granting lane-change permission.
        from .junction_area import junction_driving_polygons
        junction_polygons, junction_info = junction_driving_polygons(reference, cache=self._junction_area_cache)
        self._geometry_diagnostics["junction_drivable_area"] = junction_info
        for key, polygon in junction_polygons.items():
            polygons[key] = polygons[key].union(polygon) if key in polygons else polygon
        occupancy = self.api.PDMOccupancyMap(list(polygons), np.array(list(polygons.values()), dtype=object))

        paths, metadata, excluded = [], [], {}
        path_details = {}
        self._geometry_diagnostics.update(excluded_lateral_paths=excluded, lateral_paths=path_details)
        for side in ("center", "left", "right"):
            reason = None
            corridor, cut = _lane_corridor(reference, closest, side, progress)
            details = path_details[side] = {"cut": cut}
            if side == "center" and cut is not None:
                # A prepared reference must be a connected route. Treat losing
                # its topology as a map/route error, not an invisible stop line
                # that IDM can accept indefinitely as a planned stop.
                raise ValueError("Reference route is disconnected at index %s: %s" %
                                 (cut.get("reference_index"), cut.get("reason")))
            # A change of two lanes is never a candidate.
            if current_side != "center" and side not in (current_side, "center"):
                reason = "would_cross_two_lanes"
            if not corridor:
                reason = reason or "no_same_direction_lane_at_ego"

            # Detect the last opportunity to return before a junction, a lane
            # ending/disconnection, or a marking that forbids returning. This
            # uses map evidence, never the requested semantic action.
            barrier = cut if side != "center" else None
            for i in range(closest, len(reference)):
                base = reference[i]
                distance_ahead = float(progress[i] - progress[closest])
                wp = _adjacent(base, side)
                if side != "center":
                    return_permission = carla.LaneChange.Right if side == "left" else carla.LaneChange.Left
                    if (wp is None or base.is_junction or wp.is_junction
                            or not (int(wp.lane_change) & int(return_permission))):
                        limit = {"reason": "return_lane_change_not_permitted_or_intersection",
                                 "reference_index": i, "distance_ahead_m": distance_ahead}
                        if barrier is None or distance_ahead < barrier["distance_ahead_m"]:
                            barrier = limit
                if reason is None and side != current_side and distance_ahead <= self.transition_m + 4:
                    source = _adjacent(base, current_side)
                    direction = side if current_side == "center" else ("right" if current_side == "left" else "left")
                    permission = carla.LaneChange.Left if direction == "left" else carla.LaneChange.Right
                    if (source is None or wp is None or base.is_junction or source.is_junction or wp.is_junction
                            or not (int(source.lane_change) & int(permission))):
                        reason = "lane_change_not_permitted_or_intersection"
                        details["permission_failure_distance_m"] = distance_ahead
                        details["permission_failure"] = {
                            "source_lane_key": _lane_key(source) if source else None,
                            "target_lane_key": _lane_key(wp) if wp else None,
                            "source_lane_change": int(source.lane_change) if source else None,
                            "reference_is_junction": bool(base.is_junction)}
            details["return_barrier"] = barrier
            recovery_required = (barrier is not None and barrier["distance_ahead_m"]
                                 <= self.transition_m + 4 + self.max_speed * self.horizon)
            details["route_recovery_required"] = recovery_required
            if barrier is not None and barrier["distance_ahead_m"] > 0:
                # If recovery is temporarily blocked, IDM can slow/stop on the
                # current lane before its known return boundary. It must not
                # cruise through an unrelated junction branch just because
                # geometrically adjacent waypoints continue to exist there.
                limit_station = progress[barrier["reference_index"]]
                corridor = [wp for wp in corridor
                            if reference_line.project(Point(*_xy(wp))) < limit_station - 1e-4]
                details["bounded_before_return_barrier"] = True
            # Do not start a lane change that cannot finish inside this finite
            # corridor. Staying on a short lane is still a valid braking path.
            endpoint = barrier or cut
            if (reason is None and side != current_side and endpoint is not None
                    and endpoint["distance_ahead_m"] <= self.transition_m + 4):
                reason = "insufficient_lane_change_corridor"
            if reason:
                excluded[side] = reason
                continue
            if side == current_side:
                label, semantics = "KEEP_LANE", ["KEEP_LANE"]
                if side == "center":
                    semantics.append("FOLLOW_ROUTE")
            elif side == "center":
                label = "CHANGE_LANE_RIGHT" if current_side == "left" else "CHANGE_LANE_LEFT"
                semantics = [label, "FOLLOW_ROUTE"]
            else:
                label = "CHANGE_LANE_" + side.upper()
                semantics = [label]
            # Following a lane needs local pose recovery. Carrying a small
            # tracking error through a 30-m lane-change blend can push the
            # outgoing footprint over a curb in a short-radius turn. Use two
            # wheelbases / two seconds of current travel for same-lane recovery;
            # actual lane changes keep the configured full transition distance.
            transition = self.transition_m
            if side == current_side:
                transition = min(transition, max(2 * self.vehicle_parameters.wheel_base,
                                                  2 * ego_state.dynamic_car_state.speed))
            details["path_kind"] = "lane_follow" if side == current_side else "lane_change"
            details["transition_length_m"] = float(transition)
            try:
                path = _smooth_path(_lane_path_points(corridor), rear,
                                    ego_state.rear_axle.heading, self.api, transition)
            except ValueError as exc:
                excluded[side] = "insufficient_path_ahead"
                details["error"] = str(exc)
                continue
            paths.append(path)
            details["path_length_m"] = float(path.length)
            metadata.append({"id": side, "label": label, "semantics": sorted(set(semantics)),
                             "route_recovery_required": recovery_required})
        if not paths:
            raise ValueError("No legal lateral path from the current lane")
        centerline = self.api.PDMPath(_path_states([_xy(wp) for wp in reference], self.api))
        return paths, metadata, centerline, occupancy, lane_dict, excluded

    def _observed_actors(self, timestamp_us, tf):
        import carla
        tracked = []
        ego_box = self.ego.bounding_box
        road_z = self.map.get_waypoint(tf.location, project_to_road=True).transform.location.z
        bottom = tf.location.z + ego_box.location.z - ego_box.extent.z - road_z - 0.3
        top = tf.location.z + ego_box.location.z + ego_box.extent.z - road_z + 0.3
        for actor in self.world.get_actors():
            if actor.id == self.ego.id or not actor.type_id.startswith(("vehicle.", "walker.pedestrian.")):
                continue
            if actor.get_location().distance(tf.location) > self.map_radius:
                continue
            transform = actor.get_transform()
            bbox = actor.bounding_box
            center = transform.transform(carla.Location(x=bbox.location.x, y=bbox.location.y, z=bbox.location.z))
            closest_road = min(self._reference_for_height,
                               key=lambda wp: (wp.transform.location.x - center.x) ** 2
                               + (wp.transform.location.y - center.y) ** 2)
            local_z = closest_road.transform.location.z
            if center.z + bbox.extent.z < local_z + bottom or center.z - bbox.extent.z > local_z + top:
                continue  # Separate bridge decks must not become 2D obstacles.
            velocity = actor.get_velocity()
            kind = (self.api.TrackedObjectType.PEDESTRIAN if actor.type_id.startswith("walker.")
                    else self.api.TrackedObjectType.VEHICLE)
            oriented_box = self.api.OrientedBox(self.api.StateSE2(center.x, -center.y,
                                                    -math.radians(transform.rotation.yaw + bbox.rotation.yaw)),
                                               length=max(0.1, 2 * bbox.extent.x), width=max(0.1, 2 * bbox.extent.y),
                                               height=max(0.1, 2 * bbox.extent.z))
            tracked.append(self.api.Agent(kind, oriented_box, self.api.StateVector2D(velocity.x, -velocity.y),
                                          self.api.SceneObjectMetadata(timestamp_us, str(actor.id), actor.id, str(actor.id))))
        return self.api.DetectionsTracks(self.api.TrackedObjects(tracked))

    def _traffic_lights(self, lane_dict, timestamp_us, ego_state):
        """Use stop waypoints, not traffic-light pole positions or a route-wide wall."""
        from shapely.geometry import Polygon
        lights, stop_polygons = [], []
        center = ego_state.center
        for actor in self.world.get_actors():
            if not actor.type_id.startswith("traffic.traffic_light"):
                continue
            state = str(actor.state).split(".")[-1]
            if state not in ("Red", "Yellow"):
                continue
            for index, wp in enumerate(actor.get_stop_waypoints()):
                if _lane_key(wp) not in lane_dict:
                    continue
                point, yaw = _xy(wp), _heading(wp)
                forward = np.array([math.cos(yaw), math.sin(yaw)])
                left = np.array([-math.sin(yaw), math.cos(yaw)])
                relative = np.array([center.x, center.y]) - point
                # A stop line behind the vehicle cannot cause an in-junction stop.
                if float(np.dot(relative, forward)) > 0 or np.linalg.norm(relative) > self.map_radius:
                    continue
                width = float(wp.lane_width) / 2 + 0.15
                polygon = Polygon([point - 0.1 * forward + width * left,
                                   point + 0.5 * forward + width * left,
                                   point + 0.5 * forward - width * left,
                                   point - 0.1 * forward - width * left])
                key = "stop_%s_%s" % (actor.id, index)
                lane_dict[key] = SimpleNamespace(polygon=polygon)
                lights.append(self.api.TrafficLightStatusData(self.api.TrafficLightStatusType.RED, key, timestamp_us))
                stop_polygons.append(polygon)
        return lights, stop_polygons

    def _policies(self, speed, limit):
        # Reject corrupted/sub-walking road limits instead of either exceeding
        # them with a target floor or creating millions of integration steps.
        if not math.isfinite(limit) or limit < 0.1:
            raise ValueError("PDM requires a finite speed limit of at least 0.1 m/s")
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("PDM policy initial speed must be finite and nonnegative")
        # Every run, including semantic-off, has identical stop and slow-yield
        # policies. STOP avoids IDM's undefined desired-velocity=0 division.
        labels = ["STOP", "YIELD", "DECELERATE", "FOLLOW", "MAINTAIN", "ACCELERATE", "STOP_HARD"]
        # DECELERATE is not a creep command: once nearly stopped, hold/stop.
        decel_to_stop = speed < 0.6
        targets = [0.1, min(2.0, limit), 0.1 if decel_to_stop else min(limit, speed * 0.5),
                   limit * 0.75, limit * 0.85, limit, 0.1]
        base = self.api.BatchIDMPolicy

        class WithStopPolicy(base):
            def propagate(self, previous_idm_states, leading_agent_states, longitudinal_idcs, sampling_time):
                indices = np.asarray(longitudinal_idcs)
                stop_mask = (indices == 0) | (indices == 6) | ((indices == 2) & decel_to_stop)
                active_targets = np.asarray(targets)[indices[~stop_mask]]
                # Exponent-10 IDM is unstable at low targets under a 0.1s Euler
                # step. Integrate the unchanged upstream derivative in smaller
                # steps, before proposal simulation/scoring, not after selection.
                max_step = min(0.01, float(np.min(active_targets)) / 40) if len(active_targets) else 0.01
                steps = max(1, int(math.ceil(sampling_time / max_step)))
                dt = sampling_time / steps
                result = np.array(previous_idm_states, dtype=float, copy=True)
                for _ in range(steps):
                    old = result.copy()
                    result = super().propagate(old, leading_agent_states, longitudinal_idcs, dt)
                    result[:, 1] = np.maximum(0, result[:, 1])
                    deceleration = np.where(indices == 6, 4.0, np.where(indices == 2, 1.5, 2.0))
                    result[stop_mask, 1] = np.maximum(0, old[stop_mask, 1] - deceleration[stop_mask] * dt)
                    result[:, 0] = old[:, 0] + (np.maximum(0, old[:, 1]) + result[:, 1]) * 0.5 * dt
                return result

        policies = WithStopPolicy(fallback_target_velocity=targets, speed_limit_fraction=[1.0] * len(labels),
                                  min_gap_to_lead_agent=1.5, headway_time=[1.5, 2.5, 2.0, 2.5, 1.5, 1.5, 1.5],
                                  accel_max=2.0, decel_max=4.0)
        policies.update(None)  # Desired velocities are already capped to the road speed limit.
        return policies, labels, targets

    def plan(self, semantic_decision, semantic_enabled=True):
        from shapely.geometry import Point
        timestamp = int(round(self.world.get_snapshot().timestamp.elapsed_seconds * 1e6))
        # Freeze the prior reference for repeatable on/off evaluation of the
        # same snapshot; never use the other arm's just-selected trajectory.
        if self._continuity_timestamp != timestamp:
            self._continuity_reference = self._previous_plan
            self._continuity_timestamp = timestamp
        state, tf = self._ego_state(timestamp)
        self._geometry_diagnostics = {"ego_location": {"x": tf.location.x, "y": tf.location.y,
                                                       "z": tf.location.z}}
        try:
            paths, lateral, centerline, area, lanes, excluded = self._road_geometry(state)
        except Exception as exc:
            raise RoadGeometryError(str(exc), {
                "backend": "pdm_closed", "failure_reason": "road_geometry_error",
                "timestamp_us": timestamp, "geometry": self._geometry_diagnostics,
            }) from exc
        lights, stop_polygons = self._traffic_lights(lanes, timestamp, state)
        # Reset per snapshot so an initial overlap cannot silently make an actor
        # permanently disappear from all future forecasts.
        observation = self.api.PDMObservation(self.sampling, self.sampling, self.map_radius)
        actors = self._observed_actors(timestamp, tf)
        observation.update(state, actors, lights, lanes)
        initial_collisions = [token for token in observation.collided_track_ids
                              if observation.red_light_token not in token]
        road_limit = float(self.ego.get_speed_limit()) / 3.6
        if not math.isfinite(road_limit) or road_limit < 0.1:
            raise ValueError("CARLA returned a non-finite speed limit or a limit below the supported 0.1 m/s minimum")
        limit = min(self.max_speed, road_limit)
        policies, longitudinal, targets = self._policies(state.dynamic_car_state.speed, limit)
        manager = self.api.PDMProposalManager(paths, policies)
        proposals = self.generator.generate_proposals(state, observation, manager)
        simulated = self.simulator.simulate_proposals(proposals, state)
        scores = self.scorer.score_proposals(simulated, state, observation, centerline, lanes, area,
                                             _IntersectionMap(self.map, tf.location.z, self.api))
        costs, continuity = trajectory_continuity_costs(
            proposals, self.generator._state_idm_array[:, :, 1], timestamp_us=timestamp,
            sample_interval_s=self.dt, previous=self._continuity_reference, weight=self.continuity_weight)
        candidates = []
        for index in range(len(manager)):
            proposal = manager[index]
            lat = lateral[proposal.lateral_idx]
            idm_states = self.generator._state_idm_array[index]
            profile = idm_states[:, 1]
            finite = bool(np.isfinite(proposals[index]).all() and np.isfinite(simulated[index]).all()
                          and np.isfinite(idm_states).all() and np.isfinite(scores[index]))
            # Upstream scorer intentionally ignores red-light tokens. Add the
            # same hard check in both conditions, including lane-change paths.
            traffic_safe = all(not polygon.intersects(stop) for polygon in self.scorer._ego_polygons[index]
                               for stop in stop_polygons)
            multi, weighted = self.scorer._multi_metrics, self.scorer._weighted_metrics
            safety = {"no_collision": float(multi[self.api.MultiMetricIndex.NO_COLLISION, index]),
                      "drivable_area": float(multi[self.api.MultiMetricIndex.DRIVABLE_AREA, index]),
                      "driving_direction": float(multi[self.api.MultiMetricIndex.DRIVING_DIRECTION, index]),
                      "ttc": float(weighted[self.api.WeightedMetricIndex.TTC, index]),
                      "traffic_light": traffic_safe, "finite": finite}
            if initial_collisions:
                safety["no_collision"] = 0.0
            offroad_steps = np.flatnonzero(self.scorer._ego_areas[index, :, self.api.EgoAreaIndex.NON_DRIVABLE_AREA])
            first_offroad = int(offroad_steps[0]) if len(offroad_steps) else None
            area_failure = None
            if first_offroad is not None and finite:
                area_failure = {"time_s": first_offroad * self.dt,
                                "footprint_corners_xy": self.scorer._ego_coords[index, first_offroad, :4].tolist()}
            collision_time = float(self.scorer.time_to_at_fault_collision(index))
            ttc_time = float(self.scorer.time_to_ttc_infraction(index))
            candidates.append({"id": "%s:%s" % (lat["id"], longitudinal[proposal.longitudinal_idx]),
                               "index": index, "lateral": lat["label"], "lateral_semantics": lat["semantics"],
                               "route_target": lat["id"] == "center",
                               "route_recovery_required": lat["route_recovery_required"],
                               "forward_progress_m": float(centerline.project(Point(*proposals[index, -1, :2]))
                                                           - centerline.project(Point(*proposals[index, 0, :2]))) if finite else 0.0,
                               "terminal_speed_mps": float(self.generator._state_idm_array[index, -1, 1]) if finite else 0.0,
                               "longitudinal": "STOP" if longitudinal[proposal.longitudinal_idx] == "STOP_HARD" else longitudinal[proposal.longitudinal_idx],
                               "continuity_cost": float(costs[index]),
                               "predicted_collision_time_s": collision_time if math.isfinite(collision_time) else None,
                               "predicted_ttc_infraction_time_s": ttc_time if math.isfinite(ttc_time) else None,
                               "drivable_area_failure": area_failure,
                               "speed_at_0_2s_mps": float(profile[2]) if finite else None,
                               "acceleration_range_mps2": [float(np.min(np.diff(profile) / self.dt)),
                                                            float(np.max(np.diff(profile) / self.dt))] if finite else None,
                               "desired_speed_mps": targets[proposal.longitudinal_idx],
                               "base_score": float(scores[index]) if finite else 0.0, "safety": safety,
                               "trajectory_sha256": hashlib.sha256(proposals[index].tobytes()).hexdigest()})
        selected, diagnostics = choose_candidate(candidates, semantic_decision, semantic_enabled,
                                                  self.lateral_weight, self.longitudinal_weight)
        diagnostics.update(backend="pdm_closed", provenance=self.provenance,
                           timestamp_us=timestamp, horizon_s=self.samples * self.dt, sample_interval_s=self.dt,
                           leading_agent_update_rate=1, lateral_transition="cubic_hermite_pose_tangent",
                           excluded_lateral_paths=excluded, traffic_stop_lines=len(stop_polygons),
                           geometry=self._geometry_diagnostics,
                           continuity=continuity, idm_integration="substeps_max_0.01s_nonnegative_trapezoid",
                           actors=[{"id": str(actor.track_token), "x": actor.center.x, "y": actor.center.y,
                                    "heading": actor.center.heading, "vx": actor.velocity.x, "vy": actor.velocity.y,
                                    "length": actor.box.length, "width": actor.box.width}
                                   for actor in actors.tracked_objects.tracked_objects],
                           initial_collision_tokens=initial_collisions,
                           vehicle={"wheelbase_m": self.vehicle_parameters.wheel_base,
                                    "max_steer_rad": self.max_steer,
                                    "width_m": self.vehicle_parameters.width,
                                    "length_m": self.vehicle_parameters.length,
                                    "bounding_box_offset_xyz": [
                                        self.ego.bounding_box.location.x, self.ego.bounding_box.location.y,
                                        self.ego.bounding_box.location.z],
                                    "rear_axle_to_center_m": self.vehicle_parameters.rear_axle_to_center})
        # Run the upstream detector against the selected candidate. Its brake
        # output is an emergency signal; CARLA performs a direct full brake.
        emergency_scores = np.asarray([c["final_score"] if c["eligible"] else -1.0 for c in diagnostics["candidates"]])
        emergency = self.emergency.brake_if_emergency(state, emergency_scores, self.scorer)
        if selected is None or emergency is not None:
            self._previous_plan = None
            diagnostics.update(emergency_brake=True, failure_reason="no_safe_candidate" if selected is None else "upstream_emergency")
            return {"trajectory": [], "target_speed": 0.0, "selected_lateral": "KEEP_LANE",
                    "selected_longitudinal": "STOP", "planner_diagnostics": diagnostics}
        index = selected["index"]
        generated = self.generator.generate_trajectory(index)
        states = generated.get_sampled_trajectory()
        from carla_bridge.state.coords import global_to_ego
        yaw = math.radians(tf.rotation.yaw)
        box_offset = np.array([self.ego.bounding_box.location.x, -self.ego.bounding_box.location.y])
        trajectory = []
        idm_velocity = self.generator._state_idm_array[index, :, 1]
        self._previous_plan = TrajectoryReference(states=proposals[index].copy(), speeds=idm_velocity.copy(),
                                                  timestamp_us=timestamp, sample_interval_s=self.dt)
        for i, point in enumerate(states):
            center = point.center
            c, s = math.cos(center.heading), math.sin(center.heading)
            # Convert footprint center back to CARLA actor origin, then reflect
            # world y and apply the project's ego-centric (forward,left) frame.
            origin = np.array([center.x, center.y]) - np.array([c * box_offset[0] - s * box_offset[1],
                                                                             s * box_offset[0] + c * box_offset[1]])
            x, y = global_to_ego(origin[0], -origin[1], tf.location.x, tf.location.y, yaw)
            trajectory.append({"t": round(i * self.dt, 8), "x": float(x), "y": float(y),
                               "optional_v": float(max(0, idm_velocity[i]))})
        diagnostics.update(emergency_brake=False, selected_lateral=selected["lateral"],
                           selected_longitudinal=selected["longitudinal"])
        return {"trajectory": trajectory, "target_speed": trajectory[-1]["optional_v"],
                "selected_lateral": selected["lateral"], "selected_longitudinal": selected["longitudinal"],
                "planner_diagnostics": diagnostics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pinned PDM core imports without CARLA or VLM")
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate_dependencies(), indent=2, ensure_ascii=False))
