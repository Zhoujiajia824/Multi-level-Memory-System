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


def _lane_key(wp):
    return "%s:%s:%s" % (wp.road_id, wp.section_id, wp.lane_id)


def _heading(wp):
    return -math.radians(wp.transform.rotation.yaw)


def _xy(wp):
    p = wp.transform.location
    return np.array([p.x, -p.y], dtype=float)


def _same_direction(first, second):
    return math.cos(_heading(first) - _heading(second)) > 0.8


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
        import carla
        location = self.ego.get_location()
        raw = self.route._route
        index = self.route._nearest_index(location)
        start = max(0, index - 6)
        points, travelled = [], 0.0
        span = max(55.0, self.max_speed * self.horizon + self.transition_m + 10)
        for p in raw[start:]:
            wp = self.map.get_waypoint(p, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                raise ValueError("Route is not on a CARLA driving lane")
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
        closest = min(range(len(reference)), key=lambda i: float(np.linalg.norm(_xy(reference[i]) - origin)))
        current_side = None
        for side, candidate in (("center", base_wp), ("left", base_wp.get_left_lane()),
                                 ("right", base_wp.get_right_lane())):
            if candidate is not None and _lane_key(candidate) == _lane_key(ego_wp):
                current_side = side
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
                if float(np.linalg.norm(_xy(a) - _xy(b))) > 6:
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
        occupancy = self.api.PDMOccupancyMap(list(polygons), np.array(list(polygons.values()), dtype=object))

        paths, metadata, excluded = [], [], {}
        for side in ("center", "left", "right"):
            lane_points = []
            reason = None
            for i, base in enumerate(reference):
                distance_ahead = sum(float(np.linalg.norm(_xy(reference[j + 1]) - _xy(reference[j])))
                                     for j in range(closest, max(closest, i)))
                wp = base if side == "center" else (base.get_left_lane() if side == "left" else base.get_right_lane())
                if wp is None or wp.lane_type != carla.LaneType.Driving or not _same_direction(base, wp):
                    if distance_ahead > self.max_speed * self.horizon + 8 and len(lane_points) > 3:
                        break  # The valid finite lane becomes an IDM stopping constraint.
                    reason = "no_continuous_same_direction_driving_lane"
                    break
                # A change of two lanes is never a candidate.
                if current_side != "center" and side not in (current_side, "center"):
                    reason = "would_cross_two_lanes"
                    break
                if side != current_side and i >= closest and distance_ahead <= self.transition_m + 4:
                    source = base if current_side == "center" else (base.get_left_lane() if current_side == "left" else base.get_right_lane())
                    direction = side if current_side == "center" else ("right" if current_side == "left" else "left")
                    permission = carla.LaneChange.Left if direction == "left" else carla.LaneChange.Right
                    if source is None or base.is_junction or wp.is_junction or not (int(source.lane_change) & int(permission)):
                        reason = "lane_change_not_permitted_or_intersection"
                        break
                lane_points.append(_xy(wp))
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
            paths.append(_smooth_path(lane_points, rear, ego_state.rear_axle.heading, self.api, self.transition_m))
            metadata.append({"id": side, "label": label, "semantics": sorted(set(semantics))})
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
        # Every run, including semantic-off, has identical stop and slow-yield
        # policies. STOP avoids IDM's undefined desired-velocity=0 division.
        labels = ["STOP", "YIELD", "DECELERATE", "FOLLOW", "MAINTAIN", "ACCELERATE"]
        targets = [0.1, min(2.0, limit), max(0.3, min(limit, speed * 0.5)),
                   max(0.3, limit * 0.75), max(0.3, limit * 0.85), limit]
        base = self.api.BatchIDMPolicy

        class WithStopPolicy(base):
            def propagate(self, previous_idm_states, leading_agent_states, longitudinal_idcs, sampling_time):
                result = super().propagate(previous_idm_states, leading_agent_states, longitudinal_idcs, sampling_time)
                mask = np.asarray(longitudinal_idcs) == 0
                previous_v = np.maximum(0, previous_idm_states[mask, 1])
                next_v = np.maximum(0, previous_v - 4.0 * sampling_time)
                result[mask, 0] = previous_idm_states[mask, 0] + (previous_v + next_v) * 0.5 * sampling_time
                result[mask, 1] = next_v
                return result

        policies = WithStopPolicy(fallback_target_velocity=targets, speed_limit_fraction=[1.0] * len(labels),
                                  min_gap_to_lead_agent=1.5, headway_time=[1.5, 2.5, 2.0, 2.5, 1.5, 1.5],
                                  accel_max=2.0, decel_max=4.0)
        policies.update(None)  # Desired velocities are already capped to the road speed limit.
        return policies, labels, targets

    def plan(self, semantic_decision, semantic_enabled=True):
        timestamp = int(round(self.world.get_snapshot().timestamp.elapsed_seconds * 1e6))
        state, tf = self._ego_state(timestamp)
        paths, lateral, centerline, area, lanes, excluded = self._road_geometry(state)
        lights, stop_polygons = self._traffic_lights(lanes, timestamp, state)
        # Reset per snapshot so an initial overlap cannot silently make an actor
        # permanently disappear from all future forecasts.
        observation = self.api.PDMObservation(self.sampling, self.sampling, self.map_radius)
        observation.update(state, self._observed_actors(timestamp, tf), lights, lanes)
        initial_collisions = [token for token in observation.collided_track_ids
                              if observation.red_light_token not in token]
        limit = min(self.max_speed, float(self.ego.get_speed_limit()) / 3.6)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("CARLA returned an invalid speed limit")
        policies, longitudinal, targets = self._policies(state.dynamic_car_state.speed, limit)
        manager = self.api.PDMProposalManager(paths, policies)
        proposals = self.generator.generate_proposals(state, observation, manager)
        simulated = self.simulator.simulate_proposals(proposals, state)
        scores = self.scorer.score_proposals(simulated, state, observation, centerline, lanes, area,
                                             _IntersectionMap(self.map, tf.location.z, self.api))
        candidates = []
        for index in range(len(manager)):
            proposal = manager[index]
            lat = lateral[proposal.lateral_idx]
            finite = bool(np.isfinite(proposals[index]).all() and np.isfinite(simulated[index]).all()
                          and np.isfinite(scores[index]))
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
            candidates.append({"id": "%s:%s" % (lat["id"], longitudinal[proposal.longitudinal_idx]),
                               "index": index, "lateral": lat["label"], "lateral_semantics": lat["semantics"],
                               "longitudinal": longitudinal[proposal.longitudinal_idx],
                               "desired_speed_mps": targets[proposal.longitudinal_idx],
                               "base_score": float(scores[index]) if finite else 0.0, "safety": safety,
                               "trajectory_sha256": hashlib.sha256(proposals[index].tobytes()).hexdigest()})
        selected, diagnostics = choose_candidate(candidates, semantic_decision, semantic_enabled,
                                                  self.lateral_weight, self.longitudinal_weight)
        diagnostics.update(backend="pdm_closed", provenance=self.provenance,
                           timestamp_us=timestamp, horizon_s=self.samples * self.dt, sample_interval_s=self.dt,
                           leading_agent_update_rate=1, lateral_transition="cubic_hermite_pose_tangent",
                           excluded_lateral_paths=excluded, traffic_stop_lines=len(stop_polygons),
                           initial_collision_tokens=initial_collisions,
                           vehicle={"wheelbase_m": self.vehicle_parameters.wheel_base,
                                    "rear_axle_to_center_m": self.vehicle_parameters.rear_axle_to_center})
        # Run the upstream detector against the selected candidate. Its brake
        # output is an emergency signal; CARLA performs a direct full brake.
        emergency_scores = np.asarray([c["final_score"] if c["safe"] else -1.0 for c in diagnostics["candidates"]])
        emergency = self.emergency.brake_if_emergency(state, emergency_scores, self.scorer)
        if selected is None or emergency is not None:
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
