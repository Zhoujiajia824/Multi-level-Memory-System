"""CARLA map/actor adapter for the independent constrained nonlinear MPC.

The optimizer uses measured rear-axle poses in CARLA world coordinates. Only
after optimization are all poses rigidly converted to the captured actor frame
(forward/left), as required by the controller, validator and replay logs. No
PDM, nuPlan or tuplan object is imported by this backend.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, replace
import hashlib
import math
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from .lane_maneuver import LaneManeuver
from .mpc_solver import (MPCConfig, MPCInitialState, MPCObstacle, MPCProblem,
                         MPCReference, MPCSolver, bicycle_step)
from .scene_context import (build_scene_context, candidate_summary, lane_key,
                            planner_health, project_polyline)
from .selection import choose_candidate


def validate_dependencies():
    """Check the independent numerical runtime; no CARLA or PDM initialization."""
    import scipy
    try:
        import shapely
        from shapely.geometry import Point, Polygon
        first = Point(0., 0.).buffer(1.)
        second = Polygon([(0., -1.), (2., -1.), (2., 1.), (0., 1.)])
        if first.intersection(second).area <= 0 or Point(3., 0.).distance(first) <= 0:
            raise ValueError("geometry operation returned an invalid result")
    except Exception as exc:
        raise RuntimeError("MPC geometry dependency Shapely/GEOS is unavailable or unusable; "
                           "install carla_bridge/planning/requirements-mpc.txt. "
                           "Original error: " + str(exc)) from exc
    return {"backend": "mpc", "optimizer": "scipy.optimize.SLSQP",
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "shapely_version": shapely.__version__,
            "geos_version": str(getattr(shapely, "geos_version_string", "unknown")),
            "geometry_operation_probe": "buffer_intersection_distance_passed",
            "upstream_pdm_used": False, "dependency_check": "imports_and_config_validated",
            "default_numerical_config": asdict(MPCConfig())}


def effective_mpc_config(planning_config=None, **vehicle_overrides):
    """Resolve numerical defaults without a CARLA connection or vehicle actor."""
    planning = dict(planning_config or {})
    values = dict(planning.get("mpc") or {})
    known = {item.name for item in fields(MPCConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError("Unknown MPC settings: " + ", ".join(sorted(unknown)))
    values.setdefault("dt", .2)
    if (isinstance(values["dt"], bool) or not isinstance(values["dt"], (int, float))
            or not math.isfinite(values["dt"]) or not 0 < values["dt"] <= .5):
        raise ValueError("CARLA MPC dt must be positive, finite and no greater than 0.5s "
                         "for executable optimized-control profiles")
    horizon = float(planning.get("horizon_s", 4.0))
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("MPC horizon_s must be positive and finite")
    values.setdefault("horizon_steps", int(math.ceil(horizon / float(values["dt"]))))
    values.update(vehicle_overrides)
    return MPCConfig(**values)


def _points(reference):
    return np.array([[wp.transform.location.x, wp.transform.location.y] for wp in reference], dtype=float)


def _curve(points, headings=None):
    points = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("MPC reference needs two distinct connected map points")
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if headings is None:
        directions = np.gradient(points, stations, axis=0)
        directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-8)
    else:
        headings = np.asarray(headings, dtype=float)[keep]
        directions = np.column_stack((np.cos(headings), np.sin(headings)))
    return points, stations, CubicHermiteSpline(stations, points, directions, axis=0)


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class MPCPlanningError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class CarlaMPCPlanner:
    """Optimize each legal lane intent, then apply optional semantic preference."""

    def __init__(self, world, ego, route, config=None):
        self.world, self.ego, self.route, self.map = world, ego, route, world.get_map()
        self.config = dict(config or {})
        self.max_speed = float(self.config.get("max_speed_mps", 8.0))
        if not math.isfinite(self.max_speed) or not .1 <= self.max_speed <= 20:
            raise ValueError("MPC max_speed_mps must be within [.1, 20]")
        self.semantic_policy = self.config.get("semantic_policy", "selective")
        if self.semantic_policy not in ("always", "selective", "shadow"):
            raise ValueError("semantic_policy must be always, selective or shadow")
        self.planning_memory_mode = self.config.get("planning_memory", "off")
        if self.planning_memory_mode not in ("off", "gate", "full"):
            raise ValueError("planning_memory must be off, gate or full")
        self.lateral_weight = float(self.config.get("lateral_semantic_weight", .8))
        self.longitudinal_weight = float(self.config.get("longitudinal_semantic_weight", .8))
        self._vehicle_model()
        numerical = effective_mpc_config(self.config, wheelbase=self.vehicle_parameters.wheel_base,
            rear_axle_to_center=self.rear_axle_to_center,
            ego_length=self.vehicle_parameters.length, ego_width=self.vehicle_parameters.width,
            steer_max=min(self.max_steer, float((self.config.get("mpc") or {}).get("steer_max", self.max_steer))))
        self.numerical_config, self.solver = numerical, MPCSolver(numerical)
        self._configured_steer_max = numerical.steer_max
        self.dt, self.samples = numerical.dt, numerical.horizon_steps
        self.horizon = self.dt * self.samples
        self.lane_maneuver = LaneManeuver(self.config.get("lane_maneuver"))
        from .recovery import validate_recovery_config
        self.recovery_config = validate_recovery_config(self.config.get("recovery"))
        from .semantic_governor import SelectiveSemanticGovernor
        self.governor = (SelectiveSemanticGovernor(self.config.get("semantic_governor"))
                         if self.semantic_policy != "always" else None)
        self.experience_memory = None
        self.last_context, self.last_candidates, self.last_query_decision = None, [], None
        self.last_visualization_snapshot = None
        self._executed_desired_acceleration = {"value_mps2": 0., "sim_time_s": None,
            "source": "no_previous_control_dispatch", "physical_acceleration_measurement": False}
        self._pending_desired_acceleration = None
        self._trajectory_visualization_config = {"enabled": False}
        self._route_source, self._resolved_route_waypoints, self._route_resolution = None, None, None
        self.provenance = {"backend": "mpc", "optimizer": "scipy.optimize.SLSQP",
            "dynamics": "rear_axle_kinematic_bicycle_midpoint",
            "constraints": "speed_acceleration_steering_rate_jerk_lateral_corridor_predicted_footprint_stop_line",
            "prediction_kind": "constant_velocity_proxy_not_future_GT",
            "upstream_pdm_used": False, "numerical_config": asdict(numerical)}

    def _vehicle_model(self):
        """CARLA wheel positions are world centimetres, not local metres."""
        from carla_bridge.control.steering_model import SteeringModel
        box, tf, physics = self.ego.bounding_box, self.ego.get_transform(), self.ego.get_physics_control()
        wheels = physics.wheels
        if len(wheels) < 4:
            raise ValueError("MPC requires a four-wheel vehicle")
        inverse = np.asarray(tf.get_inverse_matrix(), dtype=float)
        local = [inverse @ np.array([w.position.x*.01, w.position.y*.01, w.position.z*.01, 1.])
                 for w in wheels[:4]]
        front, rear = np.mean(local[:2], axis=0), np.mean(local[2:], axis=0)
        wheelbase = float(np.linalg.norm(front[:3] - rear[:3]))
        track = abs(float(local[0][1] - local[1][1]))
        self.rear_axle_actor_xyz_m = tuple(float(value) for value in rear[:3])
        self.rear_axle_to_center = float(box.location.x - rear[0])
        if not 0 < self.rear_axle_to_center < 2*float(box.extent.x):
            raise ValueError("CARLA physical rear axle lies outside the supported body geometry")
        self.physical_max_steer = math.radians(max(float(w.max_steer_angle) for w in wheels[:2]))
        self.steering_model = SteeringModel(self.physical_max_steer,
            [(point.x, point.y) for point in getattr(physics, "steering_curve", ())],
            wheelbase_m=wheelbase, front_track_m=track)
        self.max_steer, self.front_track_m = self.steering_model.max_steer_rad, track
        self.vehicle_parameters = SimpleNamespace(wheel_base=wheelbase,
            width=2*float(box.extent.y), length=2*float(box.extent.x),
            rear_axle_to_center=self.rear_axle_to_center)

    def configure_trajectory_visualization(self, config=None):
        from carla_bridge.visualization.trajectory_renderer import validate_trajectory_visualization_config
        self._trajectory_visualization_config = validate_trajectory_visualization_config(config)
        self.last_visualization_snapshot = None

    def execution_vehicle_signature(self):
        """Return the stable planar execution model used to partition memory.

        World wheel coordinates carry floating-point transform noise, and
        rear-wheel height varies with suspension/terrain. This signature uses
        the actual two-dimensional dimensions at an explicit 1 mm resolution;
        it neither hashes spawn/yaw/height nor discards meaningful size changes.
        Physical steering limits and the CARLA km/h steering curve retain six
        decimal places. Only already measured model values are read here.
        """
        def quantized(value, digits):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Execution vehicle signature requires finite measured model values")
            rounded = round(value, digits)
            return 0. if rounded == 0. else rounded  # Canonicalize negative zero.

        params = self.vehicle_parameters
        return {"signature_model": "carla_rear_axle_planar_vehicle_v1",
            "blueprint": str(self.ego.type_id), "measurement_resolution_m": .001,
            "wheelbase_m": quantized(params.wheel_base, 3),
            "length_m": quantized(params.length, 3), "width_m": quantized(params.width, 3),
            "front_track_m": quantized(self.front_track_m, 3),
            "rear_axle_to_center_m": quantized(self.rear_axle_to_center, 3),
            "rear_axle_actor_xy_m": [quantized(v, 3) for v in self.rear_axle_actor_xyz_m[:2]],
            "physical_inner_max_steer_rad": quantized(self.physical_max_steer, 6),
            "steering_curve_kmh_scale": [[quantized(speed, 6), quantized(scale, 6)]
                                         for speed, scale in self.steering_model.curve],
            "steering_mapping": self.steering_model.mapping}

    def _bind_dynamic_steering_limit(self, speed, target_limit):
        """Use a conservative bound valid throughout the predicted speed range.

        CARLA's curve scales the physical inner wheel before Ackermann
        conversion. The optimizer and actual-control guard must share it.
        """
        upper = max(float(speed), float(target_limit))
        probes = [0., upper]
        probes.extend(kmh/3.6 for kmh, _ in self.steering_model.curve if 0 <= kmh/3.6 <= upper)
        limit = min(self._configured_steer_max, *(self.steering_model.limit_at(v) for v in probes))
        if not math.isclose(limit, self.numerical_config.steer_max, abs_tol=1e-10):
            self.numerical_config = replace(self.numerical_config, steer_max=limit)
            self.solver = MPCSolver(self.numerical_config)
        self._dynamic_steer_diagnostics = {"method": "minimum_over_predicted_speed_range",
            "speed_range_mps": [0., upper], "effective_bicycle_limit_rad": limit,
            "configured_bicycle_limit_rad": self._configured_steer_max}

    def record_control_execution(self, diagnostics, mpc_tracking_audit, *, now_s, source,
                                 tracker_diagnostics=None):
        """Receive feedback only after the loop actually dispatched a control.

        Jerk limits relate consecutive desired acceleration inputs. A CARLA
        contact impulse or a stationary brake's measured deceleration is not
        that previous input. Dispatch is pending until a later physical time:
        another optimization in the same frozen frame must not consume it as
        if an extra time step had elapsed. Repeated dispatches replace pending.
        The optimizer never writes either desired-control cache.
        """
        self._promote_dispatched_input(float(now_s))
        value, reason, sample_index = 0., "interrupted_control_dispatch", None
        audit = mpc_tracking_audit or {}
        tracker = tracker_diagnostics or {}
        if source == "direct_brake":
            reason = "direct_brake_reset_desired_input"
        elif tracker.get("reason") == "stationary_hold":
            reason = "stationary_hold_zero_desired_acceleration"
        elif source == "tracker" and audit.get("applied") is True:
            profile = (diagnostics or {}).get("mpc_tracking") or {}
            index = audit.get("profile_sample_index")
            values = profile.get("accelerations_mps2")
            if (type(index) is int and isinstance(values, (tuple, list))
                    and 0 <= index < len(values) and type(values[index]) in (int, float)
                    and math.isfinite(values[index])):
                value = float(np.clip(values[index], self.numerical_config.accel_min,
                                      self.numerical_config.accel_max))
                reason, sample_index = "dispatched_optimized_desired_acceleration", index
            else:
                reason = "invalid_dispatched_profile_reset_desired_input"
        elif source == "tracker":
            reason = "mpc_control_override_rejected_reset_desired_input"
        self._pending_desired_acceleration = {"value_mps2": value, "sim_time_s": float(now_s),
            "source": reason, "profile_sample_index": sample_index,
            "physical_acceleration_measurement": False,
            "longitudinal_actuation": "desired_speed_profile_tracked_by_PID_not_exact_acceleration"}

    def _promote_dispatched_input(self, now_s):
        """Only a physical clock advance makes the last dispatch previous."""
        pending = self._pending_desired_acceleration
        if pending is not None and now_s > pending["sim_time_s"]+1e-8:
            self._executed_desired_acceleration = pending
            self._pending_desired_acceleration = None

    def _initial_state(self, now_s=None):
        import carla
        tf = self.ego.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        rear = tf.transform(carla.Location(*self.rear_axle_actor_xyz_m))
        velocity, acceleration = self.ego.get_velocity(), self.ego.get_acceleration()
        forward = velocity.x*math.cos(yaw) + velocity.y*math.sin(yaw)
        if forward < -.2:
            raise ValueError("Reverse driving is not supported by this forward MPC")
        speed = max(0., float(forward))
        observed = "command_ackermann_estimate"
        steer = self.steering_model.normalized_to_bicycle(float(self.ego.get_control().steer), speed)
        getter = getattr(self.ego, "get_wheel_steer_angle", None)
        locations = getattr(carla, "VehicleWheelLocation", None)
        if callable(getter) and hasattr(locations, "FL_Wheel") and hasattr(locations, "FR_Wheel"):
            steer = self.steering_model.front_wheels_to_bicycle(
                math.radians(getter(locations.FL_Wheel)), math.radians(getter(locations.FR_Wheel)))
            observed = "carla_measured_front_wheels"
        accel = float(acceleration.x*math.cos(yaw) + acceleration.y*math.sin(yaw))
        if now_s is not None:
            self._promote_dispatched_input(now_s)
        previous = deepcopy(self._executed_desired_acceleration)
        if (previous["sim_time_s"] is None or now_s is None
                or now_s < previous["sim_time_s"]-1e-8
                or now_s-previous["sim_time_s"] > max(.5, 3*self.dt)):
            previous.update(value_mps2=0., source="no_recent_dispatched_desired_input")
        requested_accel = previous["value_mps2"]
        self._steering_state_diagnostics = {"source": observed, "bicycle_angle_rad": steer,
            "measured_longitudinal_acceleration_mps2": accel,
            "optimization_initial_acceleration_mps2": requested_accel,
            "previous_desired_acceleration": previous,
            "pending_desired_acceleration": deepcopy(self._pending_desired_acceleration),
            "input_history_clock": "strictly_previous_simulation_tick_not_same_frame_dispatch",
            "input_latency_modeled": False}
        return MPCInitialState(rear.x, rear.y, yaw, speed, steer, requested_accel), tf

    def _resolve_reference(self):
        raw = self.route._route
        if self._route_source is not raw or self._resolved_route_waypoints is None:
            retained = getattr(self.route, "_route_waypoints", None)
            if retained is not None and len(retained) == len(raw):
                self._resolved_route_waypoints = list(retained)
                self._route_resolution = getattr(self.route, "_waypoint_resolution", None)
            else:
                from .route_waypoints import resolve_route_waypoints
                self._resolved_route_waypoints, self._route_resolution = resolve_route_waypoints(self.map, raw)
            self._route_source = raw
            self._reference_topology_corrections = []
            # OpenDRIVE may expose the same junction-exit position twice:
            # retained route topology still belongs to the preceding junction,
            # while the current map projection is the connected ordinary lane.
            # Correct only this coincident initial endpoint, never an interior
            # intersection or any evaluation checkpoint/coordinate.
            if len(self._resolved_route_waypoints) >= 2:
                first, second = self._resolved_route_waypoints[:2]
                current = self.map.get_waypoint(self.ego.get_location())
                if current is not None and first.is_junction and not current.is_junction:
                    p, q, nxt = first.transform, current.transform, second.transform
                    distance = math.hypot(nxt.location.x-p.location.x, nxt.location.y-p.location.y)
                    coincident = (math.hypot(p.location.x-q.location.x, p.location.y-q.location.y) < .05
                        and abs(p.location.z-q.location.z) < .1
                        and abs(_angle(math.radians(p.rotation.yaw-q.rotation.yaw))) < math.radians(1.))
                    connected = (coincident and .1 < distance < 5.
                        and lane_key(current) == lane_key(second)
                        and any(lane_key(candidate) == lane_key(second)
                            and math.hypot(candidate.transform.location.x-nxt.location.x,
                                           candidate.transform.location.y-nxt.location.y) < .15
                            and abs(candidate.transform.location.z-nxt.location.z) < .1
                            and abs(_angle(math.radians(candidate.transform.rotation.yaw-nxt.rotation.yaw)))
                                < math.radians(3.) for candidate in current.next(distance)))
                    if connected:
                        self._resolved_route_waypoints[0] = current
                        self._reference_topology_corrections.append({"index": 0,
                            "from_lane": lane_key(first), "to_lane": lane_key(current),
                            "reason": "coincident_junction_exit_connected_to_next_reference",
                            "evaluation_route_modified": False})
        full = self._resolved_route_waypoints
        index = min(len(full)-1, max(0, int(self.route._nearest_index(self.ego.get_location()))))
        local = list(full[max(0, index-6):min(len(full), index+61)])
        if len(local) < 2:
            local = list(full[-2:])
        # Evaluation endpoints never become physical stopped leaders. Continue
        # actual successors only; ordered evaluation checkpoints remain unchanged.
        length = sum(np.linalg.norm(_points([a, b])[1]-_points([a, b])[0]) for a, b in zip(local, local[1:]))
        span = max(65., self.max_speed*self.horizon+20.)
        while local and length < span:
            options = local[-1].next(2.)
            if not options:
                break
            previous = local[-1]
            chosen = min(options, key=lambda wp: abs(_angle(math.radians(
                wp.transform.rotation.yaw-previous.transform.rotation.yaw))))
            distance = float(np.linalg.norm(_points([previous, chosen])[1]-_points([previous, chosen])[0]))
            if distance < .1:
                break
            local.append(chosen)
            length += distance
        if len(local) < 2:
            raise ValueError("MPC has no connected local reference")
        self._reference_for_height = local
        return local, index

    def _side(self, reference):
        location = self.ego.get_location()
        current = self.map.get_waypoint(location)
        closest_index = min(range(len(reference)), key=lambda i:
            (reference[i].transform.location.x-location.x)**2+(reference[i].transform.location.y-location.y)**2)
        if current is not None:
            # The closest knot alone can lie across a genuine road/section
            # seam. Match the bracketing connected route identities as well.
            nearby = reference[max(0, closest_index-3):min(len(reference), closest_index+4)]
            for side in ("center", "left", "right"):
                for base in nearby:
                    value = (base if side == "center" else base.get_left_lane()
                             if side == "left" else base.get_right_lane())
                    if value is not None and lane_key(value) == lane_key(current):
                        return side
        return "center" if getattr(current, "is_junction", False) else "unknown"

    def _obstacles(self, initial, tf):
        predictions, audit = [], []
        times = np.arange(self.samples+1)*self.dt
        for actor in self.world.get_actors():
            if actor.id == self.ego.id or not actor.type_id.startswith(("vehicle.", "walker.pedestrian", "static.prop.")):
                continue
            pose = actor.get_transform()
            p = pose.location
            if abs(p.z-tf.location.z) > 2.5 or math.hypot(p.x-tf.location.x, p.y-tf.location.y) > 70:
                continue
            velocity, box = actor.get_velocity(), actor.bounding_box
            yaw = math.radians(pose.rotation.yaw)
            offset = getattr(box, "location", SimpleNamespace(x=0., y=0.))
            center = np.array([p.x+offset.x*math.cos(yaw)-offset.y*math.sin(yaw),
                               p.y+offset.x*math.sin(yaw)+offset.y*math.cos(yaw)])
            centers = center + times[:, None]*np.array([velocity.x, velocity.y])
            body_yaw = yaw + math.radians(getattr(getattr(box, "rotation", None), "yaw", 0.))
            # Pedestrian circles use the exact ego rectangle/circle solver
            # constraint; a tall longitudinal vehicle envelope is unnecessary.
            if actor.type_id.startswith("walker."):
                radii = max(.1, math.hypot(box.extent.x, box.extent.y))
            else:
                # Circumscribed ellipse contains the CARLA rectangular body.
                radii = np.tile(np.sqrt(2.)*np.array([max(.1, box.extent.x), max(.1, box.extent.y)]), (self.samples+1, 1))
            predictions.append(MPCObstacle(str(actor.id), centers, radii, body_yaw))
            audit.append({"id": int(actor.id), "type": actor.type_id, "x": float(center[0]),
                "y": float(center[1]), "heading": body_yaw, "vx": float(velocity.x),
                "vy": float(velocity.y), "length": 2*float(box.extent.x), "width": 2*float(box.extent.y)})
        return predictions, audit

    def _reference(self, identity, points, initial, desired, width, metadata, *, headings=None, stop_line=None):
        raw_points = np.asarray(points, dtype=float)
        kept = np.r_[True, np.linalg.norm(np.diff(raw_points, axis=0), axis=1) > 1e-5]
        widths = np.asarray(width, dtype=float)
        if widths.ndim == 0:
            widths = np.full(len(raw_points), float(widths))
        if widths.shape != (len(raw_points),) or not np.isfinite(widths).all():
            raise ValueError("MPC lane widths must match the connected map samples")
        points, stations, spline = _curve(points, headings)
        projection = project_polyline(initial.x, initial.y, points)
        station = projection[0]
        stop_station = None
        if stop_line is not None:
            stop_station = project_polyline(*stop_line["point"], points)[0]
        speeds, targets, travelled = [initial.speed], [station], station
        for _ in range(self.samples):
            previous = speeds[-1]
            step = np.clip(desired-previous, self.numerical_config.accel_min*self.dt,
                           self.numerical_config.accel_max*self.dt)
            speed = max(0., previous+step)
            travelled += .5*(previous+speed)*self.dt
            target_station = min(float(stations[-1]), travelled)
            if stop_station is not None:
                target_station = min(target_station, stop_station)
            targets.append(target_station)
            speeds.append(speed)
        xy = spline(targets)
        tangents = spline(targets, 1)
        yaw = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
        speed_profile = np.minimum(speeds, desired)
        if stop_line is not None:
            normal = np.asarray(stop_line["normal"])
            remaining = np.maximum(0., (np.asarray(stop_line["point"])-xy) @ normal)
            speed_profile = np.minimum(speed_profile, np.sqrt(2*abs(self.numerical_config.accel_min)*remaining))
        refmeta = dict(metadata)
        if stop_line is not None:
            refmeta["stop_line"] = stop_line
        road_width = np.interp(targets, stations, widths[kept])
        half_width = np.maximum(.05, road_width/2-self.vehicle_parameters.width/2-.12)
        return MPCReference(identity, np.column_stack((xy, yaw, speed_profile)), half_width,
                            speed_limit=self._speed_limit, metadata=refmeta)

    def _stop_line(self, local, initial, context):
        applicable = [s for s in context["signals"] if s["applicable"]
            and s["state"].upper() in ("RED", "YELLOW") and s["stop_line_distance_m"] >= -.5]
        if not applicable:
            return None
        signal = min(applicable, key=lambda s: s["stop_line_distance_m"])
        pts, stations, spline = _curve(_points(local), np.radians([wp.transform.rotation.yaw for wp in local]))
        actor = self.ego.get_location()
        actor_station = project_polyline(actor.x, actor.y, pts)[0]
        # Context distance is to front bumper. Convert the line target to a
        # rear-axle limit, including measured bbox offset and a half-metre gap.
        rear_station = actor_station+signal["stop_line_distance_m"]-self.rear_axle_to_center-.5
        if rear_station > stations[-1]:
            return None  # A finite planning reference is not a fictitious nearer stop line.
        rear_station = max(0., rear_station)
        tangent = spline(rear_station, 1)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
        return {"point": spline(rear_station).tolist(), "normal": tangent.tolist(), "id": signal["id"]}

    def _candidate(self, reference, result, index, initial, limit):
        states, metadata = result.states, reference.metadata
        finite = bool(np.isfinite(states).all() and np.isfinite(result.controls).all())
        usable = finite and bool(result.feasible)
        distance = float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()) if finite else 0.
        route_advance = 0.
        if finite:
            route_points = _points(self._reference_for_height)
            first = project_polyline(initial.x, initial.y, route_points)[0]
            last = project_polyline(states[-1, 0], states[-1, 1], route_points)[0]
            route_advance = max(0., float(last-first))
        # Compare real optimized progression, not a model's judgement that a
        # slow-follow trajectory is "normal". All references share this scale.
        base = min(1., max(0., route_advance/max(1., limit*self.horizon)))
        tracking_error = np.mean(np.abs(np.sum((states[:, :2]-reference.points[:, :2])
            * np.column_stack((-np.sin(reference.points[:, 2]), np.cos(reference.points[:, 2]))), axis=1))) if finite else 0.
        comfort = float(np.mean(result.controls[:, 0]**2))/max(1., self.numerical_config.accel_max**2) if finite else 0.
        cost = min(.16, .025*float(tracking_error)+.012*comfort)
        lateral, longitudinal = metadata["label"], metadata["longitudinal"]
        return {"id": reference.name, "index": index, "lateral": lateral,
            "lateral_semantics": metadata["semantics"], "longitudinal": longitudinal,
            "base_score": base if usable else 0., "numerical_cost": cost,
            "forward_progress_m": route_advance, "optimized_travel_distance_m": distance,
            "terminal_speed_mps": max(0., float(states[-1, 3])) if finite else 0.,
            "desired_speed_mps": float(metadata["desired_speed_mps"]),
            "optimization_feasible": bool(result.feasible), "trajectory_finite": finite,
            "optimizer_converged": bool(result.converged), "solver": deepcopy(result.diagnostics),
            "optimizer_status": result.status, "optimizer_message": result.message,
            "solver_cost": float(result.cost) if math.isfinite(result.cost) else None,
            "physical_lane_side": metadata.get("physical_lane_side", "center"),
            "maneuver": bool(metadata.get("maneuver")), "recovery": bool(metadata.get("recovery")),
            "lane_offset_m": metadata.get("lane_offset_m", 0.),
            # Legacy experience schema names are retained as storage keys only;
            # this backend has no IDM propagation or following policy.
            "parameter_model": "nonlinear_mpc_not_IDM", "idm_headway_s": 1.5,
            "idm_min_gap_m": 1.5, "idm_accel_max_mps2": self.numerical_config.accel_max,
            "idm_decel_max_mps2": abs(self.numerical_config.accel_min),
            "safety": {"finite": usable, "no_collision": float(result.feasible),
                "drivable_area": float(result.feasible), "driving_direction": 1.,
                "ttc": float(result.feasible), "traffic_light": bool(result.feasible)},
            "feasibility_reason": "constrained_solution" if usable else "nonlinear_mpc_infeasible"}

    def _expanded_references(self, baseline):
        """A model may invite more optimization, never relax physical limits.

        Bounded lateral references retain a tube entirely inside their source
        tube. They do not widen road geometry, cross a solid line or replay an
        old trajectory. Each added reference still needs a fresh constrained
        solution from this measured snapshot.
        """
        additions = []
        maximum = self.recovery_config["max_paths"]
        for original in baseline:
            if not original.name.endswith(":CRUISE"):
                continue
            width = float(np.min(original.half_width))
            progress = np.linalg.norm(original.points[:, :2]-original.points[0, :2], axis=1)
            ramp = np.clip(progress/6., 0., 1.)
            ramp = 10*ramp**3-15*ramp**4+6*ramp**5
            normal = np.column_stack((-np.sin(original.points[:, 2]), np.cos(original.points[:, 2])))
            offset_limit = min(self.recovery_config["max_offset_m"], max(0., width-.15))
            for offset in sorted({sign*min(amount, offset_limit) for sign in (-1, 1)
                                  for amount in (.25, .45)}):
                if abs(offset) < .05:
                    continue
                points = original.points.copy()
                points[:, :2] += normal*(ramp*offset)[:, None]
                metadata = dict(original.metadata, recovery=True,
                    recovery_recipe="bounded_mpc_reference_offset", lane_offset_m=offset)
                additions.append(MPCReference(original.name+"@recovery%+.2f" % offset,
                    points, width-abs(offset), speed_limit=original.speed_limit, metadata=metadata))
                if len(additions) >= maximum:
                    return additions
        return additions

    def _execution(self, result, tf):
        from carla_bridge.state.coords import global_to_ego
        states = result.states
        output_dt = min(.1, self.dt)
        ts = np.arange(0., self.horizon+output_dt/2, output_dt)
        rows = []
        for time in ts:
            index = min(self.samples, int(math.floor((time+1e-10)/self.dt)))
            fraction = float(time-index*self.dt)
            if index == self.samples or fraction <= 1e-9:
                rows.append(states[index].copy())
            else:
                rows.append(bicycle_step(states[index], result.controls[index], fraction,
                                         self.vehicle_parameters.wheel_base)[0])
        rows = np.asarray(rows)
        # The physical actor->rear transform is constant in the rigid vehicle.
        initial = states[0]
        delta = initial[:2]-[tf.location.x, tf.location.y]
        c, s = math.cos(initial[2]), math.sin(initial[2])
        dx, dy = c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]
        actor_x = rows[:, 0]-np.cos(rows[:, 2])*dx+np.sin(rows[:, 2])*dy
        actor_y = rows[:, 1]-np.sin(rows[:, 2])*dx-np.cos(rows[:, 2])*dy
        yaw = math.radians(tf.rotation.yaw)
        trajectory = []
        for t, x, y, heading, speed in zip(ts, actor_x, actor_y, rows[:, 2], rows[:, 3]):
            forward, left = global_to_ego(x, y, tf.location.x, tf.location.y, yaw)
            trajectory.append({"t": round(float(t), 8), "x": float(forward), "y": float(left),
                "optional_heading_rad": _angle(yaw-float(heading)), "optional_v": max(0., float(speed))})
        return trajectory

    def _visualization(self, references, results, candidates, selected, tf, timestamp):
        if not self._trajectory_visualization_config["enabled"]:
            return
        from carla_bridge.visualization.trajectory_renderer import _elevations
        first = results[0].states[0]
        delta = first[:2]-[tf.location.x, tf.location.y]
        c, s = math.cos(first[2]), math.sin(first[2])
        dx, dy = c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]
        surface = [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                   for wp in self._reference_for_height]
        snapshots = []
        for candidate, result in zip(candidates, results):
            states = result.states
            if not np.isfinite(states).all():
                continue
            xy = states[:, :2]+np.column_stack((-np.cos(states[:, 2])*dx+np.sin(states[:, 2])*dy,
                                               -np.sin(states[:, 2])*dx-np.cos(states[:, 2])*dy))
            zs = _elevations(xy, surface, tf.location.z)
            snapshots.append({"id": candidate["id"], "index": candidate["index"],
                "points_world": np.column_stack((xy, zs)).tolist(),
                "headings_world_rad": states[:, 2].tolist(), "base_score": candidate["base_score"],
                "final_score": candidate.get("final_score")})
        self.last_visualization_snapshot = {"source": "nonlinear_mpc_optimized_rollout",
            "frame": "carla_world_actor", "timestamp_us": timestamp,
            "selected_index": selected["index"] if selected else None,
            "candidate_pool_count": len(candidates), "candidates": snapshots,
            "reference_surface_xyz": surface, "elevation_source": "resolved_route_surface_projection"}

    def plan(self, semantic_decision=None, semantic_enabled=True, *, semantic_update=False,
             sim_time_s=None, special_situations=()):
        self.last_visualization_snapshot = None
        timestamp = int(round(self.world.get_snapshot().timestamp.elapsed_seconds*1e6))
        now = timestamp/1e6 if sim_time_s is None else float(sim_time_s)
        initial, tf = self._initial_state(now)
        try:
            local, index = self._resolve_reference()
            context = build_scene_context(self.world, self.ego, self.route, now,
                reference=self._resolved_route_waypoints, special_situations=special_situations)
        except Exception as exc:
            raise MPCPlanningError(str(exc), {"backend": "mpc", "failure_reason": "road_geometry_error",
                "timestamp_us": timestamp, "error_type": type(exc).__name__}) from exc
        limit = float(self.ego.get_speed_limit())/3.6
        if not math.isfinite(limit) or limit < .1:
            raise ValueError("CARLA speed limit must be finite and at least .1 m/s")
        self._speed_limit = limit = min(limit, self.max_speed)
        self._bind_dynamic_steering_limit(initial.speed, limit)
        evidence = self.experience_memory.retrieve(context) if self.experience_memory else None
        maneuver = self.lane_maneuver.update(self.world, self.ego, self._resolved_route_waypoints, now,
            context=context, current_side=self._side(local), route_index=index, candidate_speed_mps=limit)
        stop_line = self._stop_line(local, initial, context)
        obstacles, actor_audit = self._obstacles(initial, tf)
        lane_width = np.array([float(wp.lane_width) for wp in local])
        base_meta = {"label": "KEEP_LANE", "semantics": ["KEEP_LANE", "FOLLOW_ROUTE"],
            "physical_lane_side": "center", "route_target": True}
        paths = [("center", _points(local), lane_width, base_meta, limit,
                  np.radians([wp.transform.rotation.yaw for wp in local]))]
        if maneuver["active"]:
            reflected = np.asarray(maneuver["points"])*np.array([1., -1.])
            width = np.asarray(maneuver["metadata"]["lane_quality_widths"])
            # The numerical manoeuvre contributes geometry only. MPC performs
            # current-state prediction, braking and footprint constraints anew.
            paths.append((maneuver["metadata"]["id"], reflected, width,
                maneuver["metadata"], min(limit, maneuver["speed_cap_mps"]), None))
        priors = (evidence or {}).get("parameter_priors", []) if self.planning_memory_mode == "full" else []
        references = []
        for identity, points, width, meta, cap, headings in paths:
            for suffix, desired, label in (("CRUISE", cap, "ACCELERATE"),
                                           ("SLOW", cap*.55, "FOLLOW"), ("STOP", 0., "STOP")):
                metadata = dict(meta, desired_speed_mps=desired, longitudinal=label)
                references.append(self._reference(identity+":"+suffix, points, initial, desired, width,
                    metadata, headings=headings, stop_line=stop_line))
            for number, prior in enumerate(priors[:2]):
                if prior.get("status") != "execution_supported" or prior.get("support_count", 0) < 3:
                    continue
                target = (prior.get("parameters") or {}).get("target_velocity_mps")
                if type(target) not in (int, float) or not math.isfinite(target):
                    continue
                desired = float(np.clip(target, min(2., cap), cap))
                meta_memory = dict(meta, desired_speed_mps=desired, longitudinal="MAINTAIN",
                                   memory_support=prior.get("record_ids", []))
                references.append(self._reference(identity+":MEMORY%d" % number, points, initial,
                    desired, width, meta_memory, headings=headings, stop_line=stop_line))
        results = [self.solver.solve(MPCProblem(initial, reference, tuple(obstacles))) for reference in references]
        candidates = [self._candidate(ref, result, i, initial, limit)
                      for i, (ref, result) in enumerate(zip(references, results))]
        baseline_candidates = list(candidates)
        baseline, base_audit = choose_candidate(candidates, {}, False)
        context["candidates_summary"] = candidate_summary(candidates, base_audit["baseline_selected_id"])
        context["planner_health"] = planner_health(candidates, base_audit["baseline_selected_id"])
        context["planner_health"]["source"] = "nonlinear_mpc_baseline"
        degradation = None
        recovery = {"authorized": False, "expanded": False, "selected": False, "added_paths": 0,
            "reason": "legal_numerical_maneuvers_in_baseline_pool", "constraints_relaxed": False}
        if self.governor is None:
            selected, diagnostics = choose_candidate(candidates, semantic_decision, semantic_enabled,
                self.lateral_weight, self.longitudinal_weight)
        else:
            degradation = self.governor.observe_planner_health(context, now)
            context["planner_degradation"] = {key: deepcopy(value) for key, value in degradation.items()
                                              if key != "native_health"}
            if semantic_update:
                self.governor.submit(semantic_decision, context, now)
            if semantic_enabled and self.recovery_config["enabled"]:
                recovery.update(self.governor.recovery_request(context, now))
                if recovery.get("authorized"):
                    extra = self._expanded_references(references)
                    for reference in extra:
                        result = self.solver.solve(MPCProblem(initial, reference, tuple(obstacles)))
                        candidates.append(self._candidate(reference, result, len(candidates), initial, limit))
                        references.append(reference)
                        results.append(result)
                    recovery.update(expanded=bool(extra), added_paths=len(extra),
                        positive_recovery_candidates=sum(bool(c.get("recovery")) and
                            c["optimization_feasible"] and c["forward_progress_m"] > .2 for c in candidates))
            selected, diagnostics = self.governor.select(candidates, context, now,
                enabled=semantic_enabled, memory_evidence=evidence)
            recovery["selected"] = bool(selected and selected.get("recovery"))
            if self.semantic_policy == "shadow":
                shadow_id = selected["id"] if selected else None
                shadow_candidates = deepcopy(diagnostics["candidates"])
                shadow_hash = diagnostics["candidate_pool_sha256"]
                selected, restored = choose_candidate(baseline_candidates, {}, False)
                diagnostics.update(selected_id=restored["selected_id"], baseline_selected_id=restored["baseline_selected_id"],
                    candidates=restored["candidates"], candidate_pool_sha256=restored["candidate_pool_sha256"],
                    shadow_selected_id=shadow_id, shadow_only=True, shadow_candidates=shadow_candidates,
                    shadow_candidate_pool_sha256=shadow_hash)
                diagnostics["governor"]["would_adopt"] = diagnostics["governor"]["adopted"]
                diagnostics["governor"]["adopted"] = False
                recovery["selected"] = False
        self.last_context, self.last_candidates = context, baseline_candidates
        diagnostics.update(backend="mpc", provenance=self.provenance, semantic_policy=self.semantic_policy,
            selection_rule="constrained_mpc_progress_minus_soft_cost_plus_optional_semantic",
            solver_feasibility_gate_applied=True, safety_is_diagnostic_only=False,
            timestamp_us=timestamp, horizon_s=self.horizon, sample_interval_s=self.dt,
            planner_context=context, planner_degradation=degradation, memory_evidence=evidence,
            planning_memory_mode=self.planning_memory_mode, query_decision=self.last_query_decision,
            original_baseline_selected_id=base_audit["baseline_selected_id"],
            original_candidate_pool_sha256=base_audit["candidate_pool_sha256"],
            original_baseline_candidates=base_audit["candidates"] if recovery["expanded"] else None,
            lane_maneuver=deepcopy(maneuver["diagnostics"]), actors=actor_audit,
            steering_state=self._steering_state_diagnostics, traffic_stop_line=stop_line,
            speed_dependent_steering=self._dynamic_steer_diagnostics,
            geometry={"reference_resolution": self._route_resolution, "current_side": self._side(local),
                "initial_topology_corrections": deepcopy(self._reference_topology_corrections)},
            recovery=recovery,
            vehicle={"wheelbase_m": self.vehicle_parameters.wheel_base, "width_m": self.vehicle_parameters.width,
                "length_m": self.vehicle_parameters.length, "rear_axle_to_center_m": self.rear_axle_to_center,
                "rear_axle_actor_xyz_m": list(self.rear_axle_actor_xyz_m), "max_steer_rad": self.max_steer})
        try:
            self._visualization(references, results, diagnostics["candidates"], selected, tf, timestamp)
        except Exception:
            self.last_visualization_snapshot = None
        if selected is None:
            diagnostics.update(emergency_brake=True, failure_reason="no_feasible_mpc_solution")
            return {"trajectory": [], "target_speed": 0., "selected_lateral": "KEEP_LANE",
                    "selected_longitudinal": "STOP", "planner_diagnostics": diagnostics}
        selected_index = selected["index"]
        result, reference = results[selected_index], references[selected_index]
        committed = self.lane_maneuver.commit_selected(selected["id"], now,
            selected_path_points=np.asarray(paths[1][1])*[1., -1.] if selected.get("maneuver") else None,
            selected_metadata=reference.metadata)
        diagnostics["lane_maneuver"].update(selected=bool(selected.get("maneuver")),
            committed_after_selection=committed, execution_commit=dict(self.lane_maneuver.last_commit_audit))
        diagnostics["lane_maneuver"]["execution_commit"].update(
            geometry_role="committed_maneuver_reference_not_optimized_execution",
            optimized_rollout_sha256=hashlib.sha256(result.states.tobytes()).hexdigest())
        trajectory = self._execution(result, tf)
        diagnostics.update(emergency_brake=False, selected_lateral=selected["lateral"],
            selected_longitudinal=selected["longitudinal"], execution_trajectory={
                "geometry_source": "nonlinear_mpc_optimized_rear_axle_rollout",
                "speed_profile_source": "nonlinear_mpc_optimized_velocity",
                "frame": "capture_time_carla_actor_forward_left",
                "sample_interval_s": min(.1, self.dt), "sample_count": len(trajectory),
                "execution_origin_distance_m": math.hypot(trajectory[0]["x"], trajectory[0]["y"]),
                "selected_states_sha256": hashlib.sha256(result.states.tobytes()).hexdigest()},
            mpc_tracking={"active": True, "source": "nonlinear_mpc_optimized_controls",
                "issued_at_s": now, "expires_at_s": now+min(.3, float(self.config.get("replan_interval_s", .2))+.1),
                "sample_interval_s": self.dt, "bicycle_angles_rad": result.controls[:, 1].tolist(),
                "accelerations_mps2": result.controls[:, 0].tolist(), "speeds_mps": result.states[:-1, 3].tolist()})
        return {"trajectory": trajectory, "target_speed": trajectory[-1]["optional_v"],
                "selected_lateral": selected["lateral"], "selected_longitudinal": selected["longitudinal"],
                "planner_diagnostics": diagnostics}

    def query_decision(self, sim_time_s):
        if self.governor is None:
            return {"query": True, "reason": "always_policy"}
        if self.last_context is None:
            return {"query": False, "reason": "no_current_context"}
        memory = self.experience_memory.retrieve(self.last_context) if self.experience_memory else None
        self.last_query_decision = self.governor.query_decision(self.last_context, self.last_candidates,
            sim_time_s, memory_evidence=memory)
        return self.last_query_decision


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Check independent MPC dependencies without CARLA or VLM")
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate_dependencies(), ensure_ascii=False, indent=2))
