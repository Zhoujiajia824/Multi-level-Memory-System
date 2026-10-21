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

from .actor_motion import ActorMotionHistory, MOTION_MODEL_VERSION, predict_actor_motion
from .lane_maneuver import LaneManeuver
from .mpc_solver import (MPCConfig, MPCInitialState, MPCObstacle, MPCProblem,
                         MPCReference, MPCSolver, bicycle_step, MPC_COLLISION_GEOMETRY_VERSION,
                         MPC_SOLVE_POLICY_VERSION)
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


MPC_BRAKING_RECOVERY_HORIZON_S = 1.2
MPC_BRAKING_RECOVERY_HORIZON_LEVELS_S = (1.2, .8, .6)
MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S = 8.0
MPC_BRAKING_RECOVERY_POLICY_VERSION = "jerk_compatible_no_added_positive_demand_v1"
MPC_MINIMUM_RISK_HOLD_POLICY_VERSION = "finite_infeasibility_full_brake_reobserve_v1"
MPC_MINIMUM_RISK_HOLD_LIFETIME_S = .3


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
            ego_body_center_lateral=self.ego_body_center_lateral, ego_body_yaw=self.ego_body_yaw,
            ego_length=self.vehicle_parameters.length, ego_width=self.vehicle_parameters.width,
            steer_max=min(self.max_steer, float((self.config.get("mpc") or {}).get("steer_max", self.max_steer))))
        self.numerical_config, self.solver = numerical, MPCSolver(numerical)
        self._configured_steer_max = numerical.steer_max
        self.dt, self.samples = numerical.dt, numerical.horizon_steps
        self.horizon = self.dt * self.samples
        self.actor_motion = ActorMotionHistory(self.config.get("actor_motion"))
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
            "prediction_kind": "connected_lane_observed_turn_rate_and_following_response_not_future_GT",
            "actor_motion_version": MOTION_MODEL_VERSION,
            "actor_motion_config": dict(self.actor_motion.config),
            "collision_geometry_version": MPC_COLLISION_GEOMETRY_VERSION,
            "solve_policy_version": MPC_SOLVE_POLICY_VERSION,
            "braking_recovery_horizon_s": MPC_BRAKING_RECOVERY_HORIZON_S,
            "braking_recovery_horizon_levels_s": list(MPC_BRAKING_RECOVERY_HORIZON_LEVELS_S),
            "braking_recovery_max_continuous_s": MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S,
            "braking_recovery_policy_version": MPC_BRAKING_RECOVERY_POLICY_VERSION,
            "minimum_risk_hold_version": MPC_MINIMUM_RISK_HOLD_POLICY_VERSION,
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
        self.ego_body_center_lateral = float(getattr(box.location, "y", 0.)-rear[1])
        self.ego_body_yaw = math.radians(float(getattr(getattr(box, "rotation", None), "yaw", 0.)))
        if not 0 < self.rear_axle_to_center < 2*float(box.extent.x):
            raise ValueError("CARLA physical rear axle lies outside the supported body geometry")
        self.physical_max_steer = math.radians(max(float(w.max_steer_angle) for w in wheels[:2]))
        self.steering_model = SteeringModel(self.physical_max_steer,
            [(point.x, point.y) for point in getattr(physics, "steering_curve", ())],
            wheelbase_m=wheelbase, front_track_m=track)
        self.max_steer, self.front_track_m = self.steering_model.max_steer_rad, track
        self.vehicle_parameters = SimpleNamespace(wheel_base=wheelbase,
            width=2*float(box.extent.y), length=2*float(box.extent.x),
            rear_axle_to_center=self.rear_axle_to_center,
            body_center_lateral=self.ego_body_center_lateral, body_yaw=self.ego_body_yaw)

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
            "body_center_lateral_m": quantized(self.ego_body_center_lateral, 3),
            "body_yaw_rad": quantized(self.ego_body_yaw, 6),
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
        elif source == "mpc_minimum_risk_hold":
            # This is direct full-brake actuation without an accepted MPC
            # acceleration profile, even if an older profile remains in audit.
            reason = "minimum_risk_hold_reset_desired_input"
        elif source == "tracker" and audit.get("applied") is True:
            # PID stationary_hold describes actual zero-speed brake actuation,
            # not a replacement acceleration demand. Preserve an accepted MPC
            # input through the unilateral stop so its next jerk bound uses
            # the actual published command history, even while speed is zero.
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
        elif tracker.get("reason") == "stationary_hold":
            reason = "stationary_hold_zero_desired_acceleration"
        elif source == "tracker":
            reason = "mpc_control_override_rejected_reset_desired_input"
        self._pending_desired_acceleration = {"value_mps2": value, "sim_time_s": float(now_s),
            "source": reason, "profile_sample_index": sample_index,
            "tracker_reason": tracker.get("reason"),
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
            # Evaluation checkpoints describe the actor origin, while MPC
            # integrates its measured rear axle. At a spawn checkpoint the
            # physical rear axle can precede station zero by ~1.4m. Without a
            # connected physical prefix even STOP is attracted forward to the
            # first checkpoint instead of holding its actual road position.
            # Keep this prefix private; route progress/checkpoints/indexing are
            # unchanged. A merge/ambiguous predecessor is never guessed.
            self._route_start_predecessors = []
            self._reference_start_extension = {"method": "verified_explicit_predecessors",
                "point_count": 0, "length_m": 0., "evaluation_route_modified": False}
            if self._resolved_route_waypoints:
                cursor = self._resolved_route_waypoints[0]
                required = max(8., self.vehicle_parameters.length+2.)
                length = 0.
                while length < required:
                    previous_getter = getattr(cursor, "previous", None)
                    if not callable(previous_getter):
                        self._reference_start_extension["stop_reason"] = "predecessor_api_unavailable"
                        break
                    predecessors = list(previous_getter(2.))
                    if len(predecessors) != 1:
                        self._reference_start_extension["stop_reason"] = ("ambiguous_predecessors"
                            if predecessors else "no_predecessor")
                        break
                    candidate = predecessors[0]
                    cp, current = candidate.transform, cursor.transform
                    distance = math.hypot(cp.location.x-current.location.x,
                                          cp.location.y-current.location.y)
                    driving = str(getattr(candidate, "lane_type", "Driving")).split(".")[-1] == "Driving"
                    linked = (driving and .25 < distance < 2.2
                        and abs(cp.location.z-current.location.z) < .5
                        and abs(_angle(math.radians(cp.rotation.yaw-current.rotation.yaw))) < math.radians(25.)
                        and any(lane_key(successor) == lane_key(cursor)
                            and math.hypot(successor.transform.location.x-current.location.x,
                                           successor.transform.location.y-current.location.y) < .08
                            and abs(successor.transform.location.z-current.location.z) < .1
                            and abs(_angle(math.radians(successor.transform.rotation.yaw-current.rotation.yaw)))
                                < math.radians(3.) for successor in candidate.next(2.)))
                    if not linked:
                        self._reference_start_extension["stop_reason"] = "predecessor_not_verified_connected"
                        break
                    self._route_start_predecessors.insert(0, candidate)
                    length += distance
                    cursor = candidate
                self._reference_start_extension.update(point_count=len(self._route_start_predecessors),
                    length_m=float(length))
        full = self._resolved_route_waypoints
        index = min(len(full)-1, max(0, int(self.route._nearest_index(self.ego.get_location()))))
        local = list(full[max(0, index-6):min(len(full), index+61)])
        if len(local) < 2:
            local = list(full[-2:])
        if local and full and local[0] is full[0]:
            local = list(self._route_start_predecessors)+local
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
        # plan() updates this shared snapshot before scene/maneuver generation.
        # Direct adapter probes can still initialize it without accumulating a
        # second history sample at the same frozen simulator frame.
        if self.actor_motion.last_time_s is None:
            self.actor_motion.update(self.world)
        current_actors = tuple(self.world.get_actors())
        for actor in current_actors:
            if actor.id == self.ego.id or not actor.type_id.startswith(("vehicle.", "walker.pedestrian", "static.prop.")):
                continue
            pose = actor.get_transform()
            p = pose.location
            if abs(p.z-tf.location.z) > 2.5 or math.hypot(p.x-tf.location.x, p.y-tf.location.y) > 70:
                continue
            motion, box = self.actor_motion.motion_for(actor), actor.bounding_box
            centers, body_yaws, prediction = predict_actor_motion(
                actor, motion, times, self.map, self.actor_motion.config,
                history=self.actor_motion, leader_actors=current_actors)
            uncertainty = np.asarray(prediction["uncertainty_margin_m"])
            # Pedestrian circles use the exact ego rectangle/circle solver
            # constraint; a tall longitudinal vehicle envelope is unnecessary.
            rectangle = None
            if actor.type_id.startswith("walker."):
                radii = max(.1, math.hypot(box.extent.x, box.extent.y)) + uncertainty
            else:
                # True physical bbox, with uncertainty kept as an independent
                # Euclidean clearance rather than baked into an outer ellipse.
                rectangle = np.array([float(box.extent.x), float(box.extent.y)])
                radii = np.tile(rectangle, (self.samples+1, 1))
            predictions.append(MPCObstacle(str(actor.id), centers, radii, body_yaws,
                rectangle_half_extents=rectangle,
                uncertainty_m=uncertainty if rectangle is not None else 0.))
            audit.append({"id": int(actor.id), "type": actor.type_id, "x": float(centers[0, 0]),
                "y": float(centers[0, 1]), "heading": float(body_yaws[0]), "vx": motion.vx,
                "vy": motion.vy, "length": 2*float(box.extent.x), "width": 2*float(box.extent.y),
                "motion": motion.audit, "prediction": prediction,
                "collision_shape": "physical_rectangle" if rectangle is not None else "pedestrian_circle",
                "rectangle_half_extents_m": rectangle.tolist() if rectangle is not None else None,
                "collision_margin_m": self.numerical_config.collision_margin,
                "uncertainty_margin_m": uncertainty.tolist()})
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
        if (str(identity).upper().split(":")[-1].split("@")[0] == "STOP"
                and desired == 0. and initial.speed <= .03):
            # An already stopped vehicle holds its measured physical pose.
            # The solver uses this for the objective only; the connected map
            # reference below still defines every road/stop-line constraint.
            # This also supports old providers without previous() waypoints,
            # where station zero must never become a forward motion request.
            refmeta["stationary_hold_pose"] = [float(initial.x),float(initial.y),float(initial.yaw)]
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
            "safety": {"finite": finite, "no_collision": float(result.feasible),
                "drivable_area": float(result.feasible), "driving_direction": 1.,
                "ttc": float(result.feasible), "traffic_light": bool(result.feasible)},
            "feasibility_reason": "constrained_solution" if usable else "nonlinear_mpc_infeasible"}

    def _maneuver_path_limit(self, points, meta, cap, maneuver, initial, signal_stop):
        """Keep ordinary continuations inside the verified merge-speed/window.

        The reservation belongs to the passing lane; a real return trajectory
        must remain free to leave that lane and join the route.
        """
        if not maneuver.get("active") or not maneuver["diagnostics"].get("committed"):
            return cap, signal_stop
        side = meta.get("physical_lane_side", "center")
        passing_side = maneuver["diagnostics"]["side"]
        target_side = maneuver["metadata"]["physical_lane_side"]
        if side in (passing_side, target_side):
            cap = min(cap, float(maneuver["speed_cap_mps"]))
        stop_xy = maneuver["metadata"].get("return_wait_world_xy",
                    maneuver["metadata"].get("maneuver_stop_world_xy"))
        if side != passing_side or stop_xy is None:
            return cap, signal_stop
        pts, stations, spline = _curve(points)
        point = np.asarray(stop_xy)*[1., -1.]
        station = project_polyline(*point, pts)[0]
        normal = spline(station, 1)
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        reservation = {"point": spline(station).tolist(), "normal": normal.tolist(),
                       "id": "preserve_legal_return_window"}
        if signal_stop is not None:
            signal_station = project_polyline(*signal_stop["point"], pts)[0]
            if signal_station <= station:
                return cap, signal_stop
        return cap, reservation

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
        samples = len(result.controls)
        if len(states) != samples + 1:
            raise ValueError("MPC execution needs one more state than optimized inputs")
        horizon = samples * self.dt
        output_dt = min(.1, self.dt)
        # A short verification horizon need not be a multiple of output_dt.
        # Include its exact endpoint and never label the final state with a
        # later, unverified timestamp by rounding the sample count upward.
        ts = np.r_[np.arange(0., horizon-1e-9, output_dt), horizon]
        rows = []
        for time in ts:
            index = min(samples, int(math.floor((time+1e-10)/self.dt)))
            fraction = float(time-index*self.dt)
            if index == samples or fraction <= 1e-9:
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

    @staticmethod
    def _recovery_slice(value, count):
        """Preserve scalar limits and truncate sampled limits without padding."""
        if value is None:
            return None
        values = np.asarray(value)
        return float(values) if values.ndim == 0 else values[:count].copy()

    def _finite_solver_result(self, result, steps):
        """Validate complete numeric solver evidence, including its tolerance."""
        try:
            states, controls = np.asarray(result.states), np.asarray(result.controls)
            minima = result.diagnostics["constraint_minimums"]
            violation = result.diagnostics["max_constraint_violation"]
            tolerance = result.diagnostics["constraint_tolerance"]
            return bool(states.shape == (steps+1, 4) and controls.shape == (steps, 2)
                and np.isfinite(states).all() and np.isfinite(controls).all()
                and math.isfinite(result.cost) and isinstance(minima, dict) and minima
                and all(math.isfinite(value) for value in minima.values())
                and math.isfinite(violation) and math.isfinite(tolerance)
                and tolerance == self.numerical_config.constraint_tolerance)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    def _finite_infeasible_result(self, result, steps):
        """Classify real finite optimizer failure without hiding numeric faults."""
        return bool(self._finite_solver_result(result, steps) and not result.feasible
            and result.status == "infeasible" and result.diagnostics["max_constraint_violation"]
            > result.diagnostics["constraint_tolerance"])

    def _recovery_braking_plan(self, references, obstacles, initial, tf, now, timestamp, diagnostics,
                               *, full_results):
        """Try a newly verified short STOP, with an explicit bounded lifetime.

        No normal full-horizon solution reaches this method. Its references,
        footprint, obstacle predictions, stop lines and actuator constraints
        are the exact same current-snapshot data; only the verified prediction
        horizon is shorter. There is no padded stationary tail or claim that a
        short solution validates the nominal four-second horizon.
        """
        episode = getattr(self, "_braking_recovery_episode", None)
        if episode is None or now < episode["started_at_s"]:
            episode = {"started_at_s": float(now),
                "deadline_s": float(now + MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S)}
            self._braking_recovery_episode = episode
        audit = {"active": False, "mode": "verified_short_horizon_stop",
            "issued_at_s": float(now), "nominal_horizon_s": float(self.horizon),
            "episode_started_s": episode["started_at_s"], "episode_deadline_s": episode["deadline_s"],
            "elapsed_s": max(0., float(now-episode["started_at_s"])),
            "max_continuous_s": MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S,
            "acceleration_policy_version": MPC_BRAKING_RECOVERY_POLICY_VERSION,
            "requested_horizon_levels_s": list(MPC_BRAKING_RECOVERY_HORIZON_LEVELS_S),
            "attempted_horizon_levels_s": [],
            "constraints_relaxed": False, "fresh_snapshot": True,
            "semantic_influence": False, "attempts": [], "optimization_feasible": False}
        diagnostics["degraded_braking"] = audit
        diagnostics["provenance"] = dict(diagnostics["provenance"],
            braking_recovery_horizon_s=MPC_BRAKING_RECOVERY_HORIZON_S,
            braking_recovery_horizon_levels_s=list(MPC_BRAKING_RECOVERY_HORIZON_LEVELS_S),
            braking_recovery_max_continuous_s=MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S,
            braking_recovery_policy_version=MPC_BRAKING_RECOVERY_POLICY_VERSION,
            minimum_risk_hold_version=MPC_MINIMUM_RISK_HOLD_POLICY_VERSION)

        finite_full_infeasible = bool(full_results and len(full_results) == len(references)
            and all(self._finite_infeasible_result(result, self.samples) for result in full_results))
        finite_short_infeasible = True
        expected_short_attempts = 0

        def failed(reason):
            audit["reason"] = reason
            diagnostics.update(emergency_brake=True, failure_reason=reason)
            if (reason == "no_feasible_mpc_solution" and finite_full_infeasible
                    and finite_short_infeasible and expected_short_attempts > 0
                    and len(audit["attempts"]) == expected_short_attempts):
                # No safe trajectory is being accepted. The separate actuator
                # contract requests full braking and a new measured snapshot;
                # it shares the existing eight-second recovery deadline.
                diagnostics["minimum_risk_hold"] = {
                    "active": True, "contract_version": MPC_MINIMUM_RISK_HOLD_POLICY_VERSION,
                    "reason": "all_finite_full_and_short_horizon_solutions_infeasible",
                    "actuation": "full_brake_reobserve", "issued_at_s": float(now),
                    "execution_expires_at_s": min(episode["deadline_s"], now+min(MPC_MINIMUM_RISK_HOLD_LIFETIME_S,
                        float(self.config.get("replan_interval_s", .2))+.1)),
                    "episode_started_s": episode["started_at_s"],
                    "episode_deadline_s": episode["deadline_s"],
                    "max_continuous_s": MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S,
                    "nominal_horizon_s": float(self.horizon), "verified_horizon_s": 0.,
                    "optimization_feasible": False, "trajectory_accepted": False,
                    "constraints_relaxed": False, "fresh_snapshot": True,
                    "full_horizon_candidate_count": len(full_results),
                    "short_horizon_attempt_count": len(audit["attempts"]),
                    "attempted_horizon_levels_s": list(audit["attempted_horizon_levels_s"])}
            return {"trajectory": [], "target_speed": 0., "selected_lateral": "KEEP_LANE",
                    "selected_longitudinal": "STOP", "planner_diagnostics": diagnostics}

        if now >= episode["deadline_s"]-1e-9:
            return failed("planner_recovery_timeout")
        if not math.isfinite(initial.speed) or initial.speed < 0.:
            return failed("invalid_mpc_initial_speed")
        if not finite_full_infeasible:
            return failed("invalid_mpc_solver_result" if not any(result.feasible for result in full_results)
                          else "mpc_feasible_candidate_selection_failed")
        audit["terminal_speed_upper_bound_mps"] = float(initial.speed)
        step_levels = []
        for requested in MPC_BRAKING_RECOVERY_HORIZON_LEVELS_S:
            steps = int(math.floor(min(requested, MPC_BRAKING_RECOVERY_HORIZON_S)/self.dt+1e-9))
            if (2 <= steps < self.samples and steps not in step_levels
                    and steps*self.dt < self.horizon-1e-9):
                step_levels.append(steps)
        audit["available_horizon_levels_s"] = [steps*self.dt for steps in step_levels]
        expected_short_attempts = len(step_levels)*sum(
            original.metadata.get("longitudinal") == "STOP" for original in references)
        if not step_levels:
            return failed("no_feasible_mpc_solution")
        solvers = getattr(self, "_braking_recovery_solvers", {})
        self._braking_recovery_solvers = solvers
        verified = []
        for steps in step_levels:
            config = replace(self.numerical_config, horizon_steps=steps)
            acceleration_upper = np.maximum(0., initial.acceleration -
                np.arange(1, steps+1)*config.jerk_max*config.dt)
            solver = solvers.get(steps)
            if solver is None or solver.config != config:
                solver = solvers[steps] = MPCSolver(config)
            audit["attempted_horizon_levels_s"].append(steps*self.dt)
            count = steps+1
            short_obstacles = tuple(MPCObstacle(obstacle.id, np.asarray(obstacle.centers)[:count].copy(),
                self._recovery_slice(obstacle.radii, count), self._recovery_slice(obstacle.yaws, count),
                rectangle_half_extents=(np.asarray(obstacle.rectangle_half_extents).copy()
                    if np.asarray(obstacle.rectangle_half_extents).shape == (2,) else
                    self._recovery_slice(obstacle.rectangle_half_extents, count)),
                uncertainty_m=self._recovery_slice(obstacle.uncertainty_m, count))
                for obstacle in obstacles)
            for original in references:
                if original.metadata.get("longitudinal") != "STOP":
                    continue
                points = np.asarray(original.points)[:count].copy()
                speed_limit = self._recovery_slice(original.speed_limit, count)
                if speed_limit is None:
                    speed_limit = np.maximum(initial.speed, np.maximum(1., points[:, 3]))
                elif np.asarray(speed_limit).ndim == 0:
                    speed_limit = np.full(count, speed_limit, dtype=float)
                # A STOP recovery must finish no faster than its observation.
                # This is an optimizer constraint as well as an output guard;
                # the objective alone could otherwise coast on an unavoidable
                # positive input inherited from the previous jerk boundary.
                speed_limit[-1] = min(speed_limit[-1], initial.speed)
                reference = MPCReference(original.name+"@verified_braking",
                    points, self._recovery_slice(original.half_width, count), speed_limit,
                    dict(original.metadata, recovery=True, degraded_braking=True,
                         desired_speed_mps=0., longitudinal="STOP"))
                result = solver.solve(MPCProblem(initial, reference, short_obstacles),
                                      acceleration_upper_bounds=acceleration_upper)
                finite_short_infeasible = (finite_short_infeasible and
                    self._finite_infeasible_result(result, steps))
                finite = bool(np.isfinite(result.states).all() and np.isfinite(result.controls).all())
                feasible = (finite and result.feasible and
                    float(result.diagnostics["max_constraint_violation"]) <= config.constraint_tolerance)
                # Terminal speed alone cannot rule out accelerating and braking
                # back to a stop. The optimizer and this output check both
                # enforce the jerk-compatible no-added-positive-demand policy.
                acceleration_policy = bool(finite and np.all(
                    result.controls[:, 0] <= acceleration_upper+config.constraint_tolerance))
                braking = (finite and result.states[-1, 3] <= initial.speed+config.constraint_tolerance)
                audit["attempts"].append({"reference_id": reference.name, "optimization_feasible": feasible,
                    "finite_result": self._finite_solver_result(result, steps),
                    "braking_or_hold": bool(braking), "acceleration_policy_passed": acceleration_policy,
                    "verified_horizon_s": steps*self.dt,
                    "solver": deepcopy(result.diagnostics), "optimizer_status": result.status})
                if feasible and braking and acceleration_policy:
                    verified.append((float(result.states[-1, 3]), result.cost, reference, result))
            if verified:
                # Preserve the longest verified horizon. Shorter levels are
                # considered only after every STOP at this level has failed.
                break
        if not verified:
            return failed("no_feasible_mpc_solution")
        audit["acceleration_upper_bounds_mps2"] = acceleration_upper.tolist()
        _, _, reference, result = min(verified, key=lambda item: item[:2])
        candidate = self._candidate(reference, result, len(diagnostics["candidates"]), initial, self._speed_limit)
        candidate.update(degraded_braking=True, verified_horizon_s=steps*self.dt)
        selected, selection_audit = choose_candidate([*diagnostics["candidates"], candidate], {}, False)
        if selected is None:
            return failed("no_feasible_mpc_solution")
        diagnostics["full_horizon_selection"] = {key: deepcopy(diagnostics.get(key)) for key in
            ("selected_id", "baseline_selected_id", "candidate_pool_sha256")}
        for key in ("selected_id", "baseline_selected_id", "candidate_pool_sha256", "candidates"):
            diagnostics[key] = selection_audit[key]
        verified_horizon = steps*self.dt
        expiry = min(now+min(.3, float(self.config.get("replan_interval_s", .2))+.1), episode["deadline_s"])
        audit.update(active=True, reason="full_horizon_infeasible_verified_braking_available",
            acceleration_policy_passed=True,
            verified_horizon_s=verified_horizon,
            verified_constraint_tolerance=config.constraint_tolerance, optimization_feasible=True,
            execution_expires_at_s=float(expiry), selected_id=selected["id"],
            verified_constraint_max_violation=float(result.diagnostics["max_constraint_violation"]))
        trajectory = self._execution(result, tf)
        diagnostics.update(emergency_brake=False, failure_reason=None,
            selection_rule="full_horizon_infeasible_then_verified_short_horizon_stop",
            nominal_horizon_s=self.horizon, horizon_s=verified_horizon,
            selected_lateral=selected["lateral"], selected_longitudinal="STOP",
            execution_trajectory={"geometry_source": "nonlinear_mpc_optimized_rear_axle_rollout",
                "speed_profile_source": "nonlinear_mpc_optimized_velocity",
                "frame": "capture_time_carla_actor_forward_left", "sample_interval_s": min(.1, self.dt),
                "sample_count": len(trajectory), "verified_horizon_s": verified_horizon,
                "execution_origin_distance_m": math.hypot(trajectory[0]["x"], trajectory[0]["y"]),
                "selected_states_sha256": hashlib.sha256(result.states.tobytes()).hexdigest()},
            mpc_tracking={"active": True, "source": "nonlinear_mpc_optimized_controls",
                "issued_at_s": now, "expires_at_s": expiry, "sample_interval_s": self.dt,
                "bicycle_angles_rad": result.controls[:, 1].tolist(),
                "accelerations_mps2": result.controls[:, 0].tolist(), "speeds_mps": result.states[:-1, 3].tolist()})
        # The old snapshot has no selected full-horizon candidate. Never show
        # one of those infeasible rollouts as the executed braking solution.
        self.last_visualization_snapshot = None
        return {"trajectory": trajectory, "target_speed": trajectory[-1]["optional_v"],
                "selected_lateral": selected["lateral"], "selected_longitudinal": "STOP",
                "planner_diagnostics": diagnostics}

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
                "final_score": candidate.get("final_score"),
                **{key: deepcopy(candidate[key]) for key in (
                    "lateral", "longitudinal", "physical_lane_side", "lane_offset_m",
                    "lateral_path_index", "longitudinal_policy_id", "maneuver", "maneuver_id",
                    "maneuver_phase", "recovery", "recovery_recipe", "recovery_transition_m",
                    "optimization_feasible", "trajectory_finite", "numerically_valid", "eligible",
                    "ineligible_reason", "forward_progress_m", "terminal_speed_mps",
                    "desired_speed_mps") if key in candidate}})
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
        self.actor_motion.update(self.world, sim_time_s=now)
        initial, tf = self._initial_state(now)
        try:
            local, index = self._resolve_reference()
            context = build_scene_context(self.world, self.ego, self.route, now,
                reference=self._resolved_route_waypoints, special_situations=special_situations,
                motion_history=self.actor_motion)
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
            context=context, current_side=self._side(local), route_index=index, candidate_speed_mps=limit,
            motion_history=self.actor_motion)
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
            cap, path_stop_line = self._maneuver_path_limit(points, meta, cap, maneuver, initial, stop_line)
            for suffix, desired, label in (("CRUISE", cap, "ACCELERATE"),
                                           ("SLOW", cap*.55, "FOLLOW"), ("STOP", 0., "STOP")):
                metadata = dict(meta, desired_speed_mps=desired, longitudinal=label)
                references.append(self._reference(identity+":"+suffix, points, initial, desired, width,
                    metadata, headings=headings, stop_line=path_stop_line))
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
                    desired, width, meta_memory, headings=headings, stop_line=path_stop_line))
        results = [self.solver.solve(MPCProblem(initial, reference, tuple(obstacles))) for reference in references]
        candidates = [self._candidate(ref, result, i, initial, limit)
                      for i, (ref, result) in enumerate(zip(references, results))]
        baseline_references, baseline_results = list(references), list(results)
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
                # Shadow proposals are observation-only. Restore the whole
                # execution pool, including the evidence used for short STOP
                # recovery and minimum-risk hold, not just ranked diagnostics.
                references, results = baseline_references, baseline_results
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
                "initial_topology_corrections": deepcopy(self._reference_topology_corrections),
                "physical_start_extension": deepcopy(self._reference_start_extension)},
            recovery=recovery,
            vehicle={"wheelbase_m": self.vehicle_parameters.wheel_base, "width_m": self.vehicle_parameters.width,
                "length_m": self.vehicle_parameters.length, "rear_axle_to_center_m": self.rear_axle_to_center,
                "body_center_lateral_m": self.ego_body_center_lateral, "body_yaw_rad": self.ego_body_yaw,
                "rear_axle_actor_xyz_m": list(self.rear_axle_actor_xyz_m), "max_steer_rad": self.max_steer})
        try:
            self._visualization(references, results, diagnostics["candidates"], selected, tf, timestamp)
        except Exception:
            self.last_visualization_snapshot = None
        if selected is None:
            return self._recovery_braking_plan(references, obstacles, initial, tf, now, timestamp,
                                               diagnostics, full_results=results)
        self._braking_recovery_episode = None
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
