"""PDM-Closed core executed on CARLA observations and road geometry.

This is a simulator adapter, not the original nuPlan benchmark planner. The
official generator, LQR/bicycle simulator and scorer are
imported unchanged from pinned upstream sources. See README.md for deviations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from .dependencies import load_dependencies, validate_dependencies
from .selection import choose_candidate

logger = logging.getLogger("carla_pdm_planner")


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
    Road compliance is still scored against independently built lane surfaces.
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


def _lane_offset_points(points, offset, api):
    """Use upstream PDM's parallel-path construction inside the same lane."""
    from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import parallel_discrete_path
    return np.array([[state.x, state.y] for state in
                     parallel_discrete_path(_path_states(points, api), offset)])


def _route_compatible_lane_points(reference, closest, side, progress, cut, transition_m,
                                  current_side):
    """Build an adjacent-lane excursion with its legal return to the route.

    Relative left/right lanes are not a route through a junction: in Town05,
    lane 24:1 continues straight/left while the reference 24:2 turns right.
    Truncating 24:1 at that junction invents a stationary IDM leader. Instead,
    include the return lane change in the spatial proposal before that seam.
    This changes proposal geometry, never candidate scoring or eligibility.
    """
    import carla
    from scipy.interpolate import CubicHermiteSpline

    if side == "center" or cut is None:
        return None, None
    end = int(cut["reference_index"])
    last = _adjacent(reference[end - 1], side)
    # Preserve actual lane endings as finite stopping corridors. The special
    # case here is a continuing lane whose successors do not take our exit.
    if not reference[end].is_junction or last is None or not last.next(2.0):
        return None, None
    start = closest
    while start > 0:
        previous, current = _adjacent(reference[start - 1], side), _adjacent(reference[start], side)
        if (previous is None or previous.lane_type != carla.LaneType.Driving
                or not _same_direction(reference[start - 1], previous) or not _connected(previous, current)):
            break
        start -= 1
    # Complete the whole vehicle's transition before the connection diverges.
    merge_end = float(progress[end - 1] - 4.0)
    merge_start = merge_end - transition_m
    ahead = float(progress[closest])
    if merge_start < progress[start] and side != current_side:
        return None, {"reason": "route_exit_return_cannot_fit", "distance_to_exit_m": float(progress[end] - ahead)}
    if side != current_side and merge_start - ahead < transition_m + 4.0:
        return None, {"reason": "insufficient_route_compatible_lane_change_corridor",
                      "distance_to_exit_m": float(progress[end] - ahead)}
    # A replan within an already developing return must retain its fixed map
    # endpoint, rather than demanding another fresh 30 metres every 0.2 s.
    if side == current_side and merge_end - ahead < 4.0:
        return None, {"reason": "route_exit_return_is_too_late", "distance_to_exit_m": float(progress[end] - ahead)}
    if end - start < 2:
        return None, {"reason": "route_exit_has_insufficient_connected_samples",
                      "distance_to_exit_m": float(progress[end] - ahead)}
    direction = carla.LaneChange.Right if side == "left" else carla.LaneChange.Left
    stations = [i for i, station in enumerate(progress) if merge_start - 2 <= station <= merge_end + 2]
    for i in stations:
        source = _adjacent(reference[i], side)
        target = reference[i]
        if (source is None or source.is_junction or target.is_junction
                or not int(source.lane_change) & int(direction)
                or source.lane_type != carla.LaneType.Driving or not _same_direction(source, target)):
            return None, {"reason": "route_exit_has_no_legal_return", "reference_index": i,
                          "source_lane_key": _lane_key(source) if source else None}
    # The shared reference station parameter keeps inner/outer curve samples
    # aligned; their own arc lengths differ and must not be mixed.
    centers = np.array([_xy(wp) for wp in reference[start:]])
    tangents = np.array([[math.cos(_heading(wp)), math.sin(_heading(wp))] for wp in reference[start:]])
    center_keep = np.r_[True, np.diff(progress[start:]) > 1e-4]
    side_keep = np.r_[True, np.diff(progress[start:end]) > 1e-4]
    if center_keep.sum() < 2 or side_keep.sum() < 2:
        return None, {"reason": "route_exit_has_insufficient_distinct_samples"}
    center_curve = CubicHermiteSpline(progress[start:][center_keep], centers[center_keep], tangents[center_keep], axis=0)
    left_points = np.array([_xy(_adjacent(wp, side)) for wp in reference[start:end]])
    left_tangents = np.array([[math.cos(_heading(_adjacent(wp, side))),
                               math.sin(_heading(_adjacent(wp, side)))] for wp in reference[start:end]])
    side_curve = CubicHermiteSpline(progress[start:end][side_keep], left_points[side_keep], left_tangents[side_keep], axis=0)
    samples = np.unique(np.r_[progress[start:], np.arange(progress[start], progress[-1], .25),
                              max(merge_start, progress[start]), merge_end])
    u = np.clip((samples - merge_start) / transition_m, 0.0, 1.0)
    blend = 1 - 10 * u ** 3 + 15 * u ** 4 - 6 * u ** 5
    # Avoid extrapolating an absent lane after it has already been left.
    side_samples = side_curve(np.minimum(samples, merge_end))
    points = center_curve(samples) + blend[:, None] * (side_samples - center_curve(samples))
    return points, {"method": "legal_return_in_spatial_proposal", "merge_start_ahead_m": merge_start - ahead,
                    "merge_end_ahead_m": merge_end - ahead, "return_length_m": transition_m,
                    "exit_reference_index": end, "selection_gate_applied": False}


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
        self.semantic_policy = self.config.get("semantic_policy", "always")
        if self.semantic_policy not in ("always", "selective", "shadow"):
            raise ValueError("semantic_policy must be always, selective or shadow")
        self.planning_memory_mode = self.config.get("planning_memory", "off")
        if self.planning_memory_mode not in ("off", "gate", "full"):
            raise ValueError("planning_memory must be off, gate or full")
        self.experience_memory = None
        self.last_context = None
        self.last_candidates = []
        self.last_query_decision = None
        self._memory_policy_metadata = {}
        if self.semantic_policy != "always":
            from .semantic_governor import SelectiveSemanticGovernor
            self.governor = SelectiveSemanticGovernor(self.config.get("semantic_governor"))
        else:
            self.governor = None
        self._reference_route_source = None
        self._resolved_route_waypoints = None
        self._route_resolution = None
        self._junction_area_cache = {}
        self._road_area_cache = {}
        for name, value in (("horizon_s", self.horizon), ("max_speed_mps", self.max_speed),
                            ("map_radius_m", self.map_radius), ("lane_change_length_m", self.transition_m)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        if self.horizon < 3 or self.max_speed > 20 or self.transition_m < 15:
            raise ValueError("PDM requires horizon >=3s, max speed <=20m/s, and lane-change length >=15m")
        self.samples = int(math.ceil(self.horizon / self.dt))
        self.lqr_lateral_horizon_s = float(self.config.get("lqr_lateral_horizon_s", 2.0))
        if not math.isfinite(self.lqr_lateral_horizon_s):
            raise ValueError("lqr_lateral_horizon_s must be finite")
        tracking_steps = self.lqr_lateral_horizon_s / self.dt
        self.lqr_lateral_horizon_samples = round(tracking_steps)
        if (not math.isclose(tracking_steps, self.lqr_lateral_horizon_samples, abs_tol=1e-8)
                or not 2 <= self.lqr_lateral_horizon_samples <= self.samples):
            raise ValueError("lqr_lateral_horizon_s must be an exact sample_interval_s multiple, "
                             "at least two samples and no longer than the planning horizon")
        self.sampling = self.api.TrajectorySampling(num_poses=self.samples, interval_length=self.dt)
        # Upstream's update_rate=2 leaves the freshly reset t=1 leader state at
        # zero, creating a phantom stopped leader on every short replan. Updating
        # at each sample initializes the first prediction and prevents repeated
        # artificial braking in the 0.2s CARLA planning loop.
        self.generator = self.api.PDMGenerator(self.sampling, self.sampling, leading_agent_update_rate=1)
        self.simulator = self.api.PDMSimulator(self.sampling)
        self.scorer = self.api.PDMScorer(self.sampling)
        self.vehicle_parameters = self._vehicle_parameters()
        # Upstream resets only the bicycle's vehicle per rollout. Its LQR
        # otherwise retains the default Pacifica wheelbase and its bicycle
        # keeps a 60-degree steering limit. Bind both official components to
        # this CARLA vehicle so the controller and simulated plant agree.
        # Lateral preview and longitudinal braking must not share a horizon:
        # extending both weakens the predicted hard-stop response. Preserve
        # native one-second longitudinal control, with a separate lateral preview.
        from .lqr_tracker import build_lqr_tracker
        self.simulator._tracker = build_lqr_tracker(
            self.api, self.vehicle_parameters, self.dt,
            longitudinal_tracking_horizon_samples=10,
            lateral_tracking_horizon_samples=self.lqr_lateral_horizon_samples)
        self.simulator._motion_model = self.api.BatchKinematicBicycleModel(
            vehicle=self.vehicle_parameters, max_steering_angle=self.max_steer)
        self.provenance = dict(self.api.provenance)

    def _vehicle_parameters(self):
        box = self.ego.bounding_box
        half_length, width = float(box.extent.x), 2 * float(box.extent.y)
        physics = self.ego.get_physics_control()
        wheels = physics.wheels
        if len(wheels) < 4:
            raise ValueError("PDM bicycle model requires a four-wheel ego vehicle")
        # CARLA 0.9.15 reports wheel positions in WORLD centimetres. Transform
        # the physical axle into the actor frame; half the wheelbase assumes
        # the actor/bounding-box centre lies midway between the axles, which
        # is false even for the default Tesla Model 3.
        front = np.mean([[w.position.x, w.position.y, w.position.z] for w in wheels[:2]], axis=0)
        rear = np.mean([[w.position.x, w.position.y, w.position.z] for w in wheels[-2:]], axis=0)
        wheelbase = float(np.linalg.norm(front - rear) * 0.01)
        if not 1.0 < wheelbase < 6.0:
            raise ValueError("CARLA returned invalid wheelbase geometry")
        inverse = np.asarray(self.ego.get_transform().get_inverse_matrix())
        rear_local = inverse @ np.r_[rear * .01, 1.0]
        front_local = [inverse @ np.array([w.position.x * .01, w.position.y * .01,
                                           w.position.z * .01, 1.0]) for w in wheels[:2]]
        self.front_track_m = float(abs(front_local[1][1] - front_local[0][1]))
        measured_offset = float(box.location.x - rear_local[0])
        center_offset = float(self.config.get("rear_axle_to_center_m", measured_offset))
        self._vehicle_geometry_source = {
            "rear_axle_to_center_source": "config_override" if "rear_axle_to_center_m" in self.config
                                         else "carla_world_wheel_positions",
            "measured_rear_axle_to_center_m": measured_offset,
            "rear_axle_actor_xyz_m": rear_local[:3].tolist(),
        }
        if not 0 < center_offset < half_length:
            raise ValueError("rear_axle_to_center_m must lie within the ego footprint")
        # The effective PDM rear axle follows its centre offset, including an
        # explicit config override. Do not confuse that offset with actor->axle.
        self.rear_axle_actor_xyz_m = (float(box.location.x - center_offset),
                                      float(box.location.y), float(rear_local[2]))
        self._vehicle_geometry_source["planning_rear_axle_actor_xyz_m"] = list(self.rear_axle_actor_xyz_m)
        self.physical_max_steer = math.radians(max(float(w.max_steer_angle) for w in wheels[:2]))
        if not 0 < self.physical_max_steer < math.pi / 2:
            raise ValueError("CARLA returned invalid front-wheel steering limits")
        from carla_bridge.control.steering_model import SteeringModel
        self.steering_model = SteeringModel(self.physical_max_steer,
            [(point.x, point.y) for point in getattr(physics, "steering_curve", ())],
            wheelbase_m=wheelbase, front_track_m=self.front_track_m)
        self.max_steer = self.steering_model.max_steer_rad
        return self.api.VehicleParameters(width=width, front_length=half_length + center_offset,
                                         rear_length=half_length - center_offset,
                                         cog_position_from_rear_axle=center_offset, wheel_base=wheelbase,
                                         vehicle_name=self.ego.type_id, vehicle_type="carla",
                                         height=2 * float(box.extent.z))

    def _observed_bicycle_steering(self, speed_mps):
        """Read both physical front wheels while the owning loop freezes ticks.

        A command is not actual wheel feedback: smoothing/sleeping can leave the
        wheels at a different angle, especially during a low-speed stall.
        """
        import carla
        command = float(self.ego.get_control().steer)
        estimate = self.steering_model.normalized_to_bicycle(command, speed_mps)
        getter = getattr(self.ego, "get_wheel_steer_angle", None)
        locations = getattr(carla, "VehicleWheelLocation", None)
        if callable(getter) and hasattr(locations, "FL_Wheel") and hasattr(locations, "FR_Wheel"):
            # CARLA returns degrees and positive-right; the PDM frame is left-positive.
            left_deg, right_deg = float(getter(locations.FL_Wheel)), float(getter(locations.FR_Wheel))
            angle = self.steering_model.front_wheels_to_bicycle(
                math.radians(left_deg), math.radians(right_deg))
            self._steering_state_diagnostics = {
                "source": "carla_measured_front_wheels", "front_left_deg": left_deg,
                "front_right_deg": right_deg, "command_estimate_bicycle_rad": -estimate,
                "bicycle_angle_rad": -angle, "normalized_command": command,
                "input_latency_modeled": False,
            }
        else:
            reason = "wheel_angle_api_unavailable"
            angle = estimate
            self._steering_state_diagnostics = {
                "source": "command_ackermann_estimate", "fallback_reason": reason,
                "bicycle_angle_rad": -angle, "normalized_command": command,
                "input_latency_modeled": False,
            }
            if not getattr(self, "_steering_fallback_warned", False):
                logger.warning("CARLA前轮角API不可用，PDM显式回退至指令转角估计（未建模输入迟滞）")
                self._steering_fallback_warned = True
        return -angle

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
            tire_steering_angle=self._observed_bicycle_steering(math.hypot(*v)),
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
        surface_seeds = []
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
                surface_seeds.extend((a, b))
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
        from .road_area import road_driving_polygons
        dense_polygons, road_info = road_driving_polygons(self.map, surface_seeds, cache=self._road_area_cache)
        self._geometry_diagnostics["road_drivable_area"] = road_info
        for key, polygon in dense_polygons.items():
            polygons[key] = polygons[key].union(polygon) if key in polygons else polygon
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

            for i in range(closest, len(reference)):
                base = reference[i]
                distance_ahead = float(progress[i] - progress[closest])
                wp = _adjacent(base, side)
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
            # Do not start a lane change that cannot finish inside this finite
            # corridor. Staying on a short lane is still a valid braking path.
            endpoint = cut
            if (reason is None and side != current_side and endpoint is not None
                    and endpoint["distance_ahead_m"] <= self.transition_m + 4):
                reason = "insufficient_lane_change_corridor"
            if reason:
                excluded[side] = reason
                continue
            route_points, route_connection = _route_compatible_lane_points(
                reference, closest, side, progress, cut, self.transition_m, current_side)
            if route_connection is not None:
                details["route_connection"] = route_connection
                if route_points is None:
                    if side != current_side:
                        excluded[side] = route_connection["reason"]
                        continue
                    # A vehicle already trapped in the wrong lane still needs
                    # a real finite braking path. Report the failed connection
                    # instead of fabricating an illegal route or dropping it.
                    details["route_connection"]["finite_stopping_corridor_retained"] = True
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
            target_points = route_points if route_points is not None else _lane_path_points(corridor)
            try:
                path = _smooth_path(target_points, rear,
                                    ego_state.rear_axle.heading, self.api, transition)
            except ValueError as exc:
                excluded[side] = "insufficient_path_ahead"
                details["error"] = str(exc)
                continue
            paths.append(path)
            details["path_length_m"] = float(path.length)
            base_metadata = {"id": side, "label": label, "semantics": sorted(set(semantics)),
                             "physical_lane_side": side, "route_target": side == "center", "lane_offset_m": 0.0}
            metadata.append(base_metadata)
            # Upstream PDM includes parallel lateral proposals, whereas the old
            # CARLA adapter had one centreline per lane. A single LQR rollout
            # can overshoot a short-radius exit by centimetres and zero every
            # moving proposal. Add actual within-lane alternatives; their real
            # footprints still receive the unchanged native road/collision score.
            if side == current_side:
                widths = [float(wp.lane_width) for wp in corridor]
                if route_points is not None:
                    widths.extend(float(wp.lane_width) for wp in reference[closest:])
                limit = max(0.0, min(.5, (min(widths)
                                         - self.vehicle_parameters.width) / 2 - .1))
                # IDs represent centimetres; quantize downward before dedup so
                # e.g. .250 and .251 never create two different @+0.25 paths.
                limit = math.floor(limit * 100 + 1e-9) / 100
                offsets = sorted({sign * min(amount, limit) for sign in (-1, 1)
                                  for amount in (.25, .5)}) if limit >= .05 else []
                details["within_lane_offset_limit_m"] = limit
                details["within_lane_offsets_m"] = [0.0] + offsets
                for offset in offsets:
                    try:
                        shifted = _smooth_path(_lane_offset_points(target_points, offset, self.api), rear,
                                               ego_state.rear_axle.heading, self.api, transition)
                    except ValueError as exc:
                        details.setdefault("excluded_offset_paths", {})["%+.2f" % offset] = str(exc)
                        continue
                    paths.append(shifted)
                    metadata.append(dict(base_metadata, id=side + "@%+.2f" % offset, lane_offset_m=offset))
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

    def _policies(self, speed, limit, memory_priors=None):
        # Reject corrupted/sub-walking road limits instead of either exceeding
        # them with a target floor or creating millions of integration steps.
        if not math.isfinite(limit) or limit < 0.1:
            raise ValueError("PDM requires a finite speed limit of at least 0.1 m/s")
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("PDM policy initial speed must be finite and nonnegative")
        # Every run, including semantic-off, has identical stop and slow-yield
        # policies. STOP avoids IDM's undefined desired-velocity=0 division.
        labels = ["STOP", "YIELD", "DECELERATE", "FOLLOW", "MAINTAIN", "ACCELERATE", "STOP_HARD",
                  "YIELD_GENTLE"]
        # DECELERATE is not a creep command: once nearly stopped, hold/stop.
        decel_to_stop = speed < 0.6
        targets = [0.1, min(2.0, limit), 0.1 if decel_to_stop else min(limit, speed * 0.5),
                   limit * 0.75, limit * 0.85, limit, 0.1, min(2.0, limit)]
        # Retain every ordinary 2m/s² proposal. An additional native IDM policy
        # can establish steering gently when starting on a tight bend; it is
        # generated in both semantic arms and competes under the same scoring.
        accelerations = [2.0] * 7 + [0.2]
        headways = [1.5, 2.5, 2.0, 2.5, 1.5, 1.5, 1.5, 2.5]
        gaps, decelerations = [1.5] * 8, [4.0] * 8
        self._memory_policy_metadata = {}
        # Memory proposes bounded IDM parameters; all original policies remain.
        # Re-simulate from the CURRENT measured state, never replay old XY points.
        for prior in (memory_priors or [])[:2]:
            if (not isinstance(prior, dict) or prior.get("status") != "execution_supported"
                    or type(prior.get("support_count")) is not int or prior["support_count"] < 3):
                continue
            action = prior.get("action") or {}
            longitudinal_action = action.get("longitudinal_decision")
            if longitudinal_action not in ("FOLLOW", "MAINTAIN", "ACCELERATE", "DECELERATE"):
                continue
            parameters = prior.get("parameters") or {}
            fields = ("target_velocity_mps", "headway_time", "min_gap_to_lead_agent", "accel_max", "decel_max")
            if not all(type(parameters.get(k)) in (int, float) and math.isfinite(parameters[k]) for k in fields):
                continue
            index = len(labels)
            labels.append("MEMORY_%d" % index)
            targets.append(float(np.clip(parameters["target_velocity_mps"], min(2.0, limit), limit)))
            headways.append(float(np.clip(parameters["headway_time"], 1.0, 3.0)))
            gaps.append(float(np.clip(parameters["min_gap_to_lead_agent"], 1.5, 4.0)))
            accelerations.append(float(np.clip(parameters["accel_max"], .2, 2.0)))
            decelerations.append(float(np.clip(parameters["decel_max"], 2.0, 4.0)))
            self._memory_policy_metadata[index] = {"action": longitudinal_action,
                "support_count": prior.get("support_count"), "record_ids": prior.get("record_ids", [])}
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
                                  min_gap_to_lead_agent=gaps, headway_time=headways,
                                  accel_max=accelerations, decel_max=decelerations)
        policies.update(None)  # Desired velocities are already capped to the road speed limit.
        return policies, labels, targets

    def plan(self, semantic_decision, semantic_enabled=True, *, semantic_update=False,
             sim_time_s=None, special_situations=()):
        from shapely.geometry import Point
        timestamp = int(round(self.world.get_snapshot().timestamp.elapsed_seconds * 1e6))
        now = timestamp/1e6 if sim_time_s is None else float(sim_time_s)
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
        context, memory_evidence = None, None
        # Pure legacy/ablation runs do not incur context construction or memory I/O.
        if getattr(self, "semantic_policy", "always") != "always" or getattr(self, "experience_memory", None) is not None:
            from .scene_context import build_scene_context
            context = build_scene_context(self.world, self.ego, self.route, now,
                        reference=self._resolved_route_waypoints, special_situations=special_situations)
            if self.experience_memory is not None:
                memory_evidence = self.experience_memory.retrieve(context)
        priors = ((memory_evidence or {}).get("parameter_priors", [])
                  if getattr(self, "planning_memory_mode", "off") == "full" else [])
        policies, longitudinal, targets = self._policies(state.dynamic_car_state.speed, limit, priors)
        manager = self.api.PDMProposalManager(paths, policies)
        proposals = self.generator.generate_proposals(state, observation, manager)
        simulated = self.simulator.simulate_proposals(proposals, state)
        scores = self.scorer.score_proposals(simulated, state, observation, centerline, lanes, area,
                                             _IntersectionMap(self.map, tf.location.z, self.api))
        candidates = []
        for index in range(len(manager)):
            proposal = manager[index]
            lat = lateral[proposal.lateral_idx]
            idm_states = self.generator._state_idm_array[index]
            profile = idm_states[:, 1]
            finite = bool(np.isfinite(proposals[index]).all() and np.isfinite(simulated[index]).all()
                          and np.isfinite(idm_states).all() and np.isfinite(scores[index]))
            multi, weighted = self.scorer._multi_metrics, self.scorer._weighted_metrics
            safety = {"no_collision": float(multi[self.api.MultiMetricIndex.NO_COLLISION, index]),
                      "drivable_area": float(multi[self.api.MultiMetricIndex.DRIVABLE_AREA, index]),
                      "driving_direction": float(multi[self.api.MultiMetricIndex.DRIVING_DIRECTION, index]),
                      "ttc": float(weighted[self.api.WeightedMetricIndex.TTC, index]), "finite": finite}
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
                               "route_target": lat["route_target"],
                               "physical_lane_side": lat["physical_lane_side"],
                               "lane_offset_m": lat["lane_offset_m"],
                               "forward_progress_m": float(centerline.project(Point(*proposals[index, -1, :2]))
                                                           - centerline.project(Point(*proposals[index, 0, :2]))) if finite else 0.0,
                               "terminal_speed_mps": float(self.generator._state_idm_array[index, -1, 1]) if finite else 0.0,
                               "longitudinal": self._memory_policy_metadata.get(proposal.longitudinal_idx, {}).get("action",
                                   {"STOP_HARD": "STOP", "YIELD_GENTLE": "YIELD"}.get(
                                   longitudinal[proposal.longitudinal_idx], longitudinal[proposal.longitudinal_idx])),
                               "longitudinal_policy_id": longitudinal[proposal.longitudinal_idx],
                               "longitudinal_policy_index": int(proposal.longitudinal_idx),
                               "idm_accel_max_mps2": float(policies._accel_max[proposal.longitudinal_idx]),
                               "idm_decel_max_mps2": float(policies._decel_max[proposal.longitudinal_idx]),
                               "idm_headway_s": float(policies._headway_time[proposal.longitudinal_idx]),
                               "idm_min_gap_m": float(policies._min_gap_to_lead_agent[proposal.longitudinal_idx]),
                               "memory_prior": self._memory_policy_metadata.get(proposal.longitudinal_idx),
                               "predicted_collision_time_s": collision_time if math.isfinite(collision_time) else None,
                               "predicted_ttc_infraction_time_s": ttc_time if math.isfinite(ttc_time) else None,
                               "drivable_area_failure": area_failure,
                               "speed_at_0_2s_mps": float(profile[2]) if finite else None,
                               "acceleration_range_mps2": [float(np.min(np.diff(profile) / self.dt)),
                                                            float(np.max(np.diff(profile) / self.dt))] if finite else None,
                               "desired_speed_mps": targets[proposal.longitudinal_idx],
                               "base_score": float(scores[index]) if finite else 0.0, "safety": safety,
                               "trajectory_sha256": hashlib.sha256(proposals[index].tobytes()).hexdigest()})
        action_memory = None
        if getattr(self, "experience_memory", None) is not None and semantic_decision:
            # Scene-level paired support can invite consultation. Recheck the
            # requested action separately; it never grants ungrounded STOP or
            # expands the score budget. Candidate-specific failures are already
            # carried by scene memory's action_feedback.
            action_memory = self.experience_memory.retrieve(context, semantic_decision)
        governor = getattr(self, "governor", None)
        if governor is None:
            selected, diagnostics = choose_candidate(candidates, semantic_decision, semantic_enabled,
                                                      self.lateral_weight, self.longitudinal_weight)
        else:
            from .scene_context import candidate_summary
            _, baseline_diagnostics = choose_candidate(candidates, None, False)
            context["candidates_summary"] = candidate_summary(candidates, baseline_diagnostics["baseline_selected_id"])
            if semantic_update:
                governor.submit(semantic_decision, context, now)
            selected, diagnostics = governor.select(candidates, context, now,
                        enabled=semantic_enabled, memory_evidence=memory_evidence)
            if self.semantic_policy == "shadow":
                shadow_selected = diagnostics.get("selected_id")
                shadow_candidates = diagnostics["candidates"]
                selected, baseline_diagnostics = choose_candidate(candidates, None, False)
                diagnostics.update(candidates=baseline_diagnostics["candidates"],
                    selected_id=baseline_diagnostics["selected_id"], shadow_selected_id=shadow_selected,
                    shadow_candidates=shadow_candidates, shadow_only=True)
                diagnostics["governor"]["would_adopt"] = diagnostics["governor"]["adopted"]
                diagnostics["governor"]["adopted"] = False
            # Query scheduling happens at actual cognition checkpoints, not on
            # every 5-Hz plan (which would consume a query without calling it).
        self.last_context, self.last_candidates = context, candidates
        diagnostics.update(semantic_policy=getattr(self, "semantic_policy", "always"),
                           planner_context=context, query_decision=self.last_query_decision,
                           planning_memory_mode=getattr(self, "planning_memory_mode", "off"),
                           memory_evidence=memory_evidence, requested_action_memory=action_memory)
        diagnostics.update(backend="pdm_closed", provenance=self.provenance,
                           timestamp_us=timestamp, horizon_s=self.samples * self.dt, sample_interval_s=self.dt,
                           lqr_longitudinal_horizon_s=1.0,
                           lqr_longitudinal_horizon_samples=10,
                           lqr_lateral_horizon_s=self.lqr_lateral_horizon_samples * self.dt,
                           lqr_lateral_horizon_samples=self.lqr_lateral_horizon_samples,
                           lqr_lateral_velocity_forecast="finite_forward_constant_acceleration_stop_at_zero",
                           leading_agent_update_rate=1, lateral_transition="cubic_hermite_pose_tangent",
                           excluded_lateral_paths=excluded, traffic_stop_lines=len(stop_polygons),
                           geometry=self._geometry_diagnostics,
                           steering_state=self._steering_state_diagnostics,
                           native_red_light_idm=True,
                           idm_integration="substeps_max_0.01s_nonnegative_trapezoid",
                           actors=[{"id": str(actor.track_token), "x": actor.center.x, "y": actor.center.y,
                                    "heading": actor.center.heading, "vx": actor.velocity.x, "vy": actor.velocity.y,
                                    "length": actor.box.length, "width": actor.box.width}
                                   for actor in actors.tracked_objects.tracked_objects],
                           initial_collision_tokens=initial_collisions,
                           vehicle={"wheelbase_m": self.vehicle_parameters.wheel_base,
                                    "max_steer_rad": self.max_steer,
                                    "physical_inner_max_steer_rad": self.physical_max_steer,
                                    "front_track_m": self.front_track_m,
                                    "steering_mapping": self.steering_model.mapping,
                                    "width_m": self.vehicle_parameters.width,
                                    "length_m": self.vehicle_parameters.length,
                                    "bounding_box_offset_xyz": [
                                        self.ego.bounding_box.location.x, self.ego.bounding_box.location.y,
                                        self.ego.bounding_box.location.z],
                                    "rear_axle_to_center_m": self.vehicle_parameters.rear_axle_to_center})
        diagnostics["vehicle"].update(self._vehicle_geometry_source)
        diagnostics["selected_zero_base_score"] = selected is not None and selected["base_score"] == 0.0
        # Selection is score-only. No separate collision/TTC detector replaces
        # the winner; only numerically invalid candidates cannot be executed.
        if selected is None:
            diagnostics.update(emergency_brake=True, failure_reason="no_finite_candidate")
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
                               "optional_heading_rad": math.atan2(math.sin(center.heading + yaw),
                                                                  math.cos(center.heading + yaw)),
                               "optional_v": float(max(0, idm_velocity[i]))})
        diagnostics.update(emergency_brake=False, selected_lateral=selected["lateral"],
                           selected_longitudinal=selected["longitudinal"])
        return {"trajectory": trajectory, "target_speed": trajectory[-1]["optional_v"],
                "selected_lateral": selected["lateral"], "selected_longitudinal": selected["longitudinal"],
                "planner_diagnostics": diagnostics}

    def query_decision(self, sim_time_s):
        if self.governor is None:
            return {"query": True, "reason": "always_policy"}
        if self.last_context is None:
            return {"query": False, "reason": "no_current_context"}
        evidence = self.experience_memory.retrieve(self.last_context) if self.experience_memory else None
        self.last_query_decision = self.governor.query_decision(
            self.last_context, self.last_candidates, sim_time_s, memory_evidence=evidence)
        return self.last_query_decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pinned PDM core imports without CARLA or VLM")
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate_dependencies(), indent=2, ensure_ascii=False))
