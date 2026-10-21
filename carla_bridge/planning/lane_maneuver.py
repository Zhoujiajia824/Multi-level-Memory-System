"""Stateful, map-anchored numerical overtaking proposals for PDM and MPC.

The current snapshot grants a proposal, never a right of way.  The numerical
backend evaluates its actual rollout and current traffic.  Neither a model call
nor a semantic CHANGE_LANE bonus is needed.  Fixed spatial transitions avoid
restarting lane changes at 5 Hz; a shared return reservation preserves room to
complete the entire excursion while traffic blocks the return.
"""
from __future__ import annotations

import math
import hashlib
from copy import deepcopy
from numbers import Real

import numpy as np

from .scene_context import lane_key, project_polyline


MANEUVER_GEOMETRY_VERSION = "fixed_phase_reference_stable_stations_v3"


DEFAULTS = {"enabled": True, "observation_s": 1.0, "minimum_gap_m": 1.0,
            "maximum_blocker_distance_m": 45.0, "slow_speed_fraction": .6,
            "maximum_pass_speed_mps": 5.0, "lateral_acceleration_mps2": 1.4,
            "return_clearance_m": 8.0, "maximum_maneuver_s": 45.0}


def validate_maneuver_config(value=None):
    result = dict(DEFAULTS)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(DEFAULTS):
            raise ValueError("lane_maneuver must contain only known settings")
        result.update(value)
    if type(result["enabled"]) is not bool:
        raise ValueError("lane_maneuver.enabled must be boolean")
    for key in set(DEFAULTS) - {"enabled"}:
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("lane_maneuver." + key + " must be positive and finite")
    if not 0 < result["slow_speed_fraction"] < 1:
        raise ValueError("lane_maneuver.slow_speed_fraction must be between zero and one")
    if result["maximum_pass_speed_mps"] > 8 or result["lateral_acceleration_mps2"] > 2:
        raise ValueError("lane maneuver must remain a low-speed numerical proposal")
    return result


def _xy(wp):
    p = wp.transform.location
    return np.array([p.x, -p.y], dtype=float)


def _adjacent(wp, side):
    return wp if side == "center" else (wp.get_left_lane() if side == "left" else wp.get_right_lane())


def _same_direction(a, b):
    return math.cos(math.radians(a.transform.rotation.yaw - b.transform.rotation.yaw)) > .8


def _connected(a, b):
    distance = float(np.linalg.norm(_xy(b) - _xy(a)))
    if distance > 6 or not _same_direction(a, b):
        return False
    return lane_key(a) == lane_key(b) or any(
        lane_key(wp) == lane_key(b) and np.linalg.norm(_xy(wp) - _xy(b)) < 1
        for wp in a.next(max(.05, distance)))


def lane_change_permitted(source, target, side):
    """Directional topology permission shared by every adapter lane change.

    This governs which numerical paths can be constructed. It does not add a
    trajectory veto or modify any score returned by the upstream PDM scorer.
    """
    import carla
    flag = carla.LaneChange.Left if side == "left" else carla.LaneChange.Right
    if (source is None or target is None or source.is_junction or target.is_junction
            or source.lane_type != carla.LaneType.Driving or target.lane_type != carla.LaneType.Driving
            or not _same_direction(source, target) or not int(source.lane_change) & int(flag)):
        return False
    # CARLA's lane_change handles directional SolidBroken/BrokenSolid markings.
    # An unambiguously solid boundary must never be overruled by a broad flag.
    marking = getattr(source, side + "_lane_marking", None)
    kind = str(getattr(marking, "type", "unknown")).split(".")[-1]
    if kind in ("Solid", "SolidSolid", "Curb", "Grass"):
        return False
    allowed = getattr(marking, "lane_change", None)
    return allowed is None or bool(int(allowed) & int(flag))


# Keep the former private name usable for downstream diagnostic scripts.
_permission = lane_change_permitted


def _quintic(value):
    value = np.clip(value, 0., 1.)
    return 10 * value ** 3 - 15 * value ** 4 + 6 * value ** 5


def _merge_station_samples(samples, required, minimum_separation_m=1e-3):
    """Coalesce redundant sampling locations, preserving transition anchors.

    CARLA float map knots can differ from a regular sample by nanometres.
    Keeping both makes a second numerical derivative ill-conditioned even on
    an exactly straight Hermite spline. This changes only the sampling set,
    never the map interpolant, curve shape or curvature/steering bound.
    Required anchors are retained in the supplied priority order; a redundant
    support endpoint within a millimetre cannot displace a transition anchor.
    """
    anchors = []
    for value in required:
        if all(abs(value - previous) >= minimum_separation_m for previous in anchors):
            anchors.append(float(value))
    anchors = np.asarray(anchors)
    samples = np.unique(np.asarray(samples, dtype=float))
    if len(anchors):
        samples = samples[np.min(abs(samples[:, None] - anchors[None, :]), axis=1)
                          >= minimum_separation_m]
    kept = []
    for value in samples:
        if not kept or value - kept[-1] >= minimum_separation_m:
            kept.append(float(value))
    return np.sort(np.r_[anchors, kept])


def _curvature(points):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    first = np.gradient(points, arc, axis=0)
    second = np.gradient(first, arc, axis=0)
    denominator = np.maximum(np.sum(first * first, axis=1) ** 1.5, 1e-8)
    return np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator


def _curvature_limit(ego, speed_mps=0., *, physics=None):
    """Equivalent bicycle steering bound from CARLA's measured wheel geometry.

    This is a proposal-generation bound, not a substitute for downstream
    dynamics/rollout. With an old actor provider use a conservative fallback.
    """
    getter = getattr(ego, "get_physics_control", None)
    if physics is None and not callable(getter):
        return .18
    physics = getter() if physics is None else physics
    wheels = physics.wheels
    if len(wheels) < 4:
        return .18
    front = np.mean([[w.position.x, w.position.y, w.position.z] for w in wheels[:2]], axis=0)
    rear = np.mean([[w.position.x, w.position.y, w.position.z] for w in wheels[-2:]], axis=0)
    wheelbase = float(np.linalg.norm(front-rear)*.01)
    track = float(np.linalg.norm(np.array([wheels[0].position.x, wheels[0].position.y,
                                          wheels[0].position.z])-np.array([
        wheels[1].position.x, wheels[1].position.y, wheels[1].position.z]))*.01)
    maximum = math.radians(min(float(w.max_steer_angle) for w in wheels[:2]))
    if not 1 < wheelbase < 6 or not .5 < track < 4 or not 0 < maximum < math.pi/2:
        return .18
    # PhysX specifies inner-wheel maximum. CARLA's steering curve reduces it
    # with speed (curve abscissas are km/h). Retain tracking headroom too.
    from carla_bridge.control.steering_model import SteeringModel
    model = SteeringModel(maximum,
        [(point.x, point.y) for point in getattr(physics, "steering_curve", ())],
        wheelbase_m=wheelbase, front_track_m=track)
    return .9*math.tan(model.limit_at(speed_mps))/wheelbase


def _actor_data(actor, reference_points, ego_z, motion_observation=None):
    p = actor.get_location()
    velocity = (actor.get_velocity() if motion_observation is None
                else motion_observation.velocity)
    if abs(p.z - ego_z) > 2.5:
        return None
    projection = project_polyline(p.x, -p.y, reference_points)
    if projection is None:
        return None
    yaw = -math.radians(actor.get_transform().rotation.yaw)
    tangent = projection[2]
    extent = actor.bounding_box.extent
    velocity_xy = np.array([velocity.x, -velocity.y], dtype=float)
    return {"id": int(actor.id), "type": actor.type_id, "xy": np.array([p.x, -p.y]),
            "velocity": velocity_xy, "station": projection[0], "lateral": projection[1],
            "along_speed": float(velocity_xy @ np.array([math.cos(tangent), math.sin(tangent)])),
            "cross_speed": float(velocity_xy @ np.array([-math.sin(tangent), math.cos(tangent)])),
            "aligned": math.cos(yaw - tangent) > .8,
            "half_length": float(extent.x), "half_width": float(extent.y),
            "speed": float(np.linalg.norm(velocity_xy)),
            "motion_source": getattr(motion_observation, "source", "actor_api_velocity")}


def _crosswalk_near(carla_map, xy):
    """Map crosswalks explain a stationary pedestrian without invented intent."""
    getter = getattr(carla_map, "get_crosswalks", None)
    if not callable(getter):
        return False
    from shapely.geometry import Point, Polygon
    current = []
    for p in getter():
        value = (float(p.x), float(-p.y))
        current.append(value)
        if len(current) >= 4 and np.linalg.norm(np.asarray(current[0]) - value) < .05:
            polygon = Polygon(current)
            if polygon.is_valid and polygon.buffer(2.).covers(Point(*xy)):
                return True
            current = []
    return False


class LaneManeuver:
    """One measured obstacle, one committed lane excursion, then a legal return.

    ``update`` is side-effect free with respect to the simulator. ``commit_selected``
    must be called only for the candidate actually executed, after shadow has
    restored its baseline. The returned points use the reflected PDM coordinate
    frame. No private future scenario schedule is read.
    """

    def __init__(self, config=None):
        self.config = validate_maneuver_config(config)
        self._observed = None
        self._active = None
        self._pending = None
        self._last_now = None
        self._counter = 0
        self._reference_source = None
        self._extended_reference = None
        self._last_reference = None
        self._last_stations = None
        self._last_update_time = None
        self._last_ego_half_width = None
        self.last_commit_audit = {"committed": False, "reason": "no_executed_candidate"}

    def _planning_reference(self, reference):
        """Retain stable global stations while removing a false route-end leader.

        The evaluation route is not a physical lane ending. Ordinary PDM paths
        extend along real map successors; an overtaking path needs the same
        planning tail or IDM brakes solely because its proposal is shorter.
        This extension never grants permission to pass a real junction seam.
        """
        if self._reference_source is reference and self._extended_reference is not None:
            return self._extended_reference
        points, tail = list(reference), 0.
        while points and tail < 80.:
            previous = points[-1]
            if previous.is_junction:
                break
            options = [wp for wp in previous.next(2.) if _same_direction(previous, wp)]
            if not options:
                break
            chosen = min(options, key=lambda wp: abs(math.atan2(
                math.sin(math.radians(wp.transform.rotation.yaw - previous.transform.rotation.yaw)),
                math.cos(math.radians(wp.transform.rotation.yaw - previous.transform.rotation.yaw)))))
            distance = float(np.linalg.norm(_xy(chosen) - _xy(previous)))
            if distance < .1 or not _connected(previous, chosen):
                break
            points.append(chosen)
            tail += distance
        self._reference_source, self._extended_reference = reference, points
        return points

    def _inactive(self, reason, **details):
        self._pending = None
        return {"active": False, "phase": "idle", "points": None, "nominal_points": None, "metadata": None,
                "target_side": None, "current_target_side": None, "speed_cap_mps": None,
                "diagnostics": dict(reason=reason, phase="idle", committed=False,
                                    target_side=None, **details)}

    @staticmethod
    def _signal_wait(world, ego, reference, points, station):
        """Signal causality also works in model-off/legacy context-free runs."""
        keys = {lane_key(wp) for wp in reference}
        for actor in world.get_actors():
            if not actor.type_id.startswith("traffic.traffic_light"):
                continue
            if str(actor.state).split(".")[-1] not in ("Red", "Yellow"):
                continue
            for wp in actor.get_stop_waypoints():
                p = wp.transform.location
                projection = project_polyline(p.x, -p.y, points)
                if (lane_key(wp) in keys and abs(p.z - ego.get_location().z) < 2.5
                        and projection is not None and projection[1] <= wp.lane_width / 2 + .2
                        and -1 <= projection[0] - station <= 80):
                    return {"signal_id": int(actor.id), "distance_m": projection[0] - station}
        return None

    @staticmethod
    def _samples(reference, side, stations):
        """Interpolate map position/tangent, not a chord-shifted fake lane."""
        from scipy.interpolate import CubicHermiteSpline
        points = np.array([_xy(_adjacent(wp, side)) for wp in reference])
        headings = -np.radians([_adjacent(wp, side).transform.rotation.yaw for wp in reference])
        tangents = np.column_stack((np.cos(headings), np.sin(headings)))
        return CubicHermiteSpline(stations, points, tangents, axis=0)

    def _corridor(self, reference, stations, start, end, side, returning=False):
        indices = [i for i, s in enumerate(stations) if start - 2 <= s <= end + 2]
        if len(indices) < 2:
            return False, "insufficient_map_samples"
        for i in indices:
            base, adjacent = reference[i], _adjacent(reference[i], side)
            source, target = (adjacent, base) if returning else (base, adjacent)
            direction = ("right" if side == "left" else "left") if returning else side
            if not lane_change_permitted(source, target, direction):
                return False, "return_not_permitted" if returning else "lane_change_not_permitted"
            if i > indices[0]:
                previous = _adjacent(reference[i - 1], side)
                if previous is None or not _connected(previous, adjacent):
                    return False, "disconnected_adjacent_lane"
        return True, None

    def _build(self, reference, stations, start, length, side, end, *, returning=False):
        valid = [i for i, s in enumerate(stations)
                 if start - 6 <= s <= end + 2 and _adjacent(reference[i], side) is not None]
        if len(valid) < 3:
            return None
        first, last = valid[0], valid[-1]
        subset, knots = reference[first:last + 1], stations[first:last + 1]
        if any(_adjacent(wp, side) is None for wp in subset):
            return None
        center = self._samples(subset, "center", knots)
        adjacent = self._samples(subset, side, knots)
        last_sample = min(end, knots[-1])
        sample_s = np.r_[knots, np.arange(knots[0], last_sample, .25)]
        sample_s = sample_s[(sample_s >= knots[0]) & (sample_s <= last_sample)]
        required = [value for value in (start, start + length, last_sample, knots[0])
                    if knots[0] <= value <= last_sample]
        sample_s = _merge_station_samples(sample_s, required)
        blend = _quintic((sample_s - start) / length)
        if returning:
            blend = 1 - blend
        points = center(sample_s) + blend[:, None] * (adjacent(sample_s) - center(sample_s))
        widths = np.interp(sample_s, stations, [wp.lane_width for wp in reference])
        return points, sample_s, widths

    @staticmethod
    def _prediction_speeds(ego, cap, candidate_speed_mps):
        """Cover both the capped proposal and ordinary uncapped IDM options.

        These are present-evidence CV proxies, not future ego ground truth or
        speed constraints on any executed proposal. Preserve the previous cap
        probe as well as the ordinary upper-speed probe: replacing one by the
        other could miss an interaction occurring at the slower arrival time.
        """
        measured = math.hypot(ego.get_velocity().x, ego.get_velocity().y)
        upper = max(cap, measured, candidate_speed_mps)
        return tuple(sorted({cap, max(cap, measured), upper})), min(cap, measured)

    def _gap_clear(self, actors, points, samples, ego_station, ego, cap, *, blocker_id=None,
                   candidate_speed_mps=None, returning=False):
        """Current front/rear envelopes and same-time constant-velocity occupancy."""
        half_l, half_w = float(ego.bounding_box.extent.x), float(ego.bounding_box.extent.y)
        direction = np.gradient(points, axis=0)
        yaw = np.arctan2(direction[:, 1], direction[:, 0])
        # Restrict the check to the traversed local segment, not a distant future
        # lane where a vehicle may safely be following well after this horizon.
        if candidate_speed_mps is None:
            candidate_speed_mps = min(8., float(ego.get_speed_limit()) / 3.6)
        speeds, rear_speed = self._prediction_speeds(ego, cap, candidate_speed_mps)
        if returning:
            measured = math.hypot(ego.get_velocity().x, ego.get_velocity().y)
            # Cover the bounded ordinary slow policy and a measured crawl as
            # well as cruise.  A slower arrival must not hide crossing traffic.
            speeds = tuple(sorted(set(speeds) | {measured, cap * .55}))
        for actor in actors:
            initial = project_polyline(*actor["xy"], points)
            if initial is not None and initial[1] < half_w + actor["half_width"] + .35:
                idx = int(np.argmin(abs(samples - actor["station"])))
                if np.linalg.norm(points[idx] - actor["xy"]) < half_w + actor["half_width"] + .5:
                    gap = actor["station"] - ego_station
                    if actor["id"] != blocker_id and gap < 0:
                        rear_gap = -gap - half_l - actor["half_length"]
                        pedestrian = actor.get("type", "").startswith("walker.")
                        minimum = 2. if returning and pedestrian else 8.
                        reserve = 2. if returning and pedestrian else 5.
                        required = max(minimum, 3 * max(0., actor["along_speed"] - rear_speed) + reserve)
                        if rear_gap < required:
                            return False, {"reason": "target_rear_gap", "actor_id": actor["id"],
                                           "gap_m": rear_gap, "required_m": required}
            for speed in speeds:
                for t in np.arange(0., 6.01, .25):
                    station = min(samples[-1], ego_station + speed * t)
                    ex, ey = np.interp(station, samples, points[:, 0]), np.interp(station, samples, points[:, 1])
                    tangent = float(np.interp(station, samples, np.unwrap(yaw)))
                    delta = actor["xy"] + actor["velocity"] * t - np.array([ex, ey])
                    longitudinal = abs(delta @ np.array([math.cos(tangent), math.sin(tangent)]))
                    lateral = abs(delta @ np.array([-math.sin(tangent), math.cos(tangent)]))
                    # Orient the other's footprint against the planned tangent.
                    other_long = actor["half_length"] if actor["aligned"] else max(actor["half_length"], actor["half_width"])
                    if longitudinal < half_l + other_long + 1. and lateral < half_w + actor["half_width"] + .35:
                        return False, {"reason": "predicted_occupied_envelope", "actor_id": actor["id"],
                                       "time_s": float(t), "ego_probe_speed_mps": float(speed)}
        return True, None

    @staticmethod
    def _target_gap_clear(actors, reference, stations, station, ego, side, cap, *,
                          candidate_speed_mps=None, returning=False):
        """Check the target lane itself, including behind the initial curve.

        Testing only the curved ego path misses a rapidly closing rear vehicle:
        at t=0 the curve is still in ego's original lane.
        """
        available = [i for i, wp in enumerate(reference) if _adjacent(wp, side) is not None]
        if len(available) < 2:
            return False, {"reason": "no_target_lane_geometry"}
        if candidate_speed_mps is None:
            candidate_speed_mps = min(8., float(ego.get_speed_limit()) / 3.6)
        speeds, rear_speed = LaneManeuver._prediction_speeds(ego, cap, candidate_speed_mps)
        front_speed = max(speeds)
        xy = np.array([_xy(_adjacent(reference[i], side)) for i in available])
        knots = stations[available]
        for actor in actors:
            target = np.array([np.interp(actor["station"], knots, xy[:, 0]),
                               np.interp(actor["station"], knots, xy[:, 1])])
            if np.linalg.norm(target - actor["xy"]) > ego.bounding_box.extent.y + actor["half_width"] + .35:
                continue
            gap = actor["station"] - station
            free_gap = abs(gap) - ego.bounding_box.extent.x - actor["half_length"]
            if gap < 0:
                pedestrian = actor.get("type", "").startswith("walker.")
                minimum = 2. if returning and pedestrian else 8.
                reserve = 2. if returning and pedestrian else 5.
                required = max(minimum, 3 * max(0., actor["along_speed"] - rear_speed) + reserve)
                kind = "target_rear_gap"
            else:
                required = max(6., 1.5 * front_speed, 3 * max(0., front_speed - actor["along_speed"]))
                kind = "target_front_gap"
            if free_gap < required:
                return False, {"reason": kind, "actor_id": actor["id"],
                               "gap_m": float(free_gap), "required_m": float(required)}
        return True, None

    def _return_reservation(self, plan, reference, stations, station, ego):
        """A longitudinal limit before the last *legal* return transition.

        The final adjacent-lane endpoint is too late to wait: a full spatial
        merge still has to fit before it.  Publish a world point so every
        adapter can apply this same limit to ordinary and maneuver proposals.
        The reserve includes ego's front overhang and two metres of map slack.
        """
        last_start = plan["corridor_end"] - plan["length"] - max(4., ego.bounding_box.extent.x + 2.)
        for start in np.arange(last_start, max(plan["anchor"], last_start - 40.) - .01, -1.):
            legal, _ = self._corridor(reference, stations, float(start),
                                      float(start) + plan["length"], plan["side"], returning=True)
            if legal:
                last_start = float(start)
                break
        stop = min(last_start, plan.get("return_wait_station", last_start))
        plan["return_wait_station"] = stop  # once reserved, never creep it forward
        stop_xy = [float(np.interp(stop, plan["sample_s"], plan["points"][:, axis]))
                   for axis in (0, 1)]
        return {"return_required": True, "return_deadline_route_station_m": last_start,
            "maneuver_stop_route_station_m": stop, "maneuver_stop_world_xy": stop_xy,
            "maneuver_stop_reason": "preserve_legal_return_window",
            "return_wait_route_station_m": stop, "return_wait_world_xy": stop_xy,
            "return_window_remaining_m": last_start - station,
            "return_stop_already_passed": bool(station > stop + .5)}

    def _return_candidate(self, plan, actors, reference, stations, station, ego):
        """Search bounded merge speeds; a fast ordinary policy is not evidence
        that this capped candidate would travel at that unrelated speed.

        Current measured speed is retained in the envelope, and the downstream
        planner still checks its actual dynamic rollout.  A candidate grants no
        right of way and never bypasses topology or a closing rear vehicle.
        """
        pending = plan.get("return_proposal")
        lengths = ([pending["length"]] if pending is not None else
                   list(dict.fromkeys([plan["length"], 24., 20., 16., 12., 10., 9., 8., 7., 6.])))
        attempts = []
        physics = ego.get_physics_control() if callable(getattr(ego, "get_physics_control", None)) else None
        for length in lengths:
            anchor = pending["anchor"] if pending is not None else station
            legal, reason = self._corridor(reference, stations, anchor, anchor + length,
                                          plan["side"], returning=True)
            if not legal or anchor + length + 4. > plan["corridor_end"]:
                attempts.append({"length_m": length, "reason": reason or "no_complete_legal_return_corridor"})
                continue
            built = ((pending["points"], pending["sample_s"], pending["widths"]) if pending is not None else
                     self._build(reference, stations, anchor, length, plan["side"],
                                 min(stations[-1], anchor + 90.), returning=True))
            if built is None:
                attempts.append({"length_m": length, "reason": "path_construction_failed"})
                continue
            kmax = max(float(np.max(_curvature(built[0]))), .001)
            cap = min(plan["speed_cap"], math.sqrt(self.config["lateral_acceleration_mps2"] / kmax))
            caps = sorted({float(cap), min(float(cap), 3.), min(float(cap), 2.),
                           min(float(cap), 1.)}, reverse=True)
            for cap in caps:
                if cap < .5 or kmax > _curvature_limit(ego, cap, physics=physics):
                    attempts.append({"length_m": length, "speed_cap_mps": cap,
                                     "reason": "steering_curvature_unreachable"})
                    continue
                clear, conflict = self._target_gap_clear(actors, reference, stations, station,
                    ego, "center", cap, candidate_speed_mps=cap, returning=True)
                if clear:
                    clear, conflict = self._gap_clear(actors, built[0], built[1], station, ego, cap,
                                                      candidate_speed_mps=cap, returning=True)
                if clear:
                    updated = dict(plan, points=built[0], nominal_points=built[0].copy(),
                        sample_s=built[1], widths=built[2], anchor=anchor, length=float(length),
                        phase="returning", speed_cap=cap)
                    updated.pop("return_proposal", None)
                    for name in ("maneuver_stop_route_station_m", "maneuver_stop_world_xy", "maneuver_stop_reason"):
                        updated.pop(name, None)
                    updated["return_checks"] = {"allowed": True, "attempts": attempts,
                        "speed_envelope": "candidate_cap_and_measured_speed", "speed_cap_mps": cap}
                    return updated, None
                attempts.append(dict(conflict, length_m=length, speed_cap_mps=cap))
        return None, {"reason": "return_currently_blocked", "attempts": attempts,
                      "speed_envelope": "candidate_cap_and_measured_speed"}

    @staticmethod
    def _observe_return_completion(plan, ego, reference, reference_points, stations,
                                   station, projection, now, current_side):
        """Release a return only after the measured complete body is stable.

        A center position can cross the route while the bonnet still crosses a
        boundary. Check the actual CARLA bbox, including its local offset and
        rotation, against the original connected lane surface. This is only a
        completion observation: it neither moves the executed anchor nor grants
        a new trajectory permission. Duplicate queries cannot add samples.
        """
        from shapely.geometry import MultiPoint, Polygon
        from shapely.ops import unary_union

        tf = ego.get_transform()
        aligned = math.cos(-math.radians(tf.rotation.yaw) - projection[2]) > .985
        end_reached = plan["phase"] != "returning" or station >= plan["anchor"] + plan["length"]
        first = max(0, int(np.searchsorted(stations, station - 6.)) - 1)
        last = min(len(reference), int(np.searchsorted(stations, station + 6.)) + 2)
        headings = -np.radians([wp.transform.rotation.yaw for wp in reference[first:last]])
        normals = np.c_[-np.sin(headings), np.cos(headings)]
        half_widths = np.array([wp.lane_width for wp in reference[first:last]]) / 2
        centers = reference_points[first:last]
        left, right = centers + half_widths[:, None] * normals, centers - half_widths[:, None] * normals
        surface = unary_union([Polygon([left[i], left[i+1], right[i+1], right[i]])
                               for i in range(len(centers)-1)])
        body = MultiPoint([(v.x, -v.y) for v in ego.bounding_box.get_world_vertices(tf)]).convex_hull
        body_inside = bool(not surface.is_empty and surface.is_valid and surface.covers(body))
        clearance = float(surface.boundary.distance(body)) if body_inside else 0.
        pose = {"current_side": current_side, "lateral_error_m": float(projection[1]),
                "heading_aligned": bool(aligned), "transition_end_reached": bool(end_reached),
                "complete_body_inside_original_lane": body_inside,
                "minimum_body_boundary_clearance_m": clearance,
                "required_body_boundary_clearance_m": .05}
        plan["return_completion_pose"] = pose
        entered = plan["phase"] == "returning" or plan.get("observed_passing_lane_at_s") is not None
        centered = (entered and end_reached and current_side == "center" and projection[1] < .35
                    and aligned and body_inside and clearance >= .05)
        if not centered:
            plan.pop("measured_return_observation", None)
            return None
        observed = plan.get("measured_return_observation")
        if observed is None or now - observed["last_time_s"] > .4 + 1e-8:
            observed = {"since_s": float(now), "samples": 0,
                "source": "measured_complete_body_in_original_lane_after_excursion"}
        distinct = now > observed.get("last_time_s", -math.inf) + 1e-8
        observed.update(last_time_s=float(now), samples=observed["samples"] + int(distinct), **pose)
        plan["measured_return_observation"] = observed
        if observed["samples"] >= 3 and now - observed["since_s"] >= .4 - 1e-8:
            return dict(observed, elapsed_s=float(now-observed["since_s"]),
                observed_passing_lane_at_s=plan.get("observed_passing_lane_at_s"),
                completion_phase=plan["phase"], dedicated_return_selection_claimed=False)
        return None

    def _result(self, plan, phase, *, committed, conflict=None):
        points = plan["points"]
        # This map-anchored target is immutable across executed within-lane
        # adjustments. Otherwise repeatedly offsetting the last winner would
        # accumulate drift and let that winner declare itself perfectly centred.
        nominal = plan.setdefault("nominal_points", points.copy())
        target = "center" if phase == "returning" else plan["side"]
        label = ("CHANGE_LANE_RIGHT" if plan["side"] == "left" else "CHANGE_LANE_LEFT") if phase == "returning" else "CHANGE_LANE_" + plan["side"].upper()
        identity = "maneuver_%s_%s" % (plan["serial"], phase)
        metadata = {"id": identity, "label": label, "semantics": [label],
            "physical_lane_side": target, "route_target": target == "center", "lane_offset_m": 0.,
            "maneuver": True, "maneuver_id": plan["serial"], "maneuver_phase": phase,
            "maneuver_blocker_id": plan["blocker_id"], "maneuver_preferred": True,
            "lane_quality_reference_points": nominal.tolist(), "lane_quality_widths": plan["widths"].tolist(),
            "lane_quality_transition": True}
        return_audit = {key: deepcopy(plan[key]) for key in (
            "return_required", "return_deadline_route_station_m", "maneuver_stop_route_station_m",
            "maneuver_stop_world_xy", "maneuver_stop_reason", "return_window_remaining_m",
            "return_wait_route_station_m", "return_wait_world_xy", "return_stop_already_passed",
            "maneuver_overdue", "return_trigger", "return_checks", "passing_checks", "pending_return_invalidation",
            "pending_return_invalidation_count", "measured_return_observation",
            "return_completion_pose") if key in plan}
        metadata.update(return_audit)
        self._pending = plan
        return {"active": True, "phase": phase, "target_side": target, "current_target_side": target,
                "points": points, "nominal_points": nominal, "metadata": metadata, "speed_cap_mps": plan["speed_cap"],
                "diagnostics": {"reason": "committed_maneuver" if committed else "persistent_slow_obstacle",
                    "geometry_version": MANEUVER_GEOMETRY_VERSION,
                    "committed": committed, "maneuver_id": plan["serial"], "blocker_id": plan["blocker_id"],
                    "side": plan["side"], "target_side": target, "phase": phase,
                    "executed_phase": self._active["phase"] if self._active is not None else None,
                    "anchor_station_m": plan["anchor"],
                    "transition_length_m": plan["length"], "speed_cap_mps": plan["speed_cap"],
                    "commit_source": plan.get("commit_source"),
                    "actual_selected_id": plan.get("actual_selected_id"),
                    "actual_selected_path_label": plan.get("actual_selected_path_label"),
                    "executed_geometry_sha256": plan.get("executed_geometry_sha256"),
                    "prediction_kind": "observed_velocity_proxy_not_future_GT", "conflict": conflict,
                    **return_audit}}

    def commit_selected(self, selected_id, now, *, selected_path_points=None, selected_metadata=None):
        """Commit only an executed proposal, including shadow's actual baseline."""
        self.last_commit_audit = {"committed": False, "reason": "executed_candidate_did_not_start_verified_maneuver",
                                  "actual_selected_id": selected_id}
        if (self._pending is not None and isinstance(selected_id, str)
                and selected_id.startswith("maneuver_%s_" % self._pending["serial"])):
            actual = np.asarray(self._pending["points"] if selected_path_points is None
                                else selected_path_points, dtype=float)
            if (actual.ndim != 2 or actual.shape[1:] != (2,) or len(actual) < 3
                    or not np.isfinite(actual).all()):
                self.last_commit_audit["reason"] = "invalid_executed_geometry"
                return False
            if self._active is None:
                self._pending["committed_at"] = float(now)
            returning = self._pending["phase"] == "returning"
            direction = ("right" if self._pending["side"] == "left" else "left") if returning else self._pending["side"]
            label = (selected_metadata or {}).get("label", "CHANGE_LANE_" + direction.upper())
            # Return proposals inherit the passing state. Provenance must name
            # this actual winner, never retain that earlier generic path/hash.
            digest = hashlib.sha256(actual.tobytes()).hexdigest()
            self._pending.update(commit_source="executed_fixed_numerical_candidate",
                actual_selected_id=selected_id, actual_selected_path_label=label,
                executed_geometry_sha256=digest, executed_geometry_point_count=len(actual),
                executed_selected_points=actual.copy())
            self._active = self._pending
            self.last_commit_audit.update(committed=True, reason="executed_fixed_numerical_candidate",
                commit_source="executed_fixed_numerical_candidate", actual_selected_id=selected_id,
                actual_selected_path_label=label, executed_geometry_sha256=digest,
                executed_geometry_point_count=len(actual), phase=self._pending["phase"],
                transition_length_m=float(self._pending["length"]))
            return True
        # A better ordinary native proposal may execute the very same verified
        # maneuver. Record that actual execution, rather than insisting our
        # additional capped proposal must win or fabricating its selection.
        if self._pending is None or selected_metadata is None or selected_path_points is None:
            return False
        if self._last_update_time is None or abs(float(now) - self._last_update_time) > 1e-6:
            self.last_commit_audit["reason"] = "stale_maneuver_evidence"
            return False
        pending, metadata = self._pending, selected_metadata
        returning = pending["phase"] == "returning"
        same_phase = self._active is not None and self._active["phase"] == pending["phase"]
        adjustment = metadata.get("maneuver_continuation_adjustment") is True
        if same_phase and not adjustment:
            return False  # An ordinary replan cannot move a committed anchor.
        target = "center" if returning else pending["side"]
        direction = ("right" if pending["side"] == "left" else "left") if returning else pending["side"]
        label = "CHANGE_LANE_" + direction.upper()
        if (metadata.get("maneuver") or metadata.get("label") not in (
                (label, "KEEP_LANE") if same_phase and adjustment else (label,))
                or metadata.get("physical_lane_side") != target
                or bool(metadata.get("route_target")) != returning):
            return False
        length = metadata.get("transition_length_m")
        if (not isinstance(length, Real) or isinstance(length, bool)
                or not math.isfinite(length) or not 6 <= length <= 60):
            self.last_commit_audit["reason"] = "executed_transition_has_no_verified_length"
            return False
        if same_phase and abs(length - pending["length"]) > 1e-6:
            self.last_commit_audit["reason"] = "adjustment_cannot_restart_transition"
            return False
        legal, reason = self._corridor(self._last_reference, self._last_stations,
            pending["anchor"], pending["anchor"] + length, pending["side"], returning=returning)
        if not legal:
            self.last_commit_audit.update(reason="executed_transition_outside_verified_corridor", detail=reason)
            return False
        if not returning:
            return_start = max(pending["anchor"] + length,
                pending["blocker_station"] + pending["blocker_half_length"] + self.config["return_clearance_m"] + 3.)
            return_legal, reason = self._corridor(self._last_reference, self._last_stations,
                return_start, return_start + length, pending["side"], returning=True)
            if not return_legal or return_start + length + 4 > pending["corridor_end"]:
                self.last_commit_audit.update(reason="executed_transition_leaves_no_legal_return", detail=reason)
                return False
        points = np.asarray(selected_path_points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3 or not np.isfinite(points).all():
            self.last_commit_audit["reason"] = "invalid_executed_geometry"
            return False
        nominal = pending.get("nominal_points", pending["points"])
        if adjustment:
            from shapely.geometry import LineString, Point
            offset, declared_limit = metadata.get("lane_offset_m"), metadata.get("maneuver_adjustment_limit_m")
            physical_limit = min(.5, max(0., (min(pending["widths"])
                - 2 * (self._last_ego_half_width or 0.)) / 2 - .1))
            if (not isinstance(offset, Real) or isinstance(offset, bool) or not math.isfinite(offset)
                    or not isinstance(declared_limit, Real) or isinstance(declared_limit, bool)
                    or not math.isfinite(declared_limit)
                    or not 0 < declared_limit <= physical_limit + 1e-6
                    or abs(offset) > declared_limit + 1e-6):
                self.last_commit_audit["reason"] = "adjustment_outside_target_lane_budget"
                return False
            curve = LineString(nominal)
            if max(curve.distance(Point(*point)) for point in points) > declared_limit + .05:
                self.last_commit_audit["reason"] = "adjustment_not_bounded_to_fixed_nominal"
                return False
        first = max(0, int(np.searchsorted(self._last_stations, pending["anchor"] - 12)) - 1)
        last = min(len(self._last_reference), int(np.searchsorted(self._last_stations, pending["anchor"] + 160)) + 2)
        reference_xy = np.array([_xy(wp) for wp in self._last_reference[first:last]])
        samples = []
        for point in points:
            projection = project_polyline(*point, reference_xy)
            value = projection[0]
            if value <= 1e-8:
                first_delta = reference_xy[1] - reference_xy[0]
                first_delta /= np.linalg.norm(first_delta)
                value = min(0., float((point - reference_xy[0]) @ first_delta))
            samples.append(value + self._last_stations[first])
        samples = np.asarray(samples)
        if np.any(np.diff(samples) < -0.1):
            self.last_commit_audit["reason"] = "executed_geometry_has_ambiguous_route_station"
            return False
        keep = np.r_[True, np.diff(samples) > 1e-4]
        preserved = points.copy()
        points, samples = points[keep], samples[keep]
        if len(points) < 3:
            self.last_commit_audit["reason"] = "executed_geometry_has_insufficient_progress"
            return False
        built = self._build(self._last_reference, self._last_stations, pending["anchor"], length,
            pending["side"], pending["corridor_end"] - 2, returning=returning)
        if built is not None:
            tail = built[1] > samples[-1] + .1
            points, samples = np.r_[points, built[0][tail]], np.r_[samples, built[1][tail]]
        widths = np.interp(samples, self._last_stations, [wp.lane_width for wp in self._last_reference])
        updated = dict(pending, points=points, nominal_points=np.asarray(nominal).copy(),
            sample_s=samples, widths=widths, length=float(length),
            committed_at=float(now) if self._active is None else self._active["committed_at"],
            commit_source="executed_generic_numerical_candidate", actual_selected_id=selected_id,
            actual_selected_path_label=metadata["label"],
            executed_geometry_sha256=hashlib.sha256(preserved.tobytes()).hexdigest(),
            executed_geometry_point_count=len(preserved), executed_selected_points=preserved)
        updated.pop("return_proposal", None)
        self._active = updated
        self.last_commit_audit.update(committed=True,
            reason="executed_generic_numerical_adjustment" if adjustment else "executed_generic_numerical_candidate",
            commit_source=updated["commit_source"], actual_selected_id=selected_id,
            actual_selected_path_label=metadata["label"], executed_geometry_sha256=updated["executed_geometry_sha256"],
            executed_geometry_point_count=len(preserved), transition_length_m=float(length), phase=updated["phase"])
        if adjustment:
            self.last_commit_audit.update(adjustment_offset_m=float(offset),
                nominal_geometry_sha256=hashlib.sha256(np.asarray(nominal).tobytes()).hexdigest())
        return True

    def update(self, world, ego, reference, now, *, context=None, current_side="center", route_index=None,
               candidate_speed_mps=None, motion_history=None):
        if not self.config["enabled"]:
            return self._inactive("disabled")
        if not math.isfinite(float(now)):
            raise ValueError("maneuver timestamp must be finite")
        if candidate_speed_mps is None:
            candidate_speed_mps = min(8., float(ego.get_speed_limit()) / 3.6)
        if (type(candidate_speed_mps) not in (int, float) or not math.isfinite(candidate_speed_mps)
                or candidate_speed_mps < 0):
            raise ValueError("ordinary candidate speed must be finite and nonnegative")
        if self._last_now is not None and (now < self._last_now or now - self._last_now > 1.):
            self._observed = None
            # A teleport/restart invalidates the anchored geometry, never reuse it.
            self._active = None
        self._last_now = float(now)
        self._last_ego_half_width = float(ego.bounding_box.extent.y)
        if len(reference) < 3:
            return self._inactive("no_resolved_route")
        evaluation_count = len(reference)
        reference = self._planning_reference(reference)
        reference_points = np.array([_xy(wp) for wp in reference])
        stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference_points, axis=0), axis=1))]
        self._last_reference, self._last_stations, self._last_update_time = reference, stations, float(now)
        if np.any(np.diff(stations) <= 1e-5):
            return self._inactive("duplicate_route_samples")
        location = ego.get_location()
        # Ordered-route station must not jump to a crossing/loop merely because
        # its Euclidean distance is a centimetre smaller. The caller's retained
        # route index is the same sequence cursor used for route completion.
        cursor = None if route_index is None else min(len(reference) - 1, max(0, int(route_index)))
        first = 0 if cursor is None else max(0, cursor - 4)
        last = (len(reference) if cursor is None or cursor >= evaluation_count - 2
                else min(len(reference), cursor + 6))
        projection = project_polyline(location.x, -location.y, reference_points[first:last])
        projection = (projection[0] + stations[first], projection[1], projection[2])
        station, _, _ = projection
        nearest = int(np.argmin(abs(stations - station)))
        actor_first, actor_last = max(0, nearest - 40), min(len(reference), nearest + 70)
        actor_points = reference_points[actor_first:actor_last]
        actors = []
        for actor in world.get_actors():
            if actor.id != ego.id and actor.type_id.startswith(("vehicle.", "walker.pedestrian.")):
                observation = motion_history.motion_for(actor) if motion_history is not None else None
                value = _actor_data(actor, actor_points, location.z, observation)
                if value is not None:
                    value["station"] += stations[actor_first]
                if value is not None and abs(value["station"] - station) < 100:
                    actors.append(value)
        # Signal causality is not interpreted as a permission to pass a queue.
        signal = self._signal_wait(world, ego, reference[actor_first:actor_last], actor_points,
                                   station - stations[actor_first])
        waiting = bool((context or {}).get("traffic_wait", {}).get("active"))
        if self._active is not None:
            plan = self._active
            # A clock deadline cannot cancel the physical obligation to return.
            # Keep the verified lane and stopping reservation while waiting for
            # traffic, rather than forgetting the excursion after 45 seconds.
            plan["maneuver_overdue"] = bool(now - plan["committed_at"] > self.config["maximum_maneuver_s"])
            # Detect an actual observation jump, not accumulated progress since
            # the original anchor. A long, continuously observed excursion
            # still owes its return, including after the maneuver time budget.
            previous_station = plan.get("last_observed_route_station_m", plan["anchor"])
            previous_time = plan.get("last_observed_time_s", plan["committed_at"])
            if now > previous_time + 1e-8:
                if abs(station - previous_station) > 150:
                    self._active = self._observed = None
                    return self._inactive("route_discontinuity_single_observation_station_jump",
                        previous_route_station_m=float(previous_station), current_route_station_m=float(station),
                        observation_station_jump_m=float(station-previous_station),
                        original_anchor_station_m=float(plan["anchor"]))
                plan.update(last_observed_route_station_m=float(station), last_observed_time_s=float(now))
            if plan["phase"] == "passing":
                # A native ordinary center path may return successfully while
                # the dedicated proposal lost or its conservative precheck was
                # blocked. Observe that physical completion without inventing
                # a dedicated-return selection. Initial center-lane launch is
                # not completion: first require the whole body in the side lane.
                if (current_side == plan["side"] and projection[1] >=
                        float(reference[nearest].lane_width) / 2 + ego.bounding_box.extent.y + .1):
                    plan.setdefault("observed_passing_lane_at_s", float(now))
            completion = self._observe_return_completion(plan, ego, reference, reference_points,
                stations, station, projection, now, current_side)
            if completion is not None:
                self._active = self._observed = None
                return self._inactive("completed", measured_return_completion=completion)
            if plan["phase"] == "returning":
                return self._result(plan, "returning", committed=True)
            plan.update(self._return_reservation(plan, reference, stations, station, ego))
            pending = plan.get("return_proposal")
            if pending is not None and current_side == plan["side"]:
                nominal_xy = np.array([np.interp(station, pending["sample_s"], pending["points"][:, axis])
                                       for axis in (0, 1)])
                normal = np.array([-math.sin(projection[2]), math.cos(projection[2])])
                mismatch = abs(float((np.array([location.x, -location.y]) - nominal_xy) @ normal))
                if station > pending["anchor"] + .5 and mismatch > .45:
                    # This proposal never executed. Reusing its old mid-merge
                    # geometry would now start metres away from the real ego.
                    # Rebuild at current evidence, inside the SAME stop limit;
                    # a committed return above never enters this branch.
                    plan["pending_return_invalidation_count"] = plan.get("pending_return_invalidation_count", 0) + 1
                    plan["pending_return_invalidation"] = {
                        "reason": "unexecuted_return_geometry_no_longer_matches_ego",
                        "time_s": float(now), "old_anchor_station_m": pending["anchor"],
                        "old_transition_end_station_m": pending["anchor"] + pending["length"],
                        "current_route_station_m": float(station), "nominal_lateral_mismatch_m": mismatch,
                        "fixed_return_wait_station_m": plan["return_wait_route_station_m"],
                        "new_anchor_station_m": None, "executed_return_reanchored": False}
                    plan.pop("return_proposal", None)
            blocker = next((actor for actor in actors if actor["id"] == plan["blocker_id"]), None)
            blocker_station = blocker["station"] if blocker else plan["blocker_station"]
            passed = station - blocker_station > plan["blocker_half_length"] + ego.bounding_box.extent.x + self.config["return_clearance_m"]
            conflict = None
            measured_speed = math.hypot(ego.get_velocity().x, ego.get_velocity().y)
            deadline_near = (plan["return_wait_route_station_m"] - station <=
                             max(3., measured_speed * 2. + measured_speed ** 2 / 4.))
            # If traffic prevents completing the pass, giving up the pass and
            # returning *behind* the blocker can be the safe legal solution.
            # The same complete gap/sweep checks apply; the deadline is never
            # permission to cut in front of a pedestrian or closing rear car.
            if passed or plan.get("return_proposal") is not None or deadline_near or plan["maneuver_overdue"]:
                plan["return_trigger"] = ("passed_original_blocker" if passed else
                    "overdue_safe_return" if plan["maneuver_overdue"] else "abort_pass_before_deadline")
                updated, conflict = self._return_candidate(plan, actors, reference, stations, station, ego)
                if updated is not None:
                    # Reserve the same spatial endpoint until its actual
                    # selection, while checking current occupancy every frame.
                    plan["return_proposal"] = updated
                    if "pending_return_invalidation" in plan:
                        plan["pending_return_invalidation"]["new_anchor_station_m"] = updated["anchor"]
                        updated["pending_return_invalidation"] = deepcopy(plan["pending_return_invalidation"])
                    return self._result(updated, "returning", committed=True)
                plan["return_checks"] = deepcopy(conflict)
            else:
                plan["return_checks"] = {"reason": "original_blocker_not_yet_cleared",
                    "blocker_id": plan["blocker_id"], "blocker_station_m": blocker_station}
            # Keep the target lane until the measured obstacle is fully behind;
            # PDM sees new actors/signals and can stop on this finite real lane.
            return self._result(plan, "passing", committed=True, conflict=conflict)
        if waiting or (signal and signal["distance_m"] < 35):
            self._observed = None
            return self._inactive("traffic_signal_or_queue", signal=signal)
        if current_side != "center" or reference[nearest].is_junction:
            self._observed = None
            return self._inactive("not_in_reference_lane_or_at_junction")
        limit = min(float(ego.get_speed_limit()) / 3.6, 8.)
        lane_width = float(reference[nearest].lane_width)
        possible = [a for a in actors if self.config["minimum_gap_m"] < a["station"] - station < self.config["maximum_blocker_distance_m"]
            and a["station"] - station - a["half_length"] - ego.bounding_box.extent.x > .5
            and a["lateral"] < min(lane_width / 2 - .1, ego.bounding_box.extent.y + a["half_width"] + .15)
            and a["along_speed"] >= -.3 and a["speed"] < self.config["slow_speed_fraction"] * limit
            and (a["aligned"] or a["speed"] < .3) and abs(a["cross_speed"]) < .35]
        blocker = min(possible, key=lambda a: a["station"]) if possible else None
        if blocker is None:
            self._observed = None
            return self._inactive("no_persistent_slow_obstacle")
        if blocker["type"].startswith("walker.") and _crosswalk_near(world.get_map(), blocker["xy"]):
            self._observed = None
            return self._inactive("pedestrian_at_crosswalk", actor_id=blocker["id"])
        if self._observed is None or self._observed["id"] != blocker["id"]:
            self._observed = {"id": blocker["id"], "since": float(now), "count": 1}
        else:
            self._observed["count"] += 1
        if now - self._observed["since"] < self.config["observation_s"] or self._observed["count"] < 3:
            return self._inactive("observing_slow_obstacle", actor_id=blocker["id"])
        clearance = blocker["station"] - station - blocker["half_length"] - ego.bounding_box.extent.x - 1.5
        # Completion of the ENTIRE lane change before the actor's longitudinal
        # station is unnecessary: the occupied bodies can already be separated
        # laterally while the transition is still in progress. Search geometry
        # and verify actual envelopes instead of imposing a fixed 9 m vacancy.
        preferred = min(24., max(6., clearance))
        # A close blocker does not make the shortest S curve the best initial
        # proposal. First try smoother legal curves at their reachable speed;
        # retain short curves when traffic/topology actually requires them.
        lengths = sorted({preferred, 24., 20., 16., 12., 10., 9., 8., 7., 6.}, reverse=True)
        getter = getattr(ego, "get_physics_control", None)
        physics = getter() if callable(getter) else None
        maximum_curvature = _curvature_limit(ego, physics=physics)
        failures = {}
        # Search lower speeds only after the normal-speed alternatives have
        # all failed. Otherwise a long 1.5 m/s curve could hide a smooth 16 m
        # curve that is safe at 4 m/s in the very same observed traffic.
        for reduced_speed, side in ((reduced, side) for reduced in (False, True)
                                     for side in ("left", "right")):
            allowed, reason = self._corridor(reference, stations, station, station + 6., side)
            if not allowed:
                failures[side] = reason
                continue
            # A finite adjacent route is not a licence to take another junction
            # exit. Leave enough real, legally returnable road after the obstacle.
            end = station
            for i in range(nearest, len(reference)):
                adjacent = _adjacent(reference[i], side)
                if (adjacent is None or reference[i].is_junction or adjacent.is_junction
                        or not _same_direction(reference[i], adjacent)
                        or (i > nearest and not _connected(_adjacent(reference[i - 1], side), adjacent))):
                    break
                end = float(stations[i])
            previous = failures.get(side)
            attempts = list(previous.get("attempts", ())) if isinstance(previous, dict) else []
            for length in lengths:
                allowed, reason = self._corridor(reference, stations, station, station + length, side)
                if not allowed:
                    attempts.append({"length_m": length, "reason": reason})
                    continue
                return_start = max(station+length, blocker["station"] + blocker["half_length"]
                    + ego.bounding_box.extent.x + self.config["return_clearance_m"])
                return_end = return_start + length
                allowed, reason = self._corridor(reference, stations, return_start, return_end, side, returning=True)
                if return_end + 4 > end or not allowed:
                    attempts.append({"length_m": length, "reason": "no_complete_legal_return_corridor"})
                    continue
                built = self._build(reference, stations, station, length, side, end - 2)
                if built is None:
                    attempts.append({"length_m": length, "reason": "path_construction_failed"})
                    continue
                kmax = max(float(np.max(_curvature(built[0]))), .001)
                if kmax > maximum_curvature:
                    attempts.append({"length_m": length, "reason": "steering_curvature_unreachable",
                                     "curvature_inv_m": kmax, "limit_inv_m": maximum_curvature})
                    continue
                cap = min(self.config["maximum_pass_speed_mps"], limit,
                          math.sqrt(self.config["lateral_acceleration_mps2"] / kmax))
                minimum_advantage = max(1.0, blocker["along_speed"] + .75)
                feasible_caps = [float(speed) for speed in np.linspace(cap, minimum_advantage, 16)
                    if minimum_advantage <= speed <= cap
                    and kmax <= _curvature_limit(ego, speed, physics=physics)]
                if not feasible_caps:
                    attempts.append({"length_m": length, "reason": "no_speed_with_reachable_steering_advantage"})
                    continue
                maximum_cap = feasible_caps[0]
                caps = ([maximum_cap] if not reduced_speed else
                        sorted({min(maximum_cap, 3.), min(maximum_cap, 2.), minimum_advantage}, reverse=True))
                for cap in caps:
                    if (cap < minimum_advantage or cap > maximum_cap
                            or (reduced_speed and cap >= maximum_cap - 1e-6)
                            or kmax > _curvature_limit(ego, cap, physics=physics)):
                        continue
                    # These proposals really execute with this cap. The
                    # measured speed is retained by _prediction_speeds; an
                    # unrelated ordinary 8 m/s policy is not their forecast.
                    clear, conflict = self._target_gap_clear(actors, reference, stations, station, ego, side, cap,
                                                           candidate_speed_mps=cap)
                    if clear:
                        clear, conflict = self._gap_clear(actors, built[0], built[1], station, ego, cap,
                            blocker_id=blocker["id"], candidate_speed_mps=cap)
                    if not clear:
                        attempts.append(dict(conflict, length_m=length, speed_cap_mps=cap,
                                             speed_search_round="reduced" if reduced_speed else "maximum"))
                        continue
                    self._counter += 1
                    plan = {"serial": self._counter, "side": side, "phase": "passing", "anchor": station,
                        "length": length, "points": built[0], "sample_s": built[1], "widths": built[2],
                        "speed_cap": cap, "blocker_id": blocker["id"], "blocker_station": blocker["station"],
                        "blocker_half_length": blocker["half_length"], "corridor_end": end,
                        "passing_checks": {"attempts": attempts,
                            "speed_search_round": "reduced" if reduced_speed else "maximum",
                            "speed_envelope": "candidate_cap_and_measured_speed"}}
                    return self._result(plan, "passing", committed=False)
            failures[side] = {"reason": "no_feasible_transition", "attempts": attempts}
        return self._inactive("no_safe_legal_overtaking_corridor", excluded=failures, actor_id=blocker["id"])


class _GroupedVelocityPolicies:
    """Keep ordinary IDM unchanged and bind a target cap only to maneuver paths.

    Delegating each group using its original policy indices preserves the
    adapter's STOP/STOP_HARD handling. Positions and velocities are integrated
    together by that same numerical IDM, before native bicycle simulation.
    """

    _arrays = ("_fallback_target_velocities", "_speed_limit_fractions", "_min_gap_to_lead_agent",
               "_headway_time", "_accel_max", "_decel_max", "_target_velocities")

    def __init__(self, base, caps):
        self.original_size = int(base.num_policies)
        self.caps = (None,) + tuple(caps)
        self.groups = [base] + [deepcopy(base) for _ in caps]
        for group, cap in zip(self.groups[1:], caps):
            group._fallback_target_velocities = np.minimum(group._fallback_target_velocities, cap)
            group._target_velocities = np.minimum(group._target_velocities, cap)
        self._num_policies = len(self.groups) * self.original_size
        self._num_longitudinal_proposals = self._num_policies
        self._refresh_arrays()

    def _refresh_arrays(self):
        for key in self._arrays:
            setattr(self, key, np.concatenate([np.asarray(getattr(group, key)) for group in self.groups]))

    @property
    def num_policies(self):
        return self._num_policies

    @property
    def max_target_velocity(self):
        return float(np.max(self._target_velocities))

    def update(self, speed_limit_mps):
        for group, cap in zip(self.groups, self.caps):
            group.update(speed_limit_mps)
            if cap is not None:
                group._target_velocities = np.minimum(group._target_velocities, cap)
        self._refresh_arrays()

    def propagate(self, previous_idm_states, leading_agent_states, longitudinal_idcs, sampling_time):
        indices = np.asarray(longitudinal_idcs, dtype=int)
        if (len(indices) != len(previous_idm_states) or len(indices) != len(leading_agent_states)
                or np.any(indices < 0) or np.any(indices >= self._num_policies)):
            raise ValueError("grouped IDM indices must match the current proposal batch")
        group_ids = indices // self.original_size
        result = np.empty_like(previous_idm_states, dtype=float)
        for group_id in np.unique(group_ids):
            mask = group_ids == group_id
            result[mask] = self.groups[group_id].propagate(previous_idm_states[mask],
                leading_agent_states[mask], (indices[mask] % self.original_size).tolist(), sampling_time)
        return result


def maneuver_proposal_manager(api, paths, lateral, policies, labels, targets):
    """Native manager with ordinary or maneuver-specific IDM combinations.

    The native manager constructs combinations; this adapter retains only the
    appropriate policy group for each path, then restores continuous proposal
    indices. The native generator iterates this actual proposal list by lateral
    index. Every retained proposal is simulated and scored in ONE shared pool.
    Ordinary path/policy pairs retain their old order, parameters and profiles.
    """
    if len(paths) != len(lateral) or len(labels) != policies.num_policies or len(targets) != len(labels):
        raise ValueError("maneuver manager needs matching paths, metadata and policy labels")
    caps = []
    path_caps = []
    for metadata in lateral:
        cap = (metadata.get("maneuver_speed_cap_mps")
               if metadata.get("maneuver") or metadata.get("maneuver_execution_limit") is True else None)
        if cap is not None:
            if type(cap) not in (int, float) or not math.isfinite(cap) or not .1 <= cap <= 20:
                raise ValueError("maneuver speed cap must be finite and within [0.1, 20]")
            cap = float(cap)
            if cap not in caps:
                caps.append(cap)
        path_caps.append(cap)
    size = len(labels)
    if not caps:
        manager = api.PDMProposalManager(paths, policies)
        if any(metadata.get("maneuver_wait_hold") for metadata in lateral):
            manager._proposals = [proposal for proposal in manager._proposals
                if not lateral[proposal.lateral_idx].get("maneuver_wait_hold")
                or labels[proposal.longitudinal_idx] in ("STOP", "STOP_HARD")]
            for index, proposal in enumerate(manager._proposals):
                proposal.proposal_idx = index
        return manager, list(labels), list(targets), list(range(size))
    grouped = _GroupedVelocityPolicies(policies, caps)
    manager = api.PDMProposalManager(paths, grouped)
    group_for_path = [0 if cap is None else 1 + caps.index(cap) for cap in path_caps]
    manager._proposals = [proposal for proposal in manager._proposals
        if proposal.longitudinal_idx // size == group_for_path[proposal.lateral_idx]
        and (not lateral[proposal.lateral_idx].get("maneuver_wait_hold")
             or labels[proposal.longitudinal_idx % size] in ("STOP", "STOP_HARD"))]
    for index, proposal in enumerate(manager._proposals):
        proposal.proposal_idx = index
    expanded_labels = list(labels) * (1 + len(caps))
    expanded_targets = list(targets) + [min(target, cap) for cap in caps for target in targets]
    source_indices = list(range(size)) * (1 + len(caps))
    return manager, expanded_labels, expanded_targets, source_indices
