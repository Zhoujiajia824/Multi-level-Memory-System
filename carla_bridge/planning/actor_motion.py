"""Auditable current actor motion and connected-road forecasts, without learning.

CARLA walker controllers can move an actor while get_velocity() stays zero.
This module estimates current velocity from past positions, verifies API motion,
and uses only explicit waypoint successors or a measured turn rate to forecast.
It never consumes future simulator states, memory, model decisions or EKF state.
All coordinates and angular signs are CARLA world x/y/yaw.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from types import SimpleNamespace

import numpy as np


MOTION_MODEL_VERSION = "carla_actor_motion_v4"
_DEFAULTS = {
    "history_window_s": .6,
    "maximum_gap_s": 1.2,
    "minimum_difference_interval_s": .04,
    "position_noise_m": .025,
    "velocity_disagreement_mps": .35,
    "velocity_disagreement_fraction": .4,
    "vehicle_max_speed_mps": 60.,
    "walker_max_speed_mps": 8.,
    "vehicle_max_yaw_rate_rad_s": 1.2,
    "vehicle_api_max_acceleration_mps2": 20.,
    "vehicle_api_integral_tolerance_m": .12,
    "vehicle_api_integral_fraction": .1,
    "vehicle_zero_api_confirmation_s": .4,
    "vehicle_zero_api_speed_epsilon_mps": .05,
    "history_retention_s": 2.,
    "map_guidance": True,
    "map_alignment_rad": math.radians(30.),
    "map_max_lateral_speed_mps": .35,
    "branch_separation_rad": math.radians(7.),
    "prediction_uncertainty_mps": .05,
    "unverified_prediction_uncertainty_mps": .15,
    "following_response_window_s": 1.5,
    "following_relation_window_s": .4,
    "following_speed_drop_mps": 1.,
    "following_speed_drop_fraction": .2,
    "following_reaction_s": .4,
    "following_max_braking_mps2": 4.,
    "following_comfort_braking_mps2": 2.,
    "following_time_headway_s": 1.2,
    "following_minimum_gap_m": 2.,
    "following_lateral_tolerance_m": .6,
    "following_heading_tolerance_rad": math.radians(20.),
    "following_response_valid_s": 2.,
    "following_renew_drop_mps": .5,
    "following_renew_drop_fraction": .15,
    "following_reacceleration_window_s": .4,
    "following_reacceleration_mps": 1.,
    "following_reacceleration_fraction": .2,
}


def validate_motion_config(config=None):
    """Return canonical execution settings, rejecting ignored/misspelled keys."""
    supplied = dict(config or {})
    unknown = set(supplied) - set(_DEFAULTS)
    if unknown:
        raise ValueError("Unknown actor_motion settings: " + ", ".join(sorted(unknown)))
    result = dict(_DEFAULTS, **supplied)
    if type(result["map_guidance"]) is not bool:
        raise ValueError("actor_motion.map_guidance must be bool")
    for key, value in result.items():
        if key == "map_guidance":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("actor_motion." + key + " must be positive and finite")
        result[key] = float(value)
    if (result["minimum_difference_interval_s"] >= result["history_window_s"]
            or result["history_window_s"] > result["maximum_gap_s"]
            or result["history_retention_s"] < result["maximum_gap_s"]):
        raise ValueError("actor_motion history intervals must be ordered")
    if result["velocity_disagreement_fraction"] > 1 or result["map_alignment_rad"] > math.pi/3:
        raise ValueError("actor_motion motion/alignment tolerance is excessive")
    if result["vehicle_zero_api_confirmation_s"] > result["history_retention_s"]:
        raise ValueError("actor_motion zero API confirmation must fit observed history")
    if (result["following_relation_window_s"] > result["following_response_window_s"]
            or result["following_response_window_s"] > result["history_retention_s"]
            or result["following_speed_drop_fraction"] >= 1
            or result["following_renew_drop_fraction"] >= 1
            or result["following_reacceleration_fraction"] >= 1
            or result["following_response_valid_s"] > result["history_retention_s"]
            or result["following_comfort_braking_mps2"] > result["following_max_braking_mps2"]):
        raise ValueError("actor_motion following evidence intervals or physical bounds are invalid")
    return result


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def _finite(value, default=0.):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _api_velocity(actor):
    try:
        raw = actor.get_velocity()
        values = tuple(float(getattr(raw, axis, 0.)) for axis in ("x", "y", "z"))
        if np.isfinite(values).all():
            return values, True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return (0., 0., 0.), False


@dataclass(frozen=True)
class ActorMotionEstimate:
    actor_id: int
    actor_type: str
    sim_time_s: float | None
    vx: float
    vy: float
    vz: float
    yaw_rate_rad_s: float
    source: str
    quality: str
    audit: dict = field(default_factory=dict)

    @property
    def velocity(self):
        """CARLA-shaped current velocity; callers do their own axis conversion."""
        return SimpleNamespace(x=self.vx, y=self.vy, z=self.vz)

    @property
    def speed_mps(self):
        return math.hypot(self.vx, self.vy)


class ActorMotionHistory:
    """One explicit world/time owner, shared across numerical planning modules.

    Repeated simulation timestamps reuse the same measurements. A backwards
    clock, long gap, actor disappearance/type change or teleport resets history.
    State is bounded by time and only exists for currently observed actors.
    """
    def __init__(self, config=None):
        self.config = validate_motion_config(config)
        self.last_time_s = None
        self._tracks, self._estimates = {}, {}
        self._following_states = {}
        self._following_invalidated_s = {}
        self._relation_geometry_cache = {}
        self.last_update_audit = {"version": MOTION_MODEL_VERSION, "status": "empty"}

    def update(self, world=None, sim_time_s=None, actors=None):
        if actors is None:
            actors = list(world.get_actors())
        else:
            actors = list(actors)
        if sim_time_s is None and world is not None:
            try:
                sim_time_s = float(world.get_snapshot().timestamp.elapsed_seconds)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                sim_time_s = None
        valid_time = (type(sim_time_s) in (int, float) and math.isfinite(sim_time_s))
        now = float(sim_time_s) if valid_time else None
        event = "updated"
        if now is not None and self.last_time_s is not None:
            delta = now - self.last_time_s
            if abs(delta) <= 1e-8:
                self.last_update_audit = {"version": MOTION_MODEL_VERSION,
                    "status": "repeated_timestamp_reused", "time_s": now,
                    "actor_count": len(self._estimates)}
                return self
            if delta < 0 or delta > self.config["maximum_gap_s"]:
                event = "clock_rewind_reset" if delta < 0 else "sampling_gap_reset"
                self._tracks.clear()
                self._following_states.clear()
                self._following_invalidated_s.clear()
        if now is None:
            event = "timestamp_unavailable_api_only"
            self._tracks.clear()
            self._following_states.clear()
            self._following_invalidated_s.clear()
        # Pure observed-pose geometry is shared by context and planner users,
        # but never survives a new physical snapshot or an actor lifecycle.
        self._relation_geometry_cache.clear()
        estimates, present = {}, set()
        for actor in actors:
            if not str(getattr(actor, "type_id", "")).startswith(("vehicle.", "walker.", "static.prop.")):
                continue
            actor_id = int(actor.id)
            present.add(actor_id)
            try:
                estimates[actor_id] = self._observe(actor, now, event)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                # Actor destruction between get_actors and RPC reads is normal.
                self._tracks.pop(actor_id, None)
                self._following_states.pop(actor_id, None)
                self._following_invalidated_s.pop(actor_id, None)
                estimates[actor_id] = self._api_estimate(actor, now,
                    "actor_observation_unavailable:" + type(exc).__name__)
        removed = set(self._tracks) - present
        for actor_id in removed:
            self._tracks.pop(actor_id, None)
            self._following_states.pop(actor_id, None)
            self._following_invalidated_s.pop(actor_id, None)
        self._estimates, self.last_time_s = estimates, now
        self.last_update_audit = {"version": MOTION_MODEL_VERSION, "status": event,
            "time_s": now, "actor_count": len(estimates), "removed_actor_count": len(removed)}
        return self

    def _api_estimate(self, actor, now, event):
        velocity, valid = _api_velocity(actor)
        return ActorMotionEstimate(int(actor.id), str(actor.type_id), now, *velocity,
            0., "api_velocity_unverified" if valid else "motion_unavailable", "unverified",
            {"version": MOTION_MODEL_VERSION, "event": event, "history_samples": 0,
             "api_velocity_mps": list(velocity), "api_velocity_valid": valid,
             "yaw_rate_source": "insufficient_observed_history"})

    def motion_for(self, actor):
        current = self._estimates.get(int(actor.id))
        if current is not None and current.actor_type == str(actor.type_id):
            return current
        return self._api_estimate(actor, self.last_time_s, "actor_not_in_current_snapshot")

    def recent_observations(self, actor, window_s=None):
        """Distinct past/current poses; never a simulator advance or future state."""
        track = self._tracks.get(int(actor.id))
        if track is None or track[0] != str(actor.type_id) or self.last_time_s is None:
            return ()
        window = self.config["history_retention_s"] if window_s is None else float(window_s)
        return tuple((time, position.copy(), yaw) for time, position, yaw in track[1]
                     if self.last_time_s-time <= window+1e-8)

    def _observe(self, actor, now, event):
        update_event = event
        if now is None:
            return self._api_estimate(actor, now, event)
        pose = actor.get_transform()
        position = np.array([pose.location.x, pose.location.y, pose.location.z], dtype=float)
        yaw = math.radians(float(pose.rotation.yaw))
        if not np.isfinite(position).all() or not math.isfinite(yaw):
            raise ValueError("invalid actor pose")
        kind, actor_id = str(actor.type_id), int(actor.id)
        maximum = self.config["walker_max_speed_mps"] if kind.startswith("walker.") else self.config["vehicle_max_speed_mps"]
        track = self._tracks.get(actor_id)
        if track is None or track[0] != kind:
            event = "new_actor" if track is None else "actor_type_changed_reset"
            track = (kind, deque(maxlen=128), deque(maxlen=128), deque(maxlen=128))
            self._tracks[actor_id] = track
            self._following_states.pop(actor_id, None)
            self._following_invalidated_s.pop(actor_id, None)
        samples = track[1]
        if samples:
            elapsed = now - samples[-1][0]
            displacement = np.linalg.norm(position[:2] - samples[-1][1][:2])
            # Permit sub-centimetre RPC noise, never treat teleportation as speed.
            if elapsed <= 0 or displacement > max(2., maximum * elapsed + .5):
                samples.clear()
                track[2].clear()
                track[3].clear()
                self._following_states.pop(actor_id, None)
                self._following_invalidated_s.pop(actor_id, None)
                event = "teleport_or_invalid_interval_reset"
        samples.append((now, position, yaw))
        while len(samples) > 2 and now - samples[0][0] > self.config["history_retention_s"]:
            samples.popleft()
        window = [sample for sample in samples if now - sample[0] <= self.config["history_window_s"] + 1e-8]
        velocity, valid_api = _api_velocity(actor)
        api = np.asarray(velocity)
        if np.linalg.norm(api[:2]) > maximum:
            valid_api = False
            api = np.zeros(3)
        integral_consistent, integral_error, api_acceleration = False, None, None
        if kind.startswith("vehicle.") and valid_api and track[2] and len(samples) >= 2:
            old_time, old_api, old_valid = track[2][-1]
            interval = now-old_time
            if old_valid and interval >= self.config["minimum_difference_interval_s"]:
                observed_delta = position[:2]-samples[-2][1][:2]
                integrated = .5*(old_api[:2]+api[:2])*interval
                integral_error = float(np.linalg.norm(integrated-observed_delta))
                api_acceleration = float(np.linalg.norm(api[:2]-old_api[:2])/interval)
                integral_consistent = (api_acceleration <= self.config["vehicle_api_max_acceleration_mps2"]
                    and integral_error <= self.config["vehicle_api_integral_tolerance_m"]
                    +self.config["vehicle_api_integral_fraction"]*float(np.linalg.norm(observed_delta)))
        track[2].append((now, api.copy(), valid_api))
        while track[2] and now-track[2][0][0] > self.config["history_retention_s"]+1e-8:
            track[2].popleft()
        result, source, quality = api.copy(), ("api_velocity_unverified" if valid_api else "motion_unavailable"), "unverified"
        fd, span, residual, yaw_rate = None, 0., None, 0.
        zero_api_observed_motion = False
        yaw_source = "insufficient_observed_history"
        if len(window) >= 2:
            stamps = np.array([sample[0] for sample in window], dtype=float)
            span = float(stamps[-1] - stamps[0])
            if span >= self.config["minimum_difference_interval_s"]:
                centered = stamps - stamps.mean()
                positions = np.array([sample[1] for sample in window])
                fd = centered @ positions / float(centered @ centered)
                predicted = positions.mean(axis=0) + centered[:, None] * fd
                residual = float(np.max(np.linalg.norm(positions[:, :2] - predicted[:, :2], axis=1)))
                displacement = float(np.linalg.norm(positions[-1, :2]-positions[0, :2]))
                historical_speed = float(np.linalg.norm(fd[:2]))
                # Short noisy intervals must not turn a stopped actor into a walker.
                fd_valid = historical_speed <= maximum and residual <= max(.25, historical_speed * span * .3)
                moving = displacement > max(self.config["position_noise_m"] * 2, .1 * span)
                disagreement = float(np.linalg.norm(fd[:2] - api[:2]))
                threshold = max(self.config["velocity_disagreement_mps"],
                    self.config["velocity_disagreement_fraction"] * historical_speed,
                    self.config["position_noise_m"] * 2 / span)
                recent_interval = float(stamps[-1]-stamps[-2])
                recent_velocity = (positions[-1]-positions[-2])/recent_interval
                recent_threshold = max(self.config["velocity_disagreement_mps"],
                    self.config["position_noise_m"]*2/recent_interval,
                    .25*float(np.linalg.norm(recent_velocity[:2])))
                recent_api_consistent = (kind.startswith("vehicle.") and valid_api
                    and (integral_consistent or np.linalg.norm(api[:2]-recent_velocity[:2]) <= recent_threshold))
                zero_samples = [(t, v, good) for t, v, good in track[2]
                    if now-t <= self.config["vehicle_zero_api_confirmation_s"]+1e-8]
                if (kind.startswith("vehicle.") and len(zero_samples) >= 3
                        and now-zero_samples[0][0] >= self.config["vehicle_zero_api_confirmation_s"]-1e-7
                        and all(good and np.linalg.norm(v[:2]) <= self.config["vehicle_zero_api_speed_epsilon_mps"]
                            for _, v, good in zero_samples)):
                    earlier = next((p for t, p, _ in samples if abs(t-zero_samples[0][0]) <= 1e-8), None)
                    # A sharp stop has zero displacement after its first zero
                    # API sample. Several zero readings with continuing real
                    # displacement instead invalidate the API's stationary
                    # claim, even below the normal disagreement deadband.
                    zero_api_observed_motion = (earlier is not None and
                        np.linalg.norm(position[:2]-earlier[:2]) > max(
                            2*self.config["position_noise_m"], .1*(now-zero_samples[0][0])))
                if fd_valid and (moving or np.linalg.norm(api[:2]) > .2):
                    if not valid_api or zero_api_observed_motion or (disagreement > threshold and not recent_api_consistent):
                        result, source = fd, "position_history_corrected_api"
                    elif kind.startswith("vehicle."):
                        # Current PhysX velocity is instantaneous. Averaging it
                        # with a backwards difference delays acceleration/turns;
                        # the past verifies it rather than adding a phase lag.
                        result, source = api, "history_validated_api_velocity"
                    else:
                        result, source = .7 * fd + .3 * api, "history_api_consistent_fusion"
                    quality = "verified" if len(window) >= 3 and span >= .15 else "provisional"
                elif fd_valid and not moving and np.linalg.norm(api[:2]) <= .2:
                    result, source, quality = np.zeros(3), "history_verified_stationary", "verified"
                yaws = np.unwrap([sample[2] for sample in window])
                rate = float(centered @ yaws / float(centered @ centered))
                turn_residual = float(np.max(np.abs(yaws - (yaws.mean()+centered*rate))))
                if (kind.startswith("vehicle.") and len(window) >= 3 and span >= .15
                        and abs(rate) <= self.config["vehicle_max_yaw_rate_rad_s"]
                        and turn_residual <= .06 and np.linalg.norm(result[:2]) >= .35):
                    yaw_rate, yaw_source = rate, "observed_yaw_history"
        if not valid_api and fd is None:
            result, source, quality = np.zeros(3), "motion_unavailable", "unverified"
        if kind.startswith("static.prop."):
            result, yaw_rate, source = np.zeros(3), 0., "static_actor"
        track[3].append((now, float(np.linalg.norm(result[:2])), quality == "verified"))
        while track[3] and now-track[3][0][0] > self.config["history_retention_s"]+1e-8:
            track[3].popleft()
        return ActorMotionEstimate(actor_id, kind, now, *map(float, result), float(yaw_rate),
            source, quality, {"version": MOTION_MODEL_VERSION, "event": event,
                "world_update_status": update_event,
                "history_samples": len(window), "history_span_s": span,
                "api_velocity_mps": list(velocity), "api_velocity_valid": bool(valid_api),
                "api_integrated_displacement_consistent": bool(integral_consistent),
                "api_integrated_displacement_error_m": integral_error,
                "api_acceleration_mps2": api_acceleration,
                "continuous_zero_api_contradicted_by_displacement": bool(zero_api_observed_motion),
                "position_difference_velocity_mps": None if fd is None else fd.tolist(),
                "position_fit_residual_m": residual, "velocity_source": source,
                "yaw_rate_source": yaw_source, "yaw_rate_rad_s": float(yaw_rate),
                "current_state_only": True})


def _turn_rate_path(position, velocity, yaw, rate, times):
    if abs(rate) < 1e-5:
        return position + times[:, None] * velocity, np.full(len(times), yaw)
    angle = times * rate
    dx = (velocity[0]*np.sin(angle) + velocity[1]*(np.cos(angle)-1))/rate
    dy = (velocity[0]*(1-np.cos(angle)) + velocity[1]*np.sin(angle))/rate
    return position + np.column_stack((dx, dy)), yaw + angle


def _lane_key(waypoint):
    values = tuple(getattr(waypoint, name, None) for name in ("road_id", "section_id", "lane_id"))
    return None if any(value is None for value in values) else values


def _connected_map_path(actor, estimate, times, carla_map, config):
    """Follow actual successor links; ambiguous branches stay an explicit fallback.

    A nearest waypoint is used only once to locate the current lane. Every
    future waypoint comes from that waypoint's next(distance); there is no
    global nearest-road lookup, cross-junction snapping, or lane-ID exclusion.
    """
    if carla_map is None or not config["map_guidance"]:
        return None, {"map_status": "unavailable_or_disabled"}
    pose = actor.get_transform()
    position = np.array([pose.location.x, pose.location.y], dtype=float)
    yaw, speed = math.radians(pose.rotation.yaw), estimate.speed_mps
    try:
        try:
            waypoint = carla_map.get_waypoint(pose.location, project_to_road=False)
        except TypeError:
            waypoint = carla_map.get_waypoint(pose.location)
        if waypoint is None or str(getattr(waypoint, "lane_type", "Driving")).split(".")[-1] != "Driving":
            return None, {"map_status": "current_position_not_on_driving_lane"}
        center = np.array([waypoint.transform.location.x, waypoint.transform.location.y], dtype=float)
        road_yaw = math.radians(waypoint.transform.rotation.yaw)
        heading_offset = _angle(yaw-road_yaw)
        velocity_heading = math.atan2(estimate.vy, estimate.vx)
        if (abs(heading_offset) > config["map_alignment_rad"]
                or abs(_angle(velocity_heading-road_yaw)) > config["map_alignment_rad"]
                or abs(waypoint.transform.location.z-pose.location.z) > 1.5
                or np.linalg.norm(position-center) > float(getattr(waypoint, "lane_width", 3.5))/2+.5):
            return None, {"map_status": "current_motion_not_aligned_with_lane"}
        tangent = np.array([math.cos(road_yaw), math.sin(road_yaw)])
        normal = np.array([-tangent[1], tangent[0]])
        along_offset, lateral_offset = float((position-center)@tangent), float((position-center)@normal)
        lateral_speed = float(np.array([estimate.vx, estimate.vy])@normal)
        # A modest heading angle can already be an actual lane change. Keep
        # that measured lateral movement instead of pinning another vehicle to
        # its current lane merely because it still projects onto that lane.
        if (abs(lateral_speed) > config["map_max_lateral_speed_mps"]
                or (abs(lateral_speed) > .12 and abs(lateral_offset) > .35
                    and lateral_speed*lateral_offset > 0)):
            return None, {"map_status": "observed_lateral_motion_preserved",
                "lateral_velocity_mps": lateral_speed, "lateral_offset_m": lateral_offset}
        positions, yaws, lane_keys = [position.copy()], [yaw], [_lane_key(waypoint)]
        max_rate = min(config["vehicle_max_yaw_rate_rad_s"], 12./max(speed, .5))
        for index, dt in enumerate(np.diff(times), start=1):
            distance = max(.001, speed*float(dt))
            options = list(waypoint.next(distance))
            expected_yaw = yaws[-1] + estimate.yaw_rate_rad_s*float(dt)
            valid = []
            for candidate in options:
                cp = candidate.transform.location
                cyaw = math.radians(candidate.transform.rotation.yaw)
                if (abs(cp.z-waypoint.transform.location.z) > 1.
                        or math.hypot(cp.x-waypoint.transform.location.x, cp.y-waypoint.transform.location.y) > distance*1.5+.3
                        or abs(_angle(cyaw-road_yaw)) > max(.12, max_rate*dt+.06)):
                    continue
                score = abs(_angle(cyaw-expected_yaw))
                if not any(math.hypot(cp.x-v[1].transform.location.x, cp.y-v[1].transform.location.y) < .05
                           and abs(_angle(cyaw-math.radians(v[1].transform.rotation.yaw))) < .03 for v in valid):
                    valid.append((score, candidate))
            valid.sort(key=lambda item: item[0])
            if not valid or (len(valid) > 1 and valid[1][0]-valid[0][0] < config["branch_separation_rad"]):
                # Do not invent a turn at an unknown branch. Continue the
                # already observed physical turn from the last connected state.
                remaining = times[index:]-times[index-1]
                velocity = speed*np.array([math.cos(yaws[-1]), math.sin(yaws[-1])])
                tail, headings = _turn_rate_path(positions[-1], velocity, yaws[-1], estimate.yaw_rate_rad_s, remaining)
                positions.extend(tail)
                yaws.extend(headings)
                return (np.asarray(positions), np.asarray(yaws)), {"map_status": "connected_prefix_then_motion_fallback",
                    "connected_samples": index, "fallback_at_s": float(times[index]),
                    "fallback_reason": "ambiguous_successors" if valid else "no_valid_successor"}
            waypoint = valid[0][1]
            lane_keys.append(_lane_key(waypoint))
            road_yaw = math.radians(waypoint.transform.rotation.yaw)
            tangent = np.array([math.cos(road_yaw), math.sin(road_yaw)])
            normal = np.array([-tangent[1], tangent[0]])
            # Preserve the observed lane offset; never teleport onto the center.
            center = np.array([waypoint.transform.location.x, waypoint.transform.location.y], dtype=float)
            positions.append(center + along_offset*tangent + lateral_offset*normal)
            yaws.append(road_yaw+heading_offset*math.exp(-float(times[index])/1.5))
        return (np.asarray(positions), np.unwrap(yaws)), {"map_status": "explicit_connected_successors",
            "connected_samples": len(times), "initial_lateral_offset_m": lateral_offset,
            "connected_lane_keys": lane_keys}
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return None, {"map_status": "map_observation_unavailable", "error_type": type(exc).__name__}


def _observed_path_relation(history, actor, leader, stamp, position, yaw,
                            lead_position, lead_yaw, carla_map, config):
    """Prove a historical leader on its unique directed road, also on bends.

    A front car and a rear car on a bend need not share a world-space tangent.
    Compare their own road-normal offsets/headings, then require the actual
    front waypoint to be reached by explicit unique successors. Lane identity
    alone cannot prove a branch, crossing or travel direction.
    """
    cache_key = (id(carla_map), int(actor.id), int(leader.id), round(stamp, 8),
        config["following_heading_tolerance_rad"], config["following_lateral_tolerance_m"],
        config["map_guidance"])
    if cache_key in history._relation_geometry_cache:
        return history._relation_geometry_cache[cache_key]

    def compute():
        if carla_map is None or not config["map_guidance"]:
            return False
        template = actor.get_transform().location

        def waypoint_for(p):
            try:
                location = type(template)(*map(float, p))
            except TypeError:
                location = SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            try:
                return carla_map.get_waypoint(location, project_to_road=False)
            except TypeError:
                return carla_map.get_waypoint(location)

        first, last = waypoint_for(position), waypoint_for(lead_position)
        if any(w is None or _lane_key(w) is None
                or str(getattr(w, "lane_type", "Driving")).split(".")[-1] != "Driving"
                for w in (first, last)):
            return False
        offsets = []
        for p, angle, waypoint in ((position, yaw, first), (lead_position, lead_yaw, last)):
            center = waypoint.transform.location
            road_yaw = math.radians(waypoint.transform.rotation.yaw)
            if (abs(_angle(angle-road_yaw)) > config["following_heading_tolerance_rad"]
                    or abs(float(p[2])-center.z) > 1.5):
                return False
            offset = float((p[:2]-[center.x, center.y]) @
                np.array([-math.sin(road_yaw), math.cos(road_yaw)]))
            if abs(offset) > float(getattr(waypoint, "lane_width", 3.5))/2:
                return False
            offsets.append(offset)
        if abs(offsets[1]-offsets[0]) > config["following_lateral_tolerance_m"]:
            return False
        target = last.transform.location

        def reaches(waypoint):
            p = waypoint.transform.location
            return (_lane_key(waypoint) == _lane_key(last)
                and math.hypot(p.x-target.x, p.y-target.y) <= .05
                and abs(p.z-target.z) <= 1.)

        # OpenDRIVE s is a physical arc station. next(|delta s|), rather than
        # the sign of s or a lane-id convention, proves the travel direction.
        fs, ls = getattr(first, "s", None), getattr(last, "s", None)
        if (_lane_key(first) == _lane_key(last) and fs is not None and ls is not None):
            distance = abs(float(ls)-float(fs))
            if distance <= .1:
                return False
            options = list(first.next(distance))
            return len(options) == 1 and reaches(options[0])
        source = first.transform.location
        chord = math.hypot(source.x-target.x, source.y-target.y)
        if chord <= .1:
            return False
        # Fast exact proof on straight fixture roads or across a straight seam.
        options = list(first.next(chord))
        if len(options) == 1 and reaches(options[0]):
            return True
        # Across road/section seams, retain a bounded explicit successor chain.
        # It stops at any branch; no nearest future road or favorable branch is
        # selected. The bound only rejects unproved long/looping relationships.
        waypoint = first
        for _ in range(min(256, int(math.ceil((2*chord+5.)/.5)))):
            options = list(waypoint.next(.5))
            if len(options) != 1:
                return False
            previous, waypoint = waypoint, options[0]
            if reaches(waypoint):
                return True
            if _lane_key(last) in (_lane_key(previous), _lane_key(waypoint)):
                a, b = previous.transform.location, waypoint.transform.location
                segment = np.array([b.x-a.x,b.y-a.y])
                squared = float(segment@segment)
                fraction = float(np.array([target.x-a.x,target.y-a.y])@segment/max(squared,1e-12))
                if (0. <= fraction <= 1. and squared > 1e-10
                        and np.linalg.norm(np.array([a.x,a.y])+fraction*segment-[target.x,target.y]) <= .05
                        and abs(a.z+fraction*(b.z-a.z)-target.z) <= 1.):
                    return True
        return False

    try:
        valid = bool(compute())
    except (AttributeError, RuntimeError, TypeError, ValueError, KeyError):
        valid = False
    history._relation_geometry_cache[cache_key] = valid
    return valid


def _response_phase(history, actor, leader, config, carla_map):
    """Advance finite observed-response evidence once per distinct measurement.

    The stored times identify actual measured speed drops, never a refreshed
    deadline obtained by looking at the same old peak on another planning call.
    Historical bootstrap supports a fresh frozen-log replay with the same data.
    """
    actor_id, leader_id, now = int(actor.id), int(leader.id), history.last_time_s
    state = history._following_states.get(actor_id)
    if state is None or state["leader_id"] != leader_id:
        changed = state is not None and state["leader_id"] != leader_id
        if changed:
            history._following_invalidated_s[actor_id] = now-1e-7
        state = {"leader_id": leader_id,
            "last_scan_s": history._following_invalidated_s.get(actor_id, -math.inf),
            "proof_s": None, "response_started_s": None, "suspended": False,
            "reason": "no_measured_recent_braking", "speeds": []}
        history._following_states[actor_id] = state
    actor_past = history.recent_observations(actor)
    lead_past = {round(t, 8): (p, yaw) for t, p, yaw in history.recent_observations(leader)}

    def established_at(stamp):
        relation = []
        for time, position, yaw in actor_past:
            if not stamp-config["following_relation_window_s"]-1e-8 <= time <= stamp+1e-8:
                continue
            other = lead_past.get(round(time, 8))
            if other is None:
                return False
            relation.append((time, _observed_path_relation(history, actor, leader,
                time, position, yaw, other[0], other[1], carla_map, config)))
        return (len(relation) >= 3 and all(valid for _, valid in relation)
            and relation[-1][0]-relation[0][0] >= config["following_relation_window_s"]-1e-7)

    track = history._tracks.get(actor_id)
    for stamp, speed, verified in (() if track is None else track[3]):
        if stamp <= state["last_scan_s"]+1e-8:
            continue
        state["last_scan_s"] = stamp
        state["speeds"].append((stamp, speed))
        state["speeds"] = [(t, v) for t, v in state["speeds"]
            if stamp-t <= config["following_response_valid_s"]+1e-8]
        if not verified or not established_at(stamp):
            state.update(proof_s=None, response_started_s=None, suspended=False,
                reason="historical_leader_relation_unavailable", speeds=[(stamp, speed)])
            continue
        if state["proof_s"] is not None and stamp-state["proof_s"] > config["following_response_valid_s"]+1e-8:
            state.update(proof_s=None, response_started_s=None, suspended=False,
                reason="observed_response_expired", speeds=[(stamp, speed)])
        since = state["proof_s"]
        prior = [(t, v) for t, v in state["speeds"]
            if t < stamp-1e-8 and (since is None or t >= since-1e-8)]
        peak = max((v for _, v in prior), default=speed)
        minimum = config["following_speed_drop_mps"] if since is None else config["following_renew_drop_mps"]
        fraction = config["following_speed_drop_fraction"] if since is None else config["following_renew_drop_fraction"]
        if peak-speed >= minimum and peak-speed >= peak*fraction:
            if state["response_started_s"] is None or state["suspended"]:
                state["response_started_s"] = stamp
            state.update(proof_s=stamp, suspended=False, reason="new_independent_measured_braking")
        recent = [(t, v) for t, v in state["speeds"]
            if stamp-t <= config["following_reacceleration_window_s"]+1e-8]
        if (state["proof_s"] is not None and len(recent) >= 3
                and recent[-1][0]-recent[0][0] >= config["following_reacceleration_window_s"]-1e-7
                and recent[-1][1]-recent[0][1] >= config["following_reacceleration_mps"]
                and recent[-1][1]-recent[0][1] >= recent[0][1]*config["following_reacceleration_fraction"]
                and all(b[1] >= a[1]-.08 for a, b in zip(recent, recent[1:]))):
            state.update(suspended=True, response_started_s=None, reason="sustained_measured_reacceleration")
    valid = (state["proof_s"] is not None and not state["suspended"]
        and now-state["proof_s"] <= config["following_response_valid_s"]+1e-8)
    remaining = (max(0., config["following_reaction_s"]-(now-state["response_started_s"]))
        if valid and state["response_started_s"] is not None else config["following_reaction_s"])
    return valid, remaining, {"response_proof_time_s": state["proof_s"],
        "response_started_s": state["response_started_s"], "response_phase_reason": state["reason"],
        "response_deadline_s": None if state["proof_s"] is None else state["proof_s"]+config["following_response_valid_s"],
        "reaction_remaining_s": remaining, "reaction_configured_s": config["following_reaction_s"],
        "response_suspended": state["suspended"]}


def _following_response(actor, estimate, times, centers, headings, map_audit,
                        history, leader_actors, carla_map, config):
    """Evidence-gated bounded following, never a position clamp or right of way.

    A four-second independent CV forecast makes a normal braking rear follower
    pass through its already observed leader. Only an established relation and
    a measured recent braking response allow this conditional traffic forecast.
    The first reaction interval is the original physical forecast, and an actor
    too close/fast to stop still passes through the leader and remains a hazard.
    """
    rejected = {"applied": False, "reason": "no_verified_following_response"}
    if (history is None or carla_map is None or estimate.quality != "verified"
            or map_audit.get("map_status") != "explicit_connected_successors"
            or not str(actor.type_id).startswith("vehicle.")):
        if history is not None:
            previous = history._following_states.get(int(actor.id))
            known_leader = next((other for other in leader_actors
                if previous is not None and int(other.id) == previous["leader_id"]), None)
            if (known_leader is not None and estimate.quality == "verified"
                    and carla_map is not None):
                actor_past = history.recent_observations(actor, config["following_relation_window_s"])
                leader_past = {round(t,8):(p,y) for t,p,y in
                    history.recent_observations(known_leader, config["following_relation_window_s"])}
                proved = [(t, _observed_path_relation(history, actor, known_leader, t, p, y,
                    *leader_past[round(t,8)], carla_map, config)) for t,p,y in actor_past
                    if round(t,8) in leader_past]
                if (len(proved) >= 3 and all(value for _,value in proved)
                        and proved[-1][0]-proved[0][0] >= config["following_relation_window_s"]-1e-7):
                    # Preserve only the original finite evidence while the
                    # current forecast itself is disabled (e.g. a lateral
                    # velocity excursion). No phase is advanced or renewed.
                    return centers, headings, {"applied":False,
                        "reason":"current_prediction_unproven_observed_relation_preserved",
                        "response_proof_time_s":previous["proof_s"],
                        "response_deadline_s":None if previous["proof_s"] is None else
                            previous["proof_s"]+config["following_response_valid_s"]}
            history._following_states.pop(int(actor.id), None)
            history._following_invalidated_s[int(actor.id)] = history.last_time_s
        return centers, headings, rejected
    past = history.recent_observations(actor, config["following_response_window_s"])
    if len(past) < 4:
        return centers, headings, rejected
    interval_speeds = [float(np.linalg.norm(second[1][:2]-first[1][:2])/(second[0]-first[0]))
        for first, second in zip(past, past[1:]) if second[0]-first[0] >= config["minimum_difference_interval_s"]]
    peak = max(interval_speeds, default=0.)
    segments = np.diff(centers, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    stations = np.r_[0., np.cumsum(lengths)]
    if stations[-1] < .1:
        return centers, headings, rejected
    lane_keys = map_audit.get("connected_lane_keys", ())
    leaders = []
    previous_state = history._following_states.get(int(actor.id))
    uncertain_existing_leader = False
    for leader in leader_actors:
        if leader.id == actor.id or not str(leader.type_id).startswith("vehicle."):
            continue
        try:
            pose, box = leader.get_transform(), leader.bounding_box
            yaw = math.radians(float(pose.rotation.yaw))
            offset = getattr(box, "location", SimpleNamespace(x=0., y=0.))
            lead_xy = np.array([pose.location.x+offset.x*math.cos(yaw)-offset.y*math.sin(yaw),
                pose.location.y+offset.x*math.sin(yaw)+offset.y*math.cos(yaw)])
            fractions = np.clip(np.sum((lead_xy-centers[:-1])*segments, axis=1)/np.maximum(lengths**2,1e-12),0.,1.)
            projections = centers[:-1]+fractions[:,None]*segments
            index = int(np.argmin(np.linalg.norm(projections-lead_xy,axis=1)))
            tangent = segments[index]/max(lengths[index],1e-12)
            extension = float((lead_xy-projections[index])@tangent)
            distance = float(stations[index]+fractions[index]*lengths[index]+extension)
            lateral = abs(float((lead_xy-projections[index])@np.array([-tangent[1],tangent[0]])))
            if (distance <= .1 or lateral > config["following_lateral_tolerance_m"]
                    or abs(_angle(yaw-headings[index])) > config["following_heading_tolerance_rad"]):
                continue
            try:
                waypoint = carla_map.get_waypoint(pose.location, project_to_road=False)
            except TypeError:
                waypoint = carla_map.get_waypoint(pose.location)
            key = _lane_key(waypoint)
            if key is None:
                uncertain_existing_leader |= previous_state is not None and previous_state["leader_id"] == int(leader.id)
                continue
            if distance > stations[-1]+1e-6:
                # A leader's rear body can overlap the horizon even while its
                # center is just beyond it. Prove that extra route explicitly;
                # never extend a tangent across an unobserved junction branch.
                actor_pose = actor.get_transform()
                try:
                    start_waypoint = carla_map.get_waypoint(actor_pose.location, project_to_road=False)
                except TypeError:
                    start_waypoint = carla_map.get_waypoint(actor_pose.location)
                options = list(start_waypoint.next(distance))
                if len(options) != 1:
                    uncertain_existing_leader |= previous_state is not None and previous_state["leader_id"] == int(leader.id)
                    continue
                if (_lane_key(options[0]) != key
                        or math.hypot(options[0].transform.location.x-pose.location.x,
                                      options[0].transform.location.y-pose.location.y) > .6):
                    continue
            elif not any(tuple(value) == key for value in lane_keys[index:index+2] if value is not None):
                continue
            lead_past = {round(t,8):(p,angle) for t,p,angle in
                history.recent_observations(leader, config["following_relation_window_s"])}
            relation = []
            for stamp, position, angle in past:
                if history.last_time_s-stamp > config["following_relation_window_s"]+1e-8:
                    continue
                measured_lead = lead_past.get(round(stamp,8))
                if measured_lead is None:
                    continue
                lp, lyaw = measured_lead
                relation.append((stamp, _observed_path_relation(history, actor, leader,
                    stamp, position, angle, lp, lyaw, carla_map, config)))
            if (len(relation) < 3 or not all(valid for _,valid in relation)
                    or relation[-1][0]-relation[0][0] < config["following_relation_window_s"]-1e-7):
                continue
            lead_motion = history.motion_for(leader)
            lead_speed = max(0., lead_motion.vx*math.cos(yaw)+lead_motion.vy*math.sin(yaw))
            body_gap = distance-float(actor.bounding_box.extent.x)-float(box.extent.x)
            leaders.append((distance, int(leader.id), body_gap, lead_speed, relation, leader))
        except (AttributeError, RuntimeError, TypeError, ValueError, KeyError):
            continue
    if not leaders:
        if uncertain_existing_leader:
            # Prediction is disabled while the current connection is unknown.
            # Its original finite proof/phase times remain unchanged; a later
            # fresh proof of the same actual leader may reuse real observations.
            # A changed leader still invalidates them in _response_phase.
            return centers, headings, {"applied":False,
                "reason":"existing_leader_connection_unproven_prediction_disabled",
                "past_position_speed_peak_mps":peak,
                "response_proof_time_s":previous_state["proof_s"],
                "response_deadline_s":None if previous_state["proof_s"] is None else
                    previous_state["proof_s"]+config["following_response_valid_s"]}
        history._following_states.pop(int(actor.id), None)
        history._following_invalidated_s[int(actor.id)] = history.last_time_s
        return centers, headings, {"applied": False, "reason": "no_established_same_path_leader",
            "past_position_speed_peak_mps": peak}
    distance, leader_id, gap0, lead_speed, relation, leader = min(leaders, key=lambda item:(item[0], item[1]))
    supported, reaction, phase_audit = _response_phase(history, actor, leader, config, carla_map)
    if not supported:
        return centers, headings, {"applied":False,"reason":phase_audit["response_phase_reason"],
            "past_position_speed_peak_mps":peak, **phase_audit}
    speed, travel, lead_travel, lead_v = estimate.speed_mps, 0., 0., lead_speed
    travel_samples, accelerations = [0.], [0.]
    maximum_braking = config["following_max_braking_mps2"]
    # Substeps prevent a sampled acceleration from reversing a stopped actor.
    # A leader may itself brake within the same physical bound; assuming that
    # conservative minimum progress does not invent an empty road ahead.
    for start,end in zip(times,times[1:]):
        stamp, acceleration = float(start), 0.
        while stamp < float(end)-1e-10:
            dt = min(.02,float(end)-stamp)
            if stamp < reaction-1e-10:
                dt = min(dt,reaction-stamp)
                acceleration = 0.
            else:
                gap = gap0+lead_travel-travel
                desired_gap = (config["following_minimum_gap_m"]+speed*config["following_time_headway_s"]
                    + max(0.,speed*(speed-lead_v)/(2*math.sqrt(config["following_comfort_braking_mps2"]))))
                acceleration = -min(maximum_braking,(desired_gap/max(.05,gap))**2)
                if speed <= 1e-10:
                    acceleration = 0.
            effective_dt = min(dt,speed/max(-acceleration,1e-12)) if acceleration < 0 else dt
            travel += speed*effective_dt+.5*acceleration*effective_dt**2
            speed = max(0.,speed+acceleration*dt)
            lead_dt = min(dt,lead_v/maximum_braking)
            lead_travel += lead_v*lead_dt-.5*maximum_braking*lead_dt**2
            lead_v = max(0.,lead_v-maximum_braking*dt)
            stamp += dt
        travel_samples.append(travel)
        accelerations.append(acceleration)
    # Reparameterize only the already established connected physical path.
    # No actor is snapped to a leader or removed when braking is insufficient.
    travel_samples = np.asarray(travel_samples)
    if travel_samples[-1] > stations[-1]+1e-8:
        return centers,headings,{"applied":False,"reason":"bounded_travel_exceeds_proven_connected_path",
            "computed_travel_m":float(travel_samples[-1]),"proven_path_length_m":float(stations[-1])}
    adjusted = np.column_stack([np.interp(travel_samples,stations,centers[:,axis]) for axis in (0,1)])
    adjusted_headings = np.interp(travel_samples,stations,np.unwrap(headings))
    prefix = times <= reaction+1e-8
    adjusted[prefix],adjusted_headings[prefix] = centers[prefix],headings[prefix]
    return adjusted,adjusted_headings,{"applied": True,"reason": "observed_braking_with_established_same_path_leader",
        "leader_id": leader_id,"initial_body_gap_m": gap0,"leader_speed_mps": lead_speed,
        "past_position_speed_peak_mps": peak,"current_speed_mps": estimate.speed_mps,
        "response_window_s": config["following_response_window_s"],
        "relation_geometry":"unique_connected_road_station_and_local_normals",
        "relation_history_span_s": relation[-1][0]-relation[0][0],"relation_distinct_frames":len(relation),
        "reaction_s":reaction,"max_braking_mps2":maximum_braking,"position_clamp_used":False,
        **phase_audit,
        "travel_m":travel_samples.tolist(),"acceleration_mps2":accelerations,
        "unconditioned_centers_world_xy_m":centers.tolist()}


def predict_actor_motion(actor, estimate, times, carla_map=None, config=None, *,
                         history=None, leader_actors=()):
    """Forecast bbox centers/headings; all actors remain collision participants.

    Returns (N x 2 centers, N body yaws, audit). Map guidance is restricted to
    aligned vehicles; crossing vehicles and pedestrians retain measured world
    motion. A map does not make a different lane immune to collision checking.
    """
    config = validate_motion_config(config)
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or abs(times[0]) > 1e-8 or np.any(np.diff(times) <= 0)):
        raise ValueError("actor prediction times must start at zero and increase")
    pose, box = actor.get_transform(), actor.bounding_box
    origin = np.array([pose.location.x, pose.location.y], dtype=float)
    yaw = math.radians(float(pose.rotation.yaw))
    velocity = np.array([estimate.vx, estimate.vy], dtype=float)
    rate = estimate.yaw_rate_rad_s if str(actor.type_id).startswith("vehicle.") else 0.
    positions, headings = _turn_rate_path(origin, velocity, yaw, rate, times)
    mode = "observed_turn_rate" if abs(rate) >= 1e-5 else "constant_velocity"
    map_audit = {"map_status": "not_an_aligned_moving_vehicle"}
    if str(actor.type_id).startswith("vehicle.") and estimate.speed_mps >= .35:
        mapped, map_audit = _connected_map_path(actor, estimate, times, carla_map, config)
        if mapped is not None:
            positions, headings = mapped
            mode = "connected_lane" if map_audit["map_status"] == "explicit_connected_successors" else "connected_lane_then_observed_motion"
    if estimate.speed_mps < .02:
        mode = "stationary"
    offset = getattr(box, "location", SimpleNamespace(x=0., y=0.))
    c, s = np.cos(headings), np.sin(headings)
    centers = positions + np.column_stack((offset.x*c-offset.y*s, offset.x*s+offset.y*c))
    body_headings = headings + math.radians(float(getattr(getattr(box, "rotation", None), "yaw", 0.)))
    centers, body_headings, following = _following_response(actor,estimate,times,centers,body_headings,
        map_audit,history,leader_actors,carla_map,config)
    verified = estimate.quality == "verified" or mode == "connected_lane"
    uncertainty = times*(config["prediction_uncertainty_mps"] if verified else config["unverified_prediction_uncertainty_mps"])
    if mode == "stationary" or str(actor.type_id).startswith("static.prop."):
        uncertainty = np.zeros(len(times))
    audit = {"version": MOTION_MODEL_VERSION,
        "prediction_kind": "connected_lane_observed_turn_rate_and_following_response_not_future_GT",
        "mode": mode, "future_ground_truth_used": False,
        "velocity_source": estimate.source, "motion_quality": estimate.quality,
        "yaw_rate_rad_s": float(rate), "uncertainty_margin_m": uncertainty.tolist(),
        "times_s": times.tolist(),
        "centers_world_xy_m": centers.tolist(), "body_yaws_rad": body_headings.tolist(),
        "following_response": following,
        **map_audit}
    return centers, body_headings, audit
