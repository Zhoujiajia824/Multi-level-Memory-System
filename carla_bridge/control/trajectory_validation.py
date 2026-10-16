"""Reject malformed or grossly infeasible VLM plans before CARLA actuation.

Coordinates are metres in the capture-time ego frame (x forward, y left),
and times are seconds from capture. This is a conservative sanity check, not
a collision checker or a replacement planner. It never changes a rejected plan
into an allegedly successful route-following decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Mapping, Optional


@dataclass
class TrajectoryValidationResult:
    accepted: bool
    reasons: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "reasons": list(self.reasons),
                "details": dict(self.details)}


def _finite_number(value: Any) -> bool:
    # bool is a subclass of int, but is never a valid VLM coordinate/speed.
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


class TrajectoryValidator:
    """Validate one plan using the current speed and its execution horizon.

    Defaults are generous urban-driving sanity limits, not controller tuning:
    25 m/s speed, +4/-8 m/s² longitudinal and 6 m/s² lateral acceleration.
    The 1.5 m/s speed and 0.25 m position allowances absorb rounded VLM points.
    Lower target speeds do not invalidate the distance needed to decelerate
    from the measured current speed. Rejection must cause braking in the caller.
    """

    DEFAULTS = {
        "max_speed_mps": 25.0,
        "max_acceleration_mps2": 4.0,
        "max_deceleration_mps2": 8.0,
        "max_lateral_acceleration_mps2": 6.0,
        "speed_tolerance_mps": 1.5,
        "position_tolerance_m": 0.25,
        "max_initial_heading_deg": 75.0,
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        cfg = config or {}
        for key, default in self.DEFAULTS.items():
            value = cfg.get(key, default)
            if not _finite_number(value) or value < 0:
                raise ValueError(f"trajectory_validation.{key} must be finite and nonnegative")
            if key.startswith("max_") and value == 0:
                raise ValueError(f"trajectory_validation.{key} must be positive")
            setattr(self, key, float(value))
        if self.max_initial_heading_deg >= 90:
            raise ValueError("max_initial_heading_deg must be below 90 for forward-only control")

    def validate(self, trajectory: Any, target_speed: Any, current_speed: Any,
                 required_horizon_s: float, *, use_speed_profile: bool = False) -> TrajectoryValidationResult:
        """Return JSON-friendly reasons and measurements; do not mutate inputs."""
        try:
            return self._validate(trajectory, target_speed, current_speed, required_horizon_s, use_speed_profile)
        except ArithmeticError:
            # A JSON number can be finite but far beyond physical arithmetic.
            return TrajectoryValidationResult(False, ["numeric_range_exceeded"], {})

    def _validate(self, trajectory, target_speed, current_speed, required_horizon_s, use_speed_profile=False):
        result = TrajectoryValidationResult(True, details={"violations": []})

        def reject(code: str, index: Optional[int] = None, **values: Any) -> None:
            result.accepted = False
            if code not in result.reasons:
                result.reasons.append(code)
            violation = {"code": code, **values}
            if index is not None:
                violation["point_index"] = index
            # Keep diagnostics bounded for a bad plan with many bad segments.
            if len(result.details["violations"]) < 20:
                result.details["violations"].append(violation)

        for name, value in (("target_speed", target_speed), ("current_speed", current_speed),
                            ("required_horizon_s", required_horizon_s)):
            if not _finite_number(value) or value < 0:
                reject(f"invalid_{name}")
        if not result.accepted:
            return result
        target_speed, current_speed = float(target_speed), float(current_speed)
        if target_speed > self.max_speed_mps:
            reject("target_speed_exceeds_limit", speed_mps=target_speed,
                   limit_mps=self.max_speed_mps)
        if not isinstance(trajectory, (list, tuple)) or not trajectory:
            reject("empty_or_malformed_trajectory")
            return result

        points = []
        previous_t = -1.0
        for i, point in enumerate(trajectory):
            if not isinstance(point, Mapping):
                reject("malformed_point", i)
                continue
            if any(not _finite_number(point.get(key)) for key in ("t", "x", "y")):
                reject("nonfinite_or_missing_coordinate", i)
                continue
            t, x, y = (float(point[key]) for key in ("t", "x", "y"))
            if t < 0 or t <= previous_t:
                reject("non_monotonic_time", i)
            previous_t = t
            speed = point.get("optional_v")
            if use_speed_profile and speed is None:
                reject("missing_speed_profile", i)
            if speed is not None:
                if not _finite_number(speed) or speed < 0:
                    reject("invalid_waypoint_speed", i)
                elif speed > max(current_speed, self.max_speed_mps) + self.speed_tolerance_mps:
                    reject("waypoint_speed_exceeds_limit", i, speed_mps=float(speed))
            points.append((t, x, y, speed))
        # Do not perform geometry with invalid numbers or time intervals.
        if not result.accepted:
            return result

        horizon = points[-1][0]
        result.details["horizon_s"] = horizon
        result.details["required_horizon_s"] = float(required_horizon_s)
        if horizon + 1e-6 < required_horizon_s:
            reject("insufficient_horizon", horizon_s=horizon,
                   required_horizon_s=float(required_horizon_s))
        if points[0][0] == 0 and math.hypot(points[0][1], points[0][2]) > self.position_tolerance_m:
            reject("nonzero_origin", 0)

        # Keep a measured high current speed in the envelope: a braking plan must
        # remain possible even if the car is already above the configured limit.
        cruise_cap = max(current_speed, target_speed)
        if use_speed_profile:
            # PDM target_speed is the terminal speed, not a cap for the whole
            # horizon. A valid plan may accelerate then stop at a red light.
            # All profile values were checked above; segment dynamics below
            # still enforce acceleration, braking and geometry consistency.
            cruise_cap = max(cruise_cap, max(float(point[3]) for point in points))
            result.details["speed_profile_peak_mps"] = cruise_cap
        previous = (0.0, 0.0, 0.0, current_speed)
        previous_mean_speed = current_speed
        previous_mid_t = 0.0
        previous_heading = 0.0
        have_heading = False
        path_length = 0.0
        max_segment_speed = 0.0
        for i, point in enumerate(points):
            t, x, y, waypoint_speed = point
            pt, px, py, previous_waypoint_speed = previous
            dt = t - pt
            if dt <= 0:  # A validated optional t=0 anchor.
                previous = point
                if waypoint_speed is not None and abs(waypoint_speed - current_speed) > self.speed_tolerance_mps:
                    reject("waypoint_speed_inconsistent", i)
                continue
            dx, dy = x - px, y - py
            distance = math.hypot(dx, dy)
            mean_speed = distance / dt
            mid_t = (t + pt) / 2.0
            derivative_dt = mid_t - previous_mid_t
            path_length += distance
            max_segment_speed = max(max_segment_speed, mean_speed)
            speed_slack = self.speed_tolerance_mps + self.position_tolerance_m / dt
            if not all(math.isfinite(value) for value in
                       (distance, mean_speed, mid_t, derivative_dt, path_length, speed_slack)):
                reject("numeric_range_exceeded", i)
                return result
            if mean_speed > cruise_cap + speed_slack:
                reject("distance_speed_mismatch", i, segment_speed_mps=round(mean_speed, 4),
                       allowed_speed_mps=round(cruise_cap + speed_slack, 4))
            if mean_speed > max(current_speed, self.max_speed_mps) + speed_slack:
                reject("segment_speed_exceeds_limit", i)

            # Prefix distance must lie within the reachability envelope from the
            # measured speed. The lower bound permits maximum emergency braking.
            time_to_cap = max(0.0, cruise_cap - current_speed) / self.max_acceleration_mps2
            accel_time = min(t, time_to_cap)
            max_distance = (current_speed * accel_time
                            + 0.5 * self.max_acceleration_mps2 * accel_time ** 2
                            + cruise_cap * max(0.0, t - accel_time))
            braking_time = min(t, current_speed / self.max_deceleration_mps2)
            min_distance = (current_speed * braking_time
                            - 0.5 * self.max_deceleration_mps2 * braking_time ** 2)
            distance_slack = self.position_tolerance_m + self.speed_tolerance_mps * t
            if not all(math.isfinite(value) for value in
                       (max_distance, min_distance, distance_slack)):
                reject("numeric_range_exceeded", i)
                return result
            if path_length > max_distance + distance_slack:
                reject("unreachable_distance", i, distance_m=round(path_length, 4),
                       maximum_m=round(max_distance + distance_slack, 4))
            if path_length + distance_slack < min_distance:
                reject("instantaneous_stop", i, distance_m=round(path_length, 4),
                       minimum_m=round(max(0.0, min_distance - distance_slack), 4))

            speed_change = mean_speed - previous_mean_speed
            if speed_change > self.max_acceleration_mps2 * derivative_dt + self.speed_tolerance_mps:
                reject("excessive_acceleration", i)
            if -speed_change > self.max_deceleration_mps2 * derivative_dt + self.speed_tolerance_mps:
                reject("excessive_deceleration", i)

            if distance > 0.05:
                heading = math.atan2(dy, dx)
                change = math.atan2(math.sin(heading - previous_heading),
                                    math.cos(heading - previous_heading))
                if not have_heading and abs(heading) > math.radians(self.max_initial_heading_deg):
                    reject("not_forward_drivable", i)
                if have_heading and abs(change) > math.pi / 2:
                    reject("direction_reversal", i)
                lateral_velocity_change = abs(mean_speed * math.sin(change))
                if lateral_velocity_change > (self.max_lateral_acceleration_mps2 * derivative_dt
                                               + self.speed_tolerance_mps):
                    reject("excessive_lateral_acceleration", i)
                previous_heading = heading
                have_heading = True

            if waypoint_speed is not None:
                if previous_waypoint_speed is not None:
                    declared_mean = (previous_waypoint_speed + waypoint_speed) / 2.0
                    if abs(mean_speed - declared_mean) > speed_slack:
                        reject("waypoint_speed_inconsistent", i)
                    declared_change = waypoint_speed - previous_waypoint_speed
                    bound = (self.max_acceleration_mps2 if declared_change >= 0
                             else self.max_deceleration_mps2)
                    if abs(declared_change) > bound * dt + self.speed_tolerance_mps:
                        reject("waypoint_acceleration_inconsistent", i)
                else:
                    # A lone supplied endpoint speed is still constrained by
                    # the segment's average and the acceleration bounds.
                    bound = max(self.max_acceleration_mps2, self.max_deceleration_mps2)
                    if abs(waypoint_speed - mean_speed) > bound * dt / 2 + speed_slack:
                        reject("waypoint_speed_inconsistent", i)

            previous, previous_mean_speed, previous_mid_t = point, mean_speed, mid_t

        result.details["path_length_m"] = round(path_length, 4)
        result.details["max_segment_speed_mps"] = round(max_segment_speed, 4)
        return result
