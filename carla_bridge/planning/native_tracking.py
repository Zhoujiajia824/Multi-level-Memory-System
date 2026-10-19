"""Execute the selected native rollout's short steering profile in CARLA.

This is a narrow execution adapter, not a trajectory generator.  The product
planner supplies steering states from the same native simulated rollout it
scored and selected.  Model text cannot provide this profile.  All PDM semantic
policies share this controller; historical Pure Pursuit remains an explicit
comparison mode.  Longitudinal controls retain their existing execution path.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import math
from numbers import Real


TRACKING_SOURCE = "native_pdm_scored_rollout"
SAMPLE_INTERVAL_S = 0.1
MAX_PROFILE_LIFETIME_S = 0.3


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _copy_control(control):
    try:
        return copy.copy(control)
    except (TypeError, RuntimeError):
        # CARLA VehicleControl does not implement pickle/copy.  Its public
        # constructor preserves every control field while changing no actor.
        fields = ("throttle", "steer", "brake", "hand_brake", "reverse",
                  "manual_gear_shift", "gear")
        return type(control)(**{field: getattr(control, field) for field in fields})


def override_native_tracking(control, diagnostics, *, now_s, speed_mps,
                             steering_model, ego_xy=None):
    """Return a copied steering command and audit, or the unchanged input.

    ``native_tracking.bicycle_angles_rad`` includes t=0 followed by up to
    three future states spaced at exactly 0.1 s.  At issue time the next state
    (index 1) supplies this tick's steering; at age 0.1 s index 2 supplies it.
    Values are never interpolated, extrapolated or held beyond their lifetime.
    PDM bicycle angles are left-positive; CARLA normalized steer is right-
    positive, so conversion receives the negated native angle.

    Call only for normal tracker controls, after direct-brake/error handling.
    ``ego_xy`` is accepted for the shared execution interface; moving tracking
    does not impose a stationary-position constraint.
    """
    def reject(reason):
        return control, {"applied": False, "reason": reason}

    if not isinstance(diagnostics, Mapping):
        return reject("invalid_diagnostics")
    if diagnostics.get("lateral_control") != "native_lqr":
        return reject("native_lqr_control_not_selected")
    profile = diagnostics.get("native_tracking")
    if not isinstance(profile, Mapping) or profile.get("active") is not True:
        return reject("no_active_native_tracking")
    if profile.get("source") != TRACKING_SOURCE:
        return reject("untrusted_tracking_source")
    if not _number(now_s) or not _number(speed_mps):
        return reject("invalid_measured_state")
    issued, expires, dt = (profile.get("issued_at_s"), profile.get("expires_at_s"),
                           profile.get("sample_interval_s"))
    if not all(_number(value) for value in (issued, expires, dt)):
        return reject("invalid_profile_timing")
    if abs(dt - SAMPLE_INTERVAL_S) > 1e-9:
        return reject("unsupported_sample_interval")
    if not 0.0 < expires - issued <= MAX_PROFILE_LIFETIME_S + 1e-9:
        return reject("invalid_profile_lifetime")
    age = now_s - issued
    if age < 0.0:
        return reject("tracking_profile_from_future")
    if now_s >= expires or age > MAX_PROFILE_LIFETIME_S:
        return reject("tracking_profile_expired")
    angles = profile.get("bicycle_angles_rad")
    if not isinstance(angles, (list, tuple)) or not 2 <= len(angles) <= 4:
        return reject("invalid_steering_profile")
    if not all(_number(angle) for angle in angles):
        return reject("invalid_steering_profile")
    maximum = getattr(steering_model, "max_steer_rad", None)
    if not _number(maximum) or not 0.0 < maximum < math.pi / 2:
        return reject("invalid_vehicle_steering_model")
    if any(abs(angle) > maximum + 1e-9 for angle in angles):
        return reject("profile_exceeds_vehicle_steering_limit")
    # Permit floating-point simulation-clock noise at exact tick boundaries,
    # without interpolating or shifting a sample by a meaningful fraction tick.
    index = math.floor((age + 1e-9) / SAMPLE_INTERVAL_S) + 1
    if index >= len(angles):
        return reject("tracking_profile_has_no_future_sample")
    predicted_angle = float(angles[index])
    try:
        steer = steering_model.bicycle_to_normalized(-predicted_angle, speed_mps)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return reject("steering_conversion_failed")
    if not _number(steer) or not -1.0 <= steer <= 1.0:
        return reject("invalid_normalized_steering")
    try:
        tracked = _copy_control(control)
        if tracked is control:
            return reject("control_copy_failed")
        tracked.steer = float(steer)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return reject("control_copy_failed")
    return tracked, {"applied": True, "reason": "native_scored_rollout_tracking",
                     "source": TRACKING_SOURCE, "profile_age_s": float(age),
                     "profile_sample_index": index,
                     "predicted_bicycle_angle_rad": predicted_angle,
                     "target_normalized_steer": float(steer)}
