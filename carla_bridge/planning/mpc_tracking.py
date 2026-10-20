"""Execute the first optimized MPC inputs; never substitute a PDM/LQR rollout."""
from __future__ import annotations

import copy
import math
from numbers import Real


TRACKING_SOURCE = "nonlinear_mpc_optimized_controls"
MAX_PROFILE_LIFETIME_S = 0.5


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _copy_control(control):
    try:
        return copy.copy(control)
    except (TypeError, RuntimeError):
        return type(control)(**{field: getattr(control, field) for field in (
            "throttle", "steer", "brake", "hand_brake", "reverse", "manual_gear_shift", "gear")})


def override_mpc_tracking(control, diagnostics, *, now_s, speed_mps, steering_model):
    """Use zero-order-held optimized wheel inputs and the PID speed-profile control.

    Inputs are CARLA right-positive equivalent bicycle angles. Array index zero
    controls [issued_at, issued_at + dt), unlike PDM's future steering states.
    Invalid/expired profiles brake explicitly; there is no silent PP fallback.
    """
    def rejected(reason):
        stopped = _copy_control(control)
        stopped.throttle, stopped.brake, stopped.steer = 0., 1., 0.
        return stopped, {"applied": False, "reason": reason, "failsafe_brake": True}

    profile = diagnostics.get("mpc_tracking") if isinstance(diagnostics, dict) else None
    if not isinstance(profile, dict) or profile.get("active") is not True:
        return rejected("missing_mpc_profile")
    if profile.get("source") != TRACKING_SOURCE:
        return rejected("untrusted_mpc_profile")
    issued, expires, dt = (profile.get("issued_at_s"), profile.get("expires_at_s"),
                           profile.get("sample_interval_s"))
    if not all(_finite(x) for x in (issued, expires, dt, now_s, speed_mps)):
        return rejected("invalid_mpc_timing_or_speed")
    if speed_mps < 0 or not 0 < dt <= .5 or not 0 < expires-issued <= MAX_PROFILE_LIFETIME_S+1e-9:
        return rejected("invalid_mpc_profile_lifetime")
    age = now_s-issued
    if age < -1e-8 or now_s >= expires-1e-9:
        return rejected("mpc_profile_expired_or_future")
    angles = profile.get("bicycle_angles_rad")
    if not isinstance(angles, (list, tuple)) or not angles or not all(_finite(v) for v in angles):
        return rejected("invalid_mpc_steering_inputs")
    index = int(math.floor((max(0., age)+1e-9)/dt))
    if index >= len(angles):
        return rejected("mpc_profile_exhausted")
    angle = float(angles[index])
    try:
        maximum = steering_model.limit_at(speed_mps)
        if abs(angle) > maximum+1e-6:
            return rejected("mpc_angle_exceeds_vehicle_limit")
        normalized = steering_model.bicycle_to_normalized(angle, speed_mps)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return rejected("mpc_steering_conversion_failed")
    if not _finite(normalized) or abs(normalized) > 1+1e-9:
        return rejected("invalid_mpc_normalized_steer")
    result = _copy_control(control)
    result.steer = float(normalized)
    return result, {"applied": True, "reason": "optimized_mpc_input", "source": TRACKING_SOURCE,
                    "profile_age_s": float(max(0., age)), "profile_sample_index": index,
                    "optimized_bicycle_angle_rad": angle, "target_normalized_steer": float(normalized)}
