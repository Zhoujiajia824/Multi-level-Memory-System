"""All PDM modes use only a fresh selected native rollout's lateral commands."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from carla_bridge.control.steering_model import SteeringModel
from carla_bridge.planning.native_tracking import (
    TRACKING_SOURCE, override_native_tracking,
)


def control():
    return NS(throttle=.2, brake=.4, steer=-.7, hand_brake=False, reverse=False,
              manual_gear_shift=False, gear=1)


def diagnostics():
    return {"lateral_control": "native_lqr", "native_tracking": {
                "active": True, "source": TRACKING_SOURCE, "issued_at_s": 2.,
                "expires_at_s": 2.3, "sample_interval_s": .1,
                "bicycle_angles_rad": [.0, -.1, -.2, -.3]}}


def apply(original=None, diag=None, **kwargs):
    return override_native_tracking(
        original if original is not None else control(),
        diag if diag is not None else diagnostics(),
        **dict(dict(now_s=2., speed_mps=1., steering_model=SteeringModel(.6)), **kwargs))


@pytest.mark.parametrize("now,index,angle", [
    (2., 1, -.1), (2.05, 1, -.1), (2.1, 2, -.2),
    (2.199, 2, -.2), (2.2, 3, -.3), (2.299, 3, -.3),
])
def test_uses_next_native_state_without_interpolation(now, index, angle):
    original, diag = control(), diagnostics()
    unchanged = deepcopy(diag)
    result, audit = apply(original, diag, now_s=now)
    assert result is not original
    assert result.steer == pytest.approx(-angle / .6)
    assert audit["applied"] and audit["profile_sample_index"] == index
    assert audit["predicted_bicycle_angle_rad"] == angle
    assert vars(result) == dict(vars(original), steer=result.steer)
    assert vars(original) == vars(control()) and diag == unchanged


def test_mapping_receives_signed_native_angle_and_actual_speed():
    calls = []
    class Model:
        max_steer_rad = .8
        def bicycle_to_normalized(self, angle, speed):
            calls.append((angle, speed))
            return .123
    result, audit = apply(steering_model=Model(), speed_mps=7.25)
    assert calls == [(.1, 7.25)]
    assert result.steer == .123 and audit["target_normalized_steer"] == .123


def test_physx_ackermann_and_speed_curve_use_physical_mapping():
    model = SteeringModel(1.2, [(0., 1.), (36., .5)], wheelbase_m=3., front_track_m=1.6)
    result, audit = apply(steering_model=model, speed_mps=5., now_s=2.2)
    assert audit["applied"]
    assert result.steer == pytest.approx(model.bicycle_to_normalized(.3, 5.))
    assert result.steer != pytest.approx(.3 / model.max_steer_rad)
    assert model.normalized_to_bicycle(result.steer, 5.) == pytest.approx(.3)


@pytest.mark.parametrize("policy", ["selective", "shadow", "always", "off", None])
@pytest.mark.parametrize("enabled", [True, False])
def test_semantic_policy_never_changes_native_execution(policy, enabled):
    original, diag = control(), diagnostics()
    diag["semantic_policy"] = policy
    diag["semantic_enabled"] = enabled
    diag["recovery"] = {"authorized": False, "expanded": False}
    result, audit = apply(original, diag)
    assert result is not original and audit["applied"]
    assert result.steer == pytest.approx(1. / 6)


@pytest.mark.parametrize("mode", ["pure_pursuit", "other", "", None])
def test_non_native_controller_keeps_original_control(mode):
    original, diag = control(), diagnostics()
    diag["lateral_control"] = mode
    result, audit = apply(original, diag)
    assert result is original and audit["reason"] == "native_lqr_control_not_selected"


def test_shadow_uses_the_profile_of_its_actual_baseline_selection():
    original, diag = control(), diagnostics()
    diag["semantic_policy"] = "shadow"
    diag["selected_id"] = "center:YIELD"
    diag["shadow_selected_id"] = "recovery@center:FOLLOW"
    # The product planner binds native_tracking to the actual selected native
    # baseline; a hypothetical recovery profile must never replace it.
    diag["native_tracking"]["bicycle_angles_rad"] = [0., .15, .2]
    diag["recovery"] = {"authorized": True, "expanded": True, "tracking": {
        "active": True, "source": TRACKING_SOURCE, "issued_at_s": 2.,
        "expires_at_s": 2.2, "sample_interval_s": .1,
        "bicycle_angles_rad": [0., -.5, -.5]}}
    result, audit = apply(original, diag)
    assert audit["applied"]
    assert result.steer == pytest.approx(-.25)


def test_old_recovery_hint_cannot_stand_in_for_a_selected_native_profile():
    original, diag = control(), diagnostics()
    old_profile = diag.pop("native_tracking")
    diag["recovery"] = {"authorized": True, "expanded": True, "tracking": old_profile}
    result, audit = apply(original, diag)
    assert result is original and audit["reason"] == "no_active_native_tracking"


@pytest.mark.parametrize("key,value,reason", [
    ("active", False, "no_active_native_tracking"),
    ("active", 1, "no_active_native_tracking"),
    ("source", "model", "untrusted_tracking_source"),
    ("sample_interval_s", .2, "unsupported_sample_interval"),
    ("sample_interval_s", 0., "unsupported_sample_interval"),
    ("expires_at_s", 2., "invalid_profile_lifetime"),
    ("expires_at_s", 2.300001, "invalid_profile_lifetime"),
])
def test_disabled_and_malformed_hints_fail_closed(key, value, reason):
    original, diag = control(), diagnostics()
    target = diag["native_tracking"]
    target[key] = value
    result, audit = apply(original, diag)
    assert result is original and audit == {"applied": False, "reason": reason}


@pytest.mark.parametrize("field", ["active", "source", "issued_at_s", "expires_at_s", "sample_interval_s", "bicycle_angles_rad"])
def test_every_native_profile_field_is_required(field):
    original, diag = control(), diagnostics()
    del diag["native_tracking"][field]
    result, audit = apply(original, diag)
    assert result is original and not audit["applied"]


@pytest.mark.parametrize("field", ["issued_at_s", "expires_at_s", "sample_interval_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "0.1", True, None])
def test_profile_timing_must_be_finite_numeric(field, value):
    original, diag = control(), diagnostics()
    diag["native_tracking"][field] = value
    result, audit = apply(original, diag)
    assert result is original and audit["reason"] == "invalid_profile_timing"


@pytest.mark.parametrize("angles,reason", [
    ([], "invalid_steering_profile"), ([0.], "invalid_steering_profile"),
    ([0.] * 5, "invalid_steering_profile"), (None, "invalid_steering_profile"),
    ([0., float("nan")], "invalid_steering_profile"),
    ([float("inf"), .1], "invalid_steering_profile"),
    ([0., True], "invalid_steering_profile"),
    ([0., ".1"], "invalid_steering_profile"),
    ([0., .600001], "profile_exceeds_vehicle_steering_limit"),
    ([0., -.600001], "profile_exceeds_vehicle_steering_limit"),
])
def test_entire_steering_profile_is_validated(angles, reason):
    original, diag = control(), diagnostics()
    diag["native_tracking"]["bicycle_angles_rad"] = angles
    result, audit = apply(original, diag)
    assert result is original and audit["reason"] == reason


@pytest.mark.parametrize("now,reason", [
    (1.999, "tracking_profile_from_future"),
    (2.3, "tracking_profile_expired"),
    (2.31, "tracking_profile_expired"),
    (float("nan"), "invalid_measured_state"),
])
def test_expired_or_invalid_time_returns_original_control(now, reason):
    original = control()
    result, audit = apply(original, now_s=now)
    assert result is original and audit["reason"] == reason


def test_short_profile_never_holds_last_sample_into_another_tick():
    original, diag = control(), diagnostics()
    diag["native_tracking"]["bicycle_angles_rad"] = [0., -.1]
    first, audit = apply(original, diag)
    assert audit["applied"]
    later, audit = apply(original, diag, now_s=2.1)
    assert later is original and audit["reason"] == "tracking_profile_has_no_future_sample"


def test_ttl_shorter_than_profile_still_expires():
    original, diag = control(), diagnostics()
    diag["native_tracking"]["expires_at_s"] = 2.15
    result, audit = apply(original, diag, now_s=2.15)
    assert result is original and audit["reason"] == "tracking_profile_expired"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1", None])
def test_measured_speed_must_be_finite(value):
    original = control()
    result, audit = apply(original, speed_mps=value)
    assert result is original and audit["reason"] == "invalid_measured_state"


@pytest.mark.parametrize("model,reason", [
    (None, "invalid_vehicle_steering_model"),
    (NS(max_steer_rad=float("nan")), "invalid_vehicle_steering_model"),
    (NS(max_steer_rad=2.), "invalid_vehicle_steering_model"),
    (NS(max_steer_rad=.6), "steering_conversion_failed"),
    (NS(max_steer_rad=.6, bicycle_to_normalized=lambda a, s: float("nan")), "invalid_normalized_steering"),
    (NS(max_steer_rad=.6, bicycle_to_normalized=lambda a, s: 1.1), "invalid_normalized_steering"),
    (NS(max_steer_rad=.6, bicycle_to_normalized=lambda a, s: True), "invalid_normalized_steering"),
])
def test_invalid_vehicle_mapping_never_produces_a_control(model, reason):
    original = control()
    result, audit = apply(original, steering_model=model)
    assert result is original and audit["reason"] == reason


def test_ego_position_does_not_prevent_tracking_a_moving_plan():
    result, audit = apply(ego_xy=(400., -400.), speed_mps=8.)
    assert audit["applied"] and result.throttle == .2 and result.brake == .4


def test_real_carla_control_preserves_every_longitudinal_and_gear_field():
    carla = pytest.importorskip("carla")
    original = carla.VehicleControl(throttle=.2, brake=.4, steer=-.7, hand_brake=True,
        reverse=True, manual_gear_shift=True, gear=-1)
    result, audit = apply(original)
    assert audit["applied"] and result is not original
    assert isinstance(result, carla.VehicleControl)
    assert result.steer == pytest.approx(1. / 6)
    for field in ("throttle", "brake", "hand_brake", "reverse", "manual_gear_shift", "gear"):
        assert getattr(result, field) == getattr(original, field)
    assert original.steer == pytest.approx(-.7)


def test_control_copy_failure_returns_original():
    class CannotCopy:
        def __copy__(self):
            raise RuntimeError("not copyable")
    original = CannotCopy()
    result, audit = apply(original)
    assert result is original and audit["reason"] == "control_copy_failed"
