"""Selected native steering reaches the actor and the measured control audit."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import carla
import pytest

from carla_bridge.control.steering_model import SteeringModel
from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.planning.native_tracking import TRACKING_SOURCE
from carla_bridge.tests.test_control_audit import audited_loop, records
from src.vla_memory.common.config import Config


def native_loop(tmp_path, *, policy="selective", semantics=True):
    loop, ego = audited_loop(tmp_path)
    loop.planner_backend = "pdm_closed"
    loop.pdm_planner = NS(steering_model=SteeringModel(1.2, [(0., 1.), (36., .5)],
        wheelbase_m=3., front_track_m=1.6))
    loop._elapsed = 2.
    loop._pdm_plan_count = 7
    loop._last_plan_diagnostics = {
        "lateral_control": "native_lqr", "semantic_policy": policy,
        "semantic_enabled": semantics, "selected_id": "center:FOLLOW",
        "recovery": {"authorized": False, "expanded": False},
        "native_tracking": {"active": True, "source": TRACKING_SOURCE,
            "issued_at_s": 2., "expires_at_s": 2.3, "sample_interval_s": .1,
            "bicycle_angles_rad": [.0, -.1, -.2, -.3]}}
    loop.tracker.last_control_diagnostics = {"target_speed_mps": 6., "reason": "speed_tracking"}
    ego.apply_control.side_effect = lambda command: setattr(ego.get_control, "return_value", command)
    ego.get_velocity.return_value = carla.Vector3D(x=3., y=4.)
    return loop, ego


def original_control():
    return carla.VehicleControl(throttle=.23, brake=.17, steer=-.6, gear=-1,
        reverse=True, manual_gear_shift=True, hand_brake=True)


@pytest.mark.parametrize("policy,semantics", [
    ("selective", True), ("selective", False), ("always", True),
    ("shadow", True), ("off", False),
])
def test_tracker_applies_same_native_lateral_law_with_or_without_model(tmp_path, policy, semantics):
    loop, ego = native_loop(tmp_path, policy=policy, semantics=semantics)
    original = original_control()
    loop._apply_control(ego, original)
    actual = ego.apply_control.call_args.args[0]
    expected = loop.pdm_planner.steering_model.bicycle_to_normalized(.1, 5.)
    assert actual is not original and actual.steer == pytest.approx(expected)
    assert original.steer == pytest.approx(-.6)
    for field in ("throttle", "brake", "gear", "reverse", "manual_gear_shift", "hand_brake"):
        assert getattr(actual, field) == getattr(original, field)
    loop._elapsed = 2.1
    loop._record_tick(ego)
    audit = records(loop)[0]
    assert audit["command"]["native_tracking"]["applied"]
    assert audit["command"]["native_tracking"]["profile_sample_index"] == 1
    assert audit["command"]["native_tracking"]["predicted_bicycle_angle_rad"] == -.1
    assert audit["command"]["steer"] == pytest.approx(expected)
    assert audit["applied_control"]["steer"] == pytest.approx(expected)
    assert audit["applied_control"]["throttle"] == original.throttle
    assert audit["applied_control"]["brake"] == original.brake
    assert audit["command"]["source"] == "tracker"


@pytest.mark.parametrize("source", ["direct_brake", "capture_failure", "error", "planner_error"])
def test_direct_brake_and_error_sources_cannot_be_overridden(tmp_path, source):
    loop, ego = native_loop(tmp_path)
    original = carla.VehicleControl(throttle=0., brake=1., steer=0.)
    loop._apply_control(ego, original, source=source)
    assert ego.apply_control.call_args.args[0] is original
    assert loop._last_control_command["native_tracking"] is None
    assert loop._last_control_command["tracking"] is None
    assert loop._last_control_command["brake"] == 1.
    assert loop._last_control_command["steer"] == 0.


def test_apply_brake_uses_direct_path_even_with_valid_native_hint(tmp_path):
    loop, ego = native_loop(tmp_path)
    loop._apply_brake()
    command = ego.apply_control.call_args.args[0]
    assert command.brake == 1. and command.throttle == 0. and command.steer == 0.
    assert loop._last_control_command["source"] == "direct_brake"
    assert loop._last_control_command["native_tracking"] is None


def test_two_control_ticks_use_native_future_samples_one_and_two(tmp_path):
    loop, ego = native_loop(tmp_path)
    original = original_control()
    loop._apply_control(ego, original)
    loop._elapsed = 2.1
    loop._record_tick(ego)
    loop._apply_control(ego, original)
    loop._elapsed = 2.2
    loop._record_tick(ego)
    first, second = records(loop)
    assert first["command"]["native_tracking"]["profile_sample_index"] == 1
    assert second["command"]["native_tracking"]["profile_sample_index"] == 2
    for row, angle in ((first, .1), (second, .2)):
        expected = loop.pdm_planner.steering_model.bicycle_to_normalized(angle, 5.)
        assert row["command"]["steer"] == pytest.approx(expected)
        assert row["applied_control"]["steer"] == pytest.approx(expected)
        assert row["command"]["throttle"] == original.throttle
        assert row["command"]["brake"] == original.brake


def test_shadow_executes_actual_baseline_rollout_instead_of_hypothetical_recovery(tmp_path):
    loop, ego = native_loop(tmp_path, policy="shadow")
    diagnostics = loop._last_plan_diagnostics
    diagnostics["selected_id"] = "center:YIELD"
    diagnostics["shadow_selected_id"] = "recovery@center:FOLLOW"
    diagnostics["native_tracking"]["bicycle_angles_rad"] = [0., .15, .2]
    diagnostics["recovery"] = {"authorized": True, "expanded": True, "tracking": {
        "active": True, "source": TRACKING_SOURCE, "issued_at_s": 2.,
        "expires_at_s": 2.2, "sample_interval_s": .1,
        "bicycle_angles_rad": [0., -.5, -.5]}}
    loop._apply_control(ego, original_control())
    actual = ego.apply_control.call_args.args[0]
    assert actual.steer == pytest.approx(loop.pdm_planner.steering_model.bicycle_to_normalized(-.15, 5.))
    assert loop._last_control_command["native_tracking"]["predicted_bicycle_angle_rad"] == .15


@pytest.mark.parametrize("case,reason", [
    ("expired", "tracking_profile_expired"),
    ("future", "tracking_profile_from_future"),
    ("source", "untrusted_tracking_source"),
    ("numeric", "invalid_steering_profile"),
    ("limit", "profile_exceeds_vehicle_steering_limit"),
    ("pure_pursuit", "native_lqr_control_not_selected"),
])
def test_unusable_hint_preserves_tracker_control_and_records_rejection(tmp_path, case, reason):
    loop, ego = native_loop(tmp_path)
    diagnostics, original = loop._last_plan_diagnostics, original_control()
    profile = diagnostics["native_tracking"]
    if case == "expired":
        loop._elapsed = 2.3
    elif case == "future":
        loop._elapsed = 1.9
    elif case == "source":
        profile["source"] = "model"
    elif case == "numeric":
        profile["bicycle_angles_rad"] = [0., float("nan")]
    elif case == "limit":
        profile["bicycle_angles_rad"] = [0., 1.5]
    else:
        diagnostics["lateral_control"] = "pure_pursuit"
    loop._apply_control(ego, original)
    assert ego.apply_control.call_args.args[0] is original
    assert loop._last_control_command["native_tracking"] == {"applied": False, "reason": reason}
    assert loop._last_control_command["steer"] == original.steer


@pytest.mark.parametrize("backend", ["trajectory", "model", None])
def test_non_pdm_backend_never_executes_native_hint(tmp_path, backend):
    loop, ego = native_loop(tmp_path)
    loop.planner_backend = backend
    original = original_control()
    loop._apply_control(ego, original)
    assert ego.apply_control.call_args.args[0] is original
    assert loop._last_control_command["native_tracking"] is None


def tracking_config(*, interval=.2, delta=.1, lateral="native_lqr", backend="pdm_closed"):
    return Config(overrides={"carla": {"fixed_delta_seconds": delta,
        "replan_interval_s": 3., "planning": {"backend": backend,
            "lateral_control": lateral, "replan_interval_s": interval}}})


@pytest.mark.parametrize("interval", [.1, .2, .25, .3])
def test_native_configuration_replans_before_the_short_profile_expires(interval):
    loop = ClosedLoop(tracking_config(interval=interval), "unused")
    assert loop._pdm_interval == interval


def test_nonintegral_replan_period_refreshes_on_third_tick_before_native_expiry():
    loop = ClosedLoop(tracking_config(interval=.25), "unused")
    loop.model_inference = False
    loop.tracker = Mock()
    loop._audit_pdm_plan = Mock()
    loop._validate_decision = Mock(return_value={"accepted": True})
    times = []
    def plan(semantic):
        times.append(loop._elapsed)
        return {"trajectory": [], "target_speed": 0., "selected_longitudinal": "STOP",
                "planner_diagnostics": {"selected_id": "center:STOP"}}
    loop._pdm_plan = plan
    for timestamp in (0., .1, .2, .3):
        loop._elapsed = timestamp
        loop._refresh_pdm_control(carla.Transform(), 0., 0.)
    # No exact-multiple constraint is necessary: actual scheduler replans at
    # 0.3 before any control for the next tick can see an expired native hint.
    assert times == [0., .3]
    assert loop.tracker.set_trajectory.call_count == 2


@pytest.mark.parametrize("interval", [.30001, .4, .5, 1.])
def test_native_configuration_rejects_periods_that_would_silently_use_pp(interval):
    with pytest.raises(ValueError, match="PDM replan_interval_s <= 0.3"):
        ClosedLoop(tracking_config(interval=interval), "unused")


@pytest.mark.parametrize("delta,interval", [(.05, .2), (.2, .2), (.10001, .2)])
def test_native_configuration_requires_matching_physics_and_native_sample_ticks(delta, interval):
    with pytest.raises(ValueError, match="fixed_delta_seconds=0.1"):
        ClosedLoop(tracking_config(delta=delta, interval=interval), "unused")


def test_historical_pp_configuration_retains_supported_tick_and_replan_choices():
    loop = ClosedLoop(tracking_config(interval=.5, delta=.05, lateral="pure_pursuit"), "unused")
    assert loop.fixed_delta == .05 and loop._pdm_interval == .5


def test_vlm_trajectory_does_not_acquire_native_controller_timing_requirements():
    loop = ClosedLoop(tracking_config(interval=.5, delta=.05, backend="vlm_trajectory"), "unused")
    assert loop.fixed_delta == .05 and loop._pdm_interval == .5
