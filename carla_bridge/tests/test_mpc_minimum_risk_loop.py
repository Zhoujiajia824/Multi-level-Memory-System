"""No-solution braking remains observable without becoming a valid trajectory."""
from copy import deepcopy
from unittest.mock import Mock

import carla
import pytest

from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.tests.test_mpc_braking_recovery import crossing_problem
from carla_bridge.tests.test_route_acceptance import fake_loop


@pytest.fixture(scope="module")
def hold_plan():
    return crossing_problem(crossing_time_s=.4).plan(None, False, sim_time_s=2.)


def configured_loop():
    obj = fake_loop(duration=30)
    obj.planner_backend, obj.model_inference = "mpc", False
    obj._elapsed, obj._pdm_interval = 2., .2
    obj.experience_memory = Mock()
    obj.pdm_planner = Mock()
    return obj


def test_no_solution_hold_is_rejected_as_trajectory_but_can_dispatch_full_brake(hold_plan):
    obj = configured_loop()
    validation = obj._validate_decision(hold_plan, {})
    assert validation["accepted"] is False
    assert validation["reasons"] == ["mpc_minimum_risk_hold"]
    assert validation["details"]["trajectory_validated"] is False
    assert obj._execute_mpc_minimum_risk_hold(hold_plan, validation, carla.Transform(), 0.)
    obj.tracker.set_trajectory.assert_called_with([], 0., 0., 0., 0.)
    obj.experience_memory.finish.assert_called_once_with("planner_minimum_risk_hold")
    # The inter-replan tick cannot execute a previously cached control.
    obj._elapsed = 2.1
    obj._apply_control(obj.scenario.ego_vehicle, carla.VehicleControl(throttle=.9, steer=.8))
    command = obj.scenario.ego_vehicle.apply_control.call_args.args[0]
    assert (command.throttle, command.brake, command.steer) == (0., 1., 0.)
    assert obj._last_control_command["source"] == "mpc_minimum_risk_hold"
    assert obj._last_control_command["minimum_risk_hold"]["observation_lease_fresh"]


def test_repeated_finite_infeasibility_does_not_consume_or_clear_illegal_trajectory_count():
    planner, obj = crossing_problem(crossing_time_s=.4), configured_loop()
    obj._consecutive_trajectory_rejections = 2
    obj._pdm_plan = Mock(side_effect=lambda _: planner.plan(None, False, sim_time_s=obj._elapsed))
    obj._audit_pdm_plan, obj._audit_pdm_error = Mock(), Mock()
    for now in (2., 2.2, 2.4, 2.6):
        obj._elapsed = now
        obj._refresh_pdm_control(carla.Transform(), 0., 0.)
        assert not obj._early_terminated
        assert obj._consecutive_trajectory_rejections == 2
        assert obj._trajectory_rejections == 0
        assert obj._mpc_minimum_risk_hold["episode_deadline_s"] == 10.
    obj._audit_pdm_error.assert_not_called()
    obj._elapsed = 10.
    obj._refresh_pdm_control(carla.Transform(), 0., 0.)
    assert obj._early_terminated and obj._termination_reason == "planner_recovery_timeout"


@pytest.mark.parametrize("field,value", [
    ("issued_at_s", 1.9), ("execution_expires_at_s", 20.),
    ("episode_deadline_s", 20.), ("optimization_feasible", True),
    ("trajectory_accepted", True), ("fresh_snapshot", False),
    ("contract_version", "unrecognized"), ("verified_horizon_s", .6),
    ("constraints_relaxed", True), ("full_horizon_candidate_count", 100),
    ("short_horizon_attempt_count", 100), ("actuation", "accelerate"),
])
def test_stale_or_malformed_hold_cannot_bypass_illegal_trajectory_handling(hold_plan, field, value):
    plan = deepcopy(hold_plan)
    plan["planner_diagnostics"]["minimum_risk_hold"][field] = value
    obj = configured_loop()
    validation = obj._validate_decision(plan, {})
    assert not validation["accepted"]
    assert validation["reasons"] == ["invalid_mpc_minimum_risk_hold_contract"]
    assert not obj._execute_mpc_minimum_risk_hold(plan, validation, carla.Transform(), 0.)
    obj.scenario.ego_vehicle.apply_control.assert_not_called()


def test_missing_stop_attempt_cannot_claim_complete_infeasibility(hold_plan):
    plan = deepcopy(hold_plan)
    d = plan["planner_diagnostics"]
    d["candidates"].append(dict(deepcopy(d["candidates"][-1]), id="second:STOP", longitudinal="STOP"))
    d["minimum_risk_hold"]["full_horizon_candidate_count"] += 1
    assert not configured_loop()._validate_decision(plan, {})["details"]["minimum_risk_contract_valid"]


def test_expired_observation_keeps_braking_but_forces_fresh_replanning(hold_plan):
    obj = configured_loop()
    obj._execute_mpc_minimum_risk_hold(hold_plan, obj._validate_decision(hold_plan, {}), carla.Transform(), 0.)
    obj._elapsed, obj._next_pdm_replan_at = 2.4, 100.
    obj._apply_control(obj.scenario.ego_vehicle, carla.VehicleControl(throttle=.9))
    assert obj._last_control_command["brake"] == 1.
    assert not obj._last_control_command["minimum_risk_hold"]["observation_lease_fresh"]
    assert obj._next_pdm_replan_at == 0.
    obj._elapsed = 10.
    obj._apply_control(obj.scenario.ego_vehicle, carla.VehicleControl(throttle=.9))
    assert obj._early_terminated and obj._termination_reason == "planner_recovery_timeout"


def test_physical_hold_ticks_count_once_despite_multiple_dispatches_and_observers(hold_plan):
    obj = configured_loop()
    obj.experience_memory = None
    obj.metrics = Mock()
    obj._execute_mpc_minimum_risk_hold(hold_plan, obj._validate_decision(hold_plan, {}), carla.Transform(), 0.)
    for now in (2.1, 2.1, 2.2):
        obj._elapsed = now
        obj._record_tick(obj.scenario.ego_vehicle)
    assert obj.advisory_stats["minimum_risk_hold_ticks"] == 2
    assert obj.advisory_stats["minimum_risk_hold_s"] == pytest.approx(.2)


def test_report_exposes_minimum_risk_without_claiming_safety():
    from carla_bridge.metrics.run_reporter import _render_md
    result = _render_md({"planner_backend": "mpc", "advisory": {
        "minimum_risk_hold_plans": 4, "minimum_risk_hold_ticks": 7, "minimum_risk_hold_s": .7}}, {})
    assert "4 次规划 / 7 个物理帧 / 0.70 s" in result
    assert "无可行轨迹，非安全性通过" in result


def test_normal_validated_plan_clears_hold_before_tracking(hold_plan):
    obj = configured_loop()
    obj._execute_mpc_minimum_risk_hold(hold_plan, obj._validate_decision(hold_plan, {}), carla.Transform(), 0.)
    result = {"trajectory": [{"x": 1., "y": 0., "t": .2, "optional_v": 2.}], "target_speed": 2.}
    obj._pdm_plan = Mock(return_value=result)
    obj._merge_plan = Mock(return_value=result)
    obj._validate_decision = Mock(return_value={"accepted": True})
    obj._audit_pdm_plan = Mock()
    obj._refresh_pdm_control(carla.Transform(), 0., 0.)
    assert obj._mpc_minimum_risk_hold is None
    obj.tracker.set_trajectory.assert_called_with(result["trajectory"], 0., 0., 0., 2., follow_speed_profile=True)


def test_hold_replans_between_periodic_model_calls_even_without_active_semantics(hold_plan):
    obj = configured_loop()
    obj.model_inference, obj._selective_execution, obj._active_semantics = True, False, None
    obj._execute_mpc_minimum_risk_hold(hold_plan, obj._validate_decision(hold_plan, {}), carla.Transform(), 0.)
    obj._pdm_plan = Mock(return_value=hold_plan)
    obj._audit_pdm_plan = Mock()
    obj._refresh_pdm_control(carla.Transform(), 0., 0.)
    obj._pdm_plan.assert_called_once_with(None)


def test_periodic_model_path_preserves_recovery_timeout_reason():
    planner, obj = crossing_problem(crossing_time_s=.4), configured_loop()
    planner.plan(None, False, sim_time_s=2.)
    result = planner.plan(None, False, sim_time_s=10.)
    assert result["planner_diagnostics"]["failure_reason"] == "planner_recovery_timeout"
    obj._elapsed, obj._consecutive_trajectory_rejections = 9.9, 2
    obj.scenario.route._route_waypoints = []
    obj._run_cognition = Mock(return_value={"decision_output": result,
        "decision_validation": {"accepted": False, "reasons": ["mpc_emergency_brake"]}})
    obj._replan_cycle()
    assert obj._early_terminated and obj._termination_reason == "planner_recovery_timeout"
    assert obj._consecutive_trajectory_rejections == 3


@pytest.mark.parametrize("field,value", [("verified_horizon_s", []), ("reference_id", {})])
def test_unhashable_attempt_fields_fail_contract_without_exception(hold_plan, field, value):
    result = deepcopy(hold_plan)
    result["planner_diagnostics"]["degraded_braking"]["attempts"][0][field] = value
    assert not configured_loop()._validate_decision(result, {})["details"]["minimum_risk_contract_valid"]


def test_consistently_deleted_horizon_levels_still_cannot_claim_exhaustive_attempts(hold_plan):
    result = deepcopy(hold_plan)
    d = result["planner_diagnostics"]
    recovery, hold = d["degraded_braking"], d["minimum_risk_hold"]
    levels = [recovery["available_horizon_levels_s"][-1]]
    recovery["attempts"] = [a for a in recovery["attempts"] if a["verified_horizon_s"] in levels]
    recovery["available_horizon_levels_s"] = recovery["attempted_horizon_levels_s"] = levels
    hold["attempted_horizon_levels_s"] = levels
    hold["short_horizon_attempt_count"] = len(recovery["attempts"])
    assert not configured_loop()._validate_decision(result, {})["details"]["minimum_risk_contract_valid"]
