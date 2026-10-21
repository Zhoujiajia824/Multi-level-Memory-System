"""The loop accepts checked short braking, never ordinary short/stale plans."""
from copy import deepcopy
from unittest.mock import Mock
import pytest

from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.control.trajectory_validation import TrajectoryValidator
from carla_bridge.tests.test_mpc_braking_recovery import crossing_problem


@pytest.fixture(scope="module")
def braking_plan():
    return crossing_problem().plan(None, False, sim_time_s=2.)


def loop():
    obj = ClosedLoop.__new__(ClosedLoop)
    obj.planner_backend = "mpc"
    obj._elapsed, obj._execution_horizon, obj._pdm_interval, obj.fixed_delta = 2., 3., .2, .1
    obj.trajectory_validator = TrajectoryValidator()
    return obj


def test_short_recovery_validates_its_actual_horizon_and_is_censored_from_experience(braking_plan):
    obj = loop()
    result = obj._validate_decision(braking_plan, {"ego_state": {"speed": 0.}})
    assert result["accepted"], result
    assert result["details"]["execution_mode"] == "verified_short_horizon_braking"
    assert result["details"]["nominal_horizon_validated"] is False
    obj.experience_memory = Mock()
    obj._experience_signature = "old_execution"
    obj._begin_execution_experience(braking_plan)
    obj.experience_memory.finish.assert_called_once_with("planner_degraded_braking")
    obj.experience_memory.begin.assert_not_called()
    assert obj._experience_signature is None


@pytest.mark.parametrize("field,value", [
    ("issued_at_s", 1.8), ("execution_expires_at_s", 2.),
    ("constraints_relaxed", True), ("fresh_snapshot", False),
    ("optimization_feasible", False), ("verified_horizon_s", .2),
    ("acceleration_policy_passed", False), ("acceleration_policy_version", "old_policy"),
    ("episode_deadline_s", 2.), ("episode_deadline_s", 100.),
    ("verified_constraint_max_violation", 1.), ("verified_constraint_tolerance", float("nan")),
])
def test_invalid_or_expired_recovery_cannot_shorten_validation(braking_plan, field, value):
    plan = deepcopy(braking_plan)
    plan["planner_diagnostics"]["degraded_braking"][field] = value
    result = loop()._validate_decision(plan, {"ego_state": {"speed": 0.}})
    assert not result["accepted"]
    assert result["reasons"] == ["invalid_mpc_braking_recovery_contract"]


def test_ordinary_short_trajectory_still_fails(braking_plan):
    plan = deepcopy(braking_plan)
    plan["planner_diagnostics"].pop("degraded_braking")
    result = loop()._validate_decision(plan, {"ego_state": {"speed": 0.}})
    assert not result["accepted"]


@pytest.mark.parametrize("tamper", ["positive_demand", "nonfinite_bound", "wrong_bound_count", "solver_bound_mismatch"])
def test_recovery_execution_must_obey_the_optimized_braking_policy(braking_plan, tamper):
    plan = deepcopy(braking_plan)
    diagnostics = plan["planner_diagnostics"]
    recovery = diagnostics["degraded_braking"]
    if tamper == "positive_demand":
        diagnostics["mpc_tracking"]["accelerations_mps2"][0] = 1.
    elif tamper == "nonfinite_bound":
        recovery["acceleration_upper_bounds_mps2"][0] = float("nan")
    elif tamper == "wrong_bound_count":
        recovery["acceleration_upper_bounds_mps2"].pop()
    else:
        chosen = next(c for c in diagnostics["candidates"] if c["id"] == diagnostics["selected_id"])
        chosen["solver"]["acceleration_upper_bounds_mps2"][0] += .1
    result = loop()._validate_decision(plan, {"ego_state": {"speed": 0.}})
    assert not result["accepted"]
    assert result["reasons"] == ["invalid_mpc_braking_recovery_contract"]


def test_finite_infeasible_mpc_cannot_win_even_with_semantic_bonus():
    from carla_bridge.planning.selection import choose_candidate
    values = [{"id": "bad", "base_score": 1., "optimization_feasible": False,
               "safety": {"finite": True}},
              {"id": "good", "base_score": .1, "optimization_feasible": True,
               "safety": {"finite": True}}]
    chosen, audit = choose_candidate(values, {}, True, semantic_bonus_by_id={"bad": 4.})
    assert chosen["id"] == "good"
    assert audit["candidates"][0]["numerically_valid"]
    assert audit["candidates"][0]["ineligible_reason"] == "optimization_infeasible"


@pytest.mark.parametrize("reason,expected", [("planner_recovery_timeout", "planner_recovery_timeout"),
                                           ("no_feasible_mpc_solution", "invalid_trajectory")])
def test_recovery_timeout_cannot_be_overwritten_by_third_rejection(reason, expected):
    import carla
    from carla_bridge.tests.test_route_acceptance import fake_loop
    obj = fake_loop()
    obj.planner_backend, obj.model_inference = "mpc", False
    obj._consecutive_trajectory_rejections = 2
    result = {"trajectory": [], "target_speed": 0., "planner_diagnostics": {
        "emergency_brake": True, "failure_reason": reason}}
    obj._pdm_plan = Mock(return_value=result)
    obj._merge_plan = Mock(return_value=result)
    obj._audit_pdm_plan, obj._audit_pdm_error, obj._apply_brake = Mock(), Mock(), Mock()
    obj._refresh_pdm_control(carla.Transform(), 0., 0.)
    assert obj._early_terminated
    assert obj._termination_reason == expected
    assert obj._consecutive_trajectory_rejections == 3
    obj._apply_brake.assert_called_once()


def test_numerical_recovery_is_not_counted_as_a_semantic_change(tmp_path, braking_plan):
    from collections import defaultdict
    from carla_bridge.tests.test_route_acceptance import fake_loop
    obj = fake_loop()
    obj.config = {"output_dir": str(tmp_path)}
    obj.experience_memory = None
    obj.semantic_influence = True
    obj.advisory_stats = defaultdict(int)
    obj._audit_pdm_plan(braking_plan, {"accepted": True}, "control_update")
    assert obj.advisory_stats["degraded_braking_plans"] == 1
    assert obj.advisory_stats["semantic_changed_plans"] == 0
