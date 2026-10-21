"""Finite infeasibility is an explicit, unverified actuator hold, not a path."""
import numpy as np
import pytest

from carla_bridge.planning.mpc_planner import (
    MPC_MINIMUM_RISK_HOLD_POLICY_VERSION, MPC_MINIMUM_RISK_HOLD_LIFETIME_S)
from carla_bridge.planning.mpc_solver import MPCProblem, MPCSolver
from carla_bridge.tests.test_mpc_braking_recovery import crossing_problem
from carla_bridge.tests.test_mpc_planner import fixture


def test_finite_near_crossing_emits_only_bounded_unverified_full_brake_contract():
    planner = crossing_problem(crossing_time_s=.4)
    plan = planner.plan(None, False, sim_time_s=10.)
    d = plan["planner_diagnostics"]
    hold = d["minimum_risk_hold"]
    assert hold["active"] and hold["contract_version"] == MPC_MINIMUM_RISK_HOLD_POLICY_VERSION
    assert hold["actuation"] == "full_brake_reobserve"
    assert not hold["optimization_feasible"] and not hold["trajectory_accepted"]
    assert hold["verified_horizon_s"] == 0. and not plan["trajectory"]
    assert hold["episode_started_s"] == 10. and hold["episode_deadline_s"] == 18.
    assert hold["execution_expires_at_s"] == pytest.approx(10.+MPC_MINIMUM_RISK_HOLD_LIFETIME_S)
    assert hold["fresh_snapshot"] and not hold["constraints_relaxed"]
    assert hold["attempted_horizon_levels_s"] == pytest.approx([1.2, .8, .6])
    assert hold["short_horizon_attempt_count"] == len(d["degraded_braking"]["attempts"])
    assert hold["full_horizon_candidate_count"] == 3
    assert d["emergency_brake"] and "mpc_tracking" not in d
    for attempt in d["degraded_braking"]["attempts"]:
        assert attempt["finite_result"] and not attempt["optimization_feasible"]
        solver = attempt["solver"]
        assert solver["max_constraint_violation"] > solver["constraint_tolerance"]


def test_hold_and_verified_short_braking_share_one_deadline_and_full_plan_resets_it():
    planner = crossing_problem(crossing_time_s=.4)
    near = planner._obstacles
    first = planner.plan(None, False, sim_time_s=10.)
    assert first["planner_diagnostics"]["minimum_risk_hold"]["episode_deadline_s"] == 18.
    planner._obstacles = crossing_problem()._obstacles
    short = planner.plan(None, False, sim_time_s=12.)
    assert short["planner_diagnostics"]["degraded_braking"]["episode_started_s"] == 10.
    planner._obstacles = near
    last = planner.plan(None, False, sim_time_s=17.8)
    hold = last["planner_diagnostics"]["minimum_risk_hold"]
    assert hold["episode_started_s"] == 10. and hold["episode_deadline_s"] == 18.
    assert hold["execution_expires_at_s"] == 18.
    timeout = planner.plan(None, False, sim_time_s=18.)
    assert timeout["planner_diagnostics"]["failure_reason"] == "planner_recovery_timeout"
    assert "minimum_risk_hold" not in timeout["planner_diagnostics"]
    planner._obstacles = lambda *_: ([], [])
    normal = planner.plan(None, False, sim_time_s=19.)
    assert normal["trajectory"] and planner._braking_recovery_episode is None
    planner._obstacles = near
    again = planner.plan(None, False, sim_time_s=20.)
    assert again["planner_diagnostics"]["minimum_risk_hold"]["episode_started_s"] == 20.


@pytest.mark.parametrize("fault", ["states", "controls", "cost", "violation", "tolerance", "minima"])
def test_nonfinite_full_solver_results_cannot_be_reclassified_as_hold(monkeypatch, fault):
    planner = crossing_problem(crossing_time_s=.4)
    real_solve = planner.solver.solve

    def corrupt(problem, **kwargs):
        result = real_solve(problem, **kwargs)
        if fault == "states": result.states[0, 0] = np.nan
        elif fault == "controls": result.controls[0, 0] = np.nan
        elif fault == "cost": result.cost = np.nan
        elif fault == "violation": result.diagnostics["max_constraint_violation"] = np.nan
        elif fault == "tolerance": result.diagnostics["constraint_tolerance"] = np.nan
        else: result.diagnostics["constraint_minimums"]["speed_min"] = np.nan
        return result

    monkeypatch.setattr(planner.solver, "solve", corrupt)
    result = planner.plan(None, False, sim_time_s=0.)
    d = result["planner_diagnostics"]
    assert d["emergency_brake"] and "minimum_risk_hold" not in d
    assert d["failure_reason"] == "invalid_mpc_solver_result"
    assert not d["degraded_braking"]["attempts"]


def test_nonfinite_short_result_does_not_get_an_unverified_hold_contract(monkeypatch):
    planner = crossing_problem(crossing_time_s=.4)
    real_solve = MPCSolver.solve

    def corrupt(solver, problem, **kwargs):
        result = real_solve(solver, problem, **kwargs)
        if solver.config.horizon_steps < planner.samples:
            result.states[0, 0] = np.nan
        return result

    monkeypatch.setattr(MPCSolver, "solve", corrupt)
    result = planner.plan(None, False, sim_time_s=0.)
    d = result["planner_diagnostics"]
    assert d["emergency_brake"] and "minimum_risk_hold" not in d
    assert not any(attempt["finite_result"] for attempt in d["degraded_braking"]["attempts"])


def test_solver_exception_propagates_instead_of_becoming_an_infeasibility_hold(monkeypatch):
    planner = crossing_problem(crossing_time_s=.4)
    def error(*args, **kwargs):
        raise RuntimeError("solver process failure")
    monkeypatch.setattr(planner.solver, "solve", error)
    with pytest.raises(RuntimeError, match="solver process failure"):
        planner.plan(None, False, sim_time_s=0.)


def test_minimum_risk_brake_resets_only_next_tick_and_never_inherits_an_old_profile():
    planner = fixture()
    old = {"mpc_tracking": {"accelerations_mps2": [1.2]}}
    applied = {"applied": True, "profile_sample_index": 0}
    planner.record_control_execution(old, applied, now_s=1., source="tracker")
    assert planner._initial_state(1.1)[0].acceleration == pytest.approx(1.2)
    # A stale, previously applied optimization audit must not override the
    # explicit minimum-risk actuator source or its independent reset reason.
    planner.record_control_execution(old, applied, now_s=1.1, source="mpc_minimum_risk_hold",
                                     tracker_diagnostics={"reason": "stationary_hold"})
    assert planner._initial_state(1.1)[0].acceleration == pytest.approx(1.2)
    assert planner._pending_desired_acceleration["source"] == "minimum_risk_hold_reset_desired_input"
    assert planner._initial_state(1.2)[0].acceleration == 0.
    assert planner._executed_desired_acceleration["source"] == "minimum_risk_hold_reset_desired_input"
    assert planner._executed_desired_acceleration["profile_sample_index"] is None


def shadow_crossing(monkeypatch, crossing_time_s):
    planner = crossing_problem(crossing_time_s=crossing_time_s)
    planner.semantic_policy = "shadow"
    monkeypatch.setattr(planner.governor, "recovery_request", lambda *_: {
        "authorized": True, "reason": "test_authorized_generation"})
    return planner


def test_shadow_expansion_all_infeasible_keeps_a_valid_baseline_hold_contract(monkeypatch):
    from carla_bridge.tests.test_mpc_minimum_risk_loop import configured_loop
    planner = shadow_crossing(monkeypatch, .4)
    plan = planner.plan(None, True, sim_time_s=2.)
    d = plan["planner_diagnostics"]
    hold = d["minimum_risk_hold"]
    assert d["shadow_only"] and len(d["shadow_candidates"]) == 7
    assert len(d["candidates"]) == hold["full_horizon_candidate_count"] == 3
    assert all(not c["optimization_feasible"] for c in d["shadow_candidates"])
    assert all(a["reference_id"] == "center:STOP@verified_braking"
               for a in d["degraded_braking"]["attempts"])
    validation = configured_loop()._validate_decision(plan, {})
    assert validation["accepted"] is False
    assert validation["details"]["minimum_risk_contract_valid"] is True


def test_shadow_only_feasible_result_cannot_block_baseline_verified_stop_recovery(monkeypatch):
    from carla_bridge.tests.test_mpc_braking_recovery_loop import loop
    planner = shadow_crossing(monkeypatch, 3.)
    solve = planner.solver.solve

    def shadow_succeeds(problem):
        # Isolate the observation-only pool's feasibility from execution.
        # Baseline and short recovery keep the real crossing obstacle; only
        # the deliberately injected shadow result comes from an empty scene.
        if "@recovery" in problem.reference.name:
            return solve(MPCProblem(problem.initial, problem.reference))
        return solve(problem)

    monkeypatch.setattr(planner.solver, "solve", shadow_succeeds)
    plan = planner.plan(None, True, sim_time_s=2.)
    d = plan["planner_diagnostics"]
    assert d["shadow_only"] and any(c["optimization_feasible"] and c["recovery"]
                                     for c in d["shadow_candidates"])
    assert all(not c["optimization_feasible"] for c in d["original_baseline_candidates"])
    assert d["degraded_braking"]["active"] and not d["emergency_brake"]
    assert d["selected_id"] == "center:STOP@verified_braking"
    assert "minimum_risk_hold" not in d and len(d["candidates"]) == 4
    assert loop()._validate_decision(plan, {"ego_state": {"speed": 0.}})["accepted"]


@pytest.mark.parametrize("constant,value", [
    ("MPC_MINIMUM_RISK_HOLD_POLICY_VERSION", "another_contract"),
    ("MPC_MINIMUM_RISK_HOLD_LIFETIME_S", .2)])
def test_unverified_hold_policy_partitions_mpc_evidence_without_changing_pdm(monkeypatch, constant, value):
    from carla_bridge.planning import mpc_planner
    from carla_bridge.planning.experience_memory import execution_planner_version
    original = execution_planner_version({"backend": "mpc"})
    pdm = execution_planner_version({"backend": "pdm_closed"})
    monkeypatch.setattr(mpc_planner, constant, value)
    assert execution_planner_version({"backend": "mpc"}) != original
    assert execution_planner_version({"backend": "pdm_closed"}) == pdm
