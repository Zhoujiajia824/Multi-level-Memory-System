"""A stopped brake warm start must not hide a feasible departure mode."""
from dataclasses import replace

import numpy as np
import pytest

from carla_bridge.planning.mpc_solver import (
    MPCConfig, MPCInitialState, MPCProblem, MPCReference, MPCSolver, _Evaluation, rollout,
    MPC_SOLVE_POLICY_VERSION,
)


def departure_problem(config, acceleration=-.03359548291660212, target=8.):
    initial = MPCInitialState(0., 0., 0., 4e-10, acceleration=acceleration)
    times = np.arange(config.horizon_steps + 1) * config.dt
    speeds = np.minimum(target, initial.speed + 1.5 * times)
    positions = np.r_[0., np.cumsum((speeds[1:] + speeds[:-1]) * config.dt / 2)]
    return MPCProblem(initial, MPCReference("center:CRUISE", np.c_[
        positions, np.zeros_like(times), np.zeros_like(times), speeds], .65, target))


@pytest.mark.parametrize("multi_start", [False, True])
@pytest.mark.parametrize("acceleration", [-.03359548291660212, -2.7])
def test_negative_brake_warm_start_is_compared_with_verified_departure(multi_start, acceleration):
    config = MPCConfig(multi_start=multi_start)
    problem = departure_problem(config, acceleration)
    solver = MPCSolver(config)
    stationary = np.tile([acceleration, 0.], (config.horizon_steps, 1))
    solver._warm_starts[problem.reference.name] = stationary
    stationary_cost = _Evaluation(config, problem).objective(stationary.ravel())
    result = solver.solve(problem)
    assert result.feasible, result.diagnostics
    assert result.states[-1, 0] > 5., result.diagnostics
    assert result.states[-1, 3] > 3.
    assert result.cost < stationary_cost
    assert [attempt["seed"] for attempt in result.diagnostics["attempts"]][:2] == [
        "warm_start", "reference_tracking"]
    assert result.diagnostics["solve_policy_version"] == MPC_SOLVE_POLICY_VERSION
    assert result.diagnostics["stationary_branch_search"]["triggered_by"] == ["warm_start"]
    assert result.diagnostics["selected_seed"] == "reference_tracking"
    np.testing.assert_allclose(result.states, rollout(problem.initial, result.controls, config), atol=1e-12)
    previous = np.vstack([[acceleration, 0.], result.controls[:-1]])
    assert np.max(np.abs(result.controls[:, 0] - previous[:, 0])) <= config.jerk_max*config.dt + 1e-8
    evaluation = _Evaluation(config, problem)
    assert np.min(evaluation.inequalities(result.controls.ravel())) >= -config.constraint_tolerance


def test_departure_search_cannot_override_a_stop_line_or_its_original_cost():
    config = MPCConfig(multi_start=False)
    problem = departure_problem(config)
    problem = replace(problem, reference=replace(problem.reference, metadata={
        "stop_line": {"point": [0., 0.], "normal": [1., 0.]}}))
    solver = MPCSolver(config)
    solver._warm_starts[problem.reference.name] = np.tile(
        [problem.initial.acceleration, 0.], (config.horizon_steps, 1))
    result = solver.solve(problem)
    assert result.feasible, result.diagnostics
    assert np.max(result.states[:, 0]) <= config.constraint_tolerance
    assert result.diagnostics["constraint_minimums"]["stop_line"] >= -config.constraint_tolerance
    feasible_costs = [attempt["cost"] for attempt in result.diagnostics["attempts"] if attempt["feasible"]]
    assert result.cost == min(feasible_costs)


def test_actual_stop_request_retains_the_single_warm_start_fast_path():
    config = MPCConfig(multi_start=False)
    problem = departure_problem(config, target=0.)
    solver = MPCSolver(config)
    solver._warm_starts[problem.reference.name] = np.tile(
        [problem.initial.acceleration, 0.], (config.horizon_steps, 1))
    result = solver.solve(problem)
    assert result.feasible
    assert len(result.diagnostics["attempts"]) == 1
    np.testing.assert_allclose(result.states[:, 3], 0., atol=1e-9)


def test_recovery_acceleration_upper_bounds_cannot_be_overridden_by_departure_search():
    config = MPCConfig(multi_start=False)
    problem = departure_problem(config)
    solver = MPCSolver(config)
    solver._warm_starts[problem.reference.name] = np.tile(
        [problem.initial.acceleration, 0.], (config.horizon_steps, 1))
    result = solver.solve(problem, acceleration_upper_bounds=np.zeros(config.horizon_steps))
    assert result.feasible
    assert np.max(result.controls[:, 0]) <= config.constraint_tolerance
    np.testing.assert_allclose(result.states[:, 3], 0., atol=1e-9)
    assert len(result.diagnostics["attempts"]) == 1


def test_moving_converged_warm_start_still_needs_only_one_optimization():
    config = MPCConfig(multi_start=True)
    solver = MPCSolver(config)
    problem = departure_problem(config)
    first = solver.solve(problem)
    measured = MPCInitialState(*first.states[5], acceleration=first.controls[4, 0],
                               steering=first.controls[4, 1])
    # Reconstruct the same local forward request at its newly measured pose.
    following = departure_problem(config)
    shifted = following.reference.points.copy()
    shifted[:, 0] += measured.x
    shifted[:, 3] = np.maximum(shifted[:, 3], measured.speed)
    following = replace(following, initial=measured,
                        reference=replace(following.reference, points=shifted))
    result = solver.solve(following)
    assert result.feasible
    assert measured.speed > .5
    assert [attempt["seed"] for attempt in result.diagnostics["attempts"]] == ["warm_start"]
    assert result.diagnostics["stationary_branch_search"]["triggered_by"] == []


def test_a_second_stationary_local_minimum_explores_delayed_departure_even_without_optional_multi_start(monkeypatch):
    import carla_bridge.planning.mpc_solver as module
    from scipy.optimize import OptimizeResult

    config = MPCConfig(multi_start=False)
    problem = departure_problem(config)
    solver = MPCSolver(config)
    stationary = np.tile([problem.initial.acceleration, 0.], (config.horizon_steps, 1))
    solver._warm_starts[problem.reference.name] = stationary
    real_minimize = module.minimize
    calls = []

    def local_branch_optimizer(*args, **kwargs):
        calls.append(np.array(args[1], copy=True))
        if len(calls) <= 2:
            # A second initialization may also converge inside the exact stop
            # branch. Its genuine rollout is feasible but cannot establish that
            # the opposite departure mode has a worse objective.
            return OptimizeResult(x=stationary.ravel(), success=True, status=0,
                                  message="Stationary local branch", nit=1)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(module, "minimize", local_branch_optimizer)
    result = solver.solve(problem)
    assert result.feasible and result.states[-1, 0] > 5., result.diagnostics
    assert result.diagnostics["selected_seed"] == "delayed_departure"
    assert [attempt["seed"] for attempt in result.diagnostics["attempts"]] == [
        "warm_start", "reference_tracking", "delayed_departure"]
    assert result.cost == min(attempt["cost"] for attempt in result.diagnostics["attempts"] if attempt["feasible"])
    assert np.min(_Evaluation(config, problem).inequalities(result.controls.ravel())) >= -config.constraint_tolerance
