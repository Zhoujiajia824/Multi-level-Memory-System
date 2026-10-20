"""Independent MPC equations, constraints and blocked-lane manoeuvre evidence."""
from dataclasses import replace
import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from carla_bridge.planning.mpc_solver import (
    MPCConfig, MPCInitialState, MPCReference, MPCObstacle, MPCProblem, MPCSolver,
    _Evaluation, bicycle_step, rollout,
)


def straight(cfg, *, initial=None, target=7., width=.7, offset=0., name="center"):
    initial = initial or MPCInitialState(0., 0., 0., 0.)
    times = np.arange(cfg.horizon_steps + 1) * cfg.dt
    speeds = np.minimum(target, initial.speed + 1.5 * times)
    positions = initial.x + np.r_[0., np.cumsum((speeds[1:] + speeds[:-1]) * cfg.dt / 2)]
    points = np.c_[positions, np.full_like(positions, offset), np.zeros_like(positions), speeds]
    return MPCProblem(initial, MPCReference(name, points, width, target))


def test_midpoint_bicycle_step_analytic_derivatives():
    state, control = np.array([2., -1., .4, 5.]), np.array([.7, -.16])
    actual, a, b = bicycle_step(state, control, .2, 2.8)
    expected = approx_derivative(lambda x: bicycle_step(x, control, .2, 2.8)[0], state)
    np.testing.assert_allclose(a, expected, rtol=2e-6, atol=1e-7)
    expected = approx_derivative(lambda x: bicycle_step(state, x, .2, 2.8)[0], control)
    np.testing.assert_allclose(b, expected, rtol=2e-6, atol=1e-7)
    assert actual[3] == pytest.approx(5.14)


def test_all_objective_and_constraint_derivatives_including_rotated_ellipse():
    cfg = MPCConfig(horizon_steps=6)
    problem = straight(cfg, initial=MPCInitialState(0, 0, .03, 2., .02, .3))
    points = problem.reference.points.copy()
    points[:, 1] = np.linspace(0, .3, 7)
    metadata = {"stop_line": {"point": [14., .2], "normal": [1., .1]}}
    obstacle = MPCObstacle("rotated", np.tile([11., 2.], (7, 1)),
                           np.tile([2., .9], (7, 1)), np.linspace(.1, .4, 7))
    circle = MPCObstacle("circle", np.tile([9., 1.7], (7, 1)), .35)
    problem = replace(problem, reference=replace(problem.reference, points=points, metadata=metadata),
                      obstacles=(obstacle, circle))
    evaluation = _Evaluation(cfg, problem)
    controls = np.c_[np.linspace(.4, .7, 6), np.linspace(.02, .07, 6)].ravel()
    np.testing.assert_allclose(evaluation.objective_jac(controls),
        approx_derivative(lambda x: np.array([evaluation.objective(x)]), controls).ravel(),
        rtol=5e-5, atol=2e-6)
    np.testing.assert_allclose(evaluation.inequalities_jac(controls),
        approx_derivative(evaluation.inequalities, controls), rtol=5e-5, atol=2e-6)


def test_open_lane_accelerates_with_real_optimized_controls():
    cfg = MPCConfig()
    problem = straight(cfg)
    result = MPCSolver(cfg).solve(problem)
    assert result.feasible and result.converged, result.diagnostics
    assert result.states[-1, 0] > 9.
    assert result.states[-1, 3] > 5.
    np.testing.assert_allclose(result.states[:, 1:3], 0., atol=1e-4)
    np.testing.assert_allclose(result.states, rollout(problem.initial, result.controls, cfg), atol=1e-12)
    assert result.diagnostics["analytic_objective_gradient"]
    assert result.diagnostics["wall_time_s"] > 0


def test_control_and_motion_limits_are_hard_constraints():
    cfg = MPCConfig(jerk_max=3., steer_rate_max=.3)
    initial = MPCInitialState(0, 0, 0, 2., .02, .1)
    problem = straight(cfg, initial=initial, target=8., width=4.)
    points = problem.reference.points.copy()
    points[:, 1] = np.linspace(0, 3., len(points))
    points[:, 2] = .15
    problem = replace(problem, reference=replace(problem.reference, points=points))
    result = MPCSolver(cfg).solve(problem)
    assert result.feasible, result.diagnostics
    delta = np.diff(np.vstack([[initial.acceleration, initial.steering], result.controls]), axis=0)
    assert np.max(np.abs(delta[:, 0])) <= cfg.jerk_max * cfg.dt + .003
    assert np.max(np.abs(delta[:, 1])) <= cfg.steer_rate_max * cfg.dt + .003
    assert np.min(result.states[:, 3]) >= -.003
    assert np.max(result.controls[:, 0]) <= cfg.accel_max + .003
    lateral_accel = result.states[1:, 3] ** 2 * np.tan(result.controls[:, 1]) / cfg.wheelbase
    assert np.max(np.abs(lateral_accel)) <= cfg.lateral_accel_max + .003


def test_stop_line_is_respected_even_when_reference_demands_progress():
    cfg = MPCConfig()
    problem = straight(cfg, initial=MPCInitialState(0, 0, 0, 3.))
    problem = replace(problem, reference=replace(problem.reference, metadata={
        "stop_line": {"point": [6., 0.], "normal": [1., 0.]}}))
    result = MPCSolver(cfg).solve(problem)
    assert result.feasible, result.diagnostics
    assert np.max(result.states[:, 0]) <= 6.003
    assert np.max(result.states[1:, 0] + result.states[1:, 3] ** 2 / (2 * abs(cfg.accel_min))) <= 6.003


def test_stationary_reference_remains_stopped_without_optimizer_fake_motion():
    cfg = MPCConfig()
    problem = straight(cfg, target=0.)
    result = MPCSolver(cfg).solve(problem)
    assert result.feasible and result.converged
    np.testing.assert_allclose(result.states, 0., atol=1e-10)
    np.testing.assert_allclose(result.controls, 0., atol=1e-10)


def test_stationary_hold_then_release_uses_neutral_dispatched_input():
    cfg = MPCConfig()
    solver = MPCSolver(cfg)
    held = solver.solve(straight(cfg, target=0.))
    assert held.feasible and held.converged
    # A stationary-hold controller dispatches zero desired acceleration. Its
    # contact/brake acceleration telemetry is a separate physical measurement.
    initial = MPCInitialState(*held.states[0], acceleration=0.)
    released = solver.solve(straight(cfg, initial=initial, target=7.))
    assert released.feasible and released.converged, released.diagnostics
    assert released.controls[0, 0] > .5
    assert released.states[-1, 0] > 9.
    assert released.states[-1, 3] > 5.


def test_measured_brake_impulse_is_not_a_valid_previous_desired_input():
    cfg = MPCConfig(max_iterations=12, multi_start=False)
    # The server observed a -12.78m/s² impulse near standstill. Clipping that
    # telemetry to -4 does not make it the previous desired control: jerk would
    # require first a <= -2.4, incompatible with nonnegative forward speed.
    incorrect = MPCInitialState(0., 0., 0., .001, acceleration=-4.)
    result = MPCSolver(cfg).solve(straight(cfg, initial=incorrect))
    assert not result.feasible and not result.converged
    assert result.status == "infeasible"
    assert result.diagnostics["max_constraint_violation"] > cfg.constraint_tolerance


def slow_walker_problem(cfg, *, bypass):
    problem = straight(cfg, target=6., width=1.8 if bypass else .6,
                       name="left_bypass" if bypass else "center")
    points = problem.reference.points.copy()
    if bypass:
        # A legal adjacent-lane reference from the same rear-axle origin, not
        # a universal following-speed cap applied to every maneuver.
        station = points[:, 0]
        u = np.clip(station / 9., 0., 1.)
        points[:, 1] = 3.5 * (3 * u ** 2 - 2 * u ** 3)
        points[:, 2] = np.arctan(3.5 / 9. * 6 * u * (1 - u))
    times = np.arange(cfg.horizon_steps + 1) * cfg.dt
    pedestrian = MPCObstacle("walker", np.c_[7. + .6 * times, np.zeros_like(times)], .35)
    return replace(problem, reference=replace(problem.reference, points=points), obstacles=(pedestrian,))


def test_legal_neighbor_corridor_enables_overtake_of_slow_same_lane_pedestrian():
    cfg = MPCConfig(max_iterations=80)
    center = MPCSolver(cfg).solve(slow_walker_problem(cfg, bypass=False))
    bypass_problem = slow_walker_problem(cfg, bypass=True)
    bypass = MPCSolver(cfg).solve(bypass_problem)
    assert center.feasible, center.diagnostics
    assert bypass.feasible, bypass.diagnostics
    assert bypass.states[-1, 1] > 2.8
    assert bypass.states[-1, 0] > center.states[-1, 0] + 1.5
    assert bypass.states[-1, 3] > center.states[-1, 3] + 1.
    assert bypass.diagnostics["constraint_minimums"]["obstacle:walker"] >= -.003


def test_pedestrian_three_metres_ahead_of_actor_center_does_not_add_false_bonnet_clearance():
    cfg = MPCConfig(max_iterations=90)
    problem = slow_walker_problem(cfg, bypass=True)
    walker = problem.obstacles[0]
    positions = walker.centers.copy()
    # Rear axle is 1.4m behind the actor centre. Pedestrian is 3m ahead
    # of that centre, just beyond the physical 2.4m front half-length.
    positions[:, 0] -= 7. - 4.4
    result = MPCSolver(cfg).solve(replace(problem, obstacles=(replace(walker, centers=positions),)))
    assert result.feasible, result.diagnostics
    assert result.states[-1, 0] > 6.
    assert result.states[-1, 1] > 2.8
    assert result.diagnostics["collision_sample_dt_s"] == cfg.dt / 2


@pytest.mark.parametrize("radius", [8., 12., 20.])
def test_real_bicycle_rollout_follows_curved_lane_center(radius):
    cfg = MPCConfig()
    speed = 4.
    times = np.arange(cfg.horizon_steps + 1) * cfg.dt
    angles = speed * times / radius
    points = np.c_[radius * np.sin(angles), radius * (1 - np.cos(angles)),
                   angles, np.full_like(angles, speed)]
    problem = MPCProblem(MPCInitialState(0, 0, 0, speed, np.arctan(cfg.wheelbase / radius)),
                         MPCReference("arc", points, .6, 8.))
    result = MPCSolver(cfg).solve(problem)
    assert result.feasible and result.converged, result.diagnostics
    assert np.max(np.linalg.norm(result.states[:, :2] - points[:, :2], axis=1)) < .03


def test_corridor_constraint_includes_vehicle_front_corner_not_only_rear_axle():
    cfg = MPCConfig(horizon_steps=4)
    problem = straight(cfg, initial=MPCInitialState(0, 0, 0, 4.), width=.7)
    evaluation = _Evaluation(cfg, problem)
    controls = np.tile([0., .3], (4, 1)).ravel()
    evaluation.evaluate(controls)
    assert np.max(np.abs(evaluation.states[:, 1])) < .7
    minimums = {}
    for name, value in zip(evaluation.constraint_names, evaluation.constraints):
        minimums[name] = min(value, minimums.get(name, float("inf")))
    assert min(minimums["corridor_left"], minimums["corridor_right"]) < -.5


def test_fast_crossing_pedestrian_cannot_tunnel_between_prediction_endpoints():
    cfg = MPCConfig(horizon_steps=2, dt=.8, steer_max=.05, accel_max=.1, max_iterations=12)
    problem = straight(cfg, target=0.)
    crossing = MPCObstacle("crossing", np.array([[2., -4.], [2., 4.], [2., 8.]]), .35)
    result = MPCSolver(cfg).solve(replace(problem, obstacles=(crossing,)))
    assert not result.feasible
    assert result.diagnostics["constraint_minimums"]["obstacle:crossing"] < -.5


def test_no_feasible_escape_is_an_explicit_failure_not_a_fallback_success():
    cfg = MPCConfig(horizon_steps=8, max_iterations=15)
    problem = straight(cfg, initial=MPCInitialState(0, 0, 0, 8.), width=.1, target=8.)
    obstacle = MPCObstacle("wall", np.tile([5., 0.], (9, 1)), 2.)
    result = MPCSolver(cfg).solve(replace(problem, obstacles=(obstacle,)))
    assert not result.feasible
    assert not result.converged
    assert result.status == "infeasible"
    assert result.diagnostics["max_constraint_violation"] > cfg.constraint_tolerance
    assert not MPCSolver(cfg)._warm_starts


def test_receding_horizon_warm_start_uses_measured_state_and_is_not_stale_trajectory():
    cfg = MPCConfig(horizon_steps=12)
    solver = MPCSolver(cfg)
    first = solver.solve(straight(cfg))
    next_state = first.states[1]
    measured = MPCInitialState(next_state[0] + .05, .02, next_state[2], next_state[3],
                               first.controls[0, 1], first.controls[0, 0])
    second = solver.solve(straight(cfg, initial=measured))
    assert first.feasible and second.feasible
    assert second.diagnostics["attempts"][0]["seed"] == "warm_start"
    np.testing.assert_allclose(second.states[0], [measured.x, measured.y, measured.yaw, measured.speed])
    solver.clear_warm_start()
    assert not solver._warm_starts


def test_iteration_limit_does_not_claim_convergence_for_a_feasible_iterate():
    cfg = MPCConfig(horizon_steps=8, max_iterations=1)
    result = MPCSolver(cfg).solve(straight(cfg))
    assert not result.converged
    assert result.status in ("feasible_not_converged", "infeasible")
    assert not result.diagnostics["attempts"][0]["optimizer_success"]


@pytest.mark.parametrize("kwargs", [
    {"dt": 0.}, {"horizon_steps": 1}, {"horizon_steps": 8.5}, {"wheelbase": 0.},
    {"accel_min": 1.}, {"steer_max": 2.}, {"jerk_max": -1.},
    {"collision_margin": -1.}, {"speed_weight": -1.}, {"dt": float("nan")},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MPCConfig(**kwargs)


def test_invalid_predictions_and_inputs_are_rejected_early():
    cfg = MPCConfig(horizon_steps=4)
    problem = straight(cfg)
    with pytest.raises(ValueError, match="N\\+1"):
        MPCSolver(cfg).solve(replace(problem, reference=replace(problem.reference, points=np.zeros((3, 4)))))
    with pytest.raises(ValueError, match="radii"):
        MPCSolver(cfg).solve(replace(problem, obstacles=(MPCObstacle("bad", np.zeros((5, 2)), -.1),)))
    with pytest.raises(ValueError, match="nonnegative speed"):
        MPCSolver(cfg).solve(replace(problem, initial=MPCInitialState(0, 0, 0, -1)))


def test_core_has_no_carla_nuplan_or_pdm_imports():
    source = Path(__file__).resolve().parents[1] / "planning" / "mpc_solver.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(any(part in name for part in ("carla", "nuplan", "tuplan", "pdm"))
                   for name in modules if name)
