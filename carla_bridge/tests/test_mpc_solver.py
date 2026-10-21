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


def test_internal_acceleration_policy_tightens_bounds_without_changing_normal_mpc():
    cfg = MPCConfig(horizon_steps=6)
    problem = straight(cfg, target=4.)
    normal = MPCSolver(cfg).solve(problem)
    constrained = MPCSolver(cfg).solve(problem, acceleration_upper_bounds=np.zeros(6))
    assert normal.feasible and normal.states[-1, 3] > 1.
    assert normal.diagnostics["acceleration_upper_bounds_mps2"] is None
    assert constrained.feasible
    np.testing.assert_allclose(constrained.states[:, 3], 0., atol=1e-9)
    np.testing.assert_array_equal(constrained.diagnostics["acceleration_upper_bounds_mps2"], 0.)


def test_terminal_stop_alone_does_not_forbid_an_unnecessary_accelerate_then_brake_profile():
    cfg = MPCConfig(horizon_steps=3)
    problem = straight(cfg, initial=MPCInitialState(0., 0., 0., 0., acceleration=.75), target=0.)
    problem = replace(problem, reference=replace(problem.reference, speed_limit=8.))
    controls = np.c_[[1.6, 0., -1.6], np.zeros(3)]
    evaluation = _Evaluation(cfg, problem)
    evaluation.evaluate(controls.ravel())
    assert np.min(evaluation.constraints) >= -1e-12
    assert evaluation.states[-1, 3] == pytest.approx(0.)
    assert evaluation.states[-1, 0] > .12
    # This physical profile passed the former terminal-only guard. The new
    # recovery policy permits zero positive demand because .75 can already
    # release to zero within the first 1.6m/s² input-change budget.
    upper = np.maximum(0., .75-np.arange(1,4)*cfg.jerk_max*cfg.dt)
    assert np.any(controls[:, 0] > upper+cfg.constraint_tolerance)
    bounded = MPCSolver(cfg).solve(problem, acceleration_upper_bounds=upper)
    assert bounded.feasible
    assert np.max(bounded.controls[:, 0]) <= cfg.constraint_tolerance
    np.testing.assert_allclose(bounded.states[:, 3], 0., atol=1e-9)


def test_rear_conflict_requiring_extra_active_acceleration_is_not_a_braking_recovery():
    cfg = MPCConfig(horizon_steps=6, max_iterations=55, multi_start=False)
    problem = straight(cfg, target=0.)
    problem = replace(problem, reference=replace(problem.reference, speed_limit=8.))
    times = np.arange(7)*cfg.dt
    rear = MPCObstacle("approaching_rear", np.c_[-1.9+times, np.zeros_like(times)], .35)
    problem = replace(problem, obstacles=(rear,))
    escape = MPCSolver(cfg).solve(problem)
    assert escape.feasible, escape.diagnostics
    assert escape.states[-1, 3] > .5
    stop = MPCSolver(cfg).solve(problem, acceleration_upper_bounds=np.zeros(6))
    assert not stop.feasible
    assert stop.diagnostics["constraint_minimums"]["obstacle:approaching_rear"] < -.1


@pytest.mark.parametrize("upper", [[0.], [0., 0., 0., float("nan")], [-5.]*4])
def test_acceleration_policy_rejects_invalid_or_outside_physical_bounds(upper):
    cfg = MPCConfig(horizon_steps=4)
    with pytest.raises(ValueError, match="acceleration_upper_bounds"):
        MPCSolver(cfg).solve(straight(cfg), acceleration_upper_bounds=upper)


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


def stopped_pose_problem(cfg, *, initial=None, name="center:STOP", width=.65):
    initial = initial or MPCInitialState(-1.3864, .03, .02, 0.)
    # A provider without preceding map samples clamps its station to this
    # forward road point. It remains the physical road constraint reference.
    points = np.tile([0., 0., 0., 0.], (cfg.horizon_steps + 1, 1))
    reference = MPCReference(name, points, width, 8., {
        "stationary_hold_pose": [initial.x, initial.y, initial.yaw],
        "stop_line": {"point": [5., 0.], "normal": [1., 0.]}})
    return MPCProblem(initial, reference)


def test_measured_stop_objective_does_not_creep_towards_forward_clamped_road_station():
    cfg = MPCConfig(horizon_steps=6)
    problem = stopped_pose_problem(cfg)
    result = MPCSolver(cfg).solve(problem)
    assert result.feasible and result.converged, result.diagnostics
    np.testing.assert_allclose(result.states, np.tile(
        [problem.initial.x, problem.initial.y, problem.initial.yaw, 0.], (7, 1)), atol=1e-9)
    np.testing.assert_allclose(result.controls[:, 0], 0., atol=1e-9)
    assert result.diagnostics["stationary_hold_objective"] is True
    np.testing.assert_array_equal(problem.reference.points[:, :2], 0.)


def test_measured_stop_objective_and_original_road_constraints_have_analytic_derivatives():
    cfg = MPCConfig(horizon_steps=6)
    problem = stopped_pose_problem(cfg, initial=MPCInitialState(-1.3864, .08, .04, .02,
                                                               steering=.02, acceleration=-.9))
    evaluation = _Evaluation(cfg, problem)
    controls = np.c_[[-.5, -.2, .15, .22, .15, .1], [.03, .04, .05, .045, .035, .025]].ravel()
    np.testing.assert_allclose(evaluation.objective_jac(controls),
        approx_derivative(lambda x: np.array([evaluation.objective(x)]), controls).ravel(),
        rtol=7e-5, atol=2e-6)
    np.testing.assert_allclose(evaluation.inequalities_jac(controls),
        approx_derivative(evaluation.inequalities, controls), rtol=7e-5, atol=2e-6)
    # The objective yaw follows the measured pose. Hard footprint orientation
    # still uses the actual map heading, and stop-line coordinates do not move.
    np.testing.assert_array_equal(evaluation.reference[:, 2], 0.)
    np.testing.assert_allclose(evaluation.objective_reference[:, 2], .04)
    np.testing.assert_allclose(evaluation.stop_line[0], [5., 0.])


def test_measured_stop_objective_cannot_move_road_corridor_or_remove_an_obstacle():
    cfg = MPCConfig(horizon_steps=4, max_iterations=12, multi_start=False)
    offroad = stopped_pose_problem(cfg, initial=MPCInitialState(-1.3864, 1., 0., 0.), width=.2)
    result = MPCSolver(cfg).solve(offroad)
    assert not result.feasible
    assert result.diagnostics["constraint_minimums"]["corridor_left"] < -.5
    centered = stopped_pose_problem(cfg)
    obstacle = MPCObstacle("occupied_hold", np.tile([0., 0.], (5, 1)), .35)
    occupied = MPCSolver(cfg).solve(replace(centered, obstacles=(obstacle,)))
    assert not occupied.feasible
    assert occupied.diagnostics["constraint_minimums"]["obstacle:occupied_hold"] < -.5


@pytest.mark.parametrize("change", ["shape", "nan", "different_pose", "cruise", "moving", "target_speed"])
def test_stationary_hold_objective_rejects_non_measured_or_non_stop_requests(change):
    cfg = MPCConfig(horizon_steps=4)
    problem = stopped_pose_problem(cfg)
    metadata = dict(problem.reference.metadata)
    if change == "shape":
        metadata["stationary_hold_pose"] = [problem.initial.x, problem.initial.y]
    elif change == "nan":
        metadata["stationary_hold_pose"] = [problem.initial.x, problem.initial.y, float("nan")]
    elif change == "different_pose":
        metadata["stationary_hold_pose"] = [problem.initial.x + .1, problem.initial.y, problem.initial.yaw]
    elif change == "cruise":
        problem = replace(problem, reference=replace(problem.reference, name="center:CRUISE"))
    elif change == "moving":
        problem = replace(problem, initial=replace(problem.initial, speed=.1))
    elif change == "target_speed":
        points = problem.reference.points.copy()
        points[-1, 3] = .1
        problem = replace(problem, reference=replace(problem.reference, points=points))
    problem = replace(problem, reference=replace(problem.reference, metadata=metadata))
    with pytest.raises(ValueError, match="stationary_hold_pose"):
        MPCSolver(cfg).solve(problem)


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


def test_dispatched_brake_demand_releases_continuously_at_standstill():
    cfg = MPCConfig(max_iterations=60, multi_start=False)
    # This is a real dispatched demand, not clipped physical-impulse telemetry.
    # A forward vehicle continues braking to zero, then releases demand with
    # hard jerk limits instead of reversing or forgetting the previous input.
    initial = MPCInitialState(0., 0., 0., .001, acceleration=-4.)
    result = MPCSolver(cfg).solve(straight(cfg, initial=initial))
    assert result.feasible and result.converged, result.diagnostics
    demand_changes = np.diff(np.r_[initial.acceleration, result.controls[:, 0]])
    assert np.max(np.abs(demand_changes)) <= cfg.jerk_max * cfg.dt + .003
    assert result.controls[0, 0] <= -2.4 + .003
    assert np.min(result.states[:, 3]) >= 0.
    assert result.states[-1, 0] > 4.
    assert result.diagnostics["stopped_under_braking_intervals"] >= 1


def test_r10_frozen_low_speed_brake_boundary_has_a_feasible_stationary_hold():
    cfg = MPCConfig()
    # Actual R10 43.9s state: the old v+a*dt constraint required v_next<0
    # because first a<=-1.16863 under jerk. Keep that command history intact.
    initial = MPCInitialState(0., 0., 0., .15367, acceleration=-2.76863)
    result = MPCSolver(cfg).solve(straight(cfg, initial=initial, target=0.))
    assert result.feasible, result.diagnostics
    # Feasibility remains separate from convergence within the finite solve
    # budget. A feasible non-converged iterate must keep that honest status.
    assert result.status in ("converged", "feasible_not_converged")
    assert result.controls[0, 0] <= -1.16863 + .003
    assert np.min(result.states[:, 3]) >= 0.
    assert np.max(result.states[:, 0]) < .02
    np.testing.assert_allclose(result.states[1:, 3], 0., atol=1e-9)
    changes = np.diff(np.r_[initial.acceleration, result.controls[:, 0]])
    assert np.max(np.abs(changes)) <= cfg.jerk_max * cfg.dt + .003
    assert result.diagnostics["previous_acceleration_demand_mps2"] == initial.acceleration
    assert result.diagnostics["jerk_limits_apply_to"] == "dispatched_acceleration_demand"


def test_braking_step_stops_at_exact_distance_and_never_turns_in_place():
    state, control = np.array([2., -1., .4, .15]), np.array([-2., -.16])
    result, a, b = bicycle_step(state, control, .2, 2.8)
    distance = .15 ** 2 / (2 * 2.)
    assert result[3] == 0.
    assert result[2] == pytest.approx(state[2] + distance * np.tan(control[1]) / 2.8)
    assert np.linalg.norm(result[:2] - state[:2]) == pytest.approx(distance)
    np.testing.assert_allclose(a,
        approx_derivative(lambda x: bicycle_step(x, control, .2, 2.8)[0], state),
        rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(b,
        approx_derivative(lambda x: bicycle_step(state, x, .2, 2.8)[0], control),
        rtol=2e-6, atol=1e-8)
    stopped, a, b = bicycle_step(result, [-2., .4], .2, 2.8)
    np.testing.assert_array_equal(stopped, result)
    np.testing.assert_allclose(b, 0., atol=0.)
    assert a[3, 3] == 0.


@pytest.mark.parametrize("speed", [0., .001, .05, .15, .5])
@pytest.mark.parametrize("demand", [-.2, -1., -4.])
def test_forward_only_stop_distance_across_acceleration_and_speed_grid(speed, demand):
    state = np.array([0., 0., 0., speed])
    result, _, _ = bicycle_step(state, [demand, 0.], .2, 2.8)
    moving_time = min(.2, speed / -demand)
    assert result[0] == pytest.approx(speed * moving_time + .5 * demand * moving_time ** 2)
    assert result[3] == pytest.approx(max(0., speed + .2 * demand))
    assert result[0] >= 0.
    assert result[3] >= 0.


def test_saturated_stop_and_restart_objective_constraints_have_consistent_derivatives():
    cfg = MPCConfig(horizon_steps=6)
    problem = straight(cfg, initial=MPCInitialState(0., 0., .04, .15, .07, -2.7), target=1.)
    obstacle = MPCObstacle("ellipse", np.tile([8., 2.], (7, 1)),
                           np.tile([2., .9], (7, 1)), .15)
    circle = MPCObstacle("circle", np.tile([7., -1.8], (7, 1)), .35)
    problem = replace(problem, obstacles=(obstacle, circle),
        reference=replace(problem.reference, metadata={
            "stop_line": {"point": [12., 0.], "normal": [1., 0.]}}))
    evaluation = _Evaluation(cfg, problem)
    controls = np.c_[[-2., -.7, .5, .2, .1, .05], [.1, .13, .15, .12, .09, .05]].ravel()
    np.testing.assert_allclose(evaluation.objective_jac(controls),
        approx_derivative(lambda x: np.array([evaluation.objective(x)]), controls).ravel(),
        rtol=7e-5, atol=2e-6)
    np.testing.assert_allclose(evaluation.inequalities_jac(controls),
        approx_derivative(evaluation.inequalities, controls), rtol=7e-5, atol=2e-6)


def test_stopping_half_step_collision_is_still_rejected():
    cfg = MPCConfig(horizon_steps=4, max_iterations=15)
    initial = MPCInitialState(0., 0., 0., .15, acceleration=-2.76863)
    problem = straight(cfg, initial=initial, target=0.)
    # Braking to rest does not license occupied vehicle space. This crossing
    # pedestrian intersects the stopped ego at the checked dt/2 sample.
    centers = np.array([[1.4, -3.], [1.4, 3.], [1.4, 6.], [1.4, 9.], [1.4, 12.]])
    result = MPCSolver(cfg).solve(replace(problem, obstacles=(MPCObstacle("crossing", centers, .35),)))
    assert not result.feasible
    assert result.diagnostics["constraint_minimums"]["obstacle:crossing"] < -.5


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


def test_verified_optimizer_iterate_survives_a_later_infeasible_final_trial(monkeypatch):
    import carla_bridge.planning.mpc_solver as module
    from scipy.optimize import OptimizeResult

    cfg = MPCConfig(horizon_steps=8, multi_start=False)
    problem = straight(cfg)
    prior = MPCSolver(cfg).solve(problem)
    assert prior.feasible and prior.converged

    def optimizer_with_bad_last_trial(*args, **kwargs):
        # The callback receives a genuine feasible optimizer iterate. Mimic a
        # finite-budget final trial jumping outside acceleration/rate bounds.
        kwargs["callback"](prior.controls.ravel())
        invalid = prior.controls.copy()
        invalid[0, 0] = cfg.accel_max + 4.
        return OptimizeResult(x=invalid.ravel(), success=False, status=9,
                              message="Iteration limit reached", nit=3)

    monkeypatch.setattr(module, "minimize", optimizer_with_bad_last_trial)
    result = MPCSolver(cfg).solve(problem)
    assert result.feasible and not result.converged
    assert result.status == "feasible_not_converged"
    np.testing.assert_array_equal(result.controls, prior.controls)
    assert result.diagnostics["attempts"][0]["selected_iterate"] == "verified_feasible_iteration"
    assert result.diagnostics["attempts"][0]["optimizer_status"] == 9
    assert result.diagnostics["max_constraint_violation"] <= cfg.constraint_tolerance


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
