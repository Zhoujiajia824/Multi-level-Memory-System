"""A short verified braking plan is bounded in time, space and execution age."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from carla_bridge.planning.mpc_solver import MPCObstacle
from carla_bridge.tests.test_mpc_planner import fixture


def crossing_problem(*, crossing_time_s=3., speed=.0):
    planner = fixture(speed=speed, allow_change=False, config={"max_speed_mps": .1,
        "mpc": {"dt": .2, "horizon_steps": 20, "max_iterations": 20, "steer_max": .01}})
    times = np.arange(planner.samples+1)*planner.dt
    # The initial body is unobstructed. This synthetic actor later crosses
    # the car's current footprint; narrow/low-speed constraints rule out a
    # spurious escape from the nominal four-second infeasible problem.
    actor = MPCObstacle("crossing", np.c_[np.full_like(times, .2),
                        4.*(times-crossing_time_s)], .35)
    planner._obstacles = lambda initial, tf: ([actor], [{"id": "crossing"}])
    return planner


def test_distant_prediction_conflict_allows_only_a_new_short_verified_stop():
    planner = crossing_problem()
    plan = planner.plan(None, False, sim_time_s=0.)
    d = plan["planner_diagnostics"]
    assert not d["emergency_brake"], d
    assert all(not c["optimization_feasible"] for c in d["candidates"] if not c.get("degraded_braking"))
    assert plan["selected_longitudinal"] == "STOP"
    assert plan["trajectory"][-1]["t"] == pytest.approx(1.2)
    assert len(plan["trajectory"]) == 13
    assert d["horizon_s"] == pytest.approx(1.2) and d["nominal_horizon_s"] == 4.
    recovery = d["degraded_braking"]
    assert recovery["active"] and recovery["optimization_feasible"]
    assert recovery["fresh_snapshot"] and not recovery["constraints_relaxed"]
    assert recovery["verified_constraint_max_violation"] <= recovery["verified_constraint_tolerance"]
    assert recovery["issued_at_s"] == 0.
    assert recovery["episode_deadline_s"] == 8.
    assert recovery["execution_expires_at_s"] == pytest.approx(.3)
    assert len(d["mpc_tracking"]["bicycle_angles_rad"]) == 6
    assert d["full_horizon_selection"]["selected_id"] is None
    assert d["selected_id"].endswith("@verified_braking")
    json.dumps(plan, allow_nan=False)


def test_near_term_crossing_is_rejected_even_by_short_recovery():
    planner = crossing_problem(crossing_time_s=.4)
    plan = planner.plan(None, False, sim_time_s=0.)
    d = plan["planner_diagnostics"]
    assert d["emergency_brake"] and not plan["trajectory"]
    assert d["failure_reason"] == "no_feasible_mpc_solution"
    assert not d["degraded_braking"]["active"]
    assert all(not attempt["optimization_feasible"] for attempt in d["degraded_braking"]["attempts"])


def test_recovery_has_a_continuous_eight_second_budget_not_an_infinite_retry():
    planner = crossing_problem()
    first = planner.plan(None, False, sim_time_s=10.)
    assert first["planner_diagnostics"]["degraded_braking"]["active"]
    last = planner.plan(None, False, sim_time_s=17.8)
    recovery = last["planner_diagnostics"]["degraded_braking"]
    assert recovery["active"] and recovery["episode_started_s"] == 10.
    assert recovery["execution_expires_at_s"] == pytest.approx(18.)
    timeout = planner.plan(None, False, sim_time_s=18.)
    d = timeout["planner_diagnostics"]
    assert d["emergency_brake"] and not timeout["trajectory"]
    assert d["failure_reason"] == "planner_recovery_timeout"
    assert d["degraded_braking"]["elapsed_s"] == 8.
    assert not d["degraded_braking"]["attempts"]


def test_new_observation_of_near_conflict_cannot_reuse_previous_safe_braking_plan():
    planner = crossing_problem()
    first = planner.plan(None, False, sim_time_s=0.)
    assert first["planner_diagnostics"]["degraded_braking"]["active"]
    other = crossing_problem(crossing_time_s=.4)
    planner._obstacles = other._obstacles
    second = planner.plan(None, False, sim_time_s=.2)
    assert second["planner_diagnostics"]["emergency_brake"]
    assert not second["trajectory"]


def test_full_horizon_recovery_clears_short_braking_episode_before_next_failure():
    planner = crossing_problem()
    original_obstacles = planner._obstacles
    first = planner.plan(None, False, sim_time_s=0.)
    assert first["planner_diagnostics"]["degraded_braking"]["active"]
    planner._obstacles = lambda initial, tf: ([], [])
    normal = planner.plan(None, False, sim_time_s=2.)
    assert not normal["planner_diagnostics"]["emergency_brake"]
    assert "degraded_braking" not in normal["planner_diagnostics"]
    assert planner._braking_recovery_episode is None
    planner._obstacles = original_obstacles
    new = planner.plan(None, False, sim_time_s=9.)
    assert new["planner_diagnostics"]["degraded_braking"]["episode_started_s"] == 9.


def test_short_recovery_keeps_initial_negative_dispatched_brake_and_jerk_limit():
    planner = crossing_problem(speed=.15367)
    planner._executed_desired_acceleration = {"value_mps2": -2.76863, "sim_time_s": 0.,
        "source": "actual_dispatch", "physical_acceleration_measurement": False}
    plan = planner.plan(None, False, sim_time_s=0.)
    d = plan["planner_diagnostics"]
    assert d["degraded_braking"]["active"] and not d["emergency_brake"]
    commands = np.asarray(d["mpc_tracking"]["accelerations_mps2"])
    assert commands[0] <= -1.16863+.003
    assert np.max(np.abs(np.diff(np.r_[-2.76863, commands]))) <= 1.603
    selected = next(c for c in d["candidates"] if c["id"] == d["selected_id"])
    assert selected["solver"]["previous_acceleration_demand_mps2"] == -2.76863


@pytest.mark.parametrize("previous", [2., .75, 0., -2.76863])
def test_recovery_only_keeps_positive_demand_that_jerk_history_makes_unavoidable(previous):
    planner = crossing_problem(speed=.15367 if previous < 0 else 0.)
    planner._executed_desired_acceleration = {"value_mps2": previous, "sim_time_s": 0.,
        "source": "actual_dispatch", "physical_acceleration_measurement": False}
    plan = planner.plan(None, False, sim_time_s=0.)
    d = plan["planner_diagnostics"]
    recovery = d["degraded_braking"]
    assert recovery["active"] and recovery["acceleration_policy_passed"], d
    assert recovery["terminal_speed_upper_bound_mps"] == pytest.approx(.15367 if previous < 0 else 0.)
    assert plan["target_speed"] <= recovery["terminal_speed_upper_bound_mps"]+planner.numerical_config.constraint_tolerance
    commands = np.asarray(d["mpc_tracking"]["accelerations_mps2"])
    upper = np.maximum(0., previous-np.arange(1,len(commands)+1)*planner.numerical_config.jerk_max*planner.dt)
    np.testing.assert_allclose(recovery["acceleration_upper_bounds_mps2"], upper)
    assert np.all(commands <= upper+planner.numerical_config.constraint_tolerance)
    assert np.max(np.abs(np.diff(np.r_[previous,commands]))) <= 1.603
    if previous == 2.:
        assert commands[0] == pytest.approx(.4, abs=.003)
    else:
        assert np.max(commands) <= planner.numerical_config.constraint_tolerance


def test_recovery_uses_the_longest_verified_level_without_trying_shorter_levels():
    planner = crossing_problem()
    plan = planner.plan(None,False,sim_time_s=0.)
    recovery = plan["planner_diagnostics"]["degraded_braking"]
    assert recovery["verified_horizon_s"] == pytest.approx(1.2)
    assert recovery["attempted_horizon_levels_s"] == pytest.approx([1.2])
    assert set(planner._braking_recovery_solvers) == {6}


def test_recovery_falls_back_to_a_verified_shorter_level_and_preserves_episode_and_solver_cache():
    planner = crossing_problem(crossing_time_s=1.35)
    # Same observed trajectory: an actor's constant motion intersects the
    # stopped car late in 1.2s, while the entire .8s window remains clear.
    # No geometry margin or constraint is removed.
    first = planner.plan(None,False,sim_time_s=10.)
    recovery = first["planner_diagnostics"]["degraded_braking"]
    assert recovery["active"], first["planner_diagnostics"]
    assert recovery["verified_horizon_s"] == pytest.approx(.8)
    assert recovery["attempted_horizon_levels_s"] == pytest.approx([1.2,.8])
    assert first["trajectory"][-1]["t"] == pytest.approx(.8)
    cached = dict(planner._braking_recovery_solvers)
    second = planner.plan(None,False,sim_time_s=12.)
    again = second["planner_diagnostics"]["degraded_braking"]
    assert again["episode_started_s"] == 10. and again["episode_deadline_s"] == 18.
    assert all(planner._braking_recovery_solvers[key] is solver for key,solver in cached.items())


def test_recovery_checks_all_longer_levels_before_using_the_point_six_second_minimum():
    planner = crossing_problem(crossing_time_s=1.15)
    plan = planner.plan(None,False,sim_time_s=0.)
    recovery = plan["planner_diagnostics"]["degraded_braking"]
    assert recovery["active"], plan["planner_diagnostics"]
    assert recovery["verified_horizon_s"] == pytest.approx(.6)
    assert recovery["attempted_horizon_levels_s"] == pytest.approx([1.2,.8,.6])
    assert plan["trajectory"][-1]["t"] == pytest.approx(.6)


def test_execution_never_rounds_a_fractional_verified_horizon_upward():
    planner = fixture(config={"mpc": {"dt": .35, "horizon_steps": 20}})
    initial, tf = planner._initial_state(0.)
    result = SimpleNamespace(states=np.tile([initial.x, initial.y, initial.yaw, 0.], (4, 1)),
                             controls=np.zeros((3, 2)))
    trajectory = planner._execution(result, tf)
    assert trajectory[-1]["t"] == pytest.approx(1.05)
    assert max(point["t"] for point in trajectory) <= 1.05
    assert all(a["t"] < b["t"] for a, b in zip(trajectory, trajectory[1:]))
