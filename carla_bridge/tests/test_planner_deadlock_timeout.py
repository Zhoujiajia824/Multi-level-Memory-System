"""A stationary numeric failure cannot inherit a legitimate stop's exemption."""
from types import SimpleNamespace

import pytest

from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.planning.semantic_governor import SelectiveSemanticGovernor
from carla_bridge.tests.test_planner_recovery_governor import context, observe_failure, signal, actor


def loop_for(state, *, planned_stop=True):
    loop = ClosedLoop.__new__(ClosedLoop)
    loop._last_plan_diagnostics = {"planner_degradation": state}
    loop._stuck_enabled = True
    loop._planned_stop = planned_stop
    loop._stuck_speed_thresh = .3
    loop._stuck_since = None
    loop._stuck_progress = .543
    loop._stuck_term_s = 2
    loop._stuck_warn_s = 1
    loop._elapsed = 2
    loop._early_terminated = False
    loop._termination_reason = None
    loop.scenario = SimpleNamespace(route=SimpleNamespace(progress_fraction=lambda: .543))
    return loop


def native_failure(governor=None):
    governor = governor or SelectiveSemanticGovernor()
    current = observe_failure(governor)
    return governor, governor.observe_planner_health(current, 2)


def test_sustained_offroad_zero_progress_planned_stop_is_subject_to_stuck_timeout():
    _, state = native_failure()
    assert state["sustained"] and state["native_health"]["positive_progressing_count"] == 0
    loop = loop_for(state)
    loop._check_stuck(0)
    assert loop._stuck_since == 2
    loop._elapsed = 4.1
    loop._check_stuck(0)
    assert loop._early_terminated and loop._termination_reason == "stuck"


@pytest.mark.parametrize("changes", [
    {"signals": [signal()]}, {"baseline_stop_expected": True},
    {"actors": [actor()]}, {"blocked": True},
])
def test_legitimate_red_queue_blockage_or_conflict_clears_deadlock_and_does_not_time_out(changes):
    governor, state = native_failure()
    loop = loop_for(state)
    loop._check_stuck(0)
    assert loop._stuck_since == 2
    state = governor.observe_planner_health(context(2.2, **changes), 2.2)
    assert not state["sustained"]
    loop._last_plan_diagnostics = {"planner_degradation": state}
    for now in (2.2, 10, 40):
        loop._elapsed = now
        loop._check_stuck(0)
    assert loop._stuck_since is None
    assert not loop._early_terminated


def test_short_lived_degradation_cannot_start_deadlock_timeout():
    governor = SelectiveSemanticGovernor()
    state = governor.observe_planner_health(context(), 0)
    assert state["degraded"] and not state["sustained"]
    loop = loop_for(state)
    loop._check_stuck(0)
    loop._elapsed = 30
    loop._check_stuck(0)
    assert loop._stuck_since is None and not loop._early_terminated


def test_old_planned_stop_without_degradation_evidence_keeps_existing_exemption():
    loop = loop_for({})
    loop._check_stuck(0)
    loop._elapsed = 30
    loop._check_stuck(0)
    assert loop._stuck_since is None and not loop._early_terminated


@pytest.mark.parametrize("speed,progress", [(1.0, .543), (0.0, .55)])
def test_actual_motion_or_route_progress_resets_deadlock_timeout(speed, progress):
    _, state = native_failure()
    loop = loop_for(state)
    loop._check_stuck(0)
    loop._elapsed = 3
    loop.scenario.route.progress_fraction = lambda: progress
    loop._check_stuck(speed)
    assert loop._stuck_since is None and not loop._early_terminated


def test_explicitly_disabled_stuck_detection_remains_disabled():
    _, state = native_failure()
    loop = loop_for(state)
    loop._stuck_enabled = False
    loop._check_stuck(0)
    loop._elapsed = 30
    loop._check_stuck(0)
    assert loop._stuck_since is None and not loop._early_terminated
