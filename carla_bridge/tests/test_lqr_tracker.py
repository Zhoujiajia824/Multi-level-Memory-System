"""Native braking equivalence with independently extended steering preview."""
from importlib.util import find_spec
from types import SimpleNamespace as NS

import numpy as np
import pytest

from carla_bridge.planning.lqr_tracker import build_lqr_tracker
from carla_bridge.tests.test_pdm_planner import make_planner, Vehicle, semantic, assert_score_only_selection

pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


def test_axis_separation_keeps_original_longitudinal_reference_and_solver():
    planner = make_planner(speed=4, allow_change=False)
    planner.plan(semantic("STOP"))
    args = dict(vehicle=planner.vehicle_parameters, discretization_time=planner.dt)
    native_short = planner.api.BatchLQRTracker(**args, tracking_horizon=10)
    native_long = planner.api.BatchLQRTracker(**args, tracking_horizon=20)
    adapted = build_lqr_tracker(planner.api, **args)
    assert isinstance(adapted, planner.api.BatchLQRTracker)
    assert adapted.longitudinal_tracking_horizon_samples == 10
    assert adapted.lateral_tracking_horizon_samples == 20
    proposals = planner.generator._state_array.copy()
    for tracker in (native_short, native_long, adapted):
        tracker.update(proposals)
    for index in (0, 15, 35, 39):
        iteration = NS(index=index)
        short_velocity, _ = native_short._compute_reference_velocity_and_curvature_profile(iteration)
        _, long_curvature = native_long._compute_reference_velocity_and_curvature_profile(iteration)
        actual_velocity, actual_curvature = adapted._compute_reference_velocity_and_curvature_profile(iteration)
        np.testing.assert_array_equal(actual_velocity, short_velocity)
        np.testing.assert_array_equal(actual_curvature, long_curvature)
    initial, reference = np.array([4., 2., 0.]), np.array([0., 1., .5])
    expected = native_short._longitudinal_lqr_controller(initial, reference)
    np.testing.assert_array_equal(adapted._longitudinal_lqr_controller(initial, reference), expected)
    assert not np.array_equal(expected, native_long._longitudinal_lqr_controller(initial, reference))
    assert adapted._tracking_horizon == 20  # No temporary mutation of lateral preview.


@pytest.mark.parametrize("enabled", [True, False])
def test_soft_and_hard_stop_preserve_native_ttc_and_entire_longitudinal_rollout(enabled):
    planner = make_planner(speed=4, allow_change=False, others=[Vehicle(actor_id=9, x=10)])
    planner.simulator._tracker = planner.api.BatchLQRTracker(
        vehicle=planner.vehicle_parameters, discretization_time=planner.dt, tracking_horizon=10)
    native = planner.plan(semantic("STOP"), semantic_enabled=enabled)["planner_diagnostics"]
    original_states = planner.scorer._states.copy()
    planner.simulator._tracker = build_lqr_tracker(planner.api, planner.vehicle_parameters, planner.dt)
    adapted = planner.plan(semantic("STOP"), semantic_enabled=enabled)["planner_diagnostics"]
    for audit in (native, adapted):
        assert_score_only_selection(audit)
        soft = next(c for c in audit["candidates"] if c["id"] == "center:STOP")
        hard = next(c for c in audit["candidates"] if c["id"] == "center:STOP_HARD")
        assert soft["safety"]["ttc"] == 0 and hard["safety"]["ttc"] == 1
        assert soft["safety"]["no_collision"] == hard["safety"]["no_collision"] == 1
        assert soft["eligible"] and hard["eligible"]
    si = planner.api.StateIndex
    for name in ("center:STOP", "center:STOP_HARD"):
        before = next(c for c in native["candidates"] if c["id"] == name)
        after = next(c for c in adapted["candidates"] if c["id"] == name)
        assert before["base_score"] == after["base_score"]
        assert before["safety"] == after["safety"]
        np.testing.assert_array_equal(original_states[before["index"]][:, [si.X, si.VELOCITY_X]],
                                      planner.scorer._states[after["index"]][:, [si.X, si.VELOCITY_X]])


@pytest.mark.parametrize("dt", [0., -.1, float("nan"), float("inf")])
def test_invalid_discretization_is_rejected_before_native_construction(dt):
    with pytest.raises(ValueError, match="discretization_time"):
        build_lqr_tracker(NS(), None, dt)


@pytest.mark.parametrize("axis", ["longitudinal", "lateral"])
@pytest.mark.parametrize("steps", [0, 1, -1, 2.5, True])
def test_invalid_preview_sample_count_is_rejected_before_native_construction(axis, steps):
    with pytest.raises(ValueError, match=axis):
        build_lqr_tracker(NS(), None, .1, **{axis + "_tracking_horizon_samples": steps})
