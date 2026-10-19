"""A bounded measured-distance preview retains native scoring and execution."""
from copy import deepcopy
from importlib.util import find_spec

import numpy as np
import pytest

from carla_bridge.planning.lqr_tracker import speed_scaled_lateral_samples


def steps(speed, wheelbase=3., **kwargs):
    return speed_scaled_lateral_samples(speed, wheelbase, discretization_time=.1,
                                        min_samples=10, max_samples=20, **kwargs)


@pytest.mark.parametrize("speed,expected", [(0., 20), (1e-8, 20), (1., 20), (3., 20),
    (4., 15), (5., 12), (6., 10), (20., 10)])
def test_preview_measures_two_wheelbases_and_keeps_low_speed_maximum(speed, expected):
    assert steps(speed) == expected


def test_preview_uses_measured_wheelbase_and_rounds_to_nearest_sample():
    assert steps(4., wheelbase=2.8) == 14
    assert steps(4., wheelbase=3.2) == 16
    assert steps(6. / 1.449) == 14
    assert steps(6. / 1.450) == 15  # Half-sample tie retains the longer preview.
    assert steps(6. / 1.451) == 15
    values = [steps(v / 10.) for v in range(201)]
    assert values == sorted(values, reverse=True)
    assert set(values) == set(range(10, 21))


@pytest.mark.parametrize("speed", [-1., float("nan"), float("inf"), -float("inf")])
def test_nonfinite_or_reverse_speed_cannot_select_a_valid_preview(speed):
    with pytest.raises(ValueError, match="speed_mps"):
        steps(speed)


@pytest.mark.parametrize("wheelbase", [0., -1., float("nan"), float("inf")])
def test_invalid_measured_geometry_is_rejected(wheelbase):
    with pytest.raises(ValueError, match="wheelbase_m"):
        steps(3., wheelbase)


@pytest.mark.parametrize("extra", [dict(discretization_time=0.), dict(discretization_time=float("nan")),
    dict(min_samples=1), dict(max_samples=2.5), dict(min_samples=True),
    dict(min_samples=21)])
def test_invalid_sample_bounds_are_rejected(extra):
    config = dict(discretization_time=.1, min_samples=10, max_samples=20)
    config.update(extra)
    with pytest.raises(ValueError):
        speed_scaled_lateral_samples(3., 3., **config)


optional_pdm = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                 reason="optional pinned PDM dependencies are not installed")


def configured_planner(config=None, speed=0.):
    from carla_bridge.tests.test_pdm_planner import make_planner
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    fixture = make_planner(speed=speed)
    return CarlaPDMPlanner(fixture.world, fixture.ego, fixture.route, config or {})


@optional_pdm
@pytest.mark.parametrize("config,field", [
    ({"lqr_lateral_preview": "adaptive"}, "lqr_lateral_preview"),
    ({"lqr_lateral_preview": True}, "lqr_lateral_preview"),
    ({"lqr_lateral_preview": "speed_scaled", "lqr_lateral_min_horizon_s": 2.1}, "lqr_lateral_min_horizon_s"),
    ({"lqr_lateral_min_horizon_s": .15}, "lqr_lateral_min_horizon_s"),
    ({"lqr_lateral_min_horizon_s": 0.}, "lqr_lateral_min_horizon_s"),
    ({"lqr_lateral_min_horizon_s": float("inf")}, "lqr_lateral_min_horizon_s"),
    ({"lqr_lateral_min_horizon_s": float("nan")}, "lqr_lateral_min_horizon_s"),
])
def test_bad_preview_configuration_fails_before_native_rollout(config, field):
    with pytest.raises(ValueError, match=field):
        configured_planner(config)


@optional_pdm
@pytest.mark.parametrize("mode", [None, "fixed"])
def test_missing_mode_keeps_fixed_history_at_every_speed(mode):
    config = {} if mode is None else {"lqr_lateral_preview": mode}
    planner = configured_planner(config)
    original_tracker = planner.simulator._tracker
    for speed in (0., 4., 8.):
        planner.ego.speed = speed
        diag = planner.plan(None, semantic_enabled=False)["planner_diagnostics"]
        assert planner.simulator._tracker is original_tracker
        assert diag["lqr_lateral_preview"] == "fixed"
        assert diag["lqr_lateral_horizon_s"] == diag["lqr_lateral_max_horizon_s"] == 2.


@optional_pdm
def test_speed_scaled_tracker_rebuilds_only_for_a_new_sample_count(monkeypatch):
    from carla_bridge.planning import lqr_tracker
    planner = configured_planner({"lqr_lateral_preview": "speed_scaled", "lateral_control": "native_lqr"})
    original_builder = lqr_tracker.build_lqr_tracker
    calls = []

    def record_builder(*args, **kwargs):
        calls.append(kwargs["lateral_tracking_horizon_samples"])
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(lqr_tracker, "build_lqr_tracker", record_builder)
    prior, outputs = planner.simulator._tracker, []
    for speed, expected in ((0., 20), (4., 14), (4.01, 14), (5., 11), (0., 20)):
        planner.ego.speed = speed
        diag = planner.plan(None, semantic_enabled=False)["planner_diagnostics"]
        tracker = planner.simulator._tracker
        outputs.append(tracker)
        assert tracker._tracking_horizon == expected
        assert tracker._longitudinal_tracker._tracking_horizon == 10
        assert tracker._wheel_base == planner.vehicle_parameters.wheel_base
        assert diag["lqr_lateral_horizon_samples"] == expected
        assert diag["lqr_lateral_horizon_s"] == pytest.approx(expected * .1)
        assert diag["lqr_lateral_max_horizon_s"] == 2.
        assert diag["lqr_lateral_min_horizon_s"] == 1.
        selected = next(c for c in diag["candidates"] if c["id"] == diag["selected_id"])
        np.testing.assert_array_equal(diag["native_tracking"]["bicycle_angles_rad"],
            planner.scorer._states[selected["index"], :4, planner.api.StateIndex.STEERING_ANGLE])
    assert outputs[0] is prior
    assert outputs[1] is outputs[2]
    assert outputs[1] is not outputs[3] and outputs[3] is not outputs[4]
    assert calls == [14, 11, 20]


# Import the map-only fixture; it never connects to a live CARLA server.
from carla_bridge.tests.test_recovery_planner_integration import road_snapshot


@optional_pdm
@pytest.mark.parametrize("policy", ["selective", "shadow"])
def test_stopped_recovery_and_shadow_replay_share_one_low_speed_native_preview(road_snapshot, policy, monkeypatch):
    from carla_bridge.tests.test_recovery_planner_integration import (
        make_snapshot_planner, observe_stopped_native, recovery_advice)
    road, snapshot = road_snapshot
    adaptive = deepcopy(snapshot)
    adaptive["planning"]["lqr_lateral_preview"] = "speed_scaled"
    planner, clock = make_snapshot_planner((road, adaptive), policy=policy)
    baseline = observe_stopped_native(planner, clock)["planner_diagnostics"]
    original_evaluate = planner._evaluate_candidates
    evaluated = []

    def record_evaluation(paths, *args):
        evaluated.append((len(paths), planner.simulator._tracker))
        assert planner.simulator._tracker._tracking_horizon == 20
        assert planner.simulator._tracker._longitudinal_tracker._tracking_horizon == 10
        return original_evaluate(paths, *args)

    monkeypatch.setattr(planner, "_evaluate_candidates", record_evaluation)
    diag = planner.plan(recovery_advice(), semantic_update=True, sim_time_s=clock.time)["planner_diagnostics"]
    assert [count for count, _ in evaluated] == ([5, 20, 5] if policy == "shadow" else [5, 20])
    assert all(tracker is evaluated[0][1] for _, tracker in evaluated)
    assert diag["recovery"]["authorized"] and diag["recovery"]["expanded"]
    assert diag["recovery"]["positive_recovery_candidates"] > 0
    assert diag["lqr_lateral_preview"] == "speed_scaled"
    assert diag["lqr_lateral_horizon_s"] == 2.
    if policy == "shadow":
        assert diag["native_tracking"] == baseline["native_tracking"]
        assert diag["selected_id"] == baseline["selected_id"] == "center:STOP"
    else:
        selected = next(c for c in diag["candidates"] if c["id"] == diag["selected_id"])
        assert selected["recovery"] and selected["base_score"] > 0
