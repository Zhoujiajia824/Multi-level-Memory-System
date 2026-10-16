"""Regression of low-speed numerics and continuity through the pinned PDM core."""
from importlib.util import find_spec
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest

from carla_bridge.tests.test_pdm_planner import make_planner, semantic, Vehicle


pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


def test_low_speed_substeps_remove_reproduced_upstream_euler_oscillation():
    planner = make_planner(speed=0.6)
    old = planner.api.BatchIDMPolicy(fallback_target_velocity=0.3, speed_limit_fraction=1.0,
                                    min_gap_to_lead_agent=1.5, headway_time=2.0,
                                    accel_max=2.0, decel_max=4.0)
    old.update(None)
    stable, labels, _ = planner._policies(0.6, 8.0)
    leader = np.array([[10000.0, 0.0, 0.0]])
    old_state = new_state = np.array([[0.0, 0.6]])
    old_history, new_history = [], []
    for _ in range(60):
        old_state = old.propagate(old_state, leader, [0], 0.1)
        new_state = stable.propagate(new_state, leader, [labels.index("DECELERATE")], 0.1)
        old_history.append(old_state[0].copy())
        new_history.append(new_state[0].copy())
    old_history, new_history = np.array(old_history), np.array(new_history)
    # The actual pinned 0.1s Euler implementation reproduces negative speed,
    # backwards progress and a limit cycle at the server's 0.3m/s target.
    assert old_history[:, 1].min() < 0
    assert np.diff(old_history[:, 0]).min() < 0
    assert np.ptp(old_history[-20:, 1]) > 0.3
    assert new_history[:, 1].min() >= 0
    assert np.diff(new_history[:, 0]).min() >= 0
    assert np.ptp(new_history[-20:, 1]) < 1e-5
    assert new_history[-1, 1] == pytest.approx(0.3, abs=1e-5)


@pytest.mark.parametrize("initial_speed", [0.0, 0.2, 0.6, 1.4])
def test_real_deceleration_rollout_never_restarts_stopped_vehicle_or_reverses(initial_speed):
    planner = make_planner(speed=initial_speed, allow_change=False)
    plan = planner.plan(semantic("DECELERATE"))
    assert plan["selected_longitudinal"] == "DECELERATE"
    speeds = np.array([point["optional_v"] for point in plan["trajectory"]])
    positions = np.array([point["x"] for point in plan["trajectory"]])
    assert speeds.min() >= 0
    assert np.diff(positions).min() >= -1e-6
    assert np.diff(speeds).max() <= 1e-6
    if initial_speed < 0.6:
        assert speeds[-1] == 0
    else:
        assert np.ptp(speeds[-10:]) < 1e-4
    if initial_speed == 0:
        np.testing.assert_allclose(speeds, 0)
        assert np.ptp(positions) < 1e-6


def test_yield_can_start_and_stably_creep_at_a_low_speed_limit():
    planner = make_planner(speed=0, allow_change=False)
    planner.ego.get_speed_limit = lambda: 1.08  # 0.3m/s, the former unstable target.
    plan = planner.plan(semantic("YIELD"))
    assert plan["selected_longitudinal"] == "YIELD"
    speeds = np.array([point["optional_v"] for point in plan["trajectory"]])
    assert speeds[-1] == pytest.approx(0.3, abs=1e-4)
    assert speeds.min() >= 0 and speeds.max() <= 0.3 + 1e-6
    assert np.ptp(speeds[-10:]) < 1e-4
    assert plan["trajectory"][-1]["x"] > 1.0


def test_normal_stop_uses_comfort_deceleration_when_the_road_is_clear():
    planner = make_planner(speed=4, allow_change=False)
    plan = planner.plan(semantic("STOP"))
    diagnostics = plan["planner_diagnostics"]
    assert diagnostics["selected_id"] == "center:STOP"
    speeds = np.array([point["optional_v"] for point in plan["trajectory"]])
    assert np.diff(speeds).min() / planner.dt == pytest.approx(-2.0)
    assert speeds[-1] == 0
    hard = next(c for c in diagnostics["candidates"] if c["id"] == "center:STOP_HARD")
    assert hard["longitudinal"] == "STOP" and hard["safe"]
    assert hard["acceleration_range_mps2"][0] == pytest.approx(-4.0)


@pytest.mark.parametrize("enabled", [True, False])
def test_hard_stop_survives_when_comfort_stop_fails_predicted_ttc(enabled):
    planner = make_planner(speed=4, allow_change=False, others=[Vehicle(actor_id=9, x=10)])
    plan = planner.plan(semantic("STOP"), semantic_enabled=enabled)
    diagnostics = plan["planner_diagnostics"]
    soft = next(c for c in diagnostics["candidates"] if c["id"] == "center:STOP")
    hard = next(c for c in diagnostics["candidates"] if c["id"] == "center:STOP_HARD")
    assert not soft["safe"] and soft["predicted_ttc_infraction_time_s"] is not None
    assert hard["safe"]
    assert diagnostics["selected_id"] == "center:STOP_HARD"
    assert plan["selected_longitudinal"] == "STOP"


def test_semantic_arms_share_prior_plan_costs_and_pool_for_repeated_snapshot():
    planner = make_planner(speed=2, allow_change=False)
    first = planner.plan(semantic("YIELD"))
    first_reference = planner._previous_plan
    planner.ego.x = first["trajectory"][2]["x"]
    planner.ego.speed = first["trajectory"][2]["optional_v"]
    planner.world.get_snapshot = lambda: NS(timestamp=NS(elapsed_seconds=2.2))
    on = planner.plan(semantic("STOP"), semantic_enabled=True)["planner_diagnostics"]
    off = planner.plan(semantic("STOP"), semantic_enabled=False)["planner_diagnostics"]
    repeated = planner.plan(semantic("STOP"), semantic_enabled=True)["planner_diagnostics"]
    assert on["continuity"]["applied"]
    assert on["continuity"]["reference_age_s"] == pytest.approx(0.2)
    assert on["continuity"] == off["continuity"] == repeated["continuity"]
    assert on["candidate_pool_sha256"] == off["candidate_pool_sha256"] == repeated["candidate_pool_sha256"]
    assert on["selected_id"] == repeated["selected_id"] != off["selected_id"]
    assert on["baseline_selected_id"] == off["selected_id"]
    assert planner._continuity_reference is first_reference
    assert any(c["continuity_cost"] > 0 for c in off["candidates"])


def test_emergency_invalidates_previous_plan_reference_before_recovery():
    planner = make_planner(speed=0, allow_change=False)
    planner.plan(semantic("YIELD"))
    planner.world.get_snapshot = lambda: NS(timestamp=NS(elapsed_seconds=2.2))
    planner.world.get_actors = lambda: [planner.ego, Vehicle(actor_id=9, x=2)]
    emergency = planner.plan(semantic("YIELD"))
    assert not emergency["trajectory"]
    assert emergency["planner_diagnostics"]["failure_reason"] == "no_safe_candidate"
    planner.world.get_snapshot = lambda: NS(timestamp=NS(elapsed_seconds=2.4))
    planner.world.get_actors = lambda: [planner.ego]
    recovered = planner.plan(semantic("YIELD"))
    assert recovered["trajectory"]
    assert recovered["planner_diagnostics"]["continuity"]["reason"] == "no_previous"


@pytest.mark.parametrize("road_limit_mps", [1e-9, 0.0, -1.0, 0.09, float("nan"), float("inf")])
def test_invalid_road_limit_is_rejected_before_target_clamping_or_substep_allocation(road_limit_mps):
    planner = make_planner()
    planner.ego.get_speed_limit = lambda: road_limit_mps * 3.6
    with pytest.raises(ValueError, match="speed limit"):
        planner.plan(semantic("YIELD"))
    # Also guard direct callers of the policy factory.
    with pytest.raises(ValueError, match="speed limit"):
        planner._policies(0.0, road_limit_mps)


@pytest.mark.parametrize("state_column", [0, 1])
def test_nonfinite_internal_idm_state_is_rejected_before_json_auditing(state_column, monkeypatch):
    planner = make_planner(speed=2, allow_change=False)
    generate = planner.generator.generate_proposals

    def corrupt_internal_state(*args, **kwargs):
        proposals = generate(*args, **kwargs)
        # Keep generated pose arrays valid to reproduce a profile-only fault.
        planner.generator._state_idm_array[0, 5, state_column] = np.nan
        return proposals

    monkeypatch.setattr(planner.generator, "generate_proposals", corrupt_internal_state)
    plan = planner.plan(semantic("MAINTAIN"))
    assert plan["trajectory"]
    diagnostics = plan["planner_diagnostics"]
    invalid = diagnostics["candidates"][0]
    assert not invalid["safety"]["finite"] and not invalid["eligible"]
    assert invalid["speed_at_0_2s_mps"] is None
    assert invalid["acceleration_range_mps2"] is None
    assert diagnostics["selected_id"] != invalid["id"]
    json.dumps(diagnostics, allow_nan=False)
