"""Selective contracts through actual pinned PDM rollouts and current geometry.

CARLA-shaped fixtures are deterministic; these tests do not claim real physics
or real VLM performance. All PDM generation, simulation and scoring run normally.
"""
from copy import deepcopy
from importlib.util import find_spec
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from carla_bridge.planning.semantic_governor import SelectiveSemanticGovernor
from carla_bridge.tests.test_pdm_planner import make_planner, Vehicle, RedLight

pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


def planner(*, fog=0, others=None):
    result = make_planner(speed=4., others=others)
    result.semantic_policy = "selective"
    result.governor = SelectiveSemanticGovernor()
    result.world.get_weather = lambda: NS(fog_density=fog)
    result.map.name = "Town05_fixture"
    result.route.nav_instruction = lambda: "STRAIGHT"
    result.route.distance_completed_m = lambda: result.ego.x
    return result


def advice(reason="visibility", ids=None, signal=None):
    return dict(lateral_decision="FOLLOW_ROUTE", longitudinal_decision="STOP",
        behavior_reason="test request", risk_level="medium", safety_notes=[],
        intervention=dict(needed=True, reason_code=reason, target_actor_ids=ids or [],
            target_stop_line_id=signal, activation="immediate", release="condition_cleared", valid_for_s=2.))


@pytest.mark.parametrize("condition", ["fog_without_obstacle", "adjacent_opposing", "far_red"])
def test_ungrounded_stop_keeps_the_identical_native_trajectory(condition):
    others = [Vehicle(actor_id=7, x=15, y=3.5, yaw=180, speed=-4)] if condition=="adjacent_opposing" else []
    engine = planner(fog=60 if condition=="fog_without_obstacle" else 0, others=others)
    if condition == "far_red":
        light = RedLight(engine.map, 35.)
        engine.world.get_actors = lambda: [engine.ego, light]
    baseline = engine.plan(None, semantic_enabled=False, sim_time_s=0.)
    if condition=="far_red":
        signal = next(s for s in engine.last_context["signals"] if s["applicable"])
        request = advice("traffic_signal", signal=signal["id"])
    elif condition=="adjacent_opposing":
        request = advice("interaction", ids=[7])
    else:
        request = advice()
    proposed = engine.plan(request, semantic_update=True, sim_time_s=0.)
    d = proposed["planner_diagnostics"]
    assert proposed["trajectory"] == baseline["trajectory"]
    assert proposed["target_speed"] == baseline["target_speed"]
    assert d["candidate_pool_sha256"] == baseline["planner_diagnostics"]["candidate_pool_sha256"]
    assert not d["governor"]["authorized"]
    assert not d["governor"]["adopted"]
    assert all(c["semantic_bonus"] == 0. for c in d["candidates"])
    assert max(p["optional_v"] for p in proposed["trajectory"][:4]) > 3.5


def test_full_memory_rebuilds_bounded_numeric_candidates_and_gate_keeps_original_pool():
    engine = planner()
    prior = dict(status="execution_supported", support_count=3, record_ids=["r1","r2","r3"],
        action=dict(lateral_decision="FOLLOW_ROUTE", longitudinal_decision="MAINTAIN"),
        parameters=dict(target_velocity_mps=90.,headway_time=.1,min_gap_to_lead_agent=.1,accel_max=20.,decel_max=90.))
    evidence = dict(parameter_priors=[prior],action_feedback=[],intervention_priors=[])
    engine.experience_memory = NS(retrieve=Mock(return_value=evidence))
    engine.planning_memory_mode = "gate"
    gate = engine.plan(None, semantic_enabled=False, sim_time_s=0.)["planner_diagnostics"]
    assert not any(c.get("memory_prior") for c in gate["candidates"])
    engine.planning_memory_mode = "full"
    full = engine.plan(None, semantic_enabled=False, sim_time_s=0.)["planner_diagnostics"]
    additions = [c for c in full["candidates"] if c.get("memory_prior")]
    assert additions
    assert len(full["candidates"]) > len(gate["candidates"])
    assert {c["id"] for c in gate["candidates"]}.issubset({c["id"] for c in full["candidates"]})
    for c in additions:
        assert 2. <= c["desired_speed_mps"] <= 8.
        assert c["idm_accel_max_mps2"] <= 2.
        assert 1. <= c["idm_headway_s"] <= 3.
        assert 1.5 <= c["idm_min_gap_m"] <= 4.
        assert 2. <= c["idm_decel_max_mps2"] <= 4.
        assert c["longitudinal"] == "MAINTAIN"
        assert c["safety"]["finite"]
    original = deepcopy(prior)
    prior["support_count"] = 2
    rejected = engine.plan(None,semantic_enabled=False,sim_time_s=0.)["planner_diagnostics"]
    assert rejected["candidate_pool_sha256"] == gate["candidate_pool_sha256"]
    assert original["parameters"] == prior["parameters"]


def test_action_specific_memory_is_checked_without_granting_stop_authority():
    engine = planner(fog=60)
    scene_memory = dict(parameter_priors=[],action_feedback=[],intervention_priors=[])
    action_memory = dict(causal_advantage=dict(status="supported",lower_bound=.9,support_count=30))
    retrieve = Mock(side_effect=lambda context, action=None: action_memory if action else scene_memory)
    engine.experience_memory = NS(retrieve=retrieve)
    request = advice()
    plan = engine.plan(request,semantic_update=True,sim_time_s=0.)
    diagnostics = plan["planner_diagnostics"]
    assert retrieve.call_count == 2
    assert retrieve.call_args.args[1] is request
    assert diagnostics["requested_action_memory"] == action_memory
    assert not diagnostics["governor"]["authorized"]
    assert diagnostics["selected_id"] == diagnostics["baseline_selected_id"]
