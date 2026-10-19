"""Numerical overtaking uses the same traffic evidence in every semantic mode.

These are native PDM rollouts against the existing deterministic road fixture.
They do not contact a CARLA server or load a model.
"""
from importlib.util import find_spec
from types import SimpleNamespace as NS

import pytest

from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
from carla_bridge.tests.test_pdm_planner import Road, RedLight, Vehicle, make_planner


pytestmark = pytest.mark.skipif(
    find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
    reason="optional pinned PDM dependencies are not installed")


def _snapshot_planner(policy, *, queue):
    road = Road()
    positions, speed, stop = ((17., 31., 45., 59.), 6., 65.) if queue else ((28.,), 1., 79.)
    actors = [Vehicle(actor_id=7 + index, x=x) for index, x in enumerate(positions)]
    fixture = make_planner(speed=speed, length=160, others=actors + [RedLight(road, stop)])
    fixture.world.get_weather = lambda: NS(fog_density=0.)
    fixture.route.distance_completed_m = lambda: 0.
    fixture.route.nav_instruction = lambda: "STRAIGHT"
    planner = CarlaPDMPlanner(fixture.world, fixture.ego, fixture.route, {
        "horizon_s": 4., "semantic_policy": policy, "planning_memory": "off",
        "lane_maneuver": {"enabled": True},
    })
    assert planner.experience_memory is None
    return planner


@pytest.mark.parametrize("queue", [True, False])
def test_common_numerical_pool_uses_causal_queue_evidence_in_every_semantic_mode(queue):
    results = []
    for policy in ("always", "selective", "shadow"):
        planner = _snapshot_planner(policy, queue=queue)
        for now in (0., .2, .4, .6, .8, 1.):
            result = planner.plan(None, semantic_enabled=False, sim_time_s=now)
        diagnostics = result["planner_diagnostics"]
        context = diagnostics["planner_context"]
        assert context["traffic_wait"]["active"] is queue
        if queue:
            assert context["traffic_wait"]["reason"] == "signal_queue"
            assert context["traffic_wait"]["actor_ids"] == [7, 8, 9, 10]
            assert diagnostics["lane_maneuver"]["reason"] == "traffic_signal_or_queue"
            assert not any(item["maneuver"] for item in diagnostics["candidates"])
        else:
            # A distant red light must not hide a separate near-lane obstacle.
            assert diagnostics["lane_maneuver"]["phase"] == "passing"
            assert any(item["maneuver"] for item in diagnostics["candidates"])
        results.append(result)
    baseline = results[0]
    for result in results[1:]:
        assert result["planner_diagnostics"]["candidate_pool_sha256"] == (
            baseline["planner_diagnostics"]["candidate_pool_sha256"])
        assert result["planner_diagnostics"]["selected_id"] == (
            baseline["planner_diagnostics"]["selected_id"])
        assert result["trajectory"] == baseline["trajectory"]
