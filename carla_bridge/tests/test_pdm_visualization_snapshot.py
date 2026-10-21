"""The optional display must not change the real pinned PDM numerical choice."""
from importlib.util import find_spec
import math

import numpy as np
import pytest

from carla_bridge.state.coords import ego_to_global
from carla_bridge.tests.test_pdm_planner import make_planner, semantic
from carla_bridge.tests.test_recovery_planner_integration import (
    make_snapshot_planner, observe_stopped_native, recovery_advice, road_snapshot,
)

pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


def test_enabled_visualization_preserves_scores_and_emits_exact_execution_geometry():
    baseline = make_planner(speed=4.)
    visual = make_planner(speed=4.)
    visual.configure_trajectory_visualization({"enabled": True, "max_candidates": 0})
    original = baseline.plan(semantic())
    displayed = visual.plan(semantic())
    assert original == displayed
    np.testing.assert_array_equal(baseline.scorer._states, visual.scorer._states)
    np.testing.assert_array_equal(baseline.generator._state_array, visual.generator._state_array)
    assert baseline.last_visualization_snapshot is None
    snapshot = visual.last_visualization_snapshot
    winner = next(c for c in displayed["planner_diagnostics"]["candidates"]
                  if c["id"] == displayed["planner_diagnostics"]["selected_id"])
    assert snapshot["selected_index"] == winner["index"]
    assert snapshot["candidate_pool_count"] == len(displayed["planner_diagnostics"]["candidates"])
    selected = next(c for c in snapshot["candidates"] if c["index"] == winner["index"])
    for key in ("lateral", "longitudinal", "physical_lane_side", "lane_offset_m",
                "lateral_path_index", "longitudinal_policy_id", "maneuver", "recovery",
                "numerically_valid", "eligible", "forward_progress_m", "terminal_speed_mps",
                "desired_speed_mps"):
        assert selected[key] == winner[key]
    selected["lateral"] = "DISPLAY_ONLY"
    assert winner["lateral"] != "DISPLAY_ONLY"
    tf = visual.ego.get_transform()
    for candidate, point in zip(selected["points_world"], displayed["trajectory"]):
        actual = ego_to_global(point["x"], point["y"], tf.location.x, tf.location.y,
                               math.radians(tf.rotation.yaw))
        assert candidate[:2] == pytest.approx(actual, abs=1e-8)
    visual.configure_trajectory_visualization({"enabled": False})
    visual.plan(semantic())
    assert visual.last_visualization_snapshot is None


def test_optional_snapshot_failure_does_not_reject_a_plan(monkeypatch):
    from carla_bridge.visualization import trajectory_renderer
    planner = make_planner()
    planner.configure_trajectory_visualization({"enabled": True})
    def fail(*args, **kwargs):
        raise RuntimeError("display error")
    monkeypatch.setattr(trajectory_renderer, "build_trajectory_visualization_snapshot", fail)
    decision = planner.plan(semantic())
    assert decision["trajectory"]
    assert planner.last_visualization_snapshot is None


@pytest.mark.parametrize("policy,expected_pool", [("selective", 160), ("shadow", 40)])
def test_snapshot_uses_executed_pool_after_optional_expansion_or_shadow_restore(
        road_snapshot, policy, expected_pool):
    planner, clock = make_snapshot_planner(road_snapshot, policy=policy)
    planner.configure_trajectory_visualization({"enabled": True})
    observe_stopped_native(planner, clock)
    result = planner.plan(recovery_advice(), semantic_update=True, sim_time_s=clock.time)
    diagnostics = result["planner_diagnostics"]
    assert diagnostics["recovery"]["expanded"]
    snapshot = planner.last_visualization_snapshot
    assert snapshot["candidate_pool_count"] == expected_pool
    assert len(snapshot["candidates"]) == expected_pool
    assert {c["id"] for c in snapshot["candidates"]} == {c["id"] for c in diagnostics["candidates"]}
    selected = next(c for c in snapshot["candidates"] if c["index"] == snapshot["selected_index"])
    assert selected["id"] == diagnostics["selected_id"]
    assert selected["id"].startswith("recovery@") == (policy == "selective")
    tf = planner.ego.get_transform()
    for rendered, executed in zip(selected["points_world"], result["trajectory"]):
        assert rendered[:2] == pytest.approx(ego_to_global(executed["x"], executed["y"],
            tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)), abs=1e-8)
