"""Real Town10HD return corner replay: native rollout, without world RPC.

The physical run returned toward the correct lane, but its only target-center
curve swept the front-left corner across the opposing-direction solid line.
This fixture preserves that pose and steering feedback. It is not a full live
traffic/physics test; independent physical completion is checked separately.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from carla_bridge.tests.test_recovery_planner_integration import (
    make_snapshot_planner, road_snapshot)


def return_snapshot(road_snapshot, *, quality=True):
    import carla
    road, _ = road_snapshot
    snap = json.loads((Path(__file__).parent / "fixtures" /
                       "town10hd_return_boundary.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(road.to_opendrive().encode()).hexdigest() == snap["map_sha256"]
    snap["planning"]["lane_quality"] = {"enabled": quality}
    planner, clock = make_snapshot_planner((road, snap))
    clock.time = snap["timestamp_s"]
    ego = planner.ego
    ego.get_velocity = lambda: carla.Vector3D(x=snap["velocity_xy"][0], y=snap["velocity_xy"][1])
    ego.get_control = lambda: carla.VehicleControl(steer=snap["steering"]["normalized_command"])
    ego.get_wheel_steer_angle = lambda wheel: snap["steering"][
        "front_left_deg" if wheel == carla.VehicleWheelLocation.FL_Wheel else "front_right_deg"]
    state, _ = planner._ego_state(round(clock.time * 1e6))
    planner._geometry_diagnostics = {}
    planner._road_geometry(state)
    lm = planner.lane_maneuver
    reference = lm._planning_reference(planner._resolved_route_waypoints)
    xy = np.array([[w.transform.location.x, -w.transform.location.y] for w in reference])
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    active = deepcopy(snap["maneuver"])
    built = lm._build(reference, stations, active["anchor"], active["length"], active["side"],
                      min(stations[-1], active["anchor"] + 90), returning=True)
    active.update(points=built[0], sample_s=built[1], widths=built[2], corridor_end=stations[-1])
    lm._active = active
    return planner, clock


def test_real_return_curve_keeps_progress_and_selects_a_body_clear_target_lane_variant(road_snapshot):
    planner, clock = return_snapshot(road_snapshot)
    nominal = planner.lane_maneuver._active["points"].copy()
    anchor = planner.lane_maneuver._active["anchor"]
    length = planner.lane_maneuver._active["length"]
    diagnostic = planner.plan(None, semantic_enabled=False, sim_time_s=clock.time)["planner_diagnostics"]
    original = next(c for c in diagnostic["candidates"] if c["id"] == "center:ACCELERATE")
    assert original["lane_quality"]["minimum_footprint_clearance_m"] < -.10
    winner = next(c for c in diagnostic["candidates"] if c["id"] == diagnostic["selected_id"])
    assert winner["physical_lane_side"] == "center", winner
    assert winner["base_score"] > .9
    assert winner["forward_progress_m"] > 25.
    assert winner["lane_quality"]["minimum_footprint_clearance_m"] > .05
    assert winner["lane_quality"]["footprint_boundary_crossing_fraction"] == 0.
    assert all(c["semantic_bonus"] == 0 for c in diagnostic["candidates"])
    assert diagnostic["safety_gate_applied"] is False
    # The real winning offset must become execution state. A scored good path
    # that cannot pass the commit interface would be rebuilt/lost next tick.
    commit = diagnostic["lane_maneuver"]["execution_commit"]
    assert commit["committed"] and commit["actual_selected_id"] == winner["id"]
    actual = planner._evaluated_lateral_paths[winner["lateral_path_index"]]
    expected = np.array([[p.x, p.y] for p in actual.discrete_path])
    active = planner.lane_maneuver._active
    np.testing.assert_array_equal(active["executed_selected_points"], expected)
    np.testing.assert_array_equal(active["nominal_points"], nominal)
    assert active["anchor"] == anchor and active["length"] == length


def test_return_curve_soft_quality_preserves_native_scores_and_simulated_geometry(road_snapshot):
    results = []
    for enabled in (False, True):
        planner, clock = return_snapshot(road_snapshot, quality=enabled)
        results.append(planner.plan(None, semantic_enabled=False, sim_time_s=clock.time)["planner_diagnostics"])
    before, after = ({c["id"]: c for c in d["candidates"]} for d in results)
    assert before.keys() == after.keys()
    for label, original in before.items():
        assert after[label]["base_score"] == original["base_score"]
        assert after[label]["trajectory_sha256"] == original["trajectory_sha256"]
