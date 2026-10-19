"""Replay a real Town10HD stopped pose through the pinned PDM pipeline.

This exercises actual CARLA OpenDRIVE topology and native generation/LQR/scoring,
but has no simulator physics, live traffic, cameras or real model call. Install
the optional PDM sources and set CARLA_TEST_TOWN10HD_XODR on other machines.
"""
import hashlib
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "town10hd_stopped_recovery.json"
DEFAULT_MAP = Path("D:/software/carla/WindowsNoEditor/CarlaUE4/Content/Carla/Maps/OpenDrive/Town10HD.xodr")


@pytest.fixture(scope="module")
def road_snapshot():
    if find_spec("tuplan_garage") is None or find_spec("nuplan") is None:
        pytest.skip("optional pinned PDM dependencies are not installed")
    path = Path(os.environ.get("CARLA_TEST_TOWN10HD_XODR", str(DEFAULT_MAP)))
    if not path.is_file():
        pytest.skip("set CARLA_TEST_TOWN10HD_XODR to the CARLA 0.9.15 Town10HD OpenDRIVE map")
    import carla
    snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode()).hexdigest() == snapshot["map_sha256"], "Town10HD map differs from the recorded server map"
    return carla.Map("Town10HD", source), snapshot


def make_snapshot_planner(road_snapshot, *, policy="selective", recovery_enabled=True):
    import carla
    from carla_bridge.env.route_planner import RoutePlanner
    from carla_bridge.env.route_progress import RouteProgress
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    road, snapshot = road_snapshot
    pose, vehicle = snapshot["pose"], snapshot["vehicle"]

    class SnapshotEgo:
        id, type_id = 1000000, "vehicle.tesla.model3"
        bounding_box = carla.BoundingBox(carla.Location(*vehicle["bounding_box_offset_xyz"]),
            carla.Vector3D(vehicle["length_m"]/2, vehicle["width_m"]/2, .75))

        def get_location(self):
            return carla.Location(x=pose["x"], y=pose["y"], z=pose["z"])

        def get_transform(self):
            return carla.Transform(self.get_location(), carla.Rotation(yaw=math.degrees(pose["yaw_rad"])))

        def get_velocity(self): return carla.Vector3D()
        def get_acceleration(self): return carla.Vector3D()
        def get_angular_velocity(self): return carla.Vector3D()
        def get_control(self): return carla.VehicleControl()
        def get_speed_limit(self): return 50.

        def get_physics_control(self):
            rear, wheels = vehicle["rear_axle_actor_xyz_m"], []
            for front in (True, False):
                for sign in (-1, 1):
                    position = self.get_transform().transform(carla.Location(
                        x=rear[0]+(vehicle["wheelbase_m"] if front else 0.),
                        y=sign*vehicle["front_track_m"]/2, z=rear[2]))
                    wheels.append(NS(position=carla.Vector3D(x=position.x*100, y=position.y*100, z=position.z*100),
                        max_steer_angle=math.degrees(vehicle["physical_inner_max_steer_rad"]) if front else 0.))
            return NS(wheels=wheels)

    ego, clock = SnapshotEgo(), NS(time=0.)
    world = NS(get_map=lambda: road, get_actors=lambda: [ego],
        get_weather=lambda: NS(fog_density=0.),
        get_snapshot=lambda: NS(timestamp=NS(elapsed_seconds=clock.time)))
    route = RoutePlanner(world, ego)
    route.load_fixed_points([dict(x=x, y=y, z=0.) for x, y in snapshot["route_xy"]])
    # Restore the recorded ordered-checkpoint state without importing logs or
    # geometrically snapping across the route's overlapping loop segments.
    route._progress = RouteProgress(route._route)
    route._progress.index = snapshot["route_progress"]["index"]
    route._progress.fraction = snapshot["route_progress"]["fraction"]
    route._progress.started = True
    route._progress._last = (pose["x"], pose["y"], 0.)
    config = dict(snapshot["planning"], semantic_policy=policy, planning_memory="off",
                  lateral_control="native_lqr",
                  recovery={"enabled": recovery_enabled})
    planner = CarlaPDMPlanner(world, ego, route, config)
    return planner, clock


def recovery_advice(*, expand=True):
    from src.vla_memory.decision.semantic_output_parser import parse_semantic_decision_output
    raw = dict(lateral_decision="FOLLOW_ROUTE", longitudinal_decision="MAINTAIN",
        behavior_reason="原生候选持续无法前进，请扩大当前车道的位姿恢复候选。", risk_level="medium", safety_notes=[],
        intervention=dict(needed=expand, reason_code="planner_degraded" if expand else "none",
            target_actor_ids=[], target_stop_line_id=None, activation="immediate",
            release="condition_cleared", valid_for_s=6.,
            planner_request="expand_candidates" if expand else "none"))
    parsed, errors = parse_semantic_decision_output(json.dumps(raw, ensure_ascii=False), require_intervention=True)
    assert not errors
    return parsed


def observe_stopped_native(planner, clock, *, semantic_enabled=True):
    results = []
    for timestamp in (0., .5, 1., 1.5, 2.):
        clock.time = timestamp
        result = planner.plan(None, semantic_enabled=semantic_enabled, sim_time_s=timestamp)
        diagnostics = result["planner_diagnostics"]
        assert diagnostics["selected_id"] == "center:STOP"
        moving = [c for c in diagnostics["candidates"] if c["forward_progress_m"] > .2]
        assert moving and all(c["base_score"] == 0. for c in moving)
        assert not diagnostics["recovery"]["expanded"]
        results.append(result)
    return results[-1]


def test_real_town10hd_baseline_reproduces_the_zero_moving_score_deadlock(road_snapshot):
    planner, clock = make_snapshot_planner(road_snapshot)
    baseline = observe_stopped_native(planner, clock)
    diagnostics = baseline["planner_diagnostics"]
    assert diagnostics["geometry"]["ego_lane_key"] == "848:0:-1"
    assert len(diagnostics["candidates"]) == 40
    assert diagnostics["planner_context"]["planner_health"]["positive_progressing_count"] == 0
    query = planner.query_decision(clock.time)
    assert query["query"]
    assert query["planner_recovery"]["sustained"]
    assert max(point["optional_v"] for point in baseline["trajectory"]) == 0.


def test_model_request_expands_and_rescores_whole_native_pool_to_a_positive_forward_option(road_snapshot, monkeypatch):
    planner, clock = make_snapshot_planner(road_snapshot)
    baseline = observe_stopped_native(planner, clock)
    original_evaluate = planner._evaluate_candidates
    evaluated = []

    def record_whole_pool(paths, *args):
        evaluated.append(len(paths))
        return original_evaluate(paths, *args)

    monkeypatch.setattr(planner, "_evaluate_candidates", record_whole_pool)
    result = planner.plan(recovery_advice(), semantic_update=True, sim_time_s=clock.time)
    diagnostics = result["planner_diagnostics"]
    recovery = diagnostics["recovery"]
    assert evaluated == [5, 20]
    assert recovery["authorized"] and recovery["expanded"] and recovery["selected"]
    assert recovery["added_paths"] == 15
    assert recovery["positive_recovery_candidates"] > 0
    tracking = diagnostics["native_tracking"]
    assert tracking["source"] == "native_pdm_scored_rollout"
    assert tracking["issued_at_s"] == clock.time
    assert tracking["expires_at_s"] == pytest.approx(clock.time + .3)
    selected_index = next(c["index"] for c in diagnostics["candidates"] if c["id"] == diagnostics["selected_id"])
    assert tracking["bicycle_angles_rad"] == planner.scorer._states[
        selected_index, :4, planner.api.StateIndex.STEERING_ANGLE].tolist()
    assert len(diagnostics["candidates"]) == 160
    assert sum(c["recovery"] for c in diagnostics["candidates"]) == 120
    selected = next(c for c in diagnostics["candidates"] if c["id"] == diagnostics["selected_id"])
    assert selected["recovery"] and selected["base_score"] > 0.
    assert selected["safety"]["drivable_area"] == 1.
    assert selected["forward_progress_m"] > .2
    assert max(point["optional_v"] for point in result["trajectory"]) > .2
    # Governor selection adds its own regret/memory audit fields. Every field
    # captured from the unmodified original scorer/selector must still match.
    for original, prior in zip(diagnostics["original_baseline_candidates"],
                               baseline["planner_diagnostics"]["candidates"]):
        assert all(prior[key] == value for key, value in original.items())
    assert diagnostics["original_candidate_pool_sha256"] == baseline["planner_diagnostics"]["candidate_pool_sha256"]


@pytest.mark.parametrize("case", ["semantic_off", "request_none", "no_request", "recovery_disabled"])
def test_recovery_requires_enabled_semantics_and_an_explicit_model_request(road_snapshot, case):
    planner, clock = make_snapshot_planner(road_snapshot, recovery_enabled=case != "recovery_disabled")
    baseline = observe_stopped_native(planner, clock, semantic_enabled=case != "semantic_off")
    request = None if case == "no_request" else recovery_advice(expand=case != "request_none")
    result = planner.plan(request, semantic_enabled=case != "semantic_off",
                          semantic_update=True, sim_time_s=clock.time)
    diagnostics = result["planner_diagnostics"]
    assert not diagnostics["recovery"]["expanded"]
    assert "tracking" not in diagnostics["recovery"]
    assert len(diagnostics["candidates"]) == 40
    assert result["trajectory"] == baseline["trajectory"]
    assert result["target_speed"] == baseline["target_speed"]
    assert diagnostics["candidate_pool_sha256"] == baseline["planner_diagnostics"]["candidate_pool_sha256"]


def test_shadow_uses_expanded_pool_only_for_audit_and_restores_exact_native_baseline(road_snapshot, monkeypatch):
    planner, clock = make_snapshot_planner(road_snapshot, policy="shadow")
    baseline = observe_stopped_native(planner, clock)
    original_evaluate = planner._evaluate_candidates
    evaluated = []

    def record_whole_pool(paths, *args):
        evaluated.append(len(paths))
        return original_evaluate(paths, *args)

    monkeypatch.setattr(planner, "_evaluate_candidates", record_whole_pool)
    result = planner.plan(recovery_advice(), semantic_update=True, sim_time_s=clock.time)
    diagnostics = result["planner_diagnostics"]
    assert evaluated == [5, 20, 5]
    assert diagnostics["recovery"]["expanded"]
    assert diagnostics["recovery"]["would_select_recovery"]
    assert not diagnostics["recovery"]["selected"]
    assert "tracking" not in diagnostics["recovery"]
    assert diagnostics["shadow_selected_id"].startswith("recovery@")
    assert len(diagnostics["shadow_candidates"]) == 160
    assert len(diagnostics["candidates"]) == 40
    assert result["trajectory"] == baseline["trajectory"]
    assert result["target_speed"] == baseline["target_speed"]
    assert diagnostics["selected_id"] == baseline["planner_diagnostics"]["selected_id"]
    assert diagnostics["candidate_pool_sha256"] == baseline["planner_diagnostics"]["candidate_pool_sha256"]
    assert diagnostics["candidates"] == baseline["planner_diagnostics"]["candidates"]
    assert diagnostics["native_tracking"] == baseline["planner_diagnostics"]["native_tracking"]


@pytest.mark.parametrize("stage", ["generate_paths", "evaluate_candidates"])
@pytest.mark.parametrize("policy", ["selective", "shadow"])
def test_optional_recovery_failure_restores_the_original_native_state_and_clears_request(
        road_snapshot, monkeypatch, stage, policy):
    planner, clock = make_snapshot_planner(road_snapshot, policy=policy)
    baseline = observe_stopped_native(planner, clock)
    original_evaluate = planner._evaluate_candidates
    native_states = planner.generator._state_array.copy()
    native_idm_states = planner.generator._state_idm_array.copy()
    evaluated, proposal_calls = [], []

    def injected_evaluate(paths, *args):
        evaluated.append(len(paths))
        result = original_evaluate(paths, *args)
        if stage == "evaluate_candidates" and len(paths) > 5:
            # Fail after the entire native generator/simulator/scorer has been
            # overwritten with the expanded pool, not before mutation occurs.
            assert planner.generator._state_array.shape[0] == 160
            raise RuntimeError("injected failure after expanded native evaluation")
        return result

    if stage == "generate_paths":
        def fail_proposal(*args, **kwargs):
            proposal_calls.append(True)
            raise ValueError("injected optional path construction failure")
        monkeypatch.setattr("carla_bridge.planning.recovery.propose_recovery_paths", fail_proposal)
    monkeypatch.setattr(planner, "_evaluate_candidates", injected_evaluate)
    result = planner.plan(recovery_advice(), semantic_update=True, sim_time_s=clock.time)
    diagnostics = result["planner_diagnostics"]
    recovery = diagnostics["recovery"]
    assert recovery["fallback_to_original"]
    assert not recovery["authorized"] and not recovery["expanded"] and not recovery["selected"]
    assert recovery["error"]["stage"] == stage
    assert recovery["error"]["type"] == ("ValueError" if stage == "generate_paths" else "RuntimeError")
    assert recovery["reason"] == "recovery_failed_original_pool_restored"
    assert evaluated == ([5, 5] if stage == "generate_paths" else [5, 20, 5])
    assert len(diagnostics["candidates"]) == 40
    assert diagnostics["candidate_pool_sha256"] == baseline["planner_diagnostics"]["candidate_pool_sha256"]
    assert diagnostics["selected_id"] == baseline["planner_diagnostics"]["selected_id"]
    assert result["trajectory"] == baseline["trajectory"]
    assert result["target_speed"] == baseline["target_speed"]
    assert planner.governor._pending is None
    np.testing.assert_array_equal(planner.generator._state_array, native_states)
    np.testing.assert_array_equal(planner.generator._state_idm_array, native_idm_states)
    assert planner.scorer._ego_coords.shape[0] == 40

    # A surviving lease must not retry a failed optional expansion on each
    # fast planning cycle. A future consultation may issue a fresh request.
    clock.time += .2
    following = planner.plan(None, sim_time_s=clock.time)
    assert len(evaluated) == (3 if stage == "generate_paths" else 4)
    assert evaluated[-1] == 5
    assert not following["planner_diagnostics"]["recovery"]["expanded"]
    assert following["trajectory"] == baseline["trajectory"]
    if stage == "generate_paths":
        assert len(proposal_calls) == 1


def test_failed_restoration_of_original_native_pool_is_not_reported_as_success(road_snapshot, monkeypatch):
    planner, clock = make_snapshot_planner(road_snapshot)
    observe_stopped_native(planner, clock)
    original_evaluate = planner._evaluate_candidates
    expanded_failed = False
    evaluated = []

    def fail_expansion_and_restoration(paths, *args):
        nonlocal expanded_failed
        evaluated.append(len(paths))
        if expanded_failed and len(paths) == 5:
            raise RuntimeError("injected original native pool restoration failure")
        result = original_evaluate(paths, *args)
        if len(paths) > 5:
            expanded_failed = True
            raise ValueError("injected expanded pool failure")
        return result

    monkeypatch.setattr(planner, "_evaluate_candidates", fail_expansion_and_restoration)
    with pytest.raises(RuntimeError, match="original native pool restoration failure"):
        planner.plan(recovery_advice(), semantic_update=True, sim_time_s=clock.time)
    assert evaluated == [5, 20, 5]
    assert planner.governor._pending is None
