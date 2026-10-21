"""The executed native planner commits an overtaking anchor without a model."""
from importlib.util import find_spec

import numpy as np
import pytest

from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
from carla_bridge.tests.test_pdm_planner import make_planner, Vehicle


pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


@pytest.mark.parametrize("pedestrian", [False, True])
def test_executed_pdm_without_model_commits_the_actual_selected_bypass_geometry(pedestrian):
    blocker = Vehicle(actor_id=7, x=28., speed=0.)
    if pedestrian:
        import carla
        blocker.type_id = "walker.pedestrian.0001"
        blocker.bounding_box.extent = carla.Vector3D(x=.25, y=.25, z=.9)
    planner = make_planner(speed=1., length=160, others=[blocker])
    for now in (0., .2, .4, .6, .8, 1.):
        result = planner.plan(None, semantic_enabled=False, sim_time_s=now)
    diagnostics = result["planner_diagnostics"]
    selected = next(item for item in diagnostics["candidates"] if item["id"] == diagnostics["selected_id"])
    assert selected["maneuver"] or selected["lateral"] == "CHANGE_LANE_LEFT"
    assert selected["base_score"] > .9
    assert selected["forward_progress_m"] > 10.
    assert diagnostics["lane_maneuver"]["committed_after_selection"]
    assert planner.lane_maneuver._active is not None
    # Native progress can prefer the uncapped generic path. Its real selected
    # geometry is committed, rather than claiming the slower fixed path won.
    if not selected["maneuver"]:
        assert diagnostics["lane_maneuver"]["selected"] is False
        selected_path = planner._evaluated_lateral_paths[selected["lateral_path_index"]]
        expected = np.array([[point.x, point.y] for point in selected_path.discrete_path])
        np.testing.assert_array_equal(planner.lane_maneuver._active["executed_selected_points"], expected)
        assert diagnostics["lane_maneuver"]["execution_commit"]["commit_source"] == "executed_generic_numerical_candidate"
    assert diagnostics["lane_maneuver"]["actual_selected_id"] == selected["id"]
    anchored = planner.lane_maneuver._active["points"].copy()
    anchor_station = diagnostics["lane_maneuver"]["anchor_station_m"]
    # A new measured pose does not restart a fresh thirty-metre lane change.
    planner.ego.x, planner.ego.y = .2, -.01
    repeat = planner.plan(None, semantic_enabled=False, sim_time_s=1.2)["planner_diagnostics"]
    assert repeat["lane_maneuver"]["anchor_station_m"] == anchor_station
    np.testing.assert_array_equal(planner.lane_maneuver._active["points"], anchored)
    assert repeat["lane_maneuver"]["committed"]
    canonical = next(item for item in repeat["candidates"] if item["id"] == "left:ACCELERATE")
    assert canonical["maneuver_continuation"]
    assert canonical["maneuver_geometry_source"] == "actual_selected_map_anchor"
    fixed = next(item for item in repeat["candidates"] if item["maneuver"])
    def coordinates(candidate):
        return np.array([[p.x, p.y] for p in
            planner._evaluated_lateral_paths[candidate["lateral_path_index"]].discrete_path])
    np.testing.assert_array_equal(coordinates(canonical), coordinates(fixed))
    assert canonical["desired_speed_mps"] <= repeat["lane_maneuver"]["speed_cap_mps"]
    # Before any execution commitment the unrelated ordinary pool is intact.
    ordinary = [item for item in diagnostics["candidates"] if not item["maneuver"]]
    assert max(item["desired_speed_mps"] for item in ordinary) == 8.
    fixed = [item for item in diagnostics["candidates"] if item["maneuver"]]
    assert fixed and max(item["maneuver_speed_cap_mps"] for item in fixed) <= 5.


def test_disabled_numerical_maneuver_never_adds_or_commits_a_candidate():
    fixture = make_planner(speed=1., length=160, others=[Vehicle(actor_id=7, x=28.)])
    planner = CarlaPDMPlanner(fixture.world, fixture.ego, fixture.route,
                             {"lane_maneuver": {"enabled": False}})
    for now in (0., .5, 1., 1.5, 2.):
        diagnostics = planner.plan(None, semantic_enabled=False, sim_time_s=now)["planner_diagnostics"]
        assert not any(item["maneuver"] for item in diagnostics["candidates"])
        assert diagnostics["lane_maneuver"]["reason"] == "disabled"
        assert not diagnostics["lane_maneuver"]["committed_after_selection"]
    assert planner.lane_maneuver._active is None


def test_without_executed_maneuver_ordinary_path_geometry_is_unchanged():
    planner = make_planner(speed=1., length=160)
    state, _ = planner._ego_state(2_000_000)
    planner._geometry_diagnostics = {}
    ordinary, metadata, *_ = planner._road_geometry(state)
    before = {entry["id"]: np.array([[p.x, p.y, p.heading] for p in path.discrete_path])
              for path, entry in zip(ordinary, metadata)}
    result = planner.plan(None, semantic_enabled=False, sim_time_s=0.)["planner_diagnostics"]
    assert planner.lane_maneuver._active is None
    for path, entry in zip(planner._evaluated_lateral_paths, planner._evaluated_lateral_metadata):
        assert not entry.get("maneuver_continuation")
        np.testing.assert_array_equal(np.array([[p.x, p.y, p.heading] for p in path.discrete_path]),
                                      before[entry["id"]])


def test_executed_geometry_continues_independently_of_fixed_or_generic_provenance():
    planner = make_planner(speed=1., length=160, others=[Vehicle(actor_id=7, x=28.)])
    for now in (0., .2, .4, .6, .8, 1.):
        planner.plan(None, semantic_enabled=False, sim_time_s=now)
    active = planner.lane_maneuver._active
    assert active is not None and active["phase"] == "passing"
    anchored = active["points"].copy()
    # This field describes which actual source won. Geometry continuity is
    # governed by the committed physical phase, not this audit label.
    active["commit_source"] = "fixed"
    planner.ego.x, planner.ego.y = .2, -.01
    diagnostics = planner.plan(None, semantic_enabled=False, sim_time_s=1.2)["planner_diagnostics"]
    canonical = next(c for c in diagnostics["candidates"] if c["id"] == "left:ACCELERATE")
    fixed = next(c for c in diagnostics["candidates"] if c["maneuver"])
    assert canonical["maneuver_geometry_source"] == "actual_selected_map_anchor"
    canonical_path = planner._evaluated_lateral_paths[canonical["lateral_path_index"]]
    fixed_path = planner._evaluated_lateral_paths[fixed["lateral_path_index"]]
    np.testing.assert_array_equal(
        [[p.x, p.y] for p in canonical_path.discrete_path],
        [[p.x, p.y] for p in fixed_path.discrete_path])
    np.testing.assert_array_equal(planner.lane_maneuver._active["points"], anchored)


def test_verified_pending_return_canonical_geometry_competes_before_execution_commit():
    planner = make_planner(speed=1., length=160, others=[Vehicle(actor_id=7, x=28.)])
    for now in (0., .2, .4, .6, .8, 1.):
        planner.plan(None, semantic_enabled=False, sim_time_s=now)
    assert planner.lane_maneuver._active["phase"] == "passing"
    planner.ego.x, planner.ego.y, planner.ego.speed = 44., -3.5, 5.
    diagnostics = planner.plan(None, semantic_enabled=False, sim_time_s=1.2)["planner_diagnostics"]
    canonical = next(c for c in diagnostics["candidates"] if c["id"] == "center:ACCELERATE")
    fixed = next(c for c in diagnostics["candidates"] if c["maneuver"])
    assert canonical["maneuver_geometry_source"] == "pending_verified_return_geometry"
    assert canonical["maneuver_executed_geometry_sha256"] is None
    a = planner._evaluated_lateral_paths[canonical["lateral_path_index"]]
    b = planner._evaluated_lateral_paths[fixed["lateral_path_index"]]
    assert a is b
    assert canonical["desired_speed_mps"] <= diagnostics["lane_maneuver"]["speed_cap_mps"]
    assert diagnostics["lane_maneuver"]["executed_phase"] == "passing"
    # A fresh proposal is never labelled as executed before actual selection.
    selected = next(c for c in diagnostics["candidates"] if c["id"] == diagnostics["selected_id"])
    if diagnostics["lane_maneuver"]["committed_after_selection"]:
        assert selected["physical_lane_side"] == "center"
        assert planner.lane_maneuver._active["phase"] == "returning"


def test_ordinary_change_retains_actual_anchor_after_lane_identity_flip_and_expires_on_time_jump():
    planner = make_planner(speed=5., length=160)
    advice = {"lateral_decision": "CHANGE_LANE_LEFT", "longitudinal_decision": "MAINTAIN"}
    planner.plan(advice, sim_time_s=0.)
    original = planner._ordinary_lane_intent["path"]
    planner.ego.x, planner.ego.y = 5., -1.9
    diagnostics = planner.plan(advice, sim_time_s=.2)["planner_diagnostics"]
    left = next(c for c in diagnostics["candidates"] if c["id"] == "left:MAINTAIN")
    assert diagnostics["geometry"]["current_side"] == "left"
    assert left["lateral"] == "KEEP_LANE" and "CHANGE_LANE_LEFT" in left["lateral_semantics"]
    assert planner._evaluated_lateral_paths[left["lateral_path_index"]] is original
    assert left["lane_quality"]["boundary_reference_kind"] == "actual_source_target_lane_union"
    assert left["lane_quality"]["boundary_cost"] == 0.
    jumped = planner.plan(None, semantic_enabled=False, sim_time_s=3.)["planner_diagnostics"]
    assert jumped["ordinary_lane_intent"]["reason"] == "time_discontinuity"


def test_active_ordinary_change_is_identical_when_replanned_on_the_same_snapshot():
    import math
    from carla_bridge.state.coords import ego_to_global
    planner = make_planner(speed=5., length=160)
    advice = {"lateral_decision": "CHANGE_LANE_LEFT", "longitudinal_decision": "MAINTAIN"}
    first = planner.plan(advice, sim_time_s=0.)
    selected = next(c for c in first["planner_diagnostics"]["candidates"]
                    if c["id"] == first["planner_diagnostics"]["selected_id"])
    point = first["trajectory"][2]
    ego = planner.ego
    ego.x, ego.y = ego_to_global(point["x"], point["y"], ego.x, ego.y, math.radians(ego.yaw))
    ego.yaw = -math.degrees(planner.generator._state_array[selected["index"], 2, planner.api.StateIndex.HEADING])
    ego.speed = point["optional_v"]
    ego.get_velocity = lambda: __import__("carla").Vector3D(
        x=ego.speed * math.cos(math.radians(ego.yaw)), y=ego.speed * math.sin(math.radians(ego.yaw)))
    before = planner.plan(advice, sim_time_s=.2)
    after = planner.plan(advice, sim_time_s=.2, semantic_update=True)
    assert before["planner_diagnostics"]["ordinary_lane_intent"]["active"]
    assert after["planner_diagnostics"]["ordinary_lane_intent"]["active"]
    assert before["trajectory"] == after["trajectory"]
    assert before["planner_diagnostics"]["candidate_pool_sha256"] == after["planner_diagnostics"]["candidate_pool_sha256"]


def test_capped_maneuver_preserves_source_memory_policy_action_and_provenance():
    planner = make_planner(speed=1., length=160)
    state, tf = planner._ego_state(2_000_000)
    planner._geometry_diagnostics = {}
    paths, metadata, centerline, area, lanes, _ = planner._road_geometry(state)
    prior = dict(status="execution_supported", support_count=4, record_ids=["paired-current-evidence"],
        action={"longitudinal_decision": "MAINTAIN"},
        parameters=dict(target_velocity_mps=6., headway_time=1.7,
                        min_gap_to_lead_agent=2., accel_max=.7, decel_max=3.))
    policies, labels, targets = planner._policies(1., 8., [prior])
    observation = planner.api.PDMObservation(planner.sampling, planner.sampling, planner.map_radius)
    actors = planner._observed_actors(2_000_000, tf)
    lights, _ = planner._traffic_lights(lanes, 2_000_000, state)
    observation.update(state, actors, lights, lanes)
    paths.append(paths[0])
    metadata.append(dict(metadata[0], id="maneuver_memory_cap", maneuver=True,
                         maneuver_speed_cap_mps=2.))
    candidates = planner._evaluate_candidates(paths, metadata, policies, labels, targets,
        state, observation, centerline, lanes, area, tf, [])
    capped = next(c for c in candidates if c["id"] == "maneuver_memory_cap:MEMORY_8")
    ordinary = next(c for c in candidates if c["id"] == "center:MEMORY_8")
    assert capped["longitudinal_policy_index"] == 17
    assert capped["source_longitudinal_policy_index"] == ordinary["source_longitudinal_policy_index"] == 8
    assert capped["longitudinal"] == ordinary["longitudinal"] == "MAINTAIN"
    assert capped["memory_prior"] == ordinary["memory_prior"] == dict(
        action="MAINTAIN", support_count=4, record_ids=["paired-current-evidence"])
    assert capped["desired_speed_mps"] == 2. and ordinary["desired_speed_mps"] == 6.
    assert capped["idm_accel_max_mps2"] == ordinary["idm_accel_max_mps2"] == .7
    assert capped["idm_headway_s"] == ordinary["idm_headway_s"] == 1.7
