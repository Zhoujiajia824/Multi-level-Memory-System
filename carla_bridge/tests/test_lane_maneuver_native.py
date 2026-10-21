"""Pinned PDM generation, bicycle rollout and scoring for a legal bypass."""
from importlib.util import find_spec

import pytest

from carla_bridge.planning.lane_maneuver import LaneManeuver, maneuver_proposal_manager
from carla_bridge.planning.pdm_planner import _path_states
from carla_bridge.tests.test_pdm_planner import make_planner, Vehicle


pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


def evaluated(*, blocker_speed=0., blocker_x=28., pedestrian=False):
    blocker = Vehicle(actor_id=7, x=blocker_x, speed=blocker_speed)
    if pedestrian:
        import carla
        blocker.type_id = "walker.pedestrian.0001"
        blocker.bounding_box.extent = carla.Vector3D(x=.25, y=.25, z=.9)
    planner = make_planner(speed=1., length=160, others=[blocker])
    state, tf = planner._ego_state(2_000_000)
    planner._geometry_diagnostics = {}
    paths, lateral, centerline, area, lanes, excluded = planner._road_geometry(state)
    maneuver = LaneManeuver()
    proposal = None
    for now in (0, .2, .4, .6, .8, 1.):
        proposal = maneuver.update(planner.world, planner.ego, planner._reference_for_height, now)
    assert proposal["active"], proposal["diagnostics"]
    policies, labels, targets = planner._policies(1., 8.)
    observation = planner.api.PDMObservation(planner.sampling, planner.sampling, planner.map_radius)
    actors = planner._observed_actors(2_000_000, tf)
    lights, _ = planner._traffic_lights(lanes, 2_000_000, state)
    observation.update(state, actors, lights, lanes)
    original = planner._evaluate_candidates(paths, lateral, policies, labels, targets, state,
        observation, centerline, lanes, area, tf, [])
    paths.append(planner.api.PDMPath(_path_states(proposal["points"], planner.api)))
    lateral.append(proposal["metadata"])
    expanded = planner._evaluate_candidates(paths, lateral, policies, labels, targets, state,
        observation, centerline, lanes, area, tf, [])
    return original, expanded, proposal


@pytest.mark.parametrize("pedestrian", [False, True])
def test_native_pool_contains_a_scored_progressing_bypass_without_model(pedestrian):
    original, expanded, proposal = evaluated(pedestrian=pedestrian)
    additions = [candidate for candidate in expanded if candidate["id"].startswith("maneuver_")]
    assert len(additions) == 8
    progressing = [candidate for candidate in additions
                   if candidate["base_score"] > 0 and candidate["forward_progress_m"] > 5]
    assert progressing, additions
    assert max(candidate["forward_progress_m"] for candidate in progressing) > 10
    assert all(candidate["safety"]["finite"] for candidate in additions)
    assert all(candidate["base_score"] >= 0 for candidate in additions)


def test_full_pdm_integration_selects_and_commits_numerical_bypass_with_semantics_off():
    import numpy as np
    planner = make_planner(speed=1., length=160, others=[Vehicle(actor_id=7, x=28)])
    audits = []
    for now in (0, .2, .4, .6, .8, 1.):
        audits.append(planner.plan(None, semantic_enabled=False, sim_time_s=now)["planner_diagnostics"])
    final = audits[-1]
    selected = next(c for c in final["candidates"] if c["id"] == final["selected_id"])
    assert selected["lateral"] in ("CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT")
    assert selected["base_score"] > 0
    assert selected["forward_progress_m"] > 10
    assert final["lane_maneuver"]["committed_after_selection"]
    assert final["lane_maneuver"]["selected"] == selected["maneuver"]
    commit = final["lane_maneuver"]["execution_commit"]
    assert commit["actual_selected_id"] == selected["id"]
    assert commit["committed"]
    if not selected["maneuver"]:
        assert commit["commit_source"] == "executed_generic_numerical_candidate"
        actual = planner._evaluated_lateral_paths[selected["lateral_path_index"]]
        expected = np.array([[point.x, point.y] for point in actual.discrete_path])
        np.testing.assert_array_equal(planner.lane_maneuver._active["executed_selected_points"], expected)
    assert all(c["semantic_bonus"] == 0 for c in final["candidates"])
    assert not any(c.get("maneuver") for c in audits[0]["candidates"])
    anchor = planner.lane_maneuver._active["anchor"]
    for now in (1.2, 1.4):
        next_audit = planner.plan(None, semantic_enabled=False, sim_time_s=now)["planner_diagnostics"]
        assert next_audit["lane_maneuver"]["anchor_station_m"] == anchor
        assert planner.lane_maneuver._active["anchor"] == anchor
        assert next_audit["lane_maneuver"]["actual_selected_id"] == next_audit["selected_id"]


def test_disabled_numerical_maneuver_leaves_original_pool_size_and_never_commits():
    planner = make_planner(speed=1., length=160, others=[Vehicle(actor_id=7, x=28)])
    planner.lane_maneuver.config["enabled"] = False
    for now in (0, .2, .4, .6, .8, 1.):
        audit = planner.plan(None, semantic_enabled=False, sim_time_s=now)["planner_diagnostics"]
        assert not any(c["maneuver"] for c in audit["candidates"])
        assert len(audit["candidates"]) == 56
    assert planner.lane_maneuver._active is None


def test_maneuver_velocity_group_keeps_ordinary_native_profiles_exact_and_stop_indices_correct():
    import numpy as np
    planner = make_planner(speed=1., length=160)
    state, tf = planner._ego_state(2_000_000)
    planner._geometry_diagnostics = {}
    paths, lateral, centerline, area, lanes, excluded = planner._road_geometry(state)
    policies, labels, targets = planner._policies(1., 8.)
    observation = planner.api.PDMObservation(planner.sampling, planner.sampling, planner.map_radius)
    actors = planner._observed_actors(2_000_000, tf)
    lights, _ = planner._traffic_lights(lanes, 2_000_000, state)
    observation.update(state, actors, lights, lanes)
    baseline_manager = planner.api.PDMProposalManager(paths, policies)
    baseline = planner.generator.generate_proposals(state, observation, baseline_manager).copy()
    baseline_idm = planner.generator._state_idm_array.copy()
    original_targets = policies._target_velocities.copy()
    size = len(baseline)
    extra = dict(lateral[0], id="maneuver_low_speed", maneuver=True, maneuver_speed_cap_mps=2.)
    manager, new_labels, new_targets, sources = maneuver_proposal_manager(planner.api,
        paths + [paths[0]], lateral + [extra], policies, labels, targets)
    result = planner.generator.generate_proposals(state, observation, manager)
    assert len(manager) == size + 8
    np.testing.assert_array_equal(result[:size], baseline)
    np.testing.assert_array_equal(planner.generator._state_idm_array[:size], baseline_idm)
    np.testing.assert_array_equal(policies._target_velocities, original_targets)
    velocities = planner.generator._state_idm_array[size:, :, 1]
    assert np.max(velocities) <= 2. + 1e-6
    assert velocities[0, 1] == pytest.approx(.8)
    assert velocities[6, 1] == pytest.approx(.6)
    assert velocities[0, -1] == 0 and velocities[6, -1] == 0
    assert new_labels == labels * 2
    assert new_targets[:8] == targets and max(new_targets[8:]) <= 2.
    assert sources == list(range(8)) * 2
    assert [manager[i].proposal_idx for i in range(len(manager))] == list(range(len(manager)))
    assert all(manager[i].longitudinal_idx < 8 for i in range(size))
    assert all(manager[i].longitudinal_idx >= 8 for i in range(size, len(manager)))


def test_committed_ordinary_continuation_obeys_explicit_execution_cap_without_forging_identity():
    import numpy as np
    planner = make_planner(speed=1., length=160)
    state, tf = planner._ego_state(2_000_000)
    planner._geometry_diagnostics = {}
    paths, lateral, centerline, area, lanes, excluded = planner._road_geometry(state)
    policies, labels, targets = planner._policies(1., 8.)
    observation = planner.api.PDMObservation(planner.sampling, planner.sampling, planner.map_radius)
    observation.update(state, planner._observed_actors(2_000_000, tf), {}, lanes)
    bounded = dict(lateral[0], id="ordinary_committed_return", maneuver=False,
        maneuver_execution_limit=True, maneuver_speed_cap_mps=1.4154)
    manager, new_labels, new_targets, sources = maneuver_proposal_manager(planner.api,
        paths + [paths[0]], lateral + [bounded], policies, labels, targets)
    result = planner.generator.generate_proposals(state, observation, manager)
    count = len(paths) * len(labels)
    assert len(result) == count + len(labels)
    assert np.max(planner.generator._state_idm_array[count:, :, 1]) <= 1.4154 + 1e-6
    assert max(new_targets[len(labels):]) <= 1.4154
    assert not bounded["maneuver"]
    assert all(manager[i].longitudinal_idx >= len(labels) for i in range(count, len(manager)))
