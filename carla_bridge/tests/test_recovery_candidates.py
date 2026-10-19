"""Finite current-lane recovery search; scoring remains the caller's task."""
from copy import deepcopy
from importlib.util import find_spec
from types import SimpleNamespace as NS

import numpy as np
import pytest

from carla_bridge.planning.recovery import propose_recovery_paths, validate_recovery_config


@pytest.mark.parametrize("config", [
    [], {"enabled": 1}, {"enabled": "on"}, {"max_paths": True},
    {"max_paths": 0}, {"max_paths": 16}, {"max_paths": 1.5},
    {"transition_lengths_m": []}, {"transition_lengths_m": "2,3,4"},
    {"transition_lengths_m": [1, 2, 3, 4]}, {"transition_lengths_m": [2, 2]},
    {"transition_lengths_m": [float("nan")]}, {"transition_lengths_m": [float("inf")]},
    {"transition_lengths_m": [.5]}, {"transition_lengths_m": [7]},
    {"transition_lengths_m": [10 ** 1000]},
    {"transition_lengths_m": [True]}, {"max_offset_m": -.1},
    {"max_offset_m": .51}, {"max_offset_m": float("nan")},
    {"max_offset_m": True}, {"max_offset_m": 10 ** 1000},
    {"side_clearance_m": 0}, {"side_clearance_m": float("inf")}, {"side_clearance_m": 10 ** 1000},

])
def test_invalid_search_budget_is_rejected_before_planning(config):
    with pytest.raises(ValueError):
        validate_recovery_config(config)


def test_defaults_are_bounded_and_validation_does_not_mutate_configuration():
    source = {"transition_lengths_m": [2, 3], "trigger_duration_s": 5}
    original = deepcopy(source)
    result = validate_recovery_config(source)
    assert source == original
    assert result["trigger_duration_s"] == 5
    result["transition_lengths_m"].append(4.)
    assert source == original
    defaults = validate_recovery_config()
    assert defaults["enabled"] is True
    assert defaults["max_paths"] == 15
    assert defaults["transition_lengths_m"] == [2., 3., 4.]



def test_disabled_extension_preserves_objects_and_needs_no_optional_dependency():
    original_path = object()
    original_metadata = {"id": "center"}
    paths, metadata, info = propose_recovery_paths([original_path], [original_metadata],
        [], None, None, None, config={"enabled": False})
    assert paths == [original_path]
    assert metadata[0] is original_metadata
    assert info["added_path_count"] == 0
    assert info["reason"] == "disabled"


def test_mismatched_original_geometry_and_metadata_fails_explicitly():
    with pytest.raises(ValueError, match="metadata"):
        propose_recovery_paths([object()], [], [], None, None, None)


def test_unresolved_current_lane_cannot_be_replaced_by_nearest_lane():
    paths, metadata, info = propose_recovery_paths([object()], [{"id": "center", "label": "CHANGE_LANE_LEFT"}],
        [], None, None, None)
    assert len(paths) == len(metadata) == 1
    assert info["reason"] == "current_lane_not_uniquely_resolved"


@pytest.fixture
def native_geometry():
    if find_spec("tuplan_garage") is None or find_spec("nuplan") is None:
        pytest.skip("optional pinned PDM dependencies are not installed")
    from carla_bridge.tests.test_pdm_planner import make_planner
    engine = make_planner(x=12.)
    engine._geometry_diagnostics = {}
    state, _ = engine._ego_state(2_000_000)
    paths, metadata, *_ = engine._road_geometry(state)
    return engine, state, paths, metadata


def expand(fixture, *, config=None, reference=None, vehicle=None, metadata=None):
    engine, state, paths, base_metadata = fixture
    return propose_recovery_paths(paths, metadata if metadata is not None else base_metadata,
        engine._reference_for_height if reference is None else reference, state,
        engine.api, vehicle if vehicle is not None else engine.vehicle_parameters, config=config)


def test_native_paths_preserve_original_pool_measured_pose_and_current_lane(native_geometry):
    engine, state, original_paths, original_metadata = native_geometry
    untouched = deepcopy(original_metadata)
    old_arrays = [path._states_se2_array.copy() for path in original_paths]
    paths, metadata, info = expand(native_geometry)
    count = len(original_paths)
    assert len(paths) == count + 15
    assert all(paths[i] is path for i, path in enumerate(original_paths))
    assert all(metadata[i] is entry for i, entry in enumerate(original_metadata))
    assert original_metadata == untouched
    for path, original in zip(original_paths, old_arrays):
        np.testing.assert_array_equal(path._states_se2_array, original)
    assert len({entry["id"] for entry in metadata}) == len(metadata)
    for path, entry in zip(paths[count:], metadata[count:]):
        pose = path.discrete_path[0]
        assert (pose.x, pose.y, pose.heading) == pytest.approx(
            (state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading))
        assert np.isfinite(path._states_se2_array).all()
        assert path.length > 40
        assert entry["physical_lane_side"] == "center"
        assert entry["semantics"] == ["FOLLOW_ROUTE", "KEEP_LANE"]
        assert entry["route_target"] is True
        assert entry["recovery"] is True
        assert entry["recovery_recipe"] == "short_pose_rejoin"
        assert 2 <= entry["transition_length_m"] <= 4
        assert abs(entry["lane_offset_m"]) <= .5
    assert info["native_resimulation_required"] is True
    assert info["score_relaxation_applied"] is False
    assert info["added_path_count"] == 15


def test_smaller_compute_budget_still_covers_each_configured_recovery_length(native_geometry):
    count = len(native_geometry[2])
    paths, metadata, info = expand(native_geometry, config={"max_paths": 3})
    assert len(paths) == count + 3
    assert {entry["transition_length_m"] for entry in metadata[count:]} == {2., 3., 4.}
    assert {entry["lane_offset_m"] for entry in metadata[count:]} == {0.}
    assert info["attempted_path_count"] == 3


@pytest.mark.parametrize("width,expected_limit", [(2.8, .25), (3.15, .07), (3.45, 0.)])
def test_lateral_search_shrinks_to_the_actual_vehicle_clearance(native_geometry, width, expected_limit):
    count = len(native_geometry[2])
    paths, metadata, info = expand(native_geometry, vehicle=NS(width=width))
    assert info["within_lane_offset_limit_m"] == pytest.approx(expected_limit)
    assert max(abs(entry["lane_offset_m"]) for entry in metadata[count:]) <= expected_limit + 1e-9
    assert len({entry["id"] for entry in metadata}) == len(paths)
    assert len(paths) - count == (3 if not expected_limit else 9)


def test_recovery_is_confined_to_resolved_adjacent_current_lane(native_geometry):
    engine, state, paths, metadata = native_geometry
    current = next(entry for entry in metadata if entry["id"] == "center")
    current["label"], current["semantics"] = "CHANGE_LANE_RIGHT", ["CHANGE_LANE_RIGHT", "FOLLOW_ROUTE"]
    left = next(entry for entry in metadata if entry["id"] == "left")
    left["label"], left["semantics"] = "KEEP_LANE", ["KEEP_LANE"]
    state.rear_axle.y = 3.5
    new_paths, new_metadata, info = expand(native_geometry)
    assert info["physical_lane_side"] == "left"
    assert all(entry["physical_lane_side"] == "left" for entry in new_metadata[len(paths):])
    assert all(not entry["route_target"] for entry in new_metadata[len(paths):])


def test_disconnected_current_lane_is_not_joined_for_escape(native_geometry):
    engine = native_geometry[0]
    reference = list(engine._reference_for_height)
    reference[20].road_id = 999
    paths, metadata, info = expand(native_geometry, reference=reference)
    assert len(paths) == len(native_geometry[2])
    assert info["added_path_count"] == 0
    assert info["reason"] == "current_lane_corridor_not_continuous"
    assert info["corridor_cut"]["reason"] == "disconnected_lane_successor"


def test_duplicate_recovery_ids_are_not_appended_again(native_geometry):
    paths, metadata, info = expand(native_geometry)
    engine, state, *_ = native_geometry
    repeated_paths, repeated_metadata, repeated = propose_recovery_paths(paths, metadata,
        engine._reference_for_height, state, engine.api, engine.vehicle_parameters)
    assert len(repeated_paths) == len(paths)
    assert len(repeated_metadata) == len(metadata)
    assert repeated["added_path_count"] == 0
    assert all(item["reason"] == "duplicate_id" for item in repeated["excluded"])


def test_reference_endpoint_does_not_create_a_path_beyond_the_map(native_geometry):
    engine, state, *_ = native_geometry
    state.rear_axle.x = 200.
    paths, metadata, info = expand(native_geometry)
    assert len(paths) == len(native_geometry[2])
    assert not info["added_ids"]
    assert all(item["reason"] == "insufficient_path_ahead" for item in info["excluded"])
