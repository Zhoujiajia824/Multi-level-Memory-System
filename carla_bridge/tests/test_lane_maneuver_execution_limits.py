"""Native scored return waiting: no uncapped ordinary-lane escape or creep."""
from copy import deepcopy
from importlib.util import find_spec

import numpy as np
import pytest

from carla_bridge.tests.test_pdm_planner import make_planner


pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


def fixture(*, phase="returning", near_stop=True):
    planner = make_planner(speed=.1, x=2.6, length=160)
    planner.ego.y = -3.5
    state, tf = planner._ego_state(2_000_000)
    planner._geometry_diagnostics = {}
    paths, metadata, centerline, area, lanes, _ = planner._road_geometry(state)
    stop = [2. if near_stop else 32., 3.5]
    maneuver = {"active": True, "phase": phase, "speed_cap_mps": 1.4154,
        "metadata": {"physical_lane_side": "center" if phase == "returning" else "left",
            "return_wait_world_xy": stop, "return_required": True},
        "diagnostics": {"committed": True, "side": "left", "executed_phase": "passing"}}
    before = {entry["id"]: float(path.length) for entry, path in zip(metadata, paths)}
    planner._apply_maneuver_limits(paths, metadata, maneuver)
    return planner, state, tf, paths, metadata, centerline, area, lanes, before


def test_pending_return_limits_every_ordinary_passing_offset_but_does_not_stop_center_return():
    _, _, _, paths, metadata, _, _, _, before = fixture()
    left = [entry for entry in metadata if entry["physical_lane_side"] == "left"]
    center = [entry for entry in metadata if entry["physical_lane_side"] == "center"]
    assert len(left) >= 3 and len(center) >= 1
    assert all(entry["maneuver_execution_limit"] and entry["maneuver_wait_hold"] for entry in left)
    for path, entry in zip(paths, metadata):
        if entry["physical_lane_side"] == "left":
            assert path.length < before[entry["id"]]
        elif entry["physical_lane_side"] == "center":
            assert not entry.get("maneuver_wait_hold")
            assert path.length == before[entry["id"]]
            assert entry["maneuver_speed_cap_mps"] == 1.4154


def test_near_stop_hold_is_generated_scored_native_stop_not_postselected_velocity_edit():
    planner, state, tf, paths, metadata, centerline, area, lanes, _ = fixture()
    policies, labels, targets = planner._policies(.1, 8.)
    observation = planner.api.PDMObservation(planner.sampling, planner.sampling, planner.map_radius)
    observation.update(state, planner._observed_actors(2_000_000, tf), {}, lanes)
    candidates = planner._evaluate_candidates(paths, metadata, policies, labels, targets,
        state, observation, centerline, lanes, area, tf, [])
    waiting = [c for c in candidates if c["physical_lane_side"] == "left"]
    returning = [c for c in candidates if c["physical_lane_side"] == "center"]
    assert waiting and returning
    assert all(c["longitudinal_policy_id"] in ("STOP", "STOP_HARD") for c in waiting)
    assert all(c["maneuver_wait_hold"] and c["safety"]["finite"] for c in waiting)
    assert all(c["terminal_speed_mps"] == 0. for c in waiting)
    assert any(c["longitudinal_policy_id"] == "ACCELERATE" for c in returning)
    assert all(c["desired_speed_mps"] <= 1.4154 for c in returning)
    for candidate in waiting:
        profile = planner.generator._state_idm_array[candidate["index"], :, 1]
        assert np.max(profile[2:]) == 0.


@pytest.mark.parametrize("phase", ["passing", "returning"])
def test_stop_policy_hold_does_not_activate_early_or_at_driving_speed(phase):
    planner, state, tf, paths, metadata, _, _, _, _ = fixture(phase=phase, near_stop=False)
    assert not any(entry.get("maneuver_wait_hold") for entry in metadata)
    planner.ego.speed = 2.
    maneuver = {"active": True, "phase": phase, "speed_cap_mps": 1.4154,
        "metadata": {"physical_lane_side": "center" if phase == "returning" else "left",
                     "return_wait_world_xy": [2., 3.5]},
        "diagnostics": {"committed": True, "side": "left"}}
    planner._apply_maneuver_limits(paths, metadata, maneuver)
    assert not any(entry.get("maneuver_wait_hold") for entry in metadata)


def test_uncommitted_maneuver_cannot_restrict_an_unexecuted_candidate_pool():
    planner, state, tf, paths, metadata, _, _, _, _ = fixture()
    before = deepcopy(metadata)
    maneuver = {"active": True, "diagnostics": {"committed": False}}
    planner._apply_maneuver_limits(paths, metadata, maneuver)
    assert metadata == before
