"""Execute the scored rollout, rather than a path-projected t=0 reference."""
from importlib.util import find_spec
import hashlib
import math

import numpy as np
import pytest

from carla_bridge.control.trajectory_validation import TrajectoryValidator
from carla_bridge.tests.test_pdm_planner import make_planner, semantic


pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


@pytest.mark.parametrize("control", ["pure_pursuit", "native_lqr"])
def test_committed_lane_change_off_reference_remains_an_anchored_executable_plan(control):
    """R05 failed here: the real car lags a fixed path by more than 0.25 m."""
    planner = make_planner(speed=6.)
    planner.lateral_control = control
    planner.plan(semantic(lateral="CHANGE_LANE_LEFT"), sim_time_s=2.)
    assert planner._ordinary_lane_intent is not None
    planner.ego.x = 8.
    result = planner.plan(semantic(lateral="CHANGE_LANE_LEFT"), sim_time_s=2.2)
    d = result["planner_diagnostics"]
    winner = next(c for c in d["candidates"] if c["id"] == d["selected_id"])
    si = planner.api.StateIndex
    reference = planner.generator._state_array[winner["index"]]
    actual = planner.scorer._states[winner["index"]]
    # Preserve the off-reference condition, not a freshly recentered path.
    assert np.linalg.norm(reference[0, si.POINT] - actual[0, si.POINT]) > .25
    check = TrajectoryValidator().validate(result["trajectory"], result["target_speed"], 6., 3.,
                                           use_speed_profile=True)
    assert check.accepted, check.to_dict()
    assert math.hypot(result["trajectory"][0]["x"], result["trajectory"][0]["y"]) < 1e-8
    audit = d["execution_trajectory"]
    assert audit["reference_origin_distance_m"] > .25
    assert audit["geometry_source"] == "native_pdm_scored_rollout"
    assert audit["selected_scored_states_sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
    from carla_bridge.state.coords import ego_to_global
    for point, row, goal in zip(result["trajectory"], actual,
                               planner.generator._state_idm_array[winner["index"], :, 1]):
        x, y = ego_to_global(point["x"], point["y"], planner.ego.x, planner.ego.y, 0.)
        heading = row[si.HEADING]
        offset = planner.vehicle_parameters.rear_axle_to_center - planner.ego.bounding_box.location.x
        assert [x, -y] == pytest.approx(row[si.POINT] + offset * np.array([math.cos(heading), math.sin(heading)]),
                                       abs=2e-6)  # CARLA Transform stores coordinates as float32.
        assert point["optional_v"] == max(0., goal)


@pytest.mark.parametrize("yaw", [0., 90., -179.9])
@pytest.mark.parametrize("pitch,roll", [(0., 0.), (10., -5.)])
def test_scored_pose_conversion_preserves_all_points_and_rotated_body_offset(yaw, pitch, roll):
    import carla
    from carla_bridge.planning.execution_trajectory import build_scored_execution_trajectory
    from carla_bridge.state.coords import ego_to_global
    planner = make_planner(speed=0.)
    planner.ego.yaw, planner.ego.x, planner.ego.y = yaw, 18., -24.
    planner.ego.get_transform = lambda: carla.Transform(planner.ego.get_location(),
        carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))
    state, transform = planner._ego_state(2_000_000)
    si = planner.api.StateIndex
    from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    row = ego_state_to_state_array(state)
    states = np.tile(row, (4, 1))
    states[:, si.X] += [0., .3, .7, 1.2]
    states[:, si.Y] += [0., -.2, -.4, -.6]
    states[:, si.HEADING] += [0., .02, .05, .09]
    reference = states.copy()
    reference[:, si.Y] += .4
    goals = np.array([0., 2., 3., 4.])
    before = states.copy(), reference.copy(), goals.copy()
    trajectory, audit = build_scored_execution_trajectory(states, reference, goals, state, transform, si, .1)
    assert math.hypot(trajectory[0]["x"], trajectory[0]["y"]) < 1e-8
    assert audit["reference_origin_distance_m"] == pytest.approx(.4)
    for point, pose in zip(trajectory, states):
        # An independent inverse rigid-body conversion recovers every scored
        # rear-axle pose, including future headings, rather than just t=0.
        capture_yaw = math.radians(transform.rotation.yaw)
        x, y = ego_to_global(point["x"], point["y"], planner.ego.x, planner.ego.y, capture_yaw)
        c, s = math.cos(pose[si.HEADING]), math.sin(pose[si.HEADING])
        dx, dy = audit["actor_from_rear_xy_m"]
        assert [x-c*dx+s*dy, -y-s*dx-c*dy] == pytest.approx(pose[si.POINT], abs=1e-8)
        assert point["optional_heading_rad"] == pytest.approx(math.atan2(
            math.sin(pose[si.HEADING]+capture_yaw), math.cos(pose[si.HEADING]+capture_yaw)))
    for array, original in zip((states, reference, goals), before):
        np.testing.assert_array_equal(array, original)


def test_stale_scored_pose_is_rejected_instead_of_being_recentred():
    from carla_bridge.planning.execution_trajectory import build_scored_execution_trajectory
    planner = make_planner()
    result = planner.plan(None, semantic_enabled=False)
    d = result["planner_diagnostics"]
    i = next(c for c in d["candidates"] if c["id"] == d["selected_id"])["index"]
    states = planner.scorer._states[i].copy()
    states[:, planner.api.StateIndex.X] += .3
    state, transform = planner._ego_state(2_000_000)
    with pytest.raises(ValueError, match="measured ego state"):
        build_scored_execution_trajectory(states, planner.generator._state_array[i],
            planner.generator._state_idm_array[i, :, 1], state, transform, planner.api.StateIndex, .1)
