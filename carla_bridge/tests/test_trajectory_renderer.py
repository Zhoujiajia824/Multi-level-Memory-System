"""Coordinate, visibility and isolation checks without a running CARLA server."""
from copy import deepcopy
import math
from types import SimpleNamespace as NS

import carla
import numpy as np
import pytest

from carla_bridge.visualization.trajectory_renderer import (
    COLORS, CarlaTrajectoryRenderer, _elevations,
    build_trajectory_visualization_snapshot, validate_trajectory_visualization_config,
)


def capture(yaw=90.):
    return carla.Transform(carla.Location(x=10., y=20., z=1.), carla.Rotation(yaw=yaw))


def world():
    calls = []
    return NS(debug=NS(draw_line=lambda *args, **kwargs: calls.append((args, kwargs))),
              calls=calls, tick=lambda: pytest.fail("Rendering must not advance simulation"))


def snapshot():
    return {"selected_index": 2, "candidate_pool_count": 3,
            "source": "native_pdm_scored_rollout",
            "reference_surface_xyz": [[10., 20., 0.], [10., 30., 1.]],
            "candidates": [{"id": str(i), "index": i, "points_world":
                            [[10. + i, 20., 0.], [10. + i, 21., .1], [10. + i, 23., .3]]}
                           for i in range(3)]}


def color(call):
    value = call[1]["color"]
    return value.r, value.g, value.b


def test_default_disabled_has_no_debug_rpc_even_without_a_world():
    renderer = CarlaTrajectoryRenderer(None)
    assert not renderer.draw(None, None, None, False)["enabled"]


@pytest.mark.parametrize("bad", [{"enabled": "on"}, {"max_candidates": -1},
                                  {"max_candidates": True}, {"point_stride": 0},
                                  {"point_stride": 1.5}, {"lifetime_s": 0},
                                  {"lifetime_s": float("inf")}, {"thickness": -1},
                                  {"z_offset": float("nan")}])
def test_invalid_config_is_rejected_before_driving(bad):
    with pytest.raises(ValueError, match="carla.visualization.trajectories"):
        validate_trajectory_visualization_config(bad)


def test_elevation_follows_sloped_resolved_route_and_separates_decks():
    result = _elevations([[2., 1.], [8., -1.]], [[0., 0., 4.], [10., 0., 9.]], 4.)
    assert result == pytest.approx([5., 8.])
    # A route can revisit exactly the same XY on another deck; select the
    # capture-height deck rather than an arbitrary first occurrence.
    crossing = [[0., 0., 0.], [10., 0., 0.], [10., 0., 10.], [0., 0., 10.]]
    assert _elevations([[5., 0.]], crossing, 10.) == pytest.approx([10.])


@pytest.mark.parametrize("yaw", [0., 90., -179.9])
def test_snapshot_rigid_body_coordinates_and_native_state_are_preserved(yaw):
    transform = capture(yaw)
    heading = -math.radians(yaw)
    c, s = math.cos(heading), math.sin(heading)
    body_offset = np.array([1.5, .2])
    delta = np.array([c * body_offset[0] - s * body_offset[1],
                      s * body_offset[0] + c * body_offset[1]])
    rear = np.array([10., -20.]) - delta
    state = NS(rear_axle=NS(x=rear[0], y=rear[1], heading=heading))
    states = np.tile([*rear, heading], (2, 4, 1))
    states[:, :, 0] += np.arange(4)[None, :] * .7
    states[:, :, 1] += np.arange(4)[None, :] * .3
    states[:, :, 2] += np.arange(4)[None, :] * .1
    # One invalid proposal should be omitted visually without changing data.
    states[1, 2, 0] = np.nan
    original = states.copy()
    candidate = [{"id": "good", "index": 0}, {"id": "bad", "index": 1}]
    result = build_trajectory_visualization_snapshot(states, candidate, 0, state,
        transform, NS(X=0, Y=1, HEADING=2), timestamp_us=12)
    assert result["frame"] == "carla_world_actor"
    assert result["candidate_pool_count"] == 2
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["points_world"][0] == pytest.approx([10., 20., 1.])
    for point, row in zip(result["candidates"][0]["points_world"], states[0]):
        c, s = math.cos(row[2]), math.sin(row[2])
        assert point[:2] == pytest.approx([row[0] + c * 1.5 - s * .2,
                                          -(row[1] + s * 1.5 + c * .2)])
    np.testing.assert_array_equal(states, original)
    assert result["timestamp_us"] == 12


def test_all_candidates_limit_winner_colors_endpoints_and_accepted_geometry():
    target = world()
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True, "max_candidates": 0,
                                                "point_stride": 2, "lifetime_s": .3})
    decision = {"trajectory": [{"x": 0., "y": 0.}, {"x": 1., "y": 0.}, {"x": 3., "y": 0.}]}
    before = deepcopy(decision)
    data = snapshot()
    audit = renderer.draw(data, decision, capture(), True, frame_id=123)
    assert audit["execution_drawn"] and not audit["rejected_drawn"]
    assert audit["drawn_candidates"] == 3
    assert audit["frame_id"] == 123
    assert [color(call) for call in target.calls] == [COLORS["candidate"], COLORS["candidate"],
                                                    COLORS["selected"], COLORS["execution"]]
    end = target.calls[-1][0][1]
    assert [end.x, end.y, end.z] == pytest.approx([10., 23., .3 + .12 + .06])
    assert all(call[1]["life_time"] == .3 and not call[1]["persistent_lines"] for call in target.calls)
    assert decision == before
    # A drawing library must never mutate its reusable pool snapshot either.
    assert data == snapshot()


def test_candidate_cap_keeps_actual_winner_even_when_it_is_outside_limit():
    target = world()
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True, "max_candidates": 1})
    result = renderer.draw(snapshot(), None, capture(), True)
    assert result["drawn_candidates"] == 1
    assert [color(call) for call in target.calls] == [COLORS["selected"]]
    assert target.calls[0][0][0].x == 12.


def test_validator_rejection_never_draws_green_execution():
    target = world()
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True})
    audit = renderer.draw(None, {"trajectory": [{"x": 0., "y": 0.}, {"x": 3., "y": 0.}]},
                          capture(), False)
    assert not audit["execution_drawn"] and audit["rejected_drawn"]
    assert [color(call) for call in target.calls] == [COLORS["rejected"]]


def test_debug_rpc_failure_is_reported_without_raising_or_touching_command():
    def fail(*args, **kwargs):
        raise RuntimeError("CARLA debug RPC unavailable")
    target = NS(debug=NS(draw_line=fail))
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True})
    command = {"trajectory": [{"x": 0., "y": 0.}, {"x": 3., "y": 0.}]}
    original = deepcopy(command)
    result = renderer.draw(snapshot(), command, capture(), True)
    assert result["error"]["type"] == "RuntimeError"
    assert not result["execution_drawn"]
    assert command == original
