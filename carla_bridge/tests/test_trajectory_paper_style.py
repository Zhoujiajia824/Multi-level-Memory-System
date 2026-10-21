"""Geometry-preserving visibility, aggregation and display-budget contracts."""
from copy import deepcopy

import numpy as np
import pytest

from carla_bridge.visualization.trajectory_renderer import (
    CarlaTrajectoryRenderer, PAPER_COLORS, _display_groups,
    validate_trajectory_visualization_config,
)
from carla_bridge.tests.test_trajectory_renderer import world, color, capture


def candidate(index, points=None, side="center", **metadata):
    return {"id": f"{side}:policy{index}", "index": index,
            "points_world": points if points is not None else [[10., 20., 0.], [10., 30., 0.]],
            "physical_lane_side": side, "base_score": 1. - index / 1000., **metadata}


def pool(candidates, selected=0):
    return {"selected_index": selected, "candidate_pool_count": len(candidates), "candidates": candidates}


@pytest.mark.parametrize("bad", [{"style": "unknown"}, {"endpoint_markers": "yes"},
                                  {"merge_distance_m": -1}, {"merge_distance_m": float("nan")},
                                  {"outline_width": 0}, {"dash_gap_m": 0}])
def test_paper_config_rejects_unbounded_or_invalid_rendering_options(bad):
    with pytest.raises(ValueError, match="carla.visualization.trajectories"):
        validate_trajectory_visualization_config(bad)


def test_near_duplicate_display_group_preserves_actual_winner_and_all_member_ids():
    data = pool([candidate(0), candidate(1), candidate(2)], selected=2)
    before = deepcopy(data)
    renderer = CarlaTrajectoryRenderer(world(), {"enabled": True})
    audit = renderer.draw(data, None, capture(), True)
    assert audit["drawn_candidates"] == 1
    assert audit["merged_candidates"] == 2 and audit["hidden_candidates"] == 0
    group = audit["candidate_groups"][0]
    assert group["representative_id"] == "center:policy2"
    assert set(group["member_indices"]) == {0, 1, 2}
    assert group["selected"] and not audit["xy_geometry_displaced"]
    assert data == before


def test_shared_prefix_different_tail_and_other_road_deck_are_not_merged():
    a = candidate(0, [[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]])
    b = candidate(1, [[0., 0., 0.], [10., 0., 0.], [20., 3., 0.]])
    c = candidate(2, [[0., 0., 1.], [10., 0., 1.], [20., 0., 1.]])
    short = candidate(3, [[0., 0., 0.], [5., 0., 0.]])
    config = validate_trajectory_visualization_config({"max_candidates": 0})
    shown, groups, invalid = _display_groups([a, b, c, short], 0, config)
    assert len(shown) == len(groups) == 4 and not invalid


def test_short_divergence_between_uniform_samples_is_not_lost_by_merging():
    a = candidate(0, [[0., 0., 0.], [100., 0., 0.]])
    b = candidate(1, [[0., 0., 0.], [41., 0., 0.], [42., .2, 0.],
                      [43., 0., 0.], [100., 0., 0.]])
    _, groups, _ = _display_groups([a, b], 0, validate_trajectory_visualization_config())
    assert len(groups) == 2


def test_budget_covers_lateral_families_and_keeps_outside_index_winner():
    candidates = [candidate(i, [[0., 0., 0.], [20. + i, 0., 0.]]) for i in range(15)]
    candidates += [candidate(20, [[0., 0., 0.], [25., 3., 0.]], "left"),
                   candidate(21, [[0., 0., 0.], [25., -3., 0.]], "right")]
    audit = CarlaTrajectoryRenderer(world(), {"enabled": True, "max_candidates": 3}).draw(
        pool(candidates, selected=14), None, capture(), True)
    assert audit["drawn_candidates"] == 3 and audit["hidden_candidates"] == 14
    assert {g["family"] for g in audit["candidate_groups"]} == {"left", "right", "center"}
    assert any(g["selected"] and g["representative_id"] == "center:policy14"
               for g in audit["candidate_groups"])


def test_no_selected_index_never_highlights_a_missing_index_as_winner():
    item = candidate(0)
    item.pop("index")
    audit = CarlaTrajectoryRenderer(world(), {"enabled": True}).draw(pool([item], selected=None), None, capture(), True)
    assert not any(g["selected"] for g in audit["candidate_groups"])


def test_infeasible_iterate_cannot_merge_into_a_feasible_display_group():
    a, b = candidate(0, optimization_feasible=True), candidate(1, optimization_feasible=False)
    target = world()
    audit = CarlaTrajectoryRenderer(target, {"enabled": True}).draw(pool([a, b]), None, capture(), True)
    assert audit["merged_candidates"] == 0
    failed = next(g for g in audit["candidate_groups"] if g["member_indices"] == [1])
    assert not failed["display_feasible"] and tuple(failed["rgb"]) == PAPER_COLORS["infeasible"]


def test_metric_dashes_and_outline_preserve_xyz_and_inputs():
    target = world()
    points = [[10., 20., 0.], [10., 21.3, .13], [10., 23.8, .38], [10., 30., 1.]]
    data = pool([candidate(0, points)], selected=None)
    before = deepcopy(data)
    config = {"enabled": True, "endpoint_markers": False, "dash_length_m": 1., "dash_gap_m": 1.}
    renderer = CarlaTrajectoryRenderer(target, config)
    audit = renderer.draw(data, None, capture(), True)
    assert audit["drawn_candidates"] == 1 and len(target.calls) > 4
    cores = [call for call in target.calls if color(call) != PAPER_COLORS["outline"]]
    assert cores and len(cores) * 2 == len(target.calls)
    intervals = []
    for args, kwargs in cores:
        a, b = args
        assert a.x == b.x == 10.
        # Python calculations are float64; CARLA Location fields use float32.
        assert a.z - renderer.config["z_offset"] - .006 == pytest.approx((a.y - 20.) * .1, abs=2e-6)
        assert b.z - renderer.config["z_offset"] - .006 == pytest.approx((b.y - 20.) * .1, abs=2e-6)
        assert np.linalg.norm([b.x - a.x, b.y - a.y, b.z - a.z]) <= 1. + 2e-6
        assert not kwargs["persistent_lines"] and kwargs["life_time"] == .2
        intervals.append((a.y, b.y))
    assert any(b[0] - a[1] > .5 for a, b in zip(intervals, intervals[1:]))
    assert data == before


@pytest.mark.parametrize("accepted,kind", [(True, "execution"), (False, "rejected")])
def test_selected_outer_stroke_and_command_inner_stroke_remain_distinguishable(accepted, kind):
    target = world()
    data = pool([candidate(0, [[10., 20., 0.], [10., 25., 0.]])])
    command = {"trajectory": [{"x": 0., "y": 0.}, {"x": 5., "y": 0.}]}
    before = deepcopy(command)
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True, "endpoint_markers": False})
    audit = renderer.draw(data, command, capture(), accepted)
    selected = [c for c in target.calls if color(c) == PAPER_COLORS["selected"]]
    executed = [c for c in target.calls if color(c) == PAPER_COLORS[kind]]
    assert selected and executed
    # The inner command casing is narrower than the visible gold plan.
    assert selected[0][1]["thickness"] > executed[0][1]["thickness"] + renderer.config["outline_width"] * 2
    assert audit[kind + "_drawn"] and command == before
    assert not audit["execution_drawn"] if not accepted else not audit["rejected_drawn"]


def test_stationary_plan_is_a_local_glyph_not_an_invented_forward_path():
    target = world()
    command = {"trajectory": [{"x": 0., "y": 0.}] * 3}
    audit = CarlaTrajectoryRenderer(target, {"enabled": True}).draw(None, command, capture(), True)
    assert audit["execution_drawn"]
    for args, _ in target.calls:
        for point in args:
            assert abs(point.x - 10.) <= .35 and abs(point.y - 20.) <= .35


def test_one_invalid_candidate_does_not_hide_the_valid_pool():
    data = pool([candidate(0), candidate(1, [[0., 0., 0.], [float("nan"), 1., 0.]])])
    audit = CarlaTrajectoryRenderer(world(), {"enabled": True}).draw(data, None, capture(), True)
    assert audit["drawn_candidates"] == 1 and audit["invalid_candidate_ids"] == ["center:policy1"]
    assert "error" not in audit


@pytest.mark.parametrize("name", ["dash_length_m", "dash_gap_m"])
def test_metric_dashes_have_a_finite_rendering_resolution(name):
    with pytest.raises(ValueError, match="carla.visualization.trajectories"):
        validate_trajectory_visualization_config({name: .049})
    assert validate_trajectory_visualization_config({name: .05})[name] == .05


@pytest.mark.parametrize("bad_candidate", [None, 7, "invalid candidate"])
def test_malformed_candidate_record_does_not_hide_valid_candidates_or_command(bad_candidate):
    target = world()
    data = pool([candidate(0), bad_candidate])
    before = deepcopy(data)
    command = {"trajectory": [{"x": 0., "y": 0.}, {"x": 5., "y": 0.}]}
    audit = CarlaTrajectoryRenderer(target, {"enabled": True}).draw(data, command, capture(), True)
    assert audit["drawn_candidates"] == 1
    assert audit["invalid_candidate_ids"] == [None]
    assert audit["execution_drawn"] and "error" not in audit
    assert data == before


def test_one_excessive_candidate_is_skipped_before_rpc_without_hiding_command():
    target = world()
    valid = candidate(0)
    excessive = candidate(1, [[0., 0., 0.], [10000., 0., 0.]])
    command = {"trajectory": [{"x": 0., "y": 0.}, {"x": 5., "y": 0.}]}
    audit = CarlaTrajectoryRenderer(target, {"enabled": True}).draw(
        pool([valid, excessive]), command, capture(), True)
    assert audit["drawn_candidates"] == 1
    assert audit["execution_drawn"] and "error" not in audit
    errors = audit["skipped_candidate_errors"]
    assert len(errors) == 1 and excessive["id"] in str(errors) and "ValueError" in str(errors)
    # The skipped long candidate cannot send a partial massive batch first.
    assert all(abs(point.x) < 100. and abs(point.y) < 100.
               for args, _ in target.calls for point in args)


@pytest.mark.parametrize("dashed,points", [
    (True, [[0., 0., 0.], [1e9, 0., 0.]]),
    (False, np.column_stack((np.linspace(0., 10., 5001), np.zeros(5001), np.zeros(5001))).tolist()),
])
def test_single_stroke_primitive_guard_precedes_any_rpc(dashed, points):
    target = world()
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True})
    with pytest.raises(ValueError):
        renderer._stroke(points, PAPER_COLORS["execution"], .1, .15, dashed=dashed)
    assert not target.calls


def test_excessive_rejected_command_is_a_rendering_error_without_emitting_a_huge_batch():
    target = world()
    points = [{"x": float(x), "y": 0.} for x in np.linspace(0., 10., 5001)]
    command = {"trajectory": points}
    before = deepcopy(command)
    audit = CarlaTrajectoryRenderer(target, {"enabled": True}).draw(None, command, capture(), False)
    assert audit["error"]["type"] == "ValueError"
    assert not audit["execution_drawn"] and not audit["rejected_drawn"]
    assert not target.calls and command == before


def test_default_large_pool_has_a_bounded_display_budget_without_reordering_source():
    target = world()
    x = np.linspace(0., 40., 41)
    data = pool([candidate(i, np.column_stack((x, np.sin(x / 40.) * i * .04,
                                              np.zeros(len(x)))).tolist(),
                           "left" if i % 2 else "right") for i in range(160)], selected=159)
    before = deepcopy(data)
    audit = CarlaTrajectoryRenderer(target, {"enabled": True}).draw(data, None, capture(), True)
    assert "error" not in audit
    assert audit["drawn_candidates"] <= 10 and len(target.calls) < 2000
    assert any(group["selected"] and 159 in group["member_indices"]
               for group in audit["candidate_groups"])
    assert data == before
