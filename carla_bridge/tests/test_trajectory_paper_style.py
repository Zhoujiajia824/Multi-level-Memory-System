"""Solid camera-only paths preserve evaluated geometry and bounded display data."""
from copy import deepcopy

import numpy as np
import pytest

from carla_bridge.visualization.trajectory_renderer import (
    CarlaTrajectoryRenderer, DISPLAY_COLORS, DISPLAY_FAMILY_COLORS, _display_groups,
    validate_trajectory_visualization_config,
)
from carla_bridge.tests.test_trajectory_renderer import world, capture


def candidate(index, points=None, side="center", **metadata):
    return {"id": f"{side}:policy{index}", "index": index,
            "points_world": points if points is not None else [[10., 20., 0.], [10., 30., 0.]],
            "physical_lane_side": side, "base_score": 1. - index / 1000., **metadata}


def pool(candidates, selected=0):
    return {"selected_index": selected, "candidate_pool_count": len(candidates), "candidates": candidates}


def no_world_debug(audit, target):
    assert not target.calls
    assert audit["backend"] == "camera_overlay" and audit["world_drawn_segments"] == 0


@pytest.mark.parametrize("bad", [{"style": "unknown"}, {"endpoint_markers": "yes"},
                                  {"merge_distance_m": -1}, {"merge_distance_m": float("nan")},
                                  {"outline_width": -1}, {"candidate_alpha": -0.1},
                                  {"selected_alpha": 1.1}, {"execution_alpha": float("nan")},
                                  {"candidate_alpha": True}])
def test_paper_config_rejects_invalid_rendering_options(bad):
    with pytest.raises(ValueError, match="carla.visualization.trajectories"):
        validate_trajectory_visualization_config(bad)


def test_default_paper_budget_widths_palette_and_no_outline():
    config = validate_trajectory_visualization_config()
    assert config["style"] == "paper" and config["max_candidates"] == 6
    assert config["thickness"] == .045
    assert config["selected_thickness"] == .075 and config["execution_thickness"] == .12
    assert config["candidate_alpha"] == .6 and config["selected_alpha"] == .85
    assert config["execution_alpha"] == 1.
    assert config["outline_width"] == 0
    assert validate_trajectory_visualization_config({"outline_width": 0})["outline_width"] == 0
    assert DISPLAY_COLORS["selected"] == (25, 92, 173)
    assert DISPLAY_COLORS["execution"] == (222, 58, 55)
    assert DISPLAY_FAMILY_COLORS == {"center": (55, 136, 190), "left": (0, 158, 150),
                                     "right": (141, 107, 188)}


def test_candidate_color_is_stable_for_lateral_family_despite_id_and_policy_changes():
    # Distinct actual geometries prevent grouping from concealing a bad ID
    # palette. Two independent plan cycles must encode the same families.
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    family_colors = {}
    for cycle in range(2):
        items = [candidate(i + cycle*100, [[0., i, 0.], [10., i, 0.]], side)
                 for i, side in enumerate(("center", "left", "right", "left", "right"))]
        renderer.draw(pool(items, selected=None), None, capture(), True)
        for shown in renderer.last_display_snapshot["paths"]:
            side = shown["id"].split(":")[0]
            rgb = tuple(shown["rgb"])
            assert rgb == DISPLAY_FAMILY_COLORS[side]
            assert rgb == family_colors.setdefault(side, rgb)
            assert shown["alpha"] == .6 and shown["width_m"] == .045
    assert len(set(family_colors.values())) == 3


def test_near_duplicate_group_preserves_winner_member_ids_and_has_no_debug_primitives():
    data = pool([candidate(0), candidate(1), candidate(2)], selected=2)
    before, target = deepcopy(data), world()
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True})
    audit = renderer.draw(data, None, capture(), True)
    assert audit["drawn_candidates"] == 1
    assert audit["merged_candidates"] == 2 and audit["hidden_candidates"] == 0
    group = audit["candidate_groups"][0]
    assert group["representative_id"] == "center:policy2"
    assert set(group["member_indices"]) == {0, 1, 2}
    assert group["selected"] and not audit["xy_geometry_displaced"]
    assert data == before
    no_world_debug(audit, target)


def test_shared_prefix_different_tail_and_other_road_deck_are_not_merged():
    a = candidate(0, [[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]])
    b = candidate(1, [[0., 0., 0.], [10., 0., 0.], [20., 3., 0.]])
    c = candidate(2, [[0., 0., 1.], [10., 0., 1.], [20., 0., 1.]])
    short = candidate(3, [[0., 0., 0.], [5., 0., 0.]])
    shown, groups, invalid = _display_groups([a, b, c, short], 0,
        validate_trajectory_visualization_config({"max_candidates": 0}))
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
    target = world()
    audit = CarlaTrajectoryRenderer(target, {"enabled": True, "max_candidates": 3}).draw(
        pool(candidates, selected=14), None, capture(), True)
    assert audit["drawn_candidates"] == 3 and audit["hidden_candidates"] == 14
    assert {g["family"] for g in audit["candidate_groups"]} == {"left", "right", "center"}
    assert any(g["selected"] and g["representative_id"] == "center:policy14"
               for g in audit["candidate_groups"])
    no_world_debug(audit, target)


def test_no_selected_index_never_highlights_a_missing_index_as_winner():
    item = candidate(0)
    item.pop("index")
    audit = CarlaTrajectoryRenderer(None, {"enabled": True}).draw(pool([item], selected=None), None, capture(), True)
    assert not any(g["selected"] for g in audit["candidate_groups"])


def test_infeasible_iterate_cannot_merge_into_feasible_display_group():
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    audit = renderer.draw(pool([candidate(0, optimization_feasible=True),
                                candidate(1, optimization_feasible=False)]), None, capture(), True)
    assert audit["merged_candidates"] == 0
    failed = next(g for g in audit["candidate_groups"] if g["member_indices"] == [1])
    assert not failed["display_feasible"]
    shown = next(p for p in renderer.last_display_snapshot["paths"] if p.get("id") == "center:policy1")
    assert tuple(shown["rgb"]) == DISPLAY_COLORS["infeasible"]


def test_paper_publishes_every_real_bend_with_only_common_surface_lift():
    target = world()
    points = [[10., 20., 0.], [10., 21.3, .13], [10.2, 23.8, .38], [10., 30., 1.]]
    data, config = pool([candidate(0, points)], selected=None), {
        "enabled": True, "point_stride": 99, "endpoint_markers": True,
        "dash_length_m": 1., "dash_gap_m": 1., "z_offset": .15}
    before = deepcopy(data)
    renderer = CarlaTrajectoryRenderer(target, config)
    audit = renderer.draw(data, None, capture(), True)
    shown = renderer.last_display_snapshot["paths"]
    assert len(shown) == 1 and audit["drawn_segments"] == len(points)-1
    expected = np.asarray(points, dtype=float)
    expected[:, 2] += .15
    np.testing.assert_allclose(shown[0]["points_world"], expected, rtol=0, atol=1e-12)
    assert shown[0]["dashed"] is False and shown[0]["marker"] is False
    assert renderer.last_display_snapshot["endpoint_markers"] is False
    assert data == before
    no_world_debug(audit, target)


@pytest.mark.parametrize("accepted,kind", [(True, "execution"), (False, "rejected")])
def test_selected_and_command_keep_distinct_width_color_but_share_surface_height(accepted, kind):
    target = world()
    data = pool([candidate(0, [[10., 20., 0.], [10., 25., 0.]])])
    data["reference_surface_xyz"] = [[10., 20., 0.], [10., 25., 0.]]
    command = {"trajectory": [{"x": 0., "y": 0.}, {"x": 5., "y": 0.}]}
    before = deepcopy(command)
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True, "endpoint_markers": True})
    audit = renderer.draw(data, command, capture(), accepted)
    selected = next(p for p in renderer.last_display_snapshot["paths"] if p["kind"] == "selected")
    executed = next(p for p in renderer.last_display_snapshot["paths"] if p["kind"] == kind)
    assert selected["width_m"] == .075 and executed["width_m"] == .12
    assert selected["width_m"] < executed["width_m"]
    assert selected["alpha"] == .85 and executed["alpha"] == 1.
    assert tuple(selected["rgb"]) == DISPLAY_COLORS["selected"]
    assert tuple(executed["rgb"]) == DISPLAY_COLORS[kind]
    np.testing.assert_allclose(selected["points_world"], executed["points_world"], atol=1e-6)
    assert np.asarray(selected["points_world"])[:, 2] == pytest.approx([.15, .15])
    assert all(p["dashed"] is False and p["marker"] is False for p in (selected, executed))
    assert audit[kind + "_drawn"] and command == before
    assert not audit["rejected_drawn"] if accepted else not audit["execution_drawn"]
    no_world_debug(audit, target)


def test_stationary_paths_are_audited_without_crosses_arrows_or_invented_segments():
    target = world()
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True, "endpoint_markers": True})
    command = {"trajectory": [{"x": 0., "y": 0.}] * 3}
    audit = renderer.draw(pool([candidate(0, [[10., 20., 1.]]*3)]), command, capture(), True)
    assert audit["drawn_segments"] == 0 and audit["drawn_candidates"] == 0
    assert not audit["execution_drawn"]
    assert any(p.get("id") == "center:policy0" and p["kind"] == "selected" for p in audit["stationary_paths"])
    assert any(p["kind"] == "execution" for p in audit["stationary_paths"])
    assert not any(len(p["points_world"]) > 1 for p in renderer.last_display_snapshot["paths"])
    no_world_debug(audit, target)


def test_one_invalid_candidate_does_not_hide_the_valid_pool():
    data = pool([candidate(0), candidate(1, [[0., 0., 0.], [float("nan"), 1., 0.]])])
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    audit = renderer.draw(data, None, capture(), True)
    assert audit["drawn_candidates"] == 1 and audit["invalid_candidate_ids"] == ["center:policy1"]
    assert "error" not in audit and len(renderer.last_display_snapshot["paths"]) == 1


@pytest.mark.parametrize("bad_candidate", [None, 7, "invalid candidate"])
def test_malformed_candidate_record_does_not_hide_valid_candidates_or_command(bad_candidate):
    target = world()
    data = pool([candidate(0), bad_candidate])
    before = deepcopy(data)
    command = {"trajectory": [{"x": 0., "y": 0.}, {"x": 5., "y": 0.}]}
    audit = CarlaTrajectoryRenderer(target, {"enabled": True}).draw(data, command, capture(), True)
    assert audit["drawn_candidates"] == 1 and audit["invalid_candidate_ids"] == [None]
    assert audit["execution_drawn"] and "error" not in audit
    assert data == before
    no_world_debug(audit, target)


def test_huge_two_point_path_is_one_solid_segment_not_a_metric_dash_budget_error():
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    points = [[0., 0., 0.], [1e9, 0., 0.]]
    audit = renderer.draw(pool([candidate(0, points)]), None, capture(), True)
    assert "error" not in audit and not audit["skipped_candidate_errors"]
    assert audit["drawn_candidates"] == 1 and audit["drawn_segments"] == 1
    np.testing.assert_array_equal(np.asarray(renderer.last_display_snapshot["paths"][0]["points_world"])[:, :2], np.asarray(points)[:, :2])


def test_excessive_actual_candidate_segments_are_isolated_without_partial_publication():
    excessive = candidate(1, np.column_stack((np.linspace(0., 10., 5002), np.zeros(5002), np.zeros(5002))).tolist())
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    command = {"trajectory": [{"x": 0., "y": 0.}, {"x": 5., "y": 0.}]}
    audit = renderer.draw(pool([candidate(0), excessive]), command, capture(), True)
    assert "error" not in audit and audit["drawn_candidates"] == 1 and audit["execution_drawn"]
    assert any(p["id"] == excessive["id"] for p in audit["skipped_candidate_errors"])
    assert all(p.get("id") != excessive["id"] for p in renderer.last_display_snapshot["paths"])


def test_exactly_4096_actual_segments_are_allowed():
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    points = np.column_stack((np.linspace(0., 10., 4097), np.zeros(4097), np.zeros(4097))).tolist()
    audit = renderer.draw(pool([candidate(0, points)]), None, capture(), True)
    assert "error" not in audit and audit["drawn_segments"] == 4096
    assert len(renderer.last_display_snapshot["paths"][0]["points_world"]) == 4097


@pytest.mark.parametrize("bad_command", [
    {"trajectory": [{"x": 0., "y": 0.}, {"x": 1.}]},
    {"trajectory": [{"x": float(x), "y": 0.} for x in np.linspace(0., 10., 5002)]},
])
def test_invalid_current_command_does_not_republish_previous_accepted_command(bad_command):
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    renderer.draw(pool([candidate(0)]), {"trajectory": [{"x": 0., "y": 0.}, {"x": 3., "y": 0.}]}, capture(), True)
    assert any(p["kind"] == "execution" for p in renderer.last_display_snapshot["paths"])
    before = deepcopy(bad_command)
    audit = renderer.draw(pool([candidate(1)]), bad_command, capture(), True)
    assert "command_error" in audit and "error" not in audit
    assert renderer.last_display_snapshot["paths"]
    assert all(p["kind"] in ("candidate", "selected") for p in renderer.last_display_snapshot["paths"])
    assert all(p["id"] == "center:policy1" for p in renderer.last_display_snapshot["paths"])
    assert not audit["execution_drawn"] and bad_command == before


def test_overall_snapshot_error_clears_previously_published_paths():
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    renderer.draw(pool([candidate(0)]), None, capture(), True)
    assert renderer.last_display_snapshot["paths"]
    audit = renderer.draw(["not a snapshot mapping"], None, capture(), True)
    assert "error" in audit and renderer.last_display_snapshot is None


def test_turning_visualization_off_clears_any_previous_display_snapshot():
    renderer = CarlaTrajectoryRenderer(None, {"enabled": True})
    renderer.draw(pool([candidate(0)]), None, capture(), True)
    assert renderer.last_display_snapshot is not None
    renderer.config["enabled"] = False
    audit = renderer.draw(None, None, None, False)
    assert not audit["enabled"] and renderer.last_display_snapshot is None


def test_default_large_pool_has_six_group_budget_without_reordering_source():
    target = world()
    x = np.linspace(0., 40., 41)
    data = pool([candidate(i, np.column_stack((x, np.sin(x/40.)*i*.04,
                                              np.zeros(len(x)))).tolist(),
                           "left" if i % 2 else "right") for i in range(160)], selected=159)
    before = deepcopy(data)
    renderer = CarlaTrajectoryRenderer(target, {"enabled": True})
    audit = renderer.draw(data, None, capture(), True)
    assert "error" not in audit and audit["drawn_candidates"] <= 6
    assert len(renderer.last_display_snapshot["paths"]) <= 6
    assert any(g["selected"] and 159 in g["member_indices"] for g in audit["candidate_groups"])
    assert data == before
    no_world_debug(audit, target)
