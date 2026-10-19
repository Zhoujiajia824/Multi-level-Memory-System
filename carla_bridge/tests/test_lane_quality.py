"""Lane centering is a bounded cost on native rollouts, never a safety veto."""
from importlib.util import find_spec

import numpy as np
import pytest

from carla_bridge.planning.lane_quality import (
    LaneQualityReference, footprint_lane_quality, lane_surface_union, validate_lane_quality_config)


def straight_footprints(offset, *, times=41, yaw=0., width=1.8):
    centers = np.column_stack((np.linspace(0., 20., times), np.full(times, offset)))
    local = np.array([[2.3, width / 2], [2.3, -width / 2],
                      [-2.3, -width / 2], [-2.3, width / 2]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return centers[:, None, :] + (local @ rotation.T)[None, :, :]


def cost(corners, **kwargs):
    reference = LaneQualityReference([[-10., 0.], [100., 0.]], 3.5)
    return footprint_lane_quality(corners, reference, 1.8, .1, **kwargs)


def test_center_free_band_does_not_charge_small_tracking_or_curvature_compensation():
    for offset in (0., .05, .149):
        value, audit = cost(straight_footprints(offset))
        assert value == 0.
        assert audit["center_error_mean_m"] == pytest.approx(abs(offset))
        assert audit["actual_simulated_footprint"]
        assert not audit["hard_veto_applied"]


def test_simulated_body_clearance_and_center_quality_distinguish_offsets():
    values = []
    for offset in (0., .25, .5, .8, 1.):
        value, audit = cost(straight_footprints(offset))
        values.append(value)
        assert audit["minimum_footprint_clearance_m"] == pytest.approx(1.75 - .9 - offset)
        assert 0. <= value <= .16
    assert values == sorted(values)
    assert values[0] == 0. and values[-1] == pytest.approx(.16)


def test_yawed_footprint_crosses_boundary_even_when_its_center_is_near_the_middle():
    value, audit = cost(straight_footprints(.1, yaw=np.deg2rad(30.)))
    assert audit["center_cost"] == 0.
    assert audit["boundary_cost"] > .1
    assert value > 0.
    assert audit["footprint_boundary_crossing_fraction"] == 1.
    assert audit["minimum_footprint_clearance_m"] < 0.


def test_stationary_and_moving_footprints_pay_identical_cost_at_the_same_offset():
    moving = straight_footprints(.6)
    stopped = np.repeat(moving[:1], len(moving), axis=0)
    stopped_cost, stopped_audit = cost(stopped)
    moving_cost, moving_audit = cost(moving)
    assert stopped_cost == pytest.approx(moving_cost)
    for key in ("center_error_mean_m", "center_error_max_m", "minimum_footprint_clearance_m"):
        assert stopped_audit[key] == pytest.approx(moving_audit[key])


def test_initial_unavoidable_state_is_not_repeated_as_a_future_penalty():
    centered = straight_footprints(0.)
    shifted_initial = centered.copy()
    shifted_initial[0, :, 1] += 2.
    assert cost(centered) == cost(shifted_initial)


def test_numeric_maneuver_reference_does_not_penalize_legal_lane_transition():
    points = np.column_stack((np.linspace(-10., 40., 501),
                              np.linspace(-10., 40., 501) * .1))
    reference = LaneQualityReference(points, 3.5, kind="legal_maneuver")
    yaw = np.arctan(.1)
    footprints = straight_footprints(0., yaw=yaw)
    footprints[:, :, 1] += np.linspace(0., 2., len(footprints))[:, None]
    value, audit = footprint_lane_quality(footprints, reference, 1.8, .1)
    assert value == 0.
    assert audit["reference_kind"] == "legal_maneuver"
    # Comparing this valid maneuver to the old route would incorrectly punish
    # the intended transition; the candidate must carry its own numerical ref.
    assert cost(footprints)[0] > 0.


def test_legal_transition_boundary_is_the_actual_two_lane_union_not_a_moving_tube():
    from shapely.geometry import box
    # At this instant the body is in the source lane while the numerical
    # transition reference has moved toward the shared dashed marking.
    footprints = straight_footprints(.2)
    reference = LaneQualityReference([[-10., 1.2], [100., 1.2]], 3.5, kind="legal_maneuver")
    source_target_union = box(-20., -1.75, 120., 5.25)
    _, moving_tube = footprint_lane_quality(footprints, reference, 1.8, .1)
    _, actual_union = footprint_lane_quality(footprints, reference, 1.8, .1,
                                           boundary_polygon=source_target_union)
    assert actual_union["center_cost"] == moving_tube["center_cost"]
    assert moving_tube["boundary_cost"] > 0.
    assert actual_union["boundary_cost"] == 0.
    assert actual_union["boundary_reference_kind"] == "actual_source_target_lane_union"
    assert actual_union["minimum_footprint_clearance_m"] > moving_tube["minimum_footprint_clearance_m"]


def test_actual_union_still_charges_crossing_its_real_outside_boundary():
    from shapely.geometry import box
    reference = LaneQualityReference([[-10., 3.5], [100., 3.5]], 3.5, kind="legal_maneuver")
    _, audit = footprint_lane_quality(straight_footprints(5.), reference, 1.8, .1,
                                     boundary_polygon=box(-20., -1.75, 120., 5.25))
    assert audit["boundary_cost"] == pytest.approx(.12)
    assert audit["minimum_footprint_clearance_m"] < 0.


def test_float32_seam_between_legal_adjacent_lanes_is_not_an_outside_boundary():
    from shapely.geometry import box
    from shapely.ops import unary_union
    # A predicted corner lands on the nominal shared marking; CARLA's two
    # independent float32 lane ribbons leave a 0.01-mm sampling seam there.
    pieces = [box(-20., -1.75, 120., 1.75), box(-20., 1.75001, 120., 5.25001)]
    reference = LaneQualityReference([[-10., .85], [100., .85]], 3.5, kind="legal_maneuver")
    footprints = straight_footprints(.85)
    _, raw = footprint_lane_quality(footprints, reference, 1.8, .1,
                                    boundary_polygon=unary_union(pieces))
    _, corrected = footprint_lane_quality(footprints, reference, 1.8, .1,
                                          boundary_polygon=lane_surface_union(pieces))
    assert raw["boundary_cost"] > .1
    assert corrected["boundary_cost"] == 0.
    assert corrected["minimum_footprint_clearance_m"] > 1.
    # Closing a numerical shared seam must retain the actual outside edge.
    _, outside = footprint_lane_quality(straight_footprints(5.), reference, 1.8, .1,
                                        boundary_polygon=lane_surface_union(pieces))
    assert outside["boundary_cost"] == pytest.approx(.12)
    assert outside["minimum_footprint_clearance_m"] < -.6


def test_lane_union_does_not_fill_a_real_gap_or_expand_outer_lane_boundary():
    from shapely.geometry import box, Point
    corrected = lane_surface_union([box(-20., -1.75, 120., 1.75),
                                    box(-20., 1.85, 120., 5.35)])
    assert not corrected.covers(Point(20., 1.80))
    assert not corrected.covers(Point(20., -1.751))


def test_disabling_only_removes_cost_and_does_not_change_footprints():
    footprints = straight_footprints(1.)
    copy = footprints.copy()
    value, audit = cost(footprints, config={"enabled": False})
    assert value == 0. and not audit["enabled"]
    np.testing.assert_array_equal(copy, footprints)


@pytest.mark.parametrize("config", [dict(enabled=1), dict(center_weight=.17),
    dict(boundary_weight=.17), dict(center_weight=.1, boundary_weight=.1),
    dict(horizon_s=.1), dict(horizon_s=float("nan")), dict(boundary_margin_m=0.),
    dict(center_tolerance_m=-.1), dict(unknown=1)])
def test_invalid_cost_configuration_is_rejected(config):
    with pytest.raises(ValueError, match="lane_quality"):
        validate_lane_quality_config(config)


@pytest.mark.parametrize("points,widths", [([], 3.5), ([[0., 0.]], 3.5),
    ([[0., 0.], [0., 0.]], 3.5), ([[0., 0.], [1., float("nan")]], 3.5),
    ([[0., 0.], [1., 0.]], 0.), ([[0., 0.], [1., 0.]], [3.5])])
def test_invalid_reference_does_not_silently_relabel_a_vehicle_as_centered(points, widths):
    with pytest.raises(ValueError, match="lane quality"):
        LaneQualityReference(points, widths)


optional_pdm = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                 reason="optional pinned PDM dependencies are not installed")


@optional_pdm
def test_native_proposals_evaluate_actual_corners_and_do_not_change_upstream_scores():
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    from carla_bridge.tests.test_pdm_planner import make_planner
    fixture = make_planner(speed=5.)
    enabled = fixture.plan(None, semantic_enabled=False)["planner_diagnostics"]
    disabled = CarlaPDMPlanner(fixture.world, fixture.ego, fixture.route,
        {"lane_quality": {"enabled": False}}).plan(None, semantic_enabled=False)["planner_diagnostics"]
    on = {item["id"]: item for item in enabled["candidates"]}
    off = {item["id"]: item for item in disabled["candidates"]}
    assert on.keys() == off.keys()
    for identity, item in on.items():
        assert item["base_score"] == off[identity]["base_score"]
        assert item["trajectory_sha256"] == off[identity]["trajectory_sha256"]
        assert off[identity]["numerical_cost"] == 0.
        assert item["lane_quality"]["actual_simulated_footprint"]
        assert 0. <= item["numerical_cost"] <= .16
    assert on["center:ACCELERATE"]["numerical_cost"] == 0.
    for item in on.values():
        audit = item["lane_quality"]
        # A label alone never incurs cost: the simulated body must actually
        # depart from the center band or approach the corridor boundary.
        if audit["center_error_max_m"] <= .15 and audit["minimum_footprint_clearance_m"] >= .15:
            assert item["numerical_cost"] == pytest.approx(audit["lane_change_start_cost"], abs=1e-20)
    # No global penalty for being in a different real lane during a legal change.
    assert on["left:ACCELERATE"]["lane_quality"]["reference_kind"] == "legal_maneuver"


@optional_pdm
def test_each_native_stopped_option_pays_the_same_measured_position_cost():
    from carla_bridge.tests.test_pdm_planner import make_planner
    fixture = make_planner(speed=0.)
    fixture.ego.y = -.5
    diagnostics = fixture.plan(None, semantic_enabled=False)["planner_diagnostics"]
    stopped = [item for item in diagnostics["candidates"]
               if item["longitudinal_policy_id"] == "STOP"]
    same_lane = [item for item in stopped if item["physical_lane_side"] == "center"]
    assert len(same_lane) == 5
    for item in same_lane:
        assert item["numerical_cost"] == pytest.approx(same_lane[0]["numerical_cost"])
        assert item["numerical_cost"] > 0.


from carla_bridge.tests.test_recovery_planner_integration import road_snapshot


@optional_pdm
def test_real_town10hd_shared_lane_marking_has_no_artificial_micrometre_boundary(road_snapshot):
    import carla
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from carla_bridge.planning.road_area import road_driving_polygons
    road, _ = road_snapshot
    seeds = []
    for x in range(40, 66):
        source = road.get_waypoint(carla.Location(x=x, y=-60.92))
        target = source.get_right_lane()
        assert (source.road_id, source.lane_id, target.lane_id) == (10, 1, 2)
        seeds.extend((source, target))
    ribbons, _ = road_driving_polygons(road, seeds)
    pieces = [ribbons["10:0:1"], ribbons["10:0:2"]]
    raw, corrected = unary_union(pieces), lane_surface_union(pieces)
    source = road.get_waypoint(carla.Location(x=50., y=-60.92))
    a, b = source.transform.location, source.get_right_lane().transform.location
    shared = Point((a.x + b.x) / 2, -(a.y + b.y) / 2)
    assert raw.boundary.distance(shared) < .0001
    assert corrected.boundary.distance(shared) > 3.499
    assert abs(corrected.area - raw.area) < .001


@optional_pdm
def test_real_town10hd_recovery_remains_positive_and_shadow_remains_exact(road_snapshot):
    from carla_bridge.tests.test_recovery_planner_integration import (
        make_snapshot_planner, observe_stopped_native, recovery_advice)
    for policy in ("selective", "shadow"):
        planner, clock = make_snapshot_planner(road_snapshot, policy=policy)
        prior = observe_stopped_native(planner, clock)
        result = planner.plan(recovery_advice(), semantic_update=True, sim_time_s=clock.time)
        diagnostics = result["planner_diagnostics"]
        assert diagnostics["recovery"]["positive_recovery_candidates"] > 0
        if policy == "selective":
            selected = next(item for item in diagnostics["candidates"] if item["id"] == diagnostics["selected_id"])
            assert selected["recovery"] and selected["base_score"] > 0.
            assert selected["planning_score"] > 0.
            assert max(point["optional_v"] for point in result["trajectory"]) > .2
        else:
            assert result["trajectory"] == prior["trajectory"]
            assert diagnostics["candidate_pool_sha256"] == prior["planner_diagnostics"]["candidate_pool_sha256"]
            assert diagnostics["native_tracking"] == prior["planner_diagnostics"]["native_tracking"]


@optional_pdm
def test_measured_town10hd_curve_prefers_less_body_boundary_error_without_losing_progress(road_snapshot):
    from copy import deepcopy
    import json
    import math
    from pathlib import Path
    import carla
    from carla_bridge.tests.test_recovery_planner_integration import make_snapshot_planner
    observed = json.loads((Path(__file__).parent / "fixtures" /
                           "town10hd_curve_tracking.json").read_text(encoding="utf-8"))
    road, original = road_snapshot
    assert observed["map_sha256"] == original["map_sha256"]
    results = []
    for enabled in (False, True):
        snapshot = deepcopy(original)
        snapshot["pose"] = observed["pose"]
        snapshot["route_progress"].update(observed["route_progress"])
        snapshot["planning"].update(lqr_lateral_preview="speed_scaled", lane_quality={"enabled": enabled})
        planner, clock = make_snapshot_planner((road, snapshot))
        yaw, speed = observed["pose"]["yaw_rad"], observed["speed_mps"]
        planner.ego.get_velocity = lambda: carla.Vector3D(x=speed * math.cos(yaw), y=speed * math.sin(yaw))
        planner.ego.get_acceleration = lambda: carla.Vector3D(*observed["acceleration_xyz"])
        planner.ego.get_angular_velocity = lambda: carla.Vector3D(z=observed["yaw_rate_deg_s"])
        planner.ego.get_control = lambda: carla.VehicleControl(steer=observed["normalized_steer"])
        planner.ego.get_wheel_steer_angle = lambda wheel: observed["wheel_steer_deg"][
            "FL_Wheel" if wheel == carla.VehicleWheelLocation.FL_Wheel else "FR_Wheel"]
        diagnostic = planner.plan(None, semantic_enabled=False, sim_time_s=0.)["planner_diagnostics"]
        results.append((diagnostic, next(item for item in diagnostic["candidates"]
                                         if item["id"] == diagnostic["selected_id"])))
    before, after = results
    old_winner = next(item for item in after[0]["candidates"] if item["id"] == before[1]["id"])
    assert after[1]["numerical_cost"] < old_winner["numerical_cost"]
    assert after[1]["lane_quality"]["center_error_mean_m"] < old_winner["lane_quality"]["center_error_mean_m"]
    assert after[1]["base_score"] > .99
    assert after[1]["forward_progress_m"] >= before[1]["forward_progress_m"] - .02
    assert before[0]["safety_gate_applied"] is after[0]["safety_gate_applied"] is False
    for item in before[0]["candidates"]:
        matched = next(candidate for candidate in after[0]["candidates"] if candidate["id"] == item["id"])
        assert matched["base_score"] == item["base_score"]
        assert matched["trajectory_sha256"] == item["trajectory_sha256"]
