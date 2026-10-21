"""Return-window regressions from R05 run 433499ab (2026-10-20).

These use production geometry/gap functions and actual observed dimensions;
the mixed-traffic values are portable constants, not private future schedules.
"""
from types import SimpleNamespace as NS

import carla
import numpy as np
import pytest

from carla_bridge.planning.lane_maneuver import LaneManeuver, _actor_data
from carla_bridge.tests.test_lane_maneuver import fixture, observed, Obstacle


def pedestrian(actor_id, x, speed=0.):
    actor = Obstacle(actor_id=actor_id, x=x, speed=speed)
    actor.type_id = "walker.pedestrian.0001"
    actor.bounding_box.extent = carla.Vector3D(x=.25, y=.25, z=.9)
    return actor


def passing_fixture():
    blockers = [pedestrian(1329, 28.)]
    engine, world, ego, reference = fixture(blockers)
    proposal = observed(engine, world, ego, reference)
    assert engine.commit_selected(proposal["metadata"]["id"] + ":MAINTAIN", 1.)
    # The real close-pedestrian maneuver used a 6 m transition and 1.4154 m/s.
    engine._active["length"] = 6.
    engine._active["speed_cap"] = 1.4154003695759112
    ego.x, ego.y, ego.speed = 44., -3.5, .1
    return engine, world, ego, reference, blockers


def test_frozen_r05_235_3_front_gap_uses_own_capped_return_speed():
    """Original gate required 24 m although this candidate cap was 1.4154."""
    engine, _, ego, reference = fixture()
    ego.speed = 1.4154003695759112
    ego.bounding_box.extent = carla.Vector3D(x=4.791779518127441/2,
                                            y=2.163450002670288/2)
    stations = np.arange(0., len(reference) * 2., 2.)
    station = 54.
    free_gap = 13.4796
    actor = dict(id=1363, type="walker.pedestrian.0001",
        xy=np.array([44. + free_gap + ego.bounding_box.extent.x + .3753577768802643/2, 0.]),
        velocity=np.array([-1.14, 0.]),
        station=station + free_gap + ego.bounding_box.extent.x + .3753577768802643/2,
        half_length=.3753577768802643/2, half_width=.3753577768802643/2,
        aligned=False, along_speed=-1.14)
    assert not engine._target_gap_clear([actor], reference, stations, station, ego,
        "center", ego.speed, candidate_speed_mps=8., returning=True)[0]
    clear, conflict = engine._target_gap_clear([actor], reference, stations, station, ego,
        "center", ego.speed, candidate_speed_mps=ego.speed, returning=True)
    assert clear and conflict is None


def test_return_blocker_rejection_is_visible_and_reserves_legal_wait_point():
    engine, world, ego, reference, actors = passing_fixture()
    actors.append(pedestrian(1363, 52., speed=-1.14))
    result = engine.update(world, ego, reference, 1.2, current_side="left", candidate_speed_mps=8.)
    assert result["phase"] == "passing"
    audit = result["diagnostics"]
    assert audit["conflict"]["reason"] == "return_currently_blocked"
    assert any(a.get("actor_id") == 1363 for a in audit["conflict"]["attempts"])
    assert audit["return_required"]
    assert audit["maneuver_stop_route_station_m"] < engine._active["corridor_end"] - 6.
    assert audit["maneuver_stop_reason"] == "preserve_legal_return_window"
    assert result["metadata"]["maneuver_stop_world_xy"] == audit["maneuver_stop_world_xy"]


def test_wait_for_oncoming_pedestrian_then_return_complete_after_actual_selection():
    engine, world, ego, reference, actors = passing_fixture()
    person = pedestrian(1363, 52., speed=-1.14)
    actors.append(person)
    waiting = engine.update(world, ego, reference, 1.2, current_side="left")
    assert waiting["phase"] == "passing"
    reserved = waiting["metadata"]["maneuver_stop_world_xy"]
    # Same actual waiting pose, measured pedestrian passes along its own lane.
    person.x = 36.
    ready = engine.update(world, ego, reference, 1.4, current_side="left", candidate_speed_mps=8.)
    assert ready["phase"] == "returning"
    assert ready["speed_cap_mps"] <= 1.4154003695759112
    assert "maneuver_stop_world_xy" not in ready["metadata"]
    assert ready["metadata"]["return_wait_world_xy"] == reserved
    assert engine._active["phase"] == "passing"  # generation cannot forge execution
    assert engine.commit_selected(ready["metadata"]["id"] + ":MAINTAIN", 1.4)
    ego.x, ego.y = 74., 0.
    for now in (1.6, 1.8):
        assert engine.update(world, ego, reference, now, current_side="center")["active"]
    completed = engine.update(world, ego, reference, 2., current_side="center")
    assert completed["diagnostics"]["reason"] == "completed"
    assert reserved[1] == pytest.approx(3.5)


def test_45_second_timeout_retains_return_obligation_and_fixed_wait_station():
    engine, world, ego, reference, actors = passing_fixture()
    actors.append(pedestrian(1363, 52., speed=-1.14))
    engine._active["committed_at"] = -45.
    first = engine.update(world, ego, reference, 1.2, current_side="left")
    assert first["active"] and first["diagnostics"]["maneuver_overdue"]
    assert engine._active is not None and first["metadata"]["return_required"]
    ego.x += .3
    second = engine.update(world, ego, reference, 1.4, current_side="left")
    assert second["metadata"]["maneuver_stop_route_station_m"] == first["metadata"]["maneuver_stop_route_station_m"]


def test_fast_rear_car_still_blocks_return_after_slow_pedestrian_clears():
    engine, world, ego, reference, actors = passing_fixture()
    actors.append(Obstacle(actor_id=1316, x=36., speed=10.))
    result = engine.update(world, ego, reference, 1.2, current_side="left")
    assert result["phase"] == "passing"
    assert any(a.get("reason") == "target_rear_gap" and a.get("actor_id") == 1316
               for a in result["diagnostics"]["conflict"]["attempts"])


def test_slow_receding_rear_pedestrian_does_not_need_vehicle_eight_metre_headway():
    engine, world, ego, reference, actors = passing_fixture()
    actors.append(pedestrian(1363, 38., speed=-1.14))
    result = engine.update(world, ego, reference, 1.2, current_side="left")
    assert result["phase"] == "returning", result["diagnostics"]


def test_wait_point_is_before_junction_and_return_crosses_no_solid_boundary():
    engine, world, ego, reference, actors = passing_fixture()
    plan = engine._active
    plan["corridor_end"] = 74.
    actors.append(pedestrian(1363, 52., speed=-1.14))
    result = engine.update(world, ego, reference, 1.2, current_side="left")
    stop_s = result["metadata"]["maneuver_stop_route_station_m"]
    assert stop_s + plan["length"] + 4. <= plan["corridor_end"]
    # There is still room to start a valid return at the reserved stop point.
    assert engine._corridor(engine._last_reference, engine._last_stations,
        stop_s, stop_s + plan["length"], "left", returning=True)[0]
    for wp in engine._last_reference:
        wp.is_junction = True
    result = engine.update(world, ego, reference, 1.4, current_side="left")
    assert result["phase"] == "passing"
    assert all(a["reason"] == "return_not_permitted"
               for a in result["diagnostics"]["conflict"]["attempts"])


def test_unselected_safe_return_keeps_anchor_even_when_ordinary_speed_is_eight():
    engine, world, ego, reference, _ = passing_fixture()
    first = engine.update(world, ego, reference, 1.2, current_side="left", candidate_speed_mps=8.)
    assert first["phase"] == "returning"
    assert not engine.commit_selected("left:ACCELERATE", 1.2)
    ego.x += .2
    second = engine.update(world, ego, reference, 1.4, current_side="left", candidate_speed_mps=8.)
    assert second["diagnostics"]["anchor_station_m"] == first["diagnostics"]["anchor_station_m"]
    np.testing.assert_array_equal(second["points"], first["points"])


def test_return_uses_shared_effective_walker_velocity_without_querying_private_future():
    person = pedestrian(1363, 52.)
    points = np.c_[np.arange(100.), np.zeros(100)]
    observation = NS(velocity=NS(x=-1.14, y=0.), source="finite_difference_walker")
    data = _actor_data(person, points, .2, observation)
    assert data["along_speed"] == pytest.approx(-1.14)
    assert data["motion_source"] == "finite_difference_walker"


@pytest.mark.parametrize("overdue", [False, True])
def test_unfinished_pass_can_safely_abort_behind_blocker_without_losing_return_window(overdue):
    engine, world, ego, reference, actors = passing_fixture()
    # A slow actor has moved ahead while ego was delayed in the passing lane.
    # At the last legal start it is better to return behind it than get stuck.
    actors[0].x = 63.
    actors[0].speed = .6
    # This fixture has 40-degree steering; unlike the real Tesla's 70-degree
    # inner-wheel limit it needs a longer physically reachable transition.
    engine._active["length"] = 20.
    engine._active["corridor_end"] = 78.
    if overdue:
        engine._active["committed_at"] = -45.
    # stop station=54, ego route station=54 (reference begins at x=-10).
    result = engine.update(world, ego, reference, 1.2, current_side="left")
    assert result["phase"] == "returning", result["diagnostics"]
    assert result["diagnostics"]["return_trigger"] == (
        "overdue_safe_return" if overdue else "abort_pass_before_deadline")
    assert engine._active["phase"] == "passing"
    assert engine.commit_selected(result["metadata"]["id"] + ":FOLLOW", 1.2)
    assert engine._active["phase"] == "returning"


def test_unexecuted_return_expires_only_after_actual_motion_invalidates_geometry():
    engine, world, ego, reference, _ = passing_fixture()
    ego.speed = 1.
    first = engine.update(world, ego, reference, 1.2, current_side="left")
    assert first["phase"] == "returning"
    old_anchor = first["diagnostics"]["anchor_station_m"]
    fixed_stop = first["metadata"]["return_wait_route_station_m"]
    invalidated = None
    # Replanning alone cannot move an anchor. Ego physically continues at
    # 1 m/s for 12 s on the side lane while this proposal never wins.
    for step in range(1, 61):
        ego.x += .2
        result = engine.update(world, ego, reference, 1.2 + step*.2, current_side="left")
        if result["diagnostics"].get("pending_return_invalidation"):
            invalidated = result
            break
    assert invalidated is not None
    audit = invalidated["diagnostics"]["pending_return_invalidation"]
    assert audit["old_anchor_station_m"] == old_anchor
    assert audit["new_anchor_station_m"] > old_anchor
    assert audit["nominal_lateral_mismatch_m"] > .45
    assert not audit["executed_return_reanchored"]
    assert invalidated["metadata"]["return_wait_route_station_m"] == fixed_stop
    assert engine._active["phase"] == "passing"


def test_stationary_wait_does_not_expire_a_return_or_move_its_anchor():
    engine, world, ego, reference, _ = passing_fixture()
    first = engine.update(world, ego, reference, 1.2, current_side="left")
    for step in range(1, 36):
        result = engine.update(world, ego, reference, 1.2 + step*.2, current_side="left")
    assert result["diagnostics"]["anchor_station_m"] == first["diagnostics"]["anchor_station_m"]
    assert "pending_return_invalidation" not in result["diagnostics"]


def test_already_executed_return_is_never_reanchored_by_pose_deviation():
    engine, world, ego, reference, _ = passing_fixture()
    first = engine.update(world, ego, reference, 1.2, current_side="left")
    assert engine.commit_selected(first["metadata"]["id"] + ":MAINTAIN", 1.2)
    anchor = engine._active["anchor"]
    ego.x += 12.
    result = engine.update(world, ego, reference, 1.4, current_side="left")
    assert engine._active["phase"] == "returning"
    assert result["diagnostics"]["anchor_station_m"] == anchor
    assert "pending_return_invalidation" not in result["diagnostics"]


def test_actual_ordinary_return_completes_only_after_side_entry_and_three_stable_frames():
    engine, world, ego, reference, _ = passing_fixture()
    # Physically entered side lane, but the dedicated return is never selected.
    first = engine.update(world, ego, reference, 1.2, current_side="left")
    assert first["phase"] == "returning"
    ego.x, ego.y = 72., 0.
    for now in (1.4, 1.6):
        result = engine.update(world, ego, reference, now, current_side="center")
        assert result["active"] and engine._active["phase"] == "passing"
    # Repeated semantic queries on the same snapshot cannot add observations.
    duplicate = engine.update(world, ego, reference, 1.6, current_side="center")
    assert duplicate["active"]
    result = engine.update(world, ego, reference, 1.8, current_side="center")
    assert not result["active"] and engine._active is None
    audit = result["diagnostics"]["measured_return_completion"]
    assert audit["samples"] == 3 and audit["elapsed_s"] == pytest.approx(.4)
    assert not audit["dedicated_return_selection_claimed"]


def test_initial_center_lane_and_transient_or_misaligned_return_do_not_complete():
    engine, world, ego, reference, _ = passing_fixture()
    # Commitment while still at the original center is not a physical excursion.
    ego.y = 0.
    for now in (1.2, 1.4, 1.6):
        assert engine.update(world, ego, reference, now, current_side="center")["active"]
    ego.y = -3.5
    engine.update(world, ego, reference, 1.8, current_side="left")
    ego.y, ego.yaw = 0., 25.
    for now in (2., 2.2, 2.4):
        assert engine.update(world, ego, reference, now, current_side="center")["active"]
    ego.yaw = 0.
    engine.update(world, ego, reference, 2.6, current_side="center")
    ego.y = -3.5
    engine.update(world, ego, reference, 2.8, current_side="left")
    ego.y = 0.
    result = engine.update(world, ego, reference, 3., current_side="center")
    assert result["active"] and result["diagnostics"]["measured_return_observation"]["samples"] == 1


def committed_return_fixture():
    engine, world, ego, reference, _ = passing_fixture()
    returning = engine.update(world, ego, reference, 1.2, current_side="left")
    assert returning["phase"] == "returning"
    assert engine.commit_selected(returning["metadata"]["id"] + ":MAINTAIN", 1.2)
    return engine, world, ego, reference


def test_committed_return_cannot_complete_with_frozen_twenty_five_degree_body_crossing():
    """Old endpoint/center-only branch released a body extending beyond 1.75m."""
    engine, world, ego, reference = committed_return_fixture()
    anchor, cap = engine._active["anchor"], engine._active["speed_cap"]
    ego.x, ego.y, ego.yaw = 74., -.4, 25.
    for now in (1.4, 1.6, 1.8):
        result = engine.update(world, ego, reference, now, current_side="center")
        pose = result["diagnostics"]["return_completion_pose"]
        assert result["active"] and engine._active["phase"] == "returning"
        assert pose["transition_end_reached"]
        assert not pose["heading_aligned"] and not pose["complete_body_inside_original_lane"]
        assert engine._active["anchor"] == anchor and result["speed_cap_mps"] == cap
        assert result["metadata"]["return_required"]
        assert "measured_return_observation" not in result["diagnostics"]


def test_committed_return_checks_actual_bbox_offset_even_when_actor_pose_is_centered():
    engine, world, ego, reference = committed_return_fixture()
    ego.x, ego.y, ego.yaw = 74., 0., 0.
    ego.bounding_box.location.y = .9
    for now in (1.4, 1.6, 1.8):
        result = engine.update(world, ego, reference, now, current_side="center")
        pose = result["diagnostics"]["return_completion_pose"]
        assert pose["heading_aligned"] and pose["lateral_error_m"] == 0.
        assert not pose["complete_body_inside_original_lane"] and result["active"]


def test_committed_return_completes_only_after_three_distinct_stable_body_observations():
    engine, world, ego, reference = committed_return_fixture()
    ego.x, ego.y, ego.yaw = 74., 0., 0.
    for now in (1.4, 1.6, 1.6):
        assert engine.update(world, ego, reference, now, current_side="center")["active"]
    assert engine._active["measured_return_observation"]["samples"] == 2
    completed = engine.update(world, ego, reference, 1.8, current_side="center")
    assert not completed["active"] and engine._active is None
    audit = completed["diagnostics"]["measured_return_completion"]
    assert audit["completion_phase"] == "returning" and audit["samples"] == 3
    assert audit["elapsed_s"] == pytest.approx(.4)
    assert audit["complete_body_inside_original_lane"] and audit["minimum_body_boundary_clearance_m"] > .05


def test_committed_return_end_and_uninterrupted_stability_are_both_required():
    engine, world, ego, reference = committed_return_fixture()
    ego.x, ego.y = engine._active["anchor"] - 10., 0.
    for now in (1.4, 1.6, 1.8):
        result = engine.update(world, ego, reference, now, current_side="center")
        assert result["active"] and not result["diagnostics"]["return_completion_pose"]["transition_end_reached"]
    ego.x = 74.
    engine.update(world, ego, reference, 2., current_side="center")
    ego.yaw = 25.
    assert engine.update(world, ego, reference, 2.2, current_side="center")["active"]
    ego.yaw = 0.
    for now in (2.4, 2.6):
        assert engine.update(world, ego, reference, now, current_side="center")["active"]
    assert not engine.update(world, ego, reference, 2.8, current_side="center")["active"]


def test_continuously_observed_one_hundred_fifty_one_metres_retains_return_obligation():
    engine, world, ego, reference = committed_return_fixture()
    plan = engine._active
    anchor, cap = plan["anchor"], plan["speed_cap"]
    reserved = list(plan["return_wait_world_xy"])
    ego.speed = 1.  # .2 m every .2 s remains below this candidate's cap.
    start = ego.x
    for step in range(1, 756):
        ego.x = start + step * .2
        result = engine.update(world, ego, reference, 1.2 + step*.2, current_side="left")
        assert result["active"] and engine._active["phase"] == "returning"
    assert ego.x - start == pytest.approx(151.)
    assert engine._active["anchor"] == anchor and result["speed_cap_mps"] == cap
    assert result["metadata"]["return_required"] and result["diagnostics"]["maneuver_overdue"]
    assert result["metadata"]["return_wait_world_xy"] == reserved
    assert engine._active["last_observed_route_station_m"] > anchor + 150.


def test_single_independent_observation_jump_discards_the_old_return_with_explicit_audit():
    engine, world, ego, reference = committed_return_fixture()
    previous = engine._active["last_observed_route_station_m"]
    ego.x += 151.
    result = engine.update(world, ego, reference, 1.4, current_side="left")
    assert not result["active"] and engine._active is None
    assert result["diagnostics"]["reason"] == "route_discontinuity_single_observation_station_jump"
    assert result["diagnostics"]["previous_route_station_m"] == previous
    assert result["diagnostics"]["observation_station_jump_m"] == pytest.approx(151.)
