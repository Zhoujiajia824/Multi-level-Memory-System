"""Current evidence and fixed geometry for ordinary blocked-lane manoeuvres."""
from copy import deepcopy
from types import SimpleNamespace as NS

import carla
import numpy as np
import pytest

from carla_bridge.planning.lane_maneuver import LaneManeuver, validate_maneuver_config
from carla_bridge.tests.test_pdm_planner import Road, Waypoint, Vehicle, RedLight


class Obstacle(Vehicle):
    def __init__(self, *args, vy=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vy = vy

    def get_velocity(self):
        return carla.Vector3D(x=self.speed, y=self.vy)


def fixture(others=None, *, road=None):
    road, ego = road or Road(), Vehicle(speed=1.)
    world = NS(get_map=lambda: road, get_actors=lambda: [ego] + (others or []))
    reference = [Waypoint(road, x) for x in range(-10, 160, 2)]
    return LaneManeuver(), world, ego, reference


def observed(engine, world, ego, reference, **kwargs):
    result = None
    for now in np.arange(0, 1.01, .2):
        result = engine.update(world, ego, reference, float(now), **kwargs)
    return result


@pytest.mark.parametrize("value", [False, [], {"enabled": 1}, {"unknown": 1},
    {"observation_s": 0}, {"observation_s": float("nan")}, {"slow_speed_fraction": 1.},
    {"maximum_pass_speed_mps": 9.}, {"lateral_acceleration_mps2": 3.}])
def test_invalid_configuration(value):
    with pytest.raises(ValueError):
        validate_maneuver_config(value)


def test_disabled_is_an_explicit_baseline_ablation():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    engine.config["enabled"] = False
    assert observed(engine, world, ego, reference)["diagnostics"]["reason"] == "disabled"
    assert engine._active is None


def test_one_observation_cannot_start_overtaking():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    result = engine.update(world, ego, reference, 0)
    assert not result["active"]
    assert result["diagnostics"]["reason"] == "observing_slow_obstacle"


def test_stationary_blocker_builds_real_adjacent_lane_path_without_model():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    result = observed(engine, world, ego, reference)
    assert result["active"] and result["target_side"] == "left"
    assert result["metadata"]["maneuver_blocker_id"] == 7
    assert not result["diagnostics"]["committed"]
    assert engine._active is None
    assert np.max(result["points"][:, 1]) == pytest.approx(3.5)
    assert result["speed_cap_mps"] <= 5.
    assert np.isfinite(result["points"]).all()
    assert result["metadata"]["lane_quality_transition"]
    assert result["metadata"]["lane_quality_reference_points"] == result["points"].tolist()


def test_fixed_anchor_and_geometry_survive_replans_after_real_selection():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    first = observed(engine, world, ego, reference)
    identity = first["metadata"]["id"] + ":ACCELERATE"
    assert engine.commit_selected(identity, 1.)
    for now, x in ((1.2, .3), (1.4, .8), (1.6, 1.2)):
        ego.x = x
        result = engine.update(world, ego, reference, now)
        assert result["diagnostics"]["committed"]
        assert result["diagnostics"]["anchor_station_m"] == first["diagnostics"]["anchor_station_m"]
        np.testing.assert_array_equal(result["points"], first["points"])


def test_selecting_an_ordinary_or_shadow_baseline_does_not_commit():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    result = observed(engine, world, ego, reference)
    assert not engine.commit_selected("center:ACCELERATE", 1.)
    assert engine._active is None
    assert not result["diagnostics"]["committed"]


@pytest.mark.parametrize("context", [{"traffic_wait": {"active": True}},
    {"traffic_wait": {"active": True, "reason": "signal_queue"}}])
def test_routine_red_queue_never_grants_a_pass(context):
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    result = observed(engine, world, ego, reference, context=context)
    assert not result["active"]
    assert result["diagnostics"]["reason"] == "traffic_signal_or_queue"


def test_signal_queue_is_recognised_without_semantic_context():
    road = Road()
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=20), RedLight(road, 30)], road=road)
    result = observed(engine, world, ego, reference)
    assert not result["active"]
    assert result["diagnostics"]["signal"]["signal_id"] == 100


def test_far_red_does_not_mask_a_separate_near_static_obstacle():
    road = Road()
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28), RedLight(road, 79)], road=road)
    assert observed(engine, world, ego, reference)["active"]


def test_no_lane_change_permission_cannot_be_overruled_by_blocker():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)], road=Road(allow_change=False))
    result = observed(engine, world, ego, reference)
    assert not result["active"]
    assert result["diagnostics"]["excluded"] == {"left": "lane_change_not_permitted", "right": "lane_change_not_permitted"}


@pytest.mark.parametrize("marking", ["Solid", "SolidSolid", "Curb"])
def test_solid_boundaries_reject_even_inconsistent_broad_lane_change_flag(marking):
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    for wp in reference:
        wp.left_lane_marking = wp.right_lane_marking = NS(type=marking, lane_change=carla.LaneChange.Both)
    assert not observed(engine, world, ego, reference)["active"]


def test_fast_closing_target_lane_vehicle_rejects_left_pass_and_chooses_clear_right():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28), Obstacle(actor_id=8, x=-8, y=-3.5, speed=10)])
    result = observed(engine, world, ego, reference)
    assert result["active"] and result["target_side"] == "right"


def test_both_adjacent_lanes_occupied_cannot_grant_overtaking():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28),
        Obstacle(actor_id=8, x=8, y=-3.5), Obstacle(actor_id=9, x=8, y=3.5)])
    result = observed(engine, world, ego, reference)
    assert not result["active"]
    assert result["diagnostics"]["reason"] == "no_safe_legal_overtaking_corridor"


def test_crossing_pedestrian_is_not_an_overtaking_target():
    person = Obstacle(actor_id=7, x=28, vy=1.2)
    person.type_id = "walker.pedestrian.0001"
    person.bounding_box.extent = carla.Vector3D(x=.25, y=.25, z=.9)
    engine, world, ego, reference = fixture([person])
    result = observed(engine, world, ego, reference)
    assert not result["active"] and result["diagnostics"]["reason"] == "no_persistent_slow_obstacle"


def test_stationary_lane_pedestrian_can_be_passed_only_with_clear_legal_adjacent_lane():
    person = Obstacle(actor_id=7, x=28)
    person.type_id = "walker.pedestrian.0001"
    person.bounding_box.extent = carla.Vector3D(x=.25, y=.25, z=.9)
    engine, world, ego, reference = fixture([person])
    assert observed(engine, world, ego, reference)["active"]


def test_normal_lead_speed_is_not_an_obstacle_even_if_persistent():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28, speed=6.)])
    assert not observed(engine, world, ego, reference)["active"]


def test_very_close_blocker_cannot_force_an_impossible_lane_change():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=10)])
    result = observed(engine, world, ego, reference)
    assert not result["active"]
    assert result["diagnostics"]["reason"] == "insufficient_space_before_obstacle"


def test_lane_return_waits_until_actual_blocker_is_behind_then_anchors_once():
    blocker = Obstacle(actor_id=7, x=28)
    engine, world, ego, reference = fixture([blocker])
    first = observed(engine, world, ego, reference)
    engine.commit_selected(first["metadata"]["id"] + ":ACCELERATE", 1.)
    ego.x, ego.y = 35., -3.5
    still_passing = engine.update(world, ego, reference, 1.2, current_side="left")
    assert still_passing["phase"] == "passing"
    ego.x = 44.
    returning = engine.update(world, ego, reference, 1.4, current_side="left")
    assert returning["phase"] == "returning" and returning["target_side"] == "center"
    assert engine.commit_selected(returning["metadata"]["id"] + ":MAINTAIN", 1.4)
    ego.x = 45.
    repeat = engine.update(world, ego, reference, 1.6, current_side="left")
    np.testing.assert_array_equal(repeat["points"], returning["points"])
    ego.x, ego.y = 72., 0.
    finished = engine.update(world, ego, reference, 1.8)
    assert not finished["active"] and finished["diagnostics"]["reason"] == "completed"


def test_changing_blocker_resets_persistence_window():
    blocker = Obstacle(actor_id=7, x=28)
    engine, world, ego, reference = fixture([blocker])
    for now in (0, .2, .4, .6, .8):
        engine.update(world, ego, reference, now)
    blocker.id = 8
    result = engine.update(world, ego, reference, 1.)
    assert not result["active"]


def test_no_real_clock_continuity_cannot_forge_sustained_observation():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    engine.update(world, ego, reference, 0)
    assert not engine.update(world, ego, reference, 20)["active"]


def test_bridge_deck_actor_does_not_become_a_false_blocker():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28, z=10)])
    assert not observed(engine, world, ego, reference)["active"]


def test_stationary_pedestrian_at_map_crosswalk_requires_yield_instead_of_pass():
    person = Obstacle(actor_id=7, x=28)
    person.type_id = "walker.pedestrian.0001"
    engine, world, ego, reference = fixture([person])
    world.get_map().get_crosswalks = lambda: [carla.Location(x=x, y=y) for x, y in
        [(26, -5), (30, -5), (30, 5), (26, 5), (26, -5)]]
    result = observed(engine, world, ego, reference)
    assert not result["active"]
    assert result["diagnostics"]["reason"] == "pedestrian_at_crosswalk"


def test_fast_car_in_original_lane_prevents_return_until_current_gap_is_clear():
    blocker = Obstacle(actor_id=7, x=28)
    actors = [blocker]
    engine, world, ego, reference = fixture(actors)
    first = observed(engine, world, ego, reference)
    engine.commit_selected(first["metadata"]["id"] + ":ACCELERATE", 1.)
    ego.x, ego.y = 44., -3.5
    actors.append(Obstacle(actor_id=8, x=38, speed=10))
    result = engine.update(world, ego, reference, 1.2, current_side="left")
    assert result["phase"] == "passing" and result["target_side"] == "left"


def test_junction_cannot_be_used_as_a_new_overtaking_entry():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    for wp in reference:
        wp.is_junction = True
    assert observed(engine, world, ego, reference)["diagnostics"]["reason"] == "not_in_reference_lane_or_at_junction"


def test_resolved_route_cursor_is_used_for_station_when_geometry_repeats():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    # Two geometrically identical route segments must not alias their station.
    reference = [Waypoint(world.get_map(), x) for x in range(-200, 160, 2)] * 2
    result = observed(engine, world, ego, reference, route_index=280)
    assert result["active"], result["diagnostics"]
    assert result["diagnostics"]["anchor_station_m"] > 200


def test_evaluation_route_end_is_not_an_invented_stationary_leader_for_the_return():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    reference = [Waypoint(world.get_map(), x) for x in range(0, 82, 2)]
    initial = observed(engine, world, ego, reference)
    assert initial["active"]
    assert initial["points"][-1, 0] >= 150
    engine.commit_selected(initial["metadata"]["id"] + ":ACCELERATE", 1.)
    ego.x, ego.y = 44., -3.5
    returning = engine.update(world, ego, reference, 1.2, current_side="left", route_index=22)
    assert returning["phase"] == "returning"
    assert returning["points"][-1, 0] >= 130


def test_unselected_return_proposal_keeps_its_anchor_and_full_tail():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    initial = observed(engine, world, ego, reference)
    engine.commit_selected(initial["metadata"]["id"] + ":ACCELERATE", 1.)
    ego.x, ego.y = 44., -3.5
    returning = engine.update(world, ego, reference, 1.2, current_side="left")
    engine.commit_selected("left:ACCELERATE", 1.2)
    for now, x in ((1.4, 45.), (1.6, 46.), (1.8, 47.)):
        ego.x = x
        proposal = engine.update(world, ego, reference, now, current_side="left")
        assert proposal["phase"] == "returning"
        assert proposal["diagnostics"]["anchor_station_m"] == returning["diagnostics"]["anchor_station_m"]
        np.testing.assert_array_equal(proposal["points"], returning["points"])
        assert engine._active["phase"] == "passing"


def generic_metadata(side="left", *, returning=False):
    direction = ("right" if side == "left" else "left") if returning else side
    return dict(label="CHANGE_LANE_" + direction.upper(), physical_lane_side="center" if returning else side,
                route_target=returning, maneuver=False, transition_length_m=30.)


def test_actual_selected_generic_change_commits_its_exact_geometry_without_claiming_fixed_win():
    import hashlib
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    proposal = observed(engine, world, ego, reference)
    executed = proposal["points"][::4].copy()
    assert engine.commit_selected("left:ACCELERATE", 1., selected_path_points=executed,
                                  selected_metadata=generic_metadata())
    assert engine._active["phase"] == "passing"
    assert engine.last_commit_audit["actual_selected_id"] == "left:ACCELERATE"
    assert engine.last_commit_audit["commit_source"] == "executed_generic_numerical_candidate"
    assert engine.last_commit_audit["executed_geometry_sha256"] == hashlib.sha256(executed.tobytes()).hexdigest()
    np.testing.assert_array_equal(engine._active["executed_selected_points"], executed)
    original_anchor = engine._active["anchor"]
    engine.update(world, ego, reference, 1.2)
    assert not engine.commit_selected("left:ACCELERATE", 1.2, selected_path_points=executed,
                                      selected_metadata=generic_metadata())
    assert engine._active["anchor"] == original_anchor


def test_actual_selected_generic_return_enters_returning_after_genuine_passing():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    proposal = observed(engine, world, ego, reference)
    engine.commit_selected("left:ACCELERATE", 1., selected_path_points=proposal["points"],
                           selected_metadata=generic_metadata())
    ego.x, ego.y = 46., -3.5
    returning = engine.update(world, ego, reference, 1.2, current_side="left")
    assert returning["phase"] == "returning"
    assert engine.commit_selected("center:ACCELERATE", 1.2, selected_path_points=returning["points"],
                                  selected_metadata=generic_metadata(returning=True))
    assert engine._active["phase"] == "returning"
    assert engine.last_commit_audit["actual_selected_id"] == "center:ACCELERATE"
    assert engine.last_commit_audit["actual_selected_path_label"] == "CHANGE_LANE_RIGHT"


@pytest.mark.parametrize("metadata", [generic_metadata("right"),
    dict(generic_metadata(), label="KEEP_LANE"), dict(generic_metadata(), route_target=True),
    dict(generic_metadata(), maneuver=True), dict(generic_metadata(), transition_length_m=None)])
def test_other_executed_action_does_not_forge_a_verified_maneuver_commit(metadata):
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    proposal = observed(engine, world, ego, reference)
    assert not engine.commit_selected("other:ACCELERATE", 1., selected_path_points=proposal["points"],
                                      selected_metadata=metadata)
    assert engine._active is None


def test_generic_longer_transition_needs_its_own_complete_permission_evidence():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    proposal = observed(engine, world, ego, reference)
    for wp in engine._last_reference:
        if 28 <= wp.x <= 32:
            wp.lane_change = carla.LaneChange.NONE
    assert not engine.commit_selected("left:ACCELERATE", 1., selected_path_points=proposal["points"],
                                      selected_metadata=generic_metadata())
    assert engine.last_commit_audit["reason"] == "executed_transition_outside_verified_corridor"
    assert engine._active is None


def test_selected_generic_cannot_claim_an_old_obstacle_snapshot():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    proposal = observed(engine, world, ego, reference)
    assert not engine.commit_selected("left:ACCELERATE", 1.2, selected_path_points=proposal["points"],
                                      selected_metadata=generic_metadata())
    assert engine.last_commit_audit["reason"] == "stale_maneuver_evidence"


@pytest.mark.parametrize("side,y", [("left", -3.5), ("right", 3.5)])
def test_return_cannot_complete_in_either_adjacent_lane_even_after_transition_end(side, y):
    actors = [Obstacle(actor_id=7, x=28)]
    if side == "right":
        actors.append(Obstacle(actor_id=8, x=8, y=-3.5))
    engine, world, ego, reference = fixture(actors)
    passing = observed(engine, world, ego, reference)
    assert passing["target_side"] == side
    assert engine.commit_selected(passing["metadata"]["id"] + ":ACCELERATE", 1.)
    ego.x, ego.y = 44., y
    returning = engine.update(world, ego, reference, 1.2, current_side=side)
    assert returning["phase"] == "returning"
    assert engine.commit_selected(returning["metadata"]["id"] + ":ACCELERATE", 1.2)
    ego.x = 78.
    still_adjacent = engine.update(world, ego, reference, 1.4, current_side=side)
    assert still_adjacent["active"] and still_adjacent["phase"] == "returning"
    ego.y = 0.
    centered = engine.update(world, ego, reference, 1.6)
    assert not centered["active"] and centered["diagnostics"]["reason"] == "completed"


def crossing_actor(x):
    return {"id": 12, "xy": np.array([x, 9.]), "velocity": np.array([0., -2.]),
            "station": x, "half_length": .25, "half_width": .25,
            "aligned": False, "along_speed": 0.}


def test_cv_gap_checks_ordinary_eight_mps_candidate_not_only_five_mps_fixed_cap():
    engine, _, ego, _ = fixture()
    samples = np.arange(0., 101.)
    points = np.c_[samples, np.zeros(len(samples))]
    actors = [crossing_actor(36.)]
    assert engine._gap_clear(actors, points, samples, 0., ego, 5., candidate_speed_mps=5.)[0]
    clear, conflict = engine._gap_clear(actors, points, samples, 0., ego, 5., candidate_speed_mps=8.)
    assert not clear and conflict["ego_probe_speed_mps"] == 8.
    assert conflict["reason"] == "predicted_occupied_envelope"
    # The default likewise covers the native ordinary maximum (min(8, limit)).
    assert not engine._gap_clear(actors, points, samples, 0., ego, 5.)[0]


def test_adding_upper_speed_probe_keeps_slower_arrival_conflict_detection():
    engine, _, ego, _ = fixture()
    samples = np.arange(0., 101.)
    points = np.c_[samples, np.zeros(len(samples))]
    clear, conflict = engine._gap_clear([crossing_actor(24.)], points, samples, 0., ego, 5.,
                                        candidate_speed_mps=8.)
    assert not clear and conflict["ego_probe_speed_mps"] == 5.


def test_prediction_never_pretends_measured_ten_mps_ego_immediately_slows_to_candidate_limit():
    engine, _, ego, _ = fixture()
    ego.speed = 10.
    samples = np.arange(0., 101.)
    points = np.c_[samples, np.zeros(len(samples))]
    clear, conflict = engine._gap_clear([crossing_actor(45.)], points, samples, 0., ego, 5.,
                                        candidate_speed_mps=8.)
    assert not clear and conflict["ego_probe_speed_mps"] == 10.


def test_target_front_gap_covers_faster_ordinary_policy_without_relaxing_rear_gap():
    engine, _, ego, reference = fixture()
    stations = np.arange(0., len(reference) * 2, 2.)
    actor = dict(crossing_actor(32.), xy=np.array([22., 3.5]), station=32., velocity=np.zeros(2),
                 aligned=True, half_length=2.3, half_width=.9)
    assert engine._target_gap_clear([actor], reference, stations, 10., ego, "left", 5.,
                                    candidate_speed_mps=5.)[0]
    clear, conflict = engine._target_gap_clear([actor], reference, stations, 10., ego, "left", 5.,
                                               candidate_speed_mps=8.)
    assert not clear and conflict["reason"] == "target_front_gap"
    assert conflict["required_m"] == 24.
    rear = dict(actor, station=25., xy=np.array([15., 3.5]), along_speed=10.)
    clear, conflict = engine._target_gap_clear([rear], reference, stations, 50., ego, "left", 5.,
                                               candidate_speed_mps=8.)
    assert not clear and conflict["reason"] == "target_rear_gap"
    assert conflict["required_m"] == 32.


@pytest.mark.parametrize("speed", [-1., float("nan"), float("inf"), True])
def test_invalid_ordinary_candidate_speed_fails_before_scene_evidence_is_used(speed):
    engine, world, ego, reference = fixture()
    with pytest.raises(ValueError, match="ordinary candidate speed"):
        engine.update(world, ego, reference, 0., candidate_speed_mps=speed)


def pending_return_fixture():
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    passing = observed(engine, world, ego, reference)
    assert engine.commit_selected(passing["metadata"]["id"] + ":ACCELERATE", 1.)
    ego.x, ego.y = 44., -3.5
    returning = engine.update(world, ego, reference, 1.2, current_side="left")
    assert returning["phase"] == "returning"
    return engine, world, ego, reference, returning


def adjustment_metadata(returning, offset, *, label="CHANGE_LANE_RIGHT", limit=.5):
    return dict(label=label, physical_lane_side="center", route_target=True,
        maneuver=False, maneuver_continuation_adjustment=True, lane_offset_m=offset,
        maneuver_adjustment_limit_m=limit,
        transition_length_m=returning["diagnostics"]["transition_length_m"])


def test_first_executed_return_offset_retains_nominal_geometry_from_its_own_phase():
    engine, world, ego, reference, returning = pending_return_fixture()
    nominal = returning["points"].copy()
    adjusted = nominal + [0., .25]
    assert engine.commit_selected("center@+0.25:ACCELERATE", 1.2,
        selected_path_points=adjusted, selected_metadata=adjustment_metadata(returning, .25)), engine.last_commit_audit
    assert engine._active["phase"] == "returning"
    np.testing.assert_array_equal(engine._active["nominal_points"], nominal)
    np.testing.assert_array_equal(engine._active["executed_selected_points"], adjusted)
    assert engine.last_commit_audit["actual_selected_id"] == "center@+0.25:ACCELERATE"
    assert engine.last_commit_audit["reason"] == "executed_generic_numerical_adjustment"
    continued = engine.update(world, ego, reference, 1.4, current_side="left")
    np.testing.assert_array_equal(continued["nominal_points"], nominal)
    assert not np.array_equal(continued["points"], continued["nominal_points"])


def test_executed_keep_lane_adjustment_keeps_anchor_time_and_immutable_nominal_without_drift():
    engine, world, ego, reference, returning = pending_return_fixture()
    assert engine.commit_selected(returning["metadata"]["id"] + ":ACCELERATE", 1.2)
    anchor, started = engine._active["anchor"], engine._active["committed_at"]
    nominal = returning["points"].copy()
    for now, offset in ((1.4, .25), (1.6, .5)):
        current = engine.update(world, ego, reference, now, current_side="left")
        actual = nominal + [0., offset]
        assert engine.commit_selected("center@%+.2f:MAINTAIN" % offset, now,
            selected_path_points=actual,
            selected_metadata=adjustment_metadata(current, offset, label="KEEP_LANE")), engine.last_commit_audit
        assert engine._active["phase"] == "returning"
        assert engine._active["anchor"] == anchor and engine._active["committed_at"] == started
        assert engine.last_commit_audit["actual_selected_path_label"] == "KEEP_LANE"
        np.testing.assert_array_equal(engine._active["nominal_points"], nominal)
        np.testing.assert_array_equal(engine._active["executed_selected_points"], actual)
    current = engine.update(world, ego, reference, 1.8, current_side="left")
    previous = engine._active["executed_selected_points"].copy()
    assert not engine.commit_selected("center@+0.50:MAINTAIN", 1.8,
        selected_path_points=previous + [0., .5],
        selected_metadata=adjustment_metadata(current, .5, label="KEEP_LANE"))
    assert engine.last_commit_audit["reason"] == "adjustment_not_bounded_to_fixed_nominal"
    np.testing.assert_array_equal(engine._active["executed_selected_points"], previous)
    np.testing.assert_array_equal(engine._active["nominal_points"], nominal)


@pytest.mark.parametrize("overrides", [
    {"maneuver_continuation_adjustment": False}, {"physical_lane_side": "left"},
    {"maneuver_adjustment_limit_m": .6}, {"lane_offset_m": .6},
    {"maneuver_adjustment_limit_m": float("nan")}, {"transition_length_m": 30.},
])
def test_unflagged_unbounded_or_restarted_adjustment_cannot_replace_actual_execution(overrides):
    engine, world, ego, reference, returning = pending_return_fixture()
    assert engine.commit_selected(returning["metadata"]["id"] + ":ACCELERATE", 1.2)
    current = engine.update(world, ego, reference, 1.4, current_side="left")
    metadata = adjustment_metadata(current, .25, label="KEEP_LANE")
    metadata.update(overrides)
    old = engine._active["points"].copy()
    assert not engine.commit_selected("center@+0.25:ACCELERATE", 1.4,
        selected_path_points=current["nominal_points"] + [0., .25], selected_metadata=metadata)
    np.testing.assert_array_equal(engine._active["points"], old)


def test_keep_lane_alone_cannot_claim_the_first_return_transition():
    engine, _, _, _, returning = pending_return_fixture()
    assert not engine.commit_selected("center@+0.25:ACCELERATE", 1.2,
        selected_path_points=returning["points"] + [0., .25],
        selected_metadata=adjustment_metadata(returning, .25, label="KEEP_LANE"))
    assert engine._active["phase"] == "passing"


def test_fixed_return_replaces_actual_generic_passing_provenance():
    import hashlib
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    passing = observed(engine, world, ego, reference)
    generic_points = passing["points"][::4].copy()
    assert engine.commit_selected("left:ACCELERATE", 1.,
        selected_path_points=generic_points, selected_metadata=generic_metadata())
    started = engine._active["committed_at"]
    ego.x, ego.y = 44., -3.5
    returning = engine.update(world, ego, reference, 1.2, current_side="left")
    assert returning["phase"] == "returning"
    nominal = returning["nominal_points"].copy()
    actual = returning["points"][::2].copy()
    identity = returning["metadata"]["id"] + ":MAINTAIN"
    assert engine.commit_selected(identity, 1.2,
        selected_path_points=actual, selected_metadata=returning["metadata"])
    assert engine._active["commit_source"] == "executed_fixed_numerical_candidate"
    assert engine._active["actual_selected_id"] == identity
    assert engine._active["actual_selected_path_label"] == "CHANGE_LANE_RIGHT"
    expected_hash = hashlib.sha256(actual.tobytes()).hexdigest()
    assert engine._active["executed_geometry_sha256"] == expected_hash
    assert engine.last_commit_audit["executed_geometry_sha256"] == expected_hash
    np.testing.assert_array_equal(engine._active["executed_selected_points"], actual)
    np.testing.assert_array_equal(engine._active["nominal_points"], nominal)
    assert engine._active["anchor"] == returning["diagnostics"]["anchor_station_m"]
    assert engine._active["committed_at"] == started
    continued = engine.update(world, ego, reference, 1.4, current_side="left")
    assert continued["diagnostics"]["commit_source"] == "executed_fixed_numerical_candidate"
    assert continued["diagnostics"]["executed_geometry_sha256"] == expected_hash


def test_direct_fixed_commit_records_pending_geometry_when_actual_array_is_not_supplied():
    import hashlib
    engine, world, ego, reference = fixture([Obstacle(actor_id=7, x=28)])
    passing = observed(engine, world, ego, reference)
    assert engine.commit_selected(passing["metadata"]["id"] + ":ACCELERATE", 1.)
    expected_hash = hashlib.sha256(passing["points"].tobytes()).hexdigest()
    assert engine._active["executed_geometry_sha256"] == expected_hash
    np.testing.assert_array_equal(engine._active["executed_selected_points"], passing["points"])
