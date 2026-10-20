"""Reproduce close walking-lead failures using real route/geometry calculations.

No CARLA server, model response or privileged future actor schedule is used.
The current-gap semantic evidence is deliberately weaker than execution safety.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import carla
import numpy as np
import pytest

from carla_bridge.planning.lane_maneuver import _curvature, _curvature_limit
from carla_bridge.planning.scene_context import build_scene_context, candidate_summary, planner_health
from carla_bridge.planning.semantic_governor import SelectiveSemanticGovernor
from carla_bridge.tests.test_lane_maneuver import fixture, observed, Obstacle
from carla_bridge.tests.test_pdm_planner import Road, Waypoint, Vehicle, RedLight
from carla_bridge.tests.test_planner_recovery_governor import recovery_advice


def pedestrian(x=6., speed=.7, vy=0.):
    person = Obstacle(actor_id=7, x=x, speed=speed, vy=vy)
    person.type_id = "walker.pedestrian.0001"
    person.bounding_box.extent = carla.Vector3D(x=.25, y=.25, z=.9)
    return person


def context_fixture(*, blocker=None, others=(), road=None):
    blocker = blocker or pedestrian()
    engine, world, ego, reference = fixture([blocker, *others], road=road)
    ego.speed = .7
    world.get_weather = lambda: NS(fog_density=0.)
    route = NS(_route_waypoints=reference,
               _nearest_index=lambda p: min(len(reference)-2, max(0, int((p.x+10)//2))),
               distance_completed_m=lambda: ego.x, nav_instruction=lambda: "STRAIGHT")
    return engine, world, ego, reference, route


def measured_context(world, ego, reference, route, now):
    result = build_scene_context(world, ego, route, now, reference=reference)
    # A high normalized score and >2m predicted progress used to declare this
    # measured 0.7m/s crawl healthy, even when the target lane was empty.
    pool = [{"id": "center:FOLLOW", "base_score": .95, "forward_progress_m": 3.,
             "longitudinal": "FOLLOW", "lateral_semantics": ["KEEP_LANE"],
             "safety": {"finite": True}}]
    result["candidates_summary"] = candidate_summary(pool, pool[0]["id"])
    result["planner_health"] = planner_health(pool, pool[0]["id"])
    result["planner_health"]["source"] = "nonlinear_mpc_baseline"
    return result, pool


def test_walking_lead_is_measured_even_above_stationary_threshold():
    _, world, ego, reference, route = context_fixture()
    context, _ = measured_context(world, ego, reference, route, 0.)
    actor = context["actors"][0]
    assert actor["speed_mps"] == pytest.approx(.7)
    assert not actor["stationary"] and not actor["path_blocked"]
    assert not context["blocked"] and not context["baseline_stop_expected"]
    assert context["slow_obstacle"]["actor_id"] == 7
    assert context["slow_obstacle"]["clearance_m"] == pytest.approx(3.45)
    assert context["slow_obstacle"]["legal_clear_sides"] == ["left", "right"]
    assert not context["slow_obstacle"]["execution_authorized"]


@pytest.mark.parametrize("kind", ["walker", "vehicle"])
def test_positive_scored_crawl_invites_model_only_after_sustained_measured_inefficiency(kind):
    blocker = pedestrian() if kind == "walker" else Obstacle(actor_id=7, x=10., speed=.7)
    _, world, ego, reference, route = context_fixture(blocker=blocker)
    governor = SelectiveSemanticGovernor()
    initial_blocker_x = blocker.x
    for now in np.arange(0., 4.01, .2):
        now = round(float(now), 6)
        ego.x, blocker.x = .7*now, initial_blocker_x+.7*now
        context, pool = measured_context(world, ego, reference, route, now)
        query = governor.query_decision(context, pool, now)
        if now < 4:
            assert not query["query"], query
    assert query["query"] and query["reason"] == "sustained_low_efficiency_following"
    assert query["moving_proposal_count"] == 1
    assert context["route_progress_m"] == pytest.approx(2.8)
    assert query["planner_efficiency"]["duration_s"] == 4
    assert query["planner_efficiency"]["execution_authorized"] is False
    assert query["planner_recovery"]["sustained"]
    assert query["planner_recovery"]["inefficient_following"]
    assert governor.submit(recovery_advice(), context, 4.)["accepted"]
    request = governor.recovery_request(context, 4.)
    assert request["authorized"] and request["native_score_override"] is False


def test_duplicate_samples_and_missing_time_cannot_forge_continuous_crawl():
    _, world, ego, reference, route = context_fixture()
    governor = SelectiveSemanticGovernor()
    context, pool = measured_context(world, ego, reference, route, 0.)
    for _ in range(30):
        governor.query_decision(context, pool, 0.)
    context, pool = measured_context(world, ego, reference, route, 8.)
    query = governor.query_decision(context, pool, 8.)
    assert not query["query"] and not query["planner_efficiency"]["sustained"]
    assert query["planner_efficiency"]["observation_count"] == 1


def test_red_queue_is_legitimate_wait_for_entire_long_observation():
    road = Road()
    _, world, ego, reference, route = context_fixture(
        blocker=Obstacle(actor_id=7, x=8., speed=.7), others=(RedLight(road, 14.),), road=road)
    governor = SelectiveSemanticGovernor()
    for now in np.arange(0., 30.01, .5):
        context, pool = measured_context(world, ego, reference, route, float(now))
        query = governor.query_decision(context, pool, float(now))
        assert context["baseline_stop_expected"]
        assert context["slow_obstacle"] is None
        assert not query["query"] and query["reason"] == "expected_signal_or_queue_wait"


def test_unavailable_neighbor_gaps_prevent_efficiency_escalation():
    # The rear car lies outside the context route window; its actual map lane
    # still identifies a fast-closing rear gap, rather than disappearing due to
    # an endpoint-clamped polyline projection.
    _, world, ego, reference, route = context_fixture(others=(
        Obstacle(actor_id=8, x=-20., y=-3.5, speed=10.),
        Obstacle(actor_id=9, x=6., y=3.5, speed=0.),))
    governor = SelectiveSemanticGovernor()
    for now in np.arange(0., 8.01, .5):
        context, pool = measured_context(world, ego, reference, route, float(now))
        query = governor.query_decision(context, pool, float(now))
    assert context["slow_obstacle"]["legal_clear_sides"] == []
    assert not query["query"] and not query["planner_efficiency"]["degraded"]
    left = next(o for o in context["overtaking_opportunities"] if o["side"] == "left")
    assert left["conflicts"][0]["actor_id"] == 8
    assert left["conflicts"][0]["kind"] == "rear"


def test_no_map_lane_permission_cannot_be_invented_for_efficiency():
    _, world, ego, reference, route = context_fixture(road=Road(allow_change=False))
    context, _ = measured_context(world, ego, reference, route, 0.)
    assert context["overtaking_opportunities"] == []
    assert context["slow_obstacle"]["legal_clear_sides"] == []


def test_crossing_or_map_crosswalk_pedestrian_is_not_an_efficiency_pass_target():
    for crosswalk, vy in ((False, .8), (True, 0.)):
        _, world, ego, reference, route = context_fixture(blocker=pedestrian(vy=vy))
        if crosswalk:
            world.get_map().get_crosswalks = lambda: [carla.Location(x=x, y=y) for x, y in
                [(4, -5), (8, -5), (8, 5), (4, 5), (4, -5)]]
        context, _ = measured_context(world, ego, reference, route, 0.)
        assert context["slow_obstacle"] is None


@pytest.mark.parametrize("distance", [6., 8., 10.])
def test_close_walking_actor_can_get_a_reachable_low_speed_path_without_complete_change_before_it(distance):
    engine, world, ego, reference, _ = context_fixture(blocker=pedestrian(x=distance))
    result = observed(engine, world, ego, reference, candidate_speed_mps=8.)
    assert result["active"], result["diagnostics"]
    assert result["diagnostics"]["transition_length_m"] > distance-2.3-.25-1.5
    assert max(_curvature(result["points"])) <= _curvature_limit(ego)
    assert .7+.75 < result["speed_cap_mps"] <= 5.
    assert result["diagnostics"]["committed"] is False
    assert engine.commit_selected(result["metadata"]["id"]+":ACCELERATE", 1.)
    anchor = engine._active["anchor"]
    ego.x += .2
    continued = engine.update(world, ego, reference, 1.2)
    assert continued["diagnostics"]["anchor_station_m"] == anchor


def test_actual_near_occupied_body_and_unreachable_steering_remain_rejected():
    engine, world, ego, reference, _ = context_fixture(blocker=pedestrian(x=3.))
    assert not observed(engine, world, ego, reference)["active"]
    engine, world, ego, reference, _ = context_fixture(blocker=pedestrian(x=6.))
    ego.max_steer_deg = 10.
    result = observed(engine, world, ego, reference)
    assert not result["active"]
    attempts = result["diagnostics"]["excluded"]["left"]["attempts"]
    assert any(a["reason"] == "steering_curvature_unreachable" for a in attempts)


def test_maneuver_speed_cap_respects_the_vehicle_steering_curve():
    engine, world, ego, reference, _ = context_fixture(blocker=pedestrian(x=6.))
    original = ego.get_physics_control
    def physics():
        value = original()
        value.steering_curve = [NS(x=0., y=1.), NS(x=10., y=.35)]
        return value
    ego.get_physics_control = physics
    result = observed(engine, world, ego, reference)
    assert result["active"], result["diagnostics"]
    curvature = max(_curvature(result["points"]))
    assert curvature <= _curvature_limit(ego, result["speed_cap_mps"])
    assert result["speed_cap_mps"] < 3.28


def test_efficiency_query_is_not_permission_to_stop_or_ignore_another_hazard():
    _, world, ego, reference, route = context_fixture()
    governor = SelectiveSemanticGovernor()
    for now in np.arange(0., 4.01, .2):
        context, pool = measured_context(world, ego, reference, route, round(float(now), 6))
        governor.query_decision(context, pool, round(float(now), 6))
    advice = recovery_advice()
    assert governor.submit(advice, context, 4.)["accepted"]
    changed = deepcopy(context)
    changed["actors"].append(dict(changed["actors"][0], id=99, same_lane=False,
        conflict=True, ttc_s=.5, path_blocked=False))
    assert not governor.recovery_request(changed, 4.)["authorized"]
    assert governor.recovery_request(changed, 4.)["reason"] == "no_recovery_request"
