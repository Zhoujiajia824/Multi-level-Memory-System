"""A model may request new numeric options, never authorize zero-score driving."""
from copy import deepcopy
import math

import pytest

from carla_bridge.planning.scene_context import candidate_summary, planner_health
from carla_bridge.planning.semantic_governor import SelectiveSemanticGovernor


def candidate(identity, score=1.0, longitudinal="MAINTAIN", progress=20.0, **safety):
    return {"id": identity, "base_score": score, "longitudinal": longitudinal,
            "lateral_semantics": ["FOLLOW_ROUTE", "KEEP_LANE"], "forward_progress_m": progress,
            "safety": dict(finite=True, no_collision=1.0, drivable_area=1.0,
                           driving_direction=1.0, ttc=1.0, **safety)}


def degenerate_pool():
    moving = candidate("offroad_moving", 0.0)
    moving["safety"]["drivable_area"] = 0.0
    return [candidate("native_stop", 1.0, "STOP", 0.0), moving]


def context(now=0.0, candidates=None, **changes):
    candidates = candidates if candidates is not None else degenerate_pool()
    scene = {"schema_version": 1, "sim_time_s": now, "ego_speed_mps": 0.0,
             "route_progress_m": 10.0, "in_junction": True, "lane_key": "1:0:1",
             "navigation": "RIGHT", "speed_limit_mps": 10.0, "signals": [],
             "actors": [], "blocked": False, "special_situations": [],
             "topology_signature": "junction1", "baseline_stop_expected": False,
             "candidates_summary": candidate_summary(candidates, candidates[0]["id"]),
             "planner_health": planner_health(candidates, candidates[0]["id"])}
    scene.update(changes)
    return scene


def recovery_advice(**changes):
    advice = {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "MAINTAIN",
              "behavior_reason": "Request numeric alternatives for persistent native degradation.",
              "risk_level": "medium", "safety_notes": [],
              "intervention": {"needed": True, "reason_code": "planner_degraded",
                "target_actor_ids": [], "target_stop_line_id": None,
                "activation": "immediate", "release": "condition_cleared", "valid_for_s": 4.0,
                "planner_request": "expand_candidates"}}
    advice["intervention"].update(changes)
    return advice


def observe_failure(governor, end=2.0):
    for step in range(round(end/.2)+1):
        now = round(step*.2, 6)
        governor.observe_planner_health(context(now), now)
    return context(end)


def actor(**changes):
    item = {"id": 7, "type": "vehicle.test", "distance_m": 6.0,
            "relative_s_m": 5.0, "relative_l_m": 1.0, "same_lane": False,
            "ahead": True, "conflict": True, "ttc_s": 2.0,
            "stationary": False, "path_blocked": False, "route_distance_m": 5.0}
    item.update(changes)
    return item


def signal(**changes):
    item = {"id": "stop_9_0", "state": "Red", "applicable": True,
            "stop_line_distance_m": 5.0}
    item.update(changes)
    return item


def test_zero_scored_forward_options_do_not_hide_native_degradation():
    health = planner_health(degenerate_pool(), "native_stop")
    assert health["baseline_score"] == 1.0 and health["baseline_stop"]
    assert health["moving_candidate_count"] == 1
    assert health["positive_progressing_count"] == 0
    assert health["drivable_area_failure_count"] == 1
    assert health["no_positive_progressing_candidate"]
    assert candidate_summary(degenerate_pool(), "native_stop")["candidate_count"] == 2


def test_health_counts_only_finite_positive_progressing_non_stop_candidates():
    candidates = degenerate_pool()+[candidate("valid", .01), candidate("nan", math.nan),
        candidate("negative", -.1), candidate("tiny", 1, progress=.5)]
    candidates[-1]["safety"]["finite"] = False
    health = planner_health(candidates, "native_stop")
    assert health["candidate_count"] == 6 and health["finite_candidate_count"] == 4
    assert health["positive_progressing_count"] == 1
    assert not health["no_positive_progressing_candidate"]


def test_native_stop_one_does_not_prevent_a_sustained_degradation_query():
    governor = SelectiveSemanticGovernor()
    for now in (0.0, .5, 1.0, 1.5):
        query = governor.query_decision(context(now), degenerate_pool(), now)
        assert not query["query"]
        assert query["reason"] == "observing_native_planner_degradation"
    query = governor.query_decision(context(2), degenerate_pool(), 2)
    assert query["query"]
    assert query["reason"] == "sustained_native_planner_degradation"
    assert query["moving_proposal_count"] == 0
    assert query["planner_recovery"]["duration_s"] == 2


def test_duplicate_checks_at_same_snapshot_do_not_create_observations_or_consume_cooldown():
    governor = SelectiveSemanticGovernor()
    for _ in range(30):
        state = governor.observe_planner_health(context(), 0)
        governor.recovery_request(context(), 0)
        governor.select(degenerate_pool(), context(), 0)
    assert state["observation_count"] == 1
    assert governor._last_query_s is None
    assert not state["sustained"]


def test_all_zero_tie_with_nominal_moving_baseline_still_detects_actual_stationary_failure():
    governor = SelectiveSemanticGovernor()
    candidates = [candidate("zero_forward", 0), candidate("zero_stop", 0, "STOP", 0)]
    for now in (0, .5, 1, 1.5, 2):
        query = governor.query_decision(context(now, candidates=candidates), candidates, now)
    assert query["query"] and query["reason"] == "sustained_native_planner_degradation"
    assert governor.submit(recovery_advice(), context(2, candidates=candidates), 2)["accepted"]


def test_missing_intermediate_snapshots_do_not_prove_continuous_degradation():
    governor = SelectiveSemanticGovernor()
    governor.observe_planner_health(context(0), 0)
    assert not governor.observe_planner_health(context(2), 2)["sustained"]
    assert governor.submit(recovery_advice(), context(2), 2)["reason"] == "planner_degradation_not_sustained"


def test_fresh_lease_requires_sustained_native_evidence_then_expires_without_renewal():
    governor = SelectiveSemanticGovernor()
    assert not governor.submit(recovery_advice(), context(), 0)["accepted"]
    current = observe_failure(governor)
    assert governor.submit(recovery_advice(), current, 2)["accepted"]
    for now in (2.2, 3, 3.8, 4.6, 5.4):
        result = governor.recovery_request(context(now), now)
        assert result["authorized"] and result["expires_at_s"] == 6.0
        assert result["native_score_override"] is False
    result = governor.recovery_request(context(6), 6)
    assert not result["authorized"] and result["reason"] == "intervention_expired"
    assert not governor.recovery_request(context(6.2), 6.2)["authorized"]


def test_candidate_generation_request_does_not_make_zero_scored_candidates_executable():
    governor = SelectiveSemanticGovernor({"base_regret_budget": 1.0})
    current = observe_failure(governor)
    assert governor.submit(recovery_advice(), current, 2)["accepted"]
    selected, diagnostics = governor.select(degenerate_pool(), current, 2)
    moving = next(c for c in diagnostics["candidates"] if c["id"] == "offroad_moving")
    assert moving["eligible"]  # Native proposals remain observable; no global restored gate.
    assert moving["semantic_bonus"] == 0
    assert selected["id"] == "native_stop"
    assert diagnostics["safety_gate_applied"] is False


def test_expanded_positive_candidates_do_not_replace_original_degradation_evidence():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(), current, 2)
    expanded = [candidate("native_stop", .05, "STOP", 0), candidate("new_forward", 1)]
    before = deepcopy(current["planner_health"])
    selected, diagnostics = governor.select(expanded, current, 2)
    assert selected["id"] == "new_forward"
    assert diagnostics["governor"]["planner_recovery"]["sustained"]
    assert current["planner_health"] == before
    assert governor.recovery_request(current, 2)["authorized"]


def test_one_healthy_original_snapshot_preserves_finite_lease_without_faking_degradation():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(), current, 2)
    healthy = context(2.2, candidates=[candidate("forward", 1)])
    result = governor.recovery_request(healthy, 2.2)
    assert result["authorized"]
    assert result["evidence"]["recovery_authorization"] == "continuing_recovery"
    assert not result["planner_health"]["degraded"]
    state = governor.observe_planner_health(context(2.4), 2.4)
    assert not state["sustained"] and state["duration_s"] == 0
    assert governor.recovery_request(context(2.4), 2.4)["authorized"]


def test_recovery_releases_only_after_continuous_native_health_and_measured_progress():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(valid_for_s=6), current, 2)
    for now, progress in ((2.2, .2), (2.7, .5), (3.2, 1), (3.7, 1.5)):
        healthy = context(now, candidates=[candidate("forward", 1)], route_progress_m=10+progress)
        assert governor.recovery_request(healthy, now)["authorized"]
    healthy = context(4.2, candidates=[candidate("forward", 1)], route_progress_m=12)
    result = governor.recovery_request(healthy, 4.2)
    assert not result["authorized"] and result["reason"] == "native_planner_recovery_confirmed"
    assert result["evidence"]["native_healthy_duration_s"] == pytest.approx(2)
    assert result["evidence"]["recovery_progress_m"] == pytest.approx(2)
    assert not governor.recovery_request(healthy, 4.2)["authorized"]


def test_health_flicker_restarts_confirmation_but_keeps_the_same_expiry():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(valid_for_s=6), current, 2)
    for now in (2.2, 2.7):
        result = governor.recovery_request(context(now, candidates=[candidate("forward", 1)],
            route_progress_m=13), now)
        assert result["authorized"]
    assert governor.recovery_request(context(2.9, route_progress_m=13), 2.9)["authorized"]
    for now in (3.1, 3.6, 4.1, 4.6):
        result = governor.recovery_request(context(now, candidates=[candidate("forward", 1)],
            route_progress_m=13), now)
        assert result["authorized"] and result["expires_at_s"] == 8
    result = governor.recovery_request(context(5.1, candidates=[candidate("forward", 1)],
        route_progress_m=13), 5.1)
    assert result["reason"] == "native_planner_recovery_confirmed"


def test_healthy_predictions_without_measured_progress_never_claim_recovery_or_extend_ttl():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(valid_for_s=6), current, 2)
    for now in (2.2, 2.7, 3.2, 3.7, 4.2, 4.7, 5.2, 5.7, 6.2, 6.7, 7.2, 7.7):
        result = governor.recovery_request(context(now, candidates=[candidate("forward", 1)]), now)
        assert result["authorized"] and result["expires_at_s"] == 8
    result = governor.recovery_request(context(8, candidates=[candidate("forward", 1)]), 8)
    assert result["reason"] == "intervention_expired" and not result["authorized"]


def test_unobserved_health_gap_cannot_satisfy_release_confirmation():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(valid_for_s=6), current, 2)
    for now in (2.2, 4.4):
        result = governor.recovery_request(context(now, candidates=[candidate("forward", 1)],
            route_progress_m=13), now)
        assert result["authorized"]
        assert result["evidence"]["native_healthy_duration_s"] == 0


def test_repeated_checks_of_one_healthy_snapshot_do_not_satisfy_release_confirmation():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(valid_for_s=6), current, 2)
    healthy = context(2.2, candidates=[candidate("forward", 1)], route_progress_m=13)
    for _ in range(30):
        assert governor.recovery_request(healthy, 2.2)["authorized"]
        governor.select([candidate("forward", 1)], healthy, 2.2)
    result = governor.recovery_request(healthy, 2.2)
    assert result["evidence"]["native_healthy_duration_s"] == 0


def test_healthy_first_submission_is_not_allowed_to_invent_an_initial_degradation():
    governor = SelectiveSemanticGovernor()
    healthy = context(candidates=[candidate("forward", 1)])
    assert not governor.submit(recovery_advice(), healthy, 0)["accepted"]
    assert not governor.recovery_request(healthy, 0)["authorized"]


def test_healthy_flicker_reuses_consultation_during_generation_lease():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(valid_for_s=6), current, 2)
    for now, candidates in ((2.2, [candidate("forward", 1)]), (2.4, degenerate_pool()),
                            (2.6, [candidate("forward", 1)])):
        query = governor.query_decision(context(now, candidates=candidates), candidates, now)
        assert not query["query"] and query["reason"] == "active_guidance_cached"


def test_candidate_generation_request_gets_no_semantic_bonus_including_stationary_decelerate():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(), current, 2)
    candidates = [candidate("native_stop", 1, "STOP", 0),
        candidate("decelerate_is_stationary", 1, "DECELERATE", 0),
        candidate("expanded_forward", .95, "MAINTAIN", 5)]
    _, diagnostics = governor.select(candidates, current, 2)
    assert all(c["semantic_bonus"] == 0 for c in diagnostics["candidates"])
    assert diagnostics["governor"]["candidate_generation_only"]


def test_current_lane_change_immediately_revokes_continuation_without_waiting_for_health_confirmation():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(), current, 2)
    changed = context(2.2, lane_key="1:0:2", candidates=[candidate("forward", 1)])
    result = governor.recovery_request(changed, 2.2)
    assert not result["authorized"] and result["reason"] == "recovery_lane_context_changed"


@pytest.mark.parametrize("changes", [
    {"signals": [signal()]}, {"baseline_stop_expected": True}, {"blocked": True},
    {"actors": [actor()]},
    {"actors": [actor(conflict=False, ttc_s=None, path_blocked=True, stationary=True)]},
])
def test_current_signal_queue_blockage_or_conflict_immediately_revokes_recovery(changes):
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(), current, 2)
    result = governor.recovery_request(context(2.2, **changes), 2.2)
    assert not result["authorized"]
    assert not governor.recovery_request(context(2.4), 2.4)["authorized"]


@pytest.mark.parametrize("changes", [
    {"signals": [signal(stop_line_distance_m=60)]},
    {"signals": [signal(stop_line_distance_m=-1)]},
    {"signals": [signal(applicable=False)]},
    {"actors": [actor(conflict=False, ttc_s=None, relative_l_m=3.5)]},
])
def test_far_or_passed_lights_and_non_conflicting_adjacent_traffic_do_not_block_recovery(changes):
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    assert governor.submit(recovery_advice(), current, 2)["accepted"]
    assert governor.recovery_request(context(2.2, **changes), 2.2)["authorized"]


@pytest.mark.parametrize("field,value", [("lane_key", "1:0:2"),
    ("navigation", "LEFT"), ("topology_signature", "junction2")])
def test_changed_route_or_lane_cannot_inherit_a_previous_failure_interval(field, value):
    governor = SelectiveSemanticGovernor()
    observe_failure(governor)
    state = governor.observe_planner_health(context(2.2, **{field: value}), 2.2)
    assert not state["sustained"] and state["observation_count"] == 1


@pytest.mark.parametrize("change", [lambda x: x.pop("planner_health"),
    lambda x: x["planner_health"].update(source="expanded_pool"),
    lambda x: x["planner_health"].update(positive_progressing_count=-1),
    lambda x: x["planner_health"].update(no_positive_progressing_candidate=False),
    lambda x: x["planner_health"].update(positive_progressing_count=True)])
def test_missing_or_inconsistent_native_health_fails_closed_for_generation_authority(change):
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    change(current)
    assert not governor.submit(recovery_advice(), current, 2)["accepted"]


@pytest.mark.parametrize("changes", [
    {"planner_request": "invalid"}, {"reason_code": "route_change"},
    {"planner_request": "none"}, {"target_actor_ids": [7]},
    {"target_stop_line_id": "stop_9_0"}, {"activation": "while_conflict"},
    {"release": "maneuver_complete"}, {"needed": False},
])
def test_request_contract_does_not_borrow_unrelated_action_authority(changes):
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    assert not governor.submit(recovery_advice(**changes), current, 2)["accepted"]


@pytest.mark.parametrize("field,value", [("lateral_decision", "CHANGE_LANE_LEFT"),
    ("longitudinal_decision", "STOP"), ("longitudinal_decision", "YIELD"),
    ("longitudinal_decision", "DECELERATE")])
def test_generation_request_only_supports_route_following_forward_semantics(field, value):
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    advice = recovery_advice()
    advice[field] = value
    assert not governor.submit(advice, current, 2)["accepted"]


@pytest.mark.parametrize("request_field", ["omitted", "none"])
def test_original_seven_field_interventions_remain_compatible_without_expansion(request_field):
    governor = SelectiveSemanticGovernor()
    advice = recovery_advice(reason_code="visibility", planner_request="none")
    if request_field == "omitted":
        advice["intervention"].pop("planner_request")
    current = context(special_situations=["limited_visibility_fog"])
    original = deepcopy(advice)
    assert governor.submit(advice, current, 0)["accepted"]
    assert not governor.recovery_request(current, 0)["authorized"]
    assert advice == original


@pytest.mark.parametrize("field,value", [("planner_request", "relax_road_constraints"),
    ("planner_request", True), ("force_recovery", True), ("recovery_trajectory", [[1, 2]])])
def test_unknown_request_or_extra_model_control_fields_are_rejected(field, value):
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    advice = recovery_advice()
    advice["intervention"][field] = value
    assert not governor.submit(advice, current, 2)["accepted"]


def test_degraded_baseline_escalates_an_existing_visibility_lease_after_two_seconds():
    governor = SelectiveSemanticGovernor()
    advice = recovery_advice(planner_request="none", reason_code="visibility", valid_for_s=6)
    first = context(special_situations=["limited_visibility_fog"])
    assert governor.query_decision(first, degenerate_pool(), 0)["query"]
    assert governor.submit(advice, first, 0)["accepted"]
    for now in (.5, 1, 1.5):
        query = governor.query_decision(context(now, special_situations=["limited_visibility_fog"]),
                                        degenerate_pool(), now)
        assert query["reason"] == "active_guidance_cached"
    query = governor.query_decision(context(2, special_situations=["limited_visibility_fog"]),
                                    degenerate_pool(), 2)
    assert query["query"] and query["reason"] == "sustained_native_planner_degradation"


def test_simulator_restart_clears_recovery_history_and_permission():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(), current, 2)
    assert not governor.recovery_request(context(), 0)["authorized"]


def test_invalid_context_permanently_discards_the_old_lease_instead_of_resurrecting_it():
    governor = SelectiveSemanticGovernor()
    current = observe_failure(governor)
    governor.submit(recovery_advice(valid_for_s=6), current, 2)
    stale = context(.5)
    assert governor.recovery_request(stale, 2.2)["reason"] == "stale_context"
    for now in (2.4, 2.9, 3.4, 3.9, 4.4):
        governor.observe_planner_health(context(now), now)
    assert governor.observe_planner_health(context(4.4), 4.4)["sustained"]
    assert not governor.recovery_request(context(4.4), 4.4)["authorized"]


@pytest.mark.parametrize("config", [{"recovery_degraded_duration_s": 0},
    {"recovery_min_observations": 1}, {"recovery_min_observations": 3.5},
    {"recovery_min_observations": True}, {"recovery_release_healthy_duration_s": 0},
    {"recovery_release_progress_m": -1}])
def test_invalid_recovery_temporal_configurations_are_rejected(config):
    with pytest.raises(ValueError):
        SelectiveSemanticGovernor(config)
