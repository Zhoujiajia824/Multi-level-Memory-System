"""Grounding, bounded influence, expiry and query scheduling contracts."""
from copy import deepcopy
import json
import math

import pytest

from carla_bridge.planning.selection import choose_candidate
from carla_bridge.planning.semantic_governor import SelectiveSemanticGovernor


def candidate(identity, score, longitudinal="MAINTAIN", progress=20.0):
    return {"id": identity, "base_score": score, "longitudinal": longitudinal,
            "lateral_semantics": ["FOLLOW_ROUTE"], "forward_progress_m": progress,
            "safety": {"finite": True, "no_collision": 1.0, "drivable_area": 1.0,
                       "driving_direction": 1.0, "ttc": 1.0}}


def pool(stop_score=0.96):
    return [candidate("moving", 0.99), candidate("stop", stop_score, "STOP", 0.0)]


def context(now=0.0, **changes):
    result = {"schema_version": 1, "sim_time_s": now, "ego_speed_mps": 2.0,
              "route_progress_m": 10.0, "in_junction": False, "lane_key": "1:0:1",
              "navigation": "STRAIGHT", "speed_limit_mps": 10.0, "signals": [],
              "actors": [], "blocked": False, "special_situations": [],
              "topology_signature": "road1", "baseline_stop_expected": False}
    result.update(changes)
    return result


def actor(**changes):
    result = {"id": 7, "distance_m": 5.0, "relative_s_m": 4.0, "relative_l_m": 1.0,
              "same_lane": False, "conflict": True, "ttc_s": 1.0,
              "ahead": True, "type": "vehicle.test", "path_blocked": False,
              "stationary": False, "route_distance_m": 4.0, "speed_mps": 2.0}
    result.update(changes)
    return result


def blocked_actor(**changes):
    values = {"same_lane": True, "conflict": False, "ttc_s": None,
              "path_blocked": True, "stationary": True, "speed_mps": 0.0}
    values.update(changes)
    return actor(**values)


def semantic(longitudinal="STOP", **changes):
    intervention = {"needed": True, "reason_code": "interaction", "target_actor_ids": [7],
                    "target_stop_line_id": None, "activation": "while_conflict",
                    "release": "condition_cleared", "valid_for_s": 2.0}
    intervention.update(changes)
    return {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": longitudinal,
            "behavior_reason": "A predicted crossing conflicts with our route.",
            "risk_level": "MEDIUM", "safety_notes": [], "intervention": intervention}


def test_simple_cruising_skips_query_and_preserves_native_selection():
    governor = SelectiveSemanticGovernor()
    request = governor.query_decision(context(), pool(), 0.0)
    assert request["query"] is False and request["reason"] == "simple_cruising"
    selected, audit = governor.select(pool(), context(), 0.0)
    assert selected["id"] == audit["baseline_selected_id"] == "moving"
    assert all(item["semantic_bonus"] == 0.0 for item in audit["candidates"])


def test_grounded_adoption_changes_preference_without_changing_candidate_pool():
    governor = SelectiveSemanticGovernor()
    candidates = pool()
    original = deepcopy(candidates)
    scene = context(actors=[actor()])
    advice = semantic()
    assert governor.submit(advice, scene, 0.0)["accepted"]
    selected, audit = governor.select(candidates, scene, 0.0)
    _, baseline = choose_candidate(candidates, {}, False)
    assert selected["id"] == "stop"
    assert audit["governor"]["adopted"] is True
    assert audit["governor"]["base_regret"] == pytest.approx(0.03)
    assert audit["candidate_pool_sha256"] == baseline["candidate_pool_sha256"]
    assert audit["governor"]["legacy_candidate_pool_sha256"] == baseline["candidate_pool_sha256"]
    assert [item["base_score"] for item in audit["candidates"]] == [0.99, 0.96]
    assert candidates == original
    assert advice == semantic()
    assert audit["safety_gate_applied"] is False
    assert audit["governor"]["global_pdm_candidate_gate_applied"] is False
    json.dumps(audit, allow_nan=False)


def test_regret_budget_withholds_bonus_but_never_excludes_native_candidates():
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    selected, audit = governor.select(pool(stop_score=0.90), scene, 0)
    assert selected["id"] == "moving"
    stopped = audit["candidates"][1]
    assert stopped["eligible"] and stopped["semantic_bonus"] == 0
    assert stopped["semantic_bonus_withheld_by_regret"]
    assert audit["governor"]["legacy_selected_id"] == "stop"


def test_large_configured_semantic_weights_are_still_capped():
    governor = SelectiveSemanticGovernor({"lateral_weight": 2.0, "longitudinal_weight": 2.0,
                                           "max_semantic_bonus": 0.05})
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    _, audit = governor.select(pool(), scene, 0)
    assert max(item["semantic_bonus"] for item in audit["candidates"]) <= 0.05


@pytest.mark.parametrize("longitudinal", ["STOP", "YIELD"])
def test_oncoming_neighbor_alone_cannot_authorize_stop_or_yield(longitudinal):
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor(conflict=False, ttc_s=None, relative_l_m=3.5)])
    submission = governor.submit(semantic(longitudinal), scene, 0)
    assert submission["accepted"] is False
    selected, audit = governor.select(pool(), scene, 0)
    assert selected["id"] == "moving"
    assert audit["governor"]["authorized"] is False


def test_distant_same_lane_vehicle_does_not_authorize_full_stop():
    governor = SelectiveSemanticGovernor()
    scene = context(blocked=True, actors=[blocked_actor(distance_m=29, relative_s_m=29,
                                                       route_distance_m=29)])
    submission = governor.submit(semantic(reason_code="path_blocked", activation="immediate"), scene, 0)
    assert submission["reason"] == "stop_or_yield_without_local_evidence"


@pytest.mark.parametrize("stop_policy", ["STOP", "YIELD"])
def test_lateral_match_cannot_indirectly_reward_unjustified_stop(stop_policy):
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor(distance_m=30, ttc_s=4)])
    advice = semantic("ACCELERATE")
    assert governor.submit(advice, scene, 0)["accepted"]
    candidates = [candidate("native", 0.99), candidate("incidental_stop", 0.97, stop_policy, 0)]
    candidates[0]["lateral_semantics"] = ["KEEP_LANE"]
    assert choose_candidate(candidates, advice, True)[0]["id"] == "incidental_stop"
    selected, audit = governor.select(candidates, scene, 0)
    assert selected["id"] == "native"
    assert audit["candidates"][1]["semantic_stop_bonus_withheld"]
    assert audit["candidates"][1]["eligible"]


def test_native_stop_is_preserved_even_without_any_semantic_stop_authority():
    candidates = [candidate("native_stop", 1.0, "STOP", 0), candidate("moving", 0.9)]
    selected, audit = SelectiveSemanticGovernor().select(candidates, context(), 0)
    assert selected["id"] == "native_stop"
    assert audit["governor"]["authorized"] is False


def test_local_blocked_lead_actor_is_evidence_but_long_explained_queue_does_not_query():
    governor = SelectiveSemanticGovernor()
    scene = context(blocked=True, baseline_stop_expected=True, ego_speed_mps=0.0,
                    actors=[blocked_actor()])
    assert governor.submit(semantic(reason_code="path_blocked", activation="immediate"), scene, 0)["accepted"]
    scene["sim_time_s"] = 20.0
    request = governor.query_decision(scene, pool(), 20.0)
    assert request["query"] is False
    assert request["reason"] == "expected_signal_or_queue_wait"
    assert request["legitimate_wait"] is True
    assert request["unexpected_stall"] is False


def test_far_red_signal_cannot_turn_global_stop_on_and_near_signal_is_scoped():
    governor = SelectiveSemanticGovernor()
    advice = semantic(reason_code="traffic_signal", target_actor_ids=[],
                      target_stop_line_id="light:17", activation="approach_stop_line",
                      release="signal_changed")
    scene = context(baseline_stop_expected=True, signals=[
        {"id": "light:17", "state": "RED", "stop_line_distance_m": 60.0, "applicable": True}])
    rejected = governor.submit(advice, scene, 0)
    assert rejected["reason"] == "stop_line_not_within_activation_range"
    assert governor.query_decision(scene, pool(), 0)["query"] is False
    scene["signals"][0]["stop_line_distance_m"] = 5.0
    assert governor.submit(advice, scene, 0)["accepted"]
    scene["sim_time_s"] = 0.2
    scene["signals"][0]["state"] = "GREEN"
    selected, audit = governor.select(pool(), scene, 0.2)
    assert selected["id"] == "moving"
    assert audit["governor"]["reason"] == "signal_changed"


@pytest.mark.parametrize("signal_changes", [{"applicable": False}, {"id": "other"},
                                            {"stop_line_distance_m": -0.1}])
def test_unrelated_or_passed_signal_never_authorizes_stop(signal_changes):
    governor = SelectiveSemanticGovernor()
    signal = {"id": "light:17", "state": "RED", "stop_line_distance_m": 2.0, "applicable": True}
    signal.update(signal_changes)
    advice = semantic(reason_code="traffic_signal", target_actor_ids=[],
                      target_stop_line_id="light:17", activation="approach_stop_line")
    assert not governor.submit(advice, context(signals=[signal]), 0)["accepted"]


def test_expiry_does_not_renew_during_repeated_planning():
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(valid_for_s=0.5), scene, 0)
    for now in (0.1, 0.3, 0.499):
        scene["sim_time_s"] = now
        selected, audit = governor.select(pool(), scene, now)
        assert selected["id"] == "stop"
        assert audit["governor"]["expires_at_s"] == 0.5
    scene["sim_time_s"] = 0.5
    selected, audit = governor.select(pool(), scene, 0.5)
    assert selected["id"] == "moving"
    assert audit["governor"]["reason"] == "intervention_expired"


def test_snapshot_time_bounds_lease_instead_of_model_return_time():
    governor = SelectiveSemanticGovernor()
    assert governor.submit(semantic(valid_for_s=0.5), context(actors=[actor()]), 0.4)["accepted"]
    scene = context(0.51, actors=[actor()])
    _, audit = governor.select(pool(), scene, 0.51)
    assert audit["governor"]["reason"] == "intervention_expired"


@pytest.mark.parametrize("changes,reason", [
    ({"actors": []}, "target_actor_missing"),
    ({"actors": [actor(conflict=False, ttc_s=None)]}, "no_current_interaction_conflict"),
    ({"actors": [actor(distance_m=61.0)]}, "target_actor_outside_scope"),
    ({"topology_signature": "other"}, "navigation_context_changed"),
    ({"navigation": "LEFT"}, "navigation_context_changed"),
])
def test_changed_evidence_permanently_drops_cached_guidance(changes, reason):
    governor = SelectiveSemanticGovernor()
    original = context(actors=[actor()])
    governor.submit(semantic(), original, 0)
    scene = context(0.2, actors=[actor()])
    scene.update(changes)
    selected, audit = governor.select(pool(), scene, 0.2)
    assert selected["id"] == "moving"
    assert audit["governor"]["reason"] == reason
    original["sim_time_s"] = 0.3
    selected, audit = governor.select(pool(), original, 0.3)
    assert selected["id"] == "moving"
    assert audit["governor"]["reason"] == "no_active_intervention"


def test_lane_change_completion_drops_preference_without_resetting_planner_state():
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic("MAINTAIN", release="maneuver_complete"), scene, 0)
    scene.update(lane_key="1:0:2", sim_time_s=0.2)
    _, audit = governor.select(pool(), scene, 0.2)
    assert audit["governor"]["reason"] == "maneuver_completed"
    assert audit["governor"]["numerical_planner_continuity_reset"] is False


def test_legacy_schema_and_model_abstention_clear_prior_authority():
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    legacy = semantic()
    del legacy["intervention"]
    result = governor.submit(legacy, scene, 0)
    assert result["reason"] == "legacy_schema_has_no_intervention_authority"
    assert governor.select(pool(), scene, 0)[0]["id"] == "moving"
    result = governor.submit(semantic(needed=False, reason_code="none", target_actor_ids=[]), scene, 0)
    assert result["accepted"] and result["active"] is False


def test_disabled_arm_never_changes_native_winner_or_reactivates_old_guidance():
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    selected, audit = governor.select(pool(), scene, 0, enabled=False)
    assert selected["id"] == "moving"
    assert audit["governor"]["reason"] == "semantic_influence_disabled"
    assert governor.select(pool(), scene, 0.1)[0]["id"] == "moving"


def test_known_native_metric_failures_are_not_restored_as_candidate_gates():
    governor = SelectiveSemanticGovernor()
    candidates = pool()
    candidates[0]["safety"].update(no_collision=0, drivable_area=0, driving_direction=0, ttc=0)
    selected, audit = governor.select(candidates, context(), 0)
    assert selected["id"] == "moving" and selected["eligible"]
    assert not selected["safe"]
    assert audit["safety_gate_applied"] is False


def test_no_finite_pool_stays_empty_and_diagnostics_are_json_serializable():
    governor = SelectiveSemanticGovernor()
    candidates = [candidate("nan", math.nan)]
    selected, audit = governor.select(candidates, context(), 0)
    assert selected is None and audit["selected_id"] is None
    assert governor.query_decision(context(), candidates, 0)["reason"] == "no_finite_pdm_proposal"
    json.dumps(audit, allow_nan=False)


def test_query_trigger_has_cooldown_and_does_not_interfere_with_native_choice():
    governor = SelectiveSemanticGovernor()
    scene = context(special_situations=["construction_officer"])
    assert governor.query_decision(scene, pool(), 0)["query"]
    scene["sim_time_s"] = 0.2
    result = governor.query_decision(scene, pool(), 0.2)
    assert not result["query"] and result["trigger_reason"] == "special_situation"
    assert governor.select(pool(), scene, 0.2)[0]["id"] == "moving"
    scene["sim_time_s"] = 2.0
    assert governor.query_decision(scene, pool(), 2.0)["query"]


def test_stall_is_measured_from_observed_progress_and_excludes_legal_waiting():
    governor = SelectiveSemanticGovernor()
    governor.query_decision(context(ego_speed_mps=0), pool(), 0)
    request = governor.query_decision(context(5.1, ego_speed_mps=0), pool(), 5.1)
    assert request["reason"] == "unexplained_stall" and request["query"]
    waiting = SelectiveSemanticGovernor()
    waiting.query_decision(context(ego_speed_mps=0, baseline_stop_expected=True), pool(), 0)
    request = waiting.query_decision(context(20, ego_speed_mps=0, baseline_stop_expected=True), pool(), 20)
    assert request["query"] is False and request["unexpected_stall"] is False
    assert request["reason"] == "expected_signal_or_queue_wait"


def test_unexpected_obstacle_is_not_misclassified_as_a_legal_queue_wait():
    governor = SelectiveSemanticGovernor()
    scene = context(blocked=True, baseline_stop_expected=False, ego_speed_mps=0,
                    actors=[blocked_actor(type="static")])
    request = governor.query_decision(scene, pool(), 0)
    assert request["query"] and request["reason"] == "path_blocked"
    assert not request["legitimate_wait"]


def test_static_blockage_can_authorize_existing_lane_change_without_requesting_stop():
    governor = SelectiveSemanticGovernor({"lateral_weight": 0.08, "longitudinal_weight": 0})
    scene = context(blocked=True, baseline_stop_expected=False,
                    actors=[blocked_actor(type="static")])
    advice = semantic("MAINTAIN", reason_code="path_blocked", activation="immediate",
                      release="maneuver_complete")
    advice["lateral_decision"] = "CHANGE_LANE_LEFT"
    assert governor.submit(advice, scene, 0)["accepted"]
    candidates = pool()
    candidates.append(candidate("left", 0.96))
    candidates[-1]["lateral_semantics"] = ["CHANGE_LANE_LEFT"]
    selected, audit = governor.select(candidates, scene, 0)
    assert selected["id"] == "left" and audit["governor"]["adopted"]


def test_memory_failure_can_request_consultation_but_cannot_authorize_stop():
    governor = SelectiveSemanticGovernor()
    memory = {"risk_feedback": {"known_failure": True, "verified": True,
                                "support_count": 3, "scope": "scene"},
              "causal_advantage": {"status": "supported", "lower_bound": 0.1, "support_count": 100}}
    request = governor.query_decision(context(), pool(), 0, memory)
    assert request["query"] and request["reason"] == "verified_memory_failure"
    assert request["memory"]["positive_causal_prior"]
    scene = context(special_situations=["officer"])
    submission = governor.submit(semantic(reason_code="special_rule", target_actor_ids=[],
                                            activation="immediate"), scene, 0)
    assert submission["reason"] == "stop_or_yield_without_local_evidence"
    selected, audit = governor.select(pool(), scene, 0, memory_evidence=memory)
    assert selected["id"] == "moving"
    assert audit["governor"]["memory"]["memory_authorizes_stop"] is False


def test_unknown_or_unverified_memory_is_not_a_query_trigger():
    memory = {"risk_feedback": {"known_failure": "true", "verified": True,
                                "support_count": 100, "scope": "scene"},
              "causal_advantage": {"status": "supported", "lower_bound": None, "support_count": 100}}
    request = SelectiveSemanticGovernor().query_decision(context(), pool(), 0, memory)
    assert request["query"] is False
    assert not request["memory"]["known_failure"]
    assert not request["memory"]["positive_causal_prior"]


@pytest.mark.parametrize("changes", [{"signals": "red"}, {"actors": [actor(ttc_s=math.nan)]},
                                     {"baseline_stop_expected": "false"}, {"schema_version": True}])
def test_invalid_context_drops_preferences_but_preserves_native_pdm(changes):
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    scene.update(changes)
    assert governor.query_decision(scene, pool(), 0)["query"] is False
    selected, audit = governor.select(pool(), scene, 0)
    assert selected["id"] == "moving" and not audit["governor"]["authorized"]


def test_stale_snapshot_is_not_refreshed_by_submission_time_and_episode_reset_drops_lease():
    governor = SelectiveSemanticGovernor()
    assert governor.submit(semantic(), context(actors=[actor()]), 2)["reason"] == "stale_context"
    scene = context(10, actors=[actor()])
    governor.submit(semantic(valid_for_s=6), scene, 10)
    _, audit = governor.select(pool(), context(0, actors=[actor()]), 0)
    assert not audit["governor"]["authorized"]


@pytest.mark.parametrize("config", [{"typo": 1}, {"base_regret_budget": True},
    {"max_semantic_bonus": math.nan}, {"lateral_weight": math.inf}, {"min_query_interval_s": 0},
    {"low_quality_threshold": 1.1}, {"max_target_distance_m": 1}, {"stall_duration_s": -1},
    {"longitudinal_weight": "0.1"}, {1: 0.2}, {"wait_review_interval_s": 0},
    {"wait_review_interval_s": True}, {"wait_review_interval_s": math.inf}])
def test_invalid_configuration_fails_loudly(config):
    with pytest.raises(ValueError):
        SelectiveSemanticGovernor(config)


@pytest.mark.parametrize("field,value", [("needed", "true"), ("valid_for_s", 6.1),
    ("valid_for_s", True), ("valid_for_s", math.nan), ("target_actor_ids", [True]),
    ("target_actor_ids", [7, 7]), ("target_actor_ids", list(range(9))),
    ("target_stop_line_id", " "), ("reason_code", []), ("activation", "forever")])
def test_invalid_intervention_never_leaks_prior_authority(field, value):
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    advice = semantic()
    advice["intervention"][field] = value
    assert governor.submit(advice, scene, 0)["accepted"] is False
    assert governor.select(pool(), scene, 0)[0]["id"] == "moving"


def test_direct_abstention_schema_cannot_retain_target_authority():
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    result = governor.submit(semantic(needed=False, reason_code="none"), scene, 0)
    assert not result["accepted"] and result["reason"] == "unneeded_intervention_has_targets"


def test_selection_override_preserves_default_behavior_and_off_arm():
    candidates = pool()
    advice = semantic()
    assert choose_candidate(candidates, advice, True)[0]["id"] == "stop"
    selected, audit = choose_candidate(candidates, advice, True,
                                       semantic_bonus_by_id={"moving": 0, "stop": 0.01})
    assert selected["id"] == "moving"
    off, off_audit = choose_candidate(candidates, advice, False,
                                      semantic_bonus_by_id={"stop": 2.0})
    assert off["id"] == "moving"
    assert audit["candidate_pool_sha256"] == off_audit["candidate_pool_sha256"]


@pytest.mark.parametrize("overrides", [[], {"missing": 0.1}, {"stop": math.nan},
                                        {"stop": -1}, {"stop": True}, {"stop": 4.1}])
def test_invalid_bonus_overrides_fail_instead_of_silently_changing_selection(overrides):
    with pytest.raises(ValueError):
        choose_candidate(pool(), {}, True, semantic_bonus_by_id=overrides)


def test_unchanged_authorized_lease_reuses_consultation_without_renewing_expiry():
    governor = SelectiveSemanticGovernor()
    scene = context(special_situations=["limited_visibility_fog"])
    assert governor.query_decision(scene, pool(), 0)["query"]
    advice = semantic("MAINTAIN", reason_code="visibility", target_actor_ids=[],
                      activation="immediate", valid_for_s=6)
    assert governor.submit(advice, scene, 0)["accepted"]
    for now in (2.0, 4.0, 5.9):
        scene["sim_time_s"] = now
        query = governor.query_decision(scene, pool(), now)
        assert not query["query"] and query["reason"] == "active_guidance_cached"
        assert query["expires_at_s"] == 6.0
    scene["sim_time_s"] = 6.0
    query = governor.query_decision(scene, pool(), 6)
    assert query["query"] and query["reason"] == "special_situation"
    assert query["lease_drop_reason"] == "intervention_expired"


def test_continuous_target_motion_inside_same_activation_band_does_not_invalidate_cache():
    governor = SelectiveSemanticGovernor()
    original = context(actors=[actor()])
    assert governor.submit(semantic(valid_for_s=6), original, 0)["accepted"]
    current = context(2, actors=[actor(distance_m=4.5, relative_s_m=3.5, ttc_s=0.8)])
    query = governor.query_decision(current, pool(), 2)
    assert not query["query"] and query["reason"] == "active_guidance_cached"


def test_new_conflict_during_authorized_lease_allows_a_new_consultation():
    governor = SelectiveSemanticGovernor()
    original = context(actors=[actor()])
    assert governor.submit(semantic(valid_for_s=6), original, 0)["accepted"]
    current = context(2, actors=[actor(), actor(id=8, distance_m=12, ttc_s=3)])
    query = governor.query_decision(current, pool(), 2)
    assert query["query"] and query["reason"] == "advisory_conditions_changed"


@pytest.mark.parametrize("changes", [{"ttc_s": 0.5}, {"distance_m": 3.0}])
def test_existing_nontarget_conflict_crossing_urgency_band_invalidates_query_cache(changes):
    governor = SelectiveSemanticGovernor()
    original = context(actors=[actor(), actor(id=8, distance_m=15, ttc_s=4.8)])
    assert governor.submit(semantic(valid_for_s=6), original, 0)["accepted"]
    other = actor(id=8, distance_m=15, ttc_s=4.8)
    other.update(changes)
    current = context(2, actors=[actor(), other])
    query = governor.query_decision(current, pool(), 2)
    assert query["query"] and query["reason"] == "advisory_conditions_changed"


def test_existing_nontarget_motion_inside_same_urgency_band_keeps_query_cache():
    governor = SelectiveSemanticGovernor()
    original = context(actors=[actor(), actor(id=8, distance_m=15, ttc_s=4.8)])
    governor.submit(semantic(valid_for_s=6), original, 0)
    current = context(2, actors=[actor(), actor(id=8, distance_m=12, ttc_s=3.0)])
    query = governor.query_decision(current, pool(), 2)
    assert not query["query"] and query["reason"] == "active_guidance_cached"


def test_changed_conditions_still_respect_minimum_query_interval():
    governor = SelectiveSemanticGovernor()
    original = context(actors=[actor()], special_situations=["fog"])
    governor.query_decision(original, pool(), 0)
    governor.submit(semantic(valid_for_s=6), original, 0)
    current = context(0.5, actors=[actor(), actor(id=8)], special_situations=["fog"])
    result = governor.query_decision(current, pool(), 0.5)
    assert not result["query"] and result["reason"] == "query_cooldown"
    current["sim_time_s"] = 2
    assert governor.query_decision(current, pool(), 2)["query"]


@pytest.mark.parametrize("rejection", ["bad_schema", "abstain", "no_evidence"])
def test_rejected_or_abstained_advice_never_suppresses_future_consultations(rejection):
    governor = SelectiveSemanticGovernor()
    scene = context(special_situations=["fog"])
    governor.query_decision(scene, pool(), 0)
    advice = semantic(reason_code="visibility", target_actor_ids=[], activation="immediate",
                      valid_for_s=6)
    if rejection == "bad_schema":
        del advice["intervention"]
    elif rejection == "abstain":
        advice["intervention"].update(needed=False, reason_code="none")
    result = governor.submit(advice, scene, 0)
    assert not result.get("active")
    scene["sim_time_s"] = 2
    result = governor.query_decision(scene, pool(), 2)
    assert result["query"] and result["reason"] == "special_situation"


def test_cleared_target_drops_cache_and_does_not_block_an_independent_trigger():
    governor = SelectiveSemanticGovernor()
    governor.submit(semantic(valid_for_s=6), context(actors=[actor()]), 0)
    current = context(2, special_situations=["fog"])
    query = governor.query_decision(current, pool(), 2)
    assert query["query"] and query["lease_drop_reason"] == "target_actor_missing"


def test_blockage_belonging_to_another_actor_cannot_authorize_stop():
    governor = SelectiveSemanticGovernor()
    moving_lead = actor(same_lane=True, conflict=False, ttc_s=None)
    obstacle = blocked_actor(id=9, distance_m=25, relative_s_m=25, route_distance_m=25)
    scene = context(blocked=True, actors=[moving_lead, obstacle])
    result = governor.submit(semantic(reason_code="path_blocked", activation="immediate"), scene, 0)
    assert not result["accepted"] and result["reason"] == "path_blockage_not_grounded"
    assert governor.select(pool(), scene, 0)[0]["id"] == "moving"


@pytest.mark.parametrize("changes", [{"path_blocked": False}, {"stationary": False},
                                     {"route_distance_m": None}, {"route_distance_m": 0},
                                     {"route_distance_m": 31}])
def test_target_requires_its_own_stationary_route_corridor_evidence(changes):
    scene = context(blocked=True, actors=[blocked_actor(**changes)])
    result = SelectiveSemanticGovernor().submit(
        semantic(reason_code="path_blocked", activation="immediate"), scene, 0)
    assert not result["accepted"]


def test_same_lane_label_and_global_blockage_do_not_allow_indirect_stop_bonus():
    governor = SelectiveSemanticGovernor()
    moving_lead = actor(same_lane=True, conflict=False, ttc_s=None)
    obstacle = blocked_actor(id=9, distance_m=25, relative_s_m=25, route_distance_m=25)
    scene = context(blocked=True, special_situations=["fog"], actors=[moving_lead, obstacle])
    advice = semantic("MAINTAIN", reason_code="visibility", activation="immediate")
    assert governor.submit(advice, scene, 0)["accepted"]
    candidates = pool()
    candidates[0]["lateral_semantics"] = ["KEEP_LANE"]
    selected, audit = governor.select(candidates, scene, 0)
    assert selected["id"] == "moving"
    assert audit["candidates"][1]["semantic_stop_bonus_withheld"]


def test_legal_red_wait_does_not_consult_periodically_or_release_the_signal():
    governor = SelectiveSemanticGovernor({"wait_review_interval_s": 15})
    signal = {"id": "red", "state": "Red", "stop_line_distance_m": 2, "applicable": True}
    scene = context(ego_speed_mps=0, baseline_stop_expected=True, signals=[signal])
    assert not governor.query_decision(scene, pool(), 0)["query"]
    for now, expected in ((14.9, False), (15, False), (16, False), (29.9, False), (30, False)):
        scene["sim_time_s"] = now
        query = governor.query_decision(scene, pool(), now)
        assert query["query"] is expected
        assert query["legitimate_wait"] and not query["unexpected_stall"]
        assert scene["signals"][0]["state"] == "Red" and scene["baseline_stop_expected"]
        assert governor.select(pool(), scene, now)[0]["id"] == "moving"


def test_wait_review_timer_resets_after_progress_and_on_simulator_restart():
    governor = SelectiveSemanticGovernor({"wait_review_interval_s": 15})
    governor.query_decision(context(ego_speed_mps=0, baseline_stop_expected=True), pool(), 0)
    governor.query_decision(context(10, ego_speed_mps=2, baseline_stop_expected=True), pool(), 10)
    assert not governor.query_decision(context(15, ego_speed_mps=0, baseline_stop_expected=True), pool(), 15)["query"]
    assert not governor.query_decision(context(0, ego_speed_mps=0, baseline_stop_expected=True), pool(), 0)["query"]
    assert not governor.query_decision(context(14, ego_speed_mps=0, baseline_stop_expected=True), pool(), 14)["query"]


def positive_intervention_memory():
    return {"intervention_priors": [{"action": {"lateral_decision": "FOLLOW_ROUTE",
        "longitudinal_decision": "STOP"}, "causal_advantage": {"status": "supported",
        "lower_bound": 0.05, "support_count": 100},
        "use": "consultation_only_recheck_requested_action_before_adoption"}]}


def test_paired_positive_action_prior_invites_consultation_but_does_not_authorize_stop():
    governor = SelectiveSemanticGovernor()
    memory = positive_intervention_memory()
    scene = context()
    query = governor.query_decision(scene, pool(), 0, memory)
    assert query["query"] and query["reason"] == "memory_supported_intervention"
    assert query["memory"]["supported_intervention_prior_count"] == 1
    advice = semantic(reason_code="special_rule", target_actor_ids=[], activation="immediate")
    assert not governor.submit(advice, scene, 0)["accepted"]
    selected, audit = governor.select(pool(), scene, 0, memory_evidence=memory)
    assert selected["id"] == "moving" and audit["governor"]["base_regret_budget"] == 0.08
    assert not audit["governor"]["memory"]["memory_authorizes_stop"]
    scene["sim_time_s"] = 0.5
    assert not governor.query_decision(scene, pool(), 0.5, memory)["query"]


@pytest.mark.parametrize("changes", [{"status": "unknown"}, {"lower_bound": None},
    {"lower_bound": 0}, {"lower_bound": -0.1}, {"lower_bound": math.nan},
    {"lower_bound": math.inf}, {"lower_bound": True}, {"support_count": 0},
    {"support_count": True}])
def test_unknown_or_malformed_positive_intervention_evidence_does_not_trigger(changes):
    memory = positive_intervention_memory()
    memory["intervention_priors"][0]["causal_advantage"].update(changes)
    assert not SelectiveSemanticGovernor().query_decision(context(), pool(), 0, memory)["query"]


def test_positive_intervention_prior_respects_an_unchanged_authorized_lease():
    governor = SelectiveSemanticGovernor()
    original = context(actors=[actor()])
    governor.submit(semantic(valid_for_s=6), original, 0)
    query = governor.query_decision(context(2, actors=[actor()]), pool(), 2, positive_intervention_memory())
    assert not query["query"] and query["reason"] == "active_guidance_cached"


def negative_action_memory(**changes):
    item = {"action": {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "STOP"},
            "known_failure": True, "verified": True, "support_count": 3, "score_residual": -0.05}
    item.update(changes)
    return {"action_feedback": [item]}


def test_verified_action_failure_discounts_only_matching_semantic_bonus():
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    selected, audit = governor.select(pool(), scene, 0, memory_evidence=negative_action_memory())
    assert selected["id"] == "moving"
    by_id = {c["id"]: c for c in audit["candidates"]}
    assert by_id["stop"]["memory_bonus_discount"] == pytest.approx(0.05)
    assert by_id["stop"]["semantic_bonus"] == pytest.approx(0.03)
    assert by_id["moving"]["memory_bonus_discount"] == 0
    assert by_id["stop"]["base_score"] == 0.96 and by_id["stop"]["eligible"]


@pytest.mark.parametrize("changes", [{"known_failure": False}, {"verified": False},
    {"verified": "true"}, {"support_count": 2}, {"support_count": True},
    {"score_residual": math.nan}, {"score_residual": -0.2}, {"score_residual": 0.1},
    {"score_residual": True}, {"action": {"lateral_decision": "KEEP_LANE", "longitudinal_decision": "STOP"}},
    {"action": {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "ACCELERATE"}}])
def test_unverified_insufficient_invalid_or_other_action_feedback_cannot_discount(changes):
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    selected, audit = governor.select(pool(), scene, 0, memory_evidence=negative_action_memory(**changes))
    assert selected["id"] == "stop"
    assert all(c["memory_bonus_discount"] == 0 for c in audit["candidates"])


def test_scene_failure_cannot_apply_a_blanket_penalty_or_reduce_native_scores():
    governor = SelectiveSemanticGovernor()
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    memory = {"score_residual": -0.15, "risk_feedback": {"known_failure": True,
        "verified": True, "support_count": 10, "negative_residual": -0.15, "scope": "scene"}}
    selected, audit = governor.select(pool(), scene, 0, memory_evidence=memory)
    assert selected["id"] == "stop"
    assert all(c["memory_bonus_discount"] == 0 for c in audit["candidates"])


def test_multiple_action_feedback_takes_maximum_discount_after_bonus_cap_and_clamps_zero():
    governor = SelectiveSemanticGovernor({"lateral_weight": 2, "longitudinal_weight": 2})
    scene = context(actors=[actor()])
    governor.submit(semantic(), scene, 0)
    memory = negative_action_memory(score_residual=-0.15)
    memory["action_feedback"] += negative_action_memory(score_residual=-0.02)["action_feedback"]
    _, audit = governor.select(pool(), scene, 0, memory_evidence=memory)
    stopped = audit["candidates"][1]
    assert stopped["semantic_bonus"] == 0
    assert stopped["memory_bonus_discount"] == pytest.approx(0.08)
    assert stopped["base_score"] == 0.96 and stopped["eligible"]


@pytest.mark.parametrize("kind", ["vehicle.test", "walker.pedestrian.0001"])
def test_first_time_urgent_crossing_invites_consultation_despite_high_native_score(kind):
    scene = context(actors=[actor(type=kind, same_lane=False, conflict=True, ttc_s=2.0)])
    governor = SelectiveSemanticGovernor()
    query = governor.query_decision(scene, pool(), 0)
    assert query["query"] and query["reason"] == "urgent_crossing_conflict"
    assert query["urgent_crossing_actor_ids"] == query["crossing_conflict_actor_ids"] == [7]
    # Consultation itself supplies no semantic authority or candidate gate.
    selected, audit = governor.select(pool(), scene, 0)
    assert selected["id"] == "moving" and not audit["governor"]["authorized"]
    assert all(c["eligible"] and c["semantic_bonus"] == 0 for c in audit["candidates"])


@pytest.mark.parametrize("other", [actor(conflict=False, ttc_s=1, relative_l_m=3.5),
    actor(same_lane=True, conflict=True, ttc_s=1), actor(type="static.prop.test", ttc_s=1),
    actor(conflict=True, ttc_s=2.01), actor(conflict=True, ttc_s=None)])
def test_neighboring_oncoming_regular_lead_and_single_nonurgent_actor_do_not_trigger(other):
    query = SelectiveSemanticGovernor().query_decision(context(actors=[other]), pool(), 0)
    assert not query["query"] and query["reason"] == "simple_cruising"


def test_two_distinct_crossing_conflicts_in_longer_window_invite_complex_consultation():
    scene = context(actors=[actor(ttc_s=3.5, distance_m=25),
                            actor(id=8, ttc_s=5.0, distance_m=35)])
    query = SelectiveSemanticGovernor().query_decision(scene, pool(), 0)
    assert query["query"] and query["reason"] == "complex_interaction"
    assert query["crossing_conflict_actor_ids"] == [7, 8]
    assert query["urgent_crossing_actor_ids"] == []


@pytest.mark.parametrize("second", [actor(id=8, ttc_s=5.01),
                                    actor(id=8, ttc_s=3.5, same_lane=True),
                                    actor(id=8, ttc_s=3.5, conflict=False)])
def test_complex_interaction_requires_two_current_crossing_conflicts_in_scope(second):
    query = SelectiveSemanticGovernor().query_decision(
        context(actors=[actor(ttc_s=3.5), second]), pool(), 0)
    assert not query["query"]


def test_explained_red_or_queue_wait_with_distant_crossing_traffic_never_gets_periodic_review():
    governor = SelectiveSemanticGovernor()
    scene = context(ego_speed_mps=0, baseline_stop_expected=True,
                    actors=[actor(ttc_s=1), actor(id=8, ttc_s=3)])
    for now in (0, 2, 4, 6, 14.9):
        scene["sim_time_s"] = now
        query = governor.query_decision(scene, pool(), now)
        assert not query["query"] and query["reason"] == "expected_signal_or_queue_wait"
    scene["sim_time_s"] = 15
    query = governor.query_decision(scene, pool(), 15)
    assert not query["query"] and query["reason"] == "expected_signal_or_queue_wait"
    assert query["legitimate_wait"] and not query["unexpected_stall"]


def test_new_crossing_during_authorized_queue_guidance_does_not_bypass_wait_review_interval():
    governor = SelectiveSemanticGovernor()
    original = context(ego_speed_mps=0, baseline_stop_expected=True, blocked=True,
                       actors=[blocked_actor()])
    advice = semantic("MAINTAIN", reason_code="path_blocked", activation="immediate", valid_for_s=6)
    assert governor.submit(advice, original, 0)["accepted"]
    current = context(2, ego_speed_mps=0, baseline_stop_expected=True, blocked=True,
                      actors=[blocked_actor(), actor(id=8, ttc_s=1)])
    query = governor.query_decision(current, pool(), 2)
    assert not query["query"] and query["reason"] == "expected_signal_or_queue_wait"


@pytest.mark.parametrize("special", [[], ["limited_visibility_fog"], ["construction_officer"]])
def test_500_ticks_red_queue_wait_never_dispatches_vlm_despite_old_review_timer_memory_or_lease_expiry(special):
    governor = SelectiveSemanticGovernor()
    signal = {"id": "red", "state": "Red", "stop_line_distance_m": 30, "applicable": True}
    lead = blocked_actor(routine_wait=True, blocking_kind="signal_queue")
    # A formerly useful piece of advice expires at six seconds. Neither expiry
    # nor 15-second wait-review multiples authorize another model consultation.
    initial = context(ego_speed_mps=0, baseline_stop_expected=True, signals=[signal],
                      special_situations=special, actors=[lead])
    advice = semantic("MAINTAIN", reason_code="traffic_signal", target_actor_ids=[],
                      target_stop_line_id="red", activation="immediate", valid_for_s=6)
    assert governor.submit(advice, initial, 0)["accepted"]
    for tick in range(500):
        now = tick*.1
        current = deepcopy(initial)
        current["sim_time_s"] = now
        request = governor.query_decision(current, [candidate("stop", 1, "STOP", 0)], now,
                                           positive_intervention_memory())
        assert not request["query"] and not request["eligible"]
        assert request["reason"] == "expected_signal_or_queue_wait"
        assert not request["wait_review_due"] and not request["unexpected_stall"]
        assert not governor.recovery_request(current, now)["authorized"]


def test_500_ticks_ordinary_slow_following_is_handled_without_model_and_blocked_static_lead_is_not():
    governor = SelectiveSemanticGovernor()
    lead = actor(same_lane=True, conflict=True, ttc_s=1, speed_mps=.4,
                 stationary=True, path_blocked=True, routine_wait=False)
    # The contextual routine-following flag explains creeping behind a moving
    # vehicle; it does not exempt an independently stationary obstacle.
    lead.update(stationary=False, path_blocked=False)
    for tick in range(500):
        now = tick*.1
        current = context(now, ego_speed_mps=.2, routine_following=True,
                          actors=[lead], special_situations=["limited_visibility_fog"])
        query = governor.query_decision(current, [candidate("slow", .1, "FOLLOW", .3)], now)
        assert not query["query"] and query["reason"] == "ordinary_same_lane_following"
    stopped = context(50, ego_speed_mps=0, blocked=True, routine_following=False,
                      actors=[blocked_actor()])
    assert governor.query_decision(stopped, pool(), 50)["reason"] == "path_blocked"
    assert governor.query_decision(context(52, **{k:v for k,v in stopped.items()
        if k != "sim_time_s"}), pool(), 52)["query"]


def test_red_to_green_still_stationary_single_lead_immediately_ends_routine_wait():
    governor = SelectiveSemanticGovernor()
    red = context(ego_speed_mps=0, baseline_stop_expected=True,
                  actors=[blocked_actor(routine_wait=True)])
    assert not governor.query_decision(red, pool(), 0)["query"]
    green_stuck = context(2, ego_speed_mps=0, blocked=True, baseline_stop_expected=False,
                          actors=[blocked_actor(routine_wait=False)])
    request = governor.query_decision(green_stuck, pool(), 2)
    assert request["query"] and request["reason"] == "path_blocked"
    green_moving = context(4, ego_speed_mps=1, routine_following=True,
        actors=[actor(same_lane=True, conflict=False, ttc_s=None, speed_mps=2)])
    request = governor.query_decision(green_moving, pool(), 4)
    assert not request["query"] and request["reason"] == "ordinary_same_lane_following"


def test_new_crossing_toward_actual_waiting_body_can_escalate_despite_red_and_cached_lease():
    governor = SelectiveSemanticGovernor()
    red = context(ego_speed_mps=0, baseline_stop_expected=True,
                  special_situations=["fog"], actors=[actor(occupied_ego_conflict=False)])
    assert not governor.query_decision(red, pool(), 0)["query"]
    dangerous = deepcopy(red)
    dangerous.update(sim_time_s=2)
    dangerous["actors"][0].update(occupied_ego_conflict=True, occupied_ego_conflict_ttc_s=.5)
    request = governor.query_decision(dangerous, pool(), 2)
    assert request["query"] and request["reason"] == "occupied_ego_crossing_conflict"
    assert request["waiting_hazard_actor_ids"] == [7]


def test_new_near_nonqueue_obstacle_during_red_wait_is_still_consultable():
    scene = context(ego_speed_mps=0, baseline_stop_expected=True, blocked=True,
        actors=[blocked_actor(type="static.prop.barrier", distance_m=5, routine_wait=False)])
    request = SelectiveSemanticGovernor().query_decision(scene, pool(), 0)
    assert request["query"] and request["reason"] == "path_blocked"


def test_semantic_regret_uses_full_numerical_planning_score_and_retains_raw_native_score():
    governor = SelectiveSemanticGovernor({"max_semantic_bonus": .2,
        "lateral_weight": .1, "longitudinal_weight": .1})
    candidates = [candidate("deviating", 1.0), candidate("center", .95)]
    candidates[0]["numerical_cost"] = .16
    candidates[1]["lateral_semantics"] = ["KEEP_LANE"]
    scene = context(actors=[actor()])
    assert governor.submit(semantic("MAINTAIN"), scene, 0)["accepted"]
    selected, audit = governor.select(candidates, scene, 0)
    assert selected["id"] == "center"
    deviating = audit["candidates"][0]
    assert deviating["base_score"] == 1 and deviating["planning_score"] == pytest.approx(.84)
    assert deviating["semantic_base_regret"] == pytest.approx(.11)
    assert deviating["semantic_bonus_withheld_by_regret"] and deviating["semantic_bonus"] == 0


def test_low_quality_query_compares_numerical_planning_score_without_rewriting_native_base():
    candidates = [candidate("sole", .5)]
    candidates[0]["numerical_cost"] = .16
    query = SelectiveSemanticGovernor().query_decision(context(), candidates, 0)
    assert query["query"] and query["reason"] == "low_pdm_quality"
    assert candidates[0]["base_score"] == .5
