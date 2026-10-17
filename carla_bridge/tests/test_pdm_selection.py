"""Score-only selection and semantic ablation contracts."""
from copy import deepcopy
import json
import math

import pytest

from carla_bridge.planning.selection import SELECTION_RULE, choose_candidate


def candidate(identity, score, longitudinal="CRUISE", lateral="FOLLOW_ROUTE"):
    return {
        "id": identity, "base_score": score, "longitudinal": longitudinal,
        "lateral_semantics": [lateral],
        "safety": {"no_collision": 1.0, "drivable_area": 1.0,
                   "driving_direction": 1.0, "ttc": 1.0,
                   "finite": True, "traffic_light": True},
    }


def pool():
    return [candidate("cruise", 0.9), candidate("stop", 0.6, "STOP"),
            candidate("left", 0.7, "DECELERATE", "CHANGE_LANE_LEFT")]


@pytest.mark.parametrize("lateral,longitudinal", [
    ("FOLLOW_ROUTE", "MAINTAIN"), ("CHANGE_LANE_LEFT", "DECELERATE"),
    ("CHANGE_LANE_RIGHT", "STOP"), ("KEEP_LANE", "YIELD"),
])
@pytest.mark.parametrize("weights", [(0.0, 0.0), (0.8, 0.8), (2.0, 0.0),
                                     (0.0, 2.0), (2.0, 2.0)])
def test_off_removes_all_semantic_score_terms(lateral, longitudinal, weights):
    candidates = pool()
    baseline, baseline_audit = choose_candidate(candidates, {}, False)
    selected, audit = choose_candidate(candidates, {
        "lateral_decision": lateral, "longitudinal_decision": longitudinal,
    }, False, *weights)
    assert selected["id"] == baseline["id"] == "cruise"
    assert audit["semantic_enabled"] is False
    assert audit["candidate_pool_sha256"] == baseline_audit["candidate_pool_sha256"]
    assert all(item["semantic_bonus"] == 0.0 for item in audit["candidates"])
    assert all(item["final_score"] == item["base_score"] for item in audit["candidates"])


def test_semantics_change_only_additive_score_not_candidate_pool_or_diagnostics():
    candidates = pool()
    original = deepcopy(candidates)
    semantic = {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "STOP"}
    baseline, off = choose_candidate(candidates, semantic, False)
    selected, on = choose_candidate(candidates, semantic, True)
    assert baseline["id"] == "cruise"
    assert selected["id"] == "stop"
    assert selected["semantic_bonus"] == pytest.approx(1.6)
    assert selected["final_score"] == pytest.approx(selected["base_score"] + 1.6)
    assert on["selection_rule"] == off["selection_rule"] == SELECTION_RULE
    assert on["candidate_pool_sha256"] == off["candidate_pool_sha256"]
    assert [c["base_score"] for c in on["candidates"]] == [c["base_score"] for c in original]
    assert [c["safety"] for c in on["candidates"]] == [c["safety"] for c in original]
    assert candidates == original
    assert semantic == {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "STOP"}


@pytest.mark.parametrize("failed_metric", [
    "no_collision", "drivable_area", "driving_direction", "ttc", "traffic_light",
])
@pytest.mark.parametrize("enabled", [True, False])
def test_failed_safety_diagnostics_do_not_exclude_the_highest_scored_candidate(failed_metric, enabled):
    safe = candidate("safe", 0.1)
    unsafe = candidate("diagnostic_failure", 1.0, "MAINTAIN", "CHANGE_LANE_LEFT")
    unsafe["safety"][failed_metric] = False if failed_metric == "traffic_light" else 0.0
    selected, audit = choose_candidate([safe, unsafe], {
        "lateral_decision": "CHANGE_LANE_LEFT", "longitudinal_decision": "MAINTAIN",
    }, enabled)
    assert selected["id"] == audit["baseline_selected_id"] == "diagnostic_failure"
    assert selected["safe"] is False and selected["eligible"] is True
    assert selected["final_score"] == pytest.approx(1.0 + (1.6 if enabled else 0))
    assert audit["safety_gate_applied"] is False
    assert audit["safety_is_diagnostic_only"] is True


def test_zero_base_score_can_win_from_semantics_without_a_hidden_safety_gate():
    failed = candidate("zero_base", 0.0, "ACCELERATE", "FOLLOW_ROUTE")
    failed["safety"].update(no_collision=0.0, drivable_area=0.0,
                            driving_direction=0.0, ttc=0.0, traffic_light=False)
    alternatives = [candidate("higher_base", 0.9, "STOP", "KEEP_LANE"), failed]
    selected, audit = choose_candidate(alternatives, {
        "lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "ACCELERATE",
    }, True)
    assert selected["id"] == "zero_base"
    assert selected["final_score"] == pytest.approx(1.6)
    assert audit["baseline_selected_id"] == "higher_base"


@pytest.mark.parametrize("value", [False, "false", "true", 1, 0, None])
def test_numerical_finite_flag_requires_the_actual_true_boolean(value):
    invalid = candidate("invalid", 1.0)
    invalid["safety"]["finite"] = value
    selected, audit = choose_candidate([invalid], {}, False)
    assert selected is None
    assert audit["candidates"][0]["eligible"] is False
    assert audit["candidates"][0]["final_score"] is None
    assert audit["candidates"][0]["invalid_numeric_reasons"] == ["non_finite_trajectory"]


@pytest.mark.parametrize("metric", [
    "no_collision", "drivable_area", "driving_direction", "ttc", "traffic_light",
])
def test_missing_optional_safety_diagnostics_do_not_affect_eligibility(metric):
    item = candidate("finite", 1.0)
    del item["safety"][metric]
    selected, _ = choose_candidate([item], {}, False)
    assert selected["id"] == "finite" and selected["eligible"]
    if metric == "traffic_light":
        # The planner no longer constructs this additional footprint check.
        assert selected["safe"]


def test_missing_finite_evidence_is_a_numerical_rejection():
    item = candidate("invalid", 1.0)
    del item["safety"]["finite"]
    selected, audit = choose_candidate([item], {}, False)
    assert selected is None
    assert audit["selected_id"] is None
    assert audit["available_lateral_semantics"] == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, True, "1"])
def test_invalid_base_score_is_rejected_without_breaking_pool_hash(value):
    invalid = candidate("invalid", value)
    valid = candidate("valid", 0.0)
    semantic = {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "CRUISE"}
    selected, off = choose_candidate([invalid, valid], semantic, False)
    _, on = choose_candidate([invalid, valid], semantic, True)
    assert selected["id"] == "valid"
    assert off["candidate_pool_sha256"] == on["candidate_pool_sha256"]
    assert off["candidates"][0]["final_score"] is None
    assert off["candidates"][0]["planning_score"] is None
    assert off["candidates"][0]["base_score"] is None
    assert off["candidates"][0]["invalid_numeric_reasons"] == ["invalid_base_score"]
    json.dumps(off, allow_nan=False)


@pytest.mark.parametrize("metric", ["no_collision", "drivable_area", "driving_direction", "ttc"])
def test_nonfinite_unused_diagnostics_do_not_become_a_new_gate(metric):
    item = candidate("finite_trajectory_and_score", 1.0)
    item["safety"][metric] = math.nan
    selected, first = choose_candidate([item], {}, False)
    _, second = choose_candidate([item], {}, True)
    assert selected["id"] == "finite_trajectory_and_score"
    assert selected["safe"] is False and selected["eligible"] is True
    assert selected["safety"][metric] is None
    assert selected["diagnostic_numeric_invalid"] == ["safety." + metric]
    assert math.isnan(item["safety"][metric])
    assert first["candidate_pool_sha256"] == second["candidate_pool_sha256"]
    json.dumps(first, allow_nan=False)


def test_all_diagnostic_failures_are_ranked_and_only_empty_pool_has_no_selection():
    candidates = pool()
    for item in candidates:
        item["safety"]["ttc"] = 0.0
    selected, _ = choose_candidate(candidates, {"longitudinal_decision": "STOP"}, True)
    assert selected["id"] == "stop"
    assert not selected["safe"] and selected["eligible"]
    selected, audit = choose_candidate([], {}, False)
    assert selected is None and audit["selected_id"] is None


def test_hash_and_ties_are_repeatable_for_the_same_ordered_pool():
    candidates = [candidate("first", 0.8), candidate("second", 0.8)]
    first, first_audit = choose_candidate(candidates, {}, False)
    reordered_keys = [dict(reversed(list(item.items()))) for item in candidates]
    second, second_audit = choose_candidate(reordered_keys, {}, False)
    assert first["id"] == second["id"] == "first"
    assert first_audit["candidate_pool_sha256"] == second_audit["candidate_pool_sha256"]


@pytest.mark.parametrize("enabled", [True, False])
def test_route_recovery_metadata_never_excludes_the_best_scored_candidate(enabled):
    center = candidate("center", 0.8, "MAINTAIN", "FOLLOW_ROUTE")
    center.update(route_target=True, forward_progress_m=25.0, terminal_speed_mps=6.4)
    left = candidate("left", 1.0, "MAINTAIN", "KEEP_LANE")
    left.update(route_target=False, route_recovery_required=True)
    selected, audit = choose_candidate([center, left], {"lateral_decision": "KEEP_LANE"}, enabled)
    assert selected["id"] == audit["baseline_selected_id"] == "left"
    assert selected["route_eligible"] and selected["eligible"]
    assert audit["route_recovery_active"] is False
    assert audit["route_recovery_gate_applied"] is False


@pytest.mark.parametrize("continuity_cost", [0.15, 0.5, 100, -1, math.nan, "obsolete"])
def test_continuity_metadata_never_changes_the_base_score(continuity_cost):
    stable = candidate("stable", 0.90, "YIELD")
    changed = candidate("changed", 0.95, "MAINTAIN")
    stable["continuity_cost"], changed["continuity_cost"] = 0.0, continuity_cost
    candidates = [stable, changed]
    off, off_audit = choose_candidate(candidates, {"longitudinal_decision": "YIELD"}, False)
    on, on_audit = choose_candidate(candidates, {"longitudinal_decision": "YIELD"}, True)
    assert off["id"] == on_audit["baseline_selected_id"] == "changed"
    assert on["id"] == "stable"
    assert off_audit["candidate_pool_sha256"] == on_audit["candidate_pool_sha256"]
    assert all(item["planning_score"] == item["base_score"] for item in off_audit["candidates"])
    assert on_audit["continuity_penalty_applied"] is False
    json.dumps(off_audit, allow_nan=False)
    if isinstance(continuity_cost, float) and math.isnan(continuity_cost):
        assert math.isnan(changed["continuity_cost"])
        assert off["continuity_cost"] is None
        assert off["diagnostic_numeric_invalid"] == ["continuity_cost"]


@pytest.mark.parametrize("weight", [-1, 2.1, math.nan, math.inf])
def test_invalid_semantic_weights_are_still_rejected(weight):
    with pytest.raises(ValueError, match="semantic weights"):
        choose_candidate(pool(), {}, True, weight, 0.8)
