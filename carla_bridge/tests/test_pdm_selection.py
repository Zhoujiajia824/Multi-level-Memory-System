"""Safety and ablation contracts for semantic candidate selection."""
from copy import deepcopy

import pytest

from carla_bridge.planning.selection import choose_candidate


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


def test_stop_bonus_can_change_selection_without_changing_pool_or_base_scores():
    candidates = pool()
    original = deepcopy(candidates)
    semantic = {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "STOP"}
    baseline, off = choose_candidate(candidates, semantic, False)
    selected, on = choose_candidate(candidates, semantic, True)
    assert baseline["id"] == "cruise"
    assert selected["id"] == "stop"
    assert selected["semantic_bonus"] > 0
    assert on["candidate_pool_sha256"] == off["candidate_pool_sha256"]
    assert [c["base_score"] for c in on["candidates"]] == [c["base_score"] for c in original]
    assert [c["safety"] for c in on["candidates"]] == [c["safety"] for c in original]
    assert candidates == original
    assert semantic == {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "STOP"}


@pytest.mark.parametrize("failed_metric", [
    "no_collision", "drivable_area", "driving_direction", "ttc", "finite", "traffic_light",
])
def test_unsafe_candidate_never_wins_even_with_maximum_semantic_bonus(failed_metric):
    safe = candidate("safe", 0.1)
    unsafe = candidate("unsafe", 1.0, "STOP", "CHANGE_LANE_LEFT")
    unsafe["safety"][failed_metric] = False if failed_metric in ("finite", "traffic_light") else 0.0
    selected, audit = choose_candidate([safe, unsafe], {
        "lateral_decision": "CHANGE_LANE_LEFT", "longitudinal_decision": "STOP",
    }, True, 2.0, 2.0)
    assert selected["id"] == "safe"
    rejected = audit["candidates"][1]
    assert rejected["safe"] is False
    assert rejected["final_score"] is None


@pytest.mark.parametrize("flag", ["finite", "traffic_light"])
@pytest.mark.parametrize("value", ["false", "true", 1, 0, None])
def test_safety_flags_require_actual_true_booleans(flag, value):
    invalid = candidate("invalid", 1.0)
    invalid["safety"][flag] = value
    selected, audit = choose_candidate([invalid], {}, False)
    assert selected is None
    assert audit["candidates"][0]["safe"] is False


@pytest.mark.parametrize("metric", [
    "no_collision", "drivable_area", "driving_direction", "ttc", "finite", "traffic_light",
])
def test_missing_safety_evidence_fails_closed(metric):
    invalid = candidate("invalid", 1.0)
    del invalid["safety"][metric]
    selected, audit = choose_candidate([invalid], {}, True)
    assert selected is None
    assert audit["selected_id"] is None


@pytest.mark.parametrize("field", ["base_score", "no_collision", "drivable_area",
                                  "driving_direction", "ttc"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_metrics_never_return_an_executable_selection(field, value):
    invalid = candidate("invalid", 1.0)
    if field == "base_score":
        invalid[field] = value
    else:
        invalid["safety"][field] = value
    # Strict JSON audit encoding may reject the whole pool; accepting it as an
    # explicitly unsafe candidate is also fail-closed. Neither may select it.
    try:
        selected, audit = choose_candidate([invalid], {}, False)
    except ValueError:
        return
    assert selected is None
    assert audit["candidates"][0]["safe"] is False


def test_all_unsafe_and_empty_pools_have_no_selected_trajectory():
    unsafe = pool()
    for item in unsafe:
        item["safety"]["ttc"] = 0.0
    for candidates in (unsafe, []):
        selected, audit = choose_candidate(candidates, {"longitudinal_decision": "STOP"}, True)
        assert selected is None
        assert audit["selected_id"] is None


def test_hash_and_ties_are_repeatable_for_the_same_ordered_pool():
    candidates = [candidate("first", 0.8), candidate("second", 0.8)]
    first, first_audit = choose_candidate(candidates, {}, False)
    # Dictionary key insertion order does not change the identity of a pool.
    reordered_keys = [dict(reversed(list(item.items()))) for item in candidates]
    second, second_audit = choose_candidate(reordered_keys, {}, False)
    assert first["id"] == second["id"]
    assert first_audit["candidate_pool_sha256"] == second_audit["candidate_pool_sha256"]


def recovery_pool():
    # Scores from the 75.1s server snapshot: KEEP_LANE formerly interrupted
    # recovery even though both alternatives passed collision/road/TTC checks.
    center = candidate("center:MAINTAIN", 0.9599316416, "MAINTAIN", "FOLLOW_ROUTE")
    center.update(route_target=True, forward_progress_m=25.0, terminal_speed_mps=6.4)
    left = candidate("left:MAINTAIN", 0.9590769486, "MAINTAIN", "KEEP_LANE")
    left.update(route_target=False, route_recovery_required=True)
    return [center, left]


@pytest.mark.parametrize("enabled", [True, False])
def test_route_deadline_precedes_semantics_and_baseline_progress(enabled):
    candidates = recovery_pool()
    candidates[1]["base_score"] = 1.0  # Also protect the semantics-off arm.
    selected, audit = choose_candidate(candidates, {"lateral_decision": "KEEP_LANE",
                                                  "longitudinal_decision": "MAINTAIN"}, enabled)
    assert selected["id"] == audit["baseline_selected_id"] == "center:MAINTAIN"
    assert audit["route_recovery_active"]
    assert audit["candidates"][1]["safe"]  # Safety and route eligibility are separate.
    assert not audit["candidates"][1]["route_eligible"]
    _, other = choose_candidate(candidates, {}, not enabled)
    assert audit["candidate_pool_sha256"] == other["candidate_pool_sha256"]


@pytest.mark.parametrize("blocked", ["collision", "red_light", "stopped"])
def test_route_deadline_never_forces_a_blocked_return(blocked):
    candidates = recovery_pool()
    if blocked == "collision":
        candidates[0]["safety"]["no_collision"] = 0.0
    elif blocked == "red_light":
        candidates[0]["safety"]["traffic_light"] = False
    else:
        candidates[0].update(forward_progress_m=0.0, terminal_speed_mps=0.0)
    selected, audit = choose_candidate(candidates, {"lateral_decision": "KEEP_LANE"}, True)
    assert selected["id"] == "left:MAINTAIN"
    assert not audit["route_recovery_active"]


def test_keep_lane_remains_effective_without_route_deadline():
    candidates = recovery_pool()
    candidates[1]["route_recovery_required"] = False
    selected, audit = choose_candidate(candidates, {"lateral_decision": "KEEP_LANE"}, True)
    assert selected["id"] == "left:MAINTAIN"
    assert not audit["route_recovery_active"]
