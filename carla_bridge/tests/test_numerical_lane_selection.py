"""Lane quality changes numerical preference while retaining native evidence."""
from copy import deepcopy
import math

import pytest

from carla_bridge.planning.selection import LANE_SELECTION_RULE, choose_candidate


def proposal(name, score, cost=0.0, lateral="FOLLOW_ROUTE"):
    return {"id": name, "base_score": score, "numerical_cost": cost,
            "longitudinal": "MAINTAIN", "lateral_semantics": [lateral],
            "safety": {"finite": True, "no_collision": 1.0, "drivable_area": 1.0,
                       "driving_direction": 1.0, "ttc": 1.0}}


def test_actual_lane_quality_resolves_native_tie_without_rewriting_native_scores():
    candidates = [proposal("line_hugging", 1.0, .14), proposal("centered", 1.0, .01)]
    original = deepcopy(candidates)
    selected, audit = choose_candidate(candidates, None, False)
    assert selected["id"] == audit["baseline_selected_id"] == "centered"
    assert selected["planning_score"] == pytest.approx(.99)
    assert audit["selection_rule"] == LANE_SELECTION_RULE
    assert audit["numerical_cost_applied"] is True
    assert candidates == original
    assert all(c["base_score"] == 1.0 for c in audit["candidates"])


def test_cost_is_soft_and_does_not_exclude_a_materially_better_bypass():
    selected, audit = choose_candidate([proposal("creep", .7),
                                       proposal("bypass", 1.0, .12, "CHANGE_LANE_LEFT")], None, False)
    assert selected["id"] == "bypass"
    assert all(c["eligible"] for c in audit["candidates"])
    assert audit["safety_gate_applied"] is False


def test_semantic_switch_uses_the_identical_numerical_pool_and_native_scores():
    candidates = [proposal("centered", 1., 0.), proposal("offset", 1., .05, "CHANGE_LANE_LEFT")]
    sem = {"lateral_decision": "CHANGE_LANE_LEFT", "longitudinal_decision": "MAINTAIN"}
    off, baseline = choose_candidate(candidates, sem, False, .08, 0.)
    on, influenced = choose_candidate(candidates, sem, True, .08, 0.)
    assert off["id"] == "centered"
    assert on["id"] == "offset"
    assert baseline["candidate_pool_sha256"] == influenced["candidate_pool_sha256"]
    assert [c["planning_score"] for c in baseline["candidates"]] == [c["planning_score"] for c in influenced["candidates"]]


@pytest.mark.parametrize("cost", [None, True, -.001, .1601, math.nan, math.inf, "0.1"])
def test_invalid_numerical_cost_is_a_numerical_error_not_a_road_veto(cost):
    selected, audit = choose_candidate([proposal("invalid", 1., cost), proposal("valid", .5)], None, False)
    assert selected["id"] == "valid"
    bad = audit["candidates"][0]
    assert not bad["eligible"]
    assert bad["invalid_numeric_reasons"] == ["invalid_numerical_cost"]
    assert bad["base_score"] == 1.


def test_omitted_cost_keeps_legacy_rank_and_rule():
    candidates = [proposal("first", .9), proposal("second", 1.)]
    for candidate in candidates:
        del candidate["numerical_cost"]
    selected, audit = choose_candidate(candidates, None, False)
    assert selected["id"] == "second"
    assert audit["selection_rule"] == "pdm_base_plus_semantic"
    assert not audit["numerical_cost_applied"]
