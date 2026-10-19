"""Rank finite proposals by the PDM score plus the optional semantic bonus.

Safety metrics remain observable diagnostics. They do not independently exclude
candidates in this experimental selection rule.
"""
from __future__ import annotations

import hashlib
import json
import math


SELECTION_RULE = "pdm_base_plus_semantic"


def _pool_identity(value):
    """Keep the pool hash deterministic even when an invalid input contains NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"__non_finite_float__": "nan" if math.isnan(value) else
                ("positive_infinity" if value > 0 else "negative_infinity")}
    if isinstance(value, dict):
        return {key: _pool_identity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pool_identity(item) for item in value]
    return value


def _audit_value(value, invalid_paths, path=""):
    """Represent non-finite diagnostic numbers as JSON null without gating."""
    if isinstance(value, float) and not math.isfinite(value):
        invalid_paths.append(path)
        return None
    if isinstance(value, dict):
        return {key: _audit_value(item, invalid_paths, path + "." + key if path else key)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_value(item, invalid_paths, "%s[%d]" % (path, index))
                for index, item in enumerate(value)]
    return value


def choose_candidate(candidates, semantic, enabled, lateral_weight=0.8, longitudinal_weight=0.8,
                     *, semantic_bonus_by_id=None):
    """Select by score only; reject numerically invalid trajectories or scores."""
    for value in (lateral_weight, longitudinal_weight):
        if not math.isfinite(value) or value < 0 or value > 2:
            raise ValueError("semantic weights must be finite and within [0, 2]")
    if semantic_bonus_by_id is not None:
        if not isinstance(semantic_bonus_by_id, dict):
            raise ValueError("semantic_bonus_by_id must be a dictionary")
        identities = {candidate.get("id") for candidate in candidates}
        if any(identity not in identities for identity in semantic_bonus_by_id):
            raise ValueError("semantic bonus refers to an unknown candidate")
        if any(type(value) not in (int, float) or not 0 <= value <= 4
               or not math.isfinite(value) for value in semantic_bonus_by_id.values()):
            raise ValueError("semantic bonus overrides must be finite and within [0, 4]")
    semantic = semantic or {}
    lateral = semantic.get("lateral_decision")
    longitudinal = semantic.get("longitudinal_decision")
    audited = []
    for candidate in candidates:
        diagnostic_invalid = []
        item = _audit_value(candidate, diagnostic_invalid)
        safety = candidate.get("safety", {})
        metrics = [safety.get(k) for k in ("no_collision", "drivable_area", "driving_direction", "ttc")]
        score_valid = (type(candidate.get("base_score")) in (int, float)
                       and math.isfinite(candidate["base_score"]))
        numerically_valid = safety.get("finite") is True and score_valid
        # Retain the former safety assessment for comparison with older logs.
        # It is deliberately independent from eligibility in this version.
        safe = (all(type(v) in (int, float) and math.isfinite(v) and 1.0 - 1e-8 <= v <= 1 for v in metrics)
                and safety.get("finite") is True and safety.get("traffic_light", True) is True
                and score_valid)
        lateral_match = lateral in item.get("lateral_semantics", [])
        longitudinal_match = longitudinal == item.get("longitudinal")
        bonus = (lateral_weight * lateral_match + longitudinal_weight * longitudinal_match) if enabled else 0.0
        if semantic_bonus_by_id is not None:
            bonus = float(semantic_bonus_by_id.get(candidate.get("id"), 0.0)) if enabled else 0.0
        planning_score = float(item["base_score"]) if score_valid else None
        invalid_reasons = []
        if safety.get("finite") is not True:
            invalid_reasons.append("non_finite_trajectory")
        if not score_valid:
            invalid_reasons.append("invalid_base_score")
            item["base_score"] = None
        item.update(safe=safe, semantic_bonus=float(bonus),
                    numerically_valid=numerically_valid,
                    invalid_numeric_reasons=invalid_reasons,
                    diagnostic_numeric_invalid=diagnostic_invalid,
                    planning_score=planning_score,
                    final_score=float(planning_score + bonus) if numerically_valid else None,
                    # Compatibility fields: no route or safety gate is applied.
                    route_eligible=True, eligible=numerically_valid)
        audited.append(item)
    eligible = [item for item in audited if item["eligible"]]
    selected = max(eligible, key=lambda item: item["final_score"]) if eligible else None
    baseline = max(eligible, key=lambda item: item["planning_score"]) if eligible else None
    # Hash the generated pool/base metrics, excluding the experimental score term.
    identity = hashlib.sha256(json.dumps(_pool_identity(candidates), sort_keys=True, allow_nan=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    available = sorted({s for c in eligible for s in c.get("lateral_semantics", [])})
    return selected, {"semantic_enabled": bool(enabled), "semantic_requested": dict(semantic),
                      "selection_rule": SELECTION_RULE,
                      "safety_gate_applied": False, "safety_is_diagnostic_only": True,
                      "continuity_penalty_applied": False,
                      "route_recovery_gate_applied": False, "route_recovery_active": False,
                      "candidate_pool_sha256": identity, "candidates": audited,
                      "selected_id": selected["id"] if selected else None,
                      "baseline_selected_id": baseline["id"] if baseline else None,
                      "available_lateral_semantics": available,
                      "unavailable_lateral_request": lateral if lateral and lateral not in available else None}
