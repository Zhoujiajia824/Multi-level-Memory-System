"""Ablation changes this score term only; candidates and safety are immutable."""
from __future__ import annotations

import hashlib
import json
import math


def choose_candidate(candidates, semantic, enabled, lateral_weight=0.8, longitudinal_weight=0.8):
    """Return an audited selection; all hard constraints precede semantic scoring."""
    for value in (lateral_weight, longitudinal_weight):
        if not math.isfinite(value) or value < 0 or value > 2:
            raise ValueError("semantic weights must be finite and within [0, 2]")
    semantic = semantic or {}
    lateral = semantic.get("lateral_decision")
    longitudinal = semantic.get("longitudinal_decision")
    audited = []
    for candidate in candidates:
        item = dict(candidate)
        safety = item.get("safety", {})
        metrics = [safety.get(k) for k in ("no_collision", "drivable_area", "driving_direction", "ttc")]
        safe = (all(type(v) in (int, float) and math.isfinite(v) and 1.0 - 1e-8 <= v <= 1 for v in metrics)
                and safety.get("finite") is True and safety.get("traffic_light") is True
                and type(item.get("base_score")) in (int, float) and math.isfinite(item["base_score"]))
        lateral_match = lateral in item.get("lateral_semantics", [])
        longitudinal_match = longitudinal == item.get("longitudinal")
        bonus = (lateral_weight * lateral_match + longitudinal_weight * longitudinal_match) if enabled else 0.0
        item.update(safe=safe, semantic_bonus=float(bonus),
                    final_score=float(item["base_score"] + bonus) if safe else None)
        audited.append(item)
    feasible = [item for item in audited if item["safe"]]
    selected = max(feasible, key=lambda item: item["final_score"]) if feasible else None
    baseline = max(feasible, key=lambda item: item["base_score"]) if feasible else None
    # Hash the generated pool/base metrics, excluding the experimental score term.
    identity = hashlib.sha256(json.dumps(candidates, sort_keys=True, allow_nan=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    available = sorted({s for c in audited for s in c.get("lateral_semantics", [])})
    return selected, {"semantic_enabled": bool(enabled), "semantic_requested": dict(semantic),
                      "candidate_pool_sha256": identity, "candidates": audited,
                      "selected_id": selected["id"] if selected else None,
                      "baseline_selected_id": baseline["id"] if baseline else None,
                      "available_lateral_semantics": available,
                      "unavailable_lateral_request": lateral if lateral and lateral not in available else None}
