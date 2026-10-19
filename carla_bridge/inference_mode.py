"""Validate model-call policy before creating simulator or memory resources."""
from __future__ import annotations


def model_inference_enabled(planning: dict, mode: str) -> bool:
    if planning.get("semantic_policy", "always") not in ("selective", "always", "shadow"):
        raise ValueError("planning.semantic_policy must be selective, always or shadow")
    if planning.get("planning_memory", "off") not in ("off", "gate", "full"):
        raise ValueError("planning.planning_memory must be off, gate or full")
    enabled = planning.get("model_inference", True)
    if type(enabled) is not bool:
        raise ValueError("planning.model_inference must be a boolean")
    if not enabled:
        if planning.get("backend", "vlm_trajectory") != "pdm_closed":
            raise ValueError("--model-inference off requires --planner pdm_closed")
        if planning.get("semantic_influence", True) is not False:
            raise ValueError("--model-inference off requires --semantic-influence off")
        if mode != "memory_off" and planning.get("planning_memory") != "full":
            raise ValueError("--model-inference off requires --mode memory_off or explicit "
                             "--planning-memory full with --mode memory_on; "
                             "model-free runs do not initialize text/visual decision memories")
    return enabled
