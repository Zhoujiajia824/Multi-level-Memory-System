"""Validate model-call policy before creating simulator or memory resources."""
from __future__ import annotations


def model_inference_enabled(planning: dict, mode: str) -> bool:
    enabled = planning.get("model_inference", True)
    if type(enabled) is not bool:
        raise ValueError("planning.model_inference must be a boolean")
    if not enabled:
        if planning.get("backend", "vlm_trajectory") != "pdm_closed":
            raise ValueError("--model-inference off requires --planner pdm_closed")
        if planning.get("semantic_influence", True) is not False:
            raise ValueError("--model-inference off requires --semantic-influence off")
        if mode != "memory_off":
            raise ValueError("--model-inference off requires --mode memory_off; "
                             "model-free runs do not retrieve or create decision memories")
    return enabled
