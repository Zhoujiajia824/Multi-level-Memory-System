"""NAVSIM-only server schema requests; core output validation stays authoritative."""
import copy
import json
import logging
import math
import re
import time
from types import SimpleNamespace

import numpy as np

from nine_grid.decision_protocol import LATERAL_ACTIONS, LONGITUDINAL_ACTIONS
from nine_grid.models import VisionClient, _goal_reference, attention_ids
from nine_grid.geometry import polyline_lengths

STRUCTURED_OUTPUT_PROTOCOL = "navsim-scoped-response-schema-1.2"
STRUCTURED_DECODING = {"policy": "greedy", "temperature": 0.0}
MODEL_ATTEMPT_PROTOCOL = "ep-navsim-model-attempts-2"
GOAL_RETRY_PROTOCOL = "navsim-model-goal-retry-2"
SEMANTIC_RETRY_STAGES = ("goal_geometry", "attention_identity", "decision_goal_mismatch")

logger = logging.getLogger(__name__)


def goal_retry_policy(settings):
    limit = settings.get("model_goal_retry_limit", 1 if settings["fast"] == "server" else 0)
    if type(limit) is not int or limit not in (0, 1) or (settings["fast"] == "off" and limit):
        raise ValueError("model_goal_retry_limit must be 0 or 1; off requires 0")
    return dict(protocol=GOAL_RETRY_PROTOCOL, limit=limit, trigger="correctable_answer_only",
                stages=list(SEMANTIC_RETRY_STAGES), reference="same_source_frame")


def rejection_stage(raw, scene, config, reference):
    """Locate a rejection without relaxing the core parser's acceptance rules."""
    from nine_grid.decision_protocol import normalize_model_output
    from nine_grid.models import _unique_answer_object
    if not isinstance(raw, str) or len(raw) > 6000:
        return "response_format"
    serialized = raw.strip()
    if serialized.startswith("```"):
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n((?:(?!```).)*?)\r?\n```",
                             serialized, re.DOTALL | re.IGNORECASE)
        if match is None:
            return "response_format"
        serialized = match[1]
    try:
        attention, _, _, _ = normalize_model_output(json.loads(serialized, object_pairs_hook=_unique_answer_object))
    except (ValueError, TypeError):
        return "response_format"
    ids = attention_ids(scene, config)
    selected = [item["id"] for item in attention]
    if len(set(selected)) != len(selected) or any(value not in ids for value in selected):
        return "attention_identity"
    if (not isinstance(reference, dict) or reference.get("frame_id") != scene.frame_id
            or reference.get("time") != scene.time or reference.get("semantic_reference_valid", True) is not True):
        return "goal_reference"
    return "goal_geometry"


def goal_retry_prompt(prompt, raw, reason, attempt, stage="goal_geometry"):
    """Ask the model again; never clamp, snap or substitute its goal locally."""
    feedback = dict(attempt=attempt, rejected_answer=raw, rejection=str(reason), stage=stage,
                    coordinate_policy="原始输入帧及原规划参考不变；不得更换坐标原点或使用未来状态")
    return (prompt + "\n回答校验未通过。请根据原始场景、同一规划参考和下面的拒绝原因，"
            "重新给出完整四部分JSON回答。程序没有修改你的目标点；仍须由你重新选择可达目标。\n"
            "必须纠正被指出的坐标、重复对象或行动矛盾；不要原样重复被拒绝的回答。"
            "delta_s_m是从自车投影向前的增量，不是source_s_ego_m；d_m表示车体中心位置。\n"
            + json.dumps(feedback, ensure_ascii=False))


class _AuditedTransport:
    def __init__(self, transport, records):
        self.transport, self.records = transport, records
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        content = kwargs.get("messages", [{}, {}])[-1].get("content", [])
        row = dict(max_tokens=kwargs.get("max_tokens"),
                   image_count=sum(isinstance(item, dict) and item.get("type") == "image_url"
                                   for item in content) if isinstance(content, list) else 0)
        started = time.monotonic()
        try:
            result = self.transport.chat.completions.create(**kwargs)
        except Exception as exc:
            row["transport_error"] = type(exc).__name__
            status = getattr(exc, "status_code", None)
            row["http_status"] = status if type(status) is int else None
            raise
        else:
            row["response_model"] = getattr(result, "model", None)
            choices = getattr(result, "choices", [])
            row["choices"] = [dict(finish_reason=getattr(choice, "finish_reason", None),
                                   raw=getattr(getattr(choice, "message", None), "content", None))
                              for choice in choices]
            usage = getattr(result, "usage", None)
            row["usage"] = {key: getattr(usage, key, None) for key in
                            ("prompt_tokens", "completion_tokens", "total_tokens")} if usage else None
            return result
        finally:
            row["wall_s"] = time.monotonic()-started
            self.records.append(row)


class AuditedVisionClient(VisionClient):
    """Preserve response evidence even when the existing transport rejects length."""
    def call(self, prompt, image_paths):
        self.last_transport_attempts = []
        original = self.client
        if not self.enabled:
            return None
        self.client = _AuditedTransport(original, self.last_transport_attempts)
        try:
            return super().call(prompt, image_paths)
        finally:
            self.client = original


class NoLegalGoalReference(ValueError):
    """Current reference has no semantic goal domain; skip a model request."""
    def __init__(self, reason_code, detail):
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        super().__init__(self.detail)


def _object(properties):
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)


def _lateral_goal_bounds(scene, reference, lower, upper):
    """Outer d envelope of the same legal lane candidates, in frozen Frenet axes.

    A constant offset at ego would cut off curved/diverging neighbouring lanes.
    Instead intersect their geometric corridors with every reference segment's
    own normal strip over the allowed longitudinal interval. Linear d extrema
    occur at intersection vertices. Full lane widths (not body-eroded widths)
    and square/miter buffer corners deliberately over-cover the valid centres.
    This box does not prove two-dimensional reachability or choose a target.
    Incomplete geometry yields no additional d constraint.
    """
    from shapely.errors import GEOSException
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
    from nine_grid.goals import goal_reference_context
    from nine_grid.local_reference import LocalReferenceManager
    from nine_grid.geometry import clean_polyline

    slope = float(reference["limits"]["maximum_lateral_slope"])
    if not math.isfinite(slope) or slope < 0:
        raise ValueError("Structured goal lateral slope must be finite and nonnegative")
    source_d = float(reference["d_ego_m"])
    reach = max(abs(lower), abs(upper)) * slope
    outer_low, outer_high = source_d - reach, source_d + reach
    if reach <= 1e-9:
        return outer_low, outer_high
    line = clean_polyline(reference["points"])
    arc = polyline_lengths(line)
    lanes = {lane.id: lane for lane in scene.lanes}
    manager = LocalReferenceManager({"local_reference": reference["local_reference_parameters"]})
    context = goal_reference_context(scene, reference)
    corridors = []
    for row in context["candidate_lanes"]:
        if not row["goal_candidate_legal"]:
            continue
        lane = lanes.get(row["lane_id"])
        if lane is None or not math.isfinite(float(lane.width)) or lane.width <= 0:
            return None
        if lane.id == reference["source_lane_id"]:
            path = line
        else:
            try:
                path = manager._continuation(scene, lane, lanes)
            except ValueError:
                # Unknown/degenerate future continuation cannot justify cutting
                # off that candidate: omit the extra lateral box altogether.
                return None
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
            return None
        # Segment rectangles are included exactly; no sampled centre offsets or
        # circle approximation can shrink the outer envelope at curve corners.
        try:
            corridors.append(LineString(path).buffer(float(lane.width)/2 + 1e-5,
                                                     cap_style=3, join_style=2))
        except GEOSException:
            return None
    if not corridors:
        return None

    def vertices(geometry):
        if hasattr(geometry, "geoms"):
            pieces = [vertices(part) for part in geometry.geoms if not part.is_empty]
            return np.vstack(pieces) if pieces else np.empty((0, 2))
        return np.asarray(geometry.exterior.coords if hasattr(geometry, "exterior") else geometry.coords)

    bounds = []
    try:
        corridor = unary_union(corridors)
        start_s, finish_s = float(reference["s_ego_m"]) + lower, float(reference["s_ego_m"]) + upper
        for index in np.flatnonzero((arc[:-1] <= finish_s) & (arc[1:] >= start_s)):
            length = arc[index+1] - arc[index]
            unit = (line[index+1] - line[index]) / length
            normal = np.array([-unit[1], unit[0]])
            first = line[index] + unit * max(0., start_s-arc[index])
            last = line[index] + unit * min(length, finish_s-arc[index])
            if np.linalg.norm(last-first) <= 1e-9:
                continue
            strip = Polygon([first+outer_low*normal, last+outer_low*normal,
                             last+outer_high*normal, first+outer_high*normal])
            overlap = corridor.intersection(strip)
            if not overlap.is_empty:
                values = (vertices(overlap)-first) @ normal
                if len(values):
                    bounds.append((float(values.min()), float(values.max())))
    except GEOSException:
        return None
    if not bounds:
        return None
    return max(outer_low, min(item[0] for item in bounds)-1e-6), \
        min(outer_high, max(item[1] for item in bounds)+1e-6)


def structured_response_format(scene, config=None, reference=None):
    """Constrain syntax, current IDs and longitudinal limits without choosing a goal."""
    reference = _goal_reference(scene, config) if reference is None else reference
    if reference.get("frame_id") != scene.frame_id or reference.get("time") != scene.time:
        raise ValueError("Structured output requires the goal reference for the current frame")
    limits = reference["limits"]
    lower = float(limits["minimum_delta_s_m"])
    maximum = float(limits["maximum_delta_s_m"])
    lateral_slope = float(limits["maximum_lateral_slope"])
    points = np.asarray(reference["points"], dtype=float)
    source_s = float(reference["s_ego_m"])
    source_d = float(reference["d_ego_m"])
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all()
            or not all(math.isfinite(value) for value in (lower, maximum, lateral_slope, source_s, source_d))):
        raise ValueError("Structured goal reference geometry and limits must be finite")
    if lateral_slope < 0:
        raise ValueError("Structured goal lateral slope must be nonnegative")
    upper = min(float(limits["maximum_delta_s_m"]),
                float(polyline_lengths(points)[-1] - source_s - 1e-6))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Structured goal interval must be finite")
    if reference.get("semantic_reference_valid", True) is not True:
        raise NoLegalGoalReference("semantic_reference_invalid",
            reference.get("reference_rejection") or "Current reference has no valid semantic goal domain")
    if upper < lower:
        raise NoLegalGoalReference("empty_longitudinal_goal_interval",
            "Structured output has no legal longitudinal goal interval")
    from .model_guidance import (GuidanceUnavailable, validate_request_state,
                                 allowed_lateral_actions)
    try:
        validate_request_state(scene, reference)
    except GuidanceUnavailable as exc:
        raise NoLegalGoalReference(exc.reason_code, str(exc)) from exc
    ids = sorted(attention_ids(scene, config))
    text = dict(type="string", minLength=1, maxLength=192)
    identity = dict(type="string", enum=ids) if ids else dict(text)
    lateral = dict(type="number")
    lateral_bounds = _lateral_goal_bounds(scene, reference, lower, upper)
    if lateral_bounds is not None:
        lateral.update(minimum=lateral_bounds[0], maximum=lateral_bounds[1])
    def decision(actions):
        return _object(dict(action=dict(type="string", enum=list(actions)), reason=dict(text)))
    schema = _object(dict(
        scene=_object(dict(description=dict(text), reason=dict(text))),
        decision=_object(dict(lateral=decision(allowed_lateral_actions(scene, reference)),
                              longitudinal=decision(LONGITUDINAL_ACTIONS))),
        attention=dict(type="array", maxItems=min(3, len(ids)),
                       items=_object(dict(id=identity, reason=dict(text)))),
        local_goal=_object(dict(delta_s_m=dict(type="number", minimum=lower, maximum=upper),
                                d_m=lateral, reason=dict(text)))))
    return dict(type="json_schema", json_schema=dict(name="navsim_driving_decision_2_0",
                                                     strict=True, schema=schema))


from .model_transport import BoundedStructuredVisionClient


class StructuredServerVisionClient(BoundedStructuredVisionClient):
    """NAVSIM-only compact structured requests with a shared bounded deadline."""
    def __init__(self, role, settings, **kwargs):
        if settings.get("backend") != "server":
            raise ValueError("NAVSIM structured_model_output requires the server backend")
        super().__init__(role, settings, **kwargs)
        self.settings["structured_decoding"] = copy.deepcopy(STRUCTURED_DECODING)
        logger.info("NAVSIM structured output protocol=%s decoding=%s",
                    STRUCTURED_OUTPUT_PROTOCOL, STRUCTURED_DECODING)
