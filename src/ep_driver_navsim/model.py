"""NAVSIM-only server schema requests; core output validation stays authoritative."""
import copy
import logging
import math
from types import SimpleNamespace

import numpy as np

from nine_grid.decision_protocol import LATERAL_ACTIONS, LONGITUDINAL_ACTIONS
from nine_grid.models import VisionClient, _goal_reference, attention_ids
from nine_grid.geometry import polyline_lengths

STRUCTURED_OUTPUT_PROTOCOL = "navsim-scoped-response-schema-1.1"
STRUCTURED_DECODING = {"policy": "greedy", "temperature": 0.0}

logger = logging.getLogger(__name__)


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
        decision=_object(dict(lateral=decision(LATERAL_ACTIONS), longitudinal=decision(LONGITUDINAL_ACTIONS))),
        attention=dict(type="array", maxItems=min(3, len(ids)),
                       items=_object(dict(id=identity, reason=dict(text)))),
        local_goal=_object(dict(delta_s_m=dict(type="number", minimum=lower, maximum=upper),
                                d_m=lateral, reason=dict(text)))))
    return dict(type="json_schema", json_schema=dict(name="navsim_driving_decision_2_0",
                                                     strict=True, schema=schema))


class _SchemaTransport:
    def __init__(self, transport, response_format):
        self.transport = transport
        self.response_format = copy.deepcopy(response_format)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        kwargs["response_format"] = copy.deepcopy(self.response_format)
        # NAVSIM server evaluation uses greedy decoding; shared API clients keep
        # their existing settings and every retry retains the same strict schema.
        kwargs["temperature"] = STRUCTURED_DECODING["temperature"]
        return self.transport.chat.completions.create(**kwargs)


class StructuredServerVisionClient(VisionClient):
    """Keep the core image payload/retries while adding one mandatory server field."""
    def __init__(self, role, settings, **kwargs):
        if settings.get("backend") != "server":
            raise ValueError("NAVSIM structured_model_output requires the server backend")
        super().__init__(role, settings, **kwargs)
        self.settings["structured_decoding"] = copy.deepcopy(STRUCTURED_DECODING)
        logger.info("NAVSIM structured output protocol=%s decoding=%s",
                    STRUCTURED_OUTPUT_PROTOCOL, STRUCTURED_DECODING)

    def call_structured(self, prompt, image_paths, response_format):
        # A worker processes anchors sequentially. Every retry keeps this exact
        # schema; unsupported response_format errors never retry unconstrained.
        original = self.client
        self.client = _SchemaTransport(original, response_format)
        try:
            return super().call(prompt, image_paths)
        finally:
            self.client = original
