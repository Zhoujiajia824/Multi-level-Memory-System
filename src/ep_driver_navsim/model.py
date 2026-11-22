"""NAVSIM-only server schema requests; core output validation stays authoritative."""
import copy
import math
from types import SimpleNamespace

from nine_grid.decision_protocol import LATERAL_ACTIONS, LONGITUDINAL_ACTIONS
from nine_grid.models import VisionClient, _goal_reference, attention_ids
from nine_grid.geometry import polyline_lengths

STRUCTURED_OUTPUT_PROTOCOL = "navsim-scoped-response-schema-1.0"


def _object(properties):
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)


def structured_response_format(scene, config=None, reference=None):
    """Constrain syntax, current IDs and longitudinal limits without choosing a goal."""
    reference = _goal_reference(scene, config) if reference is None else reference
    if (reference.get("frame_id") != scene.frame_id or reference.get("time") != scene.time
            or reference.get("semantic_reference_valid", True) is not True):
        raise ValueError("Structured output requires a valid goal reference for the current frame")
    limits = reference["limits"]
    lower = float(limits["minimum_delta_s_m"])
    upper = min(float(limits["maximum_delta_s_m"]),
                float(polyline_lengths(reference["points"])[-1] - reference["s_ego_m"] - 1e-6))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        raise ValueError("Structured output has no legal longitudinal goal interval")
    ids = sorted(attention_ids(scene, config))
    text = dict(type="string", minLength=1)
    identity = dict(type="string", enum=ids) if ids else dict(text)
    def decision(actions):
        return _object(dict(action=dict(type="string", enum=list(actions)), reason=dict(text)))
    schema = _object(dict(
        scene=_object(dict(description=dict(text), reason=dict(text))),
        decision=_object(dict(lateral=decision(LATERAL_ACTIONS), longitudinal=decision(LONGITUDINAL_ACTIONS))),
        attention=dict(type="array", maxItems=min(3, len(ids)),
                       items=_object(dict(id=identity, reason=dict(text)))),
        local_goal=_object(dict(delta_s_m=dict(type="number", minimum=lower, maximum=upper),
                                d_m=dict(type="number"), reason=dict(text)))))
    return dict(type="json_schema", json_schema=dict(name="navsim_driving_decision_2_0",
                                                     strict=True, schema=schema))


class _SchemaTransport:
    def __init__(self, transport, response_format):
        self.transport = transport
        self.response_format = copy.deepcopy(response_format)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        kwargs["response_format"] = copy.deepcopy(self.response_format)
        return self.transport.chat.completions.create(**kwargs)


class StructuredServerVisionClient(VisionClient):
    """Keep the core image payload/retries while adding one mandatory server field."""
    def __init__(self, role, settings, **kwargs):
        if settings.get("backend") != "server":
            raise ValueError("NAVSIM structured_model_output requires the server backend")
        super().__init__(role, settings, **kwargs)

    def call_structured(self, prompt, image_paths, response_format):
        # A worker processes anchors sequentially. Every retry keeps this exact
        # schema; unsupported response_format errors never retry unconstrained.
        original = self.client
        self.client = _SchemaTransport(original, response_format)
        try:
            return super().call(prompt, image_paths)
        finally:
            self.client = original
