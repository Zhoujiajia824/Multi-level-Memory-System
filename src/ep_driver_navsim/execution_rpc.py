"""Bound numerical-to-SDK controller previews; never transmits a Scene/cache."""
import json
import uuid

import numpy as np


PREVIEW_PREFIX = "EP_NAVSIM_PREVIEW "
PROTOCOL = "ep-navsim-current-controller-preview-1"
MAX_BATCH = 65
# One initial preview plus, per iteration, one 49-point linearization,
# one batched line search and at most three second-order corrections.
# The same request computes both the main and the independent reference.
MAX_CALLS = 2 * (1 + 8 * (1 + 1 + 3))


def validate_poses(value):
    poses = np.asarray(value, dtype=np.float32)
    if (poses.ndim != 3 or poses.shape[1:] != (8, 3)
            or not 1 <= len(poses) <= MAX_BATCH or not np.isfinite(poses).all()):
        raise ValueError("Controller preview requires 1..65 finite eight-pose trajectories")
    return poses


def validate_states(value, count):
    states = np.asarray(value, dtype=float)
    if states.shape != (count, 41, 11) or not np.isfinite(states).all():
        raise ValueError("Controller preview returned invalid executed-state arrays")
    return states


class WorkerPreview:
    """Synchronous nested RPC while the SDK parent awaits this planning request."""
    def __init__(self, output, input_stream, request_id, token, packet_sha256=None):
        self.output = output
        self.input_stream = input_stream
        self.request_id = request_id
        self.token = token
        self.packet_sha256 = packet_sha256
        self.calls = 0
        self.provenance = None

    def __call__(self, value):
        poses = validate_poses(value)
        self.calls += 1
        if self.calls > MAX_CALLS:
            raise RuntimeError("Controller preview request exceeded its bounded call count")
        preview_id = uuid.uuid4().hex
        request = dict(protocol=PROTOCOL, operation="native_preview", request_id=self.request_id,
                       preview_id=preview_id, frame_id=self.token, poses=poses.tolist())
        self.output.write(PREVIEW_PREFIX + json.dumps(request, allow_nan=False) + "\n")
        self.output.flush()
        line = self.input_stream.readline()
        if not line:
            raise RuntimeError("SDK parent closed while a controller preview was pending")
        reply = json.loads(line)
        for key in ("protocol", "request_id", "preview_id", "frame_id"):
            if reply.get(key) != request[key]:
                raise RuntimeError("Controller preview response is not bound to this request/frame")
        if reply.get("operation") != "native_preview_result":
            raise RuntimeError("Unexpected nested controller-preview response")
        if reply.get("ok") is not True:
            raise RuntimeError("SDK controller preview failed: " + str(reply.get("error")))
        provenance = reply.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("Controller preview lacks causal provenance")
        if self.packet_sha256 is not None and provenance.get('input_packet_sha256') != self.packet_sha256:
            raise ValueError("Controller preview did not use the active source packet")
        if self.provenance is not None and provenance != self.provenance:
            raise ValueError("Controller preview source changed within one planning request")
        self.provenance = provenance
        return validate_states(reply["states"], len(poses))
