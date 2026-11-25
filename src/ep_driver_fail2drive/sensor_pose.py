"""Read raw camera-pose metadata without changing the official image parser.

CARLA 0.9.15 serializes SensorData.transform in the Unreal local world, in
metres. Actor snapshots use the LargeMap global world. Keeping the raw camera
pose and its configured rigid mount provides a shared-frame wheel anchor.

The official callback class is instrumented once per process. Only registered
provider identities and camera tags are tapped; its original parser still runs
exactly once. No sensor is spawned, subscribed, stopped, or advanced here.
"""
import copy
import functools
import inspect
import math
import threading
import weakref


SCHEMA = "ep-driver-f2d-sensor-pose-1.0"
CAPTURE_METHOD = "official_callback_raw_image_transform_tap"
TIMESTAMP_TOLERANCE_S = 1e-6
_METHOD = "_parse_image_cb"
_MISSING = object()
_REGISTRY_LOCK = threading.RLock()
_PATCHES = {}
_ACTIVE_CAPTURE = threading.local()


class SensorPoseError(ValueError):
    """A source-frame pose failure with JSON-safe evidence for the archive."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = copy.deepcopy(evidence)


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _matrix(value):
    try:
        rows = [list(row) for row in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("camera transform must be a 4x4 rigid matrix") from exc
    if (len(rows) != 4 or any(len(row) != 4 for row in rows)
            or any(not _finite_number(v) for row in rows for v in row)):
        raise ValueError("camera transform must contain 16 finite numeric values")
    rows = [[float(v) for v in row] for row in rows]
    if any(abs(rows[3][i] - (1.0 if i == 3 else 0.0)) > 1e-7 for i in range(4)):
        raise ValueError("camera transform has an invalid homogeneous row")
    # CARLA matrices use float32 rotation values. Reject scale/reflection while
    # allowing their ordinary roundoff; do not silently repair a bad matrix.
    for i in range(3):
        for j in range(3):
            dot = sum(rows[k][i]*rows[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 2e-4:
                raise ValueError("camera transform rotation is not orthonormal")
    det = (rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1])
           - rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0])
           + rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]))
    if abs(det-1.0) > 2e-4:
        raise ValueError("camera transform rotation must preserve handedness")
    return rows


def _parser_description(callback_class, original):
    try:
        source = inspect.getsourcefile(original)
    except (OSError, TypeError):
        source = None
    return {"class_module": callback_class.__module__,
            "class_name": callback_class.__qualname__,
            "method": _METHOD,
            "parser_module": getattr(original, "__module__", None),
            "parser_name": getattr(original, "__qualname__", None),
            "source_file": source}


class _Patch:
    def __init__(self, callback_class, original):
        self.callback_class = callback_class
        self.original = original
        self.previous_attribute = callback_class.__dict__.get(_METHOD, _MISSING)
        self.providers = {}
        self.provider_parsers = {}
        self.description = _parser_description(callback_class, original)

        @functools.wraps(original)
        def wrapper(callback, image, tag):
            # Never hold the registry lock while calling CARLA accessors, a tap,
            # or the official parser. A close racing this snapshot is harmless:
            # each tap checks its own closed state before publishing metadata.
            captures = []
            provider = getattr(callback, "_data_provider", None)
            try:
                with _REGISTRY_LOCK:
                    references = tuple(self.providers.get(id(provider), {}).values())
                for reference in references:
                    tap = reference()
                    if tap is not None and tap._provider is provider and tag == tap.camera_id:
                        try:
                            record = tap._capture(image)
                        except Exception as exc:
                            # Preserve an unexpected tap failure as evidence,
                            # rather than dropping the image or hiding the error.
                            record = tap._unexpected_error(image, exc)
                        captures.append((tap, record))
            finally:
                # Even an unexpected instrumentation failure must not prevent
                # the original parser from receiving this image exactly once.
                previous = getattr(_ACTIVE_CAPTURE, "current", _MISSING)
                _ACTIVE_CAPTURE.current = (provider, tag, captures)
                try:
                    result = original(callback, image, tag)
                finally:
                    if previous is _MISSING:
                        del _ACTIVE_CAPTURE.current
                    else:
                        _ACTIVE_CAPTURE.current = previous
            return result

        self.wrapper = wrapper

    def remove(self, provider_id, tap_id):
        with _REGISTRY_LOCK:
            providers = self.providers.get(provider_id)
            if providers is not None:
                providers.pop(tap_id, None)
                if not providers:
                    self.providers.pop(provider_id, None)
                    parser = self.provider_parsers.pop(provider_id, None)
                    if parser is not None:
                        parser.close()
            if self.providers:
                return
            # Do not overwrite instrumentation installed by another owner later.
            if getattr(self.callback_class, _METHOD, None) is self.wrapper:
                if self.previous_attribute is _MISSING:
                    delattr(self.callback_class, _METHOD)
                else:
                    setattr(self.callback_class, _METHOD, self.previous_attribute)
            if _PATCHES.get(self.callback_class) is self:
                _PATCHES.pop(self.callback_class, None)

    def bind_provider(self, provider):
        identity = id(provider)
        if identity not in self.provider_parsers and callable(getattr(provider, "update_sensor", None)):
            self.provider_parsers[identity] = _ProviderParser(provider)


class _ProviderParser:
    """Bind the exact parser output array before its original queue insertion.

The official parser builds an owned ndarray then passes that same object to
update_sensor. SensorInterface.get_data returns it without copying. Thread-local
callback context prevents parallel camera callbacks from exchanging poses.
"""
    def __init__(self, provider):
        self.provider = provider
        self.original = provider.update_sensor
        self.previous_attribute = getattr(provider, "__dict__", {}).get("update_sensor", _MISSING)

        def wrapper(tag, data, frame):
            context = getattr(_ACTIVE_CAPTURE, "current", None)
            if context is not None and context[0] is provider and context[1] == tag:
                for tap, record in context[2]:
                    try:
                        tap._bind_array(data, frame, record)
                    except Exception as exc:
                        # Instrumentation must not drop an official image.
                        tap._binding_error(frame, exc)
            return self.original(tag, data, frame)

        self.wrapper = wrapper
        provider.update_sensor = wrapper

    def close(self):
        if getattr(self.provider, "update_sensor", None) is self.wrapper:
            if self.previous_attribute is _MISSING:
                delattr(self.provider, "update_sensor")
            else:
                self.provider.update_sensor = self.previous_attribute


class SensorPoseTap:
    def __init__(self, provider, camera_id="front_rgb", camera_mount_matrix_m=None,
                 callback_class=None, max_frames=64):
        if provider is None:
            raise ValueError("sensor pose tap requires a provider identity")
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("camera_id must be a nonempty string")
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        self._provider = provider
        self.camera_id = camera_id
        self._mount = _matrix(camera_mount_matrix_m)
        self._callback_class = callback_class
        self._maximum = max_frames
        self._lock = threading.RLock()
        self._records = {}
        self._array_records = {}
        self._unframed_error = None
        self._patch = None
        self._parser = None
        self._closed = False

    def install(self):
        with self._lock:
            if self._closed:
                raise self._failure("tap_closed", "cannot install a closed sensor pose tap")
            if self._patch is not None:
                return self
            callback_class = self._callback_class
            if callback_class is None:
                from leaderboard.envs.sensor_interface import CallBack
                callback_class = CallBack
            with _REGISTRY_LOCK:
                patch = _PATCHES.get(callback_class)
                current = getattr(callback_class, _METHOD, None)
                if patch is not None and current is not patch.wrapper:
                    raise self._failure("parser_changed", "official image parser changed while taps were active")
                if patch is None:
                    if not inspect.isfunction(current):
                        raise self._failure("parser_unavailable", "official image parser is not an instance method")
                    patch = _Patch(callback_class, current)
                    setattr(callback_class, _METHOD, patch.wrapper)
                    _PATCHES[callback_class] = patch
                provider_id, tap_id = id(self._provider), id(self)
                patch.bind_provider(self._provider)
                reference = weakref.ref(self, lambda _reference, state=patch,
                    owner=provider_id, key=tap_id: state.remove(owner, key))
                patch.providers.setdefault(provider_id, {})[tap_id] = reference
                self._patch = patch
                self._parser = copy.deepcopy(patch.description)
        return self

    def _failure(self, reason, message, **details):
        evidence = {"schema": SCHEMA, "reason": reason, "camera_id": self.camera_id,
                    "capture_method": CAPTURE_METHOD}
        evidence.update(details)
        return SensorPoseError(message, evidence)

    def _capture(self, image):
        with self._lock:
            if self._closed:
                return
        frame = None
        try:
            raw_frame = image.frame
            if type(raw_frame) is not int or raw_frame < 0:
                raise ValueError("raw camera frame must be a nonnegative integer")
            frame = raw_frame
            timestamp = image.timestamp
            if not _finite_number(timestamp) or timestamp < 0:
                raise ValueError("raw camera timestamp must be finite and nonnegative")
            pose = _matrix(image.transform.get_matrix())
            record = {"schema": SCHEMA, "frame": frame, "timestamp": float(timestamp),
                      "camera_id": self.camera_id, "camera_world_matrix_m": pose,
                      "camera_mount_matrix_m": copy.deepcopy(self._mount),
                      "capture_method": CAPTURE_METHOD}
        except Exception as exc:
            record = {"error": {"schema": SCHEMA, "reason": "metadata_copy_failed",
                      "camera_id": self.camera_id, "frame": frame,
                      "capture_method": CAPTURE_METHOD, "error_type": type(exc).__name__,
                      "error": str(exc)}}
        self._publish(frame, record)
        return record

    def _unexpected_error(self, image, exc):
        try:
            frame = image.frame
            if type(frame) is not int or frame < 0:
                frame = None
        except Exception:
            frame = None
        record = {"error": {"schema": SCHEMA, "reason": "metadata_copy_failed",
            "camera_id": self.camera_id, "frame": frame, "capture_method": CAPTURE_METHOD,
            "stage": "unexpected_tap_error", "error_type": type(exc).__name__, "error": str(exc)}}
        self._publish(frame, record)
        return record

    def _binding_error(self, frame, exc):
        self._publish(frame, {"error": {"schema": SCHEMA, "reason": "array_identity_binding_failed",
            "camera_id": self.camera_id, "frame": frame, "capture_method": CAPTURE_METHOD,
            "error_type": type(exc).__name__, "error": str(exc)}})

    def _bind_array(self, array, frame, record):
        if record is None:
            raise ValueError("callback metadata was not captured")
        if record.get("frame", record.get("error", {}).get("frame")) != frame:
            raise ValueError("official parser output frame differs from its raw image")
        identity = id(array)
        owner = weakref.ref(self)

        def discarded(reference):
            tap = owner()
            if tap is not None:
                with tap._lock:
                    saved = tap._array_records.get(identity)
                    if saved is not None and saved[0] is reference:
                        tap._array_records.pop(identity, None)

        reference = weakref.ref(array, discarded)
        with self._lock:
            if self._closed:
                return
            self._array_records[identity] = (reference, copy.deepcopy(record))
            # Queued images retain their own arrays. Only metadata and weak
            # references are held here, with a bound even for duplicate frames.
            while len(self._array_records) > 4*self._maximum:
                self._array_records.pop(next(iter(self._array_records)))

    def _publish(self, frame, record):
        with self._lock:
            if self._closed:
                return
            if frame is None:
                self._unframed_error = record["error"]
                return
            previous = self._records.get(frame)
            if previous is not None and previous != record:
                initial = previous.get("error", {}).get("first_observation", previous)
                if "error" in initial:
                    initial = {"error": initial["error"].get("reason")}
                record = {"error": {"schema": SCHEMA, "reason": "conflicting_frame_metadata",
                          "camera_id": self.camera_id, "frame": frame,
                          "capture_method": CAPTURE_METHOD,
                          "first_observation": copy.deepcopy(initial),
                          "incoming_observation": copy.deepcopy(record),
                          "conflict_count": previous.get("error", {}).get("conflict_count", 0)+1}}
            self._records[frame] = record
            # Retain the latest source frames, even if a late callback is old.
            while len(self._records) > self._maximum:
                self._records.pop(min(self._records))

    def for_frame(self, frame, timestamp, *, pixels=None):
        if type(frame) is not int or frame < 0:
            raise self._failure("invalid_requested_frame", "pose lookup requires a nonnegative integer frame",
                                requested_frame=repr(frame))
        if not _finite_number(timestamp) or timestamp < 0:
            raise self._failure("invalid_requested_timestamp", "pose lookup requires a finite nonnegative timestamp",
                                requested_frame=frame, requested_timestamp=repr(timestamp))
        with self._lock:
            if self._closed:
                raise self._failure("tap_closed", "sensor pose tap is closed", requested_frame=frame)
            if self._patch is None:
                raise self._failure("tap_not_installed", "sensor pose tap has not been installed", requested_frame=frame)
            if pixels is None:
                record = copy.deepcopy(self._records.get(frame))
            else:
                bound = self._array_records.get(id(pixels))
                if bound is None or bound[0]() is not pixels:
                    raise self._failure("source_array_not_bound", "exact official parser image array has no raw pose binding",
                        requested_frame=frame, requested_timestamp=float(timestamp))
                record = copy.deepcopy(bound[1])
                if record.get("frame", record.get("error", {}).get("frame")) != frame:
                    raise self._failure("source_array_frame_mismatch", "requested frame differs from the exact image array",
                        requested_frame=frame, captured_frame=record.get("frame"))
            available = sorted(self._records)
            unframed_error = copy.deepcopy(self._unframed_error)
        if record is None:
            raise self._failure("source_frame_missing", "raw camera pose is unavailable for the exact source frame",
                requested_frame=frame, requested_timestamp=float(timestamp),
                available_frames=available, unframed_error=unframed_error)
        if "error" in record:
            raise SensorPoseError("raw camera pose could not be captured for the source frame", record["error"])
        if abs(record["timestamp"]-float(timestamp)) > TIMESTAMP_TOLERANCE_S:
            raise self._failure("source_timestamp_mismatch", "raw camera pose timestamp differs from the source state",
                requested_frame=frame, requested_timestamp=float(timestamp),
                captured_timestamp=record["timestamp"], tolerance_s=TIMESTAMP_TOLERANCE_S)
        if pixels is not None:
            record["image_selection"] = "exact_official_parser_array_identity"
        return record

    def describe(self):
        with self._lock:
            return {"schema": SCHEMA, "capture_method": CAPTURE_METHOD,
                    "camera_id": self.camera_id, "camera_mount_matrix_m": copy.deepcopy(self._mount),
                    "original_parser": copy.deepcopy(self._parser),
                    "filter": "callback._data_provider is registered_provider and exact camera_id",
                    "instrumentation": "shared class method wrapper; original image parser called exactly once",
                    "image_selection": "optional exact parser output array identity; frame-only lookup rejects conflicts",
                    "lifecycle": "one wrapper per callback class; last close restores only its own wrapper",
                    "installed": self._patch is not None, "closed": self._closed,
                    "max_frames": self._maximum, "timestamp_tolerance_s": TIMESTAMP_TOLERANCE_S,
                    "coordinate_contract": "raw CARLA SensorData.transform in metres; rigid configured mount",
                    "sensor_subscription_changed": False, "world_advanced": False}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._array_records.clear()
            patch, self._patch = self._patch, None
        if patch is not None:
            patch.remove(id(self._provider), id(self))
