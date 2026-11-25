"""NAVSIM-only bounded, audited streaming requests; no shared-client changes.

The unchanged response schema remains authoritative. Compact JSON is a declared
deployment requirement; request-level whitespace flags in vLLM 0.32 do not
override its backend configuration and are deliberately not sent here.
"""
import asyncio
import base64
from contextlib import asynccontextmanager
import copy
import inspect
import math
from pathlib import Path
import time

from nine_grid.models import CommunicationFailure, VisionClient


TRANSPORT_PROTOCOL = "navsim-bounded-stream-1.0"
DEFAULT_POLICY = dict(request_timeout_s=120., total_deadline_s=300., transport_retry_limit=1)
OUTSIDE_WHITESPACE_LIMIT = 256
MAX_RESPONSE_CHARACTERS = 65536


def model_request_policy(settings):
    """Normalize one policy for both a frozen identity and the actual client."""
    value = settings.get("model_request_policy", {})
    if not isinstance(value, dict) or set(value) - set(DEFAULT_POLICY):
        raise ValueError("Unknown NAVSIM model request policy")
    policy = dict(DEFAULT_POLICY, **value)
    for key in ("request_timeout_s", "total_deadline_s"):
        number = policy[key]
        if (isinstance(number, bool) or not isinstance(number, (float, int))
                or not math.isfinite(number) or number <= 0):
            raise ValueError("NAVSIM " + key + " must be finite and positive")
        policy[key] = float(number)
    retries = policy["transport_retry_limit"]
    if type(retries) is not int or retries not in (0, 1):
        raise ValueError("NAVSIM transport_retry_limit must be zero or one")
    if policy["request_timeout_s"] > policy["total_deadline_s"]:
        raise ValueError("NAVSIM request timeout exceeds its shared phase deadline")
    return policy


class ModelRequestFailure(CommunicationFailure):
    validation_stage = "transport_error"

    @property
    def safe_detail(self):
        # All our messages are constructed locally. Remote exception bodies
        # (which may contain headers or credentials) never enter this string.
        return str(self)


class ModelOutputTruncated(ModelRequestFailure):
    validation_stage = "response_truncated"


class ModelWhitespaceLoop(ModelRequestFailure):
    validation_stage = "response_whitespace_loop"


class ModelDeadlineExceeded(ModelRequestFailure):
    validation_stage = "model_deadline"


class ModelResponseInvalid(ModelRequestFailure):
    validation_stage = "response_format"


class ModelAuditFailure(ModelRequestFailure):
    validation_stage = "model_audit_error"


class _WhitespaceGuard:
    """Detect only JSON syntax whitespace, never spaces inside quoted reasons."""
    def __init__(self):
        self.quoted, self.escaped, self.run = False, False, 0
        self.maximum = 0

    def feed(self, text):
        for char in text:
            if self.quoted:
                self.run = 0
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.quoted = False
            elif char == '"':
                self.quoted, self.run = True, 0
            elif char in " \t\r\n":
                self.run += 1
                self.maximum = max(self.maximum, self.run)
                if self.run >= OUTSIDE_WHITESPACE_LIMIT:
                    raise ModelWhitespaceLoop("JSON generation repeated syntax whitespace; request cancelled")
            else:
                self.run = 0


def _retryable(exc):
    # An output failure is not a failed network connection. In particular,
    # length/whitespace/malformed-finish must never re-enter this retry loop.
    if isinstance(exc, ModelRequestFailure):
        return False
    from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
    return isinstance(exc, (TimeoutError, ConnectionError, APIConnectionError,
                            APITimeoutError, RateLimitError, InternalServerError))


class BoundedStructuredVisionClient(VisionClient):
    """A bounded server client retaining VisionClient's public state contract.

    A worker passes the same monotonic ``deadline`` to all semantic attempts.
    ``on_attempt`` runs immediately after closing each HTTP stream, with that
    record already appended to ``last_transport_attempts``. ``on_progress`` is a
    separate, optional heartbeat; it does not manufacture completed attempts.
    """
    def __init__(self, role, settings, **kwargs):
        if settings.get("backend") != "server":
            raise ValueError("Bounded NAVSIM structured output requires backend=server")
        super().__init__(role, settings, **kwargs)
        self.policy = model_request_policy(settings)
        self.settings["model_request_policy"] = dict(self.policy)
        self.settings["transport_protocol"] = TRANSPORT_PROTOCOL
        self.last_transport_attempts = []
        self.phase_deadline = None
        self.attempt_callback = None
        self._default_sync_client = self.client if kwargs.get("transport") is None else None

    def set_phase_deadline(self, deadline):
        if deadline is not None and (isinstance(deadline, bool)
                or not isinstance(deadline, (float, int)) or not math.isfinite(deadline)):
            raise ValueError("Model phase deadline must be a monotonic finite timestamp")
        self.phase_deadline = deadline

    def set_attempt_callback(self, callback):
        if callback is not None and not callable(callback):
            raise ValueError("Model attempt callback must be callable")
        self.attempt_callback = callback

    def call(self, prompt, image_paths):
        raise ValueError("NAVSIM bounded client requires the explicit response schema")

    def call_structured(self, prompt, image_paths, response_format, *, deadline=None,
                        on_attempt=None, on_progress=None):
        self.last_transport_attempts = []
        if (not isinstance(response_format, dict) or response_format.get("type") != "json_schema"
                or response_format.get("json_schema", {}).get("strict") is not True):
            raise ValueError("NAVSIM requires a strict JSON response schema")
        started = time.monotonic()
        phase_end = deadline if deadline is not None else self.phase_deadline
        if phase_end is not None and (isinstance(phase_end, bool)
                or not isinstance(phase_end, (float, int)) or not math.isfinite(phase_end)):
            raise ValueError("Model phase deadline is invalid")
        phase_end = min(phase_end if phase_end is not None else float("inf"),
                        started+self.policy["total_deadline_s"])
        callback = on_attempt if on_attempt is not None else self.attempt_callback
        content = [{"type": "text", "text": prompt}]
        for filename in image_paths:
            path = Path(filename)
            mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(dict(type="image_url", image_url=dict(url="data:"+mime+";base64,"+encoded)))
        image_count = len(content)-1
        if image_count != 9:
            raise ValueError("NAVSIM structured server request requires all eight RGB views and BEV")
        return asyncio.run(self._call_async(content, image_count, response_format,
            phase_end, callback, on_progress))

    @asynccontextmanager
    async def _request_client(self):
        if self.client is self._default_sync_client:
            # A client/pool belongs to exactly one event loop and one call. In
            # particular, do not reuse async pools across asyncio.run loops or
            # mutate the shared synchronous client's timeout during a request.
            from openai import AsyncOpenAI
            async with AsyncOpenAI(api_key=self.client.api_key,
                    base_url=str(self.client.base_url), max_retries=0) as client:
                yield client
        else:
            # Explicit injected transports support deterministic offline replay.
            # Production HTTP always uses the cancellable async client above.
            yield self.client

    async def _call_async(self, content, image_count, response_format, phase_end,
                          callback, on_progress):
        async with self._request_client() as client:
            return await self._call_with_client(client, content, image_count,
                response_format, phase_end, callback, on_progress)

    async def _call_with_client(self, client, content, image_count, response_format,
                                phase_end, callback, on_progress):
        for index in range(self.policy["transport_retry_limit"]+1):
            remaining = phase_end-time.monotonic()
            if remaining <= 0:
                raise ModelDeadlineExceeded("Shared model phase deadline expired before request")
            budget = min(remaining, self.policy["request_timeout_s"])
            request_end = time.monotonic()+budget
            row = dict(protocol=TRANSPORT_PROTOCOL, attempt=index+1,
                       max_tokens=self.settings.get("max_tokens", 4096), image_count=image_count,
                       request_timeout_s=budget, choices=[], usage=None, stream=True)
            raw, guard = [], _WhitespaceGuard()
            state = dict(stream=None, finish=None, first_content=None, last_progress=0., characters=0)
            request_start = time.monotonic()
            error = None
            self.attempts += 1
            try:
                if on_progress is not None:
                    on_progress(dict(event="request_started", attempt=index+1, remaining_s=remaining))
                # HTTPX timeouts alone are per-read inactivity limits. wait_for
                # cancels the active connect/prefill/SSE read at this wall-clock
                # deadline, even when no next chunk ever arrives. Awaiting its
                # cancellation and closing this call's client leaves no worker
                # thread, pending HTTP request, or shared timeout mutation.
                try:
                    await asyncio.wait_for(self._exchange(client, content,
                        response_format, row, raw, guard, state, request_start,
                        request_end, budget, index+1, on_progress), timeout=budget)
                except asyncio.TimeoutError:
                    if state["finish"] == "length":
                        raise ModelOutputTruncated("Model output reached token limit before normal completion") from None
                    raise ModelDeadlineExceeded("Bounded model request deadline expired") from None
            except BaseException as exc:
                error = exc
                row["error"] = type(exc).__name__
                row["validation_stage"] = ("model_interrupted" if not isinstance(exc, Exception)
                    else getattr(exc, "validation_stage", "transport_error"))
                if isinstance(exc, Exception) and not isinstance(exc, ModelRequestFailure):
                    row["transport_error"] = type(exc).__name__
                    status = getattr(exc, "status_code", None)
                    row["http_status"] = status if type(status) is int else None
            finally:
                stream = state["stream"]
                if stream is not None and hasattr(stream, "close"):
                    try:
                        closed = stream.close()
                        if inspect.isawaitable(closed):
                            await closed
                    except Exception as exc:
                        row["stream_close_error"] = type(exc).__name__
                row.update(wall_s=time.monotonic()-request_start, time_to_first_content_s=state["first_content"],
                           max_outside_whitespace=guard.maximum,
                           choices=[dict(finish_reason=state["finish"], raw="".join(raw))])
                if error is None:
                    row["validation_stage"] = "response_complete"
                self.last_transport_attempts.append(row)
                if callback is not None:
                    try:
                        callback(copy.deepcopy(row))
                    except Exception as exc:
                        raise ModelAuditFailure("Immediate model attempt audit could not be saved") from exc
            if error is None:
                return row["choices"][0]["raw"]
            if not isinstance(error, Exception):
                raise error
            if isinstance(error, ModelRequestFailure):
                raise error
            if not _retryable(error) or index >= self.policy["transport_retry_limit"]:
                raise ModelRequestFailure("NAVSIM model transport failed: "+type(error).__name__) from None
            delay = min(float(2**index), max(0., phase_end-time.monotonic()))
            if delay <= 0:
                raise ModelDeadlineExceeded("Shared model phase deadline expired after transport failure")
            if self.sleeper is time.sleep:
                await asyncio.sleep(delay)
            else:
                self.sleeper(delay)
        raise AssertionError("Unreachable bounded model request state")

    async def _exchange(self, client, content, response_format, row, raw, guard,
                        state, request_start, request_end, budget, attempt,
                        on_progress):
        stream = client.chat.completions.create(model=self.settings["model"],
            messages=[dict(role="system", content=self.system_prompt),
                      dict(role="user", content=content)],
            response_format=copy.deepcopy(response_format), temperature=0.,
            max_tokens=row["max_tokens"], stream=True,
            stream_options=dict(include_usage=True), timeout=budget)
        if inspect.isawaitable(stream):
            stream = await stream
        state["stream"] = stream
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                self._consume_chunk(chunk, row, raw, guard, state, request_start,
                    request_end, attempt, on_progress)
        elif hasattr(stream, "__iter__"):
            for chunk in stream:
                self._consume_chunk(chunk, row, raw, guard, state, request_start,
                    request_end, attempt, on_progress)
        else:
            raise ModelResponseInvalid("Model endpoint did not honor streaming")
        if state["finish"] == "length":
            raise ModelOutputTruncated("Model output reached token limit before normal completion")
        if state["finish"] != "stop":
            raise ModelResponseInvalid("Model stream ended without a normal finish marker")

    @staticmethod
    def _consume_chunk(chunk, row, raw, guard, state, request_start, request_end,
                       attempt, on_progress):
        now = time.monotonic()
        if now >= request_end:
            raise ModelDeadlineExceeded("Bounded model request deadline expired")
        row["response_model"] = getattr(chunk, "model", row.get("response_model"))
        row["request_id"] = getattr(chunk, "id", row.get("request_id"))
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            row["usage"] = {key: getattr(usage, key, None) for key in
                            ("prompt_tokens", "completion_tokens", "total_tokens")}
        for choice in getattr(chunk, "choices", []):
            if getattr(choice, "index", 0) != 0:
                raise ModelResponseInvalid("Unexpected multiple model completions")
            received_finish = getattr(choice, "finish_reason", None)
            if received_finish is not None:
                state["finish"] = received_finish
            text = getattr(getattr(choice, "delta", None), "content", None)
            if text:
                if not isinstance(text, str):
                    raise ModelResponseInvalid("Model content is not textual JSON")
                raw.append(text)
                state["characters"] += len(text)
                if state["first_content"] is None:
                    state["first_content"] = now-request_start
                guard.feed(text)
                if state["characters"] > MAX_RESPONSE_CHARACTERS:
                    raise ModelResponseInvalid("Model response exceeded its bounded raw audit size")
            if received_finish is not None and received_finish not in ("stop", "length"):
                raise ModelResponseInvalid("Model did not complete with finish_reason=stop")
        # Keep reading the provider's usage-only tail after a length marker,
        # while retaining it as a failure. No partial JSON is repaired/accepted.
        if on_progress is not None and now-state["last_progress"] >= 5.:
            on_progress(dict(event="response_progress", attempt=attempt,
                             characters=state["characters"], elapsed_s=now-request_start))
            state["last_progress"] = now
