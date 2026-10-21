"""Optional per-consultation request observer; no simulator or SDK dependency.

The caller owns persistence. Nothing is recorded outside an explicit scope.
One attempt means one SDK dispatch or local generation, not an unobservable
SDK-internal HTTP retry. Model loading and prompt preparation are not dispatches.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4
import logging

_observer = ContextVar("model_request_observer", default=None)
_request = ContextVar("model_request_description", default=None)


@contextmanager
def observe_model_requests(callback):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


@contextmanager
def describe_model_request(stage, messages, image_paths):
    token = _request.set({"request_id": uuid4().hex, "stage": stage,
                          "messages": messages, "image_paths": list(image_paths)})
    try:
        yield
    finally:
        _request.reset(token)


def emit_model_request(event, **values):
    callback, request = _observer.get(), _request.get()
    if callback is not None and request is not None:
        try:
            callback({**request, "event": event, **values})
        except Exception:
            logging.getLogger(__name__).exception("Model request audit unavailable")
