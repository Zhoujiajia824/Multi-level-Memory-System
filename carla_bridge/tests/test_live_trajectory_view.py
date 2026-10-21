"""Real local HTTP, immutable frame metadata and nonblocking preview sinks."""
import json
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

import cv2
import numpy as np
import pytest

from carla_bridge.visualization import live_view as module


def request(view, path):
    return build_opener(ProxyHandler({})).open(view.url + path.lstrip("/"), timeout=2.)


def wait_for(view, predicate, timeout=2.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = view.summary()
        if predicate(result):
            return result
        time.sleep(.005)
    pytest.fail(f"preview did not reach expected state: {view.summary()}")


@pytest.fixture
def view():
    result = module.LiveTrajectoryView({"enabled": True, "port": 0})
    result.start()
    try:
        yield result
    finally:
        result.close()


def pixels(value=90):
    return np.full((48, 80, 3), value, dtype=np.uint8)


def test_off_has_no_server_worker_encoder_or_input_copy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled preview must not allocate a sink")
    monkeypatch.setattr(module, "ThreadingHTTPServer", forbidden)
    monkeypatch.setattr(module.threading, "Thread", forbidden)
    monkeypatch.setattr(module.cv2, "imencode", forbidden)
    preview = module.LiveTrajectoryView()
    assert preview.start() is None
    assert not preview.publish(object(), {"invalid": float("nan")})
    preview.close()
    preview.close()
    assert preview.summary()["state"] == "disabled"
    assert preview.url is None and preview._worker is None and preview._server is None


@pytest.mark.parametrize("config", [
    {"enabled": "on"}, {"port": -1}, {"port": 65536}, {"host": ""},
    {"jpeg_quality": 0}, {"jpeg_quality": 101},
    {"stale_after_s": 0}, {"stale_after_s": float("nan")},
])
def test_invalid_config_fails_before_allocating_socket(config, monkeypatch):
    monkeypatch.setattr(module, "ThreadingHTTPServer", lambda *args: pytest.fail("unexpected socket"))
    with pytest.raises(ValueError):
        module.LiveTrajectoryView(config)


def test_waiting_page_status_and_no_arbitrary_file_service(view):
    assert view.url.startswith("http://127.0.0.1:") and view.port > 0
    assert view.start() == view.url
    with request(view, "/") as response:
        page = response.read().decode("utf-8")
        assert "X-Frame-Metadata" in page and "Connected" in page and "Disconnected" in page
        assert "frame age" in page and "future plans" in page
        assert response.headers["Cache-Control"] == "no-store"
    with request(view, "/status.json") as response:
        status = json.load(response)
        assert status["state"] == "waiting" and status["frame"] is None
        assert status["host"] == "127.0.0.1" and status["started"] is True
    for path, expected in (("/frame.jpg", 503), ("/../../config/api_models.yaml", 404),
                           ("/%2e%2e/config/api_models.yaml", 404), ("/manifest.json", 404)):
        with pytest.raises(HTTPError) as error:
            request(view, path)
        assert error.value.code == expected
        error.value.close()


def test_jpeg_and_response_metadata_are_one_atomic_capture(view, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_encode = module.cv2.imencode
    captured = []

    def encode(extension, image, options):
        entered.set()
        assert release.wait(2.)
        captured.append(image.copy())
        return original_encode(extension, image, options)

    monkeypatch.setattr(module.cv2, "imencode", encode)
    image = pixels(67)
    metadata = {"carla_frame_id": 123, "sim_time_s": 1.7, "selected_id": "lane_中文",
                "details": {"value": [2]}}
    try:
        assert view.publish(image, metadata)
        assert entered.wait(1.)
        image[:] = 0
        metadata["carla_frame_id"] = 999
        metadata["details"]["value"][0] = 999
    finally:
        release.set()
    wait_for(view, lambda s: s["encoded_frame_count"] == 1)
    with request(view, "/frame.jpg") as response:
        header = json.loads(response.headers["X-Frame-Metadata"])
        decoded = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        assert header == {"carla_frame_id": 123, "sim_time_s": 1.7, "selected_id": "lane_中文"}
        assert response.headers["X-Live-Sequence"] == "1"
        assert response.headers["X-Live-State"] == "live"
        assert float(response.headers["X-Frame-Age"]) >= 0
        assert decoded.shape == image.shape and np.allclose(decoded, 67, atol=1)
    assert np.all(captured[0] == 67)
    status = view.summary()
    assert status["frame"]["details"]["value"] == [2]
    status["frame"]["details"]["value"][0] = 300
    assert view.summary()["frame"]["details"]["value"] == [2]


def test_blocked_jpeg_encoder_keeps_only_latest_frame_without_blocking_publish(view, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_encode = module.cv2.imencode
    first = True

    def encode(*args):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(2.)
        return original_encode(*args)

    monkeypatch.setattr(module.cv2, "imencode", encode)
    assert view.publish(pixels(), {"carla_frame_id": 0})
    assert entered.wait(1.)
    accepted = []

    def produce():
        for frame in range(1, 61):
            accepted.append(view.publish(pixels(frame), {"carla_frame_id": frame}))

    producer = threading.Thread(target=produce)
    try:
        producer.start()
        producer.join(timeout=.75)
        assert not producer.is_alive(), "publish waited for the blocked encoder"
        assert len(accepted) == 60 and all(accepted)
        state = view.summary()
        assert state["pending_frame_count"] == 1 and state["queue_capacity"] == 1
        assert state["dropped_frame_count"] == 59
    finally:
        release.set()
        producer.join(timeout=2.)
    state = wait_for(view, lambda s: (s["frame"] or {}).get("carla_frame_id") == 60)
    assert state["encoded_frame_count"] == 2


def test_mjpeg_stream_contains_exact_frame_and_valid_jpeg(view):
    assert view.publish(pixels(131), {"carla_frame_id": 341, "sim_time_s": 2.1})
    wait_for(view, lambda s: s["encoded_frame_count"] == 1)
    with request(view, "/stream.mjpg") as response:
        assert response.headers["Content-Type"] == "multipart/x-mixed-replace; boundary=frame"
        assert response.readline() == b"--frame\r\n"
        headers = {}
        while True:
            line = response.readline()
            if line == b"\r\n":
                break
            key, value = line.decode("ascii").split(":", 1)
            headers[key] = value.strip()
        assert headers["X-Carla-Frame"] == "341"
        jpeg = response.read(int(headers["Content-Length"]))
        assert response.read(2) == b"\r\n"
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert np.allclose(decoded, 131, atol=1)


def test_blocked_stream_write_does_not_hold_publication_lock(view):
    assert view.publish(pixels(), {"carla_frame_id": 10})
    wait_for(view, lambda s: s["encoded_frame_count"] == 1)
    entered, release = threading.Event(), threading.Event()

    class SlowSocket:
        def write(self, data):
            entered.set()
            assert release.wait(2.)
            raise BrokenPipeError("test browser disconnected")

        def flush(self):
            pass

    handler = object.__new__(view._server.RequestHandlerClass)
    handler.send_response = handler.send_header = handler.end_headers = lambda *args: None
    handler.wfile = SlowSocket()

    def stream():
        try:
            handler._stream()
        except BrokenPipeError:
            pass

    client = threading.Thread(target=stream)
    try:
        client.start()
        assert entered.wait(1.)
        assert view.publish(pixels(180), {"carla_frame_id": 11})
        state = wait_for(view, lambda s: (s["frame"] or {}).get("carla_frame_id") == 11)
        assert state["client_count"] == 1
    finally:
        release.set()
        client.join(timeout=2.)
    assert not client.is_alive() and view.summary()["client_count"] == 0


def test_encoder_exception_is_isolated_and_worker_recovers(view, monkeypatch):
    original_encode = module.cv2.imencode

    def broken(*args):
        raise RuntimeError("test JPEG failure")

    monkeypatch.setattr(module.cv2, "imencode", broken)
    assert view.publish(pixels(), {"carla_frame_id": 1})
    state = wait_for(view, lambda s: s["jpeg_error_count"] == 1)
    assert state["state"] == "waiting" and state["last_error"] == "RuntimeError"
    monkeypatch.setattr(module.cv2, "imencode", original_encode)
    assert view.publish(pixels(), {"carla_frame_id": 2})
    assert wait_for(view, lambda s: s["encoded_frame_count"] == 1)["frame"]["carla_frame_id"] == 2


@pytest.mark.parametrize("image,metadata", [
    (None, {}), (np.zeros((3, 3, 3), dtype=float), {}),
    (np.zeros((3, 3, 4), dtype=np.uint8), {}),
    (np.zeros((0, 3, 3), dtype=np.uint8), {}),
    (pixels(), [1]), (pixels(), {"time": float("nan")}),
])
def test_invalid_optional_input_returns_false_without_killing_server(view, image, metadata):
    assert not view.publish(image, metadata)
    assert view.summary()["publish_error_count"] == 1
    with request(view, "/status.json") as response:
        assert json.load(response)["state"] == "waiting"


def test_stale_state_retains_image_and_closed_state_releases_resources(view):
    assert view.publish(pixels(), {"carla_frame_id": 80})
    wait_for(view, lambda s: s["encoded_frame_count"] == 1)
    with view._condition:
        seq, jpeg, metadata, _, wall_clock = view._latest
        view._latest = (seq, jpeg, metadata, time.monotonic() - 10., wall_clock)
    assert view.summary()["state"] == "stale"
    with request(view, "/frame.jpg") as response:
        assert response.headers["X-Live-State"] == "stale"
        assert json.loads(response.headers["X-Frame-Metadata"])["carla_frame_id"] == 80
    worker, server_thread = view._worker, view._server_thread
    view.close()
    view.close()
    assert not worker.is_alive() and not server_thread.is_alive()
    assert view.summary()["state"] == "closed"
    assert view.summary()["pending_frame_count"] == 0
    assert not view.publish(pixels(), {})
    with pytest.raises(URLError):
        request(view, "/status.json")
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        view.start()


def test_port_conflict_never_starts_worker():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        preview = module.LiveTrajectoryView({"enabled": True, "port": occupied.getsockname()[1]})
        try:
            with pytest.raises(OSError):
                preview.start()
            assert preview._worker is None and preview._server_thread is None
        finally:
            preview.close()


def test_partial_startup_failure_cleans_worker_and_bound_socket(monkeypatch):
    original_start = threading.Thread.start

    def start(thread):
        if thread.name == "carla-live-http":
            raise RuntimeError("test HTTP thread failure")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    preview = module.LiveTrajectoryView({"enabled": True, "port": 0})
    with pytest.raises(RuntimeError, match="HTTP thread failure"):
        preview.start()
    assert preview.summary()["state"] == "closed"
    assert not preview._worker.is_alive()
    assert preview._server.socket.fileno() == -1
    preview.close()
