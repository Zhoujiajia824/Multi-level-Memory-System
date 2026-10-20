"""Video/simulation correspondence, bounded callbacks and real MP4 decoding."""
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from carla_bridge.video import recorder as module


def camera_frame(frame_id, color=128, width=320, height=180, timestamp=None):
    image = np.full((height, width, 4), color, dtype=np.uint8)
    image[:, :, 3] = 255
    return SimpleNamespace(frame=frame_id, timestamp=frame_id * .1 if timestamp is None else timestamp,
                           width=width, height=height, raw_data=image.tobytes())


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def start_mock(tmp_path, monkeypatch, **kwargs):
    writer = Mock()
    writer.isOpened.return_value = True
    monkeypatch.setattr(module.cv2, "VideoWriter", Mock(return_value=writer))
    world = Mock()
    recorder = module.VideoRecorder(world, Mock(), str(tmp_path / "drive.mp4"),
                                    width=320, height=180, fps=10, **kwargs)
    recorder.start()
    return recorder, writer, world


def test_exact_frame_waits_for_callback_without_ticking(tmp_path, monkeypatch):
    recorder, writer, world = start_mock(tmp_path, monkeypatch, frame_timeout_s=.5)
    recorder._receive_frame(camera_frame(49))

    def camera_callback():
        time.sleep(.02)
        recorder._receive_frame(camera_frame(50, timestamp=705.5))

    callback = threading.Thread(target=camera_callback)
    callback.start()
    try:
        assert recorder.on_tick(frame_id=50, sim_time_s=2.4)
        assert writer.write.call_count == 1
        assert rows(recorder.index_path) == [{"video_frame_index": 0, "carla_frame_id": 50,
            "requested_frame": 50, "sim_time_s": 2.4, "sensor_timestamp_s": 705.5,
            "video_time_s": 0.0, "fps": 10.0}]
        assert rows(recorder.events_path)[0]["reason"] == "stale_camera_frame"
        world.tick.assert_not_called()
        world.wait_for_tick.assert_not_called()
    finally:
        callback.join()
        recorder.flush()


def test_future_frame_is_kept_and_missing_target_is_only_retired_on_close(tmp_path, monkeypatch):
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, frame_timeout_s=0, flush_timeout_s=0)
    try:
        recorder._receive_frame(camera_frame(102))
        assert not recorder.on_tick(frame_id=101, sim_time_s=.1)
        assert not recorder.on_tick(frame_id=102, sim_time_s=.2)
        assert not recorder.on_tick(frame_id=103, sim_time_s=.3)
        assert recorder.on_tick(frame_id=102, sim_time_s=.4) is False
        assert writer.write.call_count == 0
        assert recorder.summary()["pending_frame_count"] == 3
        assert recorder.summary()["missing_frame_count"] == 1
    finally:
        recorder.flush()
    assert writer.write.call_count == 1
    assert [row["reason"] for row in rows(recorder.events_path)] == [
        "duplicate_or_out_of_order_request", "unfulfilled_on_close", "unfulfilled_on_close"]
    assert recorder.summary()["missing_frame_count"] == 3
    assert recorder.summary()["pending_frame_count"] == 0
    assert rows(recorder.index_path)[0]["carla_frame_id"] == 102


def test_legacy_no_arguments_records_actual_latest_sensor_time(tmp_path, monkeypatch):
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, hud=False)
    try:
        recorder._receive_frame(camera_frame(12))
        recorder._receive_frame(camera_frame(13, timestamp=91.3))
        assert recorder.on_tick()
        assert rows(recorder.index_path)[0]["carla_frame_id"] == 13
        assert rows(recorder.index_path)[0]["sim_time_s"] == 91.3
        assert rows(recorder.index_path)[0]["requested_frame"] is None
        assert recorder.summary()["stale_frame_count"] == 1
        assert not recorder.on_tick()
        assert writer.write.call_count == 1
    finally:
        recorder.flush()


def test_callback_storage_is_bounded_and_overflow_is_counted(tmp_path, monkeypatch):
    recorder, _, _ = start_mock(tmp_path, monkeypatch, frame_timeout_s=0)
    try:
        for frame_id in range(100):
            recorder._receive_frame(camera_frame(frame_id))
        assert recorder._queue.qsize() == 32
        assert recorder.summary()["queue_overflow_count"] == 68
        assert recorder.on_tick(frame_id=99)
        assert recorder.summary()["encoded_frame_count"] == 1
    finally:
        recorder.flush()


def test_writer_error_has_event_but_no_successful_frame_index(tmp_path, monkeypatch):
    recorder, writer, _ = start_mock(tmp_path, monkeypatch)
    writer.write.side_effect = RuntimeError("disk full")
    try:
        recorder._receive_frame(camera_frame(5))
        assert recorder.on_tick(frame_id=5, sim_time_s=.5) is False
        assert rows(recorder.index_path) == []
        assert rows(recorder.events_path)[0]["reason"] == "encode_error"
        assert recorder.summary()["encode_error_count"] == 1
    finally:
        recorder.flush()


def test_partial_start_failure_closes_sensor_writer_and_indexes_once(tmp_path, monkeypatch):
    world, writer = Mock(), Mock()
    writer.isOpened.return_value = False
    monkeypatch.setattr(module.cv2, "VideoWriter", Mock(return_value=writer))
    recorder = module.VideoRecorder(world, Mock(), str(tmp_path / "drive.mp4"))
    with pytest.raises(RuntimeError, match="无法打开"):
        recorder.start()
    recorder.flush()
    recorder.flush()
    world.spawn_actor.return_value.stop.assert_called_once()
    world.spawn_actor.return_value.destroy.assert_called_once()
    writer.release.assert_called_once()
    assert recorder._index_file is None
    assert recorder._events_file is None
    summary = json.loads(recorder.summary_path.read_text(encoding="utf-8"))
    assert summary["closed"]
    assert summary["encoded_frame_count"] == 0
    assert summary["encode_error_count"] == 1


def test_flush_on_interrupt_keeps_video_index_and_stops_late_callbacks(tmp_path, monkeypatch):
    recorder, writer, world = start_mock(tmp_path, monkeypatch)
    recorder._receive_frame(camera_frame(1))
    recorder.on_tick(frame_id=1, sim_time_s=.1)
    recorder.flush()
    recorder._receive_frame(camera_frame(2))
    assert not recorder.on_tick(frame_id=2, sim_time_s=.2)
    recorder.flush()
    assert recorder._queue.empty()
    assert len(rows(recorder.index_path)) == 1
    assert recorder.manifest()["closed"]
    assert recorder.manifest()["video_duration_s"] == .1
    writer.release.assert_called_once()
    world.spawn_actor.return_value.destroy.assert_called_once()


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": float("nan")},
                                   {"frame_timeout_s": -1}, {"width": 0},
                                   {"max_pending_frames": 0}, {"flush_timeout_s": -1}])
def test_invalid_video_configuration_fails_before_creating_sensor(tmp_path, kwargs):
    with pytest.raises(ValueError):
        module.VideoRecorder(Mock(), Mock(), str(tmp_path / "test.mp4"), **kwargs)


def test_real_mp4_frames_decode_and_hud_changes_pixels(tmp_path):
    """Uses real OpenCV codec, no CARLA service or model inference."""
    recorder = module.VideoRecorder(Mock(), Mock(), str(tmp_path / "real.mp4"),
                                    fps=10, width=320, height=180, frame_timeout_s=0)
    recorder.start()
    try:
        for i in range(3):
            recorder._receive_frame(camera_frame(300 + i, color=160))
            assert recorder.on_tick(frame_id=300 + i, sim_time_s=i * .1,
                overlay={"model_call_count": 2, "latest_call_id": "call_2",
                         "latest_trigger": "blocked_lane", "selected_id": 5, "status": "active"})
    finally:
        recorder.flush()
    capture = cv2.VideoCapture(recorder.output_path)
    decoded = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        decoded.append(image)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    assert len(decoded) == 3
    assert fps == pytest.approx(10)
    assert np.mean(decoded[0][:98]) < np.mean(decoded[0][120:]) - 50
    indexed = rows(recorder.index_path)
    assert [row["carla_frame_id"] for row in indexed] == [300, 301, 302]
    assert [row["video_time_s"] for row in indexed] == [0.0, .1, .2]
    assert [row["sim_time_s"] for row in indexed] == [0.0, .1, .2]


def test_two_tick_delayed_callbacks_use_original_times_and_hud(tmp_path, monkeypatch):
    recorder, writer, world = start_mock(tmp_path, monkeypatch, flush_timeout_s=0)
    hud = []
    monkeypatch.setattr(recorder, "_draw_hud", lambda image, row, overlay:
                        hud.append((row["carla_frame_id"], overlay["model_call_count"])))
    for tick in range(6):
        if tick >= 2:
            recorder._receive_frame(camera_frame(100 + tick - 2))
        recorder.on_tick(frame_id=100 + tick, sim_time_s=tick * .1,
                         overlay={"model_call_count": tick})
    assert recorder.summary()["pending_frame_count"] == 2
    assert recorder.summary()["missing_frame_count"] == 0
    for tick in (4, 5):
        recorder._receive_frame(camera_frame(100 + tick))
    recorder.flush()
    indexed = rows(recorder.index_path)
    assert [row["carla_frame_id"] for row in indexed] == list(range(100, 106))
    assert [row["sim_time_s"] for row in indexed] == [tick * .1 for tick in range(6)]
    assert [row["video_time_s"] for row in indexed] == [tick / 10 for tick in range(6)]
    assert hud == [(100 + tick, tick) for tick in range(6)]
    assert writer.write.call_count == 6
    assert recorder.summary()["missing_frame_count"] == 0
    world.tick.assert_not_called()


def test_out_of_order_images_wait_for_missing_older_frame(tmp_path, monkeypatch):
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, flush_timeout_s=0)
    for frame_id in (10, 11, 12):
        recorder.on_tick(frame_id=frame_id, sim_time_s=frame_id / 10)
    recorder._receive_frame(camera_frame(12))
    recorder._receive_frame(camera_frame(10))
    recorder.on_tick(frame_id=13, sim_time_s=1.3)
    assert [row["carla_frame_id"] for row in rows(recorder.index_path)] == [10]
    recorder._receive_frame(camera_frame(11))
    recorder.on_tick(frame_id=14, sim_time_s=1.4)
    assert [row["carla_frame_id"] for row in rows(recorder.index_path)] == [10, 11, 12]
    recorder.flush()
    assert writer.write.call_count == 3
    assert recorder.summary()["unfulfilled_frame_count"] == 2


def test_pending_window_expires_missing_frames_and_late_images_stay_dropped(tmp_path, monkeypatch):
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, max_pending_frames=3, flush_timeout_s=0)
    for frame_id in (1, 2, 3, 4):
        recorder.on_tick(frame_id=frame_id, sim_time_s=frame_id / 10)
    assert recorder.summary()["pending_frame_count"] == 3
    assert recorder.summary()["expired_frame_count"] == 1
    recorder._receive_frame(camera_frame(4))
    recorder._receive_frame(camera_frame(1))
    recorder.flush()
    assert [row["carla_frame_id"] for row in rows(recorder.index_path)] == [4]
    assert [row["reason"] for row in rows(recorder.events_path)] == [
        "pending_window_expired", "stale_camera_frame", "unfulfilled_on_close", "unfulfilled_on_close"]
    assert recorder.summary()["missing_frame_count"] == 3
    assert recorder.summary()["stale_frame_count"] == 1
    assert writer.write.call_count == 1


def test_final_callback_is_drained_within_bounded_flush_without_world_tick(tmp_path, monkeypatch):
    recorder, writer, world = start_mock(tmp_path, monkeypatch, flush_timeout_s=.3)
    recorder.on_tick(frame_id=7, sim_time_s=.7, overlay={"model_call_count": 1})

    def camera_callback():
        time.sleep(.02)
        recorder._receive_frame(camera_frame(7))

    callback = threading.Thread(target=camera_callback)
    callback.start()
    recorder.flush()
    callback.join()
    assert writer.write.call_count == 1
    assert rows(recorder.index_path)[0]["sim_time_s"] == .7
    assert recorder.summary()["unfulfilled_frame_count"] == 0
    world.tick.assert_not_called()


def test_default_registration_does_not_perform_blocking_wait(tmp_path, monkeypatch):
    recorder, _, _ = start_mock(tmp_path, monkeypatch, flush_timeout_s=0)
    waits = []
    real_get = recorder._queue.get

    def observed_get(*args, **kwargs):
        waits.append((args, kwargs))
        return real_get(*args, **kwargs)

    monkeypatch.setattr(recorder._queue, "get", observed_get)
    recorder.on_tick(frame_id=9, sim_time_s=.9)
    assert all(args == () and kwargs.get("block") is False for args, kwargs in waits)
    assert recorder.summary()["pending_frame_count"] == 1
    assert recorder.summary()["missing_frame_count"] == 0
    recorder.flush()
    assert recorder.summary()["unfulfilled_frame_count"] == 1
