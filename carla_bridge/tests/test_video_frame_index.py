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


def start_mock(tmp_path, monkeypatch, width=320, height=180, **kwargs):
    writer = Mock()
    writer.isOpened.return_value = True
    monkeypatch.setattr(module.cv2, "VideoWriter", Mock(return_value=writer))
    world = Mock()
    recorder = module.VideoRecorder(world, Mock(), str(tmp_path / "drive.mp4"),
                                    width=width, height=height, fps=10, **kwargs)
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


@pytest.mark.parametrize("width,height", [(1280, 720), (960, 540), (640, 360), (320, 180)])
def test_trajectory_video_legend_has_real_rgb_colors_bounded_area_and_planning_labels(
        tmp_path, monkeypatch, width, height):
    recorder = module.VideoRecorder(Mock(), Mock(), str(tmp_path / "legend.mp4"),
                                    width=width, height=height)
    image = np.full((height, width, 3), 160, dtype=np.uint8)
    labels = []
    original_put_text = module.cv2.putText

    def observed_text(image, text, *args, **kwargs):
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(module.cv2, "putText", observed_text)
    recorder._draw_hud(image, {"carla_frame_id": 91, "sim_time_s": 3.2},
        {"selected_id": "left:CRUISE", "latest_trigger": "blocked_lane",
         "trajectory_visualization": {"enabled": True, "style": "paper", "drawn_candidates": 8,
            "candidate_pool_count": 40, "merged_candidates": 20, "hidden_candidates": 12}})
    for color in module.TRAJECTORY_HUD_COLORS.values():
        assert max(color) > 32  # Ordinary muted BGR, never world-debug 0/1/2 input.
        assert np.any(np.all(image == color, axis=2))
    assert any("8/40" in label for label in labels)
    assert any("20" in label and "12" in label for label in labels)
    assert any("Selected plan" in label for label in labels)
    assert any("Accepted c" in label for label in labels)
    assert any("Rejected c" in label for label in labels)
    assert all(label.isascii() for label in labels)
    if width >= 800:
        assert "Plans, not the vehicle's driven trail" in labels
        assert np.all(image[96:] == 160)
    else:
        # At the smallest supported fixture only the top 70 pixels change.
        assert np.all(image[70:] == 160)


@pytest.mark.parametrize("visual", [None, {}, {"enabled": False}])
def test_disabled_trajectory_drawing_keeps_the_original_video_hud(tmp_path, monkeypatch, visual):
    recorder = module.VideoRecorder(Mock(), Mock(), str(tmp_path / "legend.mp4"), width=960, height=540)
    image = np.full((540, 960, 3), 160, dtype=np.uint8)
    before = image.copy()
    row = {"carla_frame_id": 91, "sim_time_s": 3.2}
    overlay = {"selected_id": "center:CRUISE"}
    recorder._draw_hud(before, row, overlay)
    recorder._draw_hud(image, row, {**overlay, "trajectory_visualization": visual})
    np.testing.assert_array_equal(image, before)


def test_trajectory_draw_error_does_not_invent_successful_candidate_counts(tmp_path, monkeypatch):
    recorder = module.VideoRecorder(Mock(), Mock(), str(tmp_path / "legend.mp4"), width=1280, height=720)
    labels = []
    monkeypatch.setattr(module.cv2, "putText", lambda image, text, *args, **kwargs: labels.append(text))
    recorder._draw_hud(np.zeros((720, 1280, 3), dtype=np.uint8),
        {"carla_frame_id": 91, "sim_time_s": 3.2},
        {"trajectory_visualization": {"enabled": True, "error": {"type": "RuntimeError"}}})
    assert any("draw error" in label and "-/-" in label for label in labels)


def test_delayed_video_frame_keeps_its_own_trajectory_legend_counts(tmp_path, monkeypatch):
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, flush_timeout_s=0)
    drawn = []
    monkeypatch.setattr(recorder, "_draw_hud", lambda image, row, overlay:
        drawn.append((row["carla_frame_id"], overlay["trajectory_visualization"]["drawn_candidates"])))
    visual = {"enabled": True, "drawn_candidates": 8}
    recorder.on_tick(frame_id=10, sim_time_s=1., overlay={"trajectory_visualization": visual})
    visual["drawn_candidates"] = 2
    recorder.on_tick(frame_id=11, sim_time_s=1.1, overlay={"trajectory_visualization": visual})
    recorder._receive_frame(camera_frame(10))
    recorder._receive_frame(camera_frame(11))
    recorder.flush()
    assert drawn == [(10, 8), (11, 2)]
    assert writer.write.call_count == 2


@pytest.mark.parametrize("style", ["paper", "classic"])
def test_video_legend_matches_the_selected_display_style_with_continuous_samples(tmp_path, monkeypatch, style):
    from carla_bridge.visualization.trajectory_renderer import DISPLAY_COLORS, DISPLAY_CANDIDATE_COLORS
    recorder = module.VideoRecorder(Mock(), Mock(), str(tmp_path / "legend.mp4"),
                                    width=1280, height=720)
    image = np.full((720, 1280, 3), 160, dtype=np.uint8)
    calls = []
    original_line = module.cv2.line

    def observed_line(image, a, b, color, *args, **kwargs):
        calls.append((a, b, color))
        return original_line(image, a, b, color, *args, **kwargs)

    monkeypatch.setattr(module.cv2, "line", observed_line)
    recorder._draw_hud(image, {"carla_frame_id": 91, "sim_time_s": 3.2},
        {"trajectory_visualization": {"enabled": True, "style": style,
            "drawn_candidates": 8, "candidate_pool_count": 40, "merged_candidates": 20,
            "hidden_candidates": 12}})
    colors = (module.CLASSIC_TRAJECTORY_HUD_COLORS if style == "classic"
              else module.TRAJECTORY_HUD_COLORS)
    for color in colors.values():
        assert np.any(np.all(image == color, axis=2))
    candidates = [call for call in calls if call[2] == colors["candidate"]]
    assert len(candidates) == 1 and candidates[0][1][0] - candidates[0][0][0] == 25
    if style == "classic":
        # Classic is the explicitly selected old native-world style.
        assert colors["execution"][1] > 2 * colors["execution"][0]
    else:
        assert colors["candidate"] == tuple(DISPLAY_CANDIDATE_COLORS[0][::-1])
        for kind in ("selected", "execution", "rejected"):
            assert colors[kind] == tuple(DISPLAY_COLORS[kind][::-1])


@pytest.mark.parametrize("style", ["paper", "classic"])
def test_video_hud_off_does_not_paint_a_trajectory_legend(tmp_path, monkeypatch, style):
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, hud=False)
    try:
        recorder._receive_frame(camera_frame(10, color=160))
        assert recorder.on_tick(frame_id=10, sim_time_s=1.,
            overlay={"trajectory_visualization": {"enabled": True, "style": style}})
        np.testing.assert_array_equal(writer.write.call_args.args[0],
                                      np.full((180, 320, 3), 160, dtype=np.uint8))
    finally:
        recorder.flush()


def test_delayed_projection_uses_each_images_capture_transform_and_retained_plan(tmp_path, monkeypatch):
    from carla_bridge.visualization import camera_overlay
    recorder, writer, world = start_mock(tmp_path, monkeypatch, flush_timeout_s=0)
    calls = []
    transforms = [SimpleNamespace(captured_pose=i) for i in (10, 11)]
    sensor = recorder._sensor
    sensor.get_transform.side_effect = AssertionError("live sensor pose is not capture pose")

    def paint(image, display, transform, fov_degrees):
        calls.append(("project", transform, display["paths"][0]["points_world"][0][0], fov_degrees))
        image[100, 100] = (10, 200, 250)
        return {"enabled": True, "drawn_segments": 3, "capture_pose": transform.captured_pose}

    monkeypatch.setattr(camera_overlay, "draw_trajectory_overlay", paint)
    monkeypatch.setattr(recorder, "_draw_hud", lambda image, row, overlay:
        calls.append(("hud", row["carla_frame_id"], tuple(image[100, 100]))))
    display = {"style": "paper", "paths": [{"points_world": [[2., 0., .156], [5., 0., .156]]}]}
    for frame_id, x in ((10, 2.), (11, 8.)):
        display["paths"][0]["points_world"][0][0] = x
        assert not recorder.on_tick(frame_id=frame_id, sim_time_s=frame_id / 10,
            overlay={"trajectory_display": display, "trajectory_visualization": {"enabled": True}})
    display["paths"][0]["points_world"][0][0] = 999.
    for frame_id, transform in ((11, transforms[1]), (10, transforms[0])):
        frame = camera_frame(frame_id)
        frame.transform = transform
        recorder._receive_frame(frame)
    recorder.flush()
    assert calls == [("project", transforms[0], 2., 90.), ("hud", 10, (10, 200, 250)),
                     ("project", transforms[1], 8., 90.), ("hud", 11, (10, 200, 250))]
    indexed = rows(recorder.index_path)
    assert [row["carla_frame_id"] for row in indexed] == [10, 11]
    assert [row["trajectory_overlay"]["capture_pose"] for row in indexed] == [10, 11]
    assert writer.write.call_count == 2
    assert recorder.summary()["encode_error_count"] == 0
    assert recorder._sensor is None
    sensor.get_transform.assert_not_called()
    world.tick.assert_not_called()
    world.wait_for_tick.assert_not_called()


def test_missing_capture_transform_skips_projection_with_audit_but_keeps_frame(tmp_path, monkeypatch):
    from carla_bridge.visualization import camera_overlay
    draw = Mock(side_effect=AssertionError("missing transform must never use live pose"))
    monkeypatch.setattr(camera_overlay, "draw_trajectory_overlay", draw)
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, hud=False)
    sensor = recorder._sensor
    sensor.get_transform.side_effect = AssertionError("never fall back to live sensor")
    try:
        recorder._receive_frame(camera_frame(10, color=160))
        assert recorder.on_tick(frame_id=10, sim_time_s=1., overlay={
            "trajectory_display": {"style": "paper", "paths": []}})
        assert rows(recorder.index_path)[0]["trajectory_overlay"]["skipped_reason"] == "missing_capture_transform"
        assert rows(recorder.events_path)[0]["reason"] == "trajectory_overlay_skipped"
        np.testing.assert_array_equal(writer.write.call_args.args[0], np.full((180, 320, 3), 160, dtype=np.uint8))
        draw.assert_not_called()
        sensor.get_transform.assert_not_called()
        assert recorder.summary()["missing_frame_count"] == 0
    finally:
        recorder.flush()


@pytest.mark.parametrize("failure", ["exception", "none", "nan", "object"])
def test_failed_optional_projection_preserves_base_pixels_and_encodes_frame(tmp_path, monkeypatch, failure):
    from carla_bridge.visualization import camera_overlay

    def broken(image, *args, **kwargs):
        image[:] = (0, 0, 0)  # A partially painted failed overlay must be discarded.
        if failure == "exception":
            raise RuntimeError("optional projection failed")
        return {"none": None, "nan": {"drawn_segments": float("nan")},
                "object": {"invalid": object()}}[failure]

    monkeypatch.setattr(camera_overlay, "draw_trajectory_overlay", broken)
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, hud=False)
    frame = camera_frame(10, color=160)
    frame.transform = SimpleNamespace(captured_pose=10)
    try:
        recorder._receive_frame(frame)
        assert recorder.on_tick(frame_id=10, sim_time_s=1.,
            overlay={"trajectory_display": {"style": "paper", "paths": []}})
        np.testing.assert_array_equal(writer.write.call_args.args[0], np.full((180, 320, 3), 160, dtype=np.uint8))
        indexed = rows(recorder.index_path)
        assert len(indexed) == 1 and indexed[0]["carla_frame_id"] == 10
        audit = indexed[0]["trajectory_overlay"]
        assert audit["skipped_reason"] == "overlay_error"
        assert audit["error"]["type"] in ("RuntimeError", "TypeError", "ValueError")
        assert rows(recorder.events_path)[0]["reason"] == "trajectory_overlay_error"
        assert recorder.summary()["encode_error_count"] == 0
        assert recorder.summary()["missing_frame_count"] == 0
    finally:
        recorder.flush()


def test_projection_event_failure_cannot_drop_an_otherwise_valid_frame(tmp_path, monkeypatch):
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, hud=False)
    monkeypatch.setattr(recorder, "_write_event", Mock(side_effect=OSError("optional event unavailable")))
    try:
        recorder._receive_frame(camera_frame(10))  # Missing capture transform produces a skip event.
        assert recorder.on_tick(frame_id=10, sim_time_s=1., overlay={"trajectory_display": {"style": "paper"}})
        assert writer.write.call_count == 1
        assert len(rows(recorder.index_path)) == 1
        assert recorder.summary()["encode_error_count"] == 0
    finally:
        recorder.flush()


@pytest.mark.parametrize("overlay", [
    {"trajectory_visualization": {"enabled": False}, "trajectory_display": {"style": "paper"}},
    {"trajectory_display": {"enabled": False, "style": "paper"}},
    {"trajectory_visualization": {"enabled": True}}])
def test_disabled_or_absent_projection_never_calls_optional_renderer(tmp_path, monkeypatch, overlay):
    from carla_bridge.visualization import camera_overlay
    draw = Mock(side_effect=AssertionError("projection is disabled"))
    monkeypatch.setattr(camera_overlay, "draw_trajectory_overlay", draw)
    recorder, writer, _ = start_mock(tmp_path, monkeypatch, hud=False)
    frame = camera_frame(10, color=160)
    frame.transform = SimpleNamespace(captured_pose=10)
    try:
        recorder._receive_frame(frame)
        assert recorder.on_tick(frame_id=10, sim_time_s=1., overlay=overlay)
        draw.assert_not_called()
        assert "trajectory_overlay" not in rows(recorder.index_path)[0]
        np.testing.assert_array_equal(writer.write.call_args.args[0], np.full((180, 320, 3), 160, dtype=np.uint8))
    finally:
        recorder.flush()


def test_real_renderer_loop_and_delayed_video_projection_share_world_geometry_and_frame_pose(tmp_path, monkeypatch):
    """Exercise every display layer; only CARLA RPCs and video encoding are fake.

    A second capture moves both ego and plan by 10m. Equal camera-relative
    pictures after delayed callbacks prove that neither the current ego box nor
    the current camera pose was substituted for the earlier capture's metadata.
    """
    from copy import deepcopy
    import carla
    from carla_bridge.tests.test_route_acceptance import fake_loop
    from carla_bridge.visualization.trajectory_renderer import CarlaTrajectoryRenderer, DISPLAY_COLORS

    recorder, writer, world = start_mock(tmp_path, monkeypatch, width=1280, height=720,
                                         hud=False, flush_timeout_s=0)
    sensor = recorder._sensor
    sensor.get_transform.side_effect = AssertionError("a live camera pose is never a capture pose")
    loop = fake_loop()
    loop.config.data["output_dir"] = tmp_path
    loop._control_audit_path = tmp_path / "control.jsonl"
    loop._world_clock_audit = Mock()
    loop.recorder = recorder
    # Legacy server YAML flags must not revive either marks or gaps in paper.
    renderer = CarlaTrajectoryRenderer(world, {"enabled": True, "endpoint_markers": True,
                                               "dash_length_m": .1, "dash_gap_m": 2.})
    loop.trajectory_renderer = renderer
    ego = loop.scenario.ego_vehicle
    ego.bounding_box = carla.BoundingBox(carla.Location(z=.75), carla.Vector3D(2.4, 1., .75))
    ego.get_control.return_value = carla.VehicleControl()
    captures = []
    for i, x in ((10, 0.), (11, 10.)):
        ego.get_transform.return_value = carla.Transform(carla.Location(x=x))
        loop._world_clock_audit.return_value = {"frame": i}
        loop._elapsed = i / 10
        points = [[x + 8., -2., 0.], [x + 28., -2., 0.]]
        snapshot = {"selected_index": 0, "candidate_pool_count": 3,
            "reference_surface_xyz": [[x, -2., 0.], [x + 30., -2., 0.]],
            "candidates": [{"id": "winner", "index": 0, "points_world": points},
                {"id": "invalid", "index": 1, "points_world": [[x, 0., 0.], [float("nan"), 0., 0.]]},
                {"id": "excessive", "index": 2, "points_world": np.column_stack(
                    (x + np.linspace(0., 10., 5001), np.zeros(5001), np.zeros(5001))).tolist()}]}
        loop.pdm_planner = SimpleNamespace(last_visualization_snapshot=snapshot)
        decision = {"trajectory": [{"x": 8., "y": 2.}, {"x": 28., "y": 2.}],
                    "planner_diagnostics": {"selected_id": "winner"}}
        before = deepcopy(decision)
        loop._audit_pdm_plan(decision, {"accepted": True}, "control_update")
        paths = renderer.last_display_snapshot["paths"]
        assert [path["kind"] for path in paths] == ["selected", "execution"]
        assert {path.get("id") for path in paths}.isdisjoint({"invalid", "excessive"})
        for path in paths:
            np.testing.assert_allclose(path["points_world"],
                np.asarray(points) + [0., 0., renderer.config["z_offset"]])
            assert not path["dashed"] and not path["marker"]
        assert decision == before
        loop._record_tick(ego)
        frame = camera_frame(i, color=160, width=1280, height=720)
        frame.transform = carla.Transform(carla.Location(x=x - 7., z=4.), carla.Rotation(pitch=-15.))
        captures.append(frame)
    # Any late telemetry access would be a bug. The exact-frame transform and
    # bounding box must already have been retained independently for each tick.
    ego.get_transform.side_effect = AssertionError("late ego pose is not capture-time ego pose")
    renderer.last_display_snapshot["paths"][0]["points_world"][0][0] = 999.
    for frame in captures[::-1]:
        recorder._receive_frame(frame)
    recorder.flush()
    indexed = rows(recorder.index_path)
    assert [row["carla_frame_id"] for row in indexed] == [10, 11]
    assert all(row["trajectory_overlay"]["drawn_paths"] == 2 for row in indexed)
    assert all(row["trajectory_overlay"]["ego_box_masked"] for row in indexed)
    images = [call.args[0] for call in writer.write.call_args_list]
    assert len(images) == 2
    for pixels in images:
        for kind in ("selected", "execution"):
            assert np.any(np.all(pixels == DISPLAY_COLORS[kind][::-1], axis=-1))
    assert np.mean(np.abs(images[0].astype(float) - images[1])) < .1
    # No emissive native primitive may enter the raw camera image. The only
    # trajectory pixels are the muted, continuous post-capture annotation.
    assert world.debug.method_calls == []
    assert [row["trajectory_visualization"]["invalid_candidate_ids"]
            for row in rows(tmp_path / "planner.jsonl")] == [["invalid"], ["invalid"]]
    sensor.get_transform.assert_not_called()
    ego.apply_control.assert_not_called()
    loop.carla_client.tick.assert_not_called()
    world.tick.assert_not_called()


def test_real_paper_video_retains_continuous_plan_between_replans_without_lifetime_expiry(tmp_path, monkeypatch):
    """The legacy world-line lifetime cannot blink a valid published video plan."""
    import carla
    from carla_bridge.tests.test_route_acceptance import fake_loop
    from carla_bridge.visualization.trajectory_renderer import CarlaTrajectoryRenderer, DISPLAY_COLORS

    recorder, writer, world = start_mock(tmp_path, monkeypatch, width=1280, height=720,
                                         hud=False, flush_timeout_s=0)
    loop = fake_loop()
    loop.config.data["output_dir"] = tmp_path
    loop._control_audit_path = tmp_path / "control.jsonl"
    loop._world_clock_audit = Mock()
    loop.recorder = recorder
    renderer = CarlaTrajectoryRenderer(world, {"enabled": True, "lifetime_s": .001,
                                               "endpoint_markers": True})
    loop.trajectory_renderer = renderer
    ego = loop.scenario.ego_vehicle
    ego.bounding_box = carla.BoundingBox(carla.Location(z=.75), carla.Vector3D(2.4, 1., .75))
    ego.get_control.return_value = carla.VehicleControl()
    loop.pdm_planner = SimpleNamespace(last_visualization_snapshot={"selected_index": 0,
        "candidate_pool_count": 1, "reference_surface_xyz": [[0., -2., 0.], [30., -2., 0.]],
        "candidates": [{"id": "winner", "index": 0, "points_world": [[8., -2., 0.], [28., -2., 0.]]}]})
    loop._audit_pdm_plan({"trajectory": [{"x": 8., "y": 2.}, {"x": 28., "y": 2.}],
                         "planner_diagnostics": {"selected_id": "winner"}}, {"accepted": True}, "control_update")
    published = renderer.last_display_snapshot
    for i in range(10, 20):
        loop._elapsed = (i - 9) / 10
        loop._world_clock_audit.return_value = {"frame": i}
        loop._record_tick(ego)
        frame = camera_frame(i, color=160, width=1280, height=720)
        frame.transform = carla.Transform(carla.Location(x=-7., z=4.), carla.Rotation(pitch=-15.))
        recorder._receive_frame(frame)
    recorder.flush()
    assert renderer.last_display_snapshot is published
    assert len(rows(tmp_path / "planner.jsonl")) == 1  # No redraw/replan was needed.
    indexed = rows(recorder.index_path)
    assert len(indexed) == 10
    assert all(row["trajectory_overlay"]["drawn_paths"] == 2 for row in indexed)
    images = [call.args[0] for call in writer.write.call_args_list]
    for pixels in images:
        assert np.any(np.all(pixels == DISPLAY_COLORS["execution"][::-1], axis=-1))
        np.testing.assert_array_equal(pixels, images[0])
    assert world.debug.method_calls == []
    ego.apply_control.assert_not_called()
    loop.carla_client.tick.assert_not_called()
    world.tick.assert_not_called()
