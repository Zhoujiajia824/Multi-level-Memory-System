"""Bounded, frame-indexed CARLA driving video with optional model-call HUD.

Tick registration is nonblocking by default; delayed GPU camera callbacks are
matched to bounded pending metadata and written in simulation-frame order.
The final drain never advances the world. A JSONL index records encoded frames,
so a missing image cannot silently shift simulation time against playback.
"""
from __future__ import annotations

import copy
import json
import math
import queue
import time
from pathlib import Path
from typing import Optional

import carla
import cv2
import numpy as np

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("carla_video")


class VideoRecorder:
    """Record an exact camera frame for each requested CARLA tick.

    ``sim_time_s`` is elapsed experiment time; the sensor's absolute timestamp is
    separate. ``video_time_s`` is actual playback time, including any frame gaps.
    """

    def __init__(self, world, ego, output_path: str, fps: float = 20,
                 width: int = 1280, height: int = 720, view: str = "chase",
                 frame_timeout_s: float = 0.0, hud: bool = True,
                 max_pending_frames: int = 32, flush_timeout_s: float = .25):
        self.world, self.ego = world, ego
        self.output_path = str(output_path)
        self.fps, self.frame_timeout_s = float(fps), float(frame_timeout_s)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video fps must be finite and positive")
        if not math.isfinite(self.frame_timeout_s) or self.frame_timeout_s < 0:
            raise ValueError("video frame timeout must be finite and non-negative")
        self.max_pending_frames = int(max_pending_frames)
        self.flush_timeout_s = float(flush_timeout_s)
        if self.max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be positive")
        if not math.isfinite(self.flush_timeout_s) or self.flush_timeout_s < 0:
            raise ValueError("flush_timeout_s must be finite and non-negative")
        self.width, self.height = int(width), int(height)
        if min(self.width, self.height) <= 0:
            raise ValueError("video dimensions must be positive")
        self.view, self.hud = view, bool(hud)
        path = Path(self.output_path)
        self.index_path = path.with_suffix(".frames.jsonl")
        self.events_path = path.with_suffix(".events.jsonl")
        self.summary_path = path.with_suffix(".summary.json")
        self._sensor = None
        self._queue: "queue.Queue" = queue.Queue(maxsize=self.max_pending_frames)
        self._available_frames = {}
        self._pending = {}
        self._last_requested_frame = None
        self._last_retired_frame = None
        self._expired_count = 0
        self._unfulfilled_count = 0
        self._writer = None
        self._index_file = None
        self._events_file = None
        self._frame_count = 0
        self._requested_count = 0
        self._missing_count = 0
        self._stale_count = 0
        self._overflow_count = 0
        self._error_count = 0
        self._first_sim_time = None
        self._last_sim_time = None
        self._last_frame_id = None
        self._started = False
        self._flushed = False
        self._closing = False

    def start(self) -> None:
        """Install the camera; clean up partial initialization on failure."""
        if self._started or self._flushed:
            raise RuntimeError("VideoRecorder instances cannot be restarted")
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._index_file = self.index_path.open("w", encoding="utf-8", newline="\n")
            self._events_file = self.events_path.open("w", encoding="utf-8", newline="\n")
            bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(self.width))
            bp.set_attribute("image_size_y", str(self.height))
            bp.set_attribute("fov", "90")
            bp.set_attribute("sensor_tick", "0.0")
            self._sensor = self.world.spawn_actor(bp, self._view_transform(), attach_to=self.ego)
            self._sensor.listen(self._receive_frame)
            self._writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                           self.fps, (self.width, self.height))
            if not self._writer.isOpened():
                raise RuntimeError(f"无法打开视频输出: {self.output_path}")
            self._started = True
        except Exception:
            self._error_count += 1
            self.flush()
            raise
        logger.info("视频录制开始: %s (%dx%d @ %.3gfps, view=%s)",
                    self.output_path, self.width, self.height, self.fps, self.view)

    def _receive_frame(self, frame) -> None:
        """Sensor thread callback: bounded storage also during failed runs."""
        if self._flushed:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._overflow_count += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                self._overflow_count += 1

    def _view_transform(self) -> "carla.Transform":
        if self.view == "front":
            return carla.Transform(carla.Location(x=2.0, y=0.0, z=1.4),
                                   carla.Rotation(pitch=0, yaw=0))
        return carla.Transform(carla.Location(x=-7.0, y=0.0, z=4.0),
                               carla.Rotation(pitch=-15, yaw=0))

    @staticmethod
    def _frame_id(frame) -> Optional[int]:
        value = getattr(frame, "frame", None)
        return int(value) if isinstance(value, (int, np.integer)) else None

    @staticmethod
    def _finite_number(value) -> Optional[float]:
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def _write_event(self, reason, requested_frame, sim_time_s, **extra) -> None:
        row = {"reason": reason, "requested_frame": requested_frame,
               "sim_time_s": self._finite_number(sim_time_s), **extra}
        if self._events_file is not None:
            self._events_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self._events_file.flush()

    def _ingest_frame(self, frame) -> None:
        actual = self._frame_id(frame)
        if actual is None or (self._last_retired_frame is not None and actual <= self._last_retired_frame):
            self._stale_count += 1
            self._write_event("stale_camera_frame", None, None, carla_frame_id=actual)
            return
        if actual in self._available_frames:
            self._stale_count += 1
            self._write_event("duplicate_camera_frame", None, None, carla_frame_id=actual)
            return
        self._available_frames[actual] = frame
        # Includes images delivered before their on_tick registration. Prefer
        # keeping registered images if unregistered future images fill storage.
        while len(self._available_frames) > self.max_pending_frames:
            unregistered = [key for key in self._available_frames if key not in self._pending]
            victim = max(unregistered) if unregistered else min(self._available_frames)
            self._available_frames.pop(victim)
            self._overflow_count += 1
            self._write_event("camera_buffer_overflow", None, None, carla_frame_id=victim)

    def _drain_available(self) -> None:
        # Fixed cap keeps a busy callback thread from prolonging on_tick forever.
        for _ in range(self.max_pending_frames):
            try:
                self._ingest_frame(self._queue.get_nowait())
            except queue.Empty:
                break

    def _encode_ready(self) -> None:
        while self._pending:
            requested = next(iter(self._pending))
            frame = self._available_frames.pop(requested, None)
            if frame is None:
                break
            metadata = self._pending.pop(requested)
            self._encode_frame(frame, requested, metadata["sim_time_s"], metadata["overlay"])
            self._last_retired_frame = requested
        if self._last_retired_frame is not None:
            for key in list(self._available_frames):
                if key <= self._last_retired_frame:
                    self._available_frames.pop(key)
                    self._stale_count += 1
                    self._write_event("stale_camera_frame", None, None, carla_frame_id=key)

    def _service_pending(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        self._drain_available()
        self._encode_ready()
        while self._pending and timeout_s > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._ingest_frame(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
            self._drain_available()
            self._encode_ready()

    def _expire_first_pending(self, reason: str) -> None:
        requested = next(iter(self._pending))
        metadata = self._pending.pop(requested)
        self._missing_count += 1
        self._last_retired_frame = requested
        self._write_event(reason, requested, metadata["sim_time_s"])

    def on_tick(self, *, frame_id=None, sim_time_s=None, overlay=None) -> bool:
        """Register exact tick metadata, then encode ready frames in time order.

        Default operation is nonblocking. A GPU image arriving two ticks later
        still receives the original frame's time and HUD, rather than the most
        recent tick's metadata. False means the target is not yet encoded, not
        that it was dropped. Expiration and final unfulfilled requests have
        explicit events. Neither registration nor close advances the world.
        Optional HUD keys: ``model_call_count``, ``latest_call_id``,
        ``latest_trigger``, ``selected_id``, ``status``.
        """
        if self._writer is None or self._flushed or self._closing:
            return False
        requested = int(frame_id) if frame_id is not None else None
        self._requested_count += 1
        if requested is None:
            # Backward compatibility for old no-argument users: encode latest
            # callback only, while still indexing its real ID and timestamp.
            self._drain_available()
            if not self._available_frames:
                self._missing_count += 1
                self._write_event("no_camera_frame", None, sim_time_s)
                return False
            latest = max(self._available_frames)
            frame = self._available_frames.pop(latest)
            self._stale_count += len(self._available_frames)
            self._available_frames.clear()
            result = self._encode_frame(frame, None, sim_time_s, overlay or {})
            self._last_retired_frame = latest
            return result
        if self._last_requested_frame is not None and requested <= self._last_requested_frame:
            self._missing_count += 1
            self._write_event("duplicate_or_out_of_order_request", requested, sim_time_s)
            return False
        self._last_requested_frame = requested
        self._pending[requested] = {"sim_time_s": sim_time_s, "overlay": copy.deepcopy(overlay or {})}
        self._service_pending(self.frame_timeout_s)
        while len(self._pending) > self.max_pending_frames:
            self._expired_count += 1
            self._expire_first_pending("pending_window_expired")
            self._encode_ready()
        return self._last_frame_id == requested

    def _encode_frame(self, frame, requested, sim_time_s, overlay) -> bool:
        actual = self._frame_id(frame)
        sensor_time = self._finite_number(getattr(frame, "timestamp", None))
        elapsed = self._finite_number(sim_time_s) if sim_time_s is not None else sensor_time
        row = {"video_frame_index": self._frame_count, "carla_frame_id": actual,
               "requested_frame": requested, "sim_time_s": elapsed,
               "sensor_timestamp_s": sensor_time,
               "video_time_s": self._frame_count / self.fps, "fps": self.fps}
        try:
            bgr = np.frombuffer(frame.raw_data, dtype=np.uint8).reshape(
                frame.height, frame.width, 4)[:, :, :3].copy()
            if (frame.width, frame.height) != (self.width, self.height):
                raise ValueError("camera frame dimensions do not match video writer")
            if self.hud:
                self._draw_hud(bgr, row, overlay or {})
            # cv2 normally returns None. Explicit False from a backend is failure.
            if self._writer.write(bgr) is False:
                raise RuntimeError("video writer rejected frame")
            if self._index_file is not None:
                self._index_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                self._index_file.flush()
        except Exception as exc:
            self._error_count += 1
            self._missing_count += 1
            self._write_event("encode_error", requested, sim_time_s,
                              carla_frame_id=actual, error=f"{type(exc).__name__}: {exc}")
            logger.warning("视频帧写入失败 (frame=%s): %s", actual, exc)
            return False
        self._frame_count += 1
        self._last_frame_id = actual
        if self._first_sim_time is None:
            self._first_sim_time = elapsed
        self._last_sim_time = elapsed
        return True

    def _draw_hud(self, image, row: dict, overlay: dict) -> None:
        def compact(value):
            # cv2 fonts are ASCII; detailed UTF-8 remains in audit artifacts.
            return str(value).encode("ascii", "replace").decode("ascii")[:150]

        elapsed = row["sim_time_s"]
        lines = [f"CARLA frame {row['carla_frame_id']} | sim {elapsed:.2f}s" if elapsed is not None
                 else f"CARLA frame {row['carla_frame_id']}"]
        if overlay:
            lines += [f"Model calls: {overlay.get('model_call_count', overlay.get('call_count', 0))}"
                      f" | latest: {compact(overlay.get('latest_call_id', '-'))}",
                      f"Trigger: {compact(overlay.get('latest_trigger', overlay.get('trigger', '-')))}",
                      f"Selected: {compact(overlay.get('selected_id', '-'))}"
                      f" | {compact(overlay.get('status', ''))}"]
        panel_height = min(self.height, 10 + 22 * len(lines))
        image[:panel_height] = (image[:panel_height].astype(float) * .35).astype(np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(image, line, (8, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (255, 255, 255), 1, cv2.LINE_AA)

    def summary(self) -> dict:
        """Small replay manifest; full frame correspondence remains on disk."""
        return {"schema_version": 1, "video_path": self.output_path,
                "frames_path": str(self.index_path), "events_path": str(self.events_path),
                "summary_path": str(self.summary_path), "fps": self.fps,
                "width": self.width, "height": self.height, "view": self.view,
                "hud": self.hud, "requested_frame_count": self._requested_count,
                "encoded_frame_count": self._frame_count,
                "pending_frame_count": len(self._pending),
                "max_pending_frames": self.max_pending_frames,
                "frame_timeout_s": self.frame_timeout_s,
                "flush_timeout_s": self.flush_timeout_s,
                "expired_frame_count": self._expired_count,
                "unfulfilled_frame_count": self._unfulfilled_count,
                "missing_frame_count": self._missing_count,
                "stale_frame_count": self._stale_count,
                "queue_overflow_count": self._overflow_count,
                "encode_error_count": self._error_count,
                "video_duration_s": self._frame_count / self.fps,
                "first_sim_time_s": self._first_sim_time, "last_sim_time_s": self._last_sim_time,
                "closed": self._flushed,
                "timeline_note": "Use frames_path for exact simulation-to-video correspondence; missing frames are not duplicated."}

    def manifest(self) -> dict:
        return self.summary()

    def flush(self) -> None:
        """Persist indexes on success, interruption or partial setup; idempotent."""
        if self._flushed or self._closing:
            return
        self._closing = True
        if self._writer is not None:
            # Leave the callback active for a small, bounded final drain. A
            # missing GPU frame cannot make close tick or wait indefinitely.
            self._service_pending(self.flush_timeout_s)
            while self._pending:
                self._encode_ready()
                if self._pending:
                    self._unfulfilled_count += 1
                    self._expire_first_pending("unfulfilled_on_close")
            self._encode_ready()
        self._flushed = True
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception:
                pass
            try:
                self._sensor.destroy()
            except Exception:
                pass
            self._sensor = None
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as exc:
                logger.warning("视频关闭失败: %s", exc)
            self._writer = None
        for name in ("_index_file", "_events_file"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)
        self._available_frames.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self.summary_path.parent.exists():
            with self.summary_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(self.summary(), handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
        logger.info("视频录制结束: %s (%d 帧, 播放 %.1fs, 缺帧 %d；时间以帧索引为准)",
                    self.output_path, self._frame_count, self._frame_count / self.fps,
                    self._missing_count)
