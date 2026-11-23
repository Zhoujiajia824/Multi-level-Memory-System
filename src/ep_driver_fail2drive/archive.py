"""Agent-owned artifacts; official scoring remains exclusively upstream-owned."""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import threading
import zlib


def _png(path, rgb):
    """Write uint8 RGB with standard library; CARLA-side needs no torch/Pillow."""
    height, width = rgb.shape[:2]
    raw = b"".join(b"\0" + rgb[row].tobytes() for row in range(height))
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind+payload)&0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


class RouteArchive:
    def __init__(self, directory, video=False, fps=20.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.planning_directory = self.directory / "planning"
        self.planning_directory.mkdir()
        self.video_requested = bool(video)
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video fps must be finite and positive")
        self.encoder = None
        self.encoder_thread = None
        self.encoder_queue = queue.Queue(maxsize=1)
        self.encoder_log = None
        self.encoder_error = None
        self.encoder_timeout_s = 30.0
        self.decoder_timeout_s = 120.0
        self.ffmpeg_path = None
        self.closed = False
        self.video_frames = 0
        self.first_frame = None
        self.last_frame = None
        self.first_time = None
        self.last_time = None
        self.files = {}
        self.final_video_path = self.planning_directory / "video.mp4"
        self.video_size = None

    def append(self, name, value):
        if name not in self.files:
            self.files[name] = (self.directory / (name+".jsonl")).open("a", encoding="utf-8")
        self.files[name].write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        self.files[name].flush()

    def json(self, name, value):
        (self.directory/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def capture_images(self, input_data, frame, timestamp, camera_ids, save=True):
        images = []
        for camera_id in camera_ids:
            if camera_id not in input_data:
                raise ValueError("missing registered RGB camera: "+camera_id)
            image_frame, data = input_data[camera_id]
            if int(image_frame) != frame:
                raise ValueError("RGB/state source frame mismatch: %s=%s, world=%s" % (camera_id, image_frame, frame))
            if len(data.shape) != 3 or data.shape[2] < 3 or str(data.dtype) != "uint8":
                raise ValueError("invalid RGB camera array: "+camera_id)
            if save:
                folder = self.planning_directory / "camera_frames" / ("%010d" % frame)
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder / (camera_id+".png")
                _png(destination, data[:, :, 2::-1])
                images.append({"id": camera_id, "frame": frame, "timestamp": timestamp,
                               "path": str(destination.resolve()), "width": data.shape[1], "height": data.shape[0]})
        return images

    def _encode_loop(self):
        """单个后台写入者；调用方等待本帧确认，不丢帧也不积压整段视频。"""
        try:
            while True:
                item = self.encoder_queue.get()
                if item is None:
                    break
                payload, completed, outcome = item
                try:
                    remaining = memoryview(payload)
                    while remaining:
                        written = self.encoder.stdin.write(remaining)
                        if not written:
                            raise BrokenPipeError("encoder accepted no bytes")
                        remaining = remaining[written:]
                    self.encoder.stdin.flush()
                except (OSError, ValueError) as exc:
                    outcome.append(type(exc).__name__ + ": " + str(exc))
                finally:
                    completed.set()
                if outcome:
                    break
        finally:
            try:
                self.encoder.stdin.close()
            except (OSError, ValueError):
                pass

    def _abort_encoder(self):
        if self.encoder is not None and self.encoder.poll() is None:
            self.encoder.terminate()
            try:
                self.encoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.encoder.kill()
                self.encoder.wait(timeout=5)
        if self.encoder_thread is not None and self.encoder_thread.is_alive():
            try:
                self.encoder_queue.put_nowait(None)
            except queue.Full:
                pass

    def _start_encoder(self, width, height):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg unavailable; cannot record the original-resolution official camera")
        self.ffmpeg_path = ffmpeg
        self.encoder_log = (self.planning_directory / "video_encoder.log").open("wb")
        try:
            self.encoder = subprocess.Popen([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "%dx%d" % (width, height),
                "-framerate", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self.final_video_path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self.encoder_log, bufsize=0)
        except OSError:
            self.encoder_log.close()
            self.encoder_log = None
            raise
        self.video_size = [width, height]
        self.encoder_thread = threading.Thread(target=self._encode_loop, name="f2d-h264-writer", daemon=True)
        self.encoder_thread.start()

    def capture_display(self, image, frame, timestamp):
        """每个物理帧录制官方 rgb 相机原尺寸；与1Hz模型PNG归档相互独立。

        image 是官方传感器包装器的 (source_frame, BGRA/BGR ndarray)，不是2Hz JPEG。
        直接 rawvideo→H264 单次编码，不缩小、不拼接、不插帧。
        """
        if not self.video_requested:
            return
        if self.closed or self.encoder_error is not None:
            raise RuntimeError("video recording is closed or failed: " + str(self.encoder_error))
        import numpy as np
        try:
            image_frame, data = image
            if (type(frame) is not int or frame < 0 or int(image_frame) != frame
                    or type(timestamp) not in (int, float) or not math.isfinite(timestamp)):
                raise ValueError("official display image/state source frame or time is invalid")
            if (len(data.shape) != 3 or data.shape[2] not in (3, 4) or str(data.dtype) != "uint8"
                    or data.shape[0] < 2 or data.shape[1] < 2 or data.shape[0] % 2 or data.shape[1] % 2):
                raise ValueError("official display image must be even-sized uint8 BGR/BGRA")
            height, width = data.shape[:2]
            if self.last_frame is not None:
                if frame != self.last_frame + 1 or abs(timestamp - self.last_time - 1.0/self.fps) > 1e-5:
                    raise ValueError("official display video requires consecutive physical frames and timestamps")
                if [width, height] != self.video_size:
                    raise ValueError("official display image size changed during the route")
            if self.encoder is None:
                self._start_encoder(width, height)
            if self.encoder.poll() is not None:
                raise RuntimeError("H264 encoder exited before receiving the next physical frame")
            completed, outcome = threading.Event(), []
            payload = np.ascontiguousarray(data[:, :, :3]).tobytes()
            self.encoder_queue.put((payload, completed, outcome), timeout=self.encoder_timeout_s)
            if not completed.wait(self.encoder_timeout_s):
                raise TimeoutError("H264 encoder stdin write exceeded bounded wait")
            if outcome:
                raise RuntimeError("H264 encoder stdin write failed: " + outcome[0])
            if self.first_frame is None:
                self.first_frame, self.first_time = frame, float(timestamp)
            self.last_frame, self.last_time = frame, float(timestamp)
            self.video_frames += 1
        except Exception as exc:
            self.encoder_error = type(exc).__name__ + ": " + str(exc)
            self._abort_encoder()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for stream in self.files.values():
            stream.close()
        status = "not_requested" if not self.video_requested else "no_frames"
        error = self.encoder_error
        decoded_frames = 0
        try:
            if self.encoder is not None:
                if error is not None:
                    raise RuntimeError(error)
                self.encoder_queue.put(None, timeout=5)
                self.encoder_thread.join(timeout=10)
                if self.encoder_thread.is_alive():
                    raise TimeoutError("H264 encoder writer did not close")
                if self.encoder.wait(timeout=120) != 0:
                    raise RuntimeError("H264 encoder failed; see video_encoder.log")
                # 实际逐帧解码在有界子进程内完成；坏文件或解码器卡住不会无限占用收尾。
                # 不使用容器声明帧数，也不执行重采样、补帧或stream copy。
                decoded = subprocess.run([self.ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-xerror", "-err_detect", "explode", "-progress", "pipe:1", "-nostats",
                    "-i", str(self.final_video_path), "-map", "0:v:0", "-an", "-sn", "-dn",
                    "-vsync", "0", "-f", "null", "-"], capture_output=True, text=True,
                    timeout=self.decoder_timeout_s)
                if decoded.returncode != 0:
                    raise RuntimeError("H264 actual decode failed: " + decoded.stderr.strip()[:500])
                counts = []
                ended = False
                for line in decoded.stdout.splitlines():
                    key, _, value = line.partition("=")
                    if key == "frame" and value.strip().isdigit():
                        counts.append(int(value))
                    if key == "progress" and value.strip() == "end":
                        ended = True
                if not ended or not counts or any(b < a for a, b in zip(counts, counts[1:])):
                    raise RuntimeError("H264 decoder did not finish with a valid actual-frame progress count")
                decoded_frames = counts[-1]
                if decoded_frames != self.video_frames:
                    raise RuntimeError("H264 decoded frame count differs from captured physical frames")
                status = "complete"
            elif error is not None:
                status = "encoding_failed"
        except (OSError, RuntimeError, TimeoutError, queue.Full, subprocess.TimeoutExpired) as exc:
            status, error = "encoding_failed", type(exc).__name__ + ": " + str(exc)
            self._abort_encoder()
        finally:
            if self.encoder_thread is not None and self.encoder_thread.is_alive():
                self.encoder_thread.join(timeout=2)
            if self.encoder_log is not None:
                self.encoder_log.close()
        self.json("planning/video_recording.json", {"requested": self.video_requested, "frames": self.video_frames,
                  "fps": self.fps, "first_source_frame": self.first_frame, "last_source_frame": self.last_frame,
                  "first_source_time": self.first_time, "last_source_time": self.last_time,
                  "source": "official_CARLA_rendered_camera", "interpolated_frames": 0,
                  "encoded_duration_s": self.video_frames/self.fps,
                  "status": status, "error": error, "decoded_frames": decoded_frames,
                  "playback_verified": status == "complete",
                  "path": self.final_video_path.name,
                  "size": self.video_size, "codec": "h264",
                  "verification": "h264_decode_verified" if status == "complete" else None,
                  "decoder": "ffmpeg_actual_decode_progress", "decoder_timeout_s": self.decoder_timeout_s,
                  "video_view": "official_display", "camera_id": "rgb",
                  "renderer_pose_reference": "Fail2Drive official team_code/viz_path_agent.py camera pose",
                  "encoder": "ffmpeg_rawvideo_libx264_single_pass_crf18",
                  "source_resolution_preserved": True, "raw_capture_preserved": False,
                  "note": "Original-resolution official-example camera at every physical tick; no mosaic or interpolated JPEGs. Project viewer; official evaluator result is authoritative."})
