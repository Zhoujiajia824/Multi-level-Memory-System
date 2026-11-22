"""Agent-owned artifacts; official scoring remains exclusively upstream-owned."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import subprocess
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
        self.writer = None
        self.video_frames = 0
        self.first_frame = None
        self.last_frame = None
        self.files = {}
        self.cv2 = None
        self.raw_video_path = self.planning_directory / "video.mpeg4.mp4"
        self.final_video_path = self.planning_directory / "video.mp4"
        self.video_size = None
        if video:
            import cv2
            self.cv2 = cv2

    def append(self, name, value):
        if name not in self.files:
            self.files[name] = (self.directory / (name+".jsonl")).open("a", encoding="utf-8")
        self.files[name].write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        self.files[name].flush()

    def json(self, name, value):
        (self.directory/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def capture_images(self, input_data, frame, timestamp, camera_ids, save=True):
        images = []
        arrays = []
        for camera_id in camera_ids:
            if camera_id not in input_data:
                raise ValueError("missing registered RGB camera: "+camera_id)
            image_frame, data = input_data[camera_id]
            if int(image_frame) != frame:
                raise ValueError("RGB/state source frame mismatch: %s=%s, world=%s" % (camera_id, image_frame, frame))
            if len(data.shape) != 3 or data.shape[2] < 3 or str(data.dtype) != "uint8":
                raise ValueError("invalid RGB camera array: "+camera_id)
            arrays.append(data[:, :, :3])  # official callback emits BGRA; VideoWriter consumes BGR.
            if save:
                folder = self.planning_directory / "camera_frames" / ("%010d" % frame)
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder / (camera_id+".png")
                _png(destination, data[:, :, 2::-1])
                images.append({"id": camera_id, "frame": frame, "timestamp": timestamp,
                               "path": str(destination.resolve()), "width": data.shape[1], "height": data.shape[0]})
        if self.video_requested and arrays:
            self._video(arrays, frame)
        return images

    def _video(self, arrays, frame):
        import numpy as np
        if self.last_frame is not None and frame != self.last_frame+1:
            raise ValueError("video requires consecutive official physical frames")
        width, height = 320, 180
        tiles = [self.cv2.resize(data, (width, height)) for data in arrays]
        while len(tiles)%3:
            tiles.append(np.zeros((height,width,3), dtype=np.uint8))
        mosaic = np.vstack([np.hstack(tiles[index:index+3]) for index in range(0,len(tiles),3)])
        if self.writer is None:
            path = str(self.raw_video_path)
            self.writer = self.cv2.VideoWriter(path, self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                              (mosaic.shape[1], mosaic.shape[0]))
            self.video_size = [mosaic.shape[1], mosaic.shape[0]]
            if not self.writer.isOpened():
                raise RuntimeError("failed to open video encoder: "+path)
            self.first_frame = frame
        self.writer.write(mosaic)
        self.video_frames += 1
        self.last_frame = frame

    def close(self):
        if self.writer is not None:
            self.writer.release()
        for stream in self.files.values():
            stream.close()
        status = "not_requested" if not self.video_requested else "no_frames"
        error = None
        decoded_frames = 0
        if self.writer is not None:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                status, error = "transcode_unavailable", "ffmpeg not found; original MPEG4 capture preserved"
            else:
                try:
                    result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                        "-i", str(self.raw_video_path), "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.final_video_path)],
                        capture_output=True, text=True, timeout=120)
                    if result.returncode:
                        raise RuntimeError(result.stderr.strip() or "ffmpeg H264 encoder failed")
                    decoder = self.cv2.VideoCapture(str(self.final_video_path))
                    try:
                        while decoder.read()[0]:
                            decoded_frames += 1
                    finally:
                        decoder.release()
                    if decoded_frames != self.video_frames:
                        raise RuntimeError("H264 decoded frame count differs from captured source frames")
                    status = "complete"
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    status, error = "transcode_failed", str(exc)
        self.json("planning/video_recording.json", {"requested": self.video_requested, "frames": self.video_frames,
                  "fps": self.fps, "first_source_frame": self.first_frame, "last_source_frame": self.last_frame,
                  "source": "official_CARLA_rendered_camera", "interpolated_frames": 0,
                  "encoded_duration_s": self.video_frames/self.fps,
                  "status": status, "error": error, "decoded_frames": decoded_frames,
                  "playback_verified": status == "complete",
                  "path": self.final_video_path.name if status == "complete" else self.raw_video_path.name,
                  "size": self.video_size, "codec": "h264" if status == "complete" else "mp4v",
                  "verification": "h264_decode_verified" if status == "complete" else None,
                  "renderer_pose_reference": "Fail2Drive official sensor wrapper; six model camera mosaic",
                  "encoder": "project_archive_ffmpeg_libx264",
                  "raw_capture_preserved": True,
                  "note": "Agent-owned camera mosaic; official evaluator result is authoritative."})
