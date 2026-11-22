"""Optional browser video copies, stored only in the viewer's own cache."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading


class MediaCache:
    def __init__(self, directory=None, ffmpeg=None):
        self.directory = Path(directory) if directory else Path(__file__).resolve().parents[1] / ".cache" / "video"
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg")
        self.jobs = {}
        self.lock = threading.RLock()

    @staticmethod
    def signature(source):
        source = Path(source)
        info = source.stat()
        return str(source.resolve()), info.st_size, info.st_mtime_ns

    @classmethod
    def key(cls, source):
        return hashlib.sha256(repr(cls.signature(source)).encode("utf-8")).hexdigest()

    def status(self, source, start=False):
        source = Path(source)
        key = self.key(source)
        target = self.directory / (key + ".mp4")
        with self.lock:
            if target.is_file() and target.stat().st_size > 0:
                return dict(status="ready", url="/api/video-cache/" + key + ".mp4",
                            codec="H.264", source_unchanged=True)
            if key in self.jobs and self.jobs[key]["status"] == "working":
                return dict(self.jobs[key])
            if not self.ffmpeg:
                return dict(status="unavailable", error="未找到FFmpeg；安装后加入PATH并重启展示程序，或通过--ffmpeg指定。原录像未修改。")
            if not start:
                return dict(self.jobs.get(key, {"status": "idle"}))
            self.directory.mkdir(parents=True, exist_ok=True)
            self.jobs[key] = dict(status="working", note="正在展示程序缓存中生成H.264副本，原录像不变")
            threading.Thread(target=self._convert, args=(source, target, key), daemon=True,
                             name="epviz-video-" + key[:8]).start()
            return dict(self.jobs[key])

    def _convert(self, source, target, key):
        # The command is an argument array, not a shell string. Source files are
        # validated episode assets; output is always under this private cache.
        partial = target.with_suffix(".partial.mp4")
        before = self.signature(source)
        command = [str(self.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-i", str(source), "-map", "0:v:0", "-an",
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(partial)]
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                    timeout=1800, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            if result.returncode:
                raise ValueError(result.stderr.decode("utf-8", errors="replace")[-1600:] or "FFmpeg转换失败")
            if before != self.signature(source):
                raise ValueError("源录像在转换中发生变化；请等待录制结束后重试")
            if not partial.is_file() or partial.stat().st_size == 0:
                raise ValueError("转换未产生有效视频")
            partial.replace(target)
            with self.lock:
                self.jobs[key] = dict(status="ready", url="/api/video-cache/" + key + ".mp4",
                                      codec="H.264", source_unchanged=True)
        except Exception as exc:
            with self.lock:
                self.jobs[key] = dict(status="failed", error=str(exc))
        finally:
            if partial.is_file():
                partial.unlink()

    def resolve(self, key):
        if not re.fullmatch(r"[a-f0-9]{64}\.mp4", key):
            raise ValueError("无效的视频缓存标识")
        target = self.directory / key
        if not target.is_file():
            raise FileNotFoundError("兼容视频尚未生成")
        return target
