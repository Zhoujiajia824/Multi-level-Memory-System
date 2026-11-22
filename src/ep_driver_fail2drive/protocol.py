"""Bounded JSONL transport between the official Python agent and numeric worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
import time

SCHEMA = "ep-driver-fail2drive-packet-v1"


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    """One synchronous request at a time; diagnostics never share protocol stdout."""

    def __init__(self, python, config_path, stderr_path, timeout_s=240.0):
        if not 0 < float(timeout_s) < 300:
            raise ValueError("worker_timeout_s must be positive and below official 300s watchdog")
        self.timeout_s = float(timeout_s)
        self.sequence = 0
        self.messages = Queue()
        self.log = Path(stderr_path).open("a", encoding="utf-8")
        source_root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        # VLM inference runs on the separate server; keep this numeric/DINO worker on CPU.
        env["CUDA_VISIBLE_DEVICES"] = ""
        self.process = subprocess.Popen(
            [str(python), "-m", "ep_driver_fail2drive.worker", "--config", str(config_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", bufsize=1, env=env,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.messages.put(line)
        except (UnicodeError, OSError) as exc:
            self.messages.put({"_reader_error": type(exc).__name__+": "+str(exc)})
        finally:
            self.messages.put(None)

    def plan(self, packet):
        self.sequence += 1
        request_id = self.sequence
        request = {"op": "plan", "request_id": request_id, "packet": packet}
        if self.process.poll() is not None:
            raise WorkerError("numeric worker exited with code %s" % self.process.returncode)
        self.process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout_s
        try:
            line = self.messages.get(timeout=max(0.0, deadline-time.monotonic()))
        except Empty as exc:
            raise WorkerError("numeric worker exceeded %.1fs wall-clock budget" % self.timeout_s) from exc
        if line is None:
            raise WorkerError("numeric worker closed protocol stream")
        if isinstance(line, dict) and "_reader_error" in line:
            raise WorkerError("numeric worker protocol reader failed: "+line["_reader_error"])
        try:
            response = json.loads(line)
        except (TypeError, json.JSONDecodeError) as exc:
            raise WorkerError("non-JSON output on numeric worker stdout") from exc
        if response.get("request_id") != request_id:
            raise WorkerError("numeric worker response request_id mismatch")
        if not response.get("ok", False):
            raise WorkerError("numeric worker failed: %s" % response.get("error", "unknown error"))
        plan = response.get("plan", response)
        if (plan.get("frame_id") != packet["scene"]["frame_id"] or
                abs(float(plan.get("time", -1))-float(packet["time"])) > 1e-7):
            raise WorkerError("plan source frame/time does not match its frozen request")
        return plan

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.log.close()
