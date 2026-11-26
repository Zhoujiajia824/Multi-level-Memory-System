"""Official NAVSIM AbstractAgent, explicitly requiring privileged Scene input."""
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import uuid

import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .input_adapter import build_packet, CAMERAS
from .protocol import digest, write_json
from .profiles import get_profile, is_v1


class CoreProcess:
    def __init__(self, python, settings, timeout_s):
        self.timeout_s = timeout_s
        self.path = Path(settings["output_dir"])
        self.path.mkdir(parents=True, exist_ok=True)
        settings_path = self.path / ("worker_settings_" + uuid.uuid4().hex + ".json")
        worker_keys = ("parameters", "dino_model", "dino_device", "prompts_config", "models_config", "fast",
                       "retrieve", "memory_read_dir", "memory_snapshot_id", "config_file_sha256", "numerical_source_sha256",
                       "evaluation_logs", "evaluation_tokens", "output_dir")
        worker_settings = {key: settings[key] for key in worker_keys}
        worker_settings["benchmark_profile"] = get_profile(settings)["id"]
        for key in ("navsim_version", "traffic_policy", "traffic_agents", "official_commit",
                    "official_source_sha256", "official_metric", "score_name"):
            if key in settings:
                worker_settings[key] = settings[key]
        worker_settings.update(resume=settings.get("resume", True),
                               model_revision=settings.get("model_revision"),
                               model_service=settings.get("model_service", {}),
                               model_service_validation=settings.get("model_service_validation"),
                               model_request_policy=settings.get("model_request_policy", {}),
                               model_goal_retry_limit=settings.get("model_goal_retry_limit",
                                   1 if settings["fast"] == "server" else 0),
                               structured_model_output=settings.get("structured_model_output", False))
        if settings["retrieve"]:
            worker_settings["memory_selection"] = settings["memory_selection"]
        write_json(settings_path, worker_settings)
        self.log = (self.path / (settings_path.stem + ".log")).open("a", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen([python, "-u", "-m", "ep_driver_navsim.worker", "--settings", str(settings_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.responses = queue.Queue()
        def read():
            try:
                for line in self.process.stdout:
                    if line.startswith("EP_NAVSIM_REPLY "):
                        try:
                            self.responses.put(json.loads(line[len("EP_NAVSIM_REPLY "):]))
                        except ValueError as exc:
                            self.responses.put({"protocol_error": str(exc)})
                    else:
                        self.log.write(line)
                        self.log.flush()
            finally:
                self.responses.put({"eof": True})
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        reply = self.receive()
        if reply != {"ready": True}:
            self.close()
            raise RuntimeError("Numerical worker failed to initialize; see " + str(self.path))

    def receive(self):
        try:
            reply = self.responses.get(timeout=self.timeout_s)
        except queue.Empty:
            self.close()
            raise TimeoutError("EP-Driver numerical worker timed out")
        if "eof" in reply or "protocol_error" in reply:
            self.close()
            raise RuntimeError("Numerical worker stopped or returned invalid protocol; inspect worker log")
        return reply

    def compute(self, packet, output_dir):
        request_id = uuid.uuid4().hex
        self.process.stdin.write(json.dumps(dict(request_id=request_id, packet=packet,
            output_dir=str(output_dir)), allow_nan=False) + "\n")
        self.process.stdin.flush()
        reply = self.receive()
        if reply.get("request_id") != request_id:
            raise RuntimeError("Numerical worker response belongs to another request")
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "Numerical worker failed"))
        return reply["result"]

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write('{"operation":"close"}\n')
                    process.stdin.flush()
                    process.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
            self.process = None
            reader = getattr(self, "reader", None)
            if reader is not None:
                reader.join(timeout=5)
        log = getattr(self, "log", None)
        if log is not None and not log.closed:
            log.close()


class EPDriverPrivilegedAgent(AbstractAgent):
    def __init__(self, settings_path, core_python, trajectory_sampling=None, timeout_s=600.0):
        sampling = trajectory_sampling or TrajectorySampling(time_horizon=4.0, interval_length=0.5)
        if sampling.num_poses != 8 or not np.isclose(sampling.interval_length, 0.5):
            raise ValueError("NAVSIM integration requires four seconds at 2Hz")
        initial_settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        if is_v1(initial_settings):
            # v1.1's official base constructor only accepts requires_scene.
            super().__init__(requires_scene=True)
            self._trajectory_sampling = sampling
        else:
            super().__init__(trajectory_sampling=sampling, requires_scene=True)
        self.benchmark_profile = get_profile(initial_settings)["id"]
        self.settings_path = str(settings_path)
        self.core_python = str(core_python)
        self.timeout_s = float(timeout_s)
        self.core = None
        atexit.register(self.close)

    def name(self):
        return "EP-Driver-privileged-current-history-" + self.benchmark_profile

    def get_sensor_config(self):
        return SensorConfig(**dict({name: [3] for name in CAMERAS}, lidar_pc=False))

    def initialize(self):
        if self.core is not None:
            return
        self.settings = json.loads(Path(self.settings_path).read_text(encoding="utf-8"))
        if get_profile(self.settings)["id"] != self.benchmark_profile:
            raise ValueError("Benchmark profile changed after the SDK agent was constructed")
        if self.settings["input_condition"] != "privileged_current_history_tracks_hd_map_route_traffic_lights":
            raise ValueError("Privileged-input condition must be explicit")
        self.core = CoreProcess(self.core_python, self.settings, self.timeout_s)

    def compute_trajectory(self, agent_input, scene=None):
        if scene is None:
            raise ValueError("This is a privileged baseline; Scene/map/tracks are required")
        self.initialize()
        token = str(scene.scene_metadata.initial_token)
        if token not in self.settings["evaluation_tokens"]:
            raise ValueError("Token is outside the frozen evaluation split")
        # Avoid interpreting external tokens as filesystem paths.
        output = Path(self.settings["output_dir"]) / "samples" / digest(token)[:24]
        packet = build_packet(agent_input, scene, output / "cameras",
            desired_speed_mps=self.settings["desired_speed_mps"], map_radius_m=self.settings["map_radius_m"],
            benchmark_profile=self.benchmark_profile)
        result = self.core.compute(packet, output)
        if result["frame_id"] != token:
            raise RuntimeError("Output trajectory has the wrong source frame")
        poses = np.asarray(result["poses"], dtype=np.float32)
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise RuntimeError("Invalid NAVSIM trajectory")
        return Trajectory(poses, self._trajectory_sampling)

    def close(self):
        if self.core is not None:
            self.core.close()
            self.core = None
