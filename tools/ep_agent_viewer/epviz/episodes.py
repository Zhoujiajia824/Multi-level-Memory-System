"""Read-only, frame-exact access to EP-Driver experiment artifacts.

The viewer never imports a driving runtime. JSONL files are indexed incrementally
by byte offset; a partially written final line is retried on the next refresh.
Camera images are not interpolated or borrowed from adjacent planning frames.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import gzip
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat as stat_mode
import threading
import zlib
from typing import Any


TIME_TOLERANCE = 1e-6
_STREAMS = ("plans", "frames", "timing", "model_inputs", "model_outputs",
            "fast", "answers", "retrieval", "quality", "goals",
            "execution_audit", "predictions", "evaluation_events")
_PRUNE = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache"}


def _disk(path: Path) -> str:
    """Support pre-existing deeply nested Windows experiment paths."""
    value = str(path.absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return value


def _stat(path: Path):
    try:
        return os.stat(_disk(path))
    except (FileNotFoundError, NotADirectoryError, OSError):
        return None


def _path(value) -> Path:
    value = str(value)
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _inside(path: Path, root: Path) -> bool:
    try:
        a, b = _path(path).resolve(), _path(root).resolve()
        return os.path.commonpath((str(a), str(b))) == str(b)
    except (ValueError, OSError):
        return False


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _identity(row: dict) -> tuple[str | None, float | None]:
    for source in (row, row.get("plan"), row.get("quality"), row.get("answer")):
        if isinstance(source, dict) and source.get("frame_id") is not None:
            return str(source["frame_id"]), _finite(source.get("time", row.get("time")))
    return None, _finite(row.get("time"))


def _same_time(a, b) -> bool:
    a, b = _finite(a), _finite(b)
    return a is not None and b is not None and abs(a - b) <= TIME_TOLERANCE


@dataclass(frozen=True)
class _Record:
    offset: int
    length: int
    frame_id: str | None
    time: float | None


class _JsonlIndex:
    """An offset index, not an in-memory copy of every trajectory."""
    def __init__(self, path: Path, root: Path | None = None):
        self.path = path
        self.root = root or path.parent
        self.records: list[_Record] = []
        self.by_frame: dict[str, _Record] = {}
        self.offset = 0
        self.size = 0
        self.signature = None
        self.modified = None
        self.prefix = b""
        self.tail = b""
        self.warnings: list[str] = []
        self.bytes_read = 0
        self.generation = 0

    def _reset(self):
        self.records.clear()
        self.by_frame.clear()
        self.offset = 0
        self.warnings.clear()
        self.generation += 1

    def refresh(self):
        stat = _stat(self.path) if _inside(self.path, self.root) else None
        if stat is None:
            if self.signature is not None:
                self._reset()
                self.signature = None
                self.size = 0
            return
        signature = (stat.st_dev, stat.st_ino)
        with open(_disk(self.path), "rb") as stream:
            prefix = stream.read(min(128, self.size, stat.st_size))
            stream.seek(max(0, self.size - len(self.tail)))
            previous_tail = stream.read(len(self.tail))
            replaced = self.signature is not None and (
                signature != self.signature or stat.st_size < self.size or
                (stat.st_size == self.size and self.modified != stat.st_mtime_ns) or
                (self.tail and previous_tail != self.tail) or
                (self.prefix and prefix != self.prefix[:len(prefix)]))
            if replaced:
                self._reset()
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                self.bytes_read += len(line)
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # The writer may currently be in the middle of this line.
                    break
                self.offset = stream.tell()
                if not line.strip():
                    continue
                try:
                    row = json.loads(line.decode("utf-8-sig"))
                    if not isinstance(row, dict):
                        raise ValueError("record is not an object")
                    frame_id, timestamp = _identity(row)
                    record = _Record(start, len(line), frame_id, timestamp)
                    self.records.append(record)
                    if frame_id is not None:
                        self.by_frame[frame_id] = record
                except (ValueError, UnicodeError) as exc:
                    if len(self.warnings) < 20:
                        self.warnings.append(f"{self.path.name}: invalid complete line at byte {start}: {exc}")
            stream.seek(0)
            self.prefix = stream.read(min(128, stat.st_size))
            stream.seek(max(0, stat.st_size - 128))
            self.tail = stream.read(128)
        self.signature = signature
        self.modified = stat.st_mtime_ns
        self.size = stat.st_size

    def read(self, record: _Record | None):
        if record is None or not _inside(self.path, self.root):
            return None
        try:
            with open(_disk(self.path), "rb") as stream:
                stream.seek(record.offset)
                value = json.loads(stream.read(record.length).decode("utf-8-sig"))
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, UnicodeError):
            return None


class _Episode:
    def __init__(self, path: Path, identifier: str):
        self.path, self.id = path, identifier
        self.streams = {name: _JsonlIndex(path / (name + ".jsonl"), path) for name in _STREAMS}
        self.timeline: list[_Record] = []
        self.captures: dict[str, Path] = {}
        self.camera_manifests: dict[str, Path] = {}
        self.json_cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self.artifact_headers: dict[str, tuple[Any, Any]] = {}
        self.warnings: list[str] = []

    def json(self, path: Path):
        if not _inside(path, self.path):
            return None
        stat = _stat(path)
        if stat is None:
            return None
        key, signature = str(path), (stat.st_size, stat.st_mtime_ns)
        cached = self.json_cache.get(key)
        if cached is not None and cached[0] == signature:
            self.json_cache.move_to_end(key)
            return cached[1]
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(_disk(path), "rt", encoding="utf-8-sig") as stream:
                value = json.load(stream)
        except (OSError, ValueError, UnicodeError, EOFError, zlib.error):
            # A JSON artifact may also still be in the process of being written.
            return None
        self.json_cache[key] = (signature, value)
        self.json_cache.move_to_end(key)
        while len(self.json_cache) > 80:
            self.json_cache.popitem(last=False)
        return value

    def _folders(self, name: str):
        path = self.path / name
        try:
            return [_path(entry.path) for entry in os.scandir(_disk(path))
                    if entry.is_dir(follow_symlinks=False) and _inside(_path(entry.path), self.path)]
        except OSError:
            return []

    def header(self, path):
        stat = _stat(path)
        if stat is None:
            return None
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self.artifact_headers.get(str(path))
        if cached is not None and cached[0] == signature:
            return cached[1]
        value = self.json(path)
        if not isinstance(value, dict):
            return None
        header = {key: value.get(key) for key in ("frame_id", "time")}
        self.artifact_headers[str(path)] = (signature, header)
        return header

    def refresh(self):
        for stream in self.streams.values():
            stream.refresh()
        # Planning is written before execution; it remains visible if execution fails.
        primary = self.streams["plans"].by_frame
        fallback = self.streams["frames"].by_frame
        all_frames = dict(fallback)
        all_frames.update(primary)
        self.timeline = sorted(all_frames.values(), key=lambda r: (
            r.time is None, r.time if r.time is not None else 0, r.offset, r.frame_id or ""))
        self.captures.clear()
        for folder in self._folders("captures"):
            path = folder / "scene.json"
            scene = self.header(path)
            if isinstance(scene, dict) and scene.get("frame_id"):
                self.captures[str(scene["frame_id"])] = folder
            else:
                evidence = self.header(folder / "planning_evidence.json")
                if isinstance(evidence, dict) and evidence.get("frame_id"):
                    self.captures[str(evidence["frame_id"])] = folder
        self.camera_manifests.clear()
        for folder in self._folders("camera_frames"):
            manifest = self.header(folder / "manifest.json")
            if isinstance(manifest, dict) and manifest.get("frame_id"):
                self.camera_manifests[str(manifest["frame_id"])] = folder / "manifest.json"


class EpisodeStore:
    """A read-only artifact store restricted to one explicitly selected root.

    ``root`` may be one episode, a results directory, or a parent containing
    several experiment groups. IDs are opaque and remain stable across polls.
    ``engine_root`` is retained for the caller's optional geometry adapter; it
    never authorizes reading assets outside an episode.
    """
    def __init__(self, root, engine_root=None):
        self.root = Path(root).expanduser().resolve()
        root_stat = _stat(self.root)
        if root_stat is None or not stat_mode.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"Experiment root is not a directory: {self.root}")
        self.engine_root = Path(engine_root).resolve() if engine_root else None
        self._episodes: dict[str, _Episode] = {}
        self._navsim_runs = {}
        self._navsim_active = OrderedDict()
        self._active_run_ids = set()
        self._lock = threading.RLock()

    @staticmethod
    def _is_episode(path: Path) -> bool:
        return any(_stat(path / name) is not None for name in ("plans.jsonl", "frames.jsonl")) or (
            _stat(path / "configuration.json") is not None and _stat(path / "summary.json") is not None)

    def discover(self) -> list[dict]:
        with self._lock:
            found = []
            for directory, dirs, _files in os.walk(_disk(self.root), followlinks=False):
                path = _path(directory)
                dirs[:] = [name for name in dirs if name not in _PRUNE]
                if not _inside(path, self.root):
                    dirs[:] = []
                    continue
                from .navsim import NavsimEpisode, NavsimRun
                if NavsimEpisode.is_sample(path):
                    relative = path.relative_to(self.root).as_posix()
                    identifier = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
                    if identifier not in self._episodes:
                        candidate = path.parent.parent
                        run = None
                        if path.parent.name == "samples" and _inside(candidate, self.root):
                            key = str(candidate)
                            if key not in self._navsim_runs:
                                self._navsim_runs[key] = NavsimRun(candidate, self.root)
                            run = self._navsim_runs[key]
                        self._episodes[identifier] = NavsimEpisode(path, identifier, self.root, run=run)
                    episode = self._episodes[identifier]
                    found.append(episode.listing())
                    dirs[:] = []
                    continue
                if self._is_episode(path):
                    relative = path.relative_to(self.root).as_posix()
                    identifier = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
                    episode = self._episodes.setdefault(identifier, _Episode(path, identifier))
                    episode.refresh()
                    found.append(self._describe(episode))
                    dirs[:] = []
            self._active_run_ids = {row["run_id"] for row in found if row.get("run_id")}
            for run in self._navsim_runs.values():
                members = [row for row in found if row.get("run_id") == run.id]
                run.sample_count = len(members)
                run.completed_samples = sum(row["frame_count"] > 1 for row in members)
                run.stages = {stage: sum(row.get("stage") == stage for row in members)
                              for stage in ("stage_one", "stage_two", "unknown")}
            return sorted(found, key=lambda row: row["path"])

    def list_runs(self) -> list[dict]:
        """Metadata for the NAVSIM runs from the latest discovery snapshot."""
        with self._lock:
            return sorted((run.metadata() for run in self._navsim_runs.values()
                           if run.id in self._active_run_ids), key=lambda row: row["path"])

    def get_run(self, identifier: str) -> dict:
        with self._lock:
            run = next((run for run in self._navsim_runs.values() if run.id == identifier), None)
            if run is None:
                self.discover()
                run = next((run for run in self._navsim_runs.values() if run.id == identifier), None)
            run_stat = _stat(run.path) if run is not None and _inside(run.path, self.root) else None
            if run_stat is None or not stat_mode.S_ISDIR(run_stat.st_mode):
                raise KeyError("Unknown NAVSIM run")
            return run.details()

    def _episode(self, identifier: str) -> _Episode:
        if identifier not in self._episodes:
            self.discover()
        if identifier not in self._episodes:
            raise KeyError("Unknown episode")
        episode = self._episodes[identifier]
        if not _inside(episode.path, self.root):
            raise ValueError("Episode escaped the authorized root")
        return episode

    def _describe(self, episode):
        if hasattr(episode, "describe"):
            return episode.describe()
        summary = episode.json(episode.path / "summary.json") or {}
        configuration = episode.json(episode.path / "configuration.json") or {}
        has_camera = bool(episode.camera_manifests)
        platform = "carla" if has_camera or "carla_runtime" in summary else "highway" if (
            _stat(episode.path / "map.json") is not None) else "unknown"
        return dict(id=episode.id, name=episode.path.name,
                    path=episode.path.relative_to(self.root).as_posix(),
                    scenario=summary.get("scenario"), platform=platform,
                    live=not bool(summary), frame_count=len(episode.timeline),
                    duration=summary.get("simulated_seconds", episode.timeline[-1].time if episode.timeline else None),
                    execution_mode=summary.get("execution_mode", configuration.get("execution_mode")),
                    summary=summary)

    def refresh(self, identifier: str) -> dict:
        with self._lock:
            episode = self._episode(identifier)
            self._refresh_episode(episode)
            return self._describe(episode)

    def _refresh_episode(self, episode):
        episode.refresh()
        if hasattr(episode, "release_payload"):
            self._navsim_active[episode.id] = episode
            self._navsim_active.move_to_end(episode.id)
            while len(self._navsim_active) > 4:
                _, previous = self._navsim_active.popitem(last=False)
                previous.release_payload()

    def _relative(self, episode: _Episode, value) -> str | None:
        """Map archived absolute paths after a move, never read their old target."""
        if not isinstance(value, (str, os.PathLike)):
            return None
        text = str(value).replace("\\", "/")
        if text.startswith("//?/"):
            text = text[4:]
        parts = PurePosixPath(text).parts
        if ".." in parts or "\x00" in text:
            return None
        absolute = text.startswith("/") or bool(re.match(r"^[A-Za-z]:/", text))
        if absolute:
            # Exact episode-name segment is the relocation anchor. A raw system
            # path or a same-named file in another episode is never followed.
            positions = [i for i, part in enumerate(parts) if part == episode.path.name]
            candidates = [PurePosixPath(*parts[i + 1:]).as_posix() for i in reversed(positions)]
            try:
                direct = Path(str(value)).resolve()
                if _inside(direct, episode.path):
                    candidates.insert(0, direct.relative_to(episode.path).as_posix())
            except (OSError, ValueError):
                pass
        else:
            candidates = [PurePosixPath(text).as_posix()]
        for relative in candidates:
            try:
                path = self._resolve(episode, relative)
                candidate_stat = _stat(path)
                if candidate_stat is not None and stat_mode.S_ISREG(candidate_stat.st_mode):
                    return path.relative_to(episode.path).as_posix()
            except (ValueError, OSError):
                continue
        return None

    @staticmethod
    def _resolve(episode, relative) -> Path:
        if not isinstance(relative, str) or not relative or "\x00" in relative:
            raise ValueError("Asset path must be an episode-relative path")
        normalized = relative.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        if normalized.startswith("/") or ":" in normalized or ".." in parts:
            raise ValueError("Absolute paths and parent traversal are not allowed")
        path = (episode.path / Path(*parts)).resolve()
        if not _inside(path, episode.path):
            raise ValueError("Asset path escaped its episode")
        return path

    def resolve_asset(self, identifier: str, relative: str) -> Path:
        with self._lock:
            path = self._resolve(self._episode(identifier), relative)
            asset_stat = _stat(path)
            if asset_stat is None or not stat_mode.S_ISREG(asset_stat.st_mode):
                raise FileNotFoundError("Asset is unavailable")
            return Path(_disk(path)) if os.name == "nt" and len(str(path)) >= 260 else path

    def get_manifest(self, identifier: str) -> dict:
        with self._lock:
            episode = self._episode(identifier)
            self._refresh_episode(episode)
            if hasattr(episode, "manifest"):
                return episode.manifest(self)
            result = self._describe(episode)
            configuration = episode.json(episode.path / "configuration.json") or {}
            frames = []
            for index, record in enumerate(episode.timeline):
                camera_path = episode.camera_manifests.get(record.frame_id)
                camera_manifest = episode.json(camera_path) if camera_path else None
                available = False
                if camera_manifest and _same_time(camera_manifest.get("time"), record.time):
                    camera_values = camera_manifest.get("cameras") or []
                    if isinstance(camera_values, dict):
                        camera_values = camera_values.values()
                    available = any(isinstance(camera, dict) and
                                    camera.get("frame") == camera_manifest.get("carla_frame") and
                                    _same_time(camera.get("timestamp"), camera_manifest.get("carla_timestamp")) and
                                    self._relative(episode, camera.get("rgb_path", camera.get("path")))
                                    for camera in camera_values)
                capture_folder = episode.captures.get(record.frame_id)
                capture_header = episode.header(capture_folder / "scene.json") if capture_folder else None
                captured = bool(capture_header and _same_time(capture_header.get("time"), record.time))
                state_ref = episode.streams["frames"].by_frame.get(record.frame_id)
                frames.append(dict(index=index, frame_id=record.frame_id, time=record.time,
                                   pending=state_ref is None or not _same_time(state_ref.time, record.time),
                                   camera_available=available,
                                   captured=captured))
            video = dict(result.get("summary", {}).get("video") or {})
            stored_video = episode.json(episode.path / "video_recording.json")
            if isinstance(stored_video, dict):
                video.update(stored_video)
            video_path = self._relative(episode, video.get("path", "video.mp4"))
            video["path"] = video_path
            video["available"] = video_path is not None
            # Constant-fps video is only an overview unless an explicit frame
            # index permits source-frame synchronization.
            video["sync_available"] = False
            return dict(**result, configuration=configuration, frames=frames,
                        times=[frame["time"] for frame in frames], video=video,
                        model_event_indices=[frame["index"] for frame in frames
                                             if frame["frame_id"] in episode.streams["model_inputs"].by_frame],
                        model_events=[dict(frame_id=r.frame_id, time=r.time) for r in episode.streams["model_outputs"].records],
                        source_frames=[f["index"] for f in frames if f["camera_available"] or f["captured"]],
                        warnings=[w for stream in episode.streams.values() for w in stream.warnings],
                        alignment_policy="exact frame_id and source time; no nearest-frame camera substitution")

    def get_frame(self, identifier: str, index: int) -> dict:
        with self._lock:
            episode = self._episode(identifier)
            self._refresh_episode(episode)
            if hasattr(episode, "frame"):
                return episode.frame(self, index)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(episode.timeline):
                raise IndexError("Planning frame index is out of range")
            record = episode.timeline[index]
            frame_id, timestamp = record.frame_id, record.time
            warnings, evidence = [], {}

            def exact(name, source_id=frame_id, source_time=timestamp):
                stream = episode.streams[name]
                ref = stream.by_frame.get(source_id)
                if ref is None:
                    return None
                if not _same_time(ref.time, source_time):
                    warnings.append(f"{name}: matching frame_id has a different/missing source time; not joined")
                    return None
                evidence[name] = dict(path=stream.path.name, byte_offset=ref.offset,
                                      frame_id=ref.frame_id, time=ref.time)
                return stream.read(ref)

            plan_record, state = exact("plans"), exact("frames")
            folder = episode.captures.get(frame_id)
            scene, planning_evidence = None, None
            if folder:
                candidate = episode.json(folder / "scene.json")
                if isinstance(candidate, dict) and str(candidate.get("frame_id")) == frame_id and _same_time(candidate.get("time"), timestamp):
                    scene = candidate
                    evidence["scene"] = (folder / "scene.json").relative_to(episode.path).as_posix()
                candidate = episode.json(folder / "planning_evidence.json")
                if isinstance(candidate, dict) and str(candidate.get("frame_id")) == frame_id and _same_time(candidate.get("time"), timestamp):
                    planning_evidence = candidate
                    evidence["planning_evidence"] = (folder / "planning_evidence.json").relative_to(episode.path).as_posix()
            cameras, camera_state = [], None
            manifest_path = episode.camera_manifests.get(frame_id)
            if manifest_path:
                manifest = episode.json(manifest_path)
                if _same_time(manifest.get("time"), timestamp):
                    evidence["camera_manifest"] = manifest_path.relative_to(episode.path).as_posix()
                    raw = episode.json(manifest_path.parent / "current_observed_state.json")
                    # Raw camera state uses CARLA frame/timestamp rather than the
                    # runner's namespace and relative episode time.
                    raw_frame = raw.get("frame", raw.get("carla_frame")) if isinstance(raw, dict) else None
                    raw_time = raw.get("timestamp", raw.get("carla_timestamp")) if isinstance(raw, dict) else None
                    raw_exact = isinstance(raw, dict) and raw_frame == manifest.get("carla_frame") and _same_time(raw_time, manifest.get("carla_timestamp"))
                    if raw_exact:
                        camera_state = raw
                        evidence["camera_state"] = (manifest_path.parent / "current_observed_state.json").relative_to(episode.path).as_posix()
                    elif raw:
                        warnings.append("Current observed camera state is not exactly matched; camera-state overlay disabled")
                    camera_values = manifest.get("cameras") or []
                    if isinstance(camera_values, dict):
                        camera_values = [dict(value, view=key) for key, value in camera_values.items() if isinstance(value, dict)]
                    for camera in camera_values:
                        if not isinstance(camera, dict):
                            continue
                        image_path = self._relative(episode, camera.get("rgb_path", camera.get("path")))
                        if camera.get("frame") != manifest.get("carla_frame") or not _same_time(camera.get("timestamp"), manifest.get("carla_timestamp")):
                            warnings.append(f"{camera.get('view', 'camera')}: frame/timestamp mismatch; image excluded")
                            continue
                        if image_path:
                            payload = dict(camera)
                            payload.update(path=image_path, image=image_path, frame_id=frame_id,
                                           time=timestamp, source_time=timestamp, aligned=True,
                                           calibrated=bool(raw_exact and camera.get("world_to_camera_ue")))
                            cameras.append(payload)
                else:
                    warnings.append("Camera manifest source time differs from planning time; images excluded")
            plan = (plan_record or {}).get("plan") or (planning_evidence or {}).get("plan")
            ego = (state or {}).get("ego") or (scene or {}).get("ego")
            actors = (state or {}).get("actors")
            if actors is None:
                actors = (scene or {}).get("actors", [])
            map_data = episode.json(episode.path / "map.json")
            # Static map geometry can be reused; dynamic rules/observations cannot.
            reconstructed = scene is None
            capture_aligned = scene is not None
            if scene is None:
                scene = dict(frame_id=frame_id, time=timestamp, ego=ego, actors=actors,
                             lanes=(map_data or {}).get("lanes", []),
                             route=(map_data or {}).get("route", []),
                             route_lane_ids=(map_data or {}).get("route_lane_ids", []),
                             drivable=(map_data or {}).get("drivable"), rules=None,
                             diagnostics=(plan or {}).get("diagnostics", {}),
                             source="same-frame state plus static episode map; no captured dynamic rules")
            assets = {}
            if folder and capture_aligned:
                try:
                    for entry in os.scandir(_disk(folder)):
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        for suffix, label in (("_bev.png", "bev"), ("_feature.png", "feature"),
                                              ("_highway_topdown.png", "highway_topdown")):
                            if entry.name.endswith(suffix):
                                assets[label] = self._relative(episode, folder / entry.name)
                except OSError:
                    pass
            overlay = self._relative(episode, (plan_record or {}).get("goal_overlay_path"))
            if overlay:
                assets["goal_overlay"] = overlay
            if manifest_path and cameras:
                assets["mosaic"] = self._relative(episode, manifest_path.parent / "multi_camera_mosaic.png")
            if assets.get("highway_topdown"):
                cameras.append(dict(view="highway_topdown", path=assets["highway_topdown"], image=assets["highway_topdown"],
                                    frame_id=frame_id, time=timestamp, source_time=timestamp, aligned=True, calibrated=False))
            model_input, model_output, retrieval = exact("model_inputs"), exact("model_outputs"), exact("retrieval")
            answer_source = (plan_record or {}).get("answer_source")
            active_model_input, active_model_output, active_retrieval = None, None, None
            if answer_source and answer_source != frame_id:
                # Explicit planner binding, never simply "most recent output".
                source_ref = episode.streams["model_inputs"].by_frame.get(answer_source) or episode.streams["plans"].by_frame.get(answer_source)
                source_time = source_ref.time if source_ref else None
                if source_time is not None and timestamp is not None and source_time <= timestamp:
                    active_model_input = exact("model_inputs", answer_source, source_time)
                    active_model_output = exact("model_outputs", answer_source, source_time)
                    active_retrieval = exact("retrieval", answer_source, source_time)
            if model_output is None:
                model_output = exact("answers") or exact("fast")
            for model_payload in (model_input, active_model_input):
                if isinstance(model_payload, dict):
                    # Actual runner records a list of source-image path strings.
                    # These are source-input evidence, not current-frame RGB.
                    images = model_payload.get("images")
                    model_payload["image_paths"] = [relative for value in images
                        if (relative := self._relative(episode, value)) is not None] if isinstance(images, list) else []
            quality_row = exact("quality")
            quality = (quality_row or {}).get("quality") or (planning_evidence or {}).get("quality")
            result = dict(episode_id=episode.id, index=index, frame_id=frame_id, time=timestamp,
                          source_time=timestamp, pending=state is None,
                          ego=ego, actors=actors, plan=plan, plan_record=plan_record,
                          scene=scene, map=map_data, rules=scene.get("rules"),
                          model_input=model_input, model_output=model_output,
                          active_model_input=active_model_input, active_model_output=active_model_output,
                          model_source_frame_id=answer_source, retrieval=retrieval,
                          active_retrieval=active_retrieval, timing=exact("timing"), quality=quality,
                          predictions=exact("predictions"), execution_audit=exact("execution_audit"),
                          goals=exact("goals"), cameras=cameras, assets=assets,
                          raw=camera_state, camera_state=camera_state,
                          configuration=episode.json(episode.path / "configuration.json") or {},
                          planning_evidence=planning_evidence,
                          alignment=dict(policy="exact_frame_id_and_source_time",
                                         camera="exact" if cameras else "unavailable_at_this_source_frame",
                                         scene="static_map_with_exact_state" if reconstructed else "exact_capture",
                                         model_binding="explicit_plan_answer_source" if answer_source else None),
                          evidence=evidence, warnings=warnings)
            if not cameras and episode.camera_manifests:
                result["warnings"].append("No archived RGB at this planning frame; images are stored at a lower rate, so no neighboring image was substituted")
            return result
