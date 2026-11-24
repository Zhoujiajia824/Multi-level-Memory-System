"""Detached NAVSIM artifacts, with one independent anchor per viewer episode.

No NAVSIM, nuPlan, model, planner, or scoring code is imported. All display
coordinates use the source rear-axle pose as the fixed SE(2) origin. Planner
states and actor boxes are centres; submitted/official ego paths are rear axles.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
import csv
import hashlib
import json
import math
from pathlib import Path
from functools import lru_cache

from .episodes import _Episode, _Record, _disk, _finite, _inside, _stat


_PACKETS = OrderedDict()
OFFICIAL_RENDERER_VERSION = "navsim-sdk-plots-1.0"

_RUN_METRICS = (
    ("NC", "no_at_fault_collisions", "无责任碰撞"),
    ("DAC", "drivable_area_compliance", "可行驶区域遵守"),
    ("DDC", "driving_direction_compliance", "行驶方向遵守"),
    ("TLC", "traffic_light_compliance", "交通灯遵守"),
    ("EP", "ego_progress", "自车行驶进度"),
    ("TTC", "time_to_collision_within_bound", "碰撞时间安全性"),
    ("LK", "lane_keeping", "车道保持"),
    ("HC", "history_comfort", "历史运动衔接舒适性"),
    ("EC", "two_frame_extended_comfort", "相邻规划扩展舒适性"),
)


def _stage(value):
    return {"original": "stage_one", "synthetic": "stage_two",
            "stage_one": "stage_one", "stage_two": "stage_two"}.get(str(value).lower())


@lru_cache(maxsize=1)
def _legacy_geometry_decoder():
    # Load only the transport helper, without modifying sys.path or importing
    # the SDK, core package, planner, or evaluator into this viewer process.
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "src/ep_driver_navsim/geometry_transfer.py"
    if not path.is_file():
        raise ValueError("Legacy NAVSIM geometry recovery requires the repository transport helper")
    spec = importlib.util.spec_from_file_location("_epviz_navsim_geometry_transfer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_drivable


def _drivable(packet):
    """Decode archived binary roads locally, without importing a planner runtime.

    GeoJSON carries Python's round-trippable float coordinates to the diagnostic
    renderer. Converting back to WKT would reintroduce the collapsed-hole bug.
    """
    if "drivable_geometry" not in packet:
        text = packet.get("drivable_wkt")
        if text is None:
            return None
        import shapely
        if shapely.from_wkt(text).is_valid:
            return text
        from shapely.geometry import mapping
        geometry, evidence = _legacy_geometry_decoder()(packet)
        return dict(mapping(geometry), geometry_transfer=evidence)
    if "drivable_wkt" in packet:
        raise ValueError("Archived NAVSIM packet has two conflicting road encodings")
    wire = packet["drivable_geometry"]
    if (not isinstance(wire, dict) or set(wire) != {"encoding", "data", "sha256"}
            or wire.get("encoding") != "wkb_hex_float64_le_2d" or not isinstance(wire.get("data"), str)):
        raise ValueError("Unrecognized archived NAVSIM road encoding")
    raw = bytes.fromhex(wire["data"])
    if hashlib.sha256(raw).hexdigest() != wire["sha256"]:
        raise ValueError("Archived NAVSIM road checksum differs")
    import numpy as np
    import shapely
    from shapely.geometry import mapping
    geometry = shapely.from_wkb(raw)
    if (geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty
            or not geometry.is_valid or geometry.area <= 0 or shapely.has_z(geometry)
            or not np.isfinite(shapely.get_coordinates(geometry)).all()
            or shapely.to_wkb(geometry, byte_order=1, output_dimension=2) != raw):
        raise ValueError("Archived NAVSIM binary road geometry is invalid")
    return mapping(geometry)


class NavsimRun(_Episode):
    """One shared cache per run, independent of which anchor is selected.

    Aggregates are read only from the SDK's named CSV summary rows. In
    particular, an average over the available anchors is never a run score.
    """
    def __init__(self, path, authorized_root):
        relative = path.relative_to(authorized_root).as_posix()
        identifier = hashlib.sha256(("navsim-run:" + relative).encode("utf-8")).hexdigest()[:16]
        super().__init__(path, identifier)
        self.authorized_root = authorized_root
        self.sample_count = 0
        self.completed_samples = 0
        self.stages = dict(stage_one=0, stage_two=0, unknown=0)
        self._csv_signature = None
        self._csv_rows = {}
        self._csv_error = None

    def benchmark(self):
        value = self.json(self.path / "benchmark_result.json")
        return value if isinstance(value, dict) else {}

    def metadata(self):
        settings = self.json(self.path / "run_settings.json") or {}
        settings = settings if isinstance(settings, dict) else {}
        benchmark = self.benchmark()
        fast = settings.get("fast")
        condition = "基础规划" if fast == "off" else "基础规划＋模型" if fast in ("api", "server") else "输入条件未归档"
        if settings.get("retrieve") is True:
            condition += "＋记忆检索"
        expected = settings.get("evaluation_tokens")
        expected = len(expected) if isinstance(expected, list) else benchmark.get("expected_tokens", benchmark.get("evaluated_tokens"))
        return dict(id=self.id, name=self.path.name, path=self.path.relative_to(self.authorized_root).as_posix(),
            label=condition + " · " + self.path.name, condition_label=condition,
            split=benchmark.get("split", settings.get("split")),
            status=benchmark.get("status", "pending"), run_phase=benchmark.get("phase"), sample_count=self.sample_count,
            completed_samples=self.completed_samples, expected_samples=expected,
            stages=dict(self.stages), model_backend=fast, retrieval_enabled=settings.get("retrieve"),
            standard_sensor_only_comparable=benchmark.get("standard_sensor_only_comparable"))

    def _official_rows(self):
        # Stored absolute paths may come from another host. Resolve only the
        # basename within this authorized run, never the original target.
        basename = str(self.benchmark().get("official_csv", "")).replace("\\", "/").split("/")[-1]
        path = self.path / "official_scores" / basename
        stat = _stat(path) if basename and _inside(path, self.path) else None
        signature = (str(path), stat.st_size, stat.st_mtime_ns) if stat else None
        if signature != self._csv_signature:
            self._csv_signature, self._csv_rows, self._csv_error = signature, {}, None
            if stat:
                try:
                    with open(_disk(path), encoding="utf-8-sig", newline="") as stream:
                        for row in csv.DictReader(stream):
                            token = row.get("token")
                            if not token:
                                continue
                            if token in self._csv_rows:
                                raise ValueError("Duplicate token in official CSV")
                            self._csv_rows[token] = row
                except (OSError, ValueError, UnicodeError, csv.Error) as exc:
                    self._csv_rows = {}
                    self._csv_error = type(exc).__name__ + ": " + str(exc)
        return (path.relative_to(self.path).as_posix() if stat else None), self._csv_rows

    def csv_record(self, token):
        return self._official_rows()[1].get(token)

    def details(self):
        result = self.metadata()
        path, rows = self._official_rows()
        first = rows.get("extended_pdm_score_stage_one", {})
        second = rows.get("extended_pdm_score_stage_two", {})
        combined = rows.get("extended_pdm_score_combined", {})
        scores = dict(combined=_finite(combined.get("score")),
                      stage_one=_finite(first.get("score")), stage_two=_finite(second.get("score")))
        metrics = [dict(key=key, label=label,
                        stage_one=_finite(first.get(field + "_stage_one")),
                        stage_two=_finite(second.get(field + "_stage_two")))
                   for key, field, label in _RUN_METRICS]
        warnings = []
        complete = result["status"] == "complete"
        scores_status = "failed" if result["status"] == "failed" else "pending"
        failure = self.benchmark().get("error")
        if scores_status == "failed" and isinstance(failure, dict):
            warnings.append(str(failure.get("type", "Error")) + ": " + str(failure.get("message", "")))
        if complete:
            valid = all(str(row.get("valid", "")).lower() == "true" for row in (first, second, combined))
            values = list(scores.values()) + [row[stage] for row in metrics for stage in ("stage_one", "stage_two")]
            scores_status = "complete" if valid and all(v is not None and 0 <= v <= 1 for v in values) else "unavailable"
            expected = _finite(self.benchmark().get("official_two_stage_epdms"))
            if expected is not None and scores["combined"] is not None and abs(expected - scores["combined"]) > 1e-9:
                scores_status = "unavailable"
                warnings.append("官方CSV综合分与benchmark_result不一致，未展示为有效汇总。")
            if scores_status != "complete":
                warnings.append("完整且有效的官方两阶段汇总行尚不可用。")
        if self._csv_error:
            warnings.append("官方CSV读取失败：" + self._csv_error)
        if scores_status != "complete":
            # A partial file or a finished anchor cannot stand in for the
            # completed run's aggregation, even if some numbers are present.
            scores = {key: None for key in scores}
            metrics = [{**row, "stage_one": None, "stage_two": None} for row in metrics]
        result.update(scores=scores, metrics=metrics, scores_status=scores_status,
            official_csv=path, score_source="official_csv_named_summary_rows",
            metric_scale="0_to_1", warnings=warnings,
            score_semantics="SDK two-stage aggregation; not an arithmetic mean of available anchors",
            input_condition=self.benchmark().get("input_condition"),
            failed_samples=self.benchmark().get("failed_tokens"),
            error=failure if isinstance(failure, dict) else None,
            condition_audit=self.benchmark().get("condition_audit"))
        return result


def _compact(value):
    """Keep text/metadata, omit embedded map, history and feature tensors."""
    bulky = {"history", "lanes", "route", "predictions", "parameters",
             "embedding", "features", "states", "controls", "drivable_wkt", "drivable_geometry",
             "visibility_wkt", "planning_scope", "iteration_history", "_representations",
             "representations", "scene", "raw_observation"}
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if k not in bulky}
    if isinstance(value, list):
        if len(value) > 100 or (value and isinstance(value[0], list) and len(value) > 16):
            return {"omitted_array_length": len(value)}
        return [_compact(v) for v in value]
    return value


def _pose(row, origin):
    """Global x/y/yaw -> source-rear-axle local x-forward/y-left/yaw."""
    x, y, yaw = map(float, row[:3])
    dx, dy = x - origin[0], y - origin[1]
    c, s = math.cos(origin[2]), math.sin(origin[2])
    angle = math.atan2(math.sin(yaw - origin[2]), math.cos(yaw - origin[2]))
    return [c * dx + s * dy, -s * dx + c * dy, angle]


def _rows(value, columns=3):
    return isinstance(value, list) and bool(value) and all(
        isinstance(row, list) and len(row) >= columns and
        all(_finite(v) is not None for v in row[:columns]) for row in value)


class NavsimEpisode(_Episode):
    @staticmethod
    def current_attempt_complete(path):
        """A new failed/running attempt cannot expose an older saved replay."""
        state_path = path / "sample_state.json"
        if _stat(state_path) is None:
            return True  # Older archives did not have an attempt marker.
        try:
            if not _inside(state_path, path):
                return False
            with open(_disk(state_path), encoding="utf-8-sig") as stream:
                state = json.load(stream)
            if (state.get("schema") != "ep-navsim-sample-state-1" or state.get("status") != "complete"
                    or not state.get("attempt_id") or not state.get("source_token")
                    or not state.get("identity_sha256")):
                return False
            marker_path = path / "sample_complete.json"
            if not _inside(marker_path, path):
                return False
            with open(_disk(marker_path), encoding="utf-8-sig") as stream:
                marker = json.load(stream)
            return (marker.get("schema") == "ep-navsim-completion-1" and marker.get("status") == "complete"
                    and marker.get("attempt_id") == state["attempt_id"]
                    and marker.get("source_token") == state.get("source_token")
                    and marker.get("identity_sha256") == state.get("identity_sha256"))
        except (OSError, ValueError, TypeError, AttributeError):
            return False

    @staticmethod
    def is_sample(path):
        return (any(_stat(path / name) is not None for name in ("input_packet.json", "input_packet.json.gz")) and any(
            _stat(path / name) is not None for name in ("trajectory.json", "planning_evidence.json"))
                and NavsimEpisode.current_attempt_complete(path))

    def __init__(self, path, identifier, authorized_root, run=None):
        super().__init__(path, identifier)
        self.authorized_root = authorized_root
        candidate = path.parent.parent
        self.run = (run or NavsimRun(candidate, authorized_root)) if (
            path.parent.name == "samples" and _inside(candidate, authorized_root)) else None
        self.identity, self.trace, self.plan, self.submitted, self.planning = {}, None, {}, {}, {}
        self.trace_path, self.packet_path = None, None
        self._listing_signature = None
        self._listing_state = None

    def _file(self, *names):
        return next((self.path / name for name in names if _inside(self.path / name, self.path)
                     and _stat(self.path / name) is not None), None)

    def validate_asset(self, relative):
        if not self.current_attempt_complete(self.path):
            raise FileNotFoundError("Current NAVSIM attempt is incomplete or failed; old assets are excluded")
        state = self.json(self.path / "sample_state.json") or {}
        attempt = state.get("attempt_id")
        if attempt and relative.replace("\\", "/").startswith("official_replay/"):
            capture = self.json(self.path / "official_replay/capture_status.json") or {}
            if (capture.get("status") != "complete" or capture.get("sample_attempt_id") != attempt
                    or capture.get("token") != state.get("source_token")):
                raise FileNotFoundError("Official replay belongs to a different sample attempt")
            if relative.endswith((".mp4", ".png")):
                render = self.json(self.path / "official_replay/render_status.json") or {}
                if (render.get("status") != "complete" or render.get("sample_attempt_id") != attempt
                        or render.get("token") != state.get("source_token")):
                    raise FileNotFoundError("Official video belongs to a different sample attempt")

    def _source_signature(self):
        result = []
        for name in ("planning_evidence.json", "trajectory.json",
                     "official_replay/trace.json.gz", "official_replay/trace.json",
                     "sample_state.json", "sample_complete.json"):
            path = self.path / name
            stat = _stat(path) if _inside(path, self.path) else None
            result.append((stat.st_size, stat.st_mtime_ns) if stat else None)
        return tuple(result)

    def listing(self):
        """Cache small, validated listing facts, without retaining every replay.

        A growing/changed sample is checked with the same refresh validation as
        a selected sample. Unchanged samples need no JSON/gzip decoding on the
        next directory refresh. Run metadata and CSV stage fallback stay live.
        """
        if self._listing_state is None or self._source_signature() != self._listing_signature:
            self.refresh()
            self.release_payload()
        return self.describe()

    def release_payload(self):
        """Drop large replay/model objects; keep only the listing facts."""
        self.identity, self.trace, self.plan, self.submitted, self.planning = {}, None, {}, {}, {}
        self.timeline = []
        self.json_cache.clear()

    def refresh(self):
        if not self.current_attempt_complete(self.path):
            self.release_payload()
            raise KeyError("Current NAVSIM sample attempt is incomplete or failed; old replay is excluded")
        signature = self._source_signature()
        self.warnings = []
        self.packet_path = self._file("input_packet.json.gz", "input_packet.json")
        self.planning = self.json(self.path / "planning_evidence.json") or {}
        self.submitted = self.json(self.path / "trajectory.json") or {}
        self.identity = self.planning.get("identity") or {}
        self.token = str(self.identity.get("token") or self.submitted.get("frame_id") or self.path.name)
        sample_state = self.json(self.path / "sample_state.json") or {}
        if sample_state and sample_state.get("source_token") != self.token:
            self.release_payload()
            raise KeyError("Current NAVSIM sample attempt has a different source token")
        self.plan = self.planning.get("plan") or {}
        if self.plan and str(self.plan.get("frame_id")) != self.token:
            self.warnings.append("Source plan token differs from anchor; plan excluded")
            self.plan = {}
        if self.submitted and str(self.submitted.get("frame_id")) != self.token:
            self.warnings.append("Submitted trajectory token differs from anchor; trajectory excluded")
            self.submitted = {}
        self.trace_path = self._file("official_replay/trace.json.gz", "official_replay/trace.json")
        trace = self.json(self.trace_path) if self.trace_path else None
        attempt = sample_state.get("attempt_id")
        self.trace = None
        if isinstance(trace, dict):
            states = (trace.get("ego") or {}).get("states")
            coords = (trace.get("provenance") or {}).get("coordinates", "")
            if (str(trace.get("token")) == self.token and _rows(states, 3)
                    and _finite(trace.get("dt_s")) is not None and float(trace["dt_s"]) > 0
                    and "global_map_xy" in coords and "rear axle" in coords
                    and (not attempt or (trace.get("provenance") or {}).get("sample_attempt_id") == attempt)):
                self.trace = trace
            else:
                self.warnings.append("Official trace source, attempt, finite states, time grid or coordinates invalid; replay excluded")
        elif self.trace_path:
            self.warnings.append("Official trace is incomplete or unreadable; showing the source anchor only")
        count = len(self.trace["ego"]["states"]) if self.trace else 1
        step = float(self.trace["dt_s"]) if self.trace else 0
        self.timeline = [_Record(i, 0, self.token if i == 0 else f"{self.token}:replay:{i:03d}", round(i * step, 9))
                         for i in range(count)]
        primary_stage = (None, None)
        sources = [("official_trace.stage", (self.trace or {}).get("stage"))]
        sources.extend(("identity." + key, self.identity.get(key)) for key in ("stage", "frame_type"))
        for source, value in sources:
            if stage := _stage(value):
                primary_stage = stage, source
                break
        self._listing_state = dict(frame_count=count, duration=self.timeline[-1].time,
                                   execution_mode=self.submitted.get("execution_mode"),
                                   primary_stage=primary_stage)
        self._listing_signature = signature

    def describe(self):
        benchmark = self.run.json(self.run.path / "benchmark_result.json") if self.run else None
        stage, stage_source = self.stage()
        run = self.run.metadata() if self.run else {}
        listing = self._listing_state
        return dict(id=self.id, name=self.token, path=self.path.relative_to(self.authorized_root).as_posix(),
                    scenario="NAVSIM · " + self.token, platform="navsim", dataset="navsim", live=False,
                    frame_count=listing["frame_count"], duration=listing["duration"],
                    execution_mode=listing["execution_mode"],
                    run_id=run.get("id"), run_name=run.get("name"), condition_label=run.get("condition_label"),
                    stage=stage, stage_source=stage_source,
                    summary=dict(scenario="navsim", execution_mode=listing["execution_mode"],
                                 independent_anchor=True, benchmark=_compact(benchmark)))

    def stage(self):
        # The scorer explicitly records original/synthetic. Missing future GT
        # or a suggestive directory name is not evidence of a stage.
        primary = (self._listing_state or {}).get("primary_stage", (None, None))
        if primary[0]:
            return primary
        row = self.run.csv_record(self.token) if self.run else None
        if row:
            first = any(isinstance(key, str) and key.endswith("_stage_one") and _finite(value) is not None for key, value in row.items())
            second = any(isinstance(key, str) and key.endswith("_stage_two") and _finite(value) is not None for key, value in row.items())
            if first != second:
                return ("stage_one" if first else "stage_two"), "official_csv_stage_columns"
        return "unknown", None

    def _packet(self):
        signature = _stat(self.packet_path) if self.packet_path else None
        key = (str(self.packet_path), signature.st_size, signature.st_mtime_ns) if signature else None
        packet = _PACKETS.get(key) if key else None
        if packet is None and self.packet_path:
            packet = self.json(self.packet_path)
            self.json_cache.pop(str(self.packet_path), None)
            if isinstance(packet, dict):
                _PACKETS[key] = packet
                while len(_PACKETS) > 4:
                    _PACKETS.popitem(last=False)
        elif key:
            _PACKETS.move_to_end(key)
        if not isinstance(packet, dict) or str((packet.get("identity") or {}).get("token")) != self.token:
            return {}, ["Source input packet is unavailable or its token differs from anchor; source geometry/images excluded"]
        if not str(packet.get("protocol", "")).startswith("ep-driver-navsim-"):
            return {}, ["Unrecognized NAVSIM input protocol; source geometry/images excluded"]
        return packet, []

    def _cameras(self, store, packet):
        result = []
        for camera in packet.get("camera_images") or []:
            if (not isinstance(camera, dict) or str(camera.get("frame_id")) != self.token
                    or _finite(camera.get("time")) != 0):
                continue
            relative = store._relative(self, camera.get("path"))
            if relative:
                result.append(dict(view=camera.get("view", "camera"), path=relative,
                    frame_id=self.token, time=0., source_time=0., aligned=True, calibrated=False,
                    projection_available=False, source=camera.get("source"),
                    projection_note="Source image only; camera calibration is not archived"))
        return result

    def _video(self, store):
        render = self.json(self.path / "official_replay/render_status.json") or {}
        attempt = (self.json(self.path / "sample_state.json") or {}).get("attempt_id")
        same_attempt = not attempt or render.get("sample_attempt_id") == attempt
        path = store._relative(self, "official_replay/official_bev.mp4")
        available = bool(path and self.trace and str(render.get("token")) == self.token
                         and render.get("status") == "complete" and same_attempt)
        official = bool(str(render.get("token")) == self.token and render.get("status") == "complete"
                        and render.get("renderer_version") == OFFICIAL_RENDERER_VERSION
                        and render.get("official_visualization") is True and same_attempt and self.trace)
        preview = store._relative(self, "official_replay/official_bev_first.png") if (
            str(render.get("token")) == self.token and same_attempt and self.trace) else None
        video_stat = _stat(self.path / path) if path else None
        legacy = not render.get("renderer_version") or str(render.get("renderer_version")).startswith("legacy-custom-")
        label = "NAVSIM SDK官方绘图 · 评分执行视频" if official else (
            "旧版自定义BEV视频（非官方绘图）" if legacy else "未认证绘图版本视频（不作为官方绘图）")
        return dict(path=path if available else None, available=available, status=render.get("status"),
            sync_available=available, simulation_start_time=0., media_start_time=0.,
            source="saved_scoring_execution_video", physical_horizon_s=self.timeline[-1].time,
            preview_path=preview, renderer_version=render.get("renderer_version"),
            revision=f"{video_stat.st_mtime_ns}-{video_stat.st_size}" if video_stat else None,
            expected_renderer_version=OFFICIAL_RENDERER_VERSION, official_visualization=official,
            visualization_label=label,
            visualization_note=None if official else "缺少已认证的官方renderer标记，不能因文件名或评分数据来自官方而视作官方绘图。",
            encoded_duration_s=render.get("encoded_duration_s"), note=render.get("note"))

    def manifest(self, store):
        packet, warnings = self._packet()
        cameras = self._cameras(store, packet)
        frames = [dict(index=i, frame_id=r.frame_id, time=r.time, pending=False,
                       camera_available=i == 0 and bool(cameras), captured=i == 0,
                       source_frame=i == 0, scoring_only=i > 0) for i, r in enumerate(self.timeline)]
        prompt = self._file("model_prompt.txt")
        return dict(**self.describe(), configuration={"platform": "navsim"}, frames=frames,
            times=[r.time for r in self.timeline], source_frames=[0] if cameras else [],
            model_event_indices=[0] if prompt or self.planning.get("model_raw") is not None else [],
            video=self._video(store), warnings=self.warnings + warnings,
            alignment_policy="One independent anchor; source images only at t=0; official replay never becomes planner input",
            navsim=dict(independent_anchor=True, anchor_token=self.token, identity=self.identity,
                        trace_available=self.trace is not None, source_frame_index=0,
                        run_id=self.run.id if self.run else None, stage=self.stage()[0]))

    def _scores(self):
        benchmark = self.run.json(self.run.path / "benchmark_result.json") if self.run else {}
        benchmark = benchmark or {}
        summary = self.run.details() if self.run else {}
        ready = summary.get("scores_status") == "complete"
        csv_row = self.run.csv_record(self.token) if self.run and ready else None
        final = ((self.trace or {}).get("metrics") or {}).get("official_final")
        return dict(internal_quality=_compact(self.planning.get("internal_quality")),
            internal_quality_is_official_metric=False,
            official_anchor=_compact(final), official_csv_anchor=csv_row,
            official_two_stage_aggregate=dict(score=(summary.get("scores") or {}).get("combined") if ready else None,
                status=summary.get("scores_status", "pending"),
                evaluated_tokens=benchmark.get("evaluated_tokens"), failed_tokens=benchmark.get("failed_tokens"),
                split=benchmark.get("split"), standard_sensor_only_comparable=benchmark.get("standard_sensor_only_comparable")),
            semantics="Internal quality, pre-aggregation official anchor metrics, CSV anchor row and official two-stage run aggregate are distinct; none are instantaneous replay metrics")

    def _model(self, store):
        prompt_path = self._file("model_prompt.txt")
        prompt = None
        if prompt_path:
            try:
                with open(_disk(prompt_path), encoding="utf-8-sig") as stream:
                    prompt = stream.read()
            except (OSError, UnicodeError):
                pass
        recorded = self.planning.get("model_input")
        model_input = _compact(recorded) if isinstance(recorded, dict) else None
        if model_input is not None:
            values = recorded.get("image_paths", recorded.get("images", []))
            model_input.update(frame_id=self.token, time=0.,
                               image_manifest_available=any(k in recorded for k in ("image_paths", "images")) and isinstance(values, list),
                               image_paths=[path for value in values if (path := store._relative(self, value))]
                               if isinstance(values, list) else [])
        if prompt is not None:
            model_input = {**(model_input or {}), "frame_id": self.token, "time": 0., "prompt": prompt,
                           "image_paths": (model_input or {}).get("image_paths", []),
                           "image_manifest_available": (model_input or {}).get("image_manifest_available", False)}
            # Older outputs archive the prompt, but not the actual image-input
            # list. Do not infer that every exported camera was sent to the VLM.
        raw, answer, error = (self.planning.get(k) for k in ("model_raw", "model_answer", "model_error"))
        output = dict(frame_id=self.token, time=0., raw=raw, answer=_compact(answer), errors=[error] if error else []) if (
            raw is not None or answer is not None or error is not None) else None
        return model_input, output

    def frame(self, store, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.timeline):
            raise IndexError("NAVSIM replay frame index is out of range")
        packet, warnings = self._packet()
        record = self.timeline[index]
        trace = self.trace or {}
        origin = (packet.get("diagnostics") or {}).get("source_rear_axle_global_pose")
        origin_source = "input_packet.diagnostics.source_rear_axle_global_pose"
        if not isinstance(origin, list) or len(origin) < 3 or any(_finite(v) is None for v in origin[:3]):
            origin = (trace.get("ego") or {}).get("states", [[0., 0., 0.]])[0][:3]
            origin_source = "official_trace_initial_rear_axle (source anchor pose unavailable)"
        trace_poses = [_pose(row, origin) for row in (trace.get("ego") or {}).get("states", [])]
        source_states = self.plan.get("states") or []
        source_path = [row[:3] for row in source_states] if _rows(source_states) else []
        submitted_path = self.submitted.get("poses") or []
        submitted_path = submitted_path if _rows(submitted_path) else []
        ego = _compact(packet.get("ego"))
        actors = [_compact(row) for row in packet.get("actors", [])]
        rear_offset = _finite((trace.get("ego") or {}).get("rear_axle_to_center_m"))
        if rear_offset is None:
            rear_offset = _finite((packet.get("vehicle") or {}).get("rear_axle_to_center_m"))
        display_states = []
        if trace_poses and rear_offset is not None:
            display_states = [[p[0] + rear_offset * math.cos(p[2]), p[1] + rear_offset * math.sin(p[2]), p[2],
                               math.hypot(*row[3:5]) if len(row) >= 5 else 0.]
                              for p, row in zip(trace_poses, trace["ego"]["states"])]
            row = trace["ego"]["states"][index]
            speed = math.hypot(*row[3:5]) if len(row) >= 5 else None
            ego = dict(id="ego", kind="vehicle", position=display_states[index][:2], heading=display_states[index][2],
                       speed=speed, length=trace["ego"].get("length_m"), width=trace["ego"].get("width_m"),
                       rear_axle_to_center_m=rear_offset)
            actor_frames = trace.get("actors") or []
            if len(actor_frames) == len(trace_poses):
                actors = []
                for actor in actor_frames[index]:
                    center = actor.get("center")
                    if not _rows([center]):
                        continue
                    pose = _pose(center, origin)
                    kind = actor.get("type", "unknown")
                    kind = {"traffic_cone": "static_obstacle", "barrier": "static_obstacle"}.get(kind, kind)
                    actors.append(dict(id=str(actor.get("id")), kind=kind, position=pose[:2], heading=pose[2],
                                       length=actor.get("length_m"), width=actor.get("width_m")))
            else:
                actors = []
                warnings.append("Official actor time grid is unavailable; source actors were not reused as future states")
        elif index > 0:
            ego, actors = None, []
            warnings.append("Rear-axle-to-centre offset is missing; future ego body geometry excluded")
        if index == 0:
            plan = copy.deepcopy(self.plan) or None
        else:
            plan = dict(frame_id=record.frame_id, time=record.time, states=display_states,
                        diagnostics={"semantic_enabled": False}, coordinates="source_anchor_local_xy_m;geometric_center",
                        source="official_simulated_execution_not_replanning")
        map_data = {k: packet.get(k) for k in ("lanes", "route", "route_lane_ids")}
        scene = dict(**map_data, frame_id=record.frame_id, time=record.time, ego=ego, actors=actors,
                     drivable=_drivable(packet), rules=packet.get("rules") if index == 0 else None,
                     diagnostics=packet.get("diagnostics", {}) if index == 0 else {},
                     coordinates="fixed_source_rear_axle_local_xy_m;x_forward_y_left")
        assets = {}
        if index == 0:
            for key, name in (("bev", "trajectory_bev.png"), ("official_replay_preview", "official_replay/official_bev_first.png")):
                relative = store._relative(self, name)
                if relative:
                    assets[key] = relative
        model_input, model_output = self._model(store)
        cameras = self._cameras(store, packet) if index == 0 else []
        navsim = dict(independent_anchor=True, anchor_token=self.token, identity=self.identity,
            source_time_us=self.identity.get("timestamp_us"), source_frame=index == 0,
            scoring_only=index > 0, trace_available=self.trace is not None,
            coordinates=dict(display="source_rear_axle_local_xy_m;x_forward_y_left", origin_global=origin,
                             origin_source=origin_source, ego_body_reference="geometric_center",
                             rear_axle_to_center_m=rear_offset),
            trajectories=dict(source_plan_center=source_path, submitted_rear_axle=submitted_path,
                              official_simulated_rear_axle=trace_poses),
            sampling=dict(source_plan_step_s=(self.planning.get("parameters", {}).get("mpc") or {}).get("step_s"),
                          submitted=self.submitted.get("sampling"), official_step_s=trace.get("dt_s")),
            scores=self._scores(), model_error=self.planning.get("model_error"),
            horizon_contract=_compact(self.planning.get("horizon_contract")),
            source_plan_metadata=_compact(self.plan), input_condition=self.planning.get("input_condition"),
            provenance=_compact(trace.get("provenance")), events=_compact(trace.get("events", [])))
        return dict(episode_id=self.id, dataset="navsim", platform="navsim", index=index,
            frame_id=record.frame_id, time=record.time, source_time=0., pending=False,
            ego=ego, actors=actors, plan=plan, plan_record=None, scene=scene, map=map_data, rules=scene["rules"],
            cameras=cameras, assets=assets, raw=None, camera_state=None,
            model_input=model_input if index == 0 else None, model_output=model_output if index == 0 else None,
            active_model_input=model_input if index > 0 else None, active_model_output=model_output if index > 0 else None,
            model_source_frame_id=self.token, retrieval=_compact(self.planning.get("retrieval")), active_retrieval=None,
            quality=None, timing=self.planning.get("timing_s"), execution_audit=None, predictions=None,
            configuration={"platform": "navsim"}, navsim=navsim,
            alignment=dict(policy="independent_navsim_anchor", camera="source_anchor_only_no_projection",
                           scene="official_scoring_replay" if trace else "source_packet"),
            evidence=dict(input_packet=self.packet_path.name if self.packet_path else None,
                planning_evidence="planning_evidence.json", submitted_trajectory="trajectory.json",
                official_trace=self.trace_path.relative_to(self.path).as_posix() if self.trace_path else None),
            warnings=self.warnings + warnings)
