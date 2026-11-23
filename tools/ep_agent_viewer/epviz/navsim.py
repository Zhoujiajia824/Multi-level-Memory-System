"""Detached NAVSIM artifacts, with one independent anchor per viewer episode.

No NAVSIM, nuPlan, model, planner, or scoring code is imported. All display
coordinates use the source rear-axle pose as the fixed SE(2) origin. Planner
states and actor boxes are centres; submitted/official ego paths are rear axles.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
import csv
import math
from pathlib import Path

from .episodes import _Episode, _Record, _disk, _finite, _inside, _stat


_PACKETS = OrderedDict()
OFFICIAL_RENDERER_VERSION = "navsim-sdk-plots-1.0"


def _compact(value):
    """Keep text/metadata, omit embedded map, history and feature tensors."""
    bulky = {"history", "lanes", "route", "predictions", "parameters",
             "embedding", "features", "states", "controls", "drivable_wkt",
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
    def is_sample(path):
        return any(_stat(path / name) is not None for name in ("input_packet.json", "input_packet.json.gz")) and any(
            _stat(path / name) is not None for name in ("trajectory.json", "planning_evidence.json"))

    def __init__(self, path, identifier, authorized_root):
        super().__init__(path, identifier)
        self.authorized_root = authorized_root
        candidate = path.parent.parent
        self.run = _Episode(candidate, "run") if path.parent.name == "samples" and _inside(candidate, authorized_root) else None
        self.identity, self.trace, self.plan, self.submitted, self.planning = {}, None, {}, {}, {}
        self.trace_path, self.packet_path = None, None

    def _file(self, *names):
        return next((self.path / name for name in names if _inside(self.path / name, self.path)
                     and _stat(self.path / name) is not None), None)

    def refresh(self):
        self.warnings = []
        self.packet_path = self._file("input_packet.json.gz", "input_packet.json")
        self.planning = self.json(self.path / "planning_evidence.json") or {}
        self.submitted = self.json(self.path / "trajectory.json") or {}
        self.identity = self.planning.get("identity") or {}
        self.token = str(self.identity.get("token") or self.submitted.get("frame_id") or self.path.name)
        self.plan = self.planning.get("plan") or {}
        if self.plan and str(self.plan.get("frame_id")) != self.token:
            self.warnings.append("Source plan token differs from anchor; plan excluded")
            self.plan = {}
        if self.submitted and str(self.submitted.get("frame_id")) != self.token:
            self.warnings.append("Submitted trajectory token differs from anchor; trajectory excluded")
            self.submitted = {}
        self.trace_path = self._file("official_replay/trace.json.gz", "official_replay/trace.json")
        trace = self.json(self.trace_path) if self.trace_path else None
        self.trace = None
        if isinstance(trace, dict):
            states = (trace.get("ego") or {}).get("states")
            coords = (trace.get("provenance") or {}).get("coordinates", "")
            if (str(trace.get("token")) == self.token and _rows(states, 3)
                    and _finite(trace.get("dt_s")) is not None and float(trace["dt_s"]) > 0
                    and "global_map_xy" in coords and "rear axle" in coords):
                self.trace = trace
            else:
                self.warnings.append("Official trace token, finite states, time grid or coordinate provenance invalid; replay excluded")
        elif self.trace_path:
            self.warnings.append("Official trace is incomplete or unreadable; showing the source anchor only")
        count = len(self.trace["ego"]["states"]) if self.trace else 1
        step = float(self.trace["dt_s"]) if self.trace else 0
        self.timeline = [_Record(i, 0, self.token if i == 0 else f"{self.token}:replay:{i:03d}", round(i * step, 9))
                         for i in range(count)]

    def describe(self):
        benchmark = self.run.json(self.run.path / "benchmark_result.json") if self.run else None
        return dict(id=self.id, name=self.token, path=self.path.relative_to(self.authorized_root).as_posix(),
                    scenario="NAVSIM · " + self.token, platform="navsim", dataset="navsim", live=False,
                    frame_count=len(self.timeline), duration=self.timeline[-1].time,
                    execution_mode=self.submitted.get("execution_mode"),
                    summary=dict(scenario="navsim", execution_mode=self.submitted.get("execution_mode"),
                                 independent_anchor=True, benchmark=_compact(benchmark)))

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
        path = store._relative(self, "official_replay/official_bev.mp4")
        available = bool(path and self.trace and str(render.get("token")) == self.token
                         and render.get("status") == "complete")
        official = bool(str(render.get("token")) == self.token and render.get("status") == "complete"
                        and render.get("renderer_version") == OFFICIAL_RENDERER_VERSION
                        and render.get("official_visualization") is True)
        preview = store._relative(self, "official_replay/official_bev_first.png") if str(render.get("token")) == self.token else None
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
                        trace_available=self.trace is not None, source_frame_index=0))

    def _scores(self):
        benchmark = self.run.json(self.run.path / "benchmark_result.json") if self.run else {}
        benchmark = benchmark or {}
        csv_row = None
        if self.run:
            # Only archive-contained files are permitted; never follow a stored
            # absolute official_csv path to a different machine's filesystem.
            relative = str(benchmark.get("official_csv", "")).replace("\\", "/").split("/")[-1]
            path = self.run.path / "official_scores" / relative
            if relative and _inside(path, self.run.path) and _stat(path):
                try:
                    with open(_disk(path), encoding="utf-8-sig", newline="") as stream:
                        csv_row = next((r for r in csv.DictReader(stream) if r.get("token") == self.token), None)
                except (OSError, ValueError, UnicodeError):
                    pass
        final = ((self.trace or {}).get("metrics") or {}).get("official_final")
        return dict(internal_quality=_compact(self.planning.get("internal_quality")),
            internal_quality_is_official_metric=False,
            official_anchor=_compact(final), official_csv_anchor=csv_row,
            official_two_stage_aggregate=dict(score=benchmark.get("official_two_stage_epdms"),
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
                     drivable=packet.get("drivable_wkt"), rules=packet.get("rules") if index == 0 else None,
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
