"""只读关联 Fail2Drive 官方成绩、真实录像与官方示例视角。

关联只使用授权目录内的归档，不导入 CARLA、模型、规划器或评分器。
官方最终成绩与逐帧内部审核分别保存，不以 Finished 冒充路线完成。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

from .episodes import _disk, _finite, _inside, _path, _same_time, _stat


SCHEMA = "ep-driver-fail2drive-runtime-v1"
PREFIX = "@fail2drive/display/"


def is_configuration(configuration):
    return (isinstance(configuration, dict) and isinstance(configuration.get("runtime"), dict)
            and configuration["runtime"].get("schema") == SCHEMA)


def planning_child(path):
    """避免上层通用 configuration/summary 将已知 planning 子目录遮住。"""
    candidate = path / "planning" / "configuration.json"
    try:
        if not _inside(candidate, path) or _stat(candidate) is None:
            return False
        with open(_disk(candidate), encoding="utf-8-sig") as stream:
            return is_configuration(json.load(stream))
    except (OSError, ValueError, UnicodeError):
        return False


class Fail2DriveArtifacts:
    def __init__(self, episode, authorized_root):
        self.episode, self.root = episode, Path(authorized_root).resolve()
        self.route = episode.path.parent
        self.assets, self.display_cache, self.execution_cache = {}, None, None

    def _read(self, path):
        if not _inside(path, self.root):
            return None
        info = _stat(path)
        if info is None or info.st_size > 32 * 1024 * 1024:
            return None
        try:
            with open(_disk(path), encoding="utf-8-sig") as stream:
                return json.load(stream)
        except (OSError, ValueError, UnicodeError):
            return None

    def _context(self):
        configuration = self.episode.json(self.episode.path / "configuration.json") or {}
        configuration = configuration if isinstance(configuration, dict) else {}
        runtime = configuration.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        association = runtime.get("archive_association")
        association = association if isinstance(association, dict) else {}
        return configuration, runtime, association

    def _layout(self, association):
        route_id, seed = association.get("official_route_id"), association.get("repetition_seed")
        if type(route_id) is not int or route_id < 0 or type(seed) is not int or seed < 0:
            return None
        parents = self.episode.path.parents
        if len(parents) < 6 or self.episode.path.name != "planning":
            return None
        if (parents[1].name != "agent" or parents[2].name != f"{route_id:04d}"
                or parents[3].name != "agent_runs" or parents[4].name != str(seed)):
            return None
        return parents[4], parents[5]

    def _relocate(self, stored, association, method_root):
        """只按归档方法根的明确相对关系重定位，绝不访问旧机器绝对路径。"""
        if not isinstance(stored, str) or not isinstance(association.get("method_results_root"), str):
            return None
        path = PurePosixPath(stored.replace("\\", "/"))
        original = PurePosixPath(association["method_results_root"].replace("\\", "/"))
        try:
            relative = path.relative_to(original)
        except ValueError:
            return None
        if ".." in relative.parts or not relative.parts:
            return None
        result = method_root.joinpath(*relative.parts)
        return result if _inside(result, self.root) else None

    def describe(self, result):
        _, runtime, association = self._context()
        route_id, seed = association.get("official_route_id"), association.get("repetition_seed")
        filename = PurePosixPath(str(association.get("official_routes_path", "")).replace("\\", "/")).stem
        return dict(result, dataset="fail2drive", platform="fail2drive",
            name=f"{filename or route_id} · seed {seed} · {self.route.name}",
            scenario=f"Fail2Drive · {filename or route_id}",
            execution_mode=runtime.get("execution_mode", result.get("execution_mode")))

    def official(self):
        _, _, association = self._context()
        unavailable = dict(available=False, official_result_valid=False, route_completed=False,
                           reason="官方成绩尚未产生，或授权目录未包含可核验的关联证据")
        layout = self._layout(association)
        if layout is None:
            return dict(unavailable, reason="归档路线/种子身份与目录结构不一致")
        seed_root, method_root = layout
        route_id = association["official_route_id"]
        expected = seed_root / "res" / f"{route_id:04d}_res.json"
        checkpoint = self._relocate(association.get("official_checkpoint"), association, method_root)
        if checkpoint is None or checkpoint.resolve() != expected.resolve():
            return dict(unavailable, reason="官方检查点不在授权根内，或与归档路线/种子不一致")
        parent = self._read(self.route / "configuration.json")
        if not isinstance(parent, dict) or parent.get("archive_association") != association:
            return dict(unavailable, reason="Agent 与数值规划归档的成绩关联不一致或未获授权")
        end, start = _stat(checkpoint), _stat(self.route / "configuration.json")
        if end is None:
            return dict(unavailable, reason="官方成绩尚未产生")
        if start is None or end.st_mtime_ns < start.st_mtime_ns:
            return dict(unavailable, reason="检查点早于本次 Agent 启动，拒绝借用旧成绩")
        # 相同输出中可能保存多次 resume 的 Agent；只有检查点更新前最近一轮可关联。
        candidates = []
        try:
            siblings = list(os.scandir(_disk(self.route.parent)))
        except OSError:
            return dict(unavailable, reason="未能核验同路线的其他 Agent 运行")
        for item in siblings:
            sibling = _path(item.path)
            path = sibling / "configuration.json"
            if not _inside(path, self.root):
                continue
            info = _stat(path)
            if info and info.st_mtime_ns <= end.st_mtime_ns:
                value = self._read(path)
                if isinstance(value, dict) and value.get("archive_association") == association:
                    candidates.append((info.st_mtime_ns, sibling.resolve()))
        latest = max((row[0] for row in candidates), default=None)
        latest_runs = [row[1] for row in candidates if row[0] == latest]
        if latest_runs != [self.route.resolve()]:
            return dict(unavailable, reason="检查点对应其他 Agent 运行，不能将另一轮成绩套到当前回放")
        document = self._read(checkpoint)
        if not isinstance(document, dict):
            return dict(unavailable, reason="官方检查点尚未写完或不可读取")
        checkpoint_data = document.get("_checkpoint")
        if not isinstance(checkpoint_data, dict):
            return dict(unavailable, reason="官方检查点结构尚未完整")
        records = checkpoint_data.get("records")
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
            return dict(unavailable, reason="官方单路线记录尚未齐全")
        record = records[0]
        if record.get("route_id") != f"RouteScenario_{route_id}_rep0":
            return dict(unavailable, reason="官方记录路线身份不匹配")
        scores = record.get("scores")
        if not isinstance(scores, dict):
            return dict(unavailable, reason="官方记录评分结构不合法")
        score_values = [scores.get(k) for k in ("score_route", "score_composed", "score_penalty")]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in score_values):
            return dict(unavailable, reason="官方记录缺少有限数值评分")
        global_record = checkpoint_data.get("global_record")
        global_record = global_record if isinstance(global_record, dict) else {}
        valid = (document.get("entry_status") == "Finished" and document.get("eligible") is True
                 and checkpoint_data.get("progress") == [1, 1]
                 and global_record.get("status") in ("Perfect", "Completed", "Failed"))
        completed = (valid and record.get("status") in ("Perfect", "Completed")
                     and scores["score_route"] >= 100.
                     and global_record.get("status") in ("Perfect", "Completed"))
        return dict(available=True, official_result_valid=valid, route_completed=completed,
            entry_status=document.get("entry_status"), eligible=document.get("eligible"),
            official_status=record.get("status"), global_status=global_record.get("status"),
            scores=copy.deepcopy(scores), infractions=copy.deepcopy(record.get("infractions") or {}),
            meta=copy.deepcopy(record.get("meta") or {}), progress=checkpoint_data.get("progress"),
            reason=None if valid else "官方结果尚未完成最终有效性检查",
            source=checkpoint.relative_to(self.root).as_posix(),
            scope="本路线最终官方结果；不随回放时间变化；不替代整套基准成绩")

    def _execution(self):
        path = self.route / "execution.jsonl"
        info = _stat(path) if _inside(path, self.root) else None
        signature = (info.st_size, info.st_mtime_ns) if info else None
        if self.execution_cache and self.execution_cache[0] == signature:
            return self.execution_cache[1]
        rows = []
        if info and info.st_size <= 64 * 1024 * 1024:
            try:
                with open(_disk(path), encoding="utf-8-sig") as stream:
                    for line in stream:
                        if not line.endswith("\n"):
                            break
                        row = json.loads(line)
                        if (not isinstance(row, dict) or type(row.get("frame")) is not int
                                or type(row.get("time")) not in (int, float) or _finite(row.get("time")) is None):
                            raise ValueError("invalid physical identity")
                        rows.append((row["frame"], float(row["time"])))
            except (OSError, ValueError, UnicodeError):
                rows = []
        self.execution_cache = signature, rows
        return rows

    def video(self, original):
        official_view = original.get("video_view") == "official_display" and original.get("camera_id") == "rgb"
        result = dict(original,
                      visualization_label="官方示例相机真实录像" if official_view else "CARLA六相机真实录像（旧版缩小拼图）",
                      video_view="official_display" if official_view else "legacy_camera_mosaic",
                      official_visualization=False, sync_available=False)
        rows = self._execution()
        count, fps = original.get("frames"), original.get("fps")
        if (not result.get("available") or original.get("status") != "complete"
                or original.get("playback_verified") is not True or original.get("interpolated_frames") != 0
                or type(count) is not int or count < 1 or original.get("decoded_frames") != count
                or type(fps) not in (int, float) or _finite(fps) is None or float(fps) <= 0 or len(rows) != count):
            result["sync_note"] = "缺少完整、已核验的逐物理帧时间证据，录像独立播放"
            return result
        first, last = original.get("first_source_frame"), original.get("last_source_frame")
        if (type(first) is not int or type(last) is not int or last - first + 1 != count
                or rows[0][0] != first or rows[-1][0] != last
                or any(frame != first + i or (i and not _same_time(time - rows[i-1][1], 1 / float(fps)))
                       for i, (frame, time) in enumerate(rows))):
            result["sync_note"] = "录像帧身份/连续性/物理时间不一致，拒绝强行同步"
            return result
        result.update(sync_available=True, simulation_start_time=rows[0][1], media_start_time=0.,
                      sync_note="已核对连续物理帧、逐帧仿真时间及解码帧数；视频0秒绑定首物理帧",
                      sync_source="execution.jsonl + video_recording.json")
        return result

    def display(self):
        empty = dict(available=False, frames=[], label="官方示例视角",
            note="采用官方示例相机设置的 CARLA 原始画面；本项目只读查看器并非官方独立查看器")
        def unavailable(reason):
            self.assets = {}
            self.display_cache = None
            return dict(empty, reason=reason)
        _, _, association = self._context()
        layout = self._layout(association)
        declaration = self._read(self.route / "official_visualization.json")
        if layout is None or not isinstance(declaration, dict) or declaration.get("source") != "official_CARLA_rendered_camera":
            return unavailable("授权范围内缺少官方示例相机归档声明")
        _, method_root = layout
        folder = self._relocate(declaration.get("path"), association, method_root)
        if folder is None or folder.name != self.route.name:
            return unavailable("展示相机目录越出授权根或运行身份不一致")
        info = _stat(folder)
        if info is None:
            return unavailable("展示画面尚未归档")
        execution = dict(self._execution())
        signature = (str(folder), info.st_mtime_ns, self.execution_cache[0])
        if self.display_cache and self.display_cache[0] == signature:
            return self.display_cache[1]
        assets, frames = {}, []
        try:
            paths = sorted(_path(item.path) for item in os.scandir(_disk(folder)) if item.name.endswith(".json"))
        except OSError:
            return unavailable("展示画面目录尚不可读取")
        for path in paths:
            metadata = self._read(path)
            if not isinstance(metadata, dict):
                continue
            frame, time = metadata.get("carla_frame"), metadata.get("carla_timestamp")
            jpeg = metadata.get("jpeg")
            if (metadata.get("schema") != "ep-driver-f2d-display-frame-v1"
                    or metadata.get("source") != "official_CARLA_rendered_camera"
                    or metadata.get("route_run") != self.route.name or type(frame) is not int
                    or type(time) not in (int, float) or _finite(time) is None
                    or frame not in execution or not _same_time(time, execution[frame])
                    or not isinstance(jpeg, str) or Path(jpeg).name != jpeg
                    or jpeg != path.with_suffix(".jpg").name):
                continue
            image = folder / jpeg
            try:
                if not _inside(image, self.root):
                    continue
                with open(_disk(image), "rb") as stream:
                    digest = hashlib.sha256(stream.read()).hexdigest()
                if digest != metadata.get("jpeg_sha256"):
                    continue
            except OSError:
                continue
            virtual = PREFIX + str(frame) + ".jpg"
            assets[virtual] = image, metadata["jpeg_sha256"]
            frames.append(dict(carla_frame=frame, time=float(time), relative_tick=metadata.get("relative_tick"),
                               path=virtual, sha256=metadata["jpeg_sha256"]))
        frames.sort(key=lambda row: (row["time"], row["carla_frame"]))
        self.assets = assets
        value = dict(empty, available=bool(frames), frames=frames,
                     reason=None if frames else "展示图像与其身份/哈希清单尚未齐全")
        self.display_cache = signature, value
        return value

    def resolve_asset(self, relative):
        self.display()
        entry = self.assets.get(relative)
        if entry is None:
            raise FileNotFoundError("未注册的 Fail2Drive 展示图像")
        path, expected = entry
        if not _inside(path, self.root):
            raise ValueError("展示图像越出授权根")
        with open(_disk(path), "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        if digest != expected:
            raise ValueError("展示图像越出授权根或已与归档哈希不符")
        return path

    def metadata(self, timestamp=None):
        configuration, _, association = self._context()
        category = self._read(self.route / "category.json")
        category = category if isinstance(category, dict) else None
        display = copy.deepcopy(self.display())
        if timestamp is not None:
            frames = display.pop("frames", [])
            selected = next((row for row in reversed(frames) if row["time"] <= timestamp + 1e-6), None)
            display["selected"] = (dict(selected, lag_s=max(0., timestamp - selected["time"]),
                                        exact_time=_same_time(timestamp, selected["time"])) if selected else None)
        return dict(official=self.official(), worker_summary=self.episode.json(self.episode.path / "summary.json") or {},
                    association=copy.deepcopy(association), input_category=category or {
                        "privileged_input": configuration.get("privileged_input"),
                        "note": "输入类别只取归档声明；缺少 category.json 时不补造传感器条件"}, display=display)
