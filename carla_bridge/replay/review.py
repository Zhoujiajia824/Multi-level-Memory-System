"""Build an offline, frame-aligned review of model use in a CARLA run.

    python -m carla_bridge.replay.review outputs/carla_runs/RUN_DIRECTORY

Only the Python standard library is needed. Large JSONL logs are read one line
at a time; candidate arrays and complete planner trajectories are not embedded.
The generated HTML embeds its data so it also works directly from ``file://``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

TRIGGER_LABELS = {
    "periodic_cognition": "周期认知对照",
    "always_policy": "每周期调用策略",
    "occupied_ego_crossing_conflict": "横穿交通与自车实际占用空间冲突",
    "path_blocked": "当前行驶路径被阻挡",
    "sustained_native_planner_degradation": "数值规划持续异常，需要协助脱困",
    "special_situation": "检测到特殊驾驶情况",
    "verified_memory_failure": "执行记忆提示相似场景曾失败",
    "memory_supported_intervention": "执行记忆支持介入处理",
    "advisory_conditions_changed": "建议依赖的环境条件已变化",
    "urgent_crossing_conflict": "存在紧迫横穿冲突",
    "complex_interaction": "多个横穿对象形成复杂交互",
    "no_finite_pdm_proposal": "数值规划没有有效候选",
    "low_pdm_quality": "数值候选质量较低",
    "no_progressing_proposal": "没有能有效前进的候选",
    "unexplained_stall": "车辆持续停滞，且非正常等待",
}


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _frame(value):
    return int(value) if type(value) is int and value >= 0 else None


def _dict(value):
    return value if isinstance(value, dict) else {}


def _first(*values):
    return next((v for v in values if v is not None), None)


def _clean(value):
    """JSON logs may contain historical NaNs; never emit invalid JS/JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _read_json(path, warnings):
    if not path.is_file():
        return {}
    try:
        return _dict(json.loads(path.read_text(encoding="utf-8-sig")))
    except (ValueError, OSError) as exc:
        warnings.append("无法读取 %s: %s" % (path.name, type(exc).__name__))
        return {}


def _rows(path, warnings):
    if not path.is_file():
        return
    malformed = 0
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("expected object")
                    yield item
                except (ValueError, TypeError):
                    malformed += 1
    except OSError as exc:
        warnings.append("无法完整读取 %s: %s" % (path.name, type(exc).__name__))
    if malformed:
        warnings.append("%s 有 %d 行损坏或未写完，已跳过；统计仅覆盖可读取记录。" % (path.name, malformed))


def _asset(path, root):
    """Resolve only files belonging to this run, including a moved Linux run.

    Never guess a basename from a memory bank: two runs may have the same token.
    Never emit a remote URL or a path outside the review's run directory.
    """
    if not isinstance(path, str) or not path or "\x00" in path:
        return None
    normalized = path.replace("\\", "/")
    if "://" in normalized or normalized.lower().startswith(("data:", "javascript:")):
        return None
    candidates = []
    native = Path(path)
    if native.is_absolute():
        candidates.append(native)
    # Native relative paths and run-relative archive references.
    if not normalized.startswith("/") and not (len(normalized) > 1 and normalized[1] == ":"):
        candidates.append(root / normalized)
    # An archive may have moved from Linux to Windows or vice versa. Preserve
    # the complete relative suffix under the exact original run directory name.
    parts = PurePosixPath(normalized).parts
    for i, part in enumerate(parts):
        if part == root.name and i + 1 < len(parts):
            candidates.append(root.joinpath(*parts[i + 1:]))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            rel = resolved.relative_to(root)
            if resolved.is_file():
                return "../" + quote(rel.as_posix(), safe="/")
        except (ValueError, OSError):
            continue
    return None


def _image_assets(event, root):
    paths = []
    value = event.get("image_paths")
    if isinstance(value, dict):
        paths.extend(value.values())
    elif isinstance(value, list):
        paths.extend(value)
    elif isinstance(value, str):
        paths.append(value)
    if event.get("mosaic_path"):
        paths.insert(0, event["mosaic_path"])
    seen, result = set(), []
    for item in paths:
        if isinstance(item, dict):
            item = _first(item.get("path"), item.get("archive_path"), item.get("image_path"))
        if not isinstance(item, str) or item in seen:
            continue
        seen.add(item)
        result.append({"path": item, "url": _asset(item, root)})
    return result


def _plan_effect(row):
    """Attribute selection changes, excluding ordinary maneuvers and shadow.

    This is candidate-selection attribution, not a claim of improved safety or
    counterfactual physical behavior. Expanded model-requested candidate search
    is accounted separately from bounded preference changes.
    """
    d = _dict(row.get("planner_diagnostics"))
    governor = _dict(d.get("governor"))
    recovery = _dict(d.get("recovery"))
    selected, baseline = d.get("selected_id"), d.get("baseline_selected_id")
    if row.get("semantic_influence") is False or d.get("shadow_only") is True:
        return False
    if row.get("semantic_influence") is not True:
        return None
    if governor:
        if governor.get("adopted") is True:
            return True
        original = d.get("original_baseline_selected_id", baseline)
        if (recovery.get("authorized") is True and recovery.get("expanded") is True
                and selected is not None and original is not None and selected != original):
            return True
        return False if selected is not None else None
    if selected is not None and baseline is not None:
        return selected != baseline
    return None


def _compact_plan(row):
    d = _dict(row.get("planner_diagnostics"))
    ids = {d.get("selected_id"), d.get("baseline_selected_id"), d.get("original_baseline_selected_id")}
    selected_candidates = []
    for candidate in d.get("candidates") or []:
        if isinstance(candidate, dict) and candidate.get("id") in ids:
            selected_candidates.append({k: v for k, v in candidate.items()
                                        if k not in ("trajectory", "states", "path", "rollout")})
    return {"sim_time_s": row.get("sim_time_s"), "sequence": row.get("sequence"),
            "source": row.get("source"), "validation": row.get("validation"),
            "semantic_frame_id": row.get("semantic_frame_id"),
            "semantic_effect": _plan_effect(row),
            "planner_diagnostics": {**{k: d.get(k) for k in (
                "selected_id", "baseline_selected_id", "original_baseline_selected_id",
                "shadow_selected_id", "shadow_only", "semantic_policy", "governor", "recovery",
                "query_decision", "planner_degradation", "ordinary_lane_intent") if k in d},
                "selected_and_baseline_candidates": selected_candidates}}


def _compact_record(record):
    record = _dict(record)
    keep = ("current_scene", "model_decision_output", "raw_response", "decision_validation",
            "parser_status", "parser_errors", "fallback_used", "nav_instruction", "ego_state",
            "perception_objects", "retrieved_memory_ids", "retrieved_memory_scores",
            "retrieved_memory_event_types", "long_term_rule_ids", "memory_mode")
    result = {k: record[k] for k in keep if k in record}
    if "decision_output" in record:
        decision = _dict(record["decision_output"])
        result["decision_output"] = {k: v for k, v in decision.items() if k != "planner_diagnostics"}
        if isinstance(decision.get("planner_diagnostics"), dict):
            result["planning_output"] = _compact_plan({"planner_diagnostics": decision["planner_diagnostics"]})["planner_diagnostics"]
    if "planning_output" not in result and isinstance(record.get("planning_output"), dict):
        result["planning_output"] = _compact_plan({"planner_diagnostics": record["planning_output"]})["planner_diagnostics"]
    return result


def _control(root, warnings):
    ticks = {}
    rows_without_frame = 0
    for row in _rows(root / "control.jsonl", warnings):
        world = _dict(row.get("world_snapshot"))
        fid = _frame(_first(world.get("frame"), row.get("carla_frame_id")))
        if fid is None:
            rows_without_frame += 1
            continue
        measured, command = _dict(row.get("measured")), _dict(row.get("command"))
        ticks[fid] = {"frame": fid, "sim_time_s": _number(row.get("sim_time_s")),
                      "x": _number(measured.get("x")), "y": _number(measured.get("y")),
                      "planner_sequence": command.get("planner_sequence"),
                      "command_source": command.get("source")}
    if rows_without_frame:
        warnings.append("control.jsonl 中 %d 条 tick 缺少真实 CARLA 帧号，不能计入帧比例分母。" % rows_without_frame)
    return ticks, rows_without_frame


def _consultations(root, warnings, audit_enabled=False):
    calls = {}
    source = "instrumented" if (root / "model_calls.jsonl").is_file() or audit_enabled else "legacy_inferred"
    if source == "instrumented":
        for number, row in enumerate(_rows(root / "model_calls.jsonl", warnings)):
            cid = str(row.get("call_id") or "unidentified_%d" % number)
            call = calls.setdefault(cid, {"call_id": cid, "requests": [], "lifecycle_events": [],
                                           "source": source})
            call["lifecycle_events"].append(row.get("event", row.get("type", "unknown")))
            call.update({k: v for k, v in row.items() if v is not None and k not in ("event", "type", "record")})
            if isinstance(row.get("record"), dict):
                call["record"] = _compact_record(row["record"])
            if row.get("event", row.get("type")) in ("consultation_end", "end"):
                call["completed"] = True
        for call in calls.values():
            call.setdefault("completed", False)
            call.setdefault("status", "incomplete" if not call["completed"] else "completed")
    else:
        # Old decision records prove a cognition result, not the number of SDK
        # requests/retries, prompts, or exact request dispatch success.
        for path in sorted(root.glob("decisions_carla_*.jsonl")):
            for row in _rows(path, warnings):
                token = str(_first(row.get("sample_token"), row.get("frame_id"), len(calls)))
                call = {"call_id": "legacy_" + token, "sample_token": token,
                        "sim_time_s": row.get("sim_time_s"), "carla_frame_id": row.get("carla_frame_id"),
                        "ego": {}, "record": _compact_record(row), "requests": [],
                        "image_paths": row.get("vlm_image_paths", []),
                        "trigger": "旧日志推断：触发原因/完整提示词/API重试不可恢复",
                        "completed": True, "status": "legacy_inferred", "source": source}
                calls[call["call_id"]] = call
        if calls:
            warnings.append("旧版日志仅可推断认知记录；实际 SDK 调用次数、失败重试和完整提示词不可恢复。")
        else:
            source = "none"
    return calls, source


def _requests(root, calls, warnings):
    attempts = {}
    for number, row in enumerate(_rows(root / "model_requests.jsonl", warnings)):
        cid = str(row.get("call_id") or "unassigned")
        rid = str(row.get("request_id") or "unidentified_%d" % number)
        attempt = row.get("attempt", 1)
        key = (cid, rid, str(attempt))
        target = attempts.setdefault(key, {"call_id": cid, "request_id": rid, "attempt": attempt,
                                           "lifecycle_events": []})
        event = row.get("event", row.get("type", "unknown"))
        target["lifecycle_events"].append(event)
        target.update({k: v for k, v in row.items() if v is not None and k not in ("event", "type")})
        if event in ("request_start", "start"):
            target["started"] = True
        if event in ("request_end", "end"):
            target["completed"] = True
    for item in attempts.values():
        # New instrumentation marks SDK dispatch explicitly. A historical row
        # with only a response/error is an attempted SDK invocation unless the
        # producer explicitly marked a skipped or preflight-only request.
        item["dispatched"] = (item.get("dispatched") is True or (
            item.get("dispatched") is not False and (item.get("started") is True
            or item.get("status") in ("success", "ok", "error", "failed", "completed"))))
        item.setdefault("status", "incomplete" if not item.get("completed") else "completed")
        if item["call_id"] not in calls:
            calls[item["call_id"]] = {"call_id": item["call_id"], "requests": [], "source": "request_only",
                                       "status": "consultation_record_missing", "completed": False}
        call = calls[item["call_id"]]
        item["carla_frame_id"] = _frame(_first(item.get("carla_frame_id"), call.get("carla_frame_id")))
        item["sim_time_s"] = _number(_first(item.get("sim_time_s"), call.get("sim_time_s")))
        call["requests"].append(item)
        for field in ("carla_frame_id", "sim_time_s"):
            if call.get(field) is None and item.get(field) is not None:
                call[field] = item[field]
    return list(attempts.values())


def _video_data(root, warnings):
    result = []
    for path in sorted((root / "carla_videos").glob("*")):
        if path.suffix.lower() not in (".mp4", ".webm", ".mkv", ".avi", ".mov"):
            continue
        # Accept foo.frames.jsonl and foo.mp4.frames.jsonl; never use metadata
        # fps as an implicit sim-time→video-time conversion.
        mappings = [path.with_suffix(".frames.jsonl"), Path(str(path) + ".frames.jsonl")]
        sidecar = next((p for p in mappings if p.is_file()), mappings[0])
        encoded = {}
        for row in _rows(sidecar, warnings):
            video_time = _number(row.get("video_time_s"))
            fid = _frame(row.get("carla_frame_id"))
            if video_time is None or video_time < 0 or fid is None:
                continue
            encoded[video_time] = [video_time, fid, _number(row.get("sim_time_s"))]
        if not encoded:
            warnings.append("视频 %s 缺少可用逐帧映射，仅可独立播放，不进行伪时间对齐。" % path.name)
        result.append({"name": path.name, "url": _asset(str(path), root),
                       "frames": sorted(encoded.values(), key=lambda v: v[0]),
                       "mapping_source": sidecar.name if encoded else None})
    return result


def _planner(root, calls, ticks, warnings):
    by_token = defaultdict(list)
    for call in calls.values():
        if call.get("sample_token") is not None:
            by_token[str(call["sample_token"])].append(call)
        call["planner"] = {"related_plan_count": 0, "semantic_effect_plan_count": 0,
                           "samples": [], "sampling": "first, changed, and latest (maximum 12 samples)"}
    plans, counts = {}, Counter()
    for row in _rows(root / "planner.jsonl", warnings):
        counts["all"] += 1
        effect = _plan_effect(row)
        accepted = _dict(row.get("validation")).get("accepted")
        if effect is None:
            counts["unknown_effect"] += 1
        elif effect:
            counts["effect"] += 1
        if accepted is True:
            counts["accepted"] += 1
        if effect is True and accepted is True:
            counts["accepted_effect"] += 1
        sequence = row.get("sequence")
        if type(sequence) is int:
            plans[sequence] = {"effect": effect, "accepted": accepted}
        token = row.get("semantic_frame_id")
        for call in by_token.get(str(token), []):
            audit = call["planner"]
            audit["related_plan_count"] += 1
            audit["semantic_effect_plan_count"] += int(effect is True)
            sample = _compact_plan(row)
            samples = audit["samples"]
            if len(samples) < 2 or (effect is True and len(samples) < 11):
                samples.append(sample)
            elif len(samples) < 12:
                samples.append(sample)
            else:
                samples[-1] = sample
    applicable, affected = set(), set()
    for fid, tick in ticks.items():
        sequence = tick.get("planner_sequence")
        plan = plans.get(sequence) if type(sequence) is int else None
        # command precedes measured physics tick. Sequence attribution is exact;
        # do not infer that a plan affected the tick recorded at its own time.
        if (plan is not None and plan["effect"] is not None and plan["accepted"] is True
                and tick.get("command_source") == "tracker"):
            applicable.add(fid)
            if plan["effect"]:
                affected.add(fid)
    return counts, applicable, affected


def _trigger_text(call):
    trigger = call.get("trigger")
    query = _dict(call.get("query"))
    if isinstance(trigger, str):
        return trigger
    if isinstance(trigger, list):
        return ", ".join(str(x) for x in trigger)
    value = _first(_dict(trigger).get("reason"), query.get("reason"), query.get("reasons"))
    return value if isinstance(value, str) else json.dumps(value or trigger or "未记录", ensure_ascii=False)


def build_review(run_dir: str | Path) -> dict[str, Any]:
    """Write ``replay/{index.html,data.json,summary.json}`` and return summary."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError("运行目录不存在: %s" % root)
    warnings = []
    manifest = _read_json(root / "manifest.json", warnings)
    metadata = _read_json(root / "replay_metadata.json", warnings)
    audit_enabled = metadata.get("audit_enabled") is True
    audit_errors = metadata.get("audit_errors") or []
    for error in audit_errors:
        warnings.append("模型审计写入失败，调用统计可能不完整: %s" % error)
    ticks, missing_tick_frames = _control(root, warnings)
    calls, source = _consultations(root, warnings, audit_enabled)
    # An enabled, healthy auditor deliberately creates no request file in a
    # model-free run. Its explicit metadata distinguishes zero from unknown.
    request_log_present = (root / "model_requests.jsonl").is_file() or audit_enabled
    attempts = _requests(root, calls, warnings)
    dispatched = [r for r in attempts if r["dispatched"]]
    unknown_attempt_frames = sum(r.get("carla_frame_id") is None for r in dispatched)
    call_frames = {r["carla_frame_id"] for r in dispatched if r.get("carla_frame_id") is not None}
    outside_frames = call_frames - ticks.keys()
    if outside_frames:
        warnings.append("%d 个实际 SDK 调用帧不在 control tick 帧日志中，调用帧比例不可准确计算。" % len(outside_frames))
    if unknown_attempt_frames:
        warnings.append("%d 次 SDK 调用缺少真实 CARLA 帧号，不能准确计算调用帧比例。" % unknown_attempt_frames)
    if not ticks:
        warnings.append("没有可用 control.jsonl 真实推进帧；帧数分母与比例未知。")
    if not request_log_present:
        warnings.append("没有 model_requests.jsonl；实际 SDK 尝试/重试计数未知，不能把认知记录数当作 API 请求数。")
    plan_counts, applicable, affected = _planner(root, calls, ticks, warnings)
    videos = _video_data(root, warnings)
    ordered_calls = sorted(calls.values(), key=lambda c: (
        _number(c.get("sim_time_s")) if _number(c.get("sim_time_s")) is not None else math.inf,
        _frame(c.get("carla_frame_id")) if _frame(c.get("carla_frame_id")) is not None else math.inf,
        c["call_id"]))
    # Fill measured trigger position only at the very same CARLA frame.
    for call in ordered_calls:
        tick = ticks.get(_frame(call.get("carla_frame_id")), {})
        if not call.get("ego") and tick:
            call["ego"] = {"x": tick.get("x"), "y": tick.get("y")}
            call["ego_source"] = "exact_control_frame"
        call["trigger_text"] = _trigger_text(call)
        label = TRIGGER_LABELS.get(call["trigger_text"])
        call["trigger_label"] = "%s（%s）" % (label, call["trigger_text"]) if label else call["trigger_text"]
        call["images"] = _image_assets(call, root)
        for request in call["requests"]:
            request["images"] = _image_assets(request, root)
        call["actual_sdk_attempt_count"] = (sum(bool(r["dispatched"]) for r in call["requests"])
                                             if request_log_present else None)
        call["has_actual_sdk_dispatch"] = (call["actual_sdk_attempt_count"] > 0
                                            if request_log_present else None)
    trigger_counts = Counter(c["trigger_text"] for c in ordered_calls if c["has_actual_sdk_dispatch"] is True)
    consultation_trigger_counts = Counter(c["trigger_text"] for c in ordered_calls)
    failed = [r for r in dispatched if r.get("status") in ("error", "failed", "failure") or r.get("error")]
    logical = {(r["call_id"], r.get("stage"), r.get("logical_request_id", r["request_id"])) for r in dispatched}
    corrupt_request_audit = any("model_requests.jsonl" in w or "model_calls.jsonl" in w for w in warnings)
    corrupt_control_audit = any("control.jsonl" in w for w in warnings)
    corrupt_planner_audit = any("planner.jsonl" in w for w in warnings)
    valid_ratio = bool(request_log_present and ticks and not unknown_attempt_frames and not audit_errors
                       and not corrupt_request_audit and not corrupt_control_audit
                       and not outside_frames and not missing_tick_frames)
    summary = {
        "schema_version": 1, "run_directory": root.name, "source_mode": source,
        "audit_complete": request_log_present and not audit_errors and not corrupt_request_audit,
        "control_audit_complete": bool(ticks) and not corrupt_control_audit,
        "planner_audit_complete": (root / "planner.jsonl").is_file() and not corrupt_planner_audit,
        "consultation_count": len(calls),
        "logical_model_request_count": len(logical) if request_log_present else None,
        "sdk_attempt_count": len(dispatched) if request_log_present else None,
        "sdk_failed_attempt_count": len(failed) if request_log_present else None,
        "sdk_incomplete_attempt_count": sum(not r.get("completed") for r in dispatched) if request_log_present else None,
        "actual_model_call_frame_count": len(call_frames) if request_log_present and not unknown_attempt_frames else None,
        "actual_model_call_frames_in_tick_log": len(call_frames & ticks.keys()) if request_log_present else None,
        "total_tick_frame_count": len(ticks) if ticks else None,
        "tick_rows_without_frame": missing_tick_frames,
        "model_call_frame_ratio": len(call_frames) / len(ticks) if valid_ratio else None,
        "model_call_frame_ratio_definition": "实际 SDK dispatch 对应的唯一 CARLA 帧 / control.jsonl 中真实推进的唯一 CARLA 帧；同帧 scene、decision、重试只计一帧。不是模型墙钟耗时占比；SDK 内部 HTTP 重试不可观察。",
        "sdk_attempt_definition": "request_id + attempt 去重的已 dispatch SDK 调用尝试；SDK 内部自动 HTTP 重试不在此计数内。",
        "planner_plan_count": plan_counts["all"],
        "semantic_effect_plan_count": plan_counts["effect"],
        "semantic_effect_unknown_plan_count": plan_counts["unknown_effect"],
        "semantic_effect_plan_ratio": (plan_counts["effect"] / plan_counts["all"]
                                       if plan_counts["all"] and not plan_counts["unknown_effect"]
                                       and not corrupt_planner_audit else None),
        "semantic_effect_accepted_plan_count": plan_counts["accepted_effect"],
        "semantic_effect_tick_frame_count": len(affected) if applicable else None,
        "semantic_attributable_tick_frame_count": len(applicable) if applicable else None,
        "semantic_effect_tick_frame_ratio": (len(affected) / len(applicable)
                                             if applicable and not corrupt_control_audit and not corrupt_planner_audit else None),
        "semantic_effect_tick_total_run_ratio": (len(affected) / len(ticks)
            if ticks and len(applicable) == len(ticks) and not corrupt_control_audit and not corrupt_planner_audit else None),
        "semantic_effect_definition": "实际选择相对同池无语义基线发生变化，或授权模型扩展候选后相对原始基线发生变化；shadow 不算生效。tick 通过 command.planner_sequence 精确关联此前已接受的规划，分母是可归因 tracker tick；不宣称物理反事实或驾驶效果。",
        "trigger_counts": dict(trigger_counts),
        "trigger_labels": {key: TRIGGER_LABELS.get(key, key) for key in trigger_counts},
        "consultation_trigger_counts": dict(consultation_trigger_counts),
        "video_count": len(videos), "warnings": warnings,
    }
    track = sorted(ticks.values(), key=lambda t: (t["sim_time_s"] if t["sim_time_s"] is not None else math.inf, t["frame"]))
    # The map is for review, not a geometry export. Stats above use every tick.
    if len(track) > 12000:
        step = math.ceil(len(track) / 12000)
        retain = set(call_frames)
        track = [v for i, v in enumerate(track) if i % step == 0 or v["frame"] in retain or i == len(track) - 1]
    track = [{k: t[k] for k in ("frame", "sim_time_s", "x", "y")} for t in track]
    data = _clean({"schema_version": 1, "summary": summary,
                   "run": {k: _first(manifest.get(k), metadata.get(k)) for k in ("run_id", "scenario_path", "mode", "status", "commit", "semantic_policy", "model_inference")},
                   "calls": ordered_calls, "track": track, "videos": videos,
                   "generated_at": datetime.now(timezone.utc).isoformat()})
    output = root / "replay"
    output.mkdir(parents=True, exist_ok=True)
    # Escape '<' as well as closing script tags: input/output text is untrusted
    # content and must never become executable HTML when opening the review.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    safe = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("review.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("__REVIEW_DATA__", safe), encoding="utf-8")
    (output / "data.json").write_text(payload + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(_clean(summary), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="离线构建 CARLA 模型调用审核与视频同步页面（不加载 CARLA/模型）")
    parser.add_argument("run_dir", help="包含 control.jsonl、planner.jsonl 和模型审计文件的本轮运行目录")
    args = parser.parse_args(argv)
    summary = build_review(args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("回放页面: %s" % (Path(args.run_dir).resolve() / "replay" / "index.html"))


if __name__ == "__main__":
    main()
