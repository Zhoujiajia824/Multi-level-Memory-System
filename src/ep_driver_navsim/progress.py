"""Attempt-bound progress around official scoring, never an alternate scorer.

The monitor reads small lifecycle files, not images, map packets, DINO features,
planning evidence or replay tensors. Unchanged records remain cached. It does
not write sample state, infer benchmark metrics or control worker processes.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time


LOG = logging.getLogger(__name__)
PROGRESS_SCHEMA = "ep-navsim-sample-progress-1"
_STAGES = {
    "representations": "表示构建", "model_preparation": "模型准备",
    "retrieval": "记忆检索",
    "model_request": "模型请求", "model_validation": "回答校验",
    "prediction": "对象预测", "planning": "轨迹规划", "reference": "参照规划",
    "quality": "质量诊断", "complete": "规划完成", "failed": "处理失败",
}


def progress_interval(value=15.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("progress_interval_s must be a finite number")
    if value != 0 and not 5 <= value <= 300:
        raise ValueError("progress_interval_s must be 0 (disabled) or between 5 and 300 seconds")
    return float(value)


def _elapsed(stamp, now):
    try:
        value = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if value.tzinfo is None:
            return None
        return max(0., now-value.timestamp())
    except (TypeError, ValueError, OverflowError):
        return None


def record_sample_progress(root, state, stage, **details):
    """Optional worker heartbeat; source/attempt IDs prevent stale-stage display.

    Call after a stage change and after each model request. A whole model phase
    need not return before the newest request's finish reason becomes visible.
    An optional progress write must not replace a numerical or scoring result.
    """
    try:
        from .protocol import write_json
        if not state or not state.get("attempt_id") or not state.get("source_token"):
            raise ValueError("Progress requires a bound sample attempt")
        path = Path(root) / "sample_progress.json"
        now = datetime.now(timezone.utc).isoformat()
        previous = {}
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        same = (previous.get("attempt_id") == state["attempt_id"] and previous.get("stage") == stage)
        row = dict(schema=PROGRESS_SCHEMA, attempt_id=state["attempt_id"],
                   source_token=state["source_token"], stage=stage, updated_utc=now,
                   stage_started_utc=previous.get("stage_started_utc", now) if same else now)
        allowed = {"model_request_attempt", "model_semantic_attempt", "last_finish_reason",
                   "model_elapsed_s", "model_deadline_remaining_s", "validation_stage"}
        row.update({key: value for key, value in details.items() if key in allowed})
        write_json(path, row)
        return row
    except (OSError, TypeError, ValueError) as exc:
        LOG.warning("NAVSIM progress heartbeat was not saved (%s)", type(exc).__name__)
        return None


class ProgressMonitor:
    def __init__(self, settings, printer=None, clock=None):
        self.root = Path(settings["output_dir"]) / "samples"
        self.video = settings.get("bev_video", True)
        self.model_requested = settings.get("fast", "off") != "off"
        self.expected = {hashlib.sha256(json.dumps(str(token), ensure_ascii=False).encode()).hexdigest()[:24]: str(token)
                         for token in settings.get("evaluation_tokens", [])}
        self._cache = {}
        self._terminal = {}
        self._printer = printer or (lambda message: print(message, flush=True))
        self._clock = clock or time.time

    def _json(self, path):
        try:
            info = path.stat()
            signature = (info.st_mtime_ns, info.st_size)
        except OSError:
            return None, None
        cached = self._cache.get(path)
        if cached is not None and cached[0] == signature:
            return signature, cached[1]
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row = row if isinstance(row, dict) else None
        except (OSError, ValueError):
            row = None
        self._cache[path] = signature, row
        return signature, row

    @staticmethod
    def _bound(row, token, attempt):
        return (isinstance(row, dict) and row.get("status") == "complete"
                and row.get("token") == token and row.get("sample_attempt_id") == attempt)

    @staticmethod
    def _present(path):
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def snapshot(self, now=None):
        now = self._clock() if now is None else now
        counts = Counter(expected=len(self.expected))
        stages = Counter()
        active = []
        try:
            entries = list(os.scandir(self.root))
        except FileNotFoundError:
            entries = []
        for entry in entries:
            token = self.expected.get(entry.name)
            if token is None or not entry.is_dir(follow_symlinks=False):
                continue
            sample = Path(entry.path)
            signature, state = self._json(sample / "sample_state.json")
            if (state is None or state.get("source_token") != token or not state.get("attempt_id")):
                counts["unbound"] += 1
                continue
            counts["attempted"] += 1
            attempt = state["attempt_id"]
            terminal = self._terminal.get(entry.name)
            if terminal is not None and terminal[0] == signature:
                counts.update(terminal[1])
                continue
            status = state.get("status")
            if status == "failed":
                counts["failed"] += 1
                continue
            if status == "processing":
                counts["processing"] += 1
                _, heartbeat = self._json(sample / "sample_progress.json")
                if (heartbeat is None or heartbeat.get("source_token") != token
                        or heartbeat.get("attempt_id") != attempt or heartbeat.get("schema") != PROGRESS_SCHEMA):
                    heartbeat = {}
                stage = str(heartbeat.get("stage") or "processing")
                stages[stage] += 1
                age = _elapsed(heartbeat.get("stage_started_utc", state.get("started_utc")), now)
                active.append(dict(token=token, stage=stage, elapsed_s=age,
                    request_attempt=heartbeat.get("model_request_attempt"),
                    last_finish_reason=heartbeat.get("last_finish_reason")))
                continue
            if status != "complete":
                counts["unbound"] += 1
                continue
            _, marker = self._json(sample / "sample_complete.json")
            if (marker is None or marker.get("status") != "complete" or marker.get("source_token") != token
                    or marker.get("attempt_id") != attempt or marker.get("identity_sha256") != state.get("identity_sha256")):
                counts["unbound"] += 1
                continue
            _, trajectory = self._json(sample / "trajectory.json")
            if trajectory is None or trajectory.get("frame_id") != token:
                counts["unbound"] += 1
                continue
            row = Counter(planned=1)
            audit = trajectory.get("condition_audit")
            if self.model_requested:
                if isinstance(audit, dict) and audit.get("model_requested") is True:
                    key = "model_applied" if audit.get("model_semantics_applied", audit.get("model_answer_valid")) is True else "base_fallback"
                    row[key] += 1
                else:
                    row["model_unknown"] += 1
            _, score = self._json(sample / "official_score_status.json")
            row["scored"] = int(self._bound(score, token, attempt) and self._present(sample / "official_score.json.gz"))
            _, video = self._json(sample / "official_replay/render_status.json")
            row["videos"] = int(self._bound(video, token, attempt) and self._present(sample / "official_replay/official_bev.mp4"))
            counts.update(row)
            if row["scored"] and (row["videos"] or not self.video):
                self._terminal[entry.name] = signature, row
        counts["not_attempted"] = max(0, counts["expected"]-counts["attempted"])
        longest = max(active, key=lambda row: row["elapsed_s"] if row["elapsed_s"] is not None else -1, default=None)
        keys = ("expected", "attempted", "planned", "scored", "videos", "model_applied", "base_fallback",
                "model_unknown", "failed", "processing", "unbound", "not_attempted")
        return dict(**{key: counts[key] for key in keys}, stages=dict(stages), longest_active=longest)

    def emit(self, phase="official_scoring"):
        try:
            row = self.snapshot()
            total = row["expected"]
            percent = 100*row["scored"]/total if total else 0.
            message = ("[NAVSIM进度] 阶段="+str(phase)+"；官方评分="+str(row["scored"])+"/"+str(total)
                       +" (%.2f%%)" % percent+"；规划完成="+str(row["planned"])+"；视频="+str(row["videos"])
                       +"；处理中="+str(row["processing"])+"；失败="+str(row["failed"]))
            if self.model_requested:
                message += "；模型有效="+str(row["model_applied"])+"；基础回退="+str(row["base_fallback"])
            if row["unbound"] or row["model_unknown"]:
                message += "；待核对记录="+str(row["unbound"]+row["model_unknown"])
            if row["longest_active"]:
                item = row["longest_active"]
                age = "未知" if item["elapsed_s"] is None else "%.0fs" % item["elapsed_s"]
                message += "；最长处理="+item["token"]+"/"+_STAGES.get(item["stage"], item["stage"])+"/"+age
                if item.get("request_attempt") is not None:
                    message += "/请求"+str(item["request_attempt"])
                if item.get("last_finish_reason"):
                    message += "/上次结束="+str(item["last_finish_reason"])
            self._printer(message)
            return row
        except Exception as exc:
            LOG.warning("NAVSIM progress unavailable (%s); official scoring continues", type(exc).__name__)
            return None
