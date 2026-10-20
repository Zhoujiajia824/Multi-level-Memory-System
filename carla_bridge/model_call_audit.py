"""Portable model-call evidence, associated with one captured simulator frame."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time
from uuid import uuid4

from src.vla_memory.common.model_call_audit import observe_model_requests

logger = logging.getLogger(__name__)


def _compact_record(record):
    """Keep actual model I/O, excluding duplicated full numerical candidate pools."""
    result = deepcopy(record or {})
    for key in ("planning_output",):
        result.pop(key, None)
    decision = result.get("decision_output")
    if isinstance(decision, dict):
        diagnostic = decision.get("planner_diagnostics") or {}
        decision["planner_diagnostics"] = {key: diagnostic.get(key) for key in (
            "selected_id", "baseline_selected_id", "original_baseline_selected_id",
            "semantic_policy", "governor", "shadow_only", "shadow_selected_id")}
    return result


class ModelCallAudit:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.latest = None
        self.consultation_count = 0
        self.errors = []
        self._assets = {}

    def _write(self, filename, row):
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with (self.run_dir / filename).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        except Exception as exc:
            self.errors.append(type(exc).__name__+": "+str(exc))
            logger.warning("Model audit could not be written: %s", exc)

    def archive_image(self, source):
        source = Path(source).resolve()
        try:
            signature = (str(source), source.stat().st_mtime_ns, source.stat().st_size)
            if signature in self._assets:
                return self._assets[signature]
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = self.run_dir / "model_call_assets" / (digest+source.suffix.lower())
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            relative = target.relative_to(self.run_dir).as_posix()
            self._assets[signature] = relative
            return relative
        except OSError as exc:
            self.errors.append("image_archive: "+str(exc))
            return None

    def _request_observer(self, metadata, row):
        row = dict(row)
        sources = row.pop("image_paths", [])
        paths = [self.archive_image(path) for path in sources]
        messages = deepcopy(row.pop("messages", []))
        image_index = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "image_url":
                        # Never duplicate base64 payloads. Archive exact images separately.
                        block["image_url"] = {"url": paths[image_index] if image_index < len(paths) else None}
                        image_index += 1
        text = []
        system = []
        for message in messages:
            content = message.get("content")
            values = [content] if isinstance(content, str) else [
                block["text"] for block in (content or []) if block.get("type") == "text"]
            (system if message.get("role") == "system" else text).extend(values)
        self._write("model_requests.jsonl", {**metadata, **row, "messages": messages,
            "prompt": "\n".join(text), "system_prompt": "\n".join(system),
            "image_paths": paths, "source_image_paths": [str(p) for p in sources]})

    @contextmanager
    def consultation(self, kf, query, ego):
        self.consultation_count += 1
        sources = list(kf.get("image_paths") or [])
        mosaic = kf.get("mosaic_path") or kf.get("image_path")
        if mosaic and mosaic not in sources:
            sources.append(mosaic)
        metadata = {"schema_version": 1, "call_id": uuid4().hex,
            "sample_token": kf.get("sample_token"), "sim_time_s": kf.get("sim_time_s"),
            "carla_frame_id": kf.get("carla_frame_id"), "ego": ego,
            "trigger": deepcopy(query), "query": deepcopy(query),
            "context": deepcopy((kf.get("planning_constraints") or {}).get("planner_context")),
            "image_paths": [self.archive_image(path) for path in sources],
            "nav_instruction": kf.get("nav_instruction"),
            "consultation_index": self.consultation_count}
        result = {"record": None, "status": "started"}
        self.latest = metadata
        self._write("model_calls.jsonl", {**metadata, "event": "consultation_start"})
        started = time.perf_counter()
        try:
            with observe_model_requests(lambda row: self._request_observer(metadata, row)):
                yield result
            if result["status"] == "started":
                result["status"] = "completed"
        except BaseException as exc:
            result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            try:
                compact = _compact_record(result.get("record"))
            except Exception as exc:
                compact = {"audit_error": type(exc).__name__+": "+str(exc)}
                self.errors.append(compact["audit_error"])
            self._write("model_calls.jsonl", {**metadata, **result,
                "record": compact, "event": "consultation_end",
                "duration_wall_s": time.perf_counter()-started})
