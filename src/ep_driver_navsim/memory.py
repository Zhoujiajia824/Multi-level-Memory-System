"""Freeze an evaluation-disjoint view of the existing, read-only memory library."""
import hashlib
import json
from pathlib import Path

from .protocol import digest


def _provenance_values(value, keys):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, (str, int)):
                found.add(str(child))
            if key in ("source", "provenance", "source_metadata", "situation", "dataset", "metadata", "frame"):
                found.update(_provenance_values(child, keys))
    return found


def memory_provenance(record):
    """Use authenticated source metadata, never arbitrary experience prose."""
    logs = _provenance_values(record, {"log_name", "source_log_name", "log_id", "source_log"})
    tokens = _provenance_values(record, {"token", "source_token", "anchor_token", "initial_token",
        "source_anchor_token", "scene_token", "source_scene_token"})
    origins = _provenance_values(record, {"platform", "simulator", "record_origin"})
    simulated = bool(origins & {"highway", "highway-env", "carla", "online_highway", "online_carla"})
    if record.get("record_origin") == "offline_retrospective":
        # MemoryStore already authenticates the complete offline publication.
        # Its returned record does not expose task.capture source IDs, so read
        # that checksummed asset explicitly rather than guessing from frame_id.
        name = "audit/task.json"
        raw = (Path(record["_asset_root"]) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["checksums"].get(name):
            raise ValueError("Memory source audit changed: " + record["id"])
        task = json.loads(raw)
        if task.get("id") != record["id"] or task.get("anchor_frame_id") != record["frame_id"]:
            raise ValueError("Memory source identity differs: " + record["id"])
        capture = task.get("capture", {})
        source = capture.get("source")
        if source != record.get("source"):
            raise ValueError("Memory source differs from authenticated capture: " + record["id"])
        if source == "openscene":
            log = capture.get("source_log_name")
            if not isinstance(log, str) or not log:
                raise ValueError("OpenScene memory has no source log: " + record["id"])
            logs.add(log)
            for row in task["timeline"]:
                prefix, separator, token = row["frame_id"].rpartition(":")
                if prefix != log or not separator or not token:
                    raise ValueError("OpenScene memory frame/log identity differs: " + record["id"])
                tokens.add(token)
        elif source == "nuscenes":
            scene = capture.get("source_scene_token")
            if not isinstance(scene, str) or not scene:
                raise ValueError("nuScenes memory has no source scene: " + record["id"])
            tokens.add(scene)
            tokens.update(row["frame_id"] for row in task["timeline"])
        elif source in ("highway-env", "carla"):
            simulated = (capture.get("record_origin") == task.get("record_origin") == "closed_loop_simulation"
                         or (source == "highway-env" and task.get("record_origin") == "scripted_simulation"
                             and capture.get("protocol") == "scripted-highway-capture-1.2"))
        else:
            raise ValueError("Unsupported offline memory source: " + record["id"])
    if not logs and not tokens and not simulated:
        raise ValueError("Memory has no auditable source log/token: " + record["id"])
    return logs, tokens


def check_memory_disjoint(records, evaluation_logs, evaluation_tokens):
    logs, tokens = set(evaluation_logs), set(evaluation_tokens)
    for record in records:
        source_logs, source_tokens = memory_provenance(record)
        if source_logs & logs or source_tokens & tokens:
            raise ValueError("Frozen memory overlaps evaluation logs/tokens: " + record["id"])


def select_memory(records, source_snapshot_id, evaluation_logs, evaluation_tokens):
    """Exclude the entire overlapping log; preserve source files and fail unknown origins."""
    forbidden_logs, forbidden_tokens = set(evaluation_logs), set(evaluation_tokens)
    selected, excluded = [], []
    for record in records:
        logs, tokens = memory_provenance(record)
        overlap_logs, overlap_tokens = logs & forbidden_logs, tokens & forbidden_tokens
        if overlap_logs or overlap_tokens:
            excluded.append(dict(id=record["id"], reason="evaluation_source_overlap",
                                 logs=sorted(overlap_logs), tokens=sorted(overlap_tokens)))
        else:
            selected.append(record)
    selection = dict(policy="exclude_evaluation_logs_and_tokens_v1", source_snapshot_id=source_snapshot_id,
                     source_records=len(records), selected_records=len(selected), excluded_records=excluded)
    snapshot_id = digest(dict(selection=selection, selected_ids=sorted(record["id"] for record in selected)))
    return selected, snapshot_id, selection


def load_memory_snapshot(path, evaluation_logs, evaluation_tokens):
    from nine_grid.memory import MemoryStore
    store = MemoryStore(path, None)
    records = store.snapshot()
    selected, snapshot_id, selection = select_memory(records, store.snapshot_id, evaluation_logs, evaluation_tokens)
    selection["reader_rejected_records"] = store.audit
    return selected, snapshot_id, selection
