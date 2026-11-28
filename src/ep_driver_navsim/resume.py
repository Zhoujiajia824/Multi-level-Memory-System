"""Content-bound sample completion and conservative checkpoint reuse.

An absent, old, partial, corrupt or mismatched checkpoint is simply a cache miss.
The completion marker is published last. It is never inferred from trajectory.json.
"""
import hashlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import uuid
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
from nine_grid.paths import filesystem_path

from . import INPUT_PROTOCOL
from .archive import read_input_packet, read_json
from .protocol import digest, horizon_contract, relative_asset_path, trajectory_from_centers, write_json
from .model import (STRUCTURED_OUTPUT_PROTOCOL, goal_retry_policy, goal_retry_prompt,
                    MODEL_ATTEMPT_PROTOCOL, SEMANTIC_RETRY_STAGES)
from .model_transport import model_request_policy

SCHEMA = "ep-navsim-completion-1"
COMPLETION_FILE = "sample_complete.json"
SAMPLE_STATE_FILE = "sample_state.json"
SAMPLE_STATE_SCHEMA = "ep-navsim-sample-state-1"


def file_sha256(path):
    with filesystem_path(path).open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _model_identity(settings, model):
    # Credentials are supplied only by the model client's named environment
    # variable. Do not hash its value or copy it into identities or diagnostics.
    for key in model:
        lowered = key.lower()
        if lowered != "api_key_env" and any(word in lowered for word in
                ("api_key", "apikey", "password", "authorization", "secret", "access_token")):
            raise ValueError("Inline model credentials are unsupported; use api_key_env")
    endpoint = str(model.get("base_url", ""))
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Model base_url must not contain credentials, query or fragment")
    revision = settings.get("model_revision")
    if settings["fast"] != "off" and settings.get("resume", True) and (
            not isinstance(revision, str) or not revision.strip()):
        raise ValueError("Enabled-model resume requires an explicit model_revision")
    return dict(backend=settings["fast"], model=model.get("model"), base_url=endpoint,
                max_tokens=model.get("max_tokens", 4096), timeout_s=model.get("timeout_s", 120),
                model_revision=revision)


def build_run_identity(settings, effective_config, model_settings, system_prompts, memory_snapshot_id):
    """Freeze effective numerical inputs once per worker, including local weights."""
    model = _model_identity(settings, model_settings)
    model_root = filesystem_path(settings["dino_model"])
    model_files = sorted(p for p in model_root.rglob("*") if p.is_file()
                         and ".git" not in p.relative_to(model_root).parts)
    if not model_files:
        raise ValueError("Cannot fingerprint an empty local DINO model")
    dino = {p.relative_to(model_root).as_posix(): file_sha256(p) for p in model_files}
    dependencies = {}
    for package in ("numpy", "scipy", "casadi", "shapely", "torch", "transformers"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return dict(schema=SCHEMA, input_protocol=INPUT_PROTOCOL,
                benchmark=dict(profile=settings.get("benchmark_profile", "v2_two_stage"),
                    navsim_version=settings.get("navsim_version", "v2.2"),
                    official_commit=settings.get("official_commit"),
                    official_source_sha256=settings.get("official_source_sha256"),
                    traffic_policy=settings.get("traffic_policy", settings.get("traffic_agents", "reactive"))),
                runtime=dict(python=sys.version, platform=platform.platform(), dependencies=dependencies),
                numerical_source_sha256=settings["numerical_source_sha256"],
                effective_parameters_sha256=digest(effective_config),
                config_file_sha256={key: settings["config_file_sha256"][key]
                                    for key in ("parameters", "prompts_config")},
                model=model, model_service=settings.get("model_service", {}),
                model_service_validation=settings.get("model_service_validation"),
                model_request_policy=(model_request_policy(settings) if settings.get("structured_model_output") else None),
                structured_model_output=settings.get("structured_model_output", False),
                model_goal_retry_policy=goal_retry_policy(settings),
                structured_output_protocol=(STRUCTURED_OUTPUT_PROTOCOL
                    if settings.get("structured_model_output", False) else None),
                system_prompts_sha256={key: digest(value) for key, value in system_prompts.items()},
                dino_files_sha256=dino, dino_device=settings["dino_device"],
                retrieval_enabled=settings["retrieve"], memory_snapshot_id=memory_snapshot_id,
                memory_selection_sha256=digest(settings.get("memory_selection")),
                sample_state_policy="independent_anchor_reset_no_online_memory_writes")


def sample_identity(packet, run_identity):
    cameras = []
    for row in packet["camera_images"]:
        path = filesystem_path(row["path"])
        cameras.append(dict(view=row["view"], frame_id=row["frame_id"], time=row["time"],
                            path=str(path.resolve()), size_bytes=path.stat().st_size,
                            sha256=file_sha256(path)))
    return dict(schema=SCHEMA, run=run_identity, source_token=packet["identity"]["token"],
                input_packet_sha256=digest(packet),
                camera_content=cameras)


def begin_sample(root, identity=None, source_token=None):
    """Publish an active attempt before invalidating old success pointers.

    Old trajectories and official evidence remain on disk. They cannot become
    the current attempt merely because a retry fails before overwriting them.
    """
    root = filesystem_path(root)
    root.mkdir(parents=True, exist_ok=True)
    state = dict(schema=SAMPLE_STATE_SCHEMA, status="processing", attempt_id=uuid.uuid4().hex,
                 source_token=identity.get("source_token") if identity is not None else source_token,
                 identity_sha256=digest(identity) if identity is not None else None,
                 started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root / SAMPLE_STATE_FILE, state)
    (root / COMPLETION_FILE).unlink(missing_ok=True)
    (root / "failure.json").unlink(missing_ok=True)
    return state


def fail_sample(root, error, attempt=None, source_token=None):
    root = filesystem_path(root)
    if attempt is None:
        attempt = begin_sample(root, source_token=source_token)
    state = read_json(root / SAMPLE_STATE_FILE)
    if state.get("attempt_id") != attempt["attempt_id"]:
        return False  # A superseded computation cannot mark another attempt failed.
    state.update(status="failed", finished_utc=datetime.now(timezone.utc).isoformat(),
                 error=dict(type=type(error).__name__, detail=str(error)))
    write_json(root / SAMPLE_STATE_FILE, state)
    return True


def _current_state(root, identity, attempt_id=None):
    state = read_json(root / SAMPLE_STATE_FILE)
    if (state.get("schema") != SAMPLE_STATE_SCHEMA or state.get("status") not in ("processing", "complete")
            or state.get("source_token") != identity.get("source_token")
            or state.get("identity_sha256") != digest(identity)
            or not isinstance(state.get("attempt_id"), str) or not state["attempt_id"]
            or (attempt_id is not None and state["attempt_id"] != attempt_id)):
        raise ValueError("Sample attempt or source identity changed")
    return state


def _local_asset(root, name):
    root = filesystem_path(root)
    path = filesystem_path(root / name)
    if Path(name).is_absolute():
        raise ValueError("Checkpoint asset must stay inside its sample directory")
    relative_asset_path(path, root)
    return path


def _unsent_semantic_preparation(result, evidence):
    """Only a fully evidenced local NoLegal condition can omit a request schema."""
    fallback = evidence.get("model_preparation_fallback")
    audit = result.get("condition_audit", {})
    if not isinstance(audit, dict):
        raise ValueError("Checkpoint model condition audit must be a mapping")
    if fallback is None and audit.get("model_preparation_fallback") is None:
        return False
    required = {"reason", "detail", "stage", "request_sent", "policy"}
    if (not isinstance(fallback, dict) or set(fallback) != required
            or not isinstance(fallback.get("reason"), str)
            or fallback["reason"] not in {"semantic_reference_invalid", "empty_longitudinal_goal_interval",
                                          "current_state_not_forward_frenet", "current_state_frenet_boundary_nonfinite"}
            or not isinstance(fallback.get("detail"), str) or not fallback["detail"].strip()
            or fallback.get("stage") != "structured_request_preparation"
            or fallback.get("request_sent") is not False
            or fallback.get("policy") != "same_base_mpc_without_model_semantics"
            or audit.get("model_preparation_fallback") != fallback
            or audit.get("model_requested") is not True
            or audit.get("model_answer_valid") is not False
            or audit.get("model_answer_parsed_valid") is not False
            or audit.get("model_semantics_applied") is not False
            or evidence.get("model_semantics_applied") is not False
            or evidence.get("model_answer") is not None or evidence.get("model_raw") is not None
            or evidence.get("model_planning_fallback") is not None
            or audit.get("model_planning_fallback") is not None
            or evidence.get("model_response_format_sha256") is not None):
        raise ValueError("Checkpoint unsent semantic preparation evidence is incomplete")
    for record in (audit, evidence):
        if (record.get("structured_model_output") is not True
                or record.get("model_error") != "NoLegalGoalReference"
                or record.get("model_error_detail") != fallback["detail"]
                or type(record.get("model_call_attempts")) is not int
                or record["model_call_attempts"] != 0
                or record.get("model_request_sent") is not False):
            raise ValueError("Checkpoint unsent semantic preparation transport evidence differs")
    return True


def _validate_output(root, identity, assets):
    required = {"trajectory.json", "planning_evidence.json", "trajectory_bev.png",
                "retrieval_diagnostics.json.gz"}
    packet_name = "input_packet.json.gz" if "input_packet.json.gz" in assets else "input_packet.json"
    required.add(packet_name)
    if not required.issubset(assets):
        raise ValueError("Checkpoint is missing required evidence")
    packet = read_input_packet(root / packet_name)
    if sample_identity(packet, identity["run"]) != identity:
        raise ValueError("Checkpoint packet/camera content differs")
    result = read_json(root / "trajectory.json")
    evidence = read_json(root / "planning_evidence.json")
    attempts = {}
    policy = identity["run"].get("model_goal_retry_policy")
    if policy is not None:
        for record in (result.get("condition_audit", {}), evidence):
            if record.get("model_goal_retry_policy") != policy:
                raise ValueError("Checkpoint model correction policy differs")
        if identity["run"]["model"]["backend"] != "off":
            if "model_attempts.json.gz" not in assets or evidence.get("model_attempts_asset") != "model_attempts.json.gz":
                raise ValueError("Checkpoint model attempt evidence is missing")
            attempts = read_json(root / "model_attempts.json.gz")
            rows = attempts.get("attempts", [])
            if (attempts.get("schema") != MODEL_ATTEMPT_PROTOCOL or attempts.get("policy") != policy
                    or attempts.get("frame_id") != packet["identity"]["token"]
                    or type(rows) is not list or len(rows) > policy["limit"]+1
                    or attempts.get("final_stage") != evidence.get("model_validation_stage")
                    or len(rows) != evidence.get("model_semantic_attempts")
                    or len(rows) != result.get("condition_audit", {}).get("model_semantic_attempts")):
                raise ValueError("Checkpoint model attempt evidence differs")
            if rows and (rows[-1].get("raw") != evidence.get("model_raw")
                         or rows[-1].get("validation_stage") != evidence.get("model_validation_stage")):
                raise ValueError("Checkpoint final model answer differs from its attempt")
            base_prompt = (root / "model_prompt.txt").read_text(encoding="utf-8")
            for index, row in enumerate(rows):
                prompt = base_prompt
                if index:
                    previous = rows[index-1]
                    stage = previous.get("validation_stage")
                    if (stage not in SEMANTIC_RETRY_STAGES
                            or row.get("feedback") != dict(rejected_attempt=index,
                                reason=previous.get("error_detail"), stage=stage)):
                        raise ValueError("Checkpoint correction does not follow a correctable rejection")
                    prompt = goal_retry_prompt(base_prompt, previous.get("raw"), previous.get("error_detail"), index+1,
                                               stage=stage)
                if (row.get("attempt") != index+1
                        or row.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        or row.get("original_reference_id") != rows[0].get("original_reference_id")
                        or row.get("response_format_sha256") != evidence.get("model_response_format_sha256")):
                    raise ValueError("Checkpoint correction source or prompt differs")
            if ((not rows and evidence.get("model_validation_stage") != "preparation_unavailable")
                    or (evidence.get("model_answer") is not None and evidence.get("model_validation_stage") != "accepted")):
                raise ValueError("Checkpoint model acceptance stage differs")
    if identity["run"].get("structured_model_output"):
        request_policy = identity["run"].get("model_request_policy")
        if request_policy is not None:
            for record in (result.get("condition_audit", {}), evidence):
                if record.get("model_request_policy") != request_policy:
                    raise ValueError("Checkpoint bounded model request policy differs")
            if attempts.get("request_policy") != request_policy:
                raise ValueError("Checkpoint model attempt deadline policy differs")
            if evidence.get("model_request_sent"):
                name = "model_transport_events.json.gz"
                if name not in assets or evidence.get("model_transport_events_asset") != name:
                    raise ValueError("Checkpoint per-request transport audit is missing")
                events = read_json(root / name)
                state = read_json(root / SAMPLE_STATE_FILE)
                if (events.get("frame_id") != packet["identity"]["token"]
                        or events.get("sample_attempt_id") != state.get("attempt_id")
                        or events.get("policy") != request_policy
                        or len(events.get("events", [])) != evidence.get("model_call_attempts")):
                    raise ValueError("Checkpoint per-request transport audit differs from this attempt")
        unsent = _unsent_semantic_preparation(result, evidence)
        has_schema = "model_response_format.json" in assets
        if not has_schema and not unsent:
            raise ValueError("Checkpoint is missing the structured model request schema")
        if has_schema and unsent:
            raise ValueError("Checkpoint unsent semantic preparation must not claim a request schema")
        if has_schema and digest(read_json(root / "model_response_format.json")) != evidence["model_response_format_sha256"]:
            raise ValueError("Checkpoint structured model request evidence differs")
    if (result["frame_id"] != packet["identity"]["token"]
            or result["sampling"] != dict(horizon_s=4.0, interval_s=0.5)
            or evidence["identity"] != packet["identity"]
            or evidence["input_packet_sha256"] != identity["input_packet_sha256"]
            or evidence["resume_identity_sha256"] != digest(identity)
            or evidence["parameters_sha256"] != identity["run"]["effective_parameters_sha256"]
            or digest(evidence["parameters"]) != evidence["parameters_sha256"]
            or evidence["memory_snapshot_id"] != identity["run"]["memory_snapshot_id"]
            or not evidence["plan"]["valid"]):
        raise ValueError("Checkpoint trajectory and numerical evidence disagree")
    poses = np.asarray(result["poses"], float)
    if poses.shape != (8, 3) or not np.isfinite(poses).all():
        raise ValueError("Checkpoint trajectory must contain eight finite SE2 poses")
    refinement_enabled = bool(evidence["parameters"].get("navsim_execution_refinement", {}).get("enabled"))
    if refinement_enabled:
        from .execution_evidence import validate_refinement_record
        expected_poses = validate_refinement_record(evidence.get("execution_refinement"), packet, poses,
            benchmark_profile=identity['run'].get('benchmark', {}).get('profile'))
        if result.get("trajectory_source") != "execution_aware_reference":
            raise ValueError("Checkpoint does not identify its execution-aware submission")
    else:
        if evidence.get("execution_refinement") is not None:
            raise ValueError("Disabled execution refinement must not claim refined evidence")
        expected_poses = trajectory_from_centers(evidence["plan"]["states"],
            evidence["parameters"]["mpc"]["step_s"], packet["ego"]["rear_axle_to_center_m"])
    if not np.array_equal(poses.astype(np.float32), expected_poses):
        raise ValueError("Checkpoint trajectory differs from its planned states")
    plans = {key: SimpleNamespace(**{field: evidence[key][field] for field in ("states", "controls")})
             for key in ("plan", "reference")}
    predictions = dict(prediction=SimpleNamespace(**{key: evidence["predictions"][key]
                                                    for key in ("times", "trajectories")}),
                       reference_prediction=SimpleNamespace(times=evidence["reference_prediction_times_s"],
                                                             trajectories={}))
    checked = horizon_contract(evidence["parameters"], plans, predictions)
    for key in ("horizon_s", "step_s", "horizon_steps", "state_count", "output_poses", "extrapolation"):
        if evidence["horizon_contract"][key] != checked[key]:
            raise ValueError("Checkpoint horizon evidence differs")
    if set(evidence["representation_assets"]) != {"bev", "feature", "grid", "occupancy"}:
        raise ValueError("Checkpoint representation assets are incomplete")
    for name in evidence["representation_assets"].values():
        required.add(name)
    # Actual camera content is checked independently and also appears in manifest.
    for row in packet["camera_images"]:
        required.add(relative_asset_path(row["path"], root))
    if not required.issubset(assets):
        raise ValueError("Checkpoint is missing representation/camera assets")
    return result, evidence


def _condition_success(result, evidence, identity):
    requested = identity["run"]["model"]["backend"] != "off"
    audit = result["condition_audit"]
    if (audit["model_requested"] is not requested
            or audit["retrieval_enabled"] is not identity["run"]["retrieval_enabled"]):
        return False
    return (not requested or (audit.get("model_answer_valid") is True
        and audit.get("model_error") is None and evidence.get("model_error") is None
        and isinstance(evidence.get("model_answer"), dict)
        and isinstance(evidence.get("model_raw"), str) and bool(evidence["model_raw"])
        and bool(identity["run"]["model"].get("model_revision"))))


def complete_sample(root, identity, artifact_paths, attempt_id=None):
    """Validate complete outputs, hash assets, then atomically publish success.

    A scored fallback can be complete, but B/C fallback completion is never reusable.
    """
    root = filesystem_path(root)
    state = _current_state(root, identity, attempt_id) if (root / SAMPLE_STATE_FILE).exists() else None
    assets = {}
    for value in artifact_paths:
        path = filesystem_path(value)
        name = relative_asset_path(path, root)
        assets[name] = dict(sha256=file_sha256(path), size_bytes=path.stat().st_size)
    result, evidence = _validate_output(root, identity, assets)
    requested = identity["run"]["model"]["backend"] != "off"
    if requested and "model_prompt.txt" not in assets:
        raise ValueError("Model participation evidence is incomplete")
    if state is not None:
        _current_state(root, identity, state["attempt_id"])
    write_json(root / COMPLETION_FILE, dict(schema=SCHEMA, status="complete",
        reusable=_condition_success(result, evidence, identity), identity=identity,
        identity_sha256=digest(identity), assets=assets,
        **(dict(attempt_id=state["attempt_id"], source_token=state["source_token"]) if state else {})))
    if state is not None:
        _current_state(root, identity, state["attempt_id"])
        state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / SAMPLE_STATE_FILE, state)


def load_completed_sample(root, identity):
    """Return a verified trajectory or None. Never trust mere file existence."""
    root = filesystem_path(root)
    try:
        if (root / "failure.json").exists():
            return None
        marker = read_json(root / COMPLETION_FILE)
        if (root / SAMPLE_STATE_FILE).exists():
            state = _current_state(root, identity)
            if (state["status"] != "complete" or marker.get("attempt_id") != state["attempt_id"]
                    or marker.get("source_token") != state["source_token"]):
                return None
        if (marker["schema"] != SCHEMA or marker["status"] != "complete"
                or marker["reusable"] is not True or marker["identity"] != identity
                or marker["identity_sha256"] != digest(identity)):
            return None
        assets = marker["assets"]
        for name, expected in assets.items():
            path = _local_asset(root, name)
            if (path.stat().st_size != expected["size_bytes"] or file_sha256(path) != expected["sha256"]):
                return None
        result, evidence = _validate_output(root, identity, assets)
        if not _condition_success(result, evidence, identity):
            return None
        if identity["run"]["model"]["backend"] != "off" and "model_prompt.txt" not in assets:
            return None
        result["resume"] = dict(reused=True, identity_sha256=digest(identity))
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, OverflowError):
        return None
