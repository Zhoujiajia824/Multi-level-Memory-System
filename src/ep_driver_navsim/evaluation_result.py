"""Validation and lifecycle evidence around the unchanged official scorer.

This module never calculates benchmark metrics or substitutes missing samples.
Only the pinned SDK's complete, valid named CSV rows may become a final score.
"""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from nine_grid.paths import filesystem_path

from .protocol import digest, write_json


METRICS = (
    ("NC", "no_at_fault_collisions"),
    ("DAC", "drivable_area_compliance"),
    ("DDC", "driving_direction_compliance"),
    ("TLC", "traffic_light_compliance"),
    ("EP", "ego_progress"),
    ("TTC", "time_to_collision_within_bound"),
    ("LK", "lane_keeping"),
    ("HC", "history_comfort"),
    ("EC", "two_frame_extended_comfort"),
)
STAGES = ("stage_one", "stage_two")
SUMMARY_TOKENS = {"extended_pdm_score_" + name for name in (*STAGES, "combined")}
V1_METRICS = (("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"),
              ("TTC", "time_to_collision_within_bound"), ("EP", "ego_progress"), ("C", "comfort"))


def _profile(settings):
    value = settings.get("benchmark_profile", "v2_two_stage")
    if value not in ("v1_navtest", "v2_navtest", "v2_two_stage"):
        raise ValueError("Unknown NAVSIM benchmark_profile")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sample_state(sample, token):
    """Check attempt-bound completion; old archives without state stay readable."""
    path = filesystem_path(sample) / "sample_state.json"
    if not path.is_file():
        return None, None
    state = {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        marker = json.loads((path.parent / "sample_complete.json").read_text(encoding="utf-8"))
        if (state["schema"] != "ep-navsim-sample-state-1" or state["status"] != "complete"
                or not isinstance(state["attempt_id"], str) or not state["attempt_id"]
                or marker["schema"] != "ep-navsim-completion-1" or marker["status"] != "complete"
                or state["source_token"] != token or marker["source_token"] != token
                or marker["attempt_id"] != state["attempt_id"]
                or marker["identity_sha256"] != state["identity_sha256"]
                or digest(marker["identity"]) != state["identity_sha256"]):
            return state, "sample state/completion marker does not bind this complete attempt"
        return state, None
    except (OSError, ValueError, KeyError, TypeError):
        return state, "sample state/completion marker is incomplete or unreadable"


def _attempt_record(sample, relative, token, state):
    if state is None:
        return True
    try:
        value = json.loads((sample / relative).read_text(encoding="utf-8"))
        return (value.get("status") == "complete" and value.get("token") == token
                and value.get("sample_attempt_id") == state["attempt_id"])
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _number(value, context):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("Official CSV has a missing/non-numeric metric: " + context) from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RuntimeError("Official CSV has a non-finite/out-of-range metric: " + context)
    return value


def validate_csv(settings, csv_path):
    """Check all samples, all three SDK summaries and all nine scalar metrics."""
    if _profile(settings) != "v2_two_stage":
        return _validate_single_stage_csv(settings, csv_path)
    with filesystem_path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        required = {"token", "valid", "score"} | {
            field + "_" + stage for _, field in METRICS for stage in STAGES}
        if not required.issubset(columns):
            raise RuntimeError("Official CSV is incomplete: required two-stage metric columns are missing")
        rows = list(reader)
    expected = set(settings["evaluation_tokens"])
    by_token = {row.get("token"): row for row in rows}
    if (len(expected) != len(settings["evaluation_tokens"]) or len(by_token) != len(rows)
            or set(by_token) != expected | SUMMARY_TOKENS):
        raise RuntimeError("Official CSV is incomplete or has duplicate/unexpected tokens")
    invalid = [token for token, row in by_token.items() if str(row.get("valid", "")).lower() != "true"]
    if invalid:
        raise RuntimeError("Official run is not a complete valid benchmark: invalid rows=%d" % len(invalid))

    membership = {stage: set() for stage in STAGES}
    for token in expected:
        row = by_token[token]
        stages = [stage for stage in STAGES if any(
            str(row.get(field + "_" + stage, "")).lower() not in ("", "none", "nan")
            for _, field in METRICS)]
        if not stages:
            # Some SDK per-anchor quantities can remain undefined even with
            # valid=True. Their semantics belong to the official aggregator;
            # only the nine final stage summaries must be finite here.
            stages = [stage for stage, key in (("stage_one", "original_tokens"),
                                               ("stage_two", "reactive_synthetic_tokens"))
                      if token in settings.get(key, ())]
        if len(stages) != 1:
            raise RuntimeError("Official CSV sample has ambiguous/missing stage: " + token)
        stage = stages[0]
        membership[stage].add(token)
    for stage, key, count_key in (("stage_one", "original_tokens", "original_count"),
                                   ("stage_two", "reactive_synthetic_tokens", "reactive_synthetic_count")):
        if key in settings and membership[stage] != set(settings[key]):
            raise RuntimeError("Official CSV sample stage differs from the pinned split: " + stage)
        if count_key in settings and len(membership[stage]) != settings[count_key]:
            raise RuntimeError("Official CSV sample stage count differs from the pinned split: " + stage)

    scores, metrics = {}, []
    for stage in (*STAGES, "combined"):
        scores[stage] = _number(by_token["extended_pdm_score_" + stage]["score"], stage + ".score")
    for key, field in METRICS:
        values = {stage: _number(by_token["extended_pdm_score_" + stage].get(field + "_" + stage),
                                stage + "." + field) for stage in STAGES}
        # The SDK copies each stage's metrics into its combined summary row.
        # A stale/mixed CSV must not quietly provide two different tables.
        for stage, value in values.items():
            combined = _number(by_token["extended_pdm_score_combined"].get(field + "_" + stage),
                               "combined." + stage + "." + field)
            if not math.isclose(value, combined, rel_tol=0., abs_tol=1e-12):
                raise RuntimeError("Official CSV stage and combined summary metrics disagree")
        metrics.append(dict(key=key, **values))
    return dict(scores=scores, metrics=metrics, stage_counts={stage: len(tokens) for stage, tokens in membership.items()})


def _validate_single_stage_csv(settings, csv_path):
    """Read the original SDK average row without inventing a paired stage.

    Per-anchor columns may contain official undefined quantities. Only v2's
    EC summary may remain undefined: its one-stage evaluator has no adjacent
    endpoint pair and gives that term zero weight. It stays None in JSON.
    """
    profile = _profile(settings)
    is_v1 = profile == "v1_navtest"
    summary_token = "average" if is_v1 else "average_all_frames"
    metrics = V1_METRICS if is_v1 else METRICS
    with filesystem_path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        columns = set(names)
        required = {"token", "valid", "score"} | {field for _, field in metrics}
        if len(columns) != len(names) or not required.issubset(columns):
            raise RuntimeError("Official single-stage CSV is missing required columns or has duplicate headers")
        rows = list(reader)
    expected = set(settings["evaluation_tokens"])
    by_token = {row.get("token"): row for row in rows}
    if (len(expected) != len(settings["evaluation_tokens"]) or not expected
            or len(by_token) != len(rows) or set(by_token) != expected | {summary_token}):
        raise RuntimeError("Official single-stage CSV is incomplete or has duplicate/unexpected tokens")
    if any(str(row.get("valid", "")).lower() != "true" for row in rows):
        raise RuntimeError("Official single-stage CSV contains invalid samples or summary")
    for token in expected:
        _number(by_token[token].get("score"), token + ".score")
    summary = by_token[summary_token]
    score = _number(summary.get("score"), summary_token + ".score")
    if is_v1 and "driving_direction_compliance" in columns:
        metrics = (*metrics, ("DDC", "driving_direction_compliance"))
    values, unavailable = [], {}
    for key, field in metrics:
        raw = summary.get(field)
        if not is_v1 and key == "EC" and str(raw).lower() in ("", "none", "nan"):
            value = None
            unavailable[key] = "official_one_stage_no_adjacent_endpoint_pair; official_EC_weight_zero"
        else:
            value = _number(raw, summary_token + "." + field)
        values.append(dict(key=key, single_stage=value))
    return dict(benchmark_profile=profile, score_name="PDMS" if is_v1 else "EPDMS",
        scores=dict(single_stage=score), metrics=values,
        stage_counts=dict(single_stage=len(expected)), unavailable_metrics=unavailable,
        summary_token=summary_token)


def artifact_audit(settings):
    """Count existing per-sample evidence without decoding map/replay payloads."""
    root = filesystem_path(settings["output_dir"])
    expected = list(settings["evaluation_tokens"])
    counts = dict(trajectory_samples=0, official_score_records=0, replay_samples=0, video_samples=0)
    failed, missing, not_attempted, state_errors = [], [], [], {}
    for token in expected:
        sample = root / "samples" / digest(token)[:24]
        state, state_error = sample_state(sample, token)
        trajectory = (sample / "trajectory.json").is_file() and state_error is None
        failure = (sample / "failure.json").is_file()
        if state_error:
            state_errors[token] = state_error
        counts["trajectory_samples"] += trajectory
        counts["official_score_records"] += (state_error is None and (sample / "official_score.json.gz").is_file()
            and _attempt_record(sample, "official_score_status.json", token, state))
        counts["replay_samples"] += (state_error is None
            and _attempt_record(sample, "official_replay/capture_status.json", token, state)
            and any((sample / "official_replay" / name).is_file() for name in ("trace.json.gz", "trace.json")))
        counts["video_samples"] += (state_error is None and (sample / "official_replay/official_bev.mp4").is_file()
            and _attempt_record(sample, "official_replay/render_status.json", token, state))
        if failure or (state or {}).get("status") == "failed":
            failed.append(token)
        if not trajectory:
            missing.append(token)
        if not trajectory and not failure and state is None:
            not_attempted.append(token)
    return dict(expected_samples=len(expected), **counts, planning_failure_tokens=failed,
                missing_trajectory_tokens=missing, not_attempted_tokens=not_attempted,
                sample_state_errors=state_errors)


def run_state(settings, phase, status="running", started_utc=None, error=None, **extra):
    record = dict(status=status, phase=phase, expected_tokens=len(settings["evaluation_tokens"]),
                  official_csv=None,
                  split=settings["split"], input_condition=settings.get("input_condition"),
                  standard_sensor_only_comparable=False, official_commit=settings.get("official_commit"),
                  settings_sha256=digest(settings), started_utc=started_utc or utc_now(),
                  updated_utc=utc_now(), **extra)
    profile = _profile(settings)
    record.update(benchmark_profile=profile, navsim_version=settings.get("navsim_version"),
                  traffic_policy=settings.get("traffic_policy", settings.get("traffic_agents")))
    if profile == "v2_two_stage":
        record.setdefault("official_two_stage_epdms", None)
    else:
        record.setdefault("official_score", None)
        record.setdefault("score_name", "PDMS" if profile == "v1_navtest" else "EPDMS")
    if status != "running":
        record["finished_utc"] = utc_now()
    if error is not None:
        record["error"] = dict(type=type(error).__name__, message=str(error))
        if hasattr(error, "returncode"):
            record["error"]["returncode"] = error.returncode
    write_json(Path(settings["output_dir"]) / "benchmark_result.json", record)
    return record
