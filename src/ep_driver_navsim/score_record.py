"""Compact evidence of the official per-anchor return, without rescoring.

These records are audit/recovery inputs, not an alternative benchmark CSV.
Two-frame comfort, mapping weights and final EPDMS still require the official
complete-set aggregation. They are saved even when BEV video is disabled.
"""
import gzip
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
from nine_grid.paths import filesystem_path

from .protocol import digest, write_bytes, write_json
from .profiles import get_profile


VERSION = "navsim-official-score-record-1.0"


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def destination(arguments, settings):
    token = str(arguments["metric_cache"].file_path).replace("\\", "/").split("/")[-2]
    if token not in settings["evaluation_tokens"]:
        raise ValueError("Official score token is outside the frozen split")
    return token, filesystem_path(settings["output_dir"]) / "samples" / digest(token)[:24]


def current_attempt(arguments, settings):
    from .evaluation_result import sample_state
    token, sample = destination(arguments, settings)
    state, error = sample_state(sample, token)
    if error:
        raise ValueError("Official score does not belong to a complete sample attempt: " + error)
    if state is not None:
        trajectory = json.loads((sample / "trajectory.json").read_text(encoding="utf-8"))
        if ((trajectory.get("resume") or {}).get("identity_sha256") != state["identity_sha256"]
                or not np.array_equal(np.asarray(trajectory["poses"], dtype=np.float32),
                                      np.asarray(arguments["model_trajectory"].poses, dtype=np.float32))):
            raise ValueError("Official scorer trajectory differs from the current sample attempt")
    return state["attempt_id"] if state is not None else None


def score_payload(arguments, result, settings, capture=None):
    """Detach the version's actual return; never run or approximate its scorer."""
    profile = get_profile(settings)
    cache = arguments["metric_cache"]
    if profile["is_v1"]:
        if not is_dataclass(result) or isinstance(result, type):
            raise ValueError("NAVSIM v1 official score must be a PDMResults dataclass")
        metrics = asdict(result)
        if set(metrics) != {"no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"}:
            raise ValueError("Unexpected NAVSIM v1 official metric fields")
        # v1 returns no states. Observe its one existing simulator call, even
        # when video is disabled; a second rollout would not be score evidence.
        if not capture or "simulated_states" not in capture:
            raise ValueError("NAVSIM v1 score requires the original official simulator capture")
        states = np.asarray(capture["simulated_states"])
        source = "official_v1_pdm_score_dataclass_and_observed_simulator_return"
    else:
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("NAVSIM v2 official score must return its dataframe and states")
        frame, states = result[0], np.asarray(result[1])
        if len(frame) != 1:
            raise ValueError("Unexpected NAVSIM v2 per-anchor score shape")
        metrics = frame.to_dict(orient="records")[0]
        source = ("official_pdm_score_return_before_two_stage_aggregation" if profile["two_stage"]
                  else "official_pdm_score_return_single_stage")
    if states.ndim != 2 or states.shape[1] != 11 or not np.isfinite(states).all():
        raise ValueError("Unexpected official per-anchor state shape")
    sampling = arguments["future_sampling"]
    if states.shape[0] != int(sampling.num_poses) + 1:
        raise ValueError("Official score record time grid differs from simulated states")
    stage = (getattr(cache.scene_type, "name", str(cache.scene_type)).lower()
             if profile["two_stage"] else "single_stage")
    path = str(cache.file_path).replace("\\", "/").split("/")
    return dict(metrics=metrics, states=states, source=source, stage=stage,
        profile=profile, log_name=str(getattr(cache, "log_name", path[-4])),
        start_time_s=float(getattr(cache, "timepoint", cache.ego_state.time_point).time_s))


def export_score_record(arguments, result, settings, capture=None):
    """Serialize detached SDK results; never edit the result or call the scorer."""
    token, sample = destination(arguments, settings)
    attempt_id = current_attempt(arguments, settings)
    cache = arguments["metric_cache"]
    payload = score_payload(arguments, result, settings, capture)
    states = payload["states"]
    sampling = arguments["future_sampling"]
    if states.shape[0] != int(sampling.num_poses) + 1:
        raise ValueError("Official score record time grid differs from returned states")
    poses = np.asarray(arguments["model_trajectory"].poses)
    if poses.shape != (8, 3) or not np.isfinite(poses).all():
        raise ValueError("Official score record requires the submitted eight finite poses")
    record = _plain(dict(version=VERSION, token=token, stage=payload["stage"], sample_attempt_id=attempt_id,
        benchmark_profile=payload["profile"]["id"], score_name=payload["profile"]["score_name"],
        settings_sha256=digest(settings), official_commit=settings.get("official_commit"),
        official_source_sha256=settings.get("official_source_sha256"),
        result=payload["metrics"], ego_simulated_states=states,
        submitted_poses=poses, dt_s=float(sampling.interval_length),
        log_name=payload["log_name"], start_time_s=payload["start_time_s"],
        source_rear_axle=[float(cache.ego_state.rear_axle.x), float(cache.ego_state.rear_axle.y),
                          float(cache.ego_state.rear_axle.heading)],
        source=payload["source"],
        nonfinite_metric_values="null means unresolved/nonfinite; never imputed as a numeric score",
        score_unchanged=True, aggregation_complete=False,
        future_data_scope="scoring_only; never forwarded to planner or memory"))
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    write_bytes(sample / "official_score.json.gz", gzip.compress(encoded, mtime=0))
    write_json(sample / "official_score_status.json", dict(status="complete", token=token,
        version=VERSION, benchmark_profile=payload["profile"]["id"],
        sample_attempt_id=attempt_id, score_unchanged=True, aggregation_complete=False))
    return sample / "official_score.json.gz"


def record_failure(arguments, settings, error):
    token, sample = destination(arguments, settings)
    write_json(sample / "official_score_status.json", dict(status="failed", token=token,
        version=VERSION, error=repr(error), score_unchanged=True,
        failure_scope="score_evidence_only; official numerical result is unchanged"))
