"""Compact evidence of the official per-anchor return, without rescoring.

These records are audit/recovery inputs, not an alternative benchmark CSV.
Two-frame comfort, mapping weights and final EPDMS still require the official
complete-set aggregation. They are saved even when BEV video is disabled.
"""
import gzip
import json
import math
from pathlib import Path

import numpy as np
from nine_grid.paths import filesystem_path

from .protocol import digest, write_bytes, write_json


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


def export_score_record(arguments, result, settings):
    """Serialize detached SDK results; never edit the result or call the scorer."""
    token, sample = destination(arguments, settings)
    attempt_id = current_attempt(arguments, settings)
    cache = arguments["metric_cache"]
    states = np.asarray(result[1])
    frame = result[0]
    if len(frame) != 1 or states.ndim != 2 or states.shape[1] != 11 or not np.isfinite(states).all():
        raise ValueError("Unexpected official per-anchor score/state shape")
    sampling = arguments["future_sampling"]
    if states.shape[0] != int(sampling.num_poses) + 1:
        raise ValueError("Official score record time grid differs from returned states")
    poses = np.asarray(arguments["model_trajectory"].poses)
    if poses.shape != (8, 3) or not np.isfinite(poses).all():
        raise ValueError("Official score record requires the submitted eight finite poses")
    stage = getattr(cache.scene_type, "name", str(cache.scene_type)).lower()
    record = _plain(dict(version=VERSION, token=token, stage=stage, sample_attempt_id=attempt_id,
        settings_sha256=digest(settings), official_commit=settings.get("official_commit"),
        official_source_sha256=settings.get("official_source_sha256"),
        result=frame.to_dict(orient="records")[0], ego_simulated_states=states,
        submitted_poses=poses, dt_s=float(sampling.interval_length),
        log_name=str(cache.log_name), start_time_s=float(cache.timepoint.time_s),
        source_rear_axle=[float(cache.ego_state.rear_axle.x), float(cache.ego_state.rear_axle.y),
                          float(cache.ego_state.rear_axle.heading)],
        source="official_pdm_score_return_before_two_stage_aggregation",
        nonfinite_metric_values="null means unresolved/nonfinite; never imputed as a numeric score",
        score_unchanged=True, aggregation_complete=False,
        future_data_scope="scoring_only; never forwarded to planner or memory"))
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    write_bytes(sample / "official_score.json.gz", gzip.compress(payload, mtime=0))
    write_json(sample / "official_score_status.json", dict(status="complete", token=token,
        version=VERSION, sample_attempt_id=attempt_id, score_unchanged=True, aggregation_complete=False))
    return sample / "official_score.json.gz"


def record_failure(arguments, settings, error):
    token, sample = destination(arguments, settings)
    write_json(sample / "official_score_status.json", dict(status="failed", token=token,
        version=VERSION, error=repr(error), score_unchanged=True,
        failure_scope="score_evidence_only; official numerical result is unchanged"))
