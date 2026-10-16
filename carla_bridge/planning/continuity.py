"""Soft trajectory continuity in physical time, independent of semantic labels.

This module never changes trajectories or decides whether they are safe. Its
bounded costs may only rank candidates that pass the planner's current safety
and route checks. Callers own the previous snapshot; repeated on/off scoring of
one snapshot must receive the same reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrajectoryReference:
    """Previously selected world-frame rear-axle poses and longitudinal speeds."""

    states: np.ndarray  # [time, >=3]: world x, world y, heading in radians
    speeds: np.ndarray  # [time]: m/s
    timestamp_us: int
    sample_interval_s: float


def trajectory_continuity_costs(
    candidate_states,
    candidate_speeds,
    *,
    timestamp_us,
    sample_interval_s,
    previous: Optional[TrajectoryReference] = None,
    weight=0.15,
    comparison_horizon_s=1.0,
    max_reference_age_s=0.5,
):
    """Return per-candidate costs in [0, weight] and auditable comparison data.

    Candidate arrays are [candidate, time, >=3] world rear-axle poses and
    [candidate, time] speeds. The previous plan is interpolated to the new
    plan's absolute future sample times; heading interpolation unwraps angles.
    Lateral displacement is measured normal to the previous path's heading.
    Along-path displacement is not charged separately from speed deviation.

    Only the next second (configurable) of common future time is compared.
    Lateral, heading and speed errors saturate at 0.5 m, 15 degrees and 2 m/s,
    respectively. Their equally weighted mean cannot exceed ``weight``.
    Missing, stale, invalid or future references disable this advisory cost.
    Non-finite candidates receive zero cost and are identified in diagnostics;
    rejecting them remains the mandatory responsibility of the safety gate.
    """
    states = np.asarray(candidate_states, dtype=float)
    speeds = np.asarray(candidate_speeds, dtype=float)
    if states.ndim != 3 or states.shape[2] < 3 or speeds.shape != states.shape[:2]:
        raise ValueError("continuity requires poses [N,T,>=3] and speeds [N,T]")
    for name, value in (("sample_interval_s", sample_interval_s),
                        ("comparison_horizon_s", comparison_horizon_s),
                        ("max_reference_age_s", max_reference_age_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be positive and finite")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("continuity weight must be nonnegative and finite")

    costs = np.zeros(len(states), dtype=float)
    context = {"applied": False, "reason": "no_previous", "weight": float(weight),
               "comparison_horizon_s": float(comparison_horizon_s),
               "compared_samples": 0,
               "scales": {"lateral_m": 0.5, "heading_rad": math.radians(15), "speed_mps": 2.0}}
    if previous is None:
        return costs, context
    if weight == 0:
        context["reason"] = "zero_weight"
        return costs, context
    if not (math.isfinite(timestamp_us) and math.isfinite(previous.timestamp_us)):
        context["reason"] = "non_finite_timestamp"
        return costs, context
    age = (timestamp_us - previous.timestamp_us) / 1e6
    context["reference_age_s"] = float(age)
    if age < 0:
        context["reason"] = "clock_rollback"
        return costs, context
    if age > max_reference_age_s:
        context["reason"] = "stale_previous"
        return costs, context
    previous_states = np.asarray(previous.states, dtype=float)
    previous_speeds = np.asarray(previous.speeds, dtype=float)
    if (previous_states.ndim != 2 or previous_states.shape[1] < 3
            or previous_speeds.shape != previous_states.shape[:1]
            or len(previous_states) < 2):
        context["reason"] = "invalid_previous_shape"
        return costs, context
    if (not np.isfinite(previous_states[:, :3]).all()
            or not np.isfinite(previous_speeds).all()
            or not math.isfinite(previous.sample_interval_s)
            or previous.sample_interval_s <= 0):
        context["reason"] = "non_finite_or_invalid_previous"
        return costs, context

    times = np.arange(states.shape[1], dtype=float) * sample_interval_s
    previous_times = np.arange(len(previous_states), dtype=float) * previous.sample_interval_s
    # Exclude t=0: every new proposal is initialized at the same measured pose.
    overlap = ((times > 0) & (times <= comparison_horizon_s + 1e-9)
               & (age + times <= previous_times[-1] + 1e-9))
    if not overlap.any():
        context["reason"] = "no_future_overlap"
        return costs, context
    aligned_times = np.minimum(age + times[overlap], previous_times[-1])
    reference_xy = np.column_stack([
        np.interp(aligned_times, previous_times, previous_states[:, axis]) for axis in (0, 1)
    ])
    reference_heading = np.interp(
        aligned_times, previous_times, np.unwrap(previous_states[:, 2]))
    reference_speed = np.interp(aligned_times, previous_times, previous_speeds)
    finite = np.isfinite(states[:, :, :3]).all(axis=(1, 2)) & np.isfinite(speeds).all(axis=1)
    context.update(compared_samples=int(overlap.sum()),
                   compared_future_start_s=float(times[overlap][0]),
                   compared_future_end_s=float(times[overlap][-1]),
                   ignored_non_finite_candidates=np.flatnonzero(~finite).tolist())
    if not finite.any():
        context["reason"] = "no_finite_candidates"
        return costs, context
    delta = states[finite][:, overlap, :2] - reference_xy[None, :, :]
    lateral_error = np.abs(-np.sin(reference_heading) * delta[:, :, 0]
                           + np.cos(reference_heading) * delta[:, :, 1])
    heading_delta = states[finite][:, overlap, 2] - reference_heading[None, :]
    heading_error = np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))
    speed_error = np.abs(speeds[finite][:, overlap] - reference_speed[None, :])
    normalized = (np.minimum(lateral_error / 0.5, 1.0)
                  + np.minimum(heading_error / math.radians(15), 1.0)
                  + np.minimum(speed_error / 2.0, 1.0)) / 3.0
    costs[finite] = np.clip(weight * normalized.mean(axis=1), 0.0, weight)
    metrics = [None] * len(states)
    for local_index, candidate_index in enumerate(np.flatnonzero(finite)):
        metrics[candidate_index] = {
            "mean_lateral_deviation_m": float(lateral_error[local_index].mean()),
            "mean_heading_deviation_rad": float(heading_error[local_index].mean()),
            "mean_speed_deviation_mps": float(speed_error[local_index].mean()),
            "cost": float(costs[candidate_index]),
        }
    context.update(applied=True, reason="ok", candidates=metrics)
    return costs, context
