"""Bounded current-lane spatial proposals for a diagnosed planning deadlock.

This module only expands geometry. It neither chooses a trajectory nor changes
the upstream simulator/scorer. The caller must regenerate and score the whole
pool on the current observation before any new path can be executed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def validate_recovery_config(config=None):
    """Validate the finite search budget; retain caller-owned trigger settings."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("recovery configuration must be a mapping")
    values = dict(config or {})
    enabled = values.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("recovery.enabled must be a boolean")
    maximum = values.get("max_paths", 15)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 15:
        raise ValueError("recovery.max_paths must be an integer in [1, 15]")
    lengths = values.get("transition_lengths_m", [2., 3., 4.])
    if not isinstance(lengths, (list, tuple)) or not 1 <= len(lengths) <= 3:
        raise ValueError("recovery.transition_lengths_m must contain one to three lengths")
    normalized = []
    for length in lengths:
        if (isinstance(length, bool) or not isinstance(length, (int, float))
                or not 1. <= length <= 6. or not math.isfinite(length)):
            raise ValueError("recovery transition lengths must be finite metres in [1, 6]")
        if float(length) in normalized:
            raise ValueError("recovery transition lengths must be distinct")
        normalized.append(float(length))
    offset = values.get("max_offset_m", .5)
    if (isinstance(offset, bool) or not isinstance(offset, (int, float))
            or not 0. <= offset <= .5 or not math.isfinite(offset)):
        raise ValueError("recovery.max_offset_m must be finite and in [0, 0.5]")
    clearance = values.get("side_clearance_m", .1)
    if (isinstance(clearance, bool) or not isinstance(clearance, (int, float))
            or not .05 <= clearance <= .5 or not math.isfinite(clearance)):
        raise ValueError("recovery.side_clearance_m must be finite and in [0.05, 0.5]")
    values.update(enabled=enabled, max_paths=maximum, transition_lengths_m=normalized,
                  max_offset_m=float(offset), side_clearance_m=float(clearance))
    return values


def propose_recovery_paths(base_paths, base_metadata, reference_waypoints, ego_state,
                           api, vehicle_parameters, *, config=None):
    """Return the unchanged original pool followed by bounded current-lane paths.

    The existing KEEP_LANE metadata supplies the already resolved physical lane;
    no nearest-map snap, lane change or model-generated XY is introduced. Local
    topology is checked again. In particular a cut current corridor is not used
    to invent an extension or replace an existing route-compatible lane return.
    """
    cfg = validate_recovery_config(config)
    paths, metadata = list(base_paths), list(base_metadata)
    if len(paths) != len(metadata):
        raise ValueError("recovery requires one metadata entry per original path")
    diagnostics = {"enabled": cfg["enabled"], "recipe": "short_pose_rejoin",
                   "original_path_count": len(paths), "added_path_count": 0,
                   "max_paths": cfg["max_paths"], "attempted_path_count": 0,
                   "native_resimulation_required": True, "score_relaxation_applied": False,
                   "added_ids": [], "excluded": []}
    if not cfg["enabled"]:
        diagnostics["reason"] = "disabled"
        return paths, metadata, diagnostics
    current = [entry for entry in metadata if isinstance(entry, Mapping)
               and not entry.get("recovery") and entry.get("lane_offset_m", 0.) == 0.
               and "KEEP_LANE" in entry.get("semantics", [entry.get("label")])
               and entry.get("physical_lane_side") in ("center", "left", "right")]
    sides = {entry["physical_lane_side"] for entry in current}
    if len(sides) != 1:
        diagnostics["reason"] = "current_lane_not_uniquely_resolved"
        return paths, metadata, diagnostics
    side = next(iter(sides))
    source = next(entry for entry in current if entry["physical_lane_side"] == side)
    reference = list(reference_waypoints)
    if len(reference) < 2:
        diagnostics["reason"] = "insufficient_reference_waypoints"
        return paths, metadata, diagnostics

    # Imported lazily so the adapter may call this module without a cycle.
    from .pdm_planner import (_lane_corridor, _lane_path_points, _lane_offset_points,
                              _smooth_path, _xy)
    rear = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y], dtype=float)
    heading = float(ego_state.rear_axle.heading)
    if not np.isfinite(rear).all() or not math.isfinite(heading):
        raise ValueError("recovery requires a finite measured rear-axle pose")
    coordinates = np.array([_xy(wp) for wp in reference])
    if not np.isfinite(coordinates).all():
        raise ValueError("recovery requires finite map geometry")
    closest = int(np.argmin(np.linalg.norm(coordinates - rear, axis=1)))
    progress = np.r_[0., np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))]
    corridor, cut = _lane_corridor(reference, closest, side, progress)
    diagnostics.update(physical_lane_side=side, closest_reference_index=closest,
                       corridor_cut=cut)
    if not corridor or cut is not None:
        diagnostics["reason"] = "current_lane_corridor_not_continuous"
        return paths, metadata, diagnostics
    width = float(vehicle_parameters.width)
    widths = [float(wp.lane_width) for wp in corridor]
    if not math.isfinite(width) or width <= 0 or any(not math.isfinite(w) or w <= 0 for w in widths):
        raise ValueError("recovery requires positive finite vehicle and lane widths")
    limit = max(0., min(cfg["max_offset_m"], (min(widths) - width) / 2 - cfg["side_clearance_m"]))
    # IDs encode centimetres, so values must be quantized before deduplication.
    limit = math.floor(limit * 100 + 1e-9) / 100
    magnitudes = sorted({min(.25, limit), min(.5, limit)} - {0.}) if limit >= .05 else []
    offsets = [0.] + [sign * magnitude for magnitude in magnitudes for sign in (-1, 1)]
    diagnostics.update(within_lane_offset_limit_m=limit, offsets_m=offsets,
                       transition_lengths_m=cfg["transition_lengths_m"])
    try:
        points = _lane_path_points(corridor)
    except ValueError as exc:
        diagnostics.update(reason="insufficient_current_lane_geometry", error=str(exc))
        return paths, metadata, diagnostics
    identifiers = {entry.get("id") for entry in metadata if isinstance(entry, Mapping)}
    # Offset-major ordering covers all short recovery lengths before exploring
    # another displacement when a smaller explicit compute budget is selected.
    for offset in offsets:
        target = points if offset == 0. else _lane_offset_points(points, offset, api)
        for transition in cfg["transition_lengths_m"]:
            if diagnostics["added_path_count"] >= cfg["max_paths"]:
                break
            identifier = "recovery@%s@t%.2f@o%+.2f" % (side, transition, offset)
            if identifier in identifiers:
                diagnostics["excluded"].append({"id": identifier, "reason": "duplicate_id"})
                continue
            diagnostics["attempted_path_count"] += 1
            try:
                path = _smooth_path(target, rear, heading, api, transition)
            except ValueError as exc:
                diagnostics["excluded"].append({"id": identifier, "reason": "insufficient_path_ahead",
                                                 "error": str(exc)})
                continue
            # Exact measured pose is enforced by the shared pose-rejoin helper.
            paths.append(path)
            metadata.append(dict(source, id=identifier, lane_offset_m=float(offset),
                                 recovery=True, recovery_recipe="short_pose_rejoin",
                                 transition_length_m=float(transition)))
            identifiers.add(identifier)
            diagnostics["added_ids"].append(identifier)
            diagnostics["added_path_count"] += 1
        if diagnostics["added_path_count"] >= cfg["max_paths"]:
            break
    diagnostics["reason"] = "expanded" if diagnostics["added_path_count"] else "no_additional_path"
    return paths, metadata, diagnostics
