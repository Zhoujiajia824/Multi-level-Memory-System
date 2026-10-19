"""Bounded lane tracking costs on the actual simulated vehicle footprint.

The upstream score treats a union of driving roads as drivable. That score is
not a lane-centering objective. These adapter costs distinguish otherwise
similar native proposals without turning a lane boundary into an extra veto.
Legal lane changes use their continuous numerical maneuver reference, so the
cost never requires a vehicle to jump between two lane centers instantaneously.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


DEFAULTS = dict(enabled=True, center_weight=.04, boundary_weight=.12,
                center_tolerance_m=.15, boundary_margin_m=.15, horizon_s=2.,
                lane_change_start_cost=.02)


def validate_lane_quality_config(config=None):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("lane_quality must be a mapping")
    values = dict(DEFAULTS, **dict(config or {}))
    if set(values) - set(DEFAULTS):
        raise ValueError("unknown lane_quality configuration field")
    if type(values["enabled"]) is not bool:
        raise ValueError("lane_quality.enabled must be a boolean")
    bounds = dict(center_weight=(0., .16), boundary_weight=(0., .16),
                  center_tolerance_m=(0., .4), boundary_margin_m=(.05, .4),
                  horizon_s=(.2, 4.), lane_change_start_cost=(0., .08))
    for key, (minimum, maximum) in bounds.items():
        value = values[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not minimum <= value <= maximum):
            raise ValueError("lane_quality.%s must be finite and within [%s, %s]" %
                             (key, minimum, maximum))
        values[key] = float(value)
    if values["center_weight"] + values["boundary_weight"] > .16 + 1e-12:
        raise ValueError("lane_quality total cost budget must not exceed 0.16")
    return values


class LaneQualityReference:
    """Resolved map/maneuver geometry, independent from an offset proposal."""

    def __init__(self, points, widths, *, kind="lane_center"):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("lane quality needs finite connected XY reference points")
        if np.isscalar(widths):
            widths = np.full(len(points), float(widths))
        widths = np.asarray(widths, dtype=float)
        if widths.shape != (len(points),) or not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("lane quality needs one positive finite lane width per reference point")
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5]
        points, widths = points[keep], widths[keep]
        if len(points) < 2:
            raise ValueError("lane quality reference has fewer than two distinct points")
        self.points, self.widths, self.kind = points, widths, str(kind)
        self.starts, self.delta = points[:-1], np.diff(points, axis=0)
        self.length2 = np.sum(self.delta ** 2, axis=1)
        self.lengths = np.sqrt(self.length2)

    def lateral(self, coordinates):
        """Signed normal displacement and width at the closest real segment.

        Longitudinal endpoint excess is not treated as a lateral deviation.
        References are local connected corridors and never nearest-road snaps.
        """
        shape = coordinates.shape[:-1]
        points = np.asarray(coordinates, dtype=float).reshape(-1, 2)
        distance, widths = [], []
        for first in range(0, len(points), 512):
            relative = points[first:first + 512, None, :] - self.starts[None, :, :]
            projection = np.clip(np.sum(relative * self.delta[None, :, :], axis=2) /
                                 self.length2[None, :], 0., 1.)
            residual = relative - projection[:, :, None] * self.delta[None, :, :]
            closest = np.argmin(np.sum(residual ** 2, axis=2), axis=1)
            row = np.arange(len(closest))
            vector = relative[row, closest]
            tangent = self.delta[closest] / self.lengths[closest, None]
            distance.append(tangent[:, 0] * vector[:, 1] - tangent[:, 1] * vector[:, 0])
            u = projection[row, closest]
            widths.append((1 - u) * self.widths[closest] + u * self.widths[closest + 1])
        return np.concatenate(distance).reshape(shape), np.concatenate(widths).reshape(shape)


def lane_surface_union(polygons, *, seam_tolerance_m=.001):
    """Join numerical seams between already verified adjacent lane ribbons.

    Independently sampled OpenDRIVE lane edges can differ by float32 rounding.
    Treating those micrometre gaps as outside road creates a false boundary
    along a permitted dashed marking. Morphological closing removes only gaps
    below two millimetres; it does not pad the outer road surface or alter the
    native PDM drivable-area geometry.
    """
    from shapely.ops import unary_union
    if not math.isfinite(seam_tolerance_m) or not 0 < seam_tolerance_m <= .005:
        raise ValueError("lane boundary seam tolerance must be within (0, .005] metres")
    pieces = [polygon for polygon in polygons if polygon is not None and not polygon.is_empty]
    if not pieces:
        return None
    joined = unary_union(pieces)
    return joined.buffer(seam_tolerance_m, join_style=2).buffer(-seam_tolerance_m, join_style=2)


def footprint_lane_quality(corners, reference, vehicle_width, interval_s, *, config=None,
                           lane_change_start=False, boundary_polygon=None):
    """Charge moving and stopped native rollouts by the same time average.

    Input is [time, four footprint corners, XY] from PDMScorer, rather than
    generated path labels. The measured t=0 footprint is omitted equally for
    every proposal; every future tick in the two-second control neighborhood
    receives equal weight. Costs saturate at the configured finite budget.
    """
    cfg = validate_lane_quality_config(config)
    if not cfg["enabled"]:
        return 0., dict(enabled=False, cost=0., reference_kind=reference.kind)
    corners = np.asarray(corners, dtype=float)
    if corners.ndim != 3 or corners.shape[1:] != (4, 2) or len(corners) < 2 or not np.isfinite(corners).all():
        raise ValueError("lane quality requires finite simulated four-corner footprints")
    if not math.isfinite(vehicle_width) or vehicle_width <= 0 or not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("lane quality requires positive finite width and sample interval")
    stop = min(len(corners), 1 + max(1, int(round(cfg["horizon_s"] / interval_s))))
    future = corners[1:stop]
    centers = np.mean(future, axis=1)
    deviation, center_widths = reference.lateral(centers)
    # Available body-center room varies with the real lane width. A small
    # tolerance lets curved-road tracking compensate for curvature naturally.
    room = np.maximum(.25, center_widths / 2 - vehicle_width / 2 - cfg["center_tolerance_m"])
    normalized = np.clip((np.abs(deviation) - cfg["center_tolerance_m"]) / room, 0., 1.)
    center_cost = cfg["center_weight"] * float(np.mean(normalized ** 2))
    if boundary_polygon is not None:
        # During a legal dashed-line transition, the shared lane marking is
        # inside the union of the two actual lane surfaces. A moving narrow
        # tube around the desired curve invents a forbidden boundary there.
        # Keep tracking error and real outside clearance as separate terms.
        from shapely import points, distance, covers
        actual_corners = points(future)
        inside = covers(boundary_polygon, actual_corners)
        clearance = distance(boundary_polygon.boundary, actual_corners) * np.where(inside, 1., -1.)
        boundary_kind = "actual_source_target_lane_union"
    else:
        corner_deviation, corner_widths = reference.lateral(future)
        clearance = corner_widths / 2 - np.abs(corner_deviation)
        boundary_kind = "resolved_lane_width"
    closest_clearance = np.min(clearance, axis=1)
    proximity = np.clip((cfg["boundary_margin_m"] - closest_clearance) / cfg["boundary_margin_m"], 0., 1.)
    boundary_cost = cfg["boundary_weight"] * float(np.mean(proximity ** 2))
    start_cost = cfg["lane_change_start_cost"] if lane_change_start else 0.
    unclipped_cost = float(center_cost + boundary_cost + start_cost)
    cost = min(.16, unclipped_cost)
    return cost, dict(enabled=True, reference_kind=reference.kind, cost=cost,
        boundary_reference_kind=boundary_kind,
        center_cost=center_cost, boundary_cost=boundary_cost,
        lane_change_start_cost=start_cost, cost_budget_saturated=unclipped_cost > .16,
        center_error_mean_m=float(np.mean(np.abs(deviation))),
        center_error_max_m=float(np.max(np.abs(deviation))),
        minimum_footprint_clearance_m=float(np.min(closest_clearance)),
        footprint_boundary_crossing_fraction=float(np.mean(closest_clearance < 0.)),
        evaluated_horizon_s=float((stop - 1) * interval_s),
        center_tolerance_m=cfg["center_tolerance_m"], boundary_margin_m=cfg["boundary_margin_m"],
        hard_veto_applied=False, actual_simulated_footprint=True)
