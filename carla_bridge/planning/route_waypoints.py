"""Recover a connected lane sequence from ordered, fixed route coordinates.

Nearest-waypoint projection alone loses the branch identity where junction
connections overlap. Keep several successor chains until later route points
resolve the branch. Evaluation coordinates are never changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _key(wp):
    return (wp.road_id, wp.section_id, wp.lane_id)


def _distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _forward(wp):
    rotation = wp.transform.rotation
    yaw, pitch = math.radians(rotation.yaw), math.radians(getattr(rotation, "pitch", 0))
    return (math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch))


def _advance(previous, target, tolerance):
    """Project onto actual next() branches, with bounded arc-length refinement.

    CARLA's next distance follows OpenDRIVE station, which can differ from
    physical distance on a curved, offset lane. Repeated nearest-map projection
    is deliberately absent: every returned point has previous as its ancestor.
    """
    start = previous.transform.location
    chord = _distance(start, target)
    if chord < 1e-5:
        return [(previous, chord)]
    # Refinement may cross a road seam, but cannot walk an unbounded loop to
    # reconnect a geometrically nearby, unrelated branch.
    upper = max(chord * 2.0 + 0.5, 1.0)
    initial = list(previous.next(max(0.01, chord)))
    found = {}
    for first in initial:
        candidate, station = first, max(0.01, chord)
        for _ in range(5):
            point = candidate.transform.location
            error = _distance(point, target)
            key = _key(candidate)
            if error <= tolerance and (key not in found or error < found[key][1]):
                found[key] = (candidate, error)
            tangent = _forward(candidate)
            along = sum(a * b for a, b in zip(
                (target.x - point.x, target.y - point.y, target.z - point.z), tangent))
            if abs(along) < 0.001:
                break
            probe = list(candidate.next(0.05))
            if not probe:
                break
            # At an overlapping branch entry either derivative is equivalent;
            # once they diverge prefer the same lane to estimate ds/distance.
            local = min(probe, key=lambda wp: (_key(wp) != key, _distance(wp.transform.location, point)))
            scale = _distance(point, local.transform.location) / 0.05
            if not 0.1 <= scale <= 5.0:
                break
            revised = min(upper, max(0.001, station + along / scale))
            if abs(revised - station) < 1e-5:
                break
            options = list(previous.next(revised))
            if not options:
                break
            # Keep the branch being refined. A different key is allowed at a
            # true road seam only through previous.next(), never lateral snap.
            candidate = min(options, key=lambda wp: (
                _key(wp) != key,
                _distance(wp.transform.location, target)))
            station = revised
    return list(found.values())


@dataclass
class _Node:
    waypoint: object
    cost: float
    error: float
    parent: object = None


def resolve_route_waypoints(carla_map, raw_points, *, max_point_error_m=0.75, beam_width=8):
    """Return (one connected waypoint per input point, diagnostic summary).

    A bounded beam retains overlapping junction branches until future fixed
    route points distinguish them. A point farther than the stated tolerance
    from every connected branch raises rather than silently changing route.
    """
    import carla

    if not raw_points:
        raise ValueError("Cannot resolve an empty fixed route")
    if (not math.isfinite(max_point_error_m) or max_point_error_m <= 0
            or not isinstance(beam_width, int) or beam_width < 1):
        raise ValueError("Route recovery requires a positive tolerance and beam width")
    if any(not all(math.isfinite(float(getattr(p, axis))) for axis in ("x", "y", "z"))
           for p in raw_points):
        raise ValueError("Fixed route coordinates must be finite")
    projected = [carla_map.get_waypoint(p, project_to_road=True, lane_type=carla.LaneType.Driving)
                 for p in raw_points]
    first = projected[0]
    if first is None or first.lane_type != carla.LaneType.Driving:
        raise ValueError("Fixed route point 0 has no driving waypoint")
    first_error = _distance(first.transform.location, raw_points[0])
    if first_error > max_point_error_m:
        raise ValueError("Fixed route point 0 is %.3f m from its driving lane (limit %.3f m)" %
                         (first_error, max_point_error_m))
    starts = {_key(first): (first, first_error)}
    # A fixed route can itself begin inside a junction. Enumerate real junction
    # connections in this one case instead of trusting a single nearest ID.
    if first.is_junction and hasattr(first, "get_junction"):
        junction = first.get_junction()
        if junction is not None:
            for entry, _ in junction.get_waypoints(carla.LaneType.Driving):
                for wp, error in _advance(entry, raw_points[0], max_point_error_m):
                    if wp.is_junction and error <= max_point_error_m:
                        key = _key(wp)
                        if key not in starts or error < starts[key][1]:
                            starts[key] = (wp, error)
    beam = [_Node(wp, error * error, error) for wp, error in starts.values()]
    beam = sorted(beam, key=lambda node: node.cost)[:beam_width]
    expanded = 0
    for index, point in enumerate(raw_points[1:], 1):
        choices = {}
        for parent in beam:
            for wp, error in _advance(parent.waypoint, point, max_point_error_m):
                if wp.lane_type != carla.LaneType.Driving:
                    continue
                # Height is part of the full 3-D error. Successor traversal
                # prevents switching to an opposite lane or another deck.
                node = _Node(wp, parent.cost + error * error, error, parent)
                key = _key(wp)
                expanded += 1
                if key not in choices or node.cost < choices[key].cost:
                    choices[key] = node
        if not choices:
            raise ValueError(
                "Fixed route point %d cannot be reached through driving-lane successors "
                "within %.3f m; refusing disconnected nearest-waypoint stitching" %
                (index, max_point_error_m))
        beam = sorted(choices.values(), key=lambda node: node.cost)[:beam_width]
    chain, errors = [], []
    node = min(beam, key=lambda value: value.cost)
    while node is not None:
        chain.append(node.waypoint)
        errors.append(node.error)
        node = node.parent
    chain.reverse()
    errors.reverse()
    mismatches = [i for i, (wp, nearest) in enumerate(zip(chain, projected))
                  if nearest is None or _key(wp) != _key(nearest)]
    return chain, {
        "method": "successor_beam", "point_count": len(chain), "beam_width": beam_width,
        "max_point_error_m": max(errors), "point_error_limit_m": max_point_error_m,
        "reprojection_mismatch_count": len(mismatches),
        "reprojection_mismatch_indices": mismatches[:30], "expanded_candidates": expanded,
        "evaluation_points_unchanged": True,
    }
