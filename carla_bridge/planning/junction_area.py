"""Actual junction lane surfaces for PDM's drivable-area map.

The reference route determines the junctions to inspect. Their full driving
surfaces include crossing/overlapping connections, while route direction and
lane-change permissions remain the planner's separate responsibilities.
"""
from __future__ import annotations

from collections import defaultdict, OrderedDict
import math


def _key(wp):
    return "%s:%s:%s" % (wp.road_id, wp.section_id, wp.lane_id)


def _distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _sample_junction(junction, distance, budget):
    import carla

    segments, count = [], 0
    pairs = sorted(junction.get_waypoints(carla.LaneType.Driving),
                   key=lambda pair: (_key(pair[0]), float(pair[0].s)))
    for entry, exit in pairs:
        key = _key(entry)
        if (key != _key(exit) or entry.lane_type != carla.LaneType.Driving
                or exit.lane_type != carla.LaneType.Driving):
            raise ValueError("Junction lane endpoints must describe one driving lane")
        current = entry
        # s is the road station; its direction depends on the lane sign. Use
        # next() to preserve that direction instead of assuming increasing s.
        remaining = abs(float(exit.s) - float(current.s))
        while remaining > 1e-5:
            if count >= budget:
                raise ValueError("Junction %s exceeds the %d-segment geometry budget" % (junction.id, budget))
            count += 1
            if remaining <= distance + 1e-5:
                following = exit  # exact endpoint supplied by get_waypoints
            else:
                options = [wp for wp in current.next(distance)
                           if _key(wp) == key and wp.is_junction
                           and wp.lane_type == carla.LaneType.Driving]
                if not options:
                    raise ValueError("Junction %s lane %s ends before its mapped exit" % (junction.id, key))
                following = min(options, key=lambda wp: abs(float(exit.s) - float(wp.s)))
            left = abs(float(exit.s) - float(following.s))
            separation = _distance(current.transform.location, following.transform.location)
            if (left >= remaining - 1e-6 or separation > distance * 5 + 0.05
                    or not math.isfinite(separation)):
                raise ValueError("Junction %s lane %s has an invalid successor" % (junction.id, key))
            if separation > 1e-5:
                segments.append((key, current, following))
            current, remaining = following, left
    return segments


def junction_driving_polygons(reference_waypoints, *, cache=None, sample_distance_m=0.5,
                              height_tolerance_m=1.0, max_segments_per_junction=20000):
    """Return (lane-key -> polygon, diagnostics), with no invented road area.

    ``cache`` belongs to one planner/map and stores unfiltered lane samples.
    Each call filters against the local route elevation, so cache reuse cannot
    accidentally include a different bridge deck. Unsupported test waypoints
    without get_junction() simply supply no additional geometry.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if (not math.isfinite(sample_distance_m) or not 0 < sample_distance_m <= 1
            or not math.isfinite(height_tolerance_m) or height_tolerance_m <= 0
            or not isinstance(max_segments_per_junction, int) or max_segments_per_junction < 1):
        raise ValueError("Invalid junction sampling, height tolerance, or segment budget")
    references = list(reference_waypoints)
    junctions = {}
    for wp in references:
        if wp.is_junction and hasattr(wp, "get_junction"):
            junction = wp.get_junction()
            if junction is not None:
                junctions[junction.id] = junction
    reference_coords = [(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z)
                        for wp in references]
    cache = {} if cache is None else cache
    polygons = defaultdict(list)
    accepted, rejected, sampled, cache_hits = 0, 0, 0, 0
    for junction_id, junction in sorted(junctions.items()):
        cache_key = (junction_id, sample_distance_m, max_segments_per_junction)
        if cache_key in cache:
            cached = cache[cache_key]
            cache_hits += 1
        else:
            segments = _sample_junction(junction, sample_distance_m, max_segments_per_junction)
            prepared = []
            for key, a, b in segments:
                corners = []
                for wp, sign in ((a, 1), (b, 1), (b, -1), (a, -1)):
                    p = wp.transform.location
                    yaw = -math.radians(wp.transform.rotation.yaw)
                    half_width = float(wp.lane_width) / 2
                    if not math.isfinite(half_width) or half_width <= 0:
                        raise ValueError("Junction lane %s has an invalid lane width" % key)
                    corners.append((p.x - sign * half_width * math.sin(yaw),
                                    -p.y + sign * half_width * math.cos(yaw)))
                polygon = Polygon(corners)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # repair only; never expand lanes
                a, b = a.transform.location, b.transform.location
                prepared.append((key, (a.x, a.y, a.z), (b.x, b.y, b.z), polygon))
            cached = {"segments": prepared, "layers": OrderedDict()}
            cache[cache_key] = cached
        segments = cached["segments"]
        sampled += len(segments)
        selected = []
        for i, (key, a, b, polygon) in enumerate(segments):
            # Both endpoint elevations must agree with nearby route geometry.
            # Junction ID alone is insufficient for multilayer road networks.
            same_deck = True
            for point in (a, b):
                nearby = min(reference_coords, key=lambda ref:
                             (ref[0] - point[0]) ** 2 + (ref[1] - point[1]) ** 2)
                if abs(point[2] - nearby[2]) > height_tolerance_m:
                    same_deck = False
                    break
            if not same_deck:
                rejected += 1
                continue
            if not polygon.is_empty:
                selected.append(i)
                accepted += 1
        # Reuse unions when the current route is on the same deck. At most four
        # deck variants are retained per junction, keeping the cache bounded.
        layer_key = tuple(selected)
        layers = cached["layers"]
        if layer_key not in layers:
            grouped = defaultdict(list)
            for i in selected:
                key, _, _, polygon = segments[i]
                grouped[key].append(polygon)
            layers[layer_key] = {key: unary_union(parts) for key, parts in grouped.items()}
            if len(layers) > 4:
                layers.popitem(last=False)
        layers.move_to_end(layer_key)
        for key, polygon in layers[layer_key].items():
            polygons[key].append(polygon)
    result = {key: unary_union(parts) for key, parts in polygons.items()}
    return result, {
        "method": "actual_junction_driving_lane_ribbons", "junction_ids": sorted(junctions),
        "lane_count": len(result), "sampled_segments": sampled, "accepted_segments": accepted,
        "excluded_height_segments": rejected, "sample_distance_m": sample_distance_m,
        "height_tolerance_m": height_tolerance_m, "cache_hits": cache_hits,
    }
