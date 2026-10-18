"""Current map/route facts for semantic decisions, separate from memory.

Nearest-road identifiers are ambiguous inside junctions. They are evidence
about route preparation, never a right-of-way or traffic clearance decision.
"""
import math


def _key(wp):
    return "%s:%s:%s" % (wp.road_id, wp.section_id, wp.lane_id)


def build_navigation_context(route, ego, sim_time_s):
    result = {"source": "CARLA_GT_and_ordered_route", "snapshot_time_s": float(sim_time_s)}
    reference = getattr(route, "_route_waypoints", None)
    if not reference:
        return result
    location = ego.get_location()
    index = route._nearest_index(location)
    index = min(range(max(0, index - 2), min(len(reference), index + 6)),
                key=lambda i: reference[i].transform.location.distance(location))
    target = reference[index]
    current = route.map.get_waypoint(location)
    result["reference_lane_key"] = _key(target)
    result["current_side"] = "unknown"
    if current is not None:
        result["ego_lane_key"] = _key(current)
        if not current.is_junction and not target.is_junction:
            for side, wp in (("center", target), ("left", target.get_left_lane()),
                             ("right", target.get_right_lane())):
                if wp is not None and _key(wp) == _key(current):
                    result["current_side"] = side
    distance = 0.0
    for i in range(index, len(reference)):
        if i > index:
            distance += reference[i - 1].transform.location.distance(reference[i].transform.location)
        if reference[i].is_junction:
            if math.isfinite(distance):
                result["route_distance_to_junction_m"] = round(distance, 2)
            break
    return result
