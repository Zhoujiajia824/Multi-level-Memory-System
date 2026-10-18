"""Dense, real lane surfaces at ordinary-road/junction seams."""
from collections import defaultdict
import math


def road_driving_polygons(carla_map, seeds, *, padding_m=3.0, step_m=.5, cache=None):
    """Sample actual OpenDRIVE lanes independently of route-point adjacency.

    An adjacent exit lane can start before its first sparse reference sample.
    Joining only neighbours at matching route indices omits that real surface.
    Extend station *sampling* around each observed lane interval, never a road
    polygon: invalid stations, other sections and non-driving lanes are absent.
    """
    import carla
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if not hasattr(carla_map, "get_waypoint_xodr"):
        return {}, {"method": "unavailable_without_xodr_lookup", "lane_count": 0}
    cache = {} if cache is None else cache
    if "bounds" not in cache:
        # CARLA get_topology() returns lane entries and entries of their next
        # roads, not both ends of every same-road lane. Its equal station values
        # cannot delimit a road. Read the authoritative station intervals.
        from xml.etree import ElementTree
        bounds = {}
        if hasattr(carla_map, "to_opendrive"):
            for road in ElementTree.fromstring(carla_map.to_opendrive()).findall("road"):
                sections = road.findall("lanes/laneSection")
                for index, section in enumerate(sections):
                    first = float(section.attrib["s"])
                    last = (float(sections[index + 1].attrib["s"]) if index + 1 < len(sections)
                            else float(road.attrib["length"]))
                    for lane in section.findall("./*/lane"):
                        if lane.attrib.get("type") == "driving":
                            bounds[(int(road.attrib["id"]), index, int(lane.attrib["id"]))] = (first, last)
        cache["bounds"] = bounds
        cache["polygons"] = {}
    groups = defaultdict(list)
    for wp in seeds:
        if not wp.is_junction and wp.lane_type == carla.LaneType.Driving:
            groups[(wp.road_id, wp.section_id, wp.lane_id)].append(wp)
    result, count = {}, 0
    for (road, section, lane), samples in groups.items():
        first = max(0.0, math.floor((min(wp.s for wp in samples) - padding_m) / step_m) * step_m)
        last = math.ceil((max(wp.s for wp in samples) + padding_m) / step_m) * step_m
        bounds = cache["bounds"].get((road, section, lane))
        if bounds is not None:
            first, last = max(first, bounds[0]), min(last, bounds[1])
            if first == bounds[0]:
                # At a nonzero lane-section start, rounding down can select
                # the preceding section and discard another half-metre strip.
                rounded = np.float32(first)
                first = float(np.nextafter(rounded, np.float32(math.inf))
                              if float(rounded) < first else rounded)
            # CARLA converts s to float32 and rejects s >= road.length.
            # Use the closest representable station inside the actual lane,
            # instead of dropping its final grid-to-end segment entirely.
            if last == bounds[1]:
                rounded = np.float32(last)
                last = float(np.nextafter(rounded, np.float32(-math.inf))
                             if float(rounded) >= last else rounded)
        cache_key = (road, section, lane, first, last, step_m)
        stored = cache["polygons"].get(cache_key)
        if stored is not None:
            result[f"{road}:{section}:{lane}"], sampled_count = stored
            count += sampled_count
            continue
        previous, parts = None, []
        # Include the actual OpenDRIVE endpoints: stopping at the last .5-metre
        # grid sample leaves a thin false off-road strip at every road seam.
        stations = [first + step_m * i for i in range(int(max(0, (last - first) // step_m)) + 1)] + [last]
        for station in sorted(set(stations)):
            wp = carla_map.get_waypoint_xodr(road, lane, station)
            if (wp is None or wp.section_id != section or wp.is_junction
                    or wp.lane_type != carla.LaneType.Driving):
                previous = None
                continue
            if previous is not None:
                a, b = previous.transform.location, wp.transform.location
                # No bridge/deck stitching or distant jumps in malformed maps.
                if math.hypot(a.x - b.x, a.y - b.y) < 5 * step_m and abs(a.z - b.z) < 1.0:
                    corners = []
                    for point, sign in ((previous, 1), (wp, 1), (wp, -1), (previous, -1)):
                        p, yaw = point.transform.location, -math.radians(point.transform.rotation.yaw)
                        half = float(point.lane_width) / 2
                        corners.append((p.x - sign * half * math.sin(yaw),
                                        -p.y + sign * half * math.cos(yaw)))
                    polygon = Polygon(corners)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if not polygon.is_empty:
                        parts.append(polygon)
                        count += 1
            previous = wp
        if parts:
            polygon = result[f"{road}:{section}:{lane}"] = unary_union(parts)
            cache["polygons"][cache_key] = (polygon, len(parts))
            if len(cache["polygons"]) > 256:
                cache["polygons"].pop(next(iter(cache["polygons"])))
    return result, {"method": "actual_xodr_station_lane_ribbons", "lane_count": len(result),
                    "sampled_segments": count, "station_step_m": step_m,
                    "station_sampling_padding_m": padding_m, "polygon_expansion_m": 0.0,
                    "exact_opendrive_endpoints": bool(cache["bounds"])}
