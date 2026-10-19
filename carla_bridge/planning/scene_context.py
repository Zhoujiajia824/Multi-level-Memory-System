"""Current, route-relative evidence for advisory planning (never a safety oracle).

Only CARLA's current state and an ordered map route are used. Constant-velocity
conflicts are explicit proxies; they are not ground-truth future trajectories.
The context authorizes semantic advice, and does not veto native PDM proposals.
"""
from __future__ import annotations

import hashlib
import math


def lane_key(wp):
    return "%s:%s:%s" % (wp.road_id, wp.section_id, wp.lane_id) if wp is not None else "unknown"


def project_polyline(x, y, points):
    """Return arc station, lateral distance and tangent of the closest segment."""
    best, arc = None, 0.0
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        u = max(0.0, min(1.0, ((x-a[0])*dx + (y-a[1])*dy) / length**2))
        px, py = a[0]+u*dx, a[1]+u*dy
        distance = math.hypot(x-px, y-py)
        if best is None or distance < best[1]:
            best = (arc+u*length, distance, math.atan2(dy, dx))
        arc += length
    return best


def point_at(station, points):
    arc = 0.0
    for a, b in zip(points, points[1:]):
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        if length > 1e-6 and arc+length >= station:
            u = max(0.0, min(1.0, (station-arc)/length))
            return a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1])
        arc += length
    return points[-1]


def constant_velocity_conflict(ego_station, speed, points, actor_xy, actor_velocity,
                               ego_radius=1.2, actor_radius=1.2, horizon_s=4.0):
    """Whether occupied envelopes meet at the SAME sampled future time.

    Longitudinal extents are deliberately not collapsed into bounding circles:
    that would falsely turn adjacent opposing lanes into conflicts again.
    A stopped ego is evaluated at a modest possible proceeding speed, rather
    than declaring every crossing clear simply because it is currently still.
    """
    travel_speed = max(2.0, speed)
    for step in range(int(horizon_s/.25)+1):
        t = step*.25
        station = ego_station+travel_speed*t
        ex, ey = point_at(station, points)
        tangent = project_polyline(ex, ey, points)[2]
        ax, ay = actor_xy[0]+actor_velocity[0]*t, actor_xy[1]+actor_velocity[1]*t
        dx, dy = ax-ex, ay-ey
        longitudinal = abs(dx*math.cos(tangent)+dy*math.sin(tangent))
        lateral = abs(-dx*math.sin(tangent)+dy*math.cos(tangent))
        if longitudinal < 4.5 and lateral < ego_radius+actor_radius+.15:
            return True, t
    return False, None


def build_scene_context(world, ego, route, sim_time_s, *, reference=None, special_situations=()):
    tf, velocity = ego.get_transform(), ego.get_velocity()
    location = tf.location
    speed = math.hypot(velocity.x, velocity.y)
    heading = math.radians(tf.rotation.yaw)
    carla_map = world.get_map()
    current = carla_map.get_waypoint(location)
    reference = reference or getattr(route, "_route_waypoints", None) or []
    if len(reference) < 2:
        raise ValueError("Selective semantics require an ordered map route")
    nearest = min(max(0, route._nearest_index(location)), len(reference)-1)
    start = max(0, nearest-3)
    local = reference[start:min(len(reference), nearest+51)]
    points = [(wp.transform.location.x, wp.transform.location.y) for wp in local]
    if len(points) < 2:
        local = reference[-2:]
        points = [(wp.transform.location.x, wp.transform.location.y) for wp in local]
    ego_projection = project_polyline(location.x, location.y, points)
    if ego_projection is None:
        raise ValueError("Ordered map route has no distinct local segments")
    ego_station = ego_projection[0]
    route_keys = {lane_key(wp) for wp in local}
    road_limit = float(ego.get_speed_limit())/3.6
    result = {"schema_version": 1, "source": "CARLA_GT_route_constant_velocity_proxy",
              "sim_time_s": float(sim_time_s), "ego_speed_mps": speed,
              "route_progress_m": float(route.distance_completed_m()),
              "in_junction": bool(current and current.is_junction),
              "lane_key": lane_key(current), "navigation": route.nav_instruction(),
              "speed_limit_mps": road_limit, "signals": [], "actors": [],
              "special_situations": sorted(set(special_situations)),
              "blocked": False, "baseline_stop_expected": False,
              "prediction_kind": "constant_velocity_proxy_not_future_GT"}
    weather = world.get_weather()
    if float(weather.fog_density) >= 40:
        result["special_situations"] = sorted(set(result["special_situations"]+["limited_visibility_fog"]))
    for actor in world.get_actors():
        if actor.id == ego.id:
            continue
        kind = actor.type_id
        if kind.startswith("traffic.traffic_light"):
            for index, wp in enumerate(actor.get_stop_waypoints()):
                p = wp.transform.location
                projection = project_polyline(p.x, p.y, points)
                distance = projection[0]-ego_station
                aligned = math.cos(math.radians(wp.transform.rotation.yaw)-projection[2]) > .8
                applicable = (lane_key(wp) in route_keys and aligned
                              and projection[1] <= wp.lane_width/2+.25
                              and abs(p.z-location.z) <= 2 and distance >= -1.0)
                if applicable and distance <= 100:
                    result["signals"].append({"id": "stop_%s_%s" % (actor.id, index),
                        "state": str(actor.state).split(".")[-1], "applicable": True,
                        # The stopping target is the front bumper, not actor origin.
                        "stop_line_distance_m": float(distance-ego.bounding_box.extent.x),
                        "lane_key": lane_key(wp)})
            continue
        if not kind.startswith(("vehicle.", "walker.pedestrian", "static.prop.")):
            continue
        p = actor.get_location()
        distance = math.hypot(p.x-location.x, p.y-location.y)
        if distance > 65 or abs(p.z-location.z) > 2.5:
            continue
        v = actor.get_velocity()
        dx, dy = p.x-location.x, p.y-location.y
        relative_s = dx*math.cos(heading)+dy*math.sin(heading)
        relative_l = dx*math.sin(heading)-dy*math.cos(heading)
        other_wp = carla_map.get_waypoint(p)
        same_lane = (current is not None and other_wp is not None
                     and lane_key(current) == lane_key(other_wp)
                     and math.cos(math.radians(actor.get_transform().rotation.yaw)-heading) > .8)
        projection = project_polyline(p.x, p.y, points)
        along = projection[0]-ego_station
        extent = actor.bounding_box.extent
        conflict, ttc = constant_velocity_conflict(ego_station, speed, points,
                        (p.x, p.y), (v.x, v.y),
                        ego_radius=float(ego.bounding_box.extent.y), actor_radius=float(extent.y))
        stationary = math.hypot(v.x, v.y) < .5
        path_block = stationary and 1 < along < 30 and projection[1] < 1.1+float(extent.y)
        result["blocked"] |= bool(path_block)
        result["actors"].append({"id": int(actor.id), "type": kind,
            "distance_m": distance, "relative_s_m": relative_s, "relative_l_m": relative_l,
            "same_lane": bool(same_lane), "ahead": bool(relative_s > 0),
            "conflict": bool(conflict), "ttc_s": ttc, "stationary": stationary,
            "path_blocked": bool(path_block), "speed_mps": math.hypot(v.x, v.y),
            "route_distance_m": along, "lane_key": lane_key(other_wp)})
    result["actors"].sort(key=lambda a: (not a["conflict"], not a["path_blocked"], a["distance_m"]))
    result["signals"].sort(key=lambda a: a["stop_line_distance_m"])
    # These explain a legitimate stop, not an unconditional command to stop now.
    signal_wait = any(s["state"] in ("Red", "Yellow") and 0 <= s["stop_line_distance_m"] < 12
                      for s in result["signals"])
    # One stationary object can be a parked car, a road barrier or a pedestrian;
    # it is insufficient evidence of a legitimate vehicle queue. This remains
    # a current-state queue proxy, never an instruction to stop or to proceed.
    queue_vehicles = sorted((a for a in result["actors"]
        if a["type"].startswith("vehicle.") and a["same_lane"] and a["ahead"]
        and a["speed_mps"] < 1.0 and 0 < a["route_distance_m"] < 30),
        key=lambda a: a["route_distance_m"])
    queue_wait = bool(queue_vehicles and queue_vehicles[0]["distance_m"] < 12
                     and any(3 <= a["route_distance_m"] - queue_vehicles[0]["route_distance_m"] <= 15
                             for a in queue_vehicles[1:]))
    result["baseline_stop_expected"] = bool(signal_wait or queue_wait)
    # No map/actor identity in the similarity key: topology and navigation generalize.
    signature = (str(getattr(carla_map, "name", "unknown")), result["navigation"],
                 result["in_junction"], round(float(local[0].lane_width), 1),
                 bool(local[0].get_left_lane()), bool(local[0].get_right_lane()))
    result["topology_signature"] = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    return result


def candidate_summary(candidates, baseline_id):
    baseline = next((c for c in candidates if c["id"] == baseline_id), None) or {}
    finite_scores = [c["base_score"] for c in candidates if math.isfinite(c["base_score"])]
    return {"baseline_candidate_id": baseline_id, "baseline_score": baseline.get("base_score"),
            "baseline_progress_m": baseline.get("forward_progress_m"),
            "baseline_speed_mps": baseline.get("terminal_speed_mps"),
            "best_score": max(finite_scores) if finite_scores else None,
            "candidate_count": len(candidates),
            "available_lateral_semantics": sorted({s for c in candidates for s in c.get("lateral_semantics", [])})}
