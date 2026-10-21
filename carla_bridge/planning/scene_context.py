"""Current, route-relative evidence for advisory planning (never a safety oracle).

Only CARLA's current/past state and an ordered map route are used. Connected-road
and observed-motion forecasts are proxies, never ground-truth future trajectories.
The context authorizes semantic advice, and does not veto native PDM proposals.
"""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_right


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


def _predicted_body_at(actor_prediction, time_s, actor_xy, actor_velocity, actor_yaw=0.):
    """Interpolate one explicit shared prediction, or preserve legacy CV calls."""
    if actor_prediction is None:
        return (actor_xy[0]+actor_velocity[0]*time_s,
                actor_xy[1]+actor_velocity[1]*time_s, actor_yaw, 0.)
    times = actor_prediction["times_s"]
    centers = actor_prediction["centers_world_xy_m"]
    yaws = actor_prediction["body_yaws_rad"]
    margins = actor_prediction.get("uncertainty_margin_m", [0.]*len(times))
    if not times or not (len(times) == len(centers) == len(yaws) == len(margins)):
        raise ValueError("actor_prediction samples must have matching times/centers/headings")
    right = min(len(times)-1, bisect_right(times, time_s))
    left = max(0, right-1)
    span = float(times[right]-times[left])
    fraction = max(0., min(1., (time_s-times[left])/span)) if span > 0 else 0.
    yaw_step = math.atan2(math.sin(yaws[right]-yaws[left]), math.cos(yaws[right]-yaws[left]))
    return (centers[left][0]+fraction*(centers[right][0]-centers[left][0]),
            centers[left][1]+fraction*(centers[right][1]-centers[left][1]),
            yaws[left]+fraction*yaw_step,
            margins[left]+fraction*(margins[right]-margins[left]))


def constant_velocity_conflict(ego_station, speed, points, actor_xy, actor_velocity,
                               ego_radius=1.2, actor_radius=1.2, horizon_s=4.0,
                               *, actor_prediction=None):
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
        ax, ay, _, uncertainty = _predicted_body_at(actor_prediction, t, actor_xy, actor_velocity)
        dx, dy = ax-ex, ay-ey
        longitudinal = abs(dx*math.cos(tangent)+dy*math.sin(tangent))
        lateral = abs(-dx*math.sin(tangent)+dy*math.cos(tangent))
        if longitudinal < 4.5+uncertainty and lateral < ego_radius+actor_radius+.15+uncertainty:
            return True, t
    return False, None


def occupied_body_conflict(ego_xy, ego_velocity, ego_yaw, ego_extent,
                          actor_xy, actor_velocity, actor_yaw, actor_extent,
                          *, actor_prediction=None):
    """Two-second body overlap using the same actor forecast as the planner.

    The route forecast above deliberately asks what could happen IF a stopped
    car proceeded. That is useful before entering a junction, but it must not
    turn normal cross traffic beyond a red stop line into a threat to a car
    that is actually waiting. This separate SAT proxy tests the occupied body.
    It is neither a safety veto nor privileged knowledge of future motion.
    """
    ex, ey = math.cos(ego_yaw), math.sin(ego_yaw)
    for step in range(9):
        t = step * .25
        px, py, predicted_yaw, uncertainty = _predicted_body_at(
            actor_prediction, t, actor_xy, actor_velocity, actor_yaw)
        ax, ay = math.cos(predicted_yaw), math.sin(predicted_yaw)
        axes = ((ex, ey), (-ey, ex), (ax, ay), (-ay, ax))
        dx = px - ego_xy[0] - ego_velocity[0]*t
        dy = py - ego_xy[1] - ego_velocity[1]*t
        if all(abs(dx*ux+dy*uy) <= (
                ego_extent[0]*abs(ex*ux+ey*uy) + ego_extent[1]*abs(-ey*ux+ex*uy)
                + actor_extent[0]*abs(ax*ux+ay*uy) + actor_extent[1]*abs(-ay*ux+ax*uy)
                + .15 + uncertainty) for ux, uy in axes):
            return True, t
    return False, None


def _route_lane_at(station, local):
    """Actual lane identities bracketing a projection on the ordered route."""
    arc = 0.0
    for a, b in zip(local, local[1:]):
        pa, pb = a.transform.location, b.transform.location
        length = math.hypot(pb.x-pa.x, pb.y-pa.y)
        if length > 1e-6 and arc+length >= station-1e-6:
            return {lane_key(a), lane_key(b)}, min(float(a.lane_width), float(b.lane_width))
        arc += length
    return {lane_key(local[-1])}, float(local[-1].lane_width)


def _body_pose(actor, pose):
    """CARLA actor origins and bounding-box centers need not coincide."""
    yaw = math.radians(pose.rotation.yaw)
    offset = getattr(actor.bounding_box, "location", None)
    dx, dy = (float(offset.x), float(offset.y)) if offset is not None else (0.0, 0.0)
    box_rotation = getattr(actor.bounding_box, "rotation", None)
    box_yaw = math.radians(box_rotation.yaw) if box_rotation is not None else 0.0
    return ((pose.location.x + dx*math.cos(yaw)-dy*math.sin(yaw),
             pose.location.y + dx*math.sin(yaw)+dy*math.cos(yaw)), yaw+box_yaw)


def _traffic_wait(result, ego_half_length):
    """Explain current traffic constraints; an isolated stopped car is not a queue."""
    speed = result["ego_speed_mps"]
    signals = [s for s in result["signals"] if s["state"].upper() in ("RED", "YELLOW")
               and 0 <= s["stop_line_distance_m"] <= 100]
    leads = sorted((a for a in result["actors"] if a["type"].startswith("vehicle.")
                    and a["same_lane"] and 0 < a["route_distance_m"] < 65),
                   key=lambda a: a["route_distance_m"])
    stopping_range = max(12.0, speed*2.0 + speed*speed/5.0)
    wait = {"active": False, "reason": "none", "signal_id": None,
            "lead_actor_id": leads[0]["id"] if leads else None, "actor_ids": [],
            "stop_line_distance_m": None}
    for signal in signals:
        stop_station = signal["stop_line_distance_m"] + ego_half_length
        # Follow a compact chain back from the stop line. A parked car far
        # before a red signal cannot acquire queue status merely from its ID.
        queue = []
        tail = stop_station
        for actor in reversed(leads):
            front = actor["route_distance_m"] + actor["length_m"]/2
            if actor["speed_mps"] <= 1.0 and -1 <= tail-front <= 14:
                queue.append(actor)
                tail = actor["route_distance_m"] - actor["length_m"]/2
        queue.reverse()
        near_queue = bool(queue and queue[0]["route_distance_m"] <= max(15.0, speed*2+5))
        if signal["stop_line_distance_m"] <= stopping_range or near_queue:
            wait.update(active=True, reason="signal_queue" if near_queue else "traffic_signal",
                        signal_id=signal["id"], actor_ids=[a["id"] for a in queue],
                        stop_line_distance_m=signal["stop_line_distance_m"])
            break
    # Moving same-route following is ordinary even when IDM predicts little
    # forward progress. Do not label a static single vehicle a legitimate wait.
    following = bool(leads and .2 < leads[0]["speed_mps"]
                     and leads[0]["route_distance_m"] < max(15.0, speed*2+5))
    result["traffic_wait"] = wait
    result["routine_following"] = bool(following and not wait["active"])
    result["baseline_stop_expected"] = bool(wait["active"])
    wait_ids = set(wait["actor_ids"])
    for actor in result["actors"]:
        actor["routine_wait"] = actor["id"] in wait_ids
        actor["blocking_kind"] = ("signal_queue" if actor["routine_wait"] else
            "moving_lead" if actor["same_lane"] and actor["speed_mps"] > .2 else
            "stationary_obstacle" if actor["path_blocked"] else
            "crossing_actor" if not actor["same_lane"] and actor["conflict"] else "none")
    # Preserve each actor's raw geometric path_blocked evidence. The scene-level
    # flag describes unexpected blockage, separate from a grounded signal queue.
    result["blocked"] = any(a["path_blocked"] and not a["routine_wait"]
                            and a["blocking_kind"] != "moving_lead"
                            for a in result["actors"])


def _overtaking_context(result, current, local, ego, carla_map, ego_route_aligned):
    """Expose measured efficiency constraints, separate from free-road failure.

    Availability is a current target-lane gap proxy, not permission to execute a
    lane change. The numerical planner must still verify the entire transition,
    return corridor, reachable steering and future occupied vehicle envelopes.
    A slow actor is not an exceptional case solely because it is visible.
    """
    result["overtaking_opportunities"] = []
    result["slow_obstacle"] = None
    if not ego_route_aligned or current is None or result["in_junction"]:
        return
    from .lane_maneuver import lane_change_permitted, _crosswalk_near
    if not hasattr(current, "lane_type") or not hasattr(current, "lane_change"):
        return  # Older context providers cannot invent map permission.
    actors = result["actors"]
    speed = result["ego_speed_mps"]
    for side in ("left", "right"):
        target = getattr(current, "get_" + side + "_lane")()
        if not lane_change_permitted(current, target, side):
            continue
        target_points = []
        for wp in local:
            adjacent = getattr(wp, "get_" + side + "_lane")()
            if adjacent is None or not lane_change_permitted(wp, adjacent, side):
                break
            p = adjacent.transform.location
            target_points.append((float(p.x), float(p.y)))
        if len(target_points) < 2:
            continue
        conflicts = []
        probe_speed = max(speed, min(5.0, result["speed_limit_mps"]))
        target_keys = {lane_key(getattr(wp, "get_"+side+"_lane")()) for wp in local}
        for actor in actors:
            position = actor.get("position_xy_m")
            if not isinstance(position, list) or len(position) != 2:
                continue
            projected = project_polyline(*position, target_points)
            # Projection clamps at the route window's first waypoint. A rear
            # car must still occupy its map lane when outside that window.
            on_target_map_lane = actor["lane_key"] in target_keys
            if (not on_target_map_lane and
                    projected[1] > float(ego.bounding_box.extent.y)+actor["width_m"]/2+.35):
                continue
            distance = (actor["relative_s_m"] if actor["relative_s_m"] < 0
                        and actor["route_distance_m"] >= 0 else actor["route_distance_m"])
            free_gap = abs(distance)-float(ego.bounding_box.extent.x)-actor["length_m"]/2
            along = actor["along_speed_mps"]
            required = (max(8., 3*max(0., along-speed)+5.) if distance < 0 else
                        max(6., probe_speed*1.5, 3*max(0., probe_speed-along)))
            if free_gap < required:
                conflicts.append({"actor_id": actor["id"], "gap_m": free_gap,
                                  "required_m": required, "kind": "rear" if distance < 0 else "front"})
        result["overtaking_opportunities"].append({"side": side,
            "target_lane_key": lane_key(target), "current_gap_clear": not conflicts,
            "conflicts": conflicts, "source": "current_map_permission_and_gap_proxy",
            "execution_authorized": False})
    if result["baseline_stop_expected"]:
        return
    limit = min(8., result["speed_limit_mps"])
    possible = [a for a in actors if a["type"].startswith(("vehicle.", "walker."))
        and 1 < a["route_distance_m"] < 45
        and a["route_lateral_distance_m"] < min(float(current.lane_width)/2-.1,
            float(ego.bounding_box.extent.y)+a["width_m"]/2+.15)
        and -.3 <= a["along_speed_mps"] < .6*limit
        and abs(a["cross_speed_mps"]) < .35
        and (a["route_heading_aligned"] or a["speed_mps"] < .3)
        and a["route_distance_m"]-float(ego.bounding_box.extent.x)-a["length_m"]/2 > .5]
    if not possible:
        return
    blocker = min(possible, key=lambda a: a["route_distance_m"])
    if blocker["type"].startswith("walker.") and _crosswalk_near(
            carla_map, (blocker["position_xy_m"][0], -blocker["position_xy_m"][1])):
        return
    sides = [item["side"] for item in result["overtaking_opportunities"] if item["current_gap_clear"]]
    result["slow_obstacle"] = {"actor_id": blocker["id"], "type": blocker["type"],
        "route_distance_m": blocker["route_distance_m"],
        "clearance_m": blocker["route_distance_m"]-float(ego.bounding_box.extent.x)-blocker["length_m"]/2,
        "along_speed_mps": blocker["along_speed_mps"], "legal_clear_sides": sides,
        "source": "measured_slow_route_obstacle_with_current_gap_proxy",
        "execution_authorized": False}


def build_scene_context(world, ego, route, sim_time_s, *, reference=None, special_situations=(),
                        motion_history=None):
    tf, velocity = ego.get_transform(), ego.get_velocity()
    location = tf.location
    speed = math.hypot(velocity.x, velocity.y)
    heading = math.radians(tf.rotation.yaw)
    ego_body_xy, ego_body_yaw = _body_pose(ego, tf)
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
    ego_route_keys, ego_route_width = _route_lane_at(ego_station, local)
    ego_route_aligned = (current is not None and lane_key(current) in ego_route_keys
                         and ego_projection[1] <= ego_route_width/2
                         and math.cos(heading-ego_projection[2]) > .8)
    route_keys = {lane_key(wp) for wp in local}
    road_limit = float(ego.get_speed_limit())/3.6
    prediction_kind = ("connected_lane_observed_turn_rate_and_following_response_not_future_GT" if motion_history is not None
                       else "constant_velocity_proxy_not_future_GT")
    result = {"schema_version": 1, "source": "CARLA_GT_route_observed_motion_proxy" if motion_history is not None
              else "CARLA_GT_route_constant_velocity_proxy",
              "sim_time_s": float(sim_time_s), "ego_speed_mps": speed,
              "route_progress_m": float(route.distance_completed_m()),
              "in_junction": bool(current and current.is_junction),
              "lane_key": lane_key(current), "navigation": route.nav_instruction(),
              "speed_limit_mps": road_limit, "signals": [], "actors": [],
              "special_situations": sorted(set(special_situations)),
              "blocked": False, "baseline_stop_expected": False,
              "prediction_kind": prediction_kind}
    weather = world.get_weather()
    if float(weather.fog_density) >= 40:
        result["special_situations"] = sorted(set(result["special_situations"]+["limited_visibility_fog"]))
    current_actors = tuple(world.get_actors())
    for actor in current_actors:
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
        motion = motion_history.motion_for(actor) if motion_history is not None else None
        v = motion.velocity if motion is not None else actor.get_velocity()
        dx, dy = p.x-location.x, p.y-location.y
        relative_s = dx*math.cos(heading)+dy*math.sin(heading)
        relative_l = dx*math.sin(heading)-dy*math.cos(heading)
        other_wp = carla_map.get_waypoint(p)
        projection = project_polyline(p.x, p.y, points)
        along = projection[0]-ego_station
        extent = actor.bounding_box.extent
        actor_tf = actor.get_transform()
        actor_yaw = math.radians(actor_tf.rotation.yaw)
        actor_body_xy, actor_body_yaw = _body_pose(actor, actor_tf)
        actor_prediction = None
        if motion is not None:
            from .actor_motion import predict_actor_motion
            _, _, actor_prediction = predict_actor_motion(actor, motion,
                [index*.25 for index in range(17)], carla_map, motion_history.config,
                history=motion_history, leader_actors=current_actors)
        projected_keys, projected_width = _route_lane_at(projection[0], local)
        route_aligned = (other_wp is not None and lane_key(other_wp) in projected_keys
                         and projection[1] <= projected_width/2
                         and math.cos(actor_yaw-projection[2]) > .8)
        same_lane = route_aligned and ego_route_aligned
        conflict, ttc = constant_velocity_conflict(ego_station, speed, points,
                        (p.x, p.y), (v.x, v.y),
                        ego_radius=float(ego.bounding_box.extent.y), actor_radius=float(extent.y),
                        actor_prediction=actor_prediction)
        stationary = math.hypot(v.x, v.y) < .5
        along_speed = v.x*math.cos(projection[2])+v.y*math.sin(projection[2])
        cross_speed = -v.x*math.sin(projection[2])+v.y*math.cos(projection[2])
        path_block = stationary and 1 < along < 30 and projection[1] < 1.1+float(extent.y)
        occupied_conflict, occupied_ttc = occupied_body_conflict(
            ego_body_xy, (velocity.x, velocity.y), ego_body_yaw,
            (float(ego.bounding_box.extent.x), float(ego.bounding_box.extent.y)),
            actor_body_xy, (v.x, v.y), actor_body_yaw, (float(extent.x), float(extent.y)),
            actor_prediction=actor_prediction)
        result["blocked"] |= bool(path_block)
        result["actors"].append({"id": int(actor.id), "type": kind,
            "distance_m": distance, "relative_s_m": relative_s, "relative_l_m": relative_l,
            "same_lane": bool(same_lane), "ahead": bool(relative_s > 0),
            "conflict": bool(conflict), "ttc_s": ttc, "stationary": stationary,
            "path_blocked": bool(path_block), "speed_mps": math.hypot(v.x, v.y),
            "route_distance_m": along, "lane_key": lane_key(other_wp),
            "route_aligned": bool(route_aligned), "route_lateral_distance_m": projection[1],
            "length_m": float(extent.x)*2, "width_m": float(extent.y)*2,
            "position_xy_m": [float(p.x), float(p.y)],
            "along_speed_mps": float(along_speed), "cross_speed_mps": float(cross_speed),
            "route_heading_aligned": bool(math.cos(actor_yaw-projection[2]) > .8),
            "occupied_ego_conflict": bool(occupied_conflict),
            "occupied_ego_conflict_ttc_s": occupied_ttc,
            "motion_source": motion.source if motion is not None else "carla_velocity_api",
            "motion_quality": motion.quality if motion is not None else "unverified_api",
            "prediction_mode": actor_prediction["mode"] if actor_prediction is not None else "constant_velocity",
            "prediction_map_status": actor_prediction.get("map_status") if actor_prediction is not None else None,
            "following_response": actor_prediction.get("following_response") if actor_prediction is not None else None,
            "yaw_rate_rad_s": motion.yaw_rate_rad_s if motion is not None else None})
    result["actors"].sort(key=lambda a: (not a["conflict"], not a["path_blocked"], a["distance_m"]))
    result["signals"].sort(key=lambda a: a["stop_line_distance_m"])
    _traffic_wait(result, float(ego.bounding_box.extent.x))
    _overtaking_context(result, current, local, ego, carla_map, ego_route_aligned)
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
            "baseline_planning_score": (baseline["base_score"]-baseline.get("numerical_cost", 0.0)
                if isinstance(baseline.get("base_score"), (int, float))
                and isinstance(baseline.get("numerical_cost", 0.0), (int, float)) else None),
            "baseline_progress_m": baseline.get("forward_progress_m"),
            "baseline_speed_mps": baseline.get("terminal_speed_mps"),
            "best_score": max(finite_scores) if finite_scores else None,
            "candidate_count": len(candidates),
            "available_lateral_semantics": sorted({s for c in candidates for s in c.get("lateral_semantics", [])})}


def planner_health(candidates, baseline_id, *, moving_progress_m=2.0):
    """Summarize the ORIGINAL native pool before model-requested expansion.

    A finite trajectory whose PDM score is zero is not a usable forward option.
    Keep this separate from ``candidate_summary`` so the original observation is
    retained when expanded trajectories subsequently enter the scored pool.
    The counts describe PDM predictions, not actual collisions or violations.
    """
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    finite_pool = [c for c in candidates
                   if c.get("safety", {}).get("finite") is True
                   and finite(c.get("base_score"))]
    moving = [c for c in finite_pool if c.get("longitudinal") != "STOP"
              and finite(c.get("forward_progress_m"))
              and c["forward_progress_m"] >= moving_progress_m]
    progressing = [c for c in moving if c["base_score"] > 0]
    baseline = next((c for c in finite_pool if c.get("id") == baseline_id), None)
    baseline_stop = (baseline is not None and (baseline.get("longitudinal") == "STOP"
                     or (finite(baseline.get("forward_progress_m"))
                         and baseline["forward_progress_m"] < moving_progress_m)))

    def failed(name):
        return sum(1 for c in finite_pool
                   if finite(c.get("safety", {}).get(name))
                   and c["safety"][name] < 1.0-1e-8)

    return {"source": "native_pdm_baseline", "candidate_count": len(candidates),
            "finite_candidate_count": len(finite_pool),
            "moving_candidate_count": len(moving),
            "positive_progressing_count": len(progressing),
            "drivable_area_failure_count": failed("drivable_area"),
            "no_collision_failure_count": failed("no_collision"),
            "baseline_candidate_id": baseline_id,
            "baseline_score": baseline.get("base_score") if baseline else None,
            "baseline_longitudinal": baseline.get("longitudinal") if baseline else None,
            "baseline_progress_m": baseline.get("forward_progress_m") if baseline else None,
            "baseline_stop": bool(baseline_stop),
            "no_positive_progressing_candidate": not progressing}
