"""Causal privileged NAVSIM history/map -> EP-Driver packet (Python 3.9).

Only the four history/current frames are indexed. Future Scene frames, metric
caches, human trajectories and extended future detections are not inspected.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import INPUT_PROTOCOL, INPUT_CONDITION
from .protocol import transform_points, rotate_vectors, validate_packet, wrap

CAMERAS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")
KINDS = {"vehicle": "vehicle", "bicycle": "bicycle", "pedestrian": "pedestrian",
         "traffic_cone": "static_obstacle", "czone_sign": "static_obstacle",
         "barrier": "static_obstacle"}
# Generic-object labels do not establish that an object is a road obstruction.


def _local_shape(shape, origin):
    translated = affinity.translate(shape, xoff=-origin[0], yoff=-origin[1])
    return affinity.rotate(translated, -origin[2], origin=(0, 0), use_radians=True)


def _project(points, position):
    line = LineString(points)
    arc = line.project(Point(position))
    foot = np.array(line.interpolate(arc).coords[0])
    start, end = max(0.0, arc - 0.05), min(line.length, arc + 0.05)
    tangent = np.array(line.interpolate(end).coords[0]) - np.array(line.interpolate(start).coords[0])
    return arc, float(np.linalg.norm(foot - position)), math.atan2(tangent[1], tangent[0])


def _clean(points):
    points = np.asarray(points, float)
    return points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]


def _correct_route_order(map_api, origin, order):
    """Reconnect current pose to the public route using the pinned PDM utility.

    Official synthetic follow-ups can still occupy the predecessor of the first
    supplied roadblock. Only current pose, map topology and supplied route enter
    this correction; no human/future trajectory or scoring cache is consulted.
    """
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction

    blocks = {}
    for key in dict.fromkeys(order):
        obj = map_api.get_map_object(key, SemanticMapLayer.ROADBLOCK)
        if obj is None:
            obj = map_api.get_map_object(key, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if obj is None:
            raise ValueError("Public navigation roadblock is missing from the map: " + key)
        blocks[key] = obj
    if not blocks:
        raise ValueError("NAVSIM privileged navigation requires a public roadblock route")
    return [str(key) for key in route_roadblock_correction(StateSE2(*origin), map_api, blocks)]


def _map_packet(map_api, origin, roadblock_ids, desired_speed_mps, map_radius_m):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    # nuPlan has no object representation for DRIVABLE_AREA. Follow the
    # official PDM map's supported vector layers rather than query its raster.
    road_layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA]
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR,
              SemanticMapLayer.CROSSWALK] + road_layers
    nearby = map_api.get_proximal_map_objects(Point2D(*origin[:2]), map_radius_m, layers)
    objects = nearby[SemanticMapLayer.LANE] + nearby[SemanticMapLayer.LANE_CONNECTOR]
    lane_objects = {str(obj.id): obj for obj in objects}
    lanes, skipped = [], []
    for key, obj in lane_objects.items():
        center = _clean(transform_points([[p.x, p.y] for p in obj.baseline_path.discrete_path], origin))
        if len(center) < 2 or not obj.polygon.is_valid or obj.polygon.is_empty:
            skipped.append(key)
            continue
        polygon = _local_shape(obj.polygon, origin)
        speed = obj.speed_limit_mps
        speed = float(speed) if speed is not None and np.isfinite(speed) and speed > 0 else desired_speed_mps
        left, right = obj.adjacent_edges
        # NuPlan supplies lane adjacency, not CARLA lane-marking permissions.
        # Keep this assumption explicit in the archived input condition.
        neighbor = lambda value: str(value.id) if value is not None and str(value.id) in lane_objects else None
        baseline = obj.baseline_path.linestring
        probes = [baseline.interpolate(fraction, normalized=True) for fraction in (0.2, 0.4, 0.6, 0.8)]
        width = float(np.median([point.distance(obj.left_boundary.linestring)
                                + point.distance(obj.right_boundary.linestring) for point in probes]))
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Invalid mapped lane width: " + key)
        lane = dict(id=key, center=center.tolist(), polygon=np.asarray(polygon.exterior.coords).tolist(),
                    width=width,
                    speed_limit=speed, successors=[str(x.id) for x in obj.outgoing_edges if str(x.id) in lane_objects],
                    left=neighbor(left), right=neighbor(right), priority=0, boundaries=[])
        for side, adjacent in (("left", left), ("right", right)):
            boundary = getattr(obj, side + "_boundary")
            points = transform_points(np.asarray(boundary.linestring.coords), origin)
            lane["boundaries"].append(dict(id=key + ":" + side, kind="lane_boundary",
                points=points.tolist(), crossable=neighbor(adjacent) is not None,
                permission_source="nuplan_directed_lane_adjacency_assumption_no_paint_labels"))
        lanes.append(lane)
    if not lanes:
        raise ValueError("No usable NAVSIM map lanes near the current frame")
    table = {lane["id"]: lane for lane in lanes}
    public_order = [str(key) for key in roadblock_ids]
    order = public_order
    def block(key):
        return str(lane_objects[key].get_roadblock_id())
    def association(lane):
        _, error, heading = _project(lane["center"], np.zeros(2))
        covers = Polygon(lane["polygon"]).buffer(0.05).covers(Point(0, 0))
        return dict(projection_error_m=error, heading_error_rad=abs(float(wrap(heading))),
                    polygon_covers_rear_axle=bool(covers))
    associations = {lane["id"]: association(lane) for lane in lanes}
    def nearest(lane):
        row = associations[lane["id"]]
        return (not row["polygon_covers_rear_axle"], row["heading_error_rad"] > math.pi / 2,
                row["projection_error_m"] + row["heading_error_rad"])
    def select(route_order):
        candidates = [lane for lane in lanes if block(lane["id"]) in set(route_order)]
        return min(candidates, key=nearest) if candidates else None
    current = select(order)
    initial_lane = None if current is None else current["id"]
    # Never snap to an unrelated nearby lane to hide a missing route prefix.
    # The official directed graph correction must first establish a connection.
    corrected = current is None or not associations[current["id"]]["polygon_covers_rear_axle"]
    if corrected:
        order = _correct_route_order(map_api, origin, order)
        current = select(order)
    if current is None or nearest(current)[1] or nearest(current)[2] > 8:
        raise ValueError("Current ego pose cannot be associated with the mapped route after topology correction")
    allowed = set(order)
    current_id = current["id"]
    route_evidence = dict(public_roadblock_ids=public_order, used_roadblock_ids=order,
        topology_correction_attempted=corrected, topology_correction_changed=order != public_order,
        correction_source="pinned_official_pdm_route_roadblock_correction_current_pose_map_public_route",
        initial_lane_id=initial_lane, current_lane_id=current_id, **associations[current_id])
    route, ids, visited = np.asarray(current["center"], float), [current_id], {current_id}
    current_block_index = order.index(block(current_id)) if block(current_id) in order else -1
    while len(ids) < 128 and LineString(route).length < map_radius_m * 1.5:
        choices = [table[key] for key in current["successors"] if key in table and key not in visited
                   and (not allowed or block(key) in allowed)]
        if not choices:
            break
        end_heading = math.atan2(*(route[-1] - route[-2])[::-1])
        def rank(lane):
            rb = block(lane["id"])
            indices = [i for i, value in enumerate(order) if value == rb and i >= current_block_index]
            index = min(indices, default=len(order))
            center = np.asarray(lane["center"])
            heading = math.atan2(*(center[1] - center[0])[::-1])
            return index, abs(float(wrap(heading - end_heading))), lane["id"]
        nxt = min(choices, key=rank)
        center = np.asarray(nxt["center"])
        # Join genuine map endpoints only, without inventing a shortcut curve.
        if np.linalg.norm(center[0] - route[-1]) > 2.0:
            raise ValueError("Mapped navigation has an unsupported endpoint gap")
        route = _clean(np.vstack([route, center]))
        current = nxt
        visited.add(current["id"])
        ids.append(current["id"])
        if block(current["id"]) in order:
            indices = [i for i, value in enumerate(order) if value == block(current["id"]) and i >= current_block_index]
            if indices:
                current_block_index = indices[0]
    road = unary_union([_local_shape(obj.polygon, origin) for layer in road_layers for obj in nearby[layer]]
                       + [Polygon(lane["polygon"]) for lane in lanes])
    crosswalks = [np.asarray(_local_shape(obj.polygon, origin).exterior.coords).tolist()
                  for obj in nearby[SemanticMapLayer.CROSSWALK]]
    return lanes, route, ids, current_id, road, crosswalks, lane_objects, skipped, route_evidence


def _history_actors(frames, origin):
    histories = {}
    ignored = {}
    t0 = frames[-1].timestamp
    for frame in frames:
        pose = np.asarray(frame.ego_status.ego_pose, float)
        time_s = (frame.timestamp - t0) / 1e6
        anns = frame.annotations
        for i, kind in enumerate(anns.names):
            if kind not in KINDS:
                ignored[kind] = ignored.get(kind, 0) + 1
                continue
            box = np.asarray(anns.boxes[i], float)
            position_global = rotate_vectors(box[:2], pose[2]) + pose[:2]
            position = transform_points(position_global, origin)
            velocity = rotate_vectors(np.asarray(anns.velocity_3d[i])[:2], pose[2] - origin[2])
            if not np.isfinite(velocity).all():
                raise ValueError("Non-finite observed track velocity; no future-based imputation")
            identity = str(anns.track_tokens[i])
            values = histories.setdefault(identity, [])
            acceleration = np.zeros(2)
            if values:
                dt = time_s - values[-1]["time"]
                if dt <= 0:
                    raise ValueError("Non-increasing track history")
                acceleration = (velocity - values[-1]["velocity"]) / dt
            values.append(dict(time=time_s, position=position.tolist(), velocity=velocity.tolist(),
                               acceleration=acceleration.tolist(), heading=float(wrap(box[6] + pose[2] - origin[2])),
                               observed=True))
    current, actors = frames[-1].annotations, []
    for i, name in enumerate(current.names):
        if name not in KINDS:
            continue
        identity = str(current.track_tokens[i])
        h = histories[identity]
        state = h[-1]
        actors.append(dict(id=identity, kind=KINDS[name], position=state["position"], velocity=state["velocity"],
                           acceleration=state["acceleration"], heading=state["heading"],
                           length=float(current.boxes[i][3]), width=float(current.boxes[i][4]),
                           lane_id=None, visible=True, history=h, radius=None))
    return actors, ignored


def build_packet(agent_input, scene, camera_dir, desired_speed_mps=10.0, map_radius_m=160.0):
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    metadata = scene.scene_metadata
    if metadata.num_history_frames != 4 or len(agent_input.ego_statuses) != 4:
        raise ValueError("The official v2.2 evaluation input must have four history/current frames")
    # This is the only access to scene.frames. Never index a future frame.
    frames = [scene.frames[i] for i in range(4)]
    now = frames[-1]
    if str(metadata.initial_token) != str(now.token):
        raise ValueError("Scene metadata/current frame token mismatch")
    origin = np.asarray(now.ego_status.ego_pose, float)
    vehicle = get_pacifica_parameters()
    rear = float(vehicle.rear_axle_to_center)
    poses = np.array([frame.ego_status.ego_pose for frame in frames], float)
    ts = np.array([frame.timestamp for frame in frames], dtype=np.int64)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("History timestamps are not increasing")
    yaw_rates = np.diff(np.unwrap(poses[:, 2])) / (np.diff(ts) / 1e6)
    omega = float(yaw_rates[-1])
    alpha = float((yaw_rates[-1] - yaw_rates[-2]) / ((ts[-1] - ts[-2]) / 1e6))
    velocity = np.asarray(now.ego_status.ego_velocity, float).copy()
    acceleration = np.asarray(now.ego_status.ego_acceleration, float).copy()
    # NAVSIM stores body-frame rear-axle velocity/acceleration. Centre kinematics
    # include the causal rigid-body yaw terms; dimensions come from nuPlan.
    velocity[1] += rear * omega
    acceleration += np.array([-rear * omega ** 2, rear * alpha])
    ego_history = []
    for index, frame in enumerate(frames):
        pose = np.asarray(frame.ego_status.ego_pose, float)
        center_global = pose[:2] + rear * np.array([np.cos(pose[2]), np.sin(pose[2])])
        history_omega = float(yaw_rates[index-1]) if index > 0 else 0.0
        history_alpha = (float((yaw_rates[index-1] - yaw_rates[index-2]) / ((ts[index] - ts[index-1]) / 1e6))
                         if index > 1 else 0.0)
        history_velocity = np.asarray(frame.ego_status.ego_velocity, float) + [0.0, rear * history_omega]
        history_acceleration = np.asarray(frame.ego_status.ego_acceleration, float) + [-rear * history_omega ** 2, rear * history_alpha]
        ego_history.append(dict(time=(frame.timestamp - now.timestamp) / 1e6,
            position=transform_points(center_global, origin).tolist(),
            velocity=rotate_vectors(history_velocity, pose[2] - origin[2]).tolist(),
            acceleration=rotate_vectors(history_acceleration, pose[2] - origin[2]).tolist(),
            heading=float(wrap(pose[2] - origin[2])), observed=True))
    ego = dict(id="ego", kind="vehicle", position=[rear, 0.0], velocity=velocity.tolist(),
               acceleration=acceleration.tolist(), heading=0.0, length=float(vehicle.length),
               width=float(vehicle.width), lane_id=None, visible=True, history=ego_history,
               radius=None, wheelbase_m=float(vehicle.wheel_base), rear_axle_to_center_m=rear)
    lanes, route, ids, current, road, crosswalks, objects, skipped, route_evidence = _map_packet(
        scene.map_api, origin, now.roadblock_ids, desired_speed_mps, map_radius_m)
    ego["lane_id"] = current
    actors, ignored = _history_actors(frames, origin)
    for actor in actors:
        matches = []
        for lane in lanes:
            if Polygon(lane["polygon"]).covers(Point(actor["position"])):
                _, error, heading = _project(lane["center"], actor["position"])
                matches.append((error + abs(float(wrap(actor["heading"] - heading))), lane["id"]))
        if matches:
            actor["lane_id"] = min(matches)[1]
    signals, stops = [], []
    for connector_id, is_red in now.traffic_lights:
        key = str(connector_id)
        row = dict(id=key, state="red" if is_red else "not_red", is_red=bool(is_red),
                   lane_ids=[key], observed_at_us=int(now.timestamp))
        obj = objects.get(key)
        if obj is not None and key in ids:
            # The controlled connector entry is also used by NAVSIM's red-light
            # polygon checker. A map stop line, where available, takes precedence.
            stop_lines = list(getattr(obj, "stop_lines", []) or [])
            if stop_lines:
                points = [_local_shape(item.polygon, origin).centroid for item in stop_lines]
                stop = min(points, key=lambda point: point.distance(LineString(route)))
                stop_point = np.array(stop.coords[0])
                row["stop_source"] = "nuplan_map_stop_line"
            else:
                point = obj.baseline_path.discrete_path[0]
                stop_point = transform_points([point.x, point.y], origin)
                row["stop_source"] = "controlled_lane_connector_entry"
            s, error, _ = _project(route, stop_point)
            row.update(stop_s=s, projection_error_m=error)
            ego_s = _project(route, ego["position"])[0]
            if is_red and s >= ego_s - vehicle.length / 2 and error < 5.0:
                stops.append(s)
        signals.append(row)
    output = Path(camera_dir)
    output.mkdir(parents=True, exist_ok=True)
    cameras = []
    for name in CAMERAS:
        camera = getattr(agent_input.cameras[-1], name)
        if camera.image is None:
            raise ValueError("Missing requested current camera: " + name)
        path = output / (name + ".jpg")
        Image.fromarray(camera.image).save(path, quality=95)
        cameras.append(dict(view=name, path=str(path.resolve()), frame_id=str(now.token), time=0.0,
                            source="NAVSIM_current_original_or_official_synthetic_camera", plan_overlay=False))
    route_s = _project(route, ego["position"])[0]
    speed_limit = next(lane["speed_limit"] for lane in lanes if lane["id"] == current)
    stop_s = min(stops) if stops else None
    return validate_packet(dict(protocol=INPUT_PROTOCOL, input_condition=INPUT_CONDITION,
        identity=dict(token=str(now.token), log_name=metadata.log_name, scene_token=metadata.scene_token,
                      map_name=metadata.map_name, timestamp_us=int(now.timestamp), history_frames=4,
                      history_timestamps_us=ts.tolist(),
                      corresponding_original_scene=metadata.corresponding_original_scene,
                      corresponding_original_initial_token=metadata.corresponding_original_initial_token),
        vehicle=dict(length_m=float(vehicle.length), width_m=float(vehicle.width),
                     wheelbase_m=float(vehicle.wheel_base), rear_axle_to_center_m=rear),
        ego=ego, actors=actors, lanes=lanes, route=route.tolist(), route_lane_ids=ids, drivable_wkt=road.wkt,
        rules=dict(desired_speed=min(desired_speed_mps, speed_limit), speed_limit=speed_limit,
                   must_wait=stop_s is not None, wait_reason="red_light" if stop_s is not None else None,
                   stop_s=stop_s, crosswalks=crosswalks, traffic_lights=signals,
                   map_available=dict(crosswalk=True, traffic_light=True, stop_line=True),
                   road_priority_available=False),
        diagnostics=dict(route_s=route_s, reference_anchor_s=route_s, stable_navigation_lane_id=current,
            lane_map_complete=True, reference_revision=0, has_occlusion=False, visibility_wkt=road.wkt,
            lane_map_complete_scope="local_map_query_radius_%gm" % map_radius_m,
            available_observation_history_s=float((ts[-1]-ts[0])/1e6),
            observation_policy=INPUT_CONDITION, previous_control=[float(np.clip(acceleration[0], -6, 3)),
                float(np.clip(math.atan(vehicle.wheel_base * omega / max(abs(velocity[0]), 0.3)), -0.6, 0.6))],
            previous_control_source="causal_observed_acceleration_and_yaw_rate_estimate_not_actuator_truth",
            acceleration_source="current_ego_body_state_and_history_track_velocity_difference",
            source_rear_axle_global_pose=origin.tolist(), ignored_annotation_types=ignored, skipped_map_lanes=skipped,
            route_association=route_evidence,
            lane_crossability_source="nuplan_directed_adjacency_not_lane_paint_annotations",
            route_source="current_privileged_roadblock_ids_and_map_topology_no_future_ego_poses"),
        camera_images=cameras))
