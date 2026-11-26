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
from .protocol import transform_points, rotate_vectors, validate_packet, wrap, HORIZON_S
from .geometry_transfer import drivable_roundtrip

CAMERAS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")
KINDS = {"vehicle": "vehicle", "bicycle": "bicycle", "pedestrian": "pedestrian",
         "traffic_cone": "static_obstacle", "czone_sign": "static_obstacle",
         "barrier": "static_obstacle"}
# Generic-object labels do not establish that an object is a road obstruction.


def navigation_command(value):
    """Preserve the official current command, without deriving it from future poses."""
    if value is None:
        return dict(one_hot=None, label="unknown", source="not_available")
    raw = np.asarray(value)
    if (raw.shape != (4,) or not np.isfinite(raw).all()
            or not np.isin(raw, [0, 1]).all() or int(raw.sum()) != 1):
        raise ValueError("NAVSIM current driving_command must contain four one-hot values")
    return dict(one_hot=raw.astype(int).tolist(),
                label=("left", "straight", "right", "unknown")[int(np.argmax(raw))],
                source="NAVSIM_current_ego_status.driving_command")


def required_forward_reference(desired_speed_mps, current_speed_mps=0.0):
    """Require a usable reference for the submitted horizon, even when starting slowly."""
    values = np.asarray([desired_speed_mps, current_speed_mps], float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid current/desired speed for navigation coverage")
    return max(15.0, HORIZON_S * float(max(values)) + 5.0)


def navigation_ego(rear_axle_velocity, yaw_rate, vehicle=None):
    """The identical current Pacifica state used by packet and map preflight."""
    if vehicle is None:
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        vehicle = get_pacifica_parameters()
    velocity = np.asarray(rear_axle_velocity,float).copy()
    if velocity.shape != (2,) or not np.isfinite(velocity).all() or not math.isfinite(yaw_rate):
        raise ValueError('NAVSIM current rear-axle velocity/yaw rate is invalid')
    rear = float(vehicle.rear_axle_to_center)
    velocity[1] += rear*float(yaw_rate)
    return dict(id='ego',kind='vehicle',position=[rear,0.],velocity=velocity.tolist(),heading=0.,
        length=float(vehicle.length),width=float(vehicle.width),lane_id=None,
        wheelbase_m=float(vehicle.wheel_base),rear_axle_to_center_m=rear)


def validate_navigation_initialization(ego, lanes, route, ids, road, association, yaw_rate, token):
    """One producer/validator path for full packets and image-free preflight."""
    from types import SimpleNamespace as NS
    road, _, geometry_evidence = drivable_roundtrip(road)
    from nine_grid.navigation_initialization import build_initialization,initialization_geometry
    route,record = build_initialization(ego,lanes,route,ids,road,association,yaw_rate=yaw_rate)
    snapshot = NS(scenario='navsim',frame_id=str(token),time=0.,assets={},
        ego=NS(**dict(ego,position=np.asarray(ego['position'],float))),
        lanes=[NS(**dict(row,center=np.asarray(row['center'],float),polygon=np.asarray(row['polygon'],float)))
               for row in lanes],route=np.asarray(route,float),route_lane_ids=ids,
        drivable=road,diagnostics=dict(navigation_initialization=record,route_association=association,
                                     lane_map_complete=True))
    initialization_geometry(snapshot)
    # Ordinary on-navigation frames need the identical short native seam
    # validation too; initialization recovery must not be the only entry point.
    from nine_grid.object_scope import lane_role_geometry
    snapshot.coordinates='world_xy_m'
    traces,_,_ = lane_role_geometry(snapshot,front_m=120.,rear_m=20.)
    record['native_seam_validation'] = dict(front_m=120.,rear_m=20.,
        connections=[dict(from_lane_id=link['from_lane_id'],to_lane_id=link['to_lane_id'])
            for trace in traces.values() if trace is not None
            for link in trace.get('navigation_connections',[])
            if link.get('kind')=='validated_short_directed_native_navigation_gap'],
        rejected_connections=[rejection for trace in traces.values() if trace is not None
                              for rejection in trace.get('native_connection_rejections',[])],
        roles=snapshot.diagnostics.get('native_navigation_validation',{}))
    reasons=list(record.get('reference_quality_reasons',[]))+association.get('reference_quality_reasons',[])
    for trace in traces.values():
        if trace is None:continue
        if trace.get('native_forward_valid') is False:reasons.append('native_forward_geometry_unverified')
        for link in trace.get('navigation_connections',[]):
            quality=link.get('reference_quality',{})
            if quality.get('full_body_on_road') is False:reasons.append('reference_body_outside_actual_road')
            if (quality.get('maximum_steering_rad') or 0.)>.61:reasons.append('reference_steering_exceeds_limit')
    record['reference_quality_reasons']=sorted(set(reasons))
    record['input_contract']='causal_finite_current_state_directed_public_topology_and_coordinate_attachments'
    record['reference_feasibility_is_input_gate']=False
    record['geometry_transfer'] = geometry_evidence
    # Preflight used to validate lane chains without constructing their final
    # lane/map intersection. Check the identical domain builder before images,
    # DINO or model calls, so numeric overlay failures are not discovered late.
    # This is the default 120m/20m base domain, not the speed-dependent extent
    # the MPC may request later, and does not certify a vehicle trajectory.
    from nine_grid.potential_domain import build_lane_potential_domain
    domain,evidence = build_lane_potential_domain(snapshot)
    record['potential_domain_validation'] = dict(valid=True,area_m2=float(domain.area),
        domain_type=domain.geom_type,checked_front_m=evidence['map_extension_front_m'],
        checked_rear_m=evidence['map_extension_rear_m'],numeric_overlay=evidence['numeric_overlay'],
        scope='base_domain_geometry_only;MPC_revalidates_actual_speed_dependent_extent',
        vehicle_safety_certificate=False)
    return route,record


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


def _roadblock(map_api, key):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    obj = map_api.get_map_object(str(key), SemanticMapLayer.ROADBLOCK)
    return obj if obj is not None else map_api.get_map_object(str(key), SemanticMapLayer.ROADBLOCK_CONNECTOR)


def _route_order_connected(map_api, order):
    """Check the public directed roadblock sequence without inventing links."""
    blocks = []
    for key in order:
        if blocks and str(blocks[-1].id) == str(key):
            continue
        obj = _roadblock(map_api, key)
        if obj is None:
            return False
        blocks.append(obj)
    return bool(blocks) and all(str(right.id) in {str(edge.id) for edge in left.outgoing_edges}
                                for left, right in zip(blocks, blocks[1:]))


def _link_public_route(map_api, order):
    """Only insert directed map links between ordered public waypoints.

    Use the same bounded BFS as official PDM's interior-link repair, without
    its off-route prefix selection or polygon-based removal of the public tail.
    """
    if _route_order_connected(map_api, order):
        return order[:]
    from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock import BreadthFirstSearchRoadBlock

    objects = {str(key): _roadblock(map_api, key) for key in order}
    if not objects or any(obj is None for obj in objects.values()):
        return None
    result = [str(order[0])]
    for key in map(str, order[1:]):
        previous = result[-1]
        if key == previous:
            continue
        if key in {str(edge.id) for edge in objects[previous].outgoing_edges}:
            result.append(key)
            continue
        graph = BreadthFirstSearchRoadBlock(previous, map_api, forward_search=True)
        (path, ids), found = graph.search(key, max_depth=30)
        ids = [str(value) for value in ids]
        if (not found or len(ids) < 2 or ids[0] != previous or ids[-1] != key
                or [str(obj.id) for obj in path] != ids
                or any(value in objects for value in ids[1:-1])):
            return None
        result.extend(ids[1:])
    return result if _route_order_connected(map_api, result) else None


def _map_packet(map_api, origin, roadblock_ids, desired_speed_mps, map_radius_m,
                current_speed_mps=0.0, driving_command=None):
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
            lane["boundaries"].append(dict(id=key + ":" + side, kind="virtual_lane_boundary",
                points=points.tolist(), crossable=neighbor(adjacent) is not None,
                permission_source="nuplan_directed_lane_adjacency_assumption_no_paint_labels",
                marking="unknown", geometry_source="nuplan_vector_lane_boundary_not_painted_line"))
        lanes.append(lane)
    if not lanes:
        raise ValueError("No usable NAVSIM map lanes near the current frame")
    table = {lane["id"]: lane for lane in lanes}
    public_order = [str(key) for key in roadblock_ids]
    order = public_order
    command = navigation_command(driving_command)
    required_forward = required_forward_reference(desired_speed_mps, current_speed_mps)
    def block(key):
        return str(lane_objects[key].get_roadblock_id())
    def association(lane):
        _, error, heading = _project(lane["center"], np.zeros(2))
        covers = Polygon(lane["polygon"]).buffer(0.05).covers(Point(0, 0))
        return dict(projection_error_m=error, heading_error_rad=abs(float(wrap(heading))),
                    polygon_covers_rear_axle=bool(covers))
    associations = {lane["id"]: association(lane) for lane in lanes}
    chains = {}
    def chain(lane, route_order):
        cache_key = (lane["id"], tuple(route_order))
        if cache_key in chains:
            return chains[cache_key]
        path, ids, visited = np.asarray(lane["center"], float), [lane["id"]], {lane["id"]}
        current, allowed = lane, set(route_order)
        index = route_order.index(block(lane["id"]))
        ego_s = _project(path, np.zeros(2))[0]
        while len(ids) < 128 and LineString(path).length - ego_s < map_radius_m * 1.5:
            choices = []
            for key in current["successors"]:
                if key not in table or key in visited or block(key) not in allowed:
                    continue
                indices = [i for i, value in enumerate(route_order) if value == block(key) and i >= index]
                if indices and np.linalg.norm(np.asarray(table[key]["center"])[0] - path[-1]) <= 2.0:
                    choices.append((min(indices), table[key]))
            if not choices:
                break
            end_heading = math.atan2(*(path[-1] - path[-2])[::-1])
            def rank(item):
                next_index, candidate = item
                center = np.asarray(candidate["center"])
                heading = math.atan2(*(center[1] - center[0])[::-1])
                return next_index, abs(float(wrap(heading - end_heading))), candidate["id"]
            index, current = min(choices, key=rank)
            path = _clean(np.vstack([path, current["center"]]))
            visited.add(current["id"])
            ids.append(current["id"])
        line = LineString(path)
        forward = float(line.length - line.project(Point(0, 0)))
        probe_s = min(line.length, line.project(Point(0, 0)) + required_forward)
        tangent_start = np.array(line.interpolate(max(0.0, probe_s - .05)).coords[0])
        tangent_end = np.array(line.interpolate(min(line.length, probe_s + .05)).coords[0])
        turn = float(wrap(math.atan2(*(tangent_end - tangent_start)[::-1])))
        # Commands include lane changes and sharp curves; only a directly
        # opposing turn is a disagreement, not the absence of an immediate turn.
        disagreement = ((command["label"] == "straight" and abs(turn) > math.pi / 3)
                        or (command["label"] == "left" and turn < -math.pi / 3)
                        or (command["label"] == "right" and turn > math.pi / 3))
        chains[cache_key] = path, ids, forward, turn, bool(disagreement)
        return chains[cache_key]
    def nearest(lane):
        row = associations[lane["id"]]
        # Overlapping connector polygons can contain opposing directions.
        # Reject direction-incompatible candidates before preferring coverage;
        # otherwise an opposite connector masks a nearby valid route branch.
        return (row["heading_error_rad"] > math.pi / 2, not row["polygon_covers_rear_axle"],
                row["projection_error_m"] + row["heading_error_rad"])
    def select(route_order):
        # A wide lane polygon can cover ego while its baseline is much
        # farther away than a valid predecessor. Apply the existing association
        # limits before coverage ranking, rather than abort on that invalid winner.
        candidates = [lane for lane in lanes if block(lane["id"]) in set(route_order)
                      and not nearest(lane)[0] and nearest(lane)[2] <= 8]
        # An ended predecessor can be geometrically closer than the current
        # connector. Prefer a reachable forward route before residual distance;
        # otherwise a behind-only reference passes with hundreds of old points.
        def route_rank(lane):
            _, _, forward, _, disagreement = chain(lane, route_order)
            return forward <= 1., forward < required_forward, disagreement, nearest(lane)
        return min(candidates, key=route_rank) if candidates else None
    current = select(order)
    initial_lane = None if current is None else current["id"]
    # PDM corrects the public route at every independent initialization, even
    # when ego already lies on it: interior links can be missing as well as the
    # prefix. Never substitute an unrelated nearby lane for this graph search.
    order = _correct_route_order(map_api, origin, order)
    attempted_order = order[:]
    rejection_reason = None
    current = select(order)
    if current is None or chain(current, order)[2] < required_forward:
        # The official correction can start on an overlapping opposite branch
        # and remove a valid public suffix as a loop. Keep the public route only
        # if its current association and every directed map link are verified.
        public_current = select(public_order)
        if public_current is not None:
            linked_public = _link_public_route(map_api, public_order)
            # Inserted links can change which nearby lane has a complete
            # forward continuation; rank again against the verified order.
            public_current = select(linked_public) if linked_public is not None else None
            if (linked_public is not None
                    and public_current is not None
                    and (chain(public_current, linked_public)[2] >= required_forward
                         or (chain(public_current, linked_public)[2]>1.
                             and (current is None or chain(current,order)[2]<=1.)))):
                order, current = linked_public, public_current
                rejection_reason = "corrected_route_has_no_valid_current_association;connected_public_route_retained"
    if current is None or nearest(current)[0] or nearest(current)[2] > 8:
        raise ValueError("Current ego pose cannot be associated with the mapped route after topology correction")
    current_id = current["id"]
    route, ids, forward, route_turn, command_disagreement = chain(current, order)
    if len(route)<2 or not np.isfinite(route).all() or LineString(route).length<=1e-6:
        raise ValueError('Mapped navigation has no finite nondegenerate directed reference')
    observed = [lane for lane in lanes if associations[lane["id"]]["polygon_covers_rear_axle"]
                and not nearest(lane)[0]]
    observed_current = min(observed, key=nearest) if observed else current
    route_evidence = dict(public_roadblock_ids=public_order, used_roadblock_ids=order,
        attempted_roadblock_ids=attempted_order, topology_correction_rejection_reason=rejection_reason,
        used_route_source=("public_route_order_with_verified_directed_map_links" if rejection_reason else
                           "pinned_official_pdm_route_roadblock_correction"),
        public_route_inserted_roadblock_ids=([key for key in order if key not in public_order] if rejection_reason else []),
        topology_correction_attempted=True, topology_correction_changed=order != public_order,
        correction_source="pinned_official_pdm_route_roadblock_correction_current_pose_map_public_route",
        initial_lane_id=initial_lane, current_lane_id=current_id,
        observed_current_lane_id=observed_current["id"],
        forward_reference_m=forward, required_forward_reference_m=required_forward,
        reference_coverage_complete=bool(forward+1e-6>=required_forward),
        reference_quality_reasons=([] if forward+1e-6>=required_forward else ['public_forward_reference_short']),
        navigation_command=command, reference_heading_ahead_rad=route_turn,
        command_disagreement=command_disagreement,
        lane_roadblock_ids={key: block(key) for key in table},
        **associations[current_id])
    # Match official PDM's DAC authorization. LANE/CONNECTOR polygons are
    # navigation semantics, not additional independently drivable surfaces.
    road = unary_union([_local_shape(obj.polygon, origin) for layer in road_layers for obj in nearby[layer]])
    if road.is_empty or not road.is_valid or road.area <= 0:
        raise ValueError('NAVSIM official drivable road layers are empty or invalid')
    road, _, _ = drivable_roundtrip(road)
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


def build_packet(agent_input, scene, camera_dir, desired_speed_mps=10.0, map_radius_m=160.0,
                 benchmark_profile="v2_two_stage"):
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from .profiles import is_v1

    metadata = scene.scene_metadata
    if metadata.num_history_frames != 4 or len(agent_input.ego_statuses) != 4:
        raise ValueError("The official NAVSIM evaluation input must have four history/current frames")
    legacy_scene = is_v1({"benchmark_profile": benchmark_profile})
    # v1.1 has no synthetic-scene provenance fields. Null explicitly means
    # inapplicable there; a v2 Scene must still provide its official fields.
    corresponding_scene = None if legacy_scene else metadata.corresponding_original_scene
    corresponding_token = None if legacy_scene else metadata.corresponding_original_initial_token
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
    ego = navigation_ego(now.ego_status.ego_velocity,omega,vehicle)
    ego.update(acceleration=acceleration.tolist(),visible=True,history=ego_history,radius=None)
    command = navigation_command(now.ego_status.driving_command)
    lanes, route, ids, current, road, crosswalks, objects, skipped, route_evidence = _map_packet(
        scene.map_api, origin, now.roadblock_ids, desired_speed_mps, map_radius_m,
        current_speed_mps=float(np.linalg.norm(velocity)), driving_command=command["one_hot"])
    # Physical lane occupancy and the lane that can continue the public route
    # are different facts near junctions and before a necessary lane change.
    ego["lane_id"] = route_evidence["observed_current_lane_id"]
    route, initialization = validate_navigation_initialization(
        ego,lanes,route,ids,road,route_evidence,omega,now.token)
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
    road, drivable_geometry, geometry_evidence = drivable_roundtrip(road)
    return validate_packet(dict(protocol=INPUT_PROTOCOL, input_condition=INPUT_CONDITION,
        identity=dict(token=str(now.token), log_name=metadata.log_name, scene_token=metadata.scene_token,
                      map_name=metadata.map_name, timestamp_us=int(now.timestamp), history_frames=4,
                      history_timestamps_us=ts.tolist(),
                      corresponding_original_scene=corresponding_scene,
                      corresponding_original_initial_token=corresponding_token),
        vehicle=dict(length_m=float(vehicle.length), width_m=float(vehicle.width),
                     wheelbase_m=float(vehicle.wheel_base), rear_axle_to_center_m=rear),
        ego=ego, actors=actors, lanes=lanes, route=route.tolist(), route_lane_ids=ids, drivable_geometry=drivable_geometry,
        rules=dict(desired_speed=min(desired_speed_mps, speed_limit), speed_limit=speed_limit,
                   navigation_command=command,
                   must_wait=stop_s is not None, wait_reason="red_light" if stop_s is not None else None,
                   stop_s=stop_s, crosswalks=crosswalks, traffic_lights=signals,
                   map_available=dict(crosswalk=True, traffic_light=True, stop_line=True),
                   road_priority_available=False),
        diagnostics=dict(route_s=route_s, reference_anchor_s=route_s, stable_navigation_lane_id=current,
            lane_map_complete=True, reference_revision=0, has_occlusion=False,
            visibility_source='scene_drivable_no_occlusion', geometry_transfer=geometry_evidence,
            lane_map_complete_scope="local_map_query_radius_%gm" % map_radius_m,
            available_observation_history_s=float((ts[-1]-ts[0])/1e6),
            observation_policy=INPUT_CONDITION, previous_control=[float(np.clip(acceleration[0], -6, 3)),
                float(np.clip(math.atan(vehicle.wheel_base * omega / max(abs(velocity[0]), 0.3)), -0.6, 0.6))],
            previous_control_source="causal_observed_acceleration_and_yaw_rate_estimate_not_actuator_truth",
            acceleration_source="current_ego_body_state_and_history_track_velocity_difference",
            source_rear_axle_global_pose=origin.tolist(), ignored_annotation_types=ignored, skipped_map_lanes=skipped,
            route_association=route_evidence,
            navigation_initialization=initialization,
            navigation_lane_groups=route_evidence["lane_roadblock_ids"],
            lane_crossability_source="nuplan_directed_adjacency_not_lane_paint_annotations",
            route_source="current_privileged_roadblock_ids_and_map_topology_no_future_ego_poses"),
        camera_images=cameras))
