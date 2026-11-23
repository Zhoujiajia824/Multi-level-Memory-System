"""Calibrated camera overlays for authenticated offline evidence, without IO.

The library store authenticates all metadata and images before calling this
module. Native dataset extrinsics are camera-to-ego, with optical z forward;
they are deliberately separate from CARLA's Unreal camera convention.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
          (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
_NEAR = .1


def _finite(value, shape=None):
    array = np.asarray(value, dtype=float)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError("标定或坐标缺少有限数值/正确维度")
    return array


def quaternion_matrix(value):
    """Return a proper rotation for dataset quaternions ordered w,x,y,z."""
    q = _finite(value, (4,))
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.) > .001:
        raise ValueError("四元数未归一化")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _pose(value):
    return quaternion_matrix(value.get("rotation")), _finite(value.get("translation"), (3,))


def camera_model(camera):
    """Compose global -> exposure ego -> optical camera, using its own pose."""
    calibration = camera.get("calibration") or {}
    rc, tc = _pose(calibration)
    re, te = _pose(camera.get("ego_pose") or {})
    intrinsic = _finite(calibration.get("camera_intrinsic"), (3, 3))
    width, height = int(camera.get("width", 0)), int(camera.get("height", 0))
    if width <= 0 or height <= 0 or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("图像尺寸或相机内参无效")
    if not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-7):
        raise ValueError("不支持的非针孔相机内参")
    # In OpenScene calibration already composes sensor2lidar and lidar2ego.
    # raw_sensor2lidar is not substituted here or labelled camera-to-ego.
    rotation = rc.T @ re.T
    translation = -rotation @ te - rc.T @ tc
    return dict(rotation=rotation, translation=translation, intrinsic=intrinsic,
                width=width, height=height, ego_rotation=re, ego_translation=te)


def _camera_point(point, model):
    return model["rotation"] @ _finite(point, (3,)) + model["translation"]


def _pixel(point, model):
    pixel = model["intrinsic"] @ point
    return pixel[:2] / pixel[2]


def _clip_image(a, b, width, height, return_parameters=False):
    delta, lo, hi = b - a, 0., 1.
    for p, q in ((-delta[0], a[0]), (delta[0], width-a[0]),
                 (-delta[1], a[1]), (delta[1], height-a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            ratio = q / p
            if p < 0:
                lo = max(lo, ratio)
            else:
                hi = min(hi, ratio)
            if lo > hi:
                return None
    points = [(a + lo*delta).tolist(), (a + hi*delta).tolist()]
    return (points, (lo, hi)) if return_parameters else points


def _project_segment_progress(a, b, model):
    """Return pixels and fractions of the original, unclipped 3D segment.

    Viewport clipping happens in perspective image space. Its interpolation
    fractions must be converted back using depth before colouring by original
    recorded point order; visible fragments must not acquire a new timeline.
    """
    a, b = _camera_point(a, model), _camera_point(b, model)
    if a[2] < _NEAR and b[2] < _NEAR:
        return None
    start, end = 0., 1.
    if a[2] < _NEAR:
        start = (_NEAR-a[2]) / (b[2]-a[2])
        a = a + start * (b-a)
    elif b[2] < _NEAR:
        end = (_NEAR-a[2]) / (b[2]-a[2])
        b = a + end * (b-a)
    clipped = _clip_image(_pixel(a, model), _pixel(b, model),
        model["width"], model["height"], return_parameters=True)
    if clipped is None:
        return None
    pixels, screen_fractions = clipped
    fractions = []
    for fraction in screen_fractions:
        world_fraction = fraction * a[2] / ((1.-fraction)*b[2] + fraction*a[2])
        fractions.append(float(start + (end-start)*world_fraction))
    return pixels, fractions


def project_segment(a, b, model):
    """Clip the actual segment at the optical near plane and image border."""
    projected = _project_segment_progress(a, b, model)
    return projected[0] if projected is not None else None


def _project_point(point, model):
    camera_point = _camera_point(point, model)
    if camera_point[2] < _NEAR:
        return None
    pixel = _pixel(camera_point, model)
    return pixel.tolist() if (0 <= pixel[0] <= model["width"] and
        0 <= pixel[1] <= model["height"]) else None


def _clip_polygon(points, inside, intersect):
    result = []
    if not points:
        return result
    previous = points[-1]
    previous_inside = inside(previous)
    for point in points:
        current_inside = inside(point)
        if current_inside != previous_inside:
            result.append(intersect(previous, point))
        if current_inside:
            result.append(point)
        previous, previous_inside = point, current_inside
    return result


def _project_polygon(points, model):
    points = [_camera_point(point, model) for point in points]
    points = _clip_polygon(points, lambda p: p[2] >= _NEAR,
        lambda a, b: a + ((_NEAR-a[2])/(b[2]-a[2]))*(b-a))
    points = [_pixel(point, model) for point in points]
    for axis, edge, lower in ((0, 0., True), (0, float(model["width"]), False),
                              (1, 0., True), (1, float(model["height"]), False)):
        points = _clip_polygon(points, lambda p: p[axis] >= edge if lower else p[axis] <= edge,
            lambda a, b: a + ((edge-a[axis])/(b[axis]-a[axis]))*(b-a))
    return [point.tolist() for point in points] if len(points) >= 3 else []


def _ground_point(value, model):
    """World XY lifted to exposure ego's local z=0 plane, explicitly approximate."""
    xy = _finite(value)[:2]
    if xy.shape != (2,):
        raise ValueError("实驶轨迹缺少世界XY")
    normal, origin = model["ego_rotation"][:, 2], model["ego_translation"]
    if abs(normal[2]) < .5:
        raise ValueError("相机位姿不能定义合理局部地面")
    z = origin[2] - float(np.dot(normal[:2], xy-origin[:2]))/normal[2]
    return np.r_[xy, z]


def _recorded_future(frame, model):
    overlay = {"trajectory_segments": [], "trajectory_segments_progress": [],
        "trajectory_samples": [], "trajectory_source_point_count": 0, "corridor_polygons": []}
    vis = frame.get("visualization") or {}
    provenance = vis.get("trajectory_provenance") or {}
    if provenance.get("kind") != "recorded_future_execution":
        return overlay, False
    if (provenance.get("anchor_frame_id") != frame["frame_id"] or
            not _same_time(provenance.get("time"), frame["time"])):
        return overlay, False
    source = vis.get("trajectory") or []
    lines = vis.get("trajectory_segments") or [source]
    count = len(source)
    denominator = max(count-1, 1)
    overlay["trajectory_source_point_count"] = count
    points = [_ground_point(point, model) for point in source]
    for index, point in enumerate(points):
        pixel = _project_point(point, model)
        if pixel is not None:
            overlay["trajectory_samples"].append(
                {"point": pixel, "index": index, "progress": index / denominator})
    # Segment arrays preserve discontinuities. Find their points in the
    # original ordered source rather than numbering visible fragments anew.
    cursor = 0
    for line in lines:
        indices = []
        for point in line:
            index = next((i for i in range(cursor, count)
                if np.allclose(source[i], point, rtol=0., atol=1e-6)), None)
            if index is None:
                indices.append(None)
                continue
            indices.append(index)
            cursor = index + 1
        if indices and indices[-1] is not None:
            cursor = indices[-1]  # Adjacent polylines may share one endpoint.
        for first, last in zip(indices, indices[1:]):
            if first is None or last != first + 1:
                continue  # Never bridge an omitted source point or a gap.
            projected = _project_segment_progress(points[first], points[last], model)
            if projected is not None:
                pixels, fractions = projected
                overlay["trajectory_segments"].append(pixels)
                overlay["trajectory_segments_progress"].append(
                    [(first+fraction) / denominator for fraction in fractions])
    polygons = []
    ribbon = vis.get("corridor_ribbon") or {}
    if ribbon.get("source") == "recorded_future_execution_schematic_ribbon":
        # Triangulation/intersection preserves archived road holes. Drawing just
        # outer rings would incorrectly cover islands or unavailable road.
        from shapely.geometry import Polygon
        from shapely.ops import triangulate
        for shape in ribbon.get("polygons") or []:
            polygon = Polygon([p[:2] for p in shape.get("outer", [])],
                              [[p[:2] for p in ring] for ring in shape.get("holes", [])])
            if polygon.is_empty or not polygon.is_valid:
                continue
            for triangle in triangulate(polygon):
                clipped = triangle.intersection(polygon)
                pieces = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
                for piece in pieces:
                    if piece.geom_type != "Polygon" or piece.is_empty or piece.interiors:
                        continue
                    projected = _project_polygon([_ground_point(p, model)
                        for p in list(piece.exterior.coords)[:-1]], model)
                    if projected:
                        polygons.append(projected)
    overlay["corridor_polygons"] = polygons
    return overlay, True


def _same_time(a, b):
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= 1e-5
    except (ValueError, TypeError):
        return False


def _raw_boxes(frame, context):
    """Only matched current actors with preserved measured 3D local boxes."""
    re, te = _pose(context.get("raw_ego_pose") or {})
    actors = {str(row.get("id")): row for row in frame.get("actors") or []}
    boxes = []
    for annotation in context.get("visible_annotation_candidates") or []:
        if not isinstance(annotation, dict):
            continue
        identity = str(annotation.get("source_id"))
        actor = actors.get(identity)
        if actor is None or not _same_time(annotation.get("time"), frame["time"]):
            continue
        try:
            local = _finite(annotation.get("raw_local_box"), (7,))
            measured = _finite(annotation.get("position_3d_for_camera"), (3,))
            if (np.linalg.norm(_finite(actor.get("position"), (2,))-
                    _finite(annotation.get("position"), (2,))) > .001 or
                    np.linalg.norm(re @ local[:3] + te - measured) > .001):
                continue
            if min(local[3:6]) <= 0:
                continue
            yaw = local[6]
            rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                           [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
            dimensions = local[3:6] / 2
            corners = []
            for zsign in (-1, 1):
                for xsign, ysign in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    offset = dimensions * [xsign, ysign, zsign]
                    corners.append(re @ (local[:3] + rz @ offset) + te)
            boxes.append((identity, corners))
        except (ValueError, TypeError):
            continue
    return boxes


def attach_library_cameras(frame, data):
    """Return a copied frame with verified evidence images and native overlays.

    ``data`` is the LibraryStore contract: source, frame_id, time, available,
    integrity, original_scene, manifest, exports, and images with authenticated
    ``view/path/url/sha256/registered/sha256_verified``. No external path is read.
    Camera time binds the selected state; timestamp_s/exposure_timestamp_s and
    time_offset_s separately describe the actual source-image exposure.
    """
    result = deepcopy(frame)
    result["cameras"], result["camera_overlays"] = [], {}
    status = {"available": False, "source": "authenticated_offline_camera_assets"}
    result["library_camera_status"] = status
    if not isinstance(data, dict) or not data.get("available"):
        status["reason"] = data.get("reason", "本证据无可验证的归档相机") if isinstance(data, dict) else "本证据无可验证的归档相机"
        return result
    source = data.get("source")
    if source not in ("openscene", "nuscenes"):
        status["reason"] = "highway-env只有环境俯视图，没有前视相机或透视标定"
        return result
    original = data.get("original_scene") or {}
    context = original.get("current_context") or {}
    if (data.get("integrity") != "verified_current_evidence_assets" or
            data.get("frame_id") != frame.get("frame_id") or
            not _same_time(data.get("time"), frame.get("time")) or
            original.get("source") != source or
            not _same_time(context.get("timestamp_s"), frame.get("time"))):
        status["reason"] = "相机元数据与当前记录/证据帧身份或时间不一致，拒绝投影"
        return result
    manifests, exports = {}, {}
    for row in data.get("manifest") or []:
        if isinstance(row, dict):
            manifests.setdefault(row.get("view"), []).append(row)
    for row in data.get("exports") or []:
        if isinstance(row, dict):
            exports.setdefault(row.get("view"), []).append(row)
    try:
        measured_boxes = _raw_boxes(frame, context) if source == "openscene" else []
    except (ValueError, TypeError):
        measured_boxes = []
    archived_matches = {str(row.get("id")): row for row in
        (frame.get("visualization") or {}).get("objects") or []}
    images = [image for image in data.get("images") or [] if isinstance(image, dict)]
    image_counts = {}
    for image in images:
        image_counts[image.get("view")] = image_counts.get(image.get("view"), 0) + 1
    for image in images:
        view = image.get("view")
        if (image_counts[view] != 1 or len(manifests.get(view, [])) != 1 or
                len(exports.get(view, [])) != 1 or not image.get("registered") or
                not image.get("sha256_verified") or not image.get("url")):
            continue
        camera, exported = manifests[view][0], exports[view][0]
        if (not image.get("sha256") or image["sha256"] != exported.get("sha256") or
                image["sha256"] != camera.get("sha256") or
                camera.get("source_filename") != exported.get("source_filename") or
                not _same_time(camera.get("sample_timestamp_s"), frame["time"]) or
                not _same_time(camera.get("timestamp_s"), exported.get("timestamp_s"))):
            continue
        bound = {"view": view, "path": image.get("path"), "url": image["url"],
            "frame_id": frame["frame_id"], "time": frame["time"],
            "width": camera.get("width"), "height": camera.get("height"),
            "timestamp_s": camera.get("timestamp_s"),
            "exposure_timestamp_s": camera.get("exposure_timestamp_s"),
            "time_offset_s": camera.get("time_offset_s"),
            "timestamp_source": camera.get("timestamp_source"), "source": source,
            "evidence_index": data.get("evidence_index"),
            "meta": {"source_frame_id": original.get("frame_id")}}
        result["cameras"].append(bound)
        overlay = {"valid": False, "frame_id": frame["frame_id"], "time": frame["time"],
            "width": camera.get("width"), "height": camera.get("height"),
            "boxes": [], "trajectory_segments": [], "trajectory_segments_progress": [],
            "trajectory_samples": [], "trajectory_source_point_count": 0, "corridor_polygons": [],
            "region_polygons": [], "goal": None, "planned_trajectory": False,
            "time_offset_s": camera.get("time_offset_s")}
        result["camera_overlays"][view] = overlay
        try:
            model = camera_model(camera)
            for identity, corners in measured_boxes:
                segments = [segment for a, b in _EDGES
                    if (segment := project_segment(corners[a], corners[b], model)) is not None]
                if segments:
                    match = archived_matches.get(identity) or {}
                    overlay["boxes"].append({"id": identity, "segments": segments,
                        "geometry_quality": "measured_source_3d_box",
                        "attention": False, "potential_source": False,
                        "grid_matched": bool(match.get("grid_matched")),
                        "grid_roles": list(match.get("grid_roles") or [])})
            trajectory, future = _recorded_future(frame, model)
            overlay.update(trajectory, valid=True, recorded_future_execution=future)
            offset = float(camera["timestamp_s"]) - float(frame["time"])
            if not _same_time(camera.get("time_offset_s"), offset):
                raise ValueError("相机曝光偏移与归档timestamp不一致")
            notes = ["原生针孔标定；使用该相机自己的ego_pose，未套用CARLA坐标"]
            if source == "openscene":
                notes.append("OpenScene只保证源帧关联，独立曝光时间未知；保存畸变参数但图像矫正状态未知，按官方可视化针孔近似，不宣称像素精确")
                notes.append("对象框仅来自ID和当前XY一致的原始full-SE3三维标注，不使用示意框高")
            else:
                notes.append("nuScenes原图曝光相对状态%+.6fs；当前库缺对象z，不绘制真实三维框" % offset)
            if future:
                notes.append("线为归档后续真实自车中心实驶记录，不是在线预测或规划；走廊仅按车宽示意，世界XY以相机ego局部地平面抬升，坡度/地面高度未测量")
            notes.append("叠加层未重建像素级深度遮挡，不推断物体实际可见像素")
            overlay["projection_note"] = "；".join(notes)
            overlay["projection_quality"] = "native_pinhole_with_explicit_ground_approximation"
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            overlay.update(valid=False, reason="原始图像可读，但不能验证投影：" + str(exc),
                           boxes=[], trajectory_segments=[], trajectory_segments_progress=[],
                           trajectory_samples=[], trajectory_source_point_count=0, corridor_polygons=[])
    front = "CAM_F0" if source == "openscene" else "CAM_FRONT"
    result["cameras"].sort(key=lambda item: (item["view"] != front, item["view"]))
    valid = sum(bool(overlay["valid"]) for overlay in result["camera_overlays"].values())
    status.update(available=bool(result["cameras"]), camera_count=len(result["cameras"]),
        calibrated_count=valid, front_view=front,
        front_available=any(item["view"] == front for item in result["cameras"]),
        reason=None if result["cameras"] else "缺少唯一且SHA一致的本证据图像/标定关联")
    capabilities = (result.get("library_scene") or {}).get("capabilities")
    if isinstance(capabilities, dict):
        capabilities["camera_projection"] = {"available": bool(valid), "source": source,
            "calibrated_count": valid, "note": "原生相机坐标，曝光偏移/地平面近似见每幅图说明"}
    return result
