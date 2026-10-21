"""Project true world polylines into a captured camera image for readable video.

CARLA shipping debug primitives are HDR emissive. This optional, post-capture
display uses ordinary RGB, never changes planning geometry or cognition images,
and uses the image's capture transform (not the camera's later live transform).
It is an annotation, with no depth-buffer occlusion against scene objects.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from carla_bridge.visualization.trajectory_renderer import MAX_STROKE_SEGMENTS, _polyline, _at_distance


def _pieces(points, distances, dashed, phase, dash, gap):
    period = dash + gap
    estimate = len(points) - 1 + (2 * math.ceil(distances[-1] / period) + 4 if dashed else 0)
    if estimate > MAX_STROKE_SEGMENTS:
        raise ValueError("Trajectory exceeds the camera display segment budget")
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        start, end = distances[i:i + 2]
        cuts = [start, end]
        if dashed:
            cuts += [v for k in range(math.floor((start + phase) / period) - 1,
                                      math.ceil((end + phase) / period) + 2)
                     for v in (k * period - phase, k * period + dash - phase) if start < v < end]
        cuts.sort()
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if dashed and ((lo + hi) * .5 + phase) % period >= dash:
                continue
            yield a + (b - a) * ((lo - start) / (end - start)), a + (b - a) * ((hi - start) / (end - start))


def _clip_viewport(a, b, width, height):
    """Clip in float before converting to OpenCV's bounded integer coords."""
    delta = b - a
    lo, hi = 0., 1.
    for p, q in ((-delta[0], a[0]), (delta[0], width - 1 - a[0]),
                 (-delta[1], a[1]), (delta[1], height - 1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            lo = max(lo, q / p)
        else:
            hi = min(hi, q / p)
        if lo > hi:
            return None
    return np.rint(a + lo * delta).astype(int), np.rint(a + hi * delta).astype(int)


def _project_segment(a, b, inverse, focal, width, height):
    # UE camera axes are forward X, right Y, up Z.
    camera = (inverse @ np.column_stack(([a, b], [1., 1.])).T).T[:, :3]
    near = .2
    if np.max(camera[:, 0]) < near:
        return None
    for i in (0, 1):
        if camera[i, 0] < near:
            other = camera[1 - i]
            u = (near - camera[i, 0]) / (other[0] - camera[i, 0])
            camera[i] += u * (other - camera[i])
    pixels = np.column_stack((width * .5 + focal * camera[:, 1] / camera[:, 0],
                              height * .5 - focal * camera[:, 2] / camera[:, 0]))
    if not np.isfinite(pixels).all():
        return None
    clipped = _clip_viewport(pixels[0], pixels[1], width, height)
    return (*clipped, float(camera[:, 0].mean())) if clipped is not None else None


def _markers(points, distances, dashed, width):
    end = points[-1]
    if distances[-1] < .15:
        radius = .34 if width >= .2 else .22
        return [[end + [-radius, 0., 0.], end + [radius, 0., 0.]],
                [end + [0., -radius, 0.], end + [0., radius, 0.]]]
    previous = _at_distance(points, distances, [max(0., distances[-1] - .8)])[0]
    tangent = end[:2] - previous[:2]
    if np.linalg.norm(tangent) < 1e-6:
        return []
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    left, right = end.copy(), end.copy()
    half = .25 if dashed else .40
    left[:2] += normal * half - (0. if dashed else .6) * tangent
    right[:2] -= normal * half + (0. if dashed else .6) * tangent
    return [[left, right]] if dashed else [[left, end, right]]


def draw_trajectory_overlay(image, display, camera_transform, fov_degrees=90.):
    """Annotate an independent BGR video frame in place; return display audit.

    All XY belong to actual evaluated/executed polylines. Only widths have
    pixel readability bounds. Metric gaps and bend vertices are preserved.
    Invalid individual paths are omitted, not repaired into plausible paths.
    """
    audit = {"enabled": False, "drawn_paths": 0, "drawn_segments": 0,
             "camera_transform_source": "captured_image", "depth_occlusion": False}
    if not display or display.get("enabled") is False or display.get("style") != "paper":
        return audit
    audit["enabled"] = True
    paths = display.get("paths") or []
    if not paths:
        return audit
    if camera_transform is None:
        audit["skipped_reason"] = "missing_capture_transform"
        return audit
    inverse = np.asarray(camera_transform.get_inverse_matrix(), dtype=float)
    if inverse.shape != (4, 4) or not np.isfinite(inverse).all():
        raise ValueError("Camera capture transform must be a finite 4x4 inverse")
    fov = float(fov_degrees)
    if not math.isfinite(fov) or not 1. < fov < 179.:
        raise ValueError("Camera horizontal FOV must be between 1 and 179 degrees")
    height, width = image.shape[:2]
    focal = width / (2 * math.tan(math.radians(fov) * .5))
    dash, gap = float(display.get("dash_length_m", 1.2)), float(display.get("dash_gap_m", .8))
    outline_m = float(display.get("outline_width_m", .045))
    if not all(math.isfinite(v) and v >= .05 for v in (dash, gap)):
        raise ValueError("Camera dash lengths must be finite and >= 0.05 m")
    if not math.isfinite(outline_m) or outline_m <= 0:
        raise ValueError("Camera outline width must be positive and finite")
    original, ego_mask = None, None
    if display.get("ego_box_world") is not None:
        try:
            box = np.asarray(display["ego_box_world"], dtype=float)
            if box.shape != (8, 3) or not np.isfinite(box).all():
                raise ValueError("Invalid ego display box")
            camera = (inverse @ np.column_stack((box, np.ones(8))).T).T[:, :3]
            if np.all(camera[:, 0] >= .2):
                pixels = np.column_stack((width * .5 + focal * camera[:, 1] / camera[:, 0],
                                          height * .5 - focal * camera[:, 2] / camera[:, 0]))
                hull = cv2.convexHull(np.rint(np.clip(pixels, -max(width, height) * 2,
                                                     max(width, height) * 3)).astype(np.int32))
                ego_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(ego_mask, hull, 1)
                original = image.copy()
        except (ValueError, TypeError):
            audit["ego_mask_skipped"] = "invalid_ego_box"
    audit["skipped_paths"] = []
    # Ordering is explicit even when a caller provides an unsorted display list.
    valid_paths = [p for p in paths if isinstance(p, dict)]
    audit["skipped_paths"].extend({"id": None, "reason": "invalid_path"} for p in paths if not isinstance(p, dict))
    order = {"candidate": 0, "selected": 1, "execution": 2, "rejected": 2}
    for path in sorted(valid_paths, key=lambda p: order.get(p.get("kind"), 0)):
        try:
            points, distances = _polyline(path["points_world"])
            rgb = np.asarray(path["rgb"], dtype=float)
            stroke_m = float(path["width_m"])
            phase = float(path.get("phase_m", 0.))
            if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 255):
                raise ValueError("Camera display color must be RGB in [0, 255]")
            if not math.isfinite(stroke_m) or stroke_m <= 0 or not math.isfinite(phase):
                raise ValueError("Camera display stroke must be positive and finite")
            dashed = bool(path.get("dashed"))
            # Resolve all fragments before touching pixels: one malformed path
            # cannot partially draw and hide otherwise valid annotations.
            pieces = list(_pieces(points, distances, dashed, phase, dash, gap))
            if display.get("endpoint_markers", True) and path.get("marker", True):
                for symbol in _markers(points, distances, dashed, stroke_m):
                    pieces.extend(zip(symbol[:-1], symbol[1:]))
            projected = [segment for a, b in pieces
                         if (segment := _project_segment(a, b, inverse, focal, width, height)) is not None]
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            audit["skipped_paths"].append({"id": path.get("id"), "reason": str(exc)})
            continue
        kind = path.get("kind")
        lower, upper = (7, 18) if kind == "selected" else (3, 9) if kind in ("execution", "rejected") else (2, 5)
        bgr = tuple(int(v) for v in rgb[::-1])
        for a, b, depth in projected:
            core_px = int(np.clip(round(stroke_m * focal / depth), lower, upper))
            outline_px = int(np.clip(round(outline_m * focal / depth), 1, 2))
            cv2.line(image, tuple(a), tuple(b), (12, 18, 25), core_px + 2 * outline_px, cv2.LINE_AA)
        for a, b, depth in projected:
            core_px = int(np.clip(round(stroke_m * focal / depth), lower, upper))
            cv2.line(image, tuple(a), tuple(b), bgr, core_px, cv2.LINE_AA)
        audit["drawn_segments"] += len(projected)
        audit["drawn_paths"] += bool(projected)
    if ego_mask is not None:
        image[ego_mask.astype(bool)] = original[ego_mask.astype(bool)]
        audit["ego_box_masked"] = True
    return audit
