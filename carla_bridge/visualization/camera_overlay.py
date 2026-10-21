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

from carla_bridge.visualization.trajectory_renderer import MAX_STROKE_SEGMENTS, _polyline


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


def draw_trajectory_overlay(image, display, camera_transform, fov_degrees=90.):
    """Annotate an independent BGR video frame in place; return display audit.

    All XY belong to actual evaluated/executed polylines. Only widths have
    pixel readability bounds. Every adjacent pair is connected as a solid line;
    legacy dash/marker flags are deliberately ignored. There is no glow layer.
    Candidate opacity is composited once per complete path, so a real bend or
    a different waypoint density cannot make its joins artificially darker.
    Invalid individual paths are omitted, not repaired into plausible paths.
    """
    audit = {"enabled": False, "drawn_paths": 0, "drawn_segments": 0,
             "camera_transform_source": "captured_image", "depth_occlusion": False,
             "line_style": "solid", "endpoint_markers": False, "emissive": False,
             "opacity_compositing": "per_path", "width_reference_height_px": 720}
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
    outline_m = float(display.get("outline_width_m", 0.))
    if not math.isfinite(outline_m) or outline_m < 0:
        raise ValueError("Camera outline width must be nonnegative and finite")
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
    audit["skipped_paths"], audit["stationary_paths"] = [], []
    # Ordering is explicit even when a caller provides an unsorted display list.
    valid_paths = [p for p in paths if isinstance(p, dict)]
    audit["skipped_paths"].extend({"id": None, "reason": "invalid_path"} for p in paths if not isinstance(p, dict))
    order = {"candidate": 0, "selected": 1, "execution": 2, "rejected": 2}
    for path in sorted(valid_paths, key=lambda p: order.get(p.get("kind"), 0)):
        try:
            points, _ = _polyline(path["points_world"])
            rgb = np.asarray(path["rgb"], dtype=float)
            stroke_m = float(path["width_m"])
            alpha = float(path.get("alpha", 1.))
            if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 255):
                raise ValueError("Camera display color must be RGB in [0, 255]")
            if not math.isfinite(stroke_m) or stroke_m <= 0:
                raise ValueError("Camera display stroke must be positive and finite")
            if not math.isfinite(alpha) or not 0. <= alpha <= 1.:
                raise ValueError("Camera display alpha must be finite in [0, 1]")
            if len(points) - 1 > MAX_STROKE_SEGMENTS:
                raise ValueError("Trajectory exceeds the camera display segment budget")
            if len(points) < 2:
                audit["stationary_paths"].append({"id": path.get("id"), "kind": path.get("kind")})
                continue
            # Resolve all fragments before touching pixels: one malformed path
            # cannot partially draw and hide otherwise valid annotations.
            projected = [segment for a, b in zip(points[:-1], points[1:])
                         if (segment := _project_segment(a, b, inverse, focal, width, height)) is not None]
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            audit["skipped_paths"].append({"id": path.get("id"), "reason": str(exc)})
            continue
        kind = path.get("kind")
        # A 720p image uses a fine 1..2 px candidate, 2..3 px selection and
        # 3..5 px executed command. The latter covers a coincident selection,
        # instead of creating a thick selected shell with a narrow warm core.
        bounds = (2, 3) if kind == "selected" else (3, 5) if kind in ("execution", "rejected") else (1, 2)
        scale = height / 720.
        lower, upper = (max(1, int(math.floor(v * scale + .5))) for v in bounds)
        bgr = tuple(int(v) for v in rgb[::-1])
        canvas = image.copy() if alpha < 1. else image
        if outline_m > 0:
            for a, b, depth in projected:
                core_px = int(np.clip(round(stroke_m * focal / depth), lower, upper))
                outline_px = int(np.clip(round(outline_m * focal / depth), 1, max(1, round(2 * scale))))
                cv2.line(canvas, tuple(a), tuple(b), (45, 53, 61), core_px + 2 * outline_px, cv2.LINE_AA)
        for a, b, depth in projected:
            core_px = int(np.clip(round(stroke_m * focal / depth), lower, upper))
            cv2.line(canvas, tuple(a), tuple(b), bgr, core_px, cv2.LINE_AA)
        if alpha < 1.:
            cv2.addWeighted(canvas, alpha, image, 1. - alpha, 0., dst=image)
        audit["drawn_segments"] += len(projected)
        audit["drawn_paths"] += bool(projected)
    if ego_mask is not None:
        image[ego_mask.astype(bool)] = original[ego_mask.astype(bool)]
        audit["ego_box_masked"] = True
    return audit
