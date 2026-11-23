"""Measure axle geometry in one UE world frame, without origin assumptions.

CARLA 0.9.15 exposes PhysX wheel locations in UE world centimetres, but
large-map Actor snapshots in API-global metres. Raw Image.transform remains
in UE world metres. A same-frame, rigid camera pose supplies the missing
vehicle pose in that frame; no body-length or symmetric-axle fallback is used.
"""
from __future__ import annotations

import math


SCHEMA = "ep-driver-f2d-wheel-geometry-1.0"


def json_safe(value):
    """Keep invalid raw input inspectable without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


class WheelGeometryError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = json_safe(dict(evidence, status="invalid", error=message))


def _rigid_matrix(value):
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise ValueError("pose must be a 4x4 matrix")
    matrix = [[float(item) for item in row] for row in value]
    if not all(math.isfinite(item) for row in matrix for item in row):
        raise ValueError("pose contains nonfinite values")
    if any(abs(matrix[3][j] - (1. if j == 3 else 0.)) > 1e-7 for j in range(4)):
        raise ValueError("invalid homogeneous pose row")
    for i in range(3):
        for j in range(3):
            product = sum(matrix[k][i]*matrix[k][j] for k in range(3))
            if abs(product - (1. if i == j else 0.)) > 1e-5:
                raise ValueError("pose rotation is not orthonormal")
    r = matrix
    determinant = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
                   - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
                   + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if abs(determinant-1.) > 1e-5:
        raise ValueError("pose rotation is not proper")
    return matrix


def _inverse(matrix):
    result = [[matrix[j][i] for j in range(3)]+[0.] for i in range(3)]
    for i in range(3):
        result[i][3] = -sum(result[i][j]*matrix[j][3] for j in range(3))
    return result+[[0., 0., 0., 1.]]


def _multiply(left, right):
    return [[sum(left[i][k]*right[k][j] for k in range(4))
             for j in range(4)] for i in range(4)]


def measure_vehicle_geometry(*, actor_transform_matrix_m, wheels, bbox_center_m, bbox_extent_m,
                             sensor_pose, frame, timestamp, bbox_heading_deg=0.):
    """Return verified, fixed body geometry and auditable source measurements.

    The camera mount maps camera-local coordinates to vehicle-local coordinates.
    The raw image pose maps camera-local coordinates to UE-world coordinates.
    Wheel order is CARLA's front-left, front-right, rear-left, rear-right.
    """
    evidence = dict(schema=SCHEMA, frame=int(frame), timestamp=float(timestamp),
        source="same_frame_raw_image_pose_and_public_physics_wheels",
        wheel_position_unit="UE_world_cm", wheel_scale_to_m=.01,
        wheel_order=["front_left", "front_right", "rear_left", "rear_right"],
        actor_global_matrix_m=actor_transform_matrix_m, sensor_pose=sensor_pose,
        bbox_center_m=bbox_center_m, bbox_extent_m=bbox_extent_m, bbox_heading_deg=bbox_heading_deg,
        geometry_policy="measured_axles_only;no_length_or_symmetry_substitution")
    try:
        wheels = list(wheels)
        wheel_values = [[float(w.position.x), float(w.position.y), float(w.position.z)] for w in wheels]
        evidence["raw_wheels_cm"] = wheel_values
        radii = [float(w.radius) for w in wheels]
        evidence["wheel_radii_cm"] = radii
        if len(wheel_values) != 4:
            raise ValueError("ego execution requires four measured wheels")
        if not all(math.isfinite(v) for wheel in wheel_values for v in wheel):
            raise ValueError("wheel position contains nonfinite values")
        if not all(math.isfinite(radius) and radius > 0. for radius in radii):
            raise ValueError("wheel radius must be measured, finite and positive")
        if not math.isfinite(float(timestamp)):
            raise ValueError("invalid source timestamp")
        if int(sensor_pose["frame"]) != int(frame):
            raise ValueError("wheel geometry camera/state source frame mismatch")
        camera_time = float(sensor_pose["timestamp"])
        if not math.isfinite(camera_time) or abs(camera_time-float(timestamp)) > 1e-6:
            raise ValueError("wheel geometry camera/state timestamp mismatch")
        actor_global = _rigid_matrix(actor_transform_matrix_m)
        camera_world = _rigid_matrix(sensor_pose["camera_world_matrix_m"])
        camera_mount = _rigid_matrix(sensor_pose["camera_mount_matrix_m"])
        actor_ue = _multiply(camera_world, _inverse(camera_mount))
        evidence["actor_ue_world_matrix_m"] = actor_ue
        rotation_error = max(abs(actor_ue[i][j]-actor_global[i][j]) for i in range(3) for j in range(3))
        evidence["rotation_max_error"] = rotation_error
        evidence["global_minus_ue_origin_m"] = [actor_global[i][3]-actor_ue[i][3] for i in range(3)]
        if rotation_error > 1e-4:
            raise ValueError("raw camera and snapshot vehicle rotations disagree")
        inverse = _inverse(actor_ue)
        local = [[sum(inverse[i][j]*([v*.01 for v in wheel]+[1.])[j] for j in range(4))
                  for i in range(3)] for wheel in wheel_values]
        evidence["vehicle_local_wheels_m"] = local
        box_center = [float(v) for v in bbox_center_m]
        box_extent = [float(v) for v in bbox_extent_m]
        box_yaw = float(bbox_heading_deg)
        if (len(box_center) != 3 or len(box_extent) != 3
                or not all(math.isfinite(v) for v in box_center)
                or not all(math.isfinite(v) and v > 0. for v in box_extent)
                or not math.isfinite(box_yaw)):
            raise ValueError("invalid vehicle bounding-box frame")
        if abs(math.remainder(box_yaw, 360.)) > 1e-3:
            raise ValueError("ego bounding-box heading must align with the measured axle model")
        front, rear = sum(v[0] for v in local[:2])/2., sum(v[0] for v in local[2:])/2.
        wheelbase, rear_distance = front-rear, box_center[0]-rear
        axle_skew = max(abs(local[0][0]-local[1][0]), abs(local[2][0]-local[3][0]))
        tracks = [abs(local[0][1]-local[1][1]), abs(local[2][1]-local[3][1])]
        evidence.update(front_axle_x_m=front, rear_axle_x_m=rear,
                        wheelbase_m=wheelbase, rear_axle_to_center_m=rear_distance,
                        axle_longitudinal_skew_m=axle_skew, track_widths_m=tracks)
        # These are validity checks on measured geometry, never replacement values.
        if wheelbase <= .1 or min(tracks) <= .1 or axle_skew > .25:
            raise ValueError("degenerate or incorrectly ordered measured axles")
        if not 0. < rear_distance < wheelbase:
            raise ValueError("bounding-box center is outside the measured front/rear axles")
        # An incorrect lateral/vertical world origin can preserve axle spacings.
        # Check all coordinates against the actual body, with measured wheel
        # radius and small explicit suspension/asset tolerances. The body never
        # supplies the axle distances themselves.
        evidence["body_validation_margin_m"] = [.15, .15, .25]
        body_clearances = [[box_extent[j]+radii[i]*.01+(.25 if j == 2 else .15)
                            -abs(wheel[j]-box_center[j]) for j in range(3)]
                           for i, wheel in enumerate(local)]
        evidence["wheel_body_clearances_m"] = body_clearances
        if any(v < 0. for wheel in body_clearances for v in wheel):
            raise ValueError("measured wheel centers disagree with the vehicle bounding box")
        evidence["status"] = "valid"
        return json_safe(evidence)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise WheelGeometryError("Invalid measured ego wheel geometry: "+str(exc), evidence) from exc
