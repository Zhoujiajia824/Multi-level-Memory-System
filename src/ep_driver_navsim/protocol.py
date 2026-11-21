"""Small, explicit interchange contract between the SDK and numerical runtimes."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import INPUT_PROTOCOL, INPUT_CONDITION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def transform_points(points, origin):
    """Global XY -> current rear-axle frame, with left-positive Y."""
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return (np.asarray(points, dtype=float) - np.asarray(origin[:2])) @ np.array([[c, -s], [s, c]])


def rotate_vectors(values, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(values, dtype=float) @ np.array([[c, s], [-s, c]])


def validate_packet(packet):
    """No Scene/pickle/metric cache is ever sent to the numerical process."""
    expected = {"protocol", "input_condition", "identity", "vehicle", "ego", "actors", "lanes",
                "route", "route_lane_ids", "drivable_wkt", "rules", "diagnostics", "camera_images"}
    if set(packet) != expected or packet["protocol"] != INPUT_PROTOCOL:
        raise ValueError("Unknown NAVSIM interchange schema")
    if packet["input_condition"] != INPUT_CONDITION:
        raise ValueError("This adapter supports the explicitly privileged input condition only")
    identity = packet["identity"]
    if (identity["history_frames"] != 4 or len(identity["history_timestamps_us"]) != 4
            or identity["history_timestamps_us"][-1] != identity["timestamp_us"]
            or any(b <= a for a, b in zip(identity["history_timestamps_us"], identity["history_timestamps_us"][1:]))):
        raise ValueError("Expected four ordered history/current frames")
    for actor in [packet["ego"]] + packet["actors"]:
        for history in actor["history"]:
            if not math.isfinite(history["time"]) or history["time"] > 1e-8:
                raise ValueError("Future observations cannot enter planning")
        values = np.r_[actor["position"], actor["velocity"], actor["acceleration"],
                       actor["heading"], actor["length"], actor["width"]]
        if not np.isfinite(values).all() or min(actor["length"], actor["width"]) <= 0:
            raise ValueError("Invalid current actor geometry/kinematics")
    route = np.asarray(packet["route"], float)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("A finite map-based route is required")
    for camera in packet["camera_images"]:
        if camera["frame_id"] != identity["token"] or camera["time"] != 0.0:
            raise ValueError("Camera does not belong to the current anchor")
    # Recursively reject NaN/Infinity, including map coordinates and rule data.
    json.dumps(packet, allow_nan=False)
    return packet


def trajectory_from_centers(states, step_s, rear_axle_to_center_m, horizon_s=4.0, interval_s=0.5):
    """Return eight future rear-axle SE(2) poses; never pad/extrapolate a 3s plan."""
    states = np.asarray(states, float)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("Invalid planner states")
    if step_s <= 0 or (len(states) - 1) * step_s < horizon_s - 1e-8:
        raise ValueError("Planner horizon does not cover the official four seconds")
    origin = states[0, :2] - rear_axle_to_center_m * np.array([np.cos(states[0, 2]), np.sin(states[0, 2])])
    rear = states[:, :2] - rear_axle_to_center_m * np.c_[np.cos(states[:, 2]), np.sin(states[:, 2])]
    rear = transform_points(rear, np.r_[origin, states[0, 2]])
    heading = np.unwrap(states[:, 2]) - states[0, 2]
    source_times = np.arange(len(states)) * step_s
    times = np.arange(1, int(round(horizon_s / interval_s)) + 1) * interval_s
    poses = np.c_[np.interp(times, source_times, rear[:, 0]), np.interp(times, source_times, rear[:, 1]),
                  wrap(np.interp(times, source_times, heading))]
    if poses.shape != (8, 3) or not np.isfinite(poses).all():
        raise ValueError("NAVSIM expects eight finite (x,y,heading) poses")
    return poses.astype(np.float32)
