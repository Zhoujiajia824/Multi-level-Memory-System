"""Small, explicit interchange contract between the SDK and numerical runtimes."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from nine_grid.paths import filesystem_path

from . import INPUT_PROTOCOL, INPUT_CONDITION

HORIZON_S = 4.0
PLANNING_STEP_S = 0.1
PLANNING_STEPS = 40
OUTPUT_INTERVAL_S = 0.5


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write_bytes(path, payload):
    """Publish a file atomically; a killed writer never exposes a partial JSON."""
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2,
                                allow_nan=False).encode("utf-8"))


def relative_asset_path(path, root):
    """Compare resolved filesystem paths in the same Windows path namespace.

    Both ordinary and extended drive/UNC paths receive the same normalization.
    Path.relative_to still enforces the directory boundary after symlink and '..'
    resolution; stripping a prefix or using a textual prefix test would not.
    """
    relative = filesystem_path(path).relative_to(filesystem_path(root))
    if not relative.parts:
        raise ValueError("An asset must be a file inside its sample directory")
    return relative.as_posix()


def horizon_contract(config, plans=None, predictions=None):
    """Check the actual grids, including the predictor's diagnostic horizon."""
    diagnostic_horizon = HORIZON_S + max(config.get("quality", {}).get("ttc_offsets_s", [0, .3, .6, .9]))
    if (config["mpc"]["horizon_steps"] != PLANNING_STEPS
            or not math.isclose(config["mpc"]["step_s"], PLANNING_STEP_S, abs_tol=1e-10)
            or any(not math.isclose(config["prediction"][key], expected, abs_tol=1e-10)
                   for key, expected in (("horizon_s", HORIZON_S),
                                         ("diagnostic_horizon_s", diagnostic_horizon),
                                         ("step_s", PLANNING_STEP_S)))):
        raise ValueError("NAVSIM planning, reference and prediction must use 40 x 0.1s = four seconds")
    report = dict(horizon_s=HORIZON_S, main_horizon_s=HORIZON_S,
                  diagnostic_horizon_s=diagnostic_horizon, step_s=PLANNING_STEP_S,
                  horizon_steps=PLANNING_STEPS, state_count=PLANNING_STEPS + 1,
                  output_interval_s=OUTPUT_INTERVAL_S, output_poses=8,
                  extrapolation=False, plans={}, predictions={})
    for name, plan in (plans or {}).items():
        states = np.asarray(plan.states, float)
        controls = np.asarray(plan.controls, float)
        if (states.shape != (PLANNING_STEPS + 1, 4) or not np.isfinite(states).all()
                or controls.shape != (PLANNING_STEPS, 2) or not np.isfinite(controls).all()):
            raise ValueError(name + " does not contain a complete finite four-second plan")
        report["plans"][name] = dict(state_count=len(states), control_count=len(controls), horizon_s=HORIZON_S)
    times = np.arange(round(diagnostic_horizon / PLANNING_STEP_S) + 1) * PLANNING_STEP_S
    for name, bundle in (predictions or {}).items():
        actual = np.asarray(bundle.times, float)
        if actual.shape != times.shape or not np.allclose(actual, times, rtol=0, atol=1e-10):
            raise ValueError(name + " does not cover the four-second plan and configured TTC diagnostic grid")
        for trajectory in bundle.trajectories.values():
            values = np.asarray(trajectory, float)
            if values.shape != (len(times), 4) or not np.isfinite(values).all():
                raise ValueError(name + " contains an incomplete/nonfinite actor prediction")
        report["predictions"][name] = dict(state_count=len(actual), horizon_s=float(actual[-1]),
            main_horizon_s=HORIZON_S, main_state_count=PLANNING_STEPS + 1,
            diagnostic_horizon_s=diagnostic_horizon, actor_count=len(bundle.trajectories))
    return report


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
