"""Draw the scored PDM rollouts and accepted command through CARLA debug RPCs.

All points are CARLA world actor positions, not ideal path references or nuPlan
rear-axle positions. Elevation follows the current resolved route's road surface.
Rendering is optional and may fail without altering a planning/control result.
"""
from __future__ import annotations

import logging
import math
from numbers import Real

import numpy as np

from carla_bridge.state.coords import ego_to_global

logger = logging.getLogger("carla_trajectory_visualization")

DEFAULTS = {
    "enabled": False,
    "max_candidates": 30,  # 0 draws the entire pool; the winner is always included.
    "point_stride": 1,
    "lifetime_s": 0.3,
    "thickness": 0.025,
    "z_offset": 0.12,
}

COLORS = {
    # CARLA 0.9.15 shipping multiplies linear debug color by 1000. Full 255
    # channels cause overlapping proposals to bloom into a white light sheet.
    "candidate": (0, 1, 2),
    "selected": (2, 1, 0),
    "execution": (0, 2, 1),
    "rejected": (2, 0, 0),
}

LINE_WIDTH_SCALE = {"candidate": 1., "selected": 1.6,
                    "execution": 2.8, "rejected": 2.8}


def validate_trajectory_visualization_config(config=None):
    if config is not None and not isinstance(config, dict):
        raise ValueError("carla.visualization.trajectories must be a mapping")
    values = dict(DEFAULTS, **(config or {}))
    if type(values["enabled"]) is not bool:
        raise ValueError("carla.visualization.trajectories.enabled must be boolean")
    for name, minimum in (("max_candidates", 0), ("point_stride", 1)):
        if type(values[name]) is not int or values[name] < minimum:
            raise ValueError("carla.visualization.trajectories.%s must be an integer >= %d" % (name, minimum))
    for name in ("lifetime_s", "thickness", "z_offset"):
        value = values[name]
        if (not isinstance(value, Real) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ValueError("carla.visualization.trajectories.%s must be positive and finite" % name)
        values[name] = float(value)
    return values


def _elevations(points, reference_xyz, fallback_z):
    """Project onto the resolved route surface without another nearest-map RPC.

    Interpolation follows a road's grade instead of drawing a flat sheet through
    hills. Resolve an exact XY crossing toward the capture's deck, never a fresh
    map lookup that could put an underpass trajectory on the overpass.
    """
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference_xyz, dtype=float)
    if reference.ndim != 2 or reference.shape[1:] != (3,) or not len(reference):
        return np.full(len(points), fallback_z, dtype=float)
    if len(reference) == 1:
        return np.full(len(points), reference[0, 2], dtype=float)
    starts = reference[:-1]
    segments = reference[1:, :2] - starts[:, :2]
    lengths2 = np.einsum("ij,ij->i", segments, segments)
    delta = points[:, None, :2] - starts[None, :, :2]
    u = np.clip(np.einsum("pij,ij->pi", delta, segments)
                / np.maximum(lengths2, 1e-12), 0., 1.)
    projected = starts[None, :, :2] + u[:, :, None] * segments[None, :, :]
    distances2 = np.sum((points[:, None, :2] - projected) ** 2, axis=2)
    heights = starts[None, :, 2] + u * (reference[1:, 2] - starts[:, 2])[None, :]
    # Only break effectively coincident projections by deck height; elevation
    # must not attract a point to a geometrically distant stretch of road.
    nearest = distances2.min(axis=1)[:, None]
    rank = np.where(distances2 <= nearest + 1e-8, np.abs(heights - fallback_z), np.inf)
    indices = np.argmin(rank, axis=1)
    return heights[np.arange(len(points)), indices]


def build_trajectory_visualization_snapshot(scored_states, candidates, selected_index,
                                            initial_state, capture_transform,
                                            state_index, reference_waypoints=(),
                                            timestamp_us=None):
    """Copy the final evaluated pool, including an actually restored shadow pool.

    This uses the identical rigid rear-axle to actor conversion as execution.
    It deliberately does not read generator references or change native arrays.
    Nonfinite proposals remain in planner diagnostics but cannot be drawn.
    """
    states = np.asarray(scored_states)
    if states.ndim != 3:
        raise ValueError("Scored visualization states must have candidate/time/state dimensions")
    rear = initial_state.rear_axle
    actor = np.array([capture_transform.location.x, -capture_transform.location.y])
    delta = actor - [rear.x, rear.y]
    c, s = math.cos(rear.heading), math.sin(rear.heading)
    dx, dy = c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]
    reference = [[float(wp.transform.location.x), float(wp.transform.location.y),
                  float(wp.transform.location.z)] for wp in reference_waypoints]
    snapshots = []
    for candidate in candidates:
        index = candidate["index"]
        row = states[index]
        poses = np.column_stack((row[:, state_index.X], row[:, state_index.Y],
                                 row[:, state_index.HEADING]))
        if len(poses) < 2 or not np.isfinite(poses).all():
            continue
        heading = poses[:, 2]
        x = poses[:, 0] + np.cos(heading) * dx - np.sin(heading) * dy
        y = -(poses[:, 1] + np.sin(heading) * dx + np.cos(heading) * dy)
        points = np.column_stack((x, y))
        z = _elevations(points, reference, capture_transform.location.z)
        snapshots.append({"id": candidate["id"], "index": int(index),
                          "points_world": np.column_stack((points, z)).tolist(),
                          "headings_world_rad": (-heading).tolist(),
                          "base_score": candidate.get("base_score"),
                          "final_score": candidate.get("final_score")})
    return {"source": "native_pdm_scored_rollout", "frame": "carla_world_actor",
            "timestamp_us": timestamp_us, "selected_index": selected_index,
            "candidate_pool_count": len(candidates), "candidates": snapshots,
            "elevation_source": "resolved_route_surface_projection",
            "reference_surface_xyz": reference}


class CarlaTrajectoryRenderer:
    """Finite lifetime debug primitives, with no ticks or control mutations."""

    def __init__(self, world, config=None):
        self.world = world
        self.config = validate_trajectory_visualization_config(config)
        self._warned = False

    def _line(self, points, kind, z_lift=0.):
        import carla
        if len(points) < 2:
            return 0
        stride = self.config["point_stride"]
        indices = list(range(0, len(points), stride))
        if indices[-1] != len(points) - 1:
            indices.append(len(points) - 1)
        color = carla.Color(*COLORS[kind])
        count = 0
        for a, b in zip(indices, indices[1:]):
            start, end = points[a], points[b]
            if not all(math.isfinite(float(value)) for value in (*start, *end)):
                continue
            # Avoid zero length primitives for stopped candidates.
            if math.dist(start, end) < 1e-6:
                continue
            lift = self.config["z_offset"] + z_lift
            self.world.debug.draw_line(
                carla.Location(x=float(start[0]), y=float(start[1]), z=float(start[2] + lift)),
                carla.Location(x=float(end[0]), y=float(end[1]), z=float(end[2] + lift)),
                thickness=self.config["thickness"] * LINE_WIDTH_SCALE[kind],
                color=color, life_time=self.config["lifetime_s"], persistent_lines=False)
            count += 1
        return count

    def draw(self, snapshot, decision, capture_transform, accepted, *, frame_id=None):
        """Call after trajectory validation; rejected commands never appear teal.

        ``decision`` is the actual control decision containing ``trajectory`` in
        captured forward/left coordinates. ``accepted`` must be the validator's
        boolean result. A failed draw is returned as audit data, never raised.
        """
        audit = {"enabled": self.config["enabled"], "frame_id": frame_id,
                 "drawn_candidates": 0, "drawn_segments": 0,
                 "execution_drawn": False, "rejected_drawn": False}
        if not self.config["enabled"]:
            return audit
        try:
            candidates = (snapshot or {}).get("candidates", [])
            selected_index = (snapshot or {}).get("selected_index")
            maximum = self.config["max_candidates"]
            chosen = list(candidates if not maximum else candidates[:maximum])
            selected = next((item for item in candidates if item["index"] == selected_index), None)
            if selected is not None and selected not in chosen:
                chosen[-1:] = [selected]
            # Draw the winner after the rest, with a small vertical separation.
            chosen.sort(key=lambda item: item["index"] == selected_index)
            for candidate in chosen:
                winner = candidate["index"] == selected_index
                count = self._line(candidate["points_world"], "selected" if winner else "candidate",
                                   .03 if winner else 0.)
                audit["drawn_candidates"] += bool(count)
                audit["drawn_segments"] += count
            trajectory = (decision or {}).get("trajectory") or []
            if len(trajectory) >= 2:
                yaw = math.radians(capture_transform.rotation.yaw)
                points = np.array([ego_to_global(float(p["x"]), float(p["y"]),
                    capture_transform.location.x, capture_transform.location.y, yaw) for p in trajectory])
                heights = _elevations(points, (snapshot or {}).get("reference_surface_xyz", []),
                                      capture_transform.location.z)
                kind = "execution" if accepted is True else "rejected"
                count = self._line(np.column_stack((points, heights)).tolist(), kind, .06)
                audit["drawn_segments"] += count
                audit[kind + "_drawn"] = bool(count)
            audit["candidate_pool_count"] = (snapshot or {}).get("candidate_pool_count", 0)
            audit["source"] = (snapshot or {}).get("source")
        except Exception as exc:
            audit["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if not self._warned:
                logger.warning("Trajectory debug rendering failed; control continues: %s", exc)
                self._warned = True
        return audit
