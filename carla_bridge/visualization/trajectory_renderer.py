"""Draw the scored PDM rollouts and accepted command through CARLA debug RPCs.

All points are CARLA world actor positions, not ideal path references or nuPlan
rear-axle positions. Elevation follows the current resolved route's road surface.
Rendering is optional and may fail without altering a planning/control result.
"""
from __future__ import annotations

import logging
import math
import hashlib
from copy import deepcopy
from numbers import Real

import numpy as np

from carla_bridge.state.coords import ego_to_global

logger = logging.getLogger("carla_trajectory_visualization")

DEFAULTS = {
    "enabled": False,
    "style": "paper",
    "max_candidates": 10,  # Display budget only; the actual winner is always included.
    "point_stride": 2,
    "lifetime_s": 0.2,
    "thickness": 0.055,
    "selected_thickness": 0.24,
    "execution_thickness": 0.11,
    "outline_width": 0.045,
    "z_offset": 0.15,
    "dash_length_m": 1.2,
    "dash_gap_m": 0.8,
    "merge_distance_m": 0.12,
    "endpoint_markers": True,
}

COLORS = {
    # CARLA 0.9.15 shipping multiplies linear debug color by 1000. Full 255
    # channels cause overlapping proposals to bloom into a white light sheet.
    "candidate": (0, 0, 2),
    "selected": (2, 1, 0),
    "execution": (0, 2, 0),
    "rejected": (2, 0, 0),
}

# Low HDR inputs are deliberate: CARLA shipping multiplies debug colors by
# 1000. Contrast comes from a black casing and separated strokes, not RGB=255.
PAPER_COLORS = {"selected": (2, 1, 0), "execution": (0, 2, 1),
                "rejected": (2, 0, 0), "infeasible": (1, 1, 1), "outline": (0, 0, 0)}
CANDIDATE_COLORS = ((0, 0, 2), (2, 0, 2), (0, 1, 2),
                    (1, 0, 2), (2, 0, 1), (0, 2, 2))
# Ordinary RGB for a camera projection, after the engine's tone mapping.
DISPLAY_COLORS = {"selected": (245, 190, 35), "execution": (20, 210, 160),
                  "rejected": (235, 60, 70), "infeasible": (155, 165, 180)}
DISPLAY_CANDIDATE_COLORS = ((45, 130, 235), (190, 80, 220), (25, 180, 225),
                            (110, 95, 225), (225, 80, 160), (65, 185, 220))
MAX_STROKE_SEGMENTS = 4096
SNAPSHOT_METADATA = (
    "lateral", "longitudinal", "physical_lane_side", "lane_offset_m",
    "lateral_path_index", "longitudinal_policy_id", "maneuver", "maneuver_id",
    "maneuver_phase", "recovery", "recovery_recipe", "recovery_transition_m",
    "optimization_feasible", "trajectory_finite", "numerically_valid", "eligible",
    "ineligible_reason", "forward_progress_m", "terminal_speed_mps", "desired_speed_mps")


def validate_trajectory_visualization_config(config=None):
    if config is not None and not isinstance(config, dict):
        raise ValueError("carla.visualization.trajectories must be a mapping")
    values = dict(DEFAULTS, **(config or {}))
    if type(values["enabled"]) is not bool:
        raise ValueError("carla.visualization.trajectories.enabled must be boolean")
    if values["style"] not in ("paper", "classic"):
        raise ValueError("carla.visualization.trajectories.style must be paper or classic")
    if type(values["endpoint_markers"]) is not bool:
        raise ValueError("carla.visualization.trajectories.endpoint_markers must be boolean")
    for name, minimum in (("max_candidates", 0), ("point_stride", 1)):
        if type(values[name]) is not int or values[name] < minimum:
            raise ValueError("carla.visualization.trajectories.%s must be an integer >= %d" % (name, minimum))
    for name in ("lifetime_s", "thickness", "z_offset", "selected_thickness",
                 "execution_thickness", "outline_width", "dash_length_m", "dash_gap_m"):
        value = values[name]
        if (not isinstance(value, Real) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ValueError("carla.visualization.trajectories.%s must be positive and finite" % name)
        values[name] = float(value)
    for name in ("dash_length_m", "dash_gap_m"):
        if values[name] < .05:
            raise ValueError("carla.visualization.trajectories.%s must be >= 0.05 m" % name)
    value = values["merge_distance_m"]
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("carla.visualization.trajectories.merge_distance_m must be finite and nonnegative")
    values["merge_distance_m"] = float(value)
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
                          "final_score": candidate.get("final_score"),
                          **{key: deepcopy(candidate[key]) for key in SNAPSHOT_METADATA if key in candidate}})
    return {"source": "native_pdm_scored_rollout", "frame": "carla_world_actor",
            "timestamp_us": timestamp_us, "selected_index": selected_index,
            "candidate_pool_count": len(candidates), "candidates": snapshots,
            "elevation_source": "resolved_route_surface_projection",
            "reference_surface_xyz": reference}


def _polyline(points):
    """Drop repeated vertices for display, preserving every remaining XYZ."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Visualization points must be finite XYZ vertices")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, lengths > 1e-6]]
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return points, distances


def _at_distance(points, distances, queries):
    return np.column_stack([np.interp(queries, distances, points[:, i]) for i in range(3)])


def _same_geometry(points, distances, group, tolerance):
    if tolerance <= 0 or abs(group["length"] - distances[-1]) > tolerance * 2:
        return False
    # Include every bend of BOTH polylines. Sparse uniform samples can miss a
    # short divergence, or a small looping manoeuvre between two sample sites.
    u = np.unique(np.r_[distances / max(distances[-1], 1e-12),
                        group["distances"] / max(group["length"], 1e-12)])
    a = _at_distance(points, distances, u * distances[-1])
    b = _at_distance(group["points"], group["distances"], u * group["length"])
    return np.max(np.linalg.norm(a - b, axis=1)) <= tolerance


def _display_groups(candidates, selected_index, config):
    """Merge near-identical FULL geometries; never merge just a common prefix.

    Time and speed variants remain in planner.jsonl; grouped members are also
    recorded in the visualization audit. Invalid candidates are not repaired.
    """
    groups, invalid = [], []
    tolerance = config["merge_distance_m"]
    valid_candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            invalid.append(None)
        else:
            valid_candidates.append(candidate)
    for candidate in sorted(valid_candidates, key=lambda item: str(item.get("id", item.get("index")))):
        try:
            points, distances = _polyline(candidate["points_world"])
        except (KeyError, ValueError, TypeError):
            invalid.append(candidate.get("id"))
            continue
        samples = _at_distance(points, distances, np.linspace(0., distances[-1], 25))
        # A failed MPC iterate must not acquire the styling of a feasible one.
        feasible = all(candidate.get(key) is not False for key in
                       ("optimization_feasible", "trajectory_finite", "numerically_valid", "eligible"))
        match = next((g for g in groups if g["feasible"] == feasible
                      and _same_geometry(points, distances, g, tolerance)), None)
        if match is None:
            side = str(candidate.get("physical_lane_side", candidate.get("lateral", "unknown")))
            match = {"members": [], "representative": candidate, "samples": samples,
                     "length": float(distances[-1]), "family": side, "feasible": feasible,
                     "points": points, "distances": distances}
            groups.append(match)
        match["members"].append(candidate)
        if selected_index is not None and candidate.get("index") == selected_index:
            match["representative"] = candidate

    def score(group):
        value = group["representative"].get("final_score", group["representative"].get("base_score"))
        return float(value) if isinstance(value, Real) and math.isfinite(value) else -math.inf

    groups.sort(key=lambda g: (-int(g["feasible"]), -score(g), str(g["representative"].get("id"))))
    selected = next((g for g in groups if selected_index is not None
                     and g["representative"].get("index") == selected_index), None)
    chosen = [selected] if selected is not None else []
    remaining = [g for g in groups if g is not selected]
    budget = config["max_candidates"] or len(groups)
    # Cover lateral families before spending the budget on more speed variants.
    families = {g["family"] for g in chosen}
    for group in remaining[:]:
        if len(chosen) >= budget:
            break
        if group["family"] not in families:
            chosen.append(group)
            families.add(group["family"])
            remaining = [g for g in remaining if g is not group]
    while remaining and len(chosen) < budget:
        # Prefer geometrically distinct paths, rather than the first N indices.
        group = max(remaining, key=lambda g: min(
            (float(np.mean(np.linalg.norm(g["samples"] - c["samples"], axis=1))) for c in chosen),
            default=0.))
        chosen.append(group)
        remaining = [g for g in remaining if g is not group]
    chosen.sort(key=lambda g: g is selected)  # The winner overlays its alternatives.
    return chosen, groups, invalid


class CarlaTrajectoryRenderer:
    """Finite lifetime debug primitives, with no ticks or control mutations."""

    def __init__(self, world, config=None):
        self.world = world
        self.config = validate_trajectory_visualization_config(config)
        self._warned = False
        self.last_display_snapshot = None

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
                thickness=self.config["thickness"] * (2 if kind in ("execution", "rejected") else 1),
                color=color, life_time=self.config["lifetime_s"], persistent_lines=False)
            count += 1
        return count

    def _stroke(self, points, rgb, width, lift, *, dashed=False, phase=0.):
        """Metric dashes follow the original polyline, including bend vertices."""
        import carla
        points, distances = _polyline(points)
        if len(points) < 2:
            return 0
        count = 0
        period = self.config["dash_length_m"] + self.config["dash_gap_m"]
        estimate = len(points) - 1
        if dashed:
            estimate += 2 * math.ceil(distances[-1] / period) + 4
        if estimate > MAX_STROKE_SEGMENTS:
            raise ValueError("Trajectory exceeds the display segment budget")
        for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
            start, end = distances[i], distances[i + 1]
            cuts = [start, end]
            if dashed:
                first = math.floor((start + phase) / period) - 1
                last = math.ceil((end + phase) / period) + 1
                cuts += [v for k in range(first, last + 1)
                         for v in (k * period - phase, k * period + self.config["dash_length_m"] - phase)
                         if start < v < end]
            cuts.sort()
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if dashed and ((lo + hi) * .5 + phase) % period >= self.config["dash_length_m"]:
                    continue
                p, q = a + (b - a) * ((lo - start) / (end - start)), a + (b - a) * ((hi - start) / (end - start))
                self.world.debug.draw_line(
                    carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2] + lift)),
                    carla.Location(x=float(q[0]), y=float(q[1]), z=float(q[2] + lift)),
                    thickness=width, color=carla.Color(*rgb),
                    life_time=self.config["lifetime_s"], persistent_lines=False)
                count += 1
        return count

    def _paper_line(self, points, rgb, width, lift, *, dashed=False, phase=0., marker=True):
        points, distances = _polyline(points)
        # A black casing remains legible over road paint, fog and bright asphalt.
        count = self._stroke(points, PAPER_COLORS["outline"], width + self.config["outline_width"] * 2,
                             lift, dashed=dashed, phase=phase)
        count += self._stroke(points, rgb, width, lift + .006, dashed=dashed, phase=phase)
        if marker and self.config["endpoint_markers"]:
            end = points[-1]
            if distances[-1] < .15:
                # Explicit stationary plan glyph; do not invent a forward path.
                radius = .22 if width < self.config["selected_thickness"] else .34
                symbols = [[end + [-radius, 0., 0.], end + [radius, 0., 0.]],
                           [end + [0., -radius, 0.], end + [0., radius, 0.]]]
            else:
                previous = _at_distance(points, distances, [max(0., distances[-1] - .8)])[0]
                tangent = end[:2] - previous[:2]
                norm = np.linalg.norm(tangent)
                if norm < 1e-6:
                    return count
                tangent /= norm
                normal = np.array([-tangent[1], tangent[0]])
                # Terminal cap for proposals; a chevron indicates the command direction.
                half = .25 if dashed else .40
                left, right = end.copy(), end.copy()
                left[:2] += normal * half - (0. if dashed else .6) * tangent
                right[:2] -= normal * half + (0. if dashed else .6) * tangent
                symbols = [[left, right]] if dashed else [[left, end, right]]
            for symbol in symbols:
                count += self._stroke(symbol, PAPER_COLORS["outline"], width + self.config["outline_width"] * 2, lift)
                count += self._stroke(symbol, rgb, width, lift + .006)
        return count

    def _draw_paper(self, snapshot, decision, capture_transform, accepted, audit):
        candidates = (snapshot or {}).get("candidates", [])
        selected_index = (snapshot or {}).get("selected_index")
        chosen, groups, invalid = _display_groups(candidates, selected_index, self.config)
        audit.update(style="paper", drawable_candidate_count=sum(len(g["members"]) for g in groups),
                     merged_candidates=sum(len(g["members"]) - 1 for g in groups),
                     hidden_candidates=len(groups) - len(chosen), invalid_candidate_ids=invalid,
                     candidate_groups=[], skipped_candidate_errors=[], xy_geometry_displaced=False)
        base_lift = self.config["z_offset"]
        # Detached display data for exact camera-frame projection in the video.
        # It never enters control, model inputs or numerical candidate arrays.
        self.last_display_snapshot = {"style": "paper", "paths": [],
            "dash_length_m": self.config["dash_length_m"], "dash_gap_m": self.config["dash_gap_m"],
            "outline_width_m": self.config["outline_width"],
            "endpoint_markers": self.config["endpoint_markers"]}
        for group in chosen:
            candidate = group["representative"]
            winner = selected_index is not None and candidate.get("index") == selected_index
            identifier = str(candidate.get("id", candidate.get("index")))
            digest = int.from_bytes(hashlib.blake2s(identifier.encode("utf-8"), digest_size=4).digest(), "big")
            rgb = (PAPER_COLORS["selected"] if winner else CANDIDATE_COLORS[digest % len(CANDIDATE_COLORS)]
                   if group["feasible"] else PAPER_COLORS["infeasible"])
            width = self.config["selected_thickness"] if winner else self.config["thickness"]
            phase = digest % 997 / 997 * (self.config["dash_length_m"] + self.config["dash_gap_m"])
            try:
                count = self._paper_line(candidate["points_world"], rgb, width,
                    base_lift + (.025 if winner else 0.), dashed=not winner, phase=phase)
            except (ValueError, TypeError, KeyError) as exc:
                audit["skipped_candidate_errors"].append({"id": candidate.get("id"),
                    "type": type(exc).__name__, "message": str(exc)})
                continue
            audit["drawn_segments"] += count
            audit["drawn_candidates"] += bool(count)
            display_rgb = (DISPLAY_COLORS["selected"] if winner else
                DISPLAY_CANDIDATE_COLORS[digest % len(DISPLAY_CANDIDATE_COLORS)]
                if group["feasible"] else DISPLAY_COLORS["infeasible"])
            display_points = np.asarray(candidate["points_world"], dtype=float).copy()
            display_points[:, 2] += base_lift + (.025 if winner else 0.) + .006
            self.last_display_snapshot["paths"].append({"kind": "selected" if winner else "candidate",
                "id": candidate.get("id"), "points_world": display_points.tolist(),
                "rgb": list(display_rgb), "width_m": width, "dashed": not winner,
                "phase_m": phase, "marker": True})
            audit["candidate_groups"].append({"representative_id": candidate.get("id"),
                "member_ids": [m.get("id") for m in group["members"]],
                "member_indices": [m.get("index") for m in group["members"]],
                "family": group["family"], "rgb": list(rgb), "display_rgb": list(display_rgb), "selected": winner,
                "display_feasible": group["feasible"]})
        trajectory = (decision or {}).get("trajectory") or []
        if trajectory:
            yaw = math.radians(capture_transform.rotation.yaw)
            xy = np.array([ego_to_global(float(p["x"]), float(p["y"]),
                capture_transform.location.x, capture_transform.location.y, yaw) for p in trajectory])
            heights = _elevations(xy, (snapshot or {}).get("reference_surface_xyz", []), capture_transform.location.z)
            kind = "execution" if accepted is True else "rejected"
            count = self._paper_line(np.column_stack((xy, heights)), PAPER_COLORS[kind],
                self.config["execution_thickness"], base_lift + .055)
            audit["drawn_segments"] += count
            audit[kind + "_drawn"] = bool(count)
            display_points = np.column_stack((xy, heights))
            display_points[:, 2] += base_lift + .055 + .006
            self.last_display_snapshot["paths"].append({"kind": kind,
                "points_world": display_points.tolist(), "rgb": list(DISPLAY_COLORS[kind]),
                "width_m": self.config["execution_thickness"], "dashed": False,
                "phase_m": 0., "marker": True})

    def draw(self, snapshot, decision, capture_transform, accepted, *, frame_id=None):
        """Call after trajectory validation; rejected commands never appear green.

        ``decision`` is the actual control decision containing ``trajectory`` in
        captured forward/left coordinates. ``accepted`` must be the validator's
        boolean result. A failed draw is returned as audit data, never raised.
        """
        audit = {"enabled": self.config["enabled"], "frame_id": frame_id,
                 "drawn_candidates": 0, "drawn_segments": 0,
                 "execution_drawn": False, "rejected_drawn": False}
        self.last_display_snapshot = None
        if not self.config["enabled"]:
            return audit
        try:
            audit["candidate_pool_count"] = (snapshot or {}).get("candidate_pool_count", 0)
            audit["source"] = (snapshot or {}).get("source")
            audit["style"] = self.config["style"]
            if self.config["style"] == "paper":
                self._draw_paper(snapshot, decision, capture_transform, accepted, audit)
                return audit
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
