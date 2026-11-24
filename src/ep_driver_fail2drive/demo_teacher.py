"""Causal scripted teacher for independent, offline CARLA demonstrations only.

This module has no CARLA, numerical-worker, model, scenario-instance or event
timer dependency. It follows an authored reference using current observed
actors and public traffic rules. It is neither the paper planner nor a safety
certificate. The collector must independently check collisions, road/rule
compliance, full interaction completion and final arrival before admission.

Coordinates are metres in the public state's right-handed XY frame; steering
is a physical wheel angle, positive left. ``bypass_reference`` is an optional
complete teaching path with the same start and finish as ``route_xy``. Only
the teacher sees that path; the ordinary navigation remains ``route_xy``.
"""
from __future__ import annotations

import math


SCHEMA = "ep-driver-independent-demo-teacher-1.0"

DEFAULT_SETTINGS = {
    "cruise_speed_mps": 5.0,
    "acceleration_min_mps2": -5.0,
    "acceleration_max_mps2": 2.0,
    "jerk_max_mps3": 6.0,
    "steering_max_rad": .6,
    "steering_rate_max_radps": .7,
    "speed_gain_per_s": 1.2,
    "lookahead_min_m": 3.0,
    "lookahead_time_s": .7,
    "lookahead_max_m": 9.0,
    "lateral_acceleration_max_mps2": 1.8,
    "service_deceleration_mps2": 2.5,
    "stop_speed_gain_per_s": .3,
    "stop_margin_m": 2.5,
    "following_distance_m": 5.0,
    "following_time_s": 1.6,
    "following_gain_per_s": .7,
    "corridor_margin_m": .45,
    "actor_front_m": 45.0,
    "actor_prediction_horizon_s": 3.0,
    "actor_prediction_step_s": .5,
    "crossing_arrival_tolerance_s": 1.0,
    "path_sample_step_m": .6,
    "clear_confirmation_s": .5,
    "stopped_speed_mps": .15,
    "finish_tolerance_m": .8,
    "projection_maximum_distance_m": 6.0,
    "projection_initial_window_m": 12.0,
    "projection_window_m": 12.0,
    "projection_backward_tolerance_m": 1.0,
    "projection_heading_limit_rad": 2.0,
    "maximum_step_s": 1.0,
}


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _point(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + " must be an XY pair")
    return (_number(value[0], name + "[0]"), _number(value[1], name + "[1]"))


def _clamp(value, low, high):
    return min(high, max(low, value))


def _wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


class _Path:
    def __init__(self, points, name):
        if not isinstance(points, (list, tuple)) or len(points) < 2:
            raise ValueError(name + " requires at least two XY points")
        self.points = [_point(p, name) for p in points]
        self.segments, self.length = [], 0.
        for a, b in zip(self.points, self.points[1:]):
            distance = math.dist(a, b)
            if distance < 1e-5:
                raise ValueError(name + " contains coincident consecutive points")
            self.segments.append((self.length, self.length + distance, a, b,
                                  math.atan2(b[1] - a[1], b[0] - a[0])))
            self.length += distance

    def at(self, station):
        station = _clamp(station, 0., self.length)
        segment = next((r for r in self.segments if r[1] >= station), self.segments[-1])
        start, end, a, b, heading = segment
        ratio = (station - start) / (end - start)
        return ((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])), heading)

    def project(self, point, heading=None, lower=0., upper=None):
        upper = self.length if upper is None else min(self.length, upper)
        lower = max(0., lower)
        choices = []
        for start, end, a, b, direction in self.segments:
            if end < lower or start > upper:
                continue
            length = end - start
            fraction = ((point[0] - a[0]) * (b[0] - a[0]) +
                        (point[1] - a[1]) * (b[1] - a[1])) / (length * length)
            fraction = _clamp(fraction, max(0., (lower - start) / length),
                              min(1., (upper - start) / length))
            station = start + fraction * length
            closest = (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))
            residual = math.dist(point, closest)
            error = abs(_wrap(direction - heading)) if heading is not None else 0.
            lateral = (-(point[0] - closest[0]) * math.sin(direction) +
                       (point[1] - closest[1]) * math.cos(direction))
            choices.append((residual + .15 * error, residual, station, error, lateral, direction))
        if not choices:
            raise ValueError("no teaching-path segment within the causal projection window")
        _, residual, station, error, lateral, direction = min(choices)
        return {"s_m": station, "distance_m": residual, "heading_error_rad": error,
                "lateral_m": lateral, "heading_rad": direction}


def _segment_distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = dx * dx + dy * dy
    fraction = _clamp(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / norm, 0., 1.) if norm else 0.
    return math.dist(point, (a[0] + fraction * dx, a[1] + fraction * dy))


def _body_distance(point, polygon):
    inside = False
    previous = polygon[-1]
    distance = float("inf")
    for current in polygon:
        distance = min(distance, _segment_distance(point, previous, current))
        if (previous[1] > point[1]) != (current[1] > point[1]):
            crossing = previous[0] + (point[1] - previous[1]) * (current[0] - previous[0]) / (current[1] - previous[1])
            if point[0] < crossing:
                inside = not inside
        previous = current
    return 0. if inside else distance


def _actor_body(actor):
    position = _point(actor.get("position"), "actor.position")
    velocity = _point(actor.get("velocity", [0., 0.]), "actor.velocity")
    heading = _number(actor.get("heading", 0.), "actor.heading")
    length = _number(actor.get("length", 1.), "actor.length")
    width = _number(actor.get("width", 1.), "actor.width")
    if length <= 0. or width <= 0.:
        raise ValueError("actor body dimensions must be positive")
    if actor.get("polygon") is not None:
        polygon = [_point(p, "actor.polygon") for p in actor["polygon"]]
        if len(polygon) < 3:
            raise ValueError("actor.polygon requires at least three points")
    else:
        c, s = math.cos(heading), math.sin(heading)
        polygon = [(position[0] + u * c - v * s, position[1] + u * s + v * c)
                   for u, v in ((length / 2., width / 2.), (length / 2., -width / 2.),
                                (-length / 2., -width / 2.), (-length / 2., width / 2.))]
    return position, velocity, polygon


class DemoTeacher:
    """Follow an offline teaching reference; never used by formal evaluation.

    Invalid inputs/projections raise ``ValueError`` so the collector records an
    unsuccessful demonstration. No hidden teleport, automatic success, model
    call, scenario timer or emergency fallback is provided here.
    """
    def __init__(self, route_xy, wheelbase_m, settings=None, bypass_reference=None):
        self.settings = dict(DEFAULT_SETTINGS)
        supplied = dict(settings or {})
        unknown = set(supplied) - set(self.settings)
        if unknown:
            raise ValueError("unknown demonstration teacher settings: " + ", ".join(sorted(unknown)))
        self.settings.update(supplied)
        for name, value in self.settings.items():
            self.settings[name] = _number(value, name)
        p = self.settings
        for name, value in p.items():
            if name != "acceleration_min_mps2" and value <= 0.:
                raise ValueError(name + " must be positive")
        if p["acceleration_min_mps2"] >= 0.:
            raise ValueError("acceleration_min_mps2 must be negative")
        if p["lookahead_max_m"] < p["lookahead_min_m"]:
            raise ValueError("lookahead maximum is smaller than minimum")
        if p["service_deceleration_mps2"] > -p["acceleration_min_mps2"]:
            raise ValueError("service deceleration exceeds the actuator deceleration limit")
        self.wheelbase = _number(wheelbase_m, "wheelbase_m")
        if self.wheelbase <= 0.:
            raise ValueError("wheelbase_m must be positive")
        self.navigation = _Path(route_xy, "route_xy")
        self.path = _Path(bypass_reference, "bypass_reference") if bypass_reference is not None else self.navigation
        if any(math.dist(self.path.points[index], self.navigation.points[index]) > .1 for index in (0, -1)):
            raise ValueError("bypass_reference must retain the original navigation start and finish")
        self.last_frame, self.last_time, self.last_position = None, None, None
        self.nav_s, self.path_s = None, None
        self.acceleration, self.steering = 0., 0.
        self.waiting = False
        self.clear_elapsed = 0.
        self.observed_interaction = False
        self.bypass_peak = 0.

    def _projection(self, path, position, heading, previous_s, speed, dt):
        p = self.settings
        lower = 0. if previous_s is None else previous_s - p["projection_backward_tolerance_m"]
        upper = (p["projection_initial_window_m"] if previous_s is None else
                 previous_s + max(p["projection_window_m"], speed * dt + 2.))
        result = path.project(position, heading, lower, upper)
        if result["distance_m"] > p["projection_maximum_distance_m"]:
            raise ValueError("teaching-path projection residual exceeds maximum distance")
        if result["heading_error_rad"] > p["projection_heading_limit_rad"]:
            raise ValueError("ego heading is incompatible with the teaching path")
        return result

    def _samples(self, station, ego_length):
        p = self.settings
        start = max(0., station - ego_length / 2.)
        end = min(self.path.length, station + p["actor_front_m"])
        count = max(1, int(math.ceil((end - start) / p["path_sample_step_m"])))
        return [(start + (end - start) * i / count, self.path.at(start + (end - start) * i / count)[0])
                for i in range(count + 1)]

    def _stop_speed(self, distance):
        """Include velocity-loop lag instead of treating sqrt(2bd) as a command.

        The linear tail approaches a finite target conservatively. A raw
        constant-deceleration envelope with a lagging speed servo overshoots
        the line/finish even in an ideal bicycle replay.
        """
        distance = max(0., distance)
        return min(math.sqrt(2. * self.settings["service_deceleration_mps2"] * distance),
                   self.settings["stop_speed_gain_per_s"] * distance)

    def _actors(self, scene, station, speed, ego_length, ego_width):
        p = self.settings
        samples = self._samples(station, ego_length)
        corridor = ego_width / 2. + p["corridor_margin_m"]
        cap, conflicts, must_stop = float("inf"), [], False
        actors = scene.get("actors", [])
        if not isinstance(actors, list):
            raise ValueError("scene.actors must be a list of current observations")
        for actor in actors:
            if not isinstance(actor, dict):
                raise ValueError("each observed actor must be a mapping")
            if actor.get("visible") is False:
                continue
            _, velocity, polygon = _actor_body(actor)
            # A spatial bound only avoids unnecessary polygon tests. Inclusion
            # itself uses the authored path corridor, not a circular range.
            center = _point(actor.get("position"), "actor.position")
            extent = max(math.dist(center, point) for point in polygon)
            motion_range = math.hypot(*velocity) * p["actor_prediction_horizon_s"]
            if min(math.dist(center, point) for _, point in samples) > corridor + extent + motion_range:
                continue
            occupied = [(s, point) for s, point in samples if _body_distance(point, polygon) <= corridor]
            if occupied:
                actor_station = occupied[0][0]
                gap = actor_station - station - ego_length / 2.
                direction = self.path.at(actor_station)[1]
                along_speed = velocity[0] * math.cos(direction) + velocity[1] * math.sin(direction)
                desired_gap = p["following_distance_m"] + p["following_time_s"] * speed
                following_cap = max(0., along_speed + p["following_gain_per_s"] * (gap - desired_gap))
                if along_speed <= .25:
                    stop_cap = self._stop_speed(gap - p["stop_margin_m"])
                    actor_cap = min(following_cap, stop_cap)
                else:
                    actor_cap = following_cap
                cap = min(cap, actor_cap)
                stopping = actor_cap <= .05
                must_stop |= stopping
                conflicts.append({"id": str(actor.get("id", "unidentified")), "reason": "observed_path_occupancy",
                                  "gap_m": gap, "observed_along_speed_mps": along_speed,
                                  "speed_cap_mps": actor_cap, "requires_stop": stopping,
                                  "evidence": "current_actor_body_and_velocity"})
                continue
            if math.hypot(*velocity) < .05:
                continue
            steps = max(1, int(math.ceil(p["actor_prediction_horizon_s"] / p["actor_prediction_step_s"])))
            for index in range(1, steps + 1):
                offset = p["actor_prediction_horizon_s"] * index / steps
                shifted = [(x + velocity[0] * offset, y + velocity[1] * offset) for x, y in polygon]
                hits = [s for s, point in samples if _body_distance(point, shifted) <= corridor]
                plausible = [s for s in hits if abs(max(0., s - station) / max(speed, 1.) - offset) <= p["crossing_arrival_tolerance_s"]]
                if not plausible:
                    continue
                gap = min(plausible) - station - ego_length / 2.
                actor_cap = self._stop_speed(gap - p["stop_margin_m"])
                cap = min(cap, actor_cap)
                stopping = actor_cap <= .05
                must_stop |= stopping
                conflicts.append({"id": str(actor.get("id", "unidentified")), "reason": "observed_motion_crosses_path",
                                  "gap_m": gap, "prediction_offset_s": offset, "speed_cap_mps": actor_cap,
                                  "requires_stop": stopping,
                                  "evidence": "constant_current_velocity_extrapolation_not_event_future"})
                break
        return cap, conflicts, must_stop

    def step(self, scene, dt):
        """Return one source-frame bound signed command without changing a world.

        Required input is the plain mapping returned by PublicStateExtractor.
        ``rules.stop_s`` is a station on the original navigation, not the
        teaching detour. Input shape/frame/geometry errors are explicit errors.
        """
        dt = _number(dt, "dt")
        p = self.settings
        if not 0. < dt <= p["maximum_step_s"]:
            raise ValueError("dt must be positive and within maximum_step_s")
        if not isinstance(scene, dict) or not isinstance(scene.get("ego"), dict):
            raise ValueError("scene must contain an ego mapping")
        time = _number(scene.get("time"), "scene.time")
        diagnostics = scene.get("diagnostics", {})
        frame = diagnostics.get("carla_frame", scene.get("frame"))
        if frame is None:
            try:
                frame = int(str(scene.get("frame_id", "")).rsplit(":", 1)[-1])
            except ValueError as exc:
                raise ValueError("scene must identify its physical source frame") from exc
        if type(frame) is not int or frame < 0:
            raise ValueError("scene source frame must be a nonnegative integer")
        if self.last_frame is not None and (frame <= self.last_frame or time <= self.last_time):
            raise ValueError("demonstration source frames and times must increase")
        if self.last_time is not None and abs(time - self.last_time - dt) > max(1e-5, dt * .01):
            raise ValueError("dt does not match elapsed observed simulation time")
        ego = scene["ego"]
        position = _point(ego.get("position"), "ego.position")
        velocity = _point(ego.get("velocity"), "ego.velocity")
        heading = _number(ego.get("heading"), "ego.heading")
        length = _number(ego.get("length"), "ego.length")
        width = _number(ego.get("width"), "ego.width")
        if length <= 0. or width <= 0.:
            raise ValueError("ego dimensions must be positive")
        speed = max(0., velocity[0] * math.cos(heading) + velocity[1] * math.sin(heading))
        if self.last_position is not None:
            reachable = max(1., speed * dt + .5 * p["acceleration_max_mps2"] * dt * dt + .5)
            if math.dist(position, self.last_position) > reachable:
                raise ValueError("observed displacement exceeds the causal teaching projection window")
        nav = self._projection(self.navigation, position, heading, self.nav_s, speed, dt)
        projection = self._projection(self.path, position, heading, self.path_s, speed, dt)
        station = projection["s_m"]
        lookahead = _clamp(p["lookahead_min_m"] + speed * p["lookahead_time_s"],
                           p["lookahead_min_m"], p["lookahead_max_m"])
        target_point, target_heading = self.path.at(station + lookahead)
        dx, dy = target_point[0] - position[0], target_point[1] - position[1]
        # Ego observations are body centres; pure pursuit is evaluated from the
        # real rear axle when its measured centre offset is available.
        axle_offset = _number(ego.get("rear_axle_to_center_m") or 0., "ego.rear_axle_to_center_m")
        dx += axle_offset * math.cos(heading)
        dy += axle_offset * math.sin(heading)
        lateral = -dx * math.sin(heading) + dy * math.cos(heading)
        distance_squared = max(.01, dx * dx + dy * dy)
        curvature = 2. * lateral / distance_squared
        requested_steering = math.atan(self.wheelbase * curvature)
        requested_steering = _clamp(requested_steering, -p["steering_max_rad"], p["steering_max_rad"])
        steering = _clamp(requested_steering, self.steering - p["steering_rate_max_radps"] * dt,
                           self.steering + p["steering_rate_max_radps"] * dt)
        rules = scene.get("rules", {})
        if not isinstance(rules, dict):
            raise ValueError("scene.rules must be a mapping")
        target_speed = p["cruise_speed_mps"]
        for name in ("desired_speed", "speed_limit"):
            if rules.get(name) is not None:
                value = _number(rules[name], "rules." + name)
                if value < 0.:
                    raise ValueError("public speed constraints cannot be negative")
                target_speed = min(target_speed, value)
        path_curvature = abs(_wrap(target_heading - projection["heading_rad"])) / max(lookahead, .1)
        curvature_bound = max(abs(curvature), path_curvature)
        if curvature_bound > 1e-5:
            target_speed = min(target_speed, math.sqrt(p["lateral_acceleration_max_mps2"] / curvature_bound))
        remaining = max(0., self.path.length - station)
        target_speed = min(target_speed, self._stop_speed(remaining))
        reasons = ["authored_reference_tracking"]
        must_stop = False
        conflicts = []
        if rules.get("stop_s") is not None:
            stop_station = _number(rules["stop_s"], "rules.stop_s")
            gap = stop_station - nav["s_m"] - length / 2. - p["stop_margin_m"]
            if gap >= -length:
                stop_cap = self._stop_speed(gap)
                target_speed = min(target_speed, stop_cap)
                must_stop |= stop_cap <= .05
                conflicts.append({"id": "public_stop_line", "reason": "current_public_traffic_rule",
                                  "navigation_stop_s_m": stop_station, "front_clearance_m": gap,
                                  "speed_cap_mps": stop_cap, "requires_stop": stop_cap <= .05})
                reasons.append("current_public_stop_line")
        actor_cap, actor_conflicts, actor_stop = self._actors(scene, station, speed, length, width)
        target_speed = min(target_speed, actor_cap)
        conflicts.extend(actor_conflicts)
        must_stop |= actor_stop
        if actor_conflicts:
            reasons.append("current_observed_actor_conflict")
        waiting = self.waiting or (must_stop and speed <= p["stopped_speed_mps"])
        clear_elapsed = 0. if must_stop else self.clear_elapsed + dt
        if waiting and not must_stop and clear_elapsed < p["clear_confirmation_s"]:
            target_speed = 0.
            reasons.append("confirming_current_corridor_clear")
        elif waiting and not must_stop:
            waiting = False
            reasons.append("observed_corridor_clear_resume")
        elif must_stop:
            target_speed = 0.
        finish_arrived = (remaining <= p["finish_tolerance_m"] and
                          math.dist(position, self.path.points[-1]) <= p["finish_tolerance_m"] and
                          speed <= p["stopped_speed_mps"])
        if finish_arrived:
            target_speed = 0.
        desired_acceleration = _clamp(p["speed_gain_per_s"] * (target_speed - speed),
                                      p["acceleration_min_mps2"], p["acceleration_max_mps2"])
        acceleration = _clamp(desired_acceleration, self.acceleration - p["jerk_max_mps3"] * dt,
                              self.acceleration + p["jerk_max_mps3"] * dt)
        interaction = self.observed_interaction or bool(conflicts)
        offset = abs(nav["lateral_m"])
        peak = max(self.bypass_peak, offset)
        target_navigation_offset = self.navigation.project(target_point,
            lower=max(0., nav["s_m"] - 2.), upper=nav["s_m"] + lookahead + 15.)["distance_m"]
        if finish_arrived:
            phase, target_speed = "DONE", 0.
            reasons.append("observed_demonstration_finish_not_official_score")
        elif waiting:
            phase = "WAIT"
        elif self.path is not self.navigation and (offset > .35 or target_navigation_offset > .35):
            phase = "RECOVER" if peak > .5 and offset < peak - .2 else "BYPASS"
        elif interaction or peak > .5:
            phase = "RECOVER" if not conflicts else "APPROACH"
        else:
            phase = "APPROACH"
        output = {"schema": SCHEMA, "acceleration_mps2": acceleration, "steering_rad_left": steering,
                  "target_speed_mps": target_speed, "phase": phase, "source_frame": frame,
                  "source_time": time, "reason": ";".join(reasons), "observed_conflicts": conflicts,
                  "projection": {"navigation": nav, "teaching": projection},
                  "tracking_target_xy": list(target_point), "teaching_remaining_m": remaining,
                  "safety_certified": False, "formal_evaluation_control": False}
        self.last_frame, self.last_time, self.last_position = frame, time, position
        self.nav_s = nav["s_m"] if self.nav_s is None else max(self.nav_s, nav["s_m"])
        self.path_s = station if self.path_s is None else max(self.path_s, station)
        self.acceleration, self.steering = acceleration, steering
        self.waiting, self.clear_elapsed = waiting, clear_elapsed
        self.observed_interaction, self.bypass_peak = interaction, peak
        return output
