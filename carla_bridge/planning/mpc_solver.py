"""Independent finite-horizon nonlinear MPC; no CARLA or nuPlan imports.

Coordinates are supplied by the caller (CARLA world x/y/yaw is supported).
The kinematic bicycle state is at the *rear axle*. Controls are longitudinal
acceleration demand and front-wheel steering angle. A negative demand brakes
to rest without engaging reverse; distance is integrated up to the exact stop
time within an interval. Each accepted trajectory is a rollout of those
controls, rather than a reference polyline renamed as MPC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize

MPC_COLLISION_GEOMETRY_VERSION = "physical_obb_euclidean_clearance_v1"
MPC_SOLVE_POLICY_VERSION = "low_speed_stationary_branch_verified_seed_comparison_v1"


@dataclass(frozen=True)
class MPCConfig:
    dt: float = .2
    horizon_steps: int = 20
    wheelbase: float = 2.8
    rear_axle_to_center: float = 1.4
    ego_length: float = 4.8
    ego_width: float = 2.0
    collision_margin: float = .2
    accel_min: float = -4.0
    accel_max: float = 2.0
    steer_max: float = .55
    steer_rate_max: float = .6
    jerk_max: float = 8.0
    lateral_accel_max: float = 3.0
    max_iterations: int = 55
    ftol: float = 1e-5
    constraint_tolerance: float = 2e-3
    lateral_weight: float = 8.0
    longitudinal_weight: float = .25
    heading_weight: float = 2.0
    speed_weight: float = 2.0
    acceleration_weight: float = .05
    steering_weight: float = .1
    jerk_weight: float = .05
    steering_rate_weight: float = .6
    terminal_weight: float = 2.0
    multi_start: bool = True
    ego_body_center_lateral: float = 0.0
    ego_body_yaw: float = 0.0

    def __post_init__(self):
        for key, value in vars(self).items():
            if key in ("multi_start",):
                if not isinstance(value, bool):
                    raise ValueError("multi_start must be boolean")
                continue
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError("MPC config %s must be finite" % key)
        if not isinstance(self.horizon_steps, int) or self.horizon_steps < 2:
            raise ValueError("horizon_steps must be an integer >= 2")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        positive = ("dt", "wheelbase", "ego_length", "ego_width", "steer_max",
                    "steer_rate_max", "jerk_max", "lateral_accel_max", "ftol",
                    "constraint_tolerance", "terminal_weight")
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError("MPC time, vehicle and constraint limits must be positive")
        if self.accel_min >= 0 or self.accel_max <= 0 or self.steer_max >= 1.4:
            raise ValueError("invalid MPC acceleration or steering limits")
        if self.collision_margin < 0:
            raise ValueError("collision_margin must be nonnegative")
        if any(value < 0 for key, value in vars(self).items() if key.endswith("weight")):
            raise ValueError("MPC objective weights must be nonnegative")


@dataclass(frozen=True)
class MPCInitialState:
    """Measured rear-axle pose/speed and the input boundary for rate limits.

    ``steering`` is the current equivalent front-wheel angle. ``acceleration``
    is the previous *dispatched desired acceleration demand*, not the measured
    physical acceleration: acceleration is not a state of this bicycle model.
    A contact impulse or stationary brake measurement must not be substituted
    for that input. Callers report a neutral zero desired input after a
    standalone stationary hold without an active optimized profile, interrupted control or absent command history,
    and keep raw acceleration telemetry separately. Hypothetical candidate
    solves must never update the dispatched-input history. A negative demand
    can persist through a stop and then release at the configured jerk rate:
    the unilateral zero-speed brake reaction is not an actuator command jump.
    """
    x: float
    y: float
    yaw: float
    speed: float
    steering: float = 0.0
    acceleration: float = 0.0


@dataclass(frozen=True)
class MPCReference:
    name: str
    points: Any  # (N+1, 4): rear-axle x, y, yaw, target speed.
    half_width: Any = .65  # Allowed rear-axle lateral error; already footprint-adjusted.
    speed_limit: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MPCObstacle:
    id: Any
    centers: Any  # (N+1, 2), with the initial observation at index zero.
    radii: Any  # scalar circular radius, (N+1,) radii, or (N+1, 2) ellipse half-axes.
    yaws: Any = 0.0  # Ellipse orientation, scalar or N+1 values.
    # Explicit physical rectangles use exact OBB distance. The original four
    # positional fields retain their circle/outer-ellipse compatibility path.
    rectangle_half_extents: Any = None  # (2,) or (N+1,2), never uncertainty-inflated.
    uncertainty_m: Any = 0.0  # Separate nonnegative scalar or N+1 clearance.


@dataclass(frozen=True)
class MPCProblem:
    initial: MPCInitialState
    reference: MPCReference
    obstacles: Any = ()


@dataclass
class MPCResult:
    states: np.ndarray
    controls: np.ndarray
    feasible: bool
    converged: bool
    status: str
    message: str
    cost: float
    iterations: int
    diagnostics: dict


def _broadcast(value, count, name):
    data = np.asarray(value, dtype=float)
    if data.ndim == 0:
        data = np.full(count, float(data))
    if data.shape != (count,) or not np.isfinite(data).all():
        raise ValueError("%s must be finite scalar or %d values" % (name, count))
    return data


def _angle(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def bicycle_step(state, control, dt, wheelbase):
    """Forward-only midpoint bicycle with an exact within-step braking stop.

    ``acceleration`` is the dispatched tractive/braking demand. Braking never
    commands reverse: after speed reaches zero, the vehicle holds position
    for the remainder of the interval. Hard jerk bounds still apply to this
    demand, so the optimizer releases brake pressure continuously instead of
    resetting its history or becoming infeasible near zero speed.

    The distance and its derivatives are exact for constant acceleration up
    to the stopping time. Pose uses the same midpoint curvature discretization
    in moving and stopping branches; derivatives are analytic within each
    active mode. At the non-smooth zero-speed boundary we take the moving
    one-sided derivative, which permits a stationary vehicle to accelerate.
    """
    x, y, yaw, speed = state
    acceleration, steering = control
    final_speed = speed + dt * acceleration
    if acceleration < 0. and final_speed < 0.:
        # Integral from t=0 to t_stop=-v/a. In particular v=0,a<0
        # produces exactly zero distance and yaw, not backwards movement.
        distance = -speed * speed / (2 * acceleration)
        distance_v = -speed / acceleration
        distance_a = speed * speed / (2 * acceleration * acceleration)
        final_speed, final_speed_v, final_speed_a = 0., 0., 0.
    else:
        distance = dt * speed + .5 * dt * dt * acceleration
        distance_v, distance_a = dt, .5 * dt * dt
        final_speed_v, final_speed_a = 1., dt
    tangent, secant2 = np.tan(steering), 1 / np.cos(steering) ** 2
    yaw_step = distance * tangent / wheelbase
    middle_yaw = yaw + .5 * yaw_step
    c, s = np.cos(middle_yaw), np.sin(middle_yaw)
    result = np.array([x + distance * c, y + distance * s, yaw + yaw_step,
                       final_speed])
    jac_state = np.eye(4)
    phi_v = .5 * distance_v * tangent / wheelbase
    jac_state[0, 2] = -distance * s
    jac_state[1, 2] = distance * c
    jac_state[0, 3] = distance_v * c - distance * s * phi_v
    jac_state[1, 3] = distance_v * s + distance * c * phi_v
    jac_state[2, 3] = distance_v * tangent / wheelbase
    jac_state[3, 3] = final_speed_v
    phi_a = .5 * distance_a * tangent / wheelbase
    phi_d = .5 * distance * secant2 / wheelbase
    jac_control = np.array([
        [distance_a * c - distance * s * phi_a, -distance * s * phi_d],
        [distance_a * s + distance * c * phi_a, distance * c * phi_d],
        [distance_a * tangent / wheelbase, distance * secant2 / wheelbase],
        [final_speed_a, 0.]])
    return result, jac_state, jac_control


def rollout(initial, controls, config, *, sensitivities=False):
    """Roll controls forward, optionally propagating d(state)/d(all controls)."""
    controls = np.asarray(controls, dtype=float).reshape(config.horizon_steps, 2)
    states = np.empty((len(controls) + 1, 4))
    states[0] = [initial.x, initial.y, initial.yaw, initial.speed]
    jac = np.zeros((len(states), 4, controls.size)) if sensitivities else None
    for index, control in enumerate(controls):
        states[index + 1], a, b = bicycle_step(states[index], control, config.dt, config.wheelbase)
        if sensitivities:
            jac[index + 1] = a @ jac[index]
            jac[index + 1, :, 2 * index:2 * index + 2] += b
    return (states, jac) if sensitivities else states


class _Evaluation:
    """Cache one rollout, objective/gradient and inequality values/Jacobian."""
    def __init__(self, config, problem):
        self.cfg, self.problem = config, problem
        n = config.horizon_steps
        self.reference = np.asarray(problem.reference.points, dtype=float)
        if self.reference.shape != (n + 1, 4) or not np.isfinite(self.reference).all():
            raise ValueError("MPC reference must contain N+1 finite [x,y,yaw,speed] rows")
        initial = problem.initial
        if not np.isfinite(list(vars(initial).values())).all() or initial.speed < 0:
            raise ValueError("MPC initial state must be finite with nonnegative speed")
        self.width = _broadcast(problem.reference.half_width, n + 1, "reference half_width")
        self.speed_limit = _broadcast(problem.reference.speed_limit if problem.reference.speed_limit
                                     is not None else np.maximum(initial.speed, np.maximum(1., self.reference[:, 3])),
                                     n + 1, "reference speed_limit")
        if np.any(self.width <= 0) or np.any(self.speed_limit < 0) or np.any(self.reference[:, 3] < 0):
            raise ValueError("MPC widths must be positive; speeds must be nonnegative")
        self.tangent = np.c_[np.cos(self.reference[:, 2]), np.sin(self.reference[:, 2])]
        self.normal = np.c_[-self.tangent[:, 1], self.tangent[:, 0]]
        # A vehicle already stopped near a route endpoint must hold its
        # measured rear-axle pose, rather than creep towards the clamped map
        # station. Keep that objective separate from the original road geometry:
        # neither the corridor nor the stop-line/collision constraints move.
        self.objective_reference = self.reference
        hold_pose = problem.reference.metadata.get("stationary_hold_pose")
        self.stationary_hold_objective = hold_pose is not None
        if hold_pose is not None:
            hold_pose = np.asarray(hold_pose, dtype=float)
            measured_pose = np.array([initial.x, initial.y, initial.yaw])
            is_stop = str(problem.reference.name).upper().split(":")[-1].split("@")[0] == "STOP"
            if (hold_pose.shape != (3,) or not np.isfinite(hold_pose).all()
                    or not is_stop or initial.speed > .03 or np.any(self.reference[:, 3] != 0.)
                    or not np.allclose(hold_pose[:2], measured_pose[:2], rtol=0., atol=1e-8)
                    or abs(_angle(hold_pose[2] - measured_pose[2])) > 1e-8):
                raise ValueError("stationary_hold_pose requires a stopped STOP reference and its measured initial pose")
            self.objective_reference = self.reference.copy()
            self.objective_reference[:, :3] = measured_pose
        self.objective_tangent = np.c_[np.cos(self.objective_reference[:, 2]),
                                      np.sin(self.objective_reference[:, 2])]
        self.objective_normal = np.c_[-self.objective_tangent[:, 1], self.objective_tangent[:, 0]]
        self.stop_line = problem.reference.metadata.get("stop_line")
        if self.stop_line is not None:
            point = np.asarray(self.stop_line["point"], dtype=float)
            normal = np.asarray(self.stop_line["normal"], dtype=float)
            if (point.shape != (2,) or normal.shape != (2,) or
                    not np.isfinite([point, normal]).all() or np.linalg.norm(normal) < 1e-6):
                raise ValueError("stop_line needs a finite point and nonzero forward normal")
            # The adapter places this line at the maximum permitted rear-axle
            # station, already accounting for front overhang and stop margin.
            self.stop_line = point, normal / np.linalg.norm(normal)
        self.previous = np.array([initial.acceleration, initial.steering])
        self.delta_matrix = np.eye(2 * n)
        self.delta_matrix[2:, :-2] -= np.eye(2 * n - 2)
        self.obstacles = []
        # Three overlapping discs cover the entire vehicle rectangle. Adding
        # the rear-axle offset is essential: most of the bonnet lies ahead of it.
        self.disc_offsets = config.rear_axle_to_center + np.array([-1., 0., 1.]) * config.ego_length / 3
        radius = np.hypot(config.ego_length / 6, config.ego_width / 2) + config.collision_margin
        for obstacle in problem.obstacles:
            centers = np.asarray(obstacle.centers, dtype=float)
            if centers.shape != (n + 1, 2) or not np.isfinite(centers).all():
                raise ValueError("obstacle centers must be N+1 finite xy predictions")
            radii = np.asarray(obstacle.radii, dtype=float)
            if radii.ndim == 0:
                radii = np.full((n + 1, 2), float(radii))
            elif radii.shape == (n + 1,):
                radii = np.repeat(radii[:, None], 2, axis=1)
            if radii.shape != (n + 1, 2) or not np.isfinite(radii).all() or np.any(radii <= 0):
                raise ValueError("obstacle radii must be positive circular radii or N+1 ellipse axes")
            yaws = _broadcast(obstacle.yaws, n + 1, "obstacle yaws")
            rectangle = None
            uncertainty = None
            if obstacle.rectangle_half_extents is not None:
                rectangle = np.asarray(obstacle.rectangle_half_extents, dtype=float)
                if rectangle.shape == (2,):
                    rectangle = np.tile(rectangle, (n + 1, 1))
                if (rectangle.shape != (n + 1, 2) or not np.isfinite(rectangle).all()
                        or np.any(rectangle <= 0)):
                    raise ValueError("rectangle_half_extents must be positive physical (2,) or (N+1,2) values")
                uncertainty = _broadcast(obstacle.uncertainty_m, n + 1, "obstacle uncertainty_m")
                if np.any(uncertainty < 0):
                    raise ValueError("obstacle uncertainty_m must be nonnegative")
            elif np.any(np.asarray(obstacle.uncertainty_m) != 0):
                raise ValueError("separate uncertainty_m requires rectangle_half_extents")
            # A conservative outer ellipse for obstacle ellipse + vehicle disc:
            # Q_total=(1+b)Q_obstacle+(1+1/b)r²I encloses their Minkowski sum.
            beta = radius / np.sqrt(radii[:, 0] * radii[:, 1])
            expanded = np.sqrt((1 + beta[:, None]) * radii ** 2 +
                               (1 + 1 / beta[:, None]) * radius ** 2)
            def sample_half_steps(data):
                result = np.empty((2 * n,) + data.shape[1:])
                result[0::2] = (data[:-1] + data[1:]) / 2
                result[1::2] = data[1:]
                return result

            circular = np.allclose(radii[:, 0], radii[:, 1], rtol=0, atol=1e-10)
            self.obstacles.append((obstacle.id, sample_half_steps(centers),
                sample_half_steps(expanded), sample_half_steps(np.unwrap(yaws)),
                sample_half_steps(radii[:, 0]) if circular else None,
                sample_half_steps(rectangle) if rectangle is not None else None,
                sample_half_steps(uncertainty) if uncertainty is not None else None))
        # These predictions stay fixed during one solve. Pack all physical
        # rectangles once, then evaluate actor x time in a single geometry call.
        # The constraint rows below still follow the original actor ordering.
        rectangular = [item for item in self.obstacles if item[5] is not None]
        self.rectangles = (np.stack([item[1] for item in rectangular]),
            np.stack([item[3] for item in rectangular]),
            np.stack([item[5] for item in rectangular]),
            np.stack([item[6] for item in rectangular])) if rectangular else None
        self.last_controls = None

    def evaluate(self, flat):
        if self.last_controls is not None and np.array_equal(flat, self.last_controls):
            return
        self.last_controls = np.array(flat, copy=True)
        cfg, n = self.cfg, self.cfg.horizon_steps
        controls = np.asarray(flat).reshape(n, 2)
        states, sensitivity = rollout(self.problem.initial, controls, cfg, sensitivities=True)
        self.states = states
        error = states[:, :2] - self.reference[:, :2]
        lateral = np.sum(error * self.normal, axis=1)
        heading = _angle(states[:, 2] - self.reference[:, 2])
        objective_error = states[:, :2] - self.objective_reference[:, :2]
        objective_along = np.sum(objective_error * self.objective_tangent, axis=1)
        objective_lateral = np.sum(objective_error * self.objective_normal, axis=1)
        objective_heading = _angle(states[:, 2] - self.objective_reference[:, 2])
        speed = states[:, 3] - self.objective_reference[:, 3]
        factor = np.ones(n + 1)
        factor[0], factor[-1] = 0., cfg.terminal_weight
        self.cost = float(np.sum(factor * (cfg.longitudinal_weight * objective_along ** 2 +
            cfg.lateral_weight * objective_lateral ** 2 + cfg.heading_weight * objective_heading ** 2 +
            cfg.speed_weight * speed ** 2)))
        state_gradient = np.zeros_like(states)
        state_gradient[:, :2] = 2 * factor[:, None] * (
            cfg.longitudinal_weight * objective_along[:, None] * self.objective_tangent +
            cfg.lateral_weight * objective_lateral[:, None] * self.objective_normal)
        state_gradient[:, 2] = 2 * factor * cfg.heading_weight * objective_heading
        state_gradient[:, 3] = 2 * factor * cfg.speed_weight * speed
        gradient = np.einsum("ni,nij->j", state_gradient, sensitivity)
        difference = np.vstack([controls[0] - self.previous, np.diff(controls, axis=0)])
        smooth_weights = np.array([cfg.jerk_weight, cfg.steering_rate_weight]) / cfg.dt ** 2
        control_weights = np.array([cfg.acceleration_weight, cfg.steering_weight])
        self.cost += float(np.sum(controls ** 2 * control_weights + difference ** 2 * smooth_weights))
        gradient += (2 * controls * control_weights).ravel()
        gradient += self.delta_matrix.T @ (2 * difference * smooth_weights).ravel()
        self.cost /= n
        self.gradient = gradient / n

        values, jacobians, names = [], [], []

        def add(name, value, jacobian):
            values.append(np.asarray(value).ravel())
            jacobians.append(np.asarray(jacobian).reshape(-1, 2 * n))
            names.extend([name] * np.size(value))

        velocity_jac = sensitivity[1:, 3]
        add("speed_min", states[1:, 3], velocity_jac)
        add("speed_max", self.speed_limit[1:] - states[1:, 3], -velocity_jac)
        lateral_jac = np.einsum("ni,nij->nj", self.normal[1:], sensitivity[1:, :2])
        # half_width is already reduced by the unrotated half-body width by
        # the adapter. Recover the physical road half-width here and include
        # front/rear corner sweep when heading differs from the lane tangent.
        # A rear axle that remains centered must not license its bonnet to cut
        # over a solid line during steering.
        relative_yaw = heading[1:]
        cs, sn = np.cos(relative_yaw), np.sin(relative_yaw)
        center_lateral = lateral[1:] + cfg.rear_axle_to_center * sn + cfg.ego_body_center_lateral * cs
        center_jac = lateral_jac + (cfg.rear_axle_to_center * cs -
            cfg.ego_body_center_lateral * sn)[:, None] * sensitivity[1:, 2]
        cs, sn = np.cos(relative_yaw+cfg.ego_body_yaw), np.sin(relative_yaw+cfg.ego_body_yaw)
        extent = cfg.ego_length / 2 * np.abs(sn) + cfg.ego_width / 2 * np.abs(cs)
        extent_yaw = cfg.ego_length / 2 * np.sign(sn) * cs - cfg.ego_width / 2 * np.sign(cs) * sn
        extent_jac = extent_yaw[:, None] * sensitivity[1:, 2]
        physical_half_width = self.width[1:] + cfg.ego_width / 2
        add("corridor_left", physical_half_width - center_lateral - extent, -center_jac - extent_jac)
        add("corridor_right", physical_half_width + center_lateral - extent, center_jac - extent_jac)
        if self.stop_line is not None:
            point, normal = self.stop_line
            clearance = (point - states[1:, :2]) @ normal
            clearance_jac = -np.einsum("i,nij->nj", normal, sensitivity[1:, :2])
            add("stop_line", clearance, clearance_jac)
            # A finite-horizon position constraint alone can end at the stop
            # line with substantial speed. Keep a braking-distance reserve so
            # this high-speed terminal state is not falsely considered safe.
            braking = abs(cfg.accel_min)
            add("stop_braking_reserve", clearance - states[1:, 3] ** 2 / (2 * braking),
                clearance_jac - (states[1:, 3] / braking)[:, None] * velocity_jac)
        rate_limits = np.tile([cfg.jerk_max * cfg.dt, cfg.steer_rate_max * cfg.dt], n)
        add("control_rate_positive", rate_limits - difference.ravel(), -self.delta_matrix)
        add("control_rate_negative", rate_limits + difference.ravel(), self.delta_matrix)
        # Limit lateral acceleration at both ends of every integration interval.
        steering_jac = np.zeros((n, 2 * n))
        steering_jac[np.arange(n), np.arange(n) * 2 + 1] = 1
        for endpoint in (0, 1):
            v = states[endpoint:n + endpoint, 3]
            tangent = np.tan(controls[:, 1])
            ay = v ** 2 * tangent / cfg.wheelbase
            ay_jac = (2 * v * tangent / cfg.wheelbase)[:, None] * sensitivity[endpoint:n + endpoint, 3]
            ay_jac += (v ** 2 / (cfg.wheelbase * np.cos(controls[:, 1]) ** 2))[:, None] * steering_jac
            add("lateral_acceleration", cfg.lateral_accel_max - ay, -ay_jac)
            add("lateral_acceleration", cfg.lateral_accel_max + ay, ay_jac)
        # Collision checks at dt/2 as well as every prediction endpoint. These
        # are bicycle rollouts with analytic sensitivities, not arbitrary
        # interpolated vehicle poses, so crossing actors cannot skip a whole
        # optimization interval unnoticed.
        collision_states = np.empty((2 * n, 4))
        collision_jac = np.empty((2 * n, 4, 2 * n))
        collision_states[1::2], collision_jac[1::2] = states[1:], sensitivity[1:]
        for index in range(n):
            middle, a, b = bicycle_step(states[index], controls[index], cfg.dt / 2, cfg.wheelbase)
            collision_states[2 * index] = middle
            collision_jac[2 * index] = a @ sensitivity[index]
            collision_jac[2 * index, :, 2 * index:2 * index + 2] += b
        cosine, sine = np.cos(collision_states[:, 2]), np.sin(collision_states[:, 2])
        # All obstacle branches use the same measured rigid bbox transform.
        # A bbox center can have a lateral offset and a local yaw of its own.
        body_shift = np.c_[cfg.rear_axle_to_center*cosine-cfg.ego_body_center_lateral*sine,
                           cfg.rear_axle_to_center*sine+cfg.ego_body_center_lateral*cosine]
        body_pose = collision_states[:, :3].copy()
        body_pose[:, :2] += body_shift
        body_pose[:, 2] += cfg.ego_body_yaw
        body_c, body_s = np.cos(body_pose[:, 2]), np.sin(body_pose[:, 2])
        rectangle_cursor = 0
        if self.rectangles is not None:
            from .mpc_geometry import rectangle_signed_distance
            packed_centers, packed_yaw, packed_dimensions, packed_uncertainty = self.rectangles
            packed_distance, packed_gradient = rectangle_signed_distance(body_pose[None, :, :],
                packed_centers, packed_yaw,
                np.array([cfg.ego_length/2, cfg.ego_width/2]), packed_dimensions)
            packed_state_grad = np.concatenate((packed_gradient,
                np.zeros(packed_gradient.shape[:-1]+(1,))), axis=-1)
            packed_state_grad[:, :, 2] += (-packed_gradient[:, :, 0]*body_shift[None, :, 1]
                +packed_gradient[:, :, 1]*body_shift[None, :, 0])
            packed_jacobian = np.einsum("rni,nij->rnj", packed_state_grad, collision_jac)
            packed_values = packed_distance-cfg.collision_margin-packed_uncertainty
        for obstacle_id, centers, axes, yaw, circular_radius, rectangle, uncertainty in self.obstacles:
            if rectangle is not None:
                add("obstacle:%s" % obstacle_id, packed_values[rectangle_cursor],
                    packed_jacobian[rectangle_cursor])
                rectangle_cursor += 1
                continue
            if circular_radius is not None:
                # Exact circle-to-oriented-rectangle signed distance avoids
                # artificial extra bonnet clearance for nearby pedestrians.
                # The entire physical ego rectangle and configured margin are
                # still enforced; no lateral body corner is omitted.
                aligned_body = cfg.ego_body_center_lateral == 0. and cfg.ego_body_yaw == 0.
                if aligned_body:
                    # Preserve the established circle solver's arithmetic and
                    # active-set behavior for its original vehicle geometry.
                    diff = centers-collision_states[:, :2]
                    u = diff[:, 0]*cosine+diff[:, 1]*sine-cfg.rear_axle_to_center
                    v = -diff[:, 0]*sine+diff[:, 1]*cosine
                else:
                    diff = centers-body_pose[:, :2]
                    u = diff[:, 0]*body_c+diff[:, 1]*body_s
                    v = -diff[:, 0]*body_s+diff[:, 1]*body_c
                q = np.c_[np.abs(u) - cfg.ego_length / 2, np.abs(v) - cfg.ego_width / 2]
                positive = np.maximum(q, 0.)
                outside = np.linalg.norm(positive, axis=1)
                distance = outside + np.minimum(np.max(q, axis=1), 0.)
                grad = positive / np.maximum(outside[:, None], 1e-12)
                inside = outside < 1e-12
                grad[inside] = 0.
                grad[np.flatnonzero(inside), np.argmax(q[inside], axis=1)] = 1.
                grad *= np.where(np.c_[u, v] >= 0, 1., -1.)
                gu, gv = grad[:, 0], grad[:, 1]
                if aligned_body:
                    state_grad = np.c_[-cosine*gu+sine*gv,-sine*gu-cosine*gv,
                        v*gu-(u+cfg.rear_axle_to_center)*gv,np.zeros(2*n)]
                else:
                    gx,gy = -body_c*gu+body_s*gv,-body_s*gu-body_c*gv
                    state_grad = np.c_[gx,gy,v*gu-u*gv-gx*body_shift[:, 1]+gy*body_shift[:, 0],np.zeros(2*n)]
                add("obstacle:%s" % obstacle_id, distance - circular_radius - cfg.collision_margin,
                    np.einsum("ni,nij->nj", state_grad, collision_jac))
                continue
            c, s = np.cos(yaw), np.sin(yaw)
            for offset in np.array([-1., 0., 1.])*cfg.ego_length/3:
                shift = body_shift + offset*np.c_[body_c, body_s]
                points = collision_states[:, :2] + shift
                diff = points - centers
                u = diff[:, 0] * c + diff[:, 1] * s
                v = -diff[:, 0] * s + diff[:, 1] * c
                value = (u / axes[:, 0]) ** 2 + (v / axes[:, 1]) ** 2 - 1
                point_grad = 2 * np.c_[u * c / axes[:, 0] ** 2 - v * s / axes[:, 1] ** 2,
                                       u * s / axes[:, 0] ** 2 + v * c / axes[:, 1] ** 2]
                state_grad = np.c_[point_grad,
                    np.sum(point_grad * np.c_[-shift[:, 1], shift[:, 0]], axis=1), np.zeros(2*n)]
                add("obstacle:%s" % obstacle_id, value,
                    np.einsum("ni,nij->nj", state_grad, collision_jac))
        self.constraints = np.concatenate(values)
        self.jacobian = np.vstack(jacobians)
        self.constraint_names = names

    def objective(self, flat):
        self.evaluate(flat)
        return self.cost

    def objective_jac(self, flat):
        self.evaluate(flat)
        return self.gradient

    def inequalities(self, flat):
        self.evaluate(flat)
        return self.constraints

    def inequalities_jac(self, flat):
        self.evaluate(flat)
        return self.jacobian


class MPCSolver:
    """SLSQP direct shooting with analytic derivatives and bounded warm starts.

    Feasibility and optimizer convergence are reported independently. A finite
    SLSQP iterate that still violates a hard constraint is never accepted as a
    substitute trajectory. The caller must handle ``feasible=False`` explicitly.
    """
    def __init__(self, config=None):
        self.config = config or MPCConfig()
        self._warm_starts = {}

    def clear_warm_start(self):
        self._warm_starts.clear()

    def _seed(self, problem, *, braking=False, departure_delay_steps=0):
        cfg, state = self.config, problem.initial
        reference = np.asarray(problem.reference.points)
        controls = np.zeros((cfg.horizon_steps, 2))
        simulated = np.array([state.x, state.y, state.yaw, state.speed])
        previous = np.array([state.acceleration, state.steering])
        for index in range(cfg.horizon_steps):
            # A geometric controller only seeds the optimizer; its rollout is
            # not declared to be an MPC solution without the same constraints.
            look = min(cfg.horizon_steps, index + max(2, int(1.2 / cfg.dt)))
            target = reference[look]
            dx, dy = target[:2] - simulated[:2]
            bearing = _angle(np.arctan2(dy, dx) - simulated[2])
            distance2 = max(.8, dx * dx + dy * dy)
            steering = np.arctan(2 * cfg.wheelbase * np.sin(bearing) / np.sqrt(distance2))
            acceleration = (-simulated[3] / cfg.dt if braking or index < departure_delay_steps else
                            1.4 * (reference[index + 1, 3] - simulated[3]))
            control = np.clip([acceleration, steering],
                              [cfg.accel_min, -cfg.steer_max], [cfg.accel_max, cfg.steer_max])
            limits = np.array([cfg.jerk_max, cfg.steer_rate_max]) * cfg.dt
            control = np.clip(control, previous - limits, previous + limits)
            controls[index] = control
            simulated, _, _ = bicycle_step(simulated, control, cfg.dt, cfg.wheelbase)
            previous = control
        return controls.ravel()

    def solve(self, problem, *, acceleration_upper_bounds=None):
        """Optimize with optional stricter, per-step acceleration box limits.

        The adapter uses this internal policy input only for verified braking
        recovery. Bounds intersect the ordinary physical limits and are checked
        by the optimizer and final feasibility test; they never relax jerk,
        collision or road constraints. Omitting them retains ordinary MPC.
        """
        started = perf_counter()
        cfg = self.config
        evaluation = _Evaluation(cfg, problem)
        lower = np.tile([cfg.accel_min, -cfg.steer_max], cfg.horizon_steps)
        upper = np.tile([cfg.accel_max, cfg.steer_max], cfg.horizon_steps)
        if acceleration_upper_bounds is not None:
            requested_upper = np.asarray(acceleration_upper_bounds, dtype=float)
            if (requested_upper.shape != (cfg.horizon_steps,) or not np.isfinite(requested_upper).all()
                    or np.any(requested_upper < cfg.accel_min)):
                raise ValueError("acceleration_upper_bounds must be N finite values >= accel_min")
            upper[::2] = np.minimum(upper[::2], requested_upper)
        if (problem.initial.speed == 0. and problem.initial.acceleration >= 0.
                and upper[0] >= 0.):
            # At rest with no brake demand to release, negative first-step
            # commands are dynamically identical to zero but create a flat
            # half-space in direct shooting. Pick its neutral representative.
            # A genuinely negative dispatched history retains its full bounds.
            lower[0] = 0.
        seeds = []
        if problem.reference.name in self._warm_starts:
            old = self._warm_starts[problem.reference.name]
            seeds.append(("warm_start", np.vstack([old[1:], old[-1]]).ravel()))
        seeds.append(("reference_tracking", self._seed(problem)))
        if cfg.multi_start and problem.obstacles:
            seeds.append(("braking", self._seed(problem, braking=True)))
            if problem.initial.speed < .5 and np.max(np.asarray(problem.reference.points)[:, 3]) > .5:
                # At zero speed the exact stop dynamics have a flat braking
                # branch. A bounded delayed-departure seed explores the other
                # side of that active mode while steering builds at its normal
                # rate. This is optimized and checked with the same constraints;
                # it is not an unchecked geometric escape trajectory.
                delay = max(1, min(cfg.horizon_steps // 3, round(.4 / cfg.dt)))
                seeds.append(("delayed_departure", self._seed(problem, departure_delay_steps=delay)))
        # Exact braking correctly has zero state sensitivity after stopping.
        # A feasible negative-demand warm start can therefore converge on that
        # flat branch even after a light clears. Explore a departure mode only
        # for this low-speed/no-restart result, including when optional ordinary
        # multi-start is disabled. This changes neither feasibility nor the
        # objective used to choose the best verified solution.
        departure_search_eligible = bool(problem.initial.speed < .5
            and np.max(evaluation.reference[:, 3]) > .03 and np.any(upper[::2] > 0.))
        stationary_branch_seeds = []
        attempts, solutions = [], []
        seen_seeds = []
        for seed_name, seed in seeds:
            seed = np.clip(seed, lower, upper)
            if any(np.array_equal(seed, previous_seed) for previous_seed in seen_seeds):
                continue
            seen_seeds.append(seed)
            verified_iterate = None
            verified_iterate_count = 0

            def retain_verified_iterate(flat):
                nonlocal verified_iterate, verified_iterate_count
                evaluation.evaluate(flat)
                if (not np.isfinite(flat).all() or not np.isfinite(evaluation.states).all()
                        or not np.isfinite(evaluation.cost) or not np.isfinite(evaluation.constraints).all()):
                    return
                violation = max(0., float(np.max(lower - flat)), float(np.max(flat - upper)),
                                -float(np.min(evaluation.constraints)))
                if violation <= cfg.constraint_tolerance:
                    verified_iterate_count += 1
                    if verified_iterate is None or evaluation.cost < verified_iterate[0]:
                        verified_iterate = evaluation.cost, np.array(flat, copy=True)

            with warnings.catch_warnings(record=True) as optimizer_warnings:
                # SciPy may clip an intermediate trial step back to its box
                # bounds. Preserve this numerical warning in the audit rather
                # than printing it every replan; final bounds are still checked.
                warnings.simplefilter("always", RuntimeWarning)
                answer = minimize(evaluation.objective, seed, jac=evaluation.objective_jac,
                    method="SLSQP", bounds=list(zip(lower, upper)),
                    constraints=[{"type": "ineq", "fun": evaluation.inequalities,
                                  "jac": evaluation.inequalities_jac}],
                    callback=retain_verified_iterate,
                    options={"maxiter": cfg.max_iterations, "ftol": cfg.ftol, "disp": False})
            evaluation.evaluate(answer.x)
            optimizer_success = bool(answer.success)
            optimizer_message = str(answer.message)
            selected_iterate = "optimizer_final"
            # A later SLSQP trial may leave a previously feasible active mode,
            # especially at a stop/restart boundary. Preserve only actual
            # optimizer iterates that passed every original constraint. This
            # retains a usable finite-budget solution, never a convergence claim.
            final_violation = max(0., float(np.max(lower - answer.x)), float(np.max(answer.x - upper)),
                                  -float(np.min(evaluation.constraints)))
            if (verified_iterate is not None and
                    (not np.isfinite(evaluation.cost) or final_violation > cfg.constraint_tolerance
                     or (not answer.success and verified_iterate[0] < evaluation.cost))):
                answer = OptimizeResult(answer)
                answer.x = verified_iterate[1]
                answer.success = False
                answer.message = "Retained verified feasible iterate; final optimizer: " + optimizer_message
                selected_iterate = "verified_feasible_iteration"
                evaluation.evaluate(answer.x)
            bound_violation = max(0., float(np.max(lower - answer.x)), float(np.max(answer.x - upper)))
            violation = max(bound_violation, max(0., -float(np.min(evaluation.constraints))))
            finite = bool(np.isfinite(answer.x).all() and np.isfinite(evaluation.states).all()
                          and np.isfinite(evaluation.cost) and np.isfinite(evaluation.constraints).all())
            feasible = finite and violation <= cfg.constraint_tolerance
            stopped_intervals = int(np.count_nonzero(
                (evaluation.states[1:, 3] == 0.) & (answer.x[::2] < 0.)))
            stationary_branch = bool(feasible and departure_search_eligible and stopped_intervals
                and evaluation.states[-1, 3] <= .03
                and np.max(evaluation.states[1:, 3]) <= max(.03, problem.initial.speed))
            if stationary_branch:
                stationary_branch_seeds.append(seed_name)
                if not any(name == "delayed_departure" for name, _ in seeds):
                    delay = max(1, min(cfg.horizon_steps // 3, round(.4 / cfg.dt)))
                    seeds.append(("delayed_departure", self._seed(problem, departure_delay_steps=delay)))
            worst = int(np.argmin(evaluation.constraints))
            summary = {"seed": seed_name, "optimizer_success": optimizer_success,
                "optimizer_status": int(answer.status), "message": optimizer_message,
                "selected_iterate": selected_iterate,
                "verified_feasible_iterate_count": verified_iterate_count,
                "iterations": int(answer.nit), "cost": float(evaluation.cost),
                "feasible": feasible, "max_constraint_violation": violation,
                "terminal_speed_mps": float(evaluation.states[-1, 3]),
                "maximum_predicted_speed_mps": float(np.max(evaluation.states[:, 3])),
                "stopped_under_braking_intervals": stopped_intervals,
                "stationary_branch_exploration_required": stationary_branch,
                "worst_constraint": evaluation.constraint_names[worst],
                "numerical_warnings": sorted(set(str(item.message) for item in optimizer_warnings))}
            attempts.append(summary)
            solutions.append((feasible, float(evaluation.cost), violation, answer,
                              np.array(evaluation.states, copy=True), summary))
            # A converged feasible warm/reference solution is sufficient; other
            # maneuver corridors are compared by the external planner.
            if feasible and answer.success and not stationary_branch:
                break
        best = min(solutions, key=lambda item: (not item[0], item[1] if item[0] else item[2]))
        feasible, cost, violation, answer, states, summary = best
        controls = answer.x.reshape(cfg.horizon_steps, 2).copy()
        if feasible:
            if len(self._warm_starts) >= 16 and problem.reference.name not in self._warm_starts:
                self._warm_starts.pop(next(iter(self._warm_starts)))
            self._warm_starts[problem.reference.name] = controls.copy()
        evaluation.evaluate(answer.x)
        minimums = {}
        for key, value in zip(evaluation.constraint_names, evaluation.constraints):
            minimums[key] = min(minimums.get(key, float("inf")), float(value))
        initial_error = states[0, :2] - evaluation.reference[0, :2]
        initial_lateral = float(initial_error @ evaluation.normal[0])
        initial_heading = float(_angle(states[0, 2] - evaluation.reference[0, 2]))
        initial_center_lateral = initial_lateral + cfg.rear_axle_to_center*np.sin(initial_heading) + cfg.ego_body_center_lateral*np.cos(initial_heading)
        initial_body_heading = initial_heading+cfg.ego_body_yaw
        initial_extent = cfg.ego_length/2*abs(np.sin(initial_body_heading)) + cfg.ego_width/2*abs(np.cos(initial_body_heading))
        initial_road_half = evaluation.width[0] + cfg.ego_width / 2
        diagnostics = {"solver": "scipy_slsqp_direct_shooting", "state_origin": "rear_axle",
            "solve_policy_version": MPC_SOLVE_POLICY_VERSION,
            "selected_seed": summary["seed"],
            "stationary_branch_search": {"eligible": departure_search_eligible,
                "triggered_by": stationary_branch_seeds,
                "selection": "minimum_original_cost_among_verified_feasible_attempts"},
            "longitudinal_dynamics": "forward_only_exact_within_step_braking_stop",
            "jerk_limits_apply_to": "dispatched_acceleration_demand",
            "previous_acceleration_demand_mps2": float(problem.initial.acceleration),
            "acceleration_upper_bounds_mps2": (upper[::2].tolist()
                if acceleration_upper_bounds is not None else None),
            "stationary_hold_objective": evaluation.stationary_hold_objective,
            "stopped_under_braking_intervals": int(np.count_nonzero(
                (states[1:, 3] == 0.) & (controls[:, 0] < 0.))),
            "dt_s": cfg.dt, "horizon_steps": cfg.horizon_steps,
            "analytic_objective_gradient": True, "analytic_constraint_jacobian": True,
            "constraint_tolerance": cfg.constraint_tolerance,
            "max_constraint_violation": violation, "constraint_minimums": minimums,
            "attempts": attempts, "wall_time_s": perf_counter() - started,
            "obstacle_count": len(evaluation.obstacles), "reference_name": problem.reference.name,
            "initial_corridor": {"rear_lateral_error_m": initial_lateral,
                "heading_error_rad": initial_heading,
                "body_left_clearance_m": float(initial_road_half - initial_center_lateral - initial_extent),
                "body_right_clearance_m": float(initial_road_half + initial_center_lateral - initial_extent)},
            "collision_geometry_version": MPC_COLLISION_GEOMETRY_VERSION,
            "vehicle_footprint_model": "physical_obb_for_explicit_rectangles_exact_circle_legacy_outer_ellipse",
            "obstacle_shape_counts": {"physical_rectangles": sum(ob.rectangle_half_extents is not None for ob in problem.obstacles),
                "legacy_shapes": sum(ob.rectangle_half_extents is None for ob in problem.obstacles)},
            "collision_check": "prediction_endpoints_and_bicycle_midpoints",
            "collision_sample_dt_s": cfg.dt / 2}
        converged = bool(answer.success and feasible)
        status = ("converged" if converged else "feasible_not_converged" if feasible else "infeasible")
        return MPCResult(states, controls, feasible, converged, status, str(answer.message),
                         cost, int(answer.nit), diagnostics)
