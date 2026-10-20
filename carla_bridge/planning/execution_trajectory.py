"""Serialize the selected native rollout in the capture-time CARLA actor frame.

PDM's generator reference starts at a path projection, which need not equal the
current vehicle pose. Its simulator starts at the measured pose instead. Only
the latter geometry was scored and supplies native steering for execution.
The IDM velocity remains the existing longitudinal *target*, not a claim about
the simulated or measured vehicle velocity.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from carla_bridge.state.coords import global_to_ego


def build_scored_execution_trajectory(scored_states, reference_states, idm_velocity,
                                      initial_state, capture_transform, state_index, dt):
    """Convert every scored pose; never move just t=0 or translate a stale plan."""
    states, reference, velocity = (np.asarray(value) for value in
                                   (scored_states, reference_states, idm_velocity))
    if (states.ndim != 2 or states.shape != reference.shape or len(states) < 2
            or velocity.shape != (len(states),) or not math.isfinite(dt) or dt <= 0
            or not all(np.isfinite(value).all() for value in (states, reference, velocity))):
        raise ValueError("Invalid selected PDM rollout or IDM velocity profile")
    rear = initial_state.rear_axle
    initial_heading = float(states[0, state_index.HEADING])
    heading_error = math.atan2(math.sin(initial_heading - rear.heading),
                               math.cos(initial_heading - rear.heading))
    pose_error = float(np.linalg.norm(states[0, state_index.POINT] - [rear.x, rear.y]))
    if pose_error > 1e-7 or abs(heading_error) > 1e-7:
        raise ValueError("Selected PDM rollout does not start at the measured ego state")

    yaw = math.radians(capture_transform.rotation.yaw)
    actor = np.array([capture_transform.location.x, -capture_transform.location.y])
    delta = actor - np.array([rear.x, rear.y])
    c, s = math.cos(rear.heading), math.sin(rear.heading)
    # One rigid planar body transform, derived from the same captured state.
    # This includes the current projected bbox offset on a pitched/rolled car;
    # the identical transform applies to every future pose and to audit data.
    actor_from_rear = np.array([c * delta[0] + s * delta[1],
                               -s * delta[0] + c * delta[1]])

    def actor_local(row):
        heading = float(row[state_index.HEADING])
        c, s = math.cos(heading), math.sin(heading)
        xy = row[state_index.POINT] + [c * actor_from_rear[0] - s * actor_from_rear[1],
                                       s * actor_from_rear[0] + c * actor_from_rear[1]]
        return global_to_ego(xy[0], -xy[1], actor[0], -actor[1], yaw)

    trajectory = []
    for i, row in enumerate(states):
        x, y = actor_local(row)
        heading = float(row[state_index.HEADING]) + yaw
        trajectory.append(dict(t=round(i * dt, 8), x=float(x), y=float(y),
            optional_heading_rad=math.atan2(math.sin(heading), math.cos(heading)),
            optional_v=float(max(0., velocity[i]))))
    reference_origin = actor_local(reference[0])
    audit = dict(geometry_source="native_pdm_scored_rollout",
        speed_profile_source="native_pdm_idm_target_velocity",
        frame="capture_time_carla_actor_forward_left",
        sample_count=len(states), sample_interval_s=float(dt),
        selected_scored_states_sha256=hashlib.sha256(states.tobytes()).hexdigest(),
        initial_scored_pose_error_m=pose_error,
        reference_origin_xy_m=[float(value) for value in reference_origin],
        reference_origin_distance_m=math.hypot(*reference_origin),
        execution_origin_distance_m=math.hypot(trajectory[0]["x"], trajectory[0]["y"]),
        actor_from_rear_xy_m=actor_from_rear.tolist())
    return trajectory, audit
