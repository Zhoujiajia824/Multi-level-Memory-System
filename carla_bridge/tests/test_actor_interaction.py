"""Independent safeguards for evidence-gated traffic following forecasts.

The ego/leader is an already observed stationary car, not a hypothetical MPC
candidate. These tests keep the original obstacle footprint and uncertainty.
"""
from types import SimpleNamespace as NS

import numpy as np

from carla_bridge.planning.actor_motion import ActorMotionHistory, predict_actor_motion
from carla_bridge.planning.mpc_solver import (
    MPCConfig, MPCInitialState, MPCObstacle, MPCProblem, MPCReference, MPCSolver,
)


class Actor:
    def __init__(self, ident, x, *, speed=0., y=0.):
        self.id, self.type_id = ident, "vehicle.test"
        self.x, self.y, self.speed = float(x), float(y), float(speed)
        self.bounding_box = NS(location=NS(x=0., y=0.), rotation=NS(yaw=0.),
                               extent=NS(x=2.4, y=1.))

    def get_transform(self):
        return NS(location=NS(x=self.x, y=self.y, z=.2), rotation=NS(yaw=0.))

    def get_velocity(self):
        return NS(x=self.speed, y=0., z=0.)


class Waypoint:
    def __init__(self, x, y=0.):
        self.transform = NS(location=NS(x=float(x), y=float(y), z=.2), rotation=NS(yaw=0.))
        self.road_id, self.section_id, self.lane_id = 5, 0, -1
        self.lane_type, self.lane_width, self.is_junction = "Driving", 3.5, False

    def next(self, distance):
        return [Waypoint(self.transform.location.x + distance, self.transform.location.y)]


class Road:
    def get_waypoint(self, location, **_kwargs):
        return Waypoint(location.x)


def observed_pair(*, current_speed=2.2, current_x=-9.56, braking=True, cut_in=False):
    history = ActorMotionHistory()
    follower, leader = Actor(90, current_x, speed=current_speed), Actor(1, 0.)
    duration, deceleration = 1.4, (2. if braking else 0.)
    for now in np.arange(8) * .2:
        remaining = duration - now
        follower.x = current_x - current_speed * remaining - .5 * deceleration * remaining ** 2
        follower.speed = current_speed + deceleration * remaining
        # The new car has no established leader relationship, although the
        # follower happened to brake before that car cut into this lane.
        leader.y = 3.5 if cut_in and now < duration - 1e-8 else 0.
        history.update(actors=[follower, leader], sim_time_s=float(now))
    return follower, leader, history


def forecast(pair, *, interaction):
    follower, leader, history = pair
    times = np.arange(21) * .2
    kwargs = {"history": history, "leader_actors": (leader,)} if interaction else {}
    return predict_actor_motion(follower, history.motion_for(follower), times, Road(), **kwargs)


def hold_result(centers, yaws, audit):
    cfg = MPCConfig(max_iterations=30, multi_start=False)
    # Ego actor center is at x=0. Both its physical rectangle and the existing
    # conservative vehicle ellipse/uncertainty remain unchanged.
    rear = MPCInitialState(-cfg.rear_axle_to_center, 0., 0., 0.)
    reference = MPCReference("center:STOP", np.tile(
        [rear.x, 0., 0., 0.], (cfg.horizon_steps + 1, 1)), .65, 8., {
        "stop_line": {"point": [rear.x + .2, 0.], "normal": [1., 0.]}})
    axes = np.tile(np.sqrt(2.) * np.array([2.4, 1.]), (cfg.horizon_steps + 1, 1))
    axes += np.asarray(audit["uncertainty_margin_m"])[:, None]
    return MPCSolver(cfg).solve(MPCProblem(rear, reference,
        (MPCObstacle("90", centers, axes, yaws),)))


def test_established_braking_follower_uses_elapsed_reaction_and_enables_verified_red_light_hold():
    pair = observed_pair()
    original = forecast(pair, interaction=False)
    following = forecast(pair, interaction=True)
    response = following[2]["following_response"]
    assert response["reaction_configured_s"] == .4
    assert response["reaction_remaining_s"] == 0.
    np.testing.assert_allclose(following[0][0], original[0][0], atol=1e-10)
    np.testing.assert_allclose(following[1][0], original[1][0], atol=1e-10)
    np.testing.assert_allclose(following[2]["uncertainty_margin_m"], original[2]["uncertainty_margin_m"])
    # A continuation that ignores the already observed braking runs into the
    # real stationary leader. Bounded following can instead maintain a gap.
    assert following[0][-1, 0] < original[0][-1, 0] - 4.
    assert not hold_result(*original).feasible
    bounded = hold_result(*following)
    assert bounded.feasible, bounded.diagnostics
    assert bounded.diagnostics["constraint_minimums"]["obstacle:90"] >= -.002


def test_a_close_fast_rear_vehicle_is_not_made_safe_by_following_assumption():
    pair = observed_pair(current_speed=8., current_x=-7.)
    original = forecast(pair, interaction=False)
    following = forecast(pair, interaction=True)
    np.testing.assert_allclose(following[0][:3], original[0][:3], atol=1e-10)
    # Even if prior braking permits the interaction model, reaction delay and
    # bounded deceleration cannot remove this real near-term rear collision.
    assert not hold_result(*following).feasible


def test_new_cut_in_does_not_inherit_braking_evidence_for_an_unrelated_leader():
    pair = observed_pair(cut_in=True)
    original, following = forecast(pair, interaction=False), forecast(pair, interaction=True)
    np.testing.assert_allclose(following[0], original[0], atol=1e-10)
    np.testing.assert_allclose(following[1], original[1], atol=1e-10)
    assert not hold_result(*following).feasible


def test_same_lane_without_observed_braking_does_not_assume_cooperation():
    pair = observed_pair(braking=False)
    original, following = forecast(pair, interaction=False), forecast(pair, interaction=True)
    np.testing.assert_allclose(following[0], original[0], atol=1e-10)
    assert not hold_result(*following).feasible


def test_interaction_does_not_clamp_travel_beyond_the_proven_path_endpoint():
    class ShortChordWaypoint(Waypoint):
        def next(self, distance):
            return [ShortChordWaypoint(self.transform.location.x + .99 * distance)]

    class ShortChordRoad(Road):
        def get_waypoint(self, location, **_kwargs):
            return ShortChordWaypoint(location.x)

    follower, leader, history = observed_pair()
    settings = {"following_max_braking_mps2": .001, "following_comfort_braking_mps2": .001}
    args = (follower, history.motion_for(follower), np.arange(21) * .2, ShortChordRoad(), settings)
    original = predict_actor_motion(*args)
    following = predict_actor_motion(*args, history=history, leader_actors=(leader,))
    assert following[2]["following_response"]["applied"] is False
    assert following[2]["following_response"]["reason"] == "bounded_travel_exceeds_proven_connected_path"
    np.testing.assert_allclose(following[0], original[0], atol=1e-10)
    np.testing.assert_allclose(following[1], original[1], atol=1e-10)
