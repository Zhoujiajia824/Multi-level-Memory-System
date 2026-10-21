"""Finite measured-response phases, without assuming an NPC always gives way."""
import numpy as np
import pytest

from carla_bridge.planning.actor_motion import ActorMotionHistory, predict_actor_motion
from carla_bridge.tests.test_actor_interaction import Actor, Road
from carla_bridge.tests.test_actor_interaction import Waypoint
from carla_bridge.tests.test_actor_motion import Actor as MotionActor


def advance(pair, time, speed):
    follower, leader, history = pair
    old = history.last_time_s
    if old is not None:
        follower.x += .5*(follower.speed+speed)*(time-old)
    follower.speed = speed
    history.update(actors=[follower, leader], sim_time_s=time)
    return predict_actor_motion(follower, history.motion_for(follower), np.arange(21)*.2,
        Road(), history=history, leader_actors=(leader,))


def established():
    pair = Actor(90, -35., speed=5.), Actor(1, 0.), ActorMotionHistory()
    outputs = []
    for time in np.arange(8)*.2:
        outputs.append(advance(pair, float(time), 5.-2*time))
    return pair, outputs


def test_reaction_is_anchored_to_first_response_and_duplicate_frames_do_not_renew_proof():
    pair, outputs = established()
    first = next(output[2]["following_response"] for output in outputs
        if output[2]["following_response"]["applied"])
    current = outputs[-1][2]["following_response"]
    assert first["reaction_remaining_s"] == pytest.approx(.4)
    assert current["response_started_s"] == first["response_started_s"]
    assert current["reaction_remaining_s"] == pytest.approx(0.)
    deadline = current["response_deadline_s"]
    again = advance(pair, 1.4, 2.2)[2]["following_response"]
    assert again["response_deadline_s"] == deadline
    assert again["response_proof_time_s"] == current["response_proof_time_s"]


@pytest.mark.parametrize("sample_counts", [(17, 21, 17, 21), (21, 17, 21, 17)])
def test_context_and_mpc_sampling_in_same_frame_do_not_mutate_response_phase(sample_counts):
    from copy import deepcopy

    pair, outputs = established()
    follower, leader, history = pair
    state = deepcopy(history._following_states[follower.id])
    expected = outputs[-1][2]["following_response"]
    for count in sample_counts:
        # scene_context samples 0:.25:4 and MPC samples 0:.2:4. A repeated
        # consumer must not re-read old speed drops as fresh braking evidence.
        times = np.linspace(0., 4., count)
        response = predict_actor_motion(follower, history.motion_for(follower), times,
            Road(), history=history, leader_actors=(leader,))[2]["following_response"]
        assert response["applied"]
        assert response["response_proof_time_s"] == expected["response_proof_time_s"]
        assert response["response_started_s"] == expected["response_started_s"]
        assert response["response_deadline_s"] == expected["response_deadline_s"]
        assert response["reaction_remaining_s"] == expected["reaction_remaining_s"]
        assert history._following_states[follower.id] == state


def test_stable_following_proof_expires_without_new_independent_deceleration():
    pair, outputs = established()
    proof = outputs[-1][2]["following_response"]["response_proof_time_s"]
    for time in np.arange(1.6, proof+2.01, .2):
        response = advance(pair, float(time), 2.2)[2]["following_response"]
        assert response["applied"]
        assert response["response_proof_time_s"] == proof
    expired = advance(pair, proof+2.2, 2.2)[2]["following_response"]
    assert not expired["applied"]
    assert expired["response_phase_reason"] == "observed_response_expired"


def test_one_speed_bump_preserves_response_but_sustained_acceleration_suspends_it():
    pair, outputs = established()
    proof = outputs[-1][2]["following_response"]["response_proof_time_s"]
    bump = advance(pair, 1.6, 2.8)[2]["following_response"]
    assert bump["applied"] and bump["response_proof_time_s"] == proof
    acceleration = advance(pair, 1.8, 3.4)[2]["following_response"]
    assert not acceleration["applied"]
    assert acceleration["reason"] == "sustained_measured_reacceleration"
    # A genuinely new, independently observed brake response can resume it.
    braking = advance(pair, 2., 2.7)[2]["following_response"]
    assert braking["applied"]
    assert braking["response_proof_time_s"] == 2.
    assert braking["reaction_remaining_s"] == pytest.approx(.4)


def test_new_leader_or_actor_teleport_cannot_inherit_an_observed_response():
    pair, _ = established()
    follower, _, history = pair
    new_leader = Actor(2, 0.)
    response = advance((follower, new_leader, history), 1.6, 2.2)[2]["following_response"]
    assert not response["applied"]
    assert follower.id not in history._following_states
    follower.x += 100.
    response = advance((follower, new_leader, history), 1.8, 2.2)[2]["following_response"]
    assert not response["applied"]
    assert history.motion_for(follower).audit["event"] == "teleport_or_invalid_interval_reset"


@pytest.mark.parametrize("acceleration,initial", [(-14.,12.), (14.,1.)])
def test_verified_api_integral_preserves_physical_hard_braking_and_acceleration(acceleration, initial):
    actor, history = MotionActor(), ActorMotionHistory()
    for stamp in np.arange(4)*.2:
        actor.x = initial*stamp+.5*acceleration*stamp**2
        actor.vx = initial+acceleration*stamp
        history.update(actors=[actor], sim_time_s=float(stamp))
    motion = history.motion_for(actor)
    assert motion.vx == pytest.approx(actor.vx)
    assert motion.source == "history_validated_api_velocity"
    assert motion.audit["api_integrated_displacement_consistent"]
    assert motion.audit["api_integrated_displacement_error_m"] < 1e-10


def test_bad_zero_vehicle_api_and_impossible_api_jump_remain_rejected():
    actor, history = MotionActor(), ActorMotionHistory()
    for stamp in np.arange(4)*.2:
        actor.x, actor.vx = 5*stamp, 0.
        history.update(actors=[actor], sim_time_s=float(stamp))
    assert history.motion_for(actor).vx == pytest.approx(5.)
    assert not history.motion_for(actor).audit["api_integrated_displacement_consistent"]
    actor.x, actor.vx = 4., 20.
    history.update(actors=[actor], sim_time_s=.8)
    assert history.motion_for(actor).vx == pytest.approx(5.)
    assert not history.motion_for(actor).audit["api_integrated_displacement_consistent"]


@pytest.mark.parametrize("speed", [.2, .3, .5])
def test_continuous_zero_api_cannot_hide_observed_vehicle_crawl(speed):
    actor, history = MotionActor(), ActorMotionHistory()
    for stamp in (0., .2, .4, .6):
        actor.x, actor.vx = speed*stamp, 0.
        history.update(actors=[actor], sim_time_s=stamp)
    motion = history.motion_for(actor)
    assert motion.vx == pytest.approx(speed)
    assert motion.source == "position_history_corrected_api"
    assert motion.audit["continuous_zero_api_contradicted_by_displacement"]


def test_physical_sharp_stop_keeps_zero_instantaneous_api_after_it_reaches_zero():
    actor, history = MotionActor(), ActorMotionHistory()
    for stamp, position, speed in ((0.,0.,8.),(.2,1.4,6.),(.4,2.2,2.),
            (.6,2.4,0.),(.8,2.4,0.),(1.,2.4,0.)):
        actor.x, actor.vx = position, speed
        history.update(actors=[actor], sim_time_s=stamp)
        if stamp >= .6:
            assert history.motion_for(actor).speed_mps == 0.
            assert not history.motion_for(actor).audit["continuous_zero_api_contradicted_by_displacement"]


def test_low_speed_leader_beyond_horizon_can_be_proved_across_connected_road_seam():
    from carla_bridge.tests.test_actor_interaction import observed_pair

    class SeamWaypoint(Waypoint):
        def __init__(self, x):
            super().__init__(x)
            self.road_id = 18 if x < -.5 else 19

        def next(self, distance):
            return [SeamWaypoint(self.transform.location.x+distance)]

    class SeamRoad(Road):
        def get_waypoint(self, location, **kwargs):
            return SeamWaypoint(location.x)

    follower, leader, history = observed_pair()
    # The 8.8m motion horizon ends at x=-.76 on road18, but the real leader
    # lies at x=0 on its uniquely connected successor road19.
    _, _, audit = predict_actor_motion(follower, history.motion_for(follower), np.arange(21)*.2,
        SeamRoad(), history=history, leader_actors=(leader,))
    assert all(key[0] == 18 for key in audit["connected_lane_keys"])
    assert audit["following_response"]["applied"]


def test_temporarily_unknown_existing_leader_disables_forecast_without_restarting_response():
    class UnknownLeaderRoad(Road):
        def get_waypoint(self, location, **kwargs):
            return None if abs(location.x) < 1e-10 else super().get_waypoint(location, **kwargs)

    pair, outputs = established()
    follower, leader, history = pair
    original = outputs[-1][2]["following_response"]
    # Advance a real measurement, then make only this current leader's map
    # location unavailable. Its pose and the follower path remain measurable.
    advance(pair, 1.6, 2.2)
    centers, _, unknown = predict_actor_motion(follower, history.motion_for(follower),
        np.arange(21)*.2, UnknownLeaderRoad(), history=history, leader_actors=(leader,))
    unconditioned = predict_actor_motion(follower, history.motion_for(follower),
        np.arange(21)*.2, UnknownLeaderRoad())[0]
    response = unknown["following_response"]
    assert not response["applied"]
    assert response["reason"] == "existing_leader_connection_unproven_prediction_disabled"
    assert response["response_proof_time_s"] == original["response_proof_time_s"]
    np.testing.assert_allclose(centers, unconditioned)
    resumed = advance(pair, 1.8, 2.2)[2]["following_response"]
    assert resumed["applied"]
    assert resumed["response_started_s"] == original["response_started_s"]
    assert resumed["response_deadline_s"] == original["response_deadline_s"]
    assert resumed["reaction_remaining_s"] == 0.


def test_unknown_connection_does_not_extend_the_original_finite_response_deadline():
    class UnknownLeaderRoad(Road):
        def get_waypoint(self, location, **kwargs):
            return None if abs(location.x) < 1e-10 else super().get_waypoint(location, **kwargs)

    pair, outputs = established()
    follower, leader, history = pair
    original = outputs[-1][2]["following_response"]
    for stamp in np.arange(1.6, 3.61, .2):
        old = history.last_time_s
        follower.x += follower.speed*(stamp-old)
        history.update(actors=[follower, leader], sim_time_s=float(stamp))
        response = predict_actor_motion(follower, history.motion_for(follower),
            np.arange(21)*.2, UnknownLeaderRoad(), history=history,
            leader_actors=(leader,))[2]["following_response"]
        assert not response["applied"]
        assert response["response_deadline_s"] == original["response_deadline_s"]
    restored = predict_actor_motion(follower, history.motion_for(follower),
        np.arange(21)*.2, Road(), history=history,
        leader_actors=(leader,))[2]["following_response"]
    assert not restored["applied"]
    # Finite observation retention may discard the earliest relation evidence
    # before this delayed recheck; neither path can manufacture a fresh proof.
    assert restored["response_phase_reason"] in (
        "observed_response_expired", "historical_leader_relation_unavailable")
    assert restored["response_proof_time_s"] is None
