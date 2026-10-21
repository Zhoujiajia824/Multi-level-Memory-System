"""Past-only motion quality and connected-road prediction, without CARLA RPC."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from carla_bridge.planning.actor_motion import (ActorMotionHistory,
    predict_actor_motion, validate_motion_config)


class Actor:
    def __init__(self, actor_id=7, kind="vehicle.test", x=0., y=0., yaw=0., vx=0., vy=0.):
        self.id, self.type_id = actor_id, kind
        self.x, self.y, self.z, self.yaw, self.vx, self.vy = x, y, .2, yaw, vx, vy
        self.bounding_box = NS(location=NS(x=.2, y=0.), extent=NS(x=2.4, y=1.), rotation=NS(yaw=0.))

    def get_transform(self):
        return NS(location=NS(x=self.x, y=self.y, z=self.z), rotation=NS(yaw=math.degrees(self.yaw)))

    def get_velocity(self):
        return NS(x=self.vx, y=self.vy, z=0.)


def observe(history, actor, time):
    history.update(actors=[actor], sim_time_s=time)
    return history.motion_for(actor)


def test_walker_zero_api_velocity_is_replaced_by_observed_motion():
    actor, history = Actor(kind="walker.pedestrian.test"), ActorMotionHistory()
    for time in (0., .2, .4, .6):
        actor.x, actor.y = .7*time, -.3*time
        current = observe(history, actor, time)
    assert current.vx == pytest.approx(.7)
    assert current.vy == pytest.approx(-.3)
    assert current.source == "position_history_corrected_api"
    assert current.quality == "verified"
    assert current.velocity.x == pytest.approx(.7)
    assert current.audit["api_velocity_mps"] == [0., 0., 0.]
    centers, _, forecast = predict_actor_motion(actor, current, [0., 1., 2.])
    assert centers[-1]-centers[0] == pytest.approx([1.4, -.6])
    assert forecast["mode"] == "constant_velocity"


def test_consistent_api_and_positions_are_fused_in_carla_coordinates():
    actor, history = Actor(vx=4., vy=-2.), ActorMotionHistory()
    for time in (0., .2, .4):
        actor.x, actor.y = 4*time, -2*time
        motion = observe(history, actor, time)
    assert (motion.vx, motion.vy) == pytest.approx((4., -2.))
    assert motion.source == "history_validated_api_velocity"


@pytest.mark.parametrize("acceleration,initial_speed", [(3.,0.),(-3.,3.)])
def test_verified_instantaneous_vehicle_velocity_does_not_lag_acceleration_or_braking(acceleration,initial_speed):
    actor, history = Actor(), ActorMotionHistory()
    for time in (0., .2, .4, .6):
        actor.x = initial_speed*time+.5*acceleration*time*time
        actor.vx = initial_speed+acceleration*time
        current = observe(history,actor,time)
    assert current.source == "history_validated_api_velocity"
    assert current.vx == pytest.approx(actor.vx)


def test_repeated_timestamp_is_idempotent_and_does_not_read_changed_actor_state():
    actor, history = Actor(kind="walker.pedestrian.test"), ActorMotionHistory()
    observe(history, actor, 0.)
    actor.x = .2
    first = observe(history, actor, .2)
    actor.x = 100.
    second = observe(history, actor, .2)
    assert second is first
    assert history.last_update_audit["status"] == "repeated_timestamp_reused"
    assert second.audit["history_samples"] == 2


@pytest.mark.parametrize("next_time,status", [(-.1, "clock_rewind_reset"), (2., "sampling_gap_reset")])
def test_clock_rewind_or_missing_samples_reset_old_trend(next_time, status):
    actor, history = Actor(), ActorMotionHistory()
    observe(history, actor, 0.)
    actor.x = 1.
    observe(history, actor, .2)
    actor.x, actor.vx = 2., 3.
    motion = observe(history, actor, next_time)
    assert history.last_update_audit["status"] == status
    assert motion.audit["history_samples"] == 1
    assert motion.vx == 3.
    assert motion.yaw_rate_rad_s == 0.


def test_teleport_is_not_reported_as_real_speed_or_turn():
    actor, history = Actor(vx=2.), ActorMotionHistory()
    observe(history, actor, 0.)
    actor.x, actor.yaw = 100., 1.
    motion = observe(history, actor, .2)
    assert motion.audit["event"] == "teleport_or_invalid_interval_reset"
    assert motion.vx == 2. and motion.yaw_rate_rad_s == 0.
    assert motion.audit["history_samples"] == 1


def test_actor_disappearance_and_type_reuse_do_not_inherit_history():
    actor, history = Actor(), ActorMotionHistory()
    observe(history, actor, 0.)
    actor.x = 1.
    observe(history, actor, .2)
    history.update(actors=[], sim_time_s=.4)
    assert history.last_update_audit["removed_actor_count"] == 1
    actor.type_id = "walker.pedestrian.other"
    actor.x = 20.
    current = observe(history, actor, .6)
    assert current.audit["history_samples"] == 1
    assert current.vx == 0.


def test_subcentimeter_stopped_actor_jitter_does_not_create_motion():
    actor, history = Actor(), ActorMotionHistory()
    for time, x in ((0., 0.), (.1, .008), (.2, -.005), (.3, .004)):
        actor.x = x
        current = observe(history, actor, time)
    assert current.vx == current.vy == current.yaw_rate_rad_s == 0.
    assert current.source == "history_verified_stationary"


def test_yaw_wrap_is_unwrapped_and_observed_turn_is_not_straight_forecast():
    actor, history = Actor(vx=-5.), ActorMotionHistory()
    start, speed, rate = math.radians(176.), 5., .3
    for time in (0., .2, .4, .6):
        angle = start + rate*time
        actor.x = speed/rate*(math.sin(angle)-math.sin(start))
        actor.y = speed/rate*(-math.cos(angle)+math.cos(start))
        actor.yaw = (angle+math.pi)%(2*math.pi)-math.pi
        actor.vx, actor.vy = speed*math.cos(angle), speed*math.sin(angle)
        current = observe(history, actor, time)
    assert current.yaw_rate_rad_s == pytest.approx(rate)
    centers, yaws, audit = predict_actor_motion(actor, current, [0., 2., 4.])
    assert audit["mode"] == "observed_turn_rate"
    assert yaws[-1]-yaws[0] == pytest.approx(1.2)
    straight = centers[0] + 4*np.array([current.vx, current.vy])
    assert np.linalg.norm(centers[-1]-straight) > 10.


def test_bbox_center_rotates_with_predicted_vehicle_heading():
    actor, history = Actor(vx=4.), ActorMotionHistory()
    actor.bounding_box.location = NS(x=2., y=1.)
    for time in (0., .2, .4):
        actor.x = 4*time
        actor.yaw = time*.2
        motion = observe(history, actor, time)
    centers, yaws, audit = predict_actor_motion(actor, motion, [0., 1.])
    expected0 = np.array([actor.x, actor.y]) + np.array([
        2*math.cos(actor.yaw)-math.sin(actor.yaw), 2*math.sin(actor.yaw)+math.cos(actor.yaw)])
    assert centers[0] == pytest.approx(expected0)
    assert yaws[-1]-yaws[0] == pytest.approx(.2)
    assert not audit["future_ground_truth_used"]


class Waypoint:
    lane_type = "Driving"
    lane_width = 3.5
    def __init__(self, station, curve=0., *, branch=False, road=1):
        self.station, self.curve, self.branch, self.road_id = station, curve, branch, road
        self.lane_id, self.section_id, self.is_junction = -1, 0, bool(curve)
        yaw = station*curve
        x, y = ((station, 0.) if curve == 0 else (math.sin(yaw)/curve, (1-math.cos(yaw))/curve))
        self.transform = NS(location=NS(x=x, y=y, z=.2), rotation=NS(yaw=math.degrees(yaw)))

    def next(self, distance):
        if self.branch and self.station >= 1.:
            # Equal left/right hypotheses cannot be resolved from straight past.
            left, right = Waypoint(self.station+distance), Waypoint(self.station+distance)
            left.transform.rotation.yaw, right.transform.rotation.yaw = 5., -5.
            left.road_id, right.road_id = 2, 3
            return [left, right]
        return [Waypoint(self.station+distance, self.curve, branch=self.branch)]


def test_map_prediction_follows_only_explicit_connected_curve_successors():
    actor, history = Actor(vx=5.), ActorMotionHistory()
    motion = observe(history, actor, 0.)
    calls = []
    road = NS(get_waypoint=lambda location, **kwargs: calls.append(location) or Waypoint(0., .04))
    centers, yaws, audit = predict_actor_motion(actor, motion, np.arange(21)*.2, road)
    assert len(calls) == 1, "Future positions must not use nearest-lane snapping"
    assert audit["mode"] == "connected_lane"
    assert yaws[-1] == pytest.approx(.8)
    assert centers[-1, 1] > 7.
    assert audit["connected_samples"] == 21


def test_ambiguous_map_branch_uses_observed_motion_instead_of_arbitrary_turn():
    actor, history = Actor(vx=5.), ActorMotionHistory()
    motion = observe(history, actor, 0.)
    road = NS(get_waypoint=lambda location, **kwargs: Waypoint(0., branch=True))
    centers, yaws, audit = predict_actor_motion(actor, motion, np.arange(21)*.2, road)
    assert audit["fallback_reason"] == "ambiguous_successors"
    assert audit["mode"] == "connected_lane_then_observed_motion"
    assert max(abs(yaws)) == 0.
    assert max(abs(centers[:, 1])) == 0.


def test_crossing_vehicle_is_not_snapped_to_nearby_parallel_driving_lane():
    actor, history = Actor(vy=-5., yaw=-math.pi/2), ActorMotionHistory()
    motion = observe(history, actor, 0.)
    road = NS(get_waypoint=lambda location, **kwargs: Waypoint(0.))
    centers, _, audit = predict_actor_motion(actor, motion, [0., 1., 2.], road)
    assert audit["map_status"] == "current_motion_not_aligned_with_lane"
    assert audit["mode"] == "constant_velocity"
    assert centers[-1, 1]-centers[0, 1] == pytest.approx(-10.)


def test_small_angle_lane_change_preserves_measured_cross_lane_motion():
    actor, history = Actor(vx=6., vy=-.8, yaw=-.12), ActorMotionHistory()
    motion = observe(history, actor, 0.)
    road = NS(get_waypoint=lambda location, **kwargs: Waypoint(0.))
    centers, _, audit = predict_actor_motion(actor, motion, [0., 1., 2., 4.], road)
    assert audit["map_status"] == "observed_lateral_motion_preserved"
    assert audit["mode"] == "constant_velocity"
    assert centers[-1, 1]-centers[0, 1] == pytest.approx(-3.2)


def test_other_level_road_is_not_used_as_motion_guidance():
    actor, history = Actor(vx=5.), ActorMotionHistory()
    motion = observe(history, actor, 0.)
    wrong = Waypoint(0.)
    wrong.transform.location.z = 5.
    road = NS(get_waypoint=lambda location, **kwargs: wrong)
    _, _, audit = predict_actor_motion(actor, motion, [0., 1.], road)
    assert audit["map_status"] == "current_motion_not_aligned_with_lane"


def test_static_props_do_not_accumulate_unobserved_motion_uncertainty():
    actor, history = Actor(kind="static.prop.barrier", vx=5.), ActorMotionHistory()
    motion = observe(history, actor, 0.)
    centers, _, audit = predict_actor_motion(actor, motion, [0., 4.])
    assert np.array_equal(centers[0], centers[-1])
    assert audit["uncertainty_margin_m"] == [0., 0.]


@pytest.mark.parametrize("bad", [{"typo": 1.}, {"map_guidance": 1},
    {"history_window_s": 0.}, {"maximum_gap_s": float("nan")},
    {"minimum_difference_interval_s": .7}, {"map_alignment_rad": 2.}])
def test_motion_config_rejects_ignored_keys_or_unphysical_parameters(bad):
    with pytest.raises(ValueError, match="actor_motion"):
        validate_motion_config(bad)


@pytest.mark.parametrize("times", [[.2, .4], [0., 0.], [0., -1.], [0., float("nan")]])
def test_prediction_times_must_preserve_observation_origin(times):
    actor, history = Actor(), ActorMotionHistory()
    motion = observe(history, actor, 0.)
    with pytest.raises(ValueError, match="prediction times"):
        predict_actor_motion(actor, motion, times)


def test_crossing_pedestrian_corrected_velocity_still_triggers_real_route_risk():
    from carla_bridge.planning.scene_context import constant_velocity_conflict
    actor, history = Actor(kind="walker.pedestrian.test", x=4., y=-6.), ActorMotionHistory()
    for time in (0., .2, .4, .6):
        actor.y = -6.+3*time
        motion = observe(history, actor, time)
    _, _, forecast = predict_actor_motion(actor, motion, np.arange(17)*.25)
    conflict, ttc = constant_velocity_conflict(0., 2., [(0.,0.),(30.,0.)],
        (actor.x,actor.y), (0.,0.), actor_prediction=forecast)
    assert conflict and ttc <= 2.


def test_occupied_body_proxy_uses_each_predicted_body_orientation():
    from carla_bridge.planning.scene_context import occupied_body_conflict
    args = ((0.,0.), (0.,0.), 0., (1.,.5), (0.,2.4), (0.,0.), 0., (2.,.5))
    forecast = {"times_s":[0.,1.,2.], "centers_world_xy_m":[[0.,2.4]]*3,
        "body_yaws_rad":[0.,math.pi/2,math.pi/2], "uncertainty_margin_m":[0.]*3}
    assert occupied_body_conflict(*args) == (False,None)
    assert occupied_body_conflict(*args, actor_prediction=forecast)[0]
