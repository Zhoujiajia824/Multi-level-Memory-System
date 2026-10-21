"""Curved-road following uses directed topology and local, measured geometry."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from carla_bridge.planning.actor_motion import (ActorMotionHistory,
    _observed_path_relation, predict_actor_motion)


class CurveWaypoint:
    def __init__(self, station, lane=1, branch=False):
        self.s, self.lane_id, self.branch = float(station), lane, branch
        self.road_id, self.section_id = 42, 0
        self.lane_type, self.lane_width = "Driving", 3.5
        angle, radius = self.s/20., 20.+3.5*(lane-1)
        self.transform = NS(location=NS(x=radius*math.sin(angle),
            y=20.-radius*math.cos(angle), z=.2), rotation=NS(yaw=math.degrees(angle)))

    def next(self, distance):
        answer = CurveWaypoint(self.s+distance, self.lane_id, self.branch)
        if self.branch and self.s < 12. <= self.s+distance:
            return [answer, CurveWaypoint(self.s+distance, self.lane_id+1)]
        return [answer]


class CurveRoad:
    def __init__(self, branch=False): self.branch = branch

    def get_waypoint(self, location, **kwargs):
        radial = math.hypot(location.x, 20.-location.y)
        lane = 2 if radial > 21.75 else 1
        return CurveWaypoint(20.*math.atan2(location.x,20.-location.y), lane, self.branch)


class CurveActor:
    def __init__(self, ident, station, speed=0., lateral=0.):
        self.id, self.station, self.speed, self.lateral = ident, station, speed, lateral
        self.type_id, self.heading_error = "vehicle.curved", 0.
        self.bounding_box = NS(location=NS(x=0.,y=0.), rotation=NS(yaw=0.),
            extent=NS(x=2.4,y=1.))

    def get_transform(self):
        angle, radius = self.station/20., 20.-self.lateral
        return NS(location=NS(x=radius*math.sin(angle), y=20.-radius*math.cos(angle),z=.2),
            rotation=NS(yaw=math.degrees(angle+self.heading_error)))

    def get_velocity(self):
        angle=self.station/20.
        return NS(x=self.speed*math.cos(angle), y=self.speed*math.sin(angle),z=0.)


def pair(*, cut_in=False):
    follower, leader, history = CurveActor(183,10.,2.2), CurveActor(138,16.), ActorMotionHistory()
    for now in np.arange(8)*.2:
        remaining=1.4-now
        follower.station=10.-2.2*remaining-remaining**2
        follower.speed=2.2+2.*remaining
        leader.lateral=-3.5 if cut_in and now<1.4-1e-8 else 0.
        history.update(actors=[follower,leader],sim_time_s=float(now))
    return follower,leader,history


def forecast(follower, leader, history, road=None, times=None, estimate=None):
    return predict_actor_motion(follower, estimate or history.motion_for(follower),
        np.arange(21)*.2 if times is None else times, road or CurveRoad(),
        history=history,leader_actors=(leader,))


def relation(follower, leader, road):
    history=ActorMotionHistory(); history.update(actors=[follower,leader],sim_time_s=0.)
    first, last = follower.get_transform(), leader.get_transform()
    return _observed_path_relation(history,follower,leader,0.,
        np.array([first.location.x,first.location.y,.2]),math.radians(first.rotation.yaw),
        np.array([last.location.x,last.location.y,.2]),math.radians(last.rotation.yaw),
        road,history.config)


def test_measured_braking_is_recognized_on_unique_curve_without_a_common_tangent():
    follower,leader,history=pair()
    first,last=follower.get_transform(),leader.get_transform()
    delta=np.array([last.location.x-first.location.x,last.location.y-first.location.y])
    angle=follower.station/20.
    assert abs(delta@np.array([-math.sin(angle),math.cos(angle)])) > .6
    result=forecast(follower,leader,history)
    response=result[2]["following_response"]
    assert response["applied"]
    assert response["relation_geometry"] == "unique_connected_road_station_and_local_normals"
    assert response["reaction_remaining_s"] == 0.
    assert response["response_proof_time_s"] < history.last_time_s
    state=deepcopy(history._following_states[follower.id])
    for count in (17,21,17):
        repeated=forecast(follower,leader,history,times=np.linspace(0.,4.,count))[2]["following_response"]
        assert repeated["response_proof_time_s"] == response["response_proof_time_s"]
        assert repeated["response_started_s"] == response["response_started_s"]
        assert history._following_states[follower.id] == state


@pytest.mark.parametrize("case", ["parallel_curve","crossing","branch","behind","near_behind"])
def test_connected_curve_proof_does_not_admit_other_lane_crossing_branch_or_wrong_direction(case):
    follower,leader=CurveActor(183,10.),CurveActor(138,16.)
    road=CurveRoad(branch=case=="branch")
    if case=="parallel_curve": leader.lateral=-3.5
    if case=="crossing": leader.heading_error=math.pi/2
    if case=="behind": leader.station=8.
    if case=="near_behind": leader.station=9.8
    assert not relation(follower,leader,road)


def test_recent_cut_in_cannot_inherit_old_braking_even_on_a_connected_curve():
    follower,leader,history=pair(cut_in=True)
    result=forecast(follower,leader,history)
    assert not result[2]["following_response"]["applied"]
    assert history._following_states.get(follower.id) is None


def test_unproven_current_lateral_forecast_does_not_apply_or_renew_a_curve_response():
    follower,leader,history=pair()
    follower.lateral=-.48
    # Rebuild measured history at this unchanged lane-normal offset.
    history=ActorMotionHistory()
    for now in np.arange(8)*.2:
        remaining=1.4-now
        follower.station=10.-2.2*remaining-remaining**2
        follower.speed=2.2+2.*remaining
        history.update(actors=[follower,leader],sim_time_s=float(now))
    normal=forecast(follower,leader,history)[2]["following_response"]
    assert normal["applied"]
    state=deepcopy(history._following_states[follower.id])
    motion=history.motion_for(follower);angle=follower.station/20.
    lateral=replace(motion,vx=motion.vx+.15*math.sin(angle),vy=motion.vy-.15*math.cos(angle))
    disabled=forecast(follower,leader,history,estimate=lateral)[2]["following_response"]
    assert not disabled["applied"]
    assert disabled["reason"] == "current_prediction_unproven_observed_relation_preserved"
    assert history._following_states[follower.id] == state
    resumed=forecast(follower,leader,history)[2]["following_response"]
    assert resumed["applied"] and resumed["response_deadline_s"] == normal["response_deadline_s"]


def test_recorded_town05_curve_replays_measurements_in_order_and_preserves_real_braking():
    from carla_bridge.tests.test_actor_motion import Actor
    from carla_bridge.tests.test_actor_response_frozen import TraceMap

    case=json.loads((Path(__file__).parent/"fixtures/actor_motion_r05_curve_v4.json").read_text(encoding="utf8"))
    road=TraceMap(case["map_trace"]);history=ActorMotionHistory();actors={};audits={}
    for frame in case["observed_frames"]:
        present=[]
        for raw in frame["actors"]:
            actor=actors.setdefault(raw["id"],Actor(actor_id=raw["id"],kind=raw["type"]))
            actor.x,actor.y,actor.yaw=raw["x"],raw["y"],raw["heading"]
            actor.vx,actor.vy=raw["vx"],raw["vy"]
            actor.bounding_box=NS(location=NS(x=0.,y=0.),rotation=NS(yaw=0.),
                extent=NS(x=raw["length"]/2,y=raw["width"]/2))
            present.append(actor)
        now=frame["time_s"];history.update(actors=present,sim_time_s=now)
        assert set(history._estimates)=={actor.id for actor in present}
        actor=actors[183]
        predict_actor_motion(actor,history.motion_for(actor),np.arange(17)*.25,
            road,history=history,leader_actors=present)
        centers,_,audit=predict_actor_motion(actor,history.motion_for(actor),np.arange(21)*.2,
            road,history=history,leader_actors=present)
        expected=frame["expected_actor183_prediction"]
        np.testing.assert_allclose(centers,expected["centers_world_xy_m"],atol=1e-6)
        for key in ("applied","reason","response_proof_time_s","response_started_s","response_deadline_s"):
            assert audit["following_response"].get(key)==expected["following_response"].get(key)
        audits[round(now,1)]=audit["following_response"]
    assert audits[107.7]["applied"]
    assert audits[107.7]["response_proof_time_s"]==pytest.approx(107.7)
    assert not audits[108.7]["applied"]
    assert audits[108.7]["reason"]=="current_prediction_unproven_observed_relation_preserved"
    assert audits[108.7]["response_proof_time_s"]==audits[108.5]["response_proof_time_s"]
    assert audits[108.9]["applied"]
    assert audits[109.5]["applied"]
    assert audits[109.5]["reaction_remaining_s"]==0.
    assert audits[109.5]["response_started_s"]==pytest.approx(107.7)
