"""Regress the two server failures with recorded past, never future telemetry.

These are frozen-state optimizer probes, not claims of a full CARLA route pass.
Every recorded collision participant is retained. The reference is the same
local stationary STOP reconstruction used during forensic diagnosis.
"""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from carla_bridge.planning.actor_motion import ActorMotionHistory, predict_actor_motion
from carla_bridge.planning.mpc_solver import (MPCConfig, MPCInitialState,
    MPCObstacle, MPCReference, MPCProblem, MPCSolver, _Evaluation)
from carla_bridge.tests.test_actor_motion import Actor


CASES = json.loads((Path(__file__).parent/"fixtures/actor_motion_server_failures.json").read_text(encoding="utf-8"))


def frozen_actors(case):
    history, actors = ActorMotionHistory(), {}
    for snapshot in case["observed_past"]:
        present = []
        for raw in snapshot["actors"]:
            actor = actors.setdefault(raw["id"], Actor(actor_id=raw["id"], kind=raw["type"]))
            actor.x, actor.y, actor.yaw = raw["x"], raw["y"], raw["heading"]
            actor.vx, actor.vy = raw["vx"], raw["vy"]
            # The source records bbox centers; no origin offset is added twice.
            actor.bounding_box = NS(location=NS(x=0., y=0.), rotation=NS(yaw=0.),
                extent=NS(x=raw["length"]/2, y=raw["width"]/2))
            present.append(actor)
        history.update(actors=present, sim_time_s=snapshot["time_s"])
    return history, present


class RecordedWaypoint:
    """Only the real static OpenDRIVE successor links captured for this probe."""
    def __init__(self, node):
        self.node = node
        self.transform = NS(location=NS(x=node["x"], y=node["y"], z=node["z"]),
                            rotation=NS(yaw=node["yaw_deg"]))
        self.lane_type, self.lane_width = node["lane_type"], node["lane_width"]
        self.road_id, self.lane_id, self.section_id = node["road_id"], node["lane_id"], node["section_id"]

    def next(self, distance):
        assert distance == pytest.approx(self.node["next_distance_m"], abs=1e-5)
        return [RecordedWaypoint(child) for child in self.node["successors"]]


class RecordedMap:
    def __init__(self, case, actors):
        self.nodes = case["connected_waypoint_subtrees"]
        self.actors = actors

    def get_waypoint(self, location, **kwargs):
        actor = min(self.actors, key=lambda actor:math.hypot(actor.x-location.x, actor.y-location.y))
        assert math.hypot(actor.x-location.x, actor.y-location.y) < 1e-5
        node = self.nodes[str(actor.id)]
        return RecordedWaypoint(node) if node is not None else None


@pytest.mark.parametrize("case", CASES, ids=lambda item:item["case"])
def test_recorded_turning_actor_does_not_make_stationary_stop_infeasible(case):
    history, actors = frozen_actors(case)
    carla_map = RecordedMap(case, actors)
    config = MPCConfig(**case["config"])
    times = np.arange(config.horizon_steps+1)*config.dt
    predicted, original_cv = [], []
    by_id = {}
    for actor in actors:
        motion = history.motion_for(actor)
        centers, yaws, audit = predict_actor_motion(actor, motion, times, carla_map)
        margin = np.asarray(audit["uncertainty_margin_m"])
        extent = actor.bounding_box.extent
        if actor.type_id.startswith("walker."):
            axes = math.hypot(extent.x, extent.y)+margin
            raw_axes = math.hypot(extent.x, extent.y)
        else:
            raw_axes = np.tile(np.sqrt(2)*np.array([extent.x, extent.y]), (len(times), 1))
            axes = raw_axes+margin[:, None]
        predicted.append(MPCObstacle(str(actor.id), centers, axes, yaws))
        original_cv.append(MPCObstacle(str(actor.id),
            np.array([actor.x, actor.y])+times[:, None]*np.array([actor.vx, actor.vy]), raw_axes, actor.yaw))
        by_id[actor.id] = (motion, audit)
    initial = MPCInitialState(**case["initial"])
    reference = MPCReference("frozen:STOP", np.asarray(case["reference_points"]), case["half_width"],
        8., {"stop_line":case["stop_line"]})
    new_problem = MPCProblem(initial, reference, predicted)
    old_problem = replace(new_problem, obstacles=original_cv)
    old_eval, new_eval = _Evaluation(config, old_problem), _Evaluation(config, new_problem)
    zeros = np.zeros((config.horizon_steps, 2)).ravel()
    old_eval.evaluate(zeros)
    new_eval.evaluate(zeros)
    binding = "obstacle:"+str(case["binding_actor_id"])
    old_min = min(value for name,value in zip(old_eval.constraint_names,old_eval.constraints) if name == binding)
    new_min = min(value for name,value in zip(new_eval.constraint_names,new_eval.constraints) if name == binding)
    assert old_min < -.2, "The fixture must reproduce the original erroneous CV conflict"
    assert new_min > .01, "A stationary stop must remain outside the observed-turn forecast envelope"
    motion, audit = by_id[case["binding_actor_id"]]
    assert .1 < motion.yaw_rate_rad_s < .4
    assert audit["mode"] == "connected_lane"
    assert audit["map_status"] == "explicit_connected_successors"
    assert not audit["future_ground_truth_used"]
    result = MPCSolver(config).solve(new_problem)
    assert result.feasible and result.converged, result.diagnostics
    assert result.diagnostics["obstacle_count"] == len(actors)


def test_recorded_walker_zero_api_is_corrected_without_using_its_future():
    # PDM R05 walker 1363: the simulator API was zero while the actor approached
    # at 1.14m/s. This synthetic local translation preserves that observed rate.
    history = ActorMotionHistory()
    actor = Actor(kind="walker.pedestrian.recorded", x=10., vx=0.)
    for time in (0., .2, .4, .6):
        actor.x = 10.-1.14*time
        history.update(actors=[actor], sim_time_s=time)
    estimate = history.motion_for(actor)
    assert estimate.vx == pytest.approx(-1.14)
    assert estimate.audit["api_velocity_mps"] == [0.,0.,0.]
    assert estimate.source == "position_history_corrected_api"


@pytest.mark.parametrize("case", CASES, ids=lambda item:item["case"])
def test_semantic_context_uses_same_map_prediction_instead_of_false_cv_trigger(case):
    from carla_bridge.planning.scene_context import constant_velocity_conflict
    history, actors = frozen_actors(case)
    actor = next(actor for actor in actors if actor.id == case["binding_actor_id"])
    motion = history.motion_for(actor)
    _, _, forecast = predict_actor_motion(actor, motion, np.arange(21)*.2, RecordedMap(case, actors))
    initial = case["initial"]
    position = np.array([initial["x"],initial["y"]])
    tangent = np.array([math.cos(initial["yaw"]),math.sin(initial["yaw"])])
    points = [(position-50*tangent).tolist(), (position+50*tangent).tolist()]
    raw = constant_velocity_conflict(50., 0., points, (actor.x,actor.y), (actor.vx,actor.vy))
    corrected = constant_velocity_conflict(50., 0., points, (actor.x,actor.y), (actor.vx,actor.vy),
                                          actor_prediction=forecast)
    assert raw[0], "Original constant-velocity semantic trigger must be reproduced"
    assert corrected == (False, None)


QUEUE_CASES = json.loads((Path(__file__).parent/"fixtures/actor_motion_r10_queue.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", QUEUE_CASES, ids=lambda item:item["case"])
@pytest.mark.parametrize("horizon", [.6, .8, 1.2])
def test_recorded_serial_queue_hold_keeps_every_actor_and_all_original_constraints(case, horizon):
    """The rejected 1.2s hold has a genuine shorter-horizon certificate.

    This test does not grant an unlimited execution horizon. It preserves the
    complete recorded participant set and safety margins, recomputes the two
    relevant rear forecasts from real observed past/static successor links,
    and retains the other past-only forecasts. No braking response is invented
    for either rear car before the recorded evidence exists.
    """
    history, actors = frozen_actors(case)
    by_id = {actor.id:actor for actor in actors}
    assert len(case["npc_ids"]) == (29 if case["time_s"] == 79.7 else 28)
    config = MPCConfig(**case["config"])
    count = round(horizon/config.dt)+1
    local_config = replace(config, horizon_steps=count-1)
    replay_map = RecordedMap(case, actors)
    forecasts = dict(case["frozen_forecasts"])
    for ident in (90, 85):
        actor = by_id[ident]
        centers, headings, audit = predict_actor_motion(actor, history.motion_for(actor),
            np.arange(config.horizon_steps+1)*config.dt, replay_map,
            history=history, leader_actors=actors)
        np.testing.assert_allclose(centers, forecasts[str(ident)]["centers_world_xy_m"], atol=1e-6)
        assert not audit["following_response"]["applied"]
        assert audit["following_response"]["reason"] in (
            "no_measured_recent_braking", "no_established_same_path_leader")
        forecasts[str(ident)] = audit
    obstacles = []
    for ident in case["npc_ids"]:
        actor, prediction = by_id[ident], forecasts[str(ident)]
        margin = np.asarray(prediction["uncertainty_margin_m"])[:count]
        extent = actor.bounding_box.extent
        if actor.type_id.startswith("walker."):
            axes = math.hypot(extent.x, extent.y)+margin
        else:
            axes = np.tile(np.sqrt(2)*np.array([extent.x,extent.y]), (count,1))+margin[:,None]
        obstacles.append(MPCObstacle(str(ident),
            np.asarray(prediction["centers_world_xy_m"])[:count], axes,
            np.asarray(prediction["body_yaws_rad"])[:count]))
    assert {int(obstacle.id) for obstacle in obstacles} == set(case["npc_ids"])
    initial = MPCInitialState(**case["initial"])
    reference = MPCReference(**dict(case["reference"], points=np.asarray(case["reference"]["points"])[:count]))
    evaluation = _Evaluation(local_config, MPCProblem(initial,reference,obstacles))
    previous = np.array([initial.acceleration,initial.steering])
    controls = []
    for _ in range(count-1):
        rate = np.array([config.jerk_max,config.steer_rate_max])*config.dt
        current = np.clip([0.,0.],previous-rate,previous+rate)
        controls.append(current)
        previous = current
    controls = np.asarray(controls)
    assert np.max(controls[:,0]) <= 0., "A hold certificate cannot accelerate to escape the rear forecast"
    evaluation.evaluate(controls.ravel())
    minima = {}
    for name,value in zip(evaluation.constraint_names,evaluation.constraints):
        minima[name] = min(minima.get(name,float("inf")),float(value))
    if horizon <= .8:
        assert min(minima.values()) >= -config.constraint_tolerance, minima
        assert minima["obstacle:90"] > .1
        np.testing.assert_allclose(evaluation.states[:,:2],
            np.tile([initial.x,initial.y],(count,1)),atol=1e-8)
    else:
        assert minima["obstacle:90"] < -.1, "The recorded failing forecast must remain a collision participant"
    assert minima["obstacle:85"] > 1., "The distant rear queue is not the near-term binding actor"
