"""Recorded R10 queue: real past motion, static topology and every participant."""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from carla_bridge.planning.actor_motion import predict_actor_motion
from carla_bridge.planning.mpc_solver import (
    MPCConfig, MPCInitialState, MPCObstacle, MPCReference, MPCProblem, _Evaluation,
)
from carla_bridge.tests.test_actor_motion_frozen import frozen_actors


CASES = json.loads((Path(__file__).parent/"fixtures/actor_motion_r10_response_v3.json").read_text(encoding="utf-8"))


class TraceWaypoint:
    def __init__(self, graph, ident):
        self.graph, self.node = graph, graph[ident]
        node = self.node
        self.transform = NS(location=NS(x=node["x"], y=node["y"], z=node["z"]),
            rotation=NS(yaw=node["yaw_deg"]))
        for name in ("lane_type", "lane_width", "road_id", "section_id", "lane_id", "s"):
            setattr(self, name, node[name])

    def next(self, distance):
        # These are the exact 0.9.15 OpenDRIVE successor queries captured in
        # this frozen measurement. No inferred branch or future simulator pose.
        keys = self.node.get("next", {})
        key = min(keys, key=lambda key:abs(float(key)-distance))
        assert float(key) == pytest.approx(distance, abs=1e-6)
        return [TraceWaypoint(self.graph, ident) for ident in keys[key]]


class TraceMap:
    def __init__(self, trace):
        self.trace = trace

    def get_waypoint(self, location, **kwargs):
        query = min(self.trace["queries"], key=lambda q:math.hypot(q[0]-location.x, q[1]-location.y))
        assert math.hypot(query[0]-location.x, query[1]-location.y) < 1e-5
        return None if query[3] is None else TraceWaypoint(self.trace["nodes"], query[3])


@pytest.mark.parametrize("case", CASES, ids=lambda case:case["case"])
@pytest.mark.parametrize("horizon", [.6, .8, 1.2])
def test_recorded_response_phase_and_physical_bodies_preserve_complete_safe_hold(case, horizon):
    history, actors = frozen_actors(case)
    by_id = {actor.id:actor for actor in actors}
    forecasts = dict(case["frozen_forecasts"])
    cfg = MPCConfig(**case["config"])
    times = np.arange(cfg.horizon_steps+1)*cfg.dt
    for ident in (215, 210):
        actor = by_id[ident]
        centers, yaws, audit = predict_actor_motion(actor, history.motion_for(actor), times,
            TraceMap(case["map_trace"]), history=history, leader_actors=actors)
        np.testing.assert_allclose(centers, forecasts[str(ident)]["centers_world_xy_m"], atol=1e-6)
        np.testing.assert_allclose(audit["uncertainty_margin_m"], forecasts[str(ident)]["uncertainty_margin_m"])
        forecasts[str(ident)] = audit
    response = forecasts["215"]["following_response"]
    if case["time_s"] == 80.5:
        motion = history.motion_for(by_id[215])
        assert motion.speed_mps == pytest.approx(1.199447, abs=1e-5)
        assert motion.audit["api_integrated_displacement_consistent"]
    else:
        assert response["applied"]
        assert response["position_clamp_used"] is False
    if case["time_s"] >= 81.7:
        assert response["reaction_remaining_s"] == pytest.approx(0., abs=1e-7)
        assert response["response_started_s"] == pytest.approx(81.3)
    count = round(horizon/cfg.dt)+1
    local = replace(cfg, horizon_steps=count-1)
    physical, legacy = [], []
    for ident in case["npc_ids"]:
        actor, forecast = by_id[ident], forecasts[str(ident)]
        margin = np.asarray(forecast["uncertainty_margin_m"])[:count]
        extent = actor.bounding_box.extent
        vehicle = actor.type_id.startswith("vehicle.")
        axes = (np.tile(np.sqrt(2)*np.array([extent.x, extent.y]), (count, 1))+margin[:, None]
            if not actor.type_id.startswith("walker.") else math.hypot(extent.x, extent.y)+margin)
        common = (str(ident), np.asarray(forecast["centers_world_xy_m"])[:count], axes,
            np.asarray(forecast["body_yaws_rad"])[:count])
        legacy.append(MPCObstacle(*common))
        physical.append(MPCObstacle(*common, rectangle_half_extents=[extent.x, extent.y] if vehicle else None,
            uncertainty_m=margin if vehicle else 0.))
    assert {int(ob.id) for ob in physical} == set(case["npc_ids"])
    assert len(physical) == (28 if case["time_s"] == 80.5 else 27)
    initial = MPCInitialState(**case["initial"])
    reference = MPCReference(**dict(case["reference"], points=np.asarray(case["reference"]["points"])[:count]))
    previous = np.array([initial.acceleration, initial.steering])
    controls = []
    for _ in range(count-1):
        rate = np.array([cfg.jerk_max, cfg.steer_rate_max])*cfg.dt
        current = np.clip([0., 0.], previous-rate, previous+rate)
        controls.append(current)
        previous = current
    controls = np.asarray(controls)
    assert np.max(controls[:, 0]) <= 0.
    evaluation = _Evaluation(local, MPCProblem(initial, reference, physical))
    evaluation.evaluate(controls.ravel())
    assert np.min(evaluation.constraints) >= -local.constraint_tolerance, dict(zip(evaluation.constraint_names, evaluation.constraints))
    old = _Evaluation(local, MPCProblem(initial, reference, legacy))
    old.evaluate(controls.ravel())
    if case["time_s"] >= 81.5:
        assert min(value for name, value in zip(old.constraint_names, old.constraints) if name == "obstacle:215") < -.05
