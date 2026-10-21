"""Physical rectangle clearance, uncertainty, body transform and short recovery."""
from dataclasses import replace
import math

import carla
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from carla_bridge.planning.mpc_solver import (MPCConfig, MPCInitialState, MPCObstacle,
    MPCProblem, MPCSolver, _Evaluation, MPC_COLLISION_GEOMETRY_VERSION)
from carla_bridge.tests.test_mpc_solver import straight
from carla_bridge.tests.test_mpc_planner import fixture
from carla_bridge.tests.test_mpc_braking_recovery import crossing_problem
from carla_bridge.tests.test_pdm_planner import Vehicle


def constraint_minimum(problem, cfg, controls=None):
    evaluation = _Evaluation(cfg, problem)
    evaluation.evaluate(np.zeros((cfg.horizon_steps,2)).ravel() if controls is None else controls.ravel())
    minima = {}
    for name, value in zip(evaluation.constraint_names,evaluation.constraints):
        minima[name] = min(minima.get(name,float("inf")),float(value))
    return minima


def test_true_rectangles_preserve_a_clear_queue_gap_rejected_by_legacy_outer_ellipses():
    cfg = MPCConfig(horizon_steps=3)
    problem = straight(cfg,target=0.)
    centers = np.tile([-5.,0.],(4,1))
    half = np.array([2.4,1.])
    legacy = MPCObstacle("rear",centers,np.tile(np.sqrt(2)*half,(4,1)),0.)
    physical = replace(legacy,rectangle_half_extents=half,uncertainty_m=.1)
    legacy_problem = replace(problem,obstacles=(legacy,))
    physical_problem = replace(problem,obstacles=(physical,))
    assert constraint_minimum(legacy_problem,cfg)["obstacle:rear"] < 0.
    # Ego rear bumper=-1, other front bumper=-2.6: exactly 1.6m raw clearance.
    assert constraint_minimum(physical_problem,cfg)["obstacle:rear"] == pytest.approx(1.3)
    result = MPCSolver(cfg).solve(physical_problem)
    assert result.feasible, result.diagnostics
    assert result.diagnostics["obstacle_shape_counts"]["physical_rectangles"] == 1
    assert result.diagnostics["collision_geometry_version"] == MPC_COLLISION_GEOMETRY_VERSION


@pytest.mark.parametrize("clearance,uncertainty,expected", [(.25,.1,-.05),(.4,.1,.1),(-.15,0.,-.35)])
def test_physical_rectangle_does_not_remove_collision_margin_or_prediction_uncertainty(clearance,uncertainty,expected):
    cfg = MPCConfig(horizon_steps=3)
    problem = straight(cfg,target=0.)
    center_x = cfg.rear_axle_to_center+cfg.ego_length/2+2.+clearance
    obstacle = MPCObstacle("front",np.tile([center_x,0.],(4,1)),np.tile([2.,1.],(4,1)),0.,
                           rectangle_half_extents=[2.,1.],uncertainty_m=uncertainty)
    assert constraint_minimum(replace(problem,obstacles=(obstacle,)),cfg)["obstacle:front"] == pytest.approx(expected)


@pytest.mark.parametrize("shape", ["rectangle","circle","legacy_ellipse"])
def test_body_offset_yaw_collision_and_corridor_gradients_match_full_rollout_differences(shape):
    cfg = MPCConfig(horizon_steps=5,rear_axle_to_center=1.1,
                    ego_body_center_lateral=.18,ego_body_yaw=.12)
    problem = straight(cfg,initial=MPCInitialState(.1,-.1,.07,2.,.02,.1),target=4.,width=3.)
    centers = np.tile([10.,3.],(6,1))
    if shape == "rectangle":
        obstacle = MPCObstacle("object",centers,np.tile([2.3,.9],(6,1)),-.3,
            rectangle_half_extents=[2.3,.9],uncertainty_m=np.linspace(.05,.15,6))
    elif shape == "circle":
        obstacle = MPCObstacle("object",centers,.35)
    else:
        obstacle = MPCObstacle("object",centers,np.tile([3.,1.2],(6,1)),-.3)
    evaluation = _Evaluation(cfg,replace(problem,obstacles=(obstacle,)))
    controls = np.c_[np.linspace(.2,.4,5),np.linspace(.03,.05,5)].ravel()
    np.testing.assert_allclose(evaluation.inequalities_jac(controls),
        approx_derivative(evaluation.inequalities,controls),rtol=7e-5,atol=3e-6)


def test_crossing_rectangle_between_prediction_endpoints_is_still_rejected():
    cfg = MPCConfig(horizon_steps=2,max_iterations=12,multi_start=False)
    problem = straight(cfg,target=0.)
    obstacle = MPCObstacle("crossing",np.array([[1.4,-4.],[1.4,4.],[1.4,12.]]),
        np.tile([.5,.4],(3,1)),0.,rectangle_half_extents=[.5,.4],uncertainty_m=.1)
    result = MPCSolver(cfg).solve(replace(problem,obstacles=(obstacle,)))
    assert not result.feasible
    assert result.diagnostics["constraint_minimums"]["obstacle:crossing"] < -.5
    assert result.diagnostics["collision_sample_dt_s"] == .1


@pytest.mark.parametrize("half,uncertainty", [([0.,1.],0.),([1.,float("nan")],0.),
    ([[1.,1.]],0.),([1.,1.],-.01),([1.,1.],[0.,0.])])
def test_physical_shape_metadata_rejects_missing_extent_or_invalid_uncertainty(half,uncertainty):
    cfg = MPCConfig(horizon_steps=3)
    problem = straight(cfg)
    obstacle = MPCObstacle("invalid",np.tile([12.,2.],(4,1)),1.,
        rectangle_half_extents=half,uncertainty_m=uncertainty)
    with pytest.raises(ValueError,match="rectangle_half_extents|uncertainty_m"):
        MPCSolver(cfg).solve(replace(problem,obstacles=(obstacle,)))


def test_carla_adapter_exports_true_bbox_offset_yaw_and_separate_uncertainty():
    other = Vehicle(actor_id=7,x=20.,y=1.,yaw=30.,speed=0.)
    other.bounding_box = carla.BoundingBox(carla.Location(x=.3,y=.2,z=.5),carla.Vector3D(x=2.2,y=.9,z=.5))
    other.bounding_box.rotation = carla.Rotation(yaw=17.)
    planner = fixture(other)
    planner.ego.bounding_box.location.y = .17
    planner.ego.bounding_box.rotation = carla.Rotation(yaw=11.)
    planner._vehicle_model()
    assert planner.ego_body_center_lateral == pytest.approx(.17,abs=1e-6)
    assert planner.ego_body_yaw == pytest.approx(math.radians(11.))
    signature = planner.execution_vehicle_signature()
    assert signature["body_center_lateral_m"] == .17
    assert signature["body_yaw_rad"] == round(math.radians(11.),6)
    initial, tf = planner._initial_state(0.)
    obstacles,audit = planner._obstacles(initial,tf)
    obstacle = next(ob for ob in obstacles if ob.id=="7")
    np.testing.assert_allclose(obstacle.rectangle_half_extents,[2.2,.9],atol=1e-6)
    np.testing.assert_allclose(obstacle.yaws,math.radians(47.),atol=1e-6)
    expected = np.array([20.,1.])+np.array([[math.cos(math.pi/6),-math.sin(math.pi/6)],
                                          [math.sin(math.pi/6),math.cos(math.pi/6)]])@np.array([.3,.2])
    np.testing.assert_allclose(obstacle.centers[0],expected,atol=1e-6)
    np.testing.assert_allclose(obstacle.uncertainty_m,audit[0]["prediction"]["uncertainty_margin_m"])


def test_short_recovery_retains_sampled_rectangles_and_their_uncertainty():
    planner = crossing_problem()
    times = np.arange(planner.samples+1)*planner.dt
    half = np.tile([.8,.7],(planner.samples+1,1))
    uncertainty = np.linspace(.05,.25,planner.samples+1)
    obstacle = MPCObstacle("rectangle",np.c_[np.full_like(times,.2),4.*(times-3.)],half,0.,
                           rectangle_half_extents=half,uncertainty_m=uncertainty)
    planner._obstacles = lambda initial,tf: ([obstacle],[{"id":"rectangle"}])
    plan = planner.plan(None,False,sim_time_s=0.)
    d = plan["planner_diagnostics"]
    assert d["degraded_braking"]["active"],d
    selected = next(candidate for candidate in d["candidates"] if candidate["id"]==d["selected_id"])
    assert selected["solver"]["obstacle_shape_counts"] == {"physical_rectangles":1,"legacy_shapes":0}
    horizon = d["degraded_braking"]["verified_horizon_s"]
    last_center = np.array([.2,4.*(horizon-3.)])
    # Closest rectangle remains directly lateral to the stopped vehicle.
    expected = abs(last_center[1])-.7-planner.numerical_config.ego_width/2
    expected -= planner.numerical_config.collision_margin+float(np.interp(horizon,times,uncertainty))
    assert selected["solver"]["constraint_minimums"]["obstacle:rectangle"] == pytest.approx(expected,abs=1e-6)


def test_collision_geometry_version_partitions_only_mpc_experiences(monkeypatch):
    from carla_bridge.planning import mpc_solver
    from carla_bridge.planning.experience_memory import execution_planner_version
    mpc = execution_planner_version({"backend":"mpc"})
    pdm = execution_planner_version({"backend":"pdm_closed"})
    monkeypatch.setattr(mpc_solver,"MPC_COLLISION_GEOMETRY_VERSION","different_geometry")
    assert execution_planner_version({"backend":"mpc"}) != mpc
    assert execution_planner_version({"backend":"pdm_closed"}) == pdm
