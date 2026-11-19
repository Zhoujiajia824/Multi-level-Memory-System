"""Planner integration checks: actual SQP, signed objective and lane domain; no simulator/model."""
from pathlib import Path
import json
import os
import sys
import unittest

os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nine_grid.contracts import Actor, Lane, ModelAnswer, PredictionBundle, SceneSnapshot
from nine_grid.configuration import validate_config
from nine_grid.goals import build_goal_reference, validate_local_goal, first_arrival_residual
from nine_grid.lane_potential import LanePotentialGeometry
from nine_grid.planner import Planner
from nine_grid.potential_domain import build_lane_potential_domain


def config():
    return json.loads((ROOT/'config/parameters.json').read_text(encoding='utf-8'))


def scene(obstacle_length=None):
    center = np.array([[-60., 0.], [160., 0.]])
    lane = Lane('own', center, np.array(box(-60, -1.75, 160, 1.75).exterior.coords), 3.5, 15.)
    lane.boundaries = [dict(id='left', points=[[-60,1.75],[160,1.75]],crossable=False),
                       dict(id='right',points=[[-60,-1.75],[160,-1.75]],crossable=False)]
    ego = Actor('ego', 'vehicle', np.array([0.,0.]), np.array([5.,0.]), np.zeros(2), 0., 4., 1.8, 'own')
    actors = [] if obstacle_length is None else [Actor('block', 'static_obstacle',np.array([25.,0.]),
        np.zeros(2),np.zeros(2),0.,obstacle_length,1.8,'own')]
    return SceneSnapshot('integration:1',0.,ego,actors,[lane],center,['own'],Polygon(lane.polygon),
        'integration',rules={'desired_speed':5.,'speed_limit':15.},diagnostics={'lane_map_complete':True})


def prediction(s):
    times=np.arange(31)*.1
    trajectories={a.id:np.tile([*a.position,a.heading,0.],(len(times),1)) for a in s.actors}
    return PredictionBundle(s.frame_id,s.time,times,trajectories,{a.id:a for a in s.actors})


class PlannerPotentialChecks(unittest.TestCase):
    def test_base_keeps_safety_constraints_and_scalar_energy(self):
        for length in (2.,8.):
            with self.subTest(source_length=length):
                s=scene(length);c=config();plan=Planner(c).solve(s,prediction(s))
                self.assertTrue(plan.valid)
                self.assertTrue(plan.diagnostics['safety_feasible'])
                self.assertEqual(plan.diagnostics['potential_field']['invalid_samples'],0)
                self.assertIn('actor_clearance',plan.diagnostics['hard_constraint_groups'])
                # Independent centre-distance formula, evaluated on the accepted actual rollout.
                rho=np.sqrt(((plan.states[1:,0]-25.)/12.)**2+(plan.states[1:,1]/4.)**2)
                t=np.maximum(1.-rho,0.)
                expected=2.*8.*np.mean(t**3*(1.+3.*rho+6.*rho**2))
                self.assertAlmostEqual(plan.diagnostics['cost_terms']['actor_risk'],expected,places=8)
                self.assertAlmostEqual(sum(plan.diagnostics['cost_terms'].values()),plan.diagnostics['objective'],places=8)
                self.assertEqual(plan.diagnostics['dense_validation']['outside_potential_domain_samples'],0)

    def test_goal_negative_shift_is_once_and_goal_lifecycle_unchanged(self):
        s=scene();c=config();ref=build_goal_reference(s,c)
        payload=validate_local_goal(s,ref,dict(delta_s_m=10.,d_m=0.))
        answer=ModelAnswer(s.frame_id,s.time,[],payload['local_goal'],{},payload['world_goal'],ref,
            payload['goal_id'],target_lane_id=payload['target_lane_id'])
        planner=Planner(c);plan=planner.solve(s,prediction(s),answer)
        self.assertTrue(plan.diagnostics['safety_feasible'])
        self.assertEqual(plan.diagnostics['goal_evaluation']['potential_coordinates'],'lane_frenet')
        self.assertEqual(plan.diagnostics['goal_evaluation']['mode'],'first_crossing')
        self.assertAlmostEqual(plan.diagnostics['cost_terms']['goal_attraction'],-6.,places=7)
        self.assertAlmostEqual(sum(plan.diagnostics['cost_terms'].values()),plan.diagnostics['objective'],places=8)
        self.assertEqual(planner.goals.active['goal_id'],answer.goal_id)

    def test_wide_crossroad_polygon_does_not_grant_side_branch(self):
        s=scene()
        s.drivable=shapely.union_all([s.drivable,box(-5,-30,5,30)])
        domain,evidence=build_lane_potential_domain(s)
        self.assertTrue(domain.covers(shapely.Point(4.,0.)))
        self.assertFalse(domain.covers(shapely.Point(0.,15.)))
        self.assertEqual(evidence['lane_ids'],['own'])

    def test_curved_goal_plan_follows_lane_without_cutting_corner(self):
        c=config();s=scene()
        theta=np.linspace(-1.,2.,241)
        center=np.c_[30.*np.sin(theta),30.*(1.-np.cos(theta))]
        shape=LineString(center).buffer(1.75,cap_style=2)
        s.lanes=[Lane('own',center,np.array(shape.exterior.coords),3.5,15.)]
        s.route=center;s.drivable=shape
        ref=build_goal_reference(s,c)
        payload=validate_local_goal(s,ref,dict(delta_s_m=10.,d_m=0.))
        answer=ModelAnswer(s.frame_id,s.time,[],payload['local_goal'],{},payload['world_goal'],ref,
            payload['goal_id'],target_lane_id='own')
        plan=Planner(c).solve(s,prediction(s),answer)
        self.assertTrue(plan.diagnostics['safety_feasible'])
        self.assertGreater(plan.states[-1,2],.1)
        self.assertLess(max(LineString(center).distance(shapely.Point(xy)) for xy in plan.states[:,:2]),.3)

    def test_uncovered_remote_source_retains_collision_constraints(self):
        s=scene();c=config()
        rear=Actor('rear','vehicle',np.array([-15.,0.]),np.array([-25.,0.]),np.zeros(2),np.pi,4.,1.8,'own')
        s.actors=[rear];times=np.arange(31)*.1
        path=np.c_[-15.-25.*times,np.zeros(31),np.full(31,np.pi),np.full(31,25.)]
        pred=PredictionBundle(s.frame_id,s.time,times,{'rear':path},{'rear':rear})
        plan=Planner(c).solve(s,pred)
        self.assertTrue(plan.diagnostics['safety_feasible'])
        self.assertGreater(plan.diagnostics['potential_field']['source_projection_uncovered_samples']['rear'],0)
        self.assertIn('actor_clearance:rear',plan.diagnostics['constraint_minima'])

    def test_invalid_goal_probe_is_rejected_without_exception(self):
        s=scene();geometry=LanePotentialGeometry(s.route,s.drivable)
        residual,jacobian,evidence=first_arrival_residual(np.array([[0.,0.],[8.,9.]]),s.route,80.,
            np.array([20.,0.]),np.eye(2),60.,frenet_geometry=geometry)
        self.assertFalse(evidence['potential_valid'])
        self.assertTrue(np.isfinite(residual).all() and np.isfinite(jacobian).all())

    def test_old_body_distance_apf_config_rejected(self):
        c=config();c['risk']['classes']['vehicle']['influence_m']=20.
        with self.assertRaisesRegex(ValueError,'旧版|未知'):
            validate_config(c)


if __name__=='__main__':
    unittest.main(verbosity=2)
