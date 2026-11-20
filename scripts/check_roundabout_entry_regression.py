"""Real roundabout map and audited stop-state regression; independent headless instance.

No simulator stepping, VLM, DINO or user experiment files are used. SQP covers
clean-road stationary fixtures at the audited entry and exit stop positions.
"""
from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import shapely
import yaml

from nine_grid.geometry import project_polyline
from nine_grid.highway import HighwayAdapter
from nine_grid.lane_roles import trace_lane
from nine_grid.navigation_connections import (active_navigation_connection,
                                              confirmed_navigation_connections)
from nine_grid.object_scope import lane_role_geometry, select_scope_objects
from nine_grid.planner import Planner
from nine_grid.potential_domain import build_lane_potential_domain
from nine_grid.prediction import Predictor


class RoundaboutEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config/parameters.json').read_text(encoding='utf-8'))
        cls.config['scenarios'] = yaml.safe_load((ROOT/'config/scenarios.yaml').read_text(encoding='utf-8'))
        cls.config.update(scenario='roundabout', vehicles=0)
        cls.adapter = HighwayAdapter(cls.config, 2001, 'isolated-roundabout-geometry')
        cls.scene = cls.adapter.snapshot()

    @classmethod
    def tearDownClass(cls):
        cls.adapter.close()

    def exit_fixture(self, position=None, heading=None):
        scene = copy.deepcopy(self.scene)
        connection = confirmed_navigation_connections(scene)[('ee|nx|1', 'nx|nxs|0')]
        middle = len(connection['points'])//2
        if position is None:
            position = connection['points'][middle].copy()
        if heading is None:
            heading = float(connection['headings'][middle])
        scene.frame_id = 'audited-exit-fixture:1'
        scene.ego = replace(scene.ego, position=np.asarray(position, float), heading=heading,
            velocity=np.zeros(2), acceleration=np.zeros(2), history=[], lane_id='nx|ne|1')
        scene.actors = []
        anchor = project_polyline(scene.ego.position, scene.route, heading=heading)[0]
        scene.diagnostics.update(route_s=anchor, reference_anchor_s=anchor,
            stable_navigation_lane_id='ee|nx|1', previous_control=[0., -.245])
        return scene

    def test_entry_and_exit_use_the_confirmed_navigation_curves(self):
        s = copy.deepcopy(self.scene)
        links = confirmed_navigation_connections(s)
        self.assertEqual(len(links), 2)
        trace = trace_lane(s, s.ego.lane_id, back_m=20., front_m=500.)
        self.assertTrue(set(s.route_lane_ids).issubset(trace['lane_ids']))
        domain, evidence = build_lane_potential_domain(s, front_m=500.)
        self.assertEqual(len(evidence['navigation_connections']), 2)
        for link in links.values():
            self.assertTrue(shapely.covers(domain, shapely.points(link['points'])).all())
            for point in link['points']:
                self.assertLess(shapely.LineString(trace['points']).distance(shapely.Point(point)), 1e-6)

    def test_ring_object_is_in_range_along_the_connected_lane(self):
        s = copy.deepcopy(self.scene)
        s.ego = replace(s.ego, position=np.array([5.378846638886971, -25.473215087677357]),
                        heading=1.2408712805929454, velocity=np.zeros(2), history=[], lane_id='ses|se|0')
        ring = next(lane for lane in s.lanes if lane.id == 'se|ex|1')
        point = ring.center[len(ring.center)//2].copy()
        actor = replace(s.ego, id='ring_vehicle', position=point, lane_id=ring.id,
                        heading=project_polyline(point, ring.center)[2])
        s.actors = [actor]
        selected, evidence = select_scope_objects(s, self.config)
        self.assertEqual([a.id for a in selected], ['ring_vehicle'])
        ds = evidence['objects']['ring_vehicle']['memberships']['front']['delta_s_m']
        self.assertGreater(ds, 0.)
        self.assertLessEqual(ds, 40.)

    def test_audited_stationary_entry_state_can_plan_forward(self):
        s = copy.deepcopy(self.scene)
        s.frame_id = 'audited-stop-fixture:1'
        s.ego = replace(s.ego, position=np.array([5.378846638886971, -25.473215087677357]),
                        heading=1.2408712805929454, velocity=np.zeros(2),
                        acceleration=np.zeros(2), history=[], lane_id='ses|se|0')
        anchor = project_polyline(s.ego.position, s.route, heading=s.ego.heading)[0]
        s.diagnostics.update(route_s=anchor, reference_anchor_s=anchor,
                             stable_navigation_lane_id=s.ego.lane_id, previous_control=[0., 0.])
        plan = Planner(self.config).solve(s, Predictor(self.config).predict(s))
        self.assertTrue(plan.valid)
        self.assertTrue(plan.diagnostics['safety_feasible'], plan.diagnostics['constraint_minima'])
        self.assertEqual(plan.diagnostics['dense_validation']['outside_potential_domain_samples'], 0)
        self.assertEqual(plan.diagnostics['potential_field']['invalid_samples'], 0)
        self.assertGreater(plan.controls[0, 0], .1)
        self.assertGreater(plan.states[-1, 3], .5)

    def test_exit_curve_and_audited_stop_keep_the_downstream_exit_domain(self):
        for position, heading in ((None, None), ([6.8934759169, 23.7768782683], 2.53)):
            with self.subTest(position=position):
                scene = self.exit_fixture(position, heading)
                active = active_navigation_connection(scene)
                self.assertIsNotNone(active)
                self.assertEqual(active['evidence']['raw_lane_id'], 'nx|ne|1')
                traces, roads, _ = lane_role_geometry(scene)
                self.assertIn('nx|nxs|0', traces['center']['lane_ids'])
                self.assertIn('nxs|nxr|0', traces['center']['lane_ids'])
                self.assertNotIn('nx|ne|1', traces['center']['lane_ids'])
                self.assertEqual(scene.ego.lane_id, 'nx|ne|1')
                self.assertTrue(all(not roads[side]['known'] for side in ('left', 'right')))
                domain, evidence = build_lane_potential_domain(scene)
                exit_lane = next(lane for lane in scene.lanes if lane.id == 'nx|nxs|0')
                self.assertTrue(domain.covers(shapely.Point(exit_lane.center[-1])))
                continuation = next(lane for lane in scene.lanes if lane.id == 'nx|ne|1')
                branch_point = continuation.center[len(continuation.center)*3//4]
                self.assertFalse(domain.covers(shapely.Point(branch_point)))
                self.assertIn({'from_lane_id': 'ee|nx|1', 'to_lane_id': 'nx|nxs|0'},
                              evidence['navigation_connections'])
                for identity, lane, index in (('exit_vehicle', exit_lane, len(exit_lane.center)*2//3),
                        ('other_branch_vehicle', continuation, len(continuation.center)*3//4)):
                    point = lane.center[index].copy()
                    scene.actors.append(replace(scene.ego, id=identity, lane_id=lane.id,
                        position=point, heading=project_polyline(point, lane.center)[2]))
                selected, scope = select_scope_objects(scene, self.config)
                self.assertEqual([actor.id for actor in selected], ['exit_vehicle'])
                self.assertIn('front', scope['objects']['exit_vehicle']['roles'])
                self.assertEqual(scope['current_lane_association']['to_lane_id'], 'nx|nxs|0')

    def test_audited_stationary_exit_state_can_resume(self):
        scene = self.exit_fixture([6.8934759169, 23.7768782683], 2.53)
        plan = Planner(self.config).solve(scene, Predictor(self.config).predict(scene))
        evidence = {key: plan.diagnostics.get(key) for key in (
            'execution_mode', 'optimizer_termination', 'initial_guess_source',
            'accepted_initialization_source', 'candidate_source', 'initial_constraints',
            'infeasible_initial_reselection', 'warm_start_recovery', 'iteration_history',
            'constraint_minima', 'dense_validation', 'local_reference', 'must_wait', 'stop_s')}
        evidence.update(first_controls=plan.controls[:5].tolist(),
                        terminal_state=plan.states[-1].tolist())
        failure = json.dumps(evidence, ensure_ascii=False, default=lambda value: value.tolist())
        self.assertTrue(plan.valid)
        self.assertTrue(plan.diagnostics['safety_feasible'], plan.diagnostics['constraint_minima'])
        self.assertEqual(plan.diagnostics['dense_validation']['outside_potential_domain_samples'], 0)
        self.assertEqual(plan.diagnostics['potential_field']['invalid_samples'], 0)
        self.assertGreater(plan.controls[0, 0], .1, failure)
        self.assertGreater(plan.states[-1, 3], .5, failure)


if __name__ == '__main__':
    unittest.main()
