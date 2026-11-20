"""Real roundabout map and audited stop-state regression; independent headless instance.

No simulator stepping, VLM, DINO or user experiment files are used. SQP is run
once on a clean-road fixture at the stop position audited in seed 2001.
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
from nine_grid.navigation_connections import confirmed_navigation_connections
from nine_grid.object_scope import select_scope_objects
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


if __name__ == '__main__':
    unittest.main()
