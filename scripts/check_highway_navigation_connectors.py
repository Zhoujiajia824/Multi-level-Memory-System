"""Navigation connector evidence checks; no simulator stepping, SQP, model or rendering."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nine_grid.geometry import rectangle_corners
from nine_grid.highway import HighwayAdapter


class StraightLane:
    def __init__(self, start, end):
        self.start = np.asarray(start, float)
        self.delta = np.asarray(end, float)-self.start
        self.length = float(np.linalg.norm(self.delta))

    def position(self, s, lateral):
        direction = self.delta/self.length
        return self.start+s*direction+lateral*np.array([-direction[1], direction[0]])


def fixture(right_handed=False):
    lanes = {('a', 'b', 0): StraightLane([0., 0.], [40., 0.]),
             ('b', 'c', 0): StraightLane([45., 4.], [85., 4.])}
    network = SimpleNamespace(graph={'a': {'b': [lanes['a', 'b', 0]]},
                                     'b': {'c': [lanes['b', 'c', 0]]}},
        shortest_path=lambda origin, target: ['b', 'c'], get_lane=lanes.__getitem__)
    adapter = HighwayAdapter.__new__(HighwayAdapter)
    adapter.env = SimpleNamespace(road=SimpleNamespace(network=network),
        vehicle=SimpleNamespace(lane_index=('a', 'b', 0), LENGTH=5., WIDTH=2.))
    adapter.scenario = 'roundabout'
    adapter.settings = {'destination': 'c'}
    adapter.steer_max = .6
    adapter._right_handed = right_handed
    adapter.drivable = box(-10., -12., 95., 12.)
    return adapter


class NavigationConnectors(unittest.TestCase):
    def metadata(self, outside=0, steering=.5, right_handed=False):
        adapter = fixture(right_handed)
        points = np.array([[36., 0.], [40., 1.], [46., 4.]])
        headings = np.array([.1, .2, .3])
        def connect(before, after):
            return np.vstack((before[:1], points)), after[1:], {
                'method': 'road_checked_cubic_hermite', 'outside_samples': outside,
                'maximum_curvature': .1, 'maximum_steering': steering,
                'from_end_trim_m': 4., 'to_start_trim_m': 1.,
                '_points_native': points.copy(), '_headings_native': headings.copy()}
        adapter._connect = connect
        identities, route, diagnostics = adapter._navigation()
        return adapter, identities, route, diagnostics['connections'][0], points, headings

    def test_confirmed_pair_and_same_curve_are_serializable_right_handed_evidence(self):
        import json
        adapter, identities, route, evidence, points, headings = self.metadata()
        self.assertEqual(identities, ['a|b|0', 'b|c|0'])
        self.assertEqual(adapter.env.vehicle.route, [('a', 'b', 0), ('b', 'c', 0)])
        self.assertEqual(evidence['version'], 'confirmed-navigation-connection-1.0')
        self.assertEqual((evidence['from_lane_id'], evidence['to_lane_id']), ('a|b|0', 'b|c|0'))
        np.testing.assert_allclose(evidence['points_world'], points*[1., -1.])
        np.testing.assert_allclose(evidence['headings_world'], -headings)
        np.testing.assert_allclose(route[1:4], points)
        self.assertTrue(evidence['validated'])
        self.assertEqual(evidence['ego_length_m'], 5.)
        self.assertEqual(evidence['ego_width_m'], 2.)
        self.assertEqual(evidence['steering_limit'], .6)
        self.assertNotIn('_points_native', evidence)
        self.assertNotIn('_headings_native', evidence)
        json.dumps(evidence, allow_nan=False)

    def test_best_candidate_with_body_outside_is_not_certified(self):
        self.assertFalse(self.metadata(outside=1)[3]['validated'])

    def test_best_candidate_exceeding_steering_is_not_certified(self):
        self.assertFalse(self.metadata(steering=.60001)[3]['validated'])

    def test_lane_change_rebuild_converts_native_evidence_exactly_once(self):
        before = self.metadata(right_handed=False)[3]
        after = self.metadata(right_handed=True)[3]
        self.assertEqual(before, after)

    def test_actual_curve_archives_the_body_headings_used_for_road_check(self):
        adapter = fixture()
        before = np.array([[0., 0.], [40., 0.]])
        after = np.array([[45., 4.], [85., 4.]])
        revised, remaining, detail = adapter._connect(before, after)
        points, headings = detail['_points_native'], detail['_headings_native']
        self.assertEqual(points.shape, (len(headings), 2))
        self.assertGreater(len(headings), 2)
        np.testing.assert_allclose(revised[-len(points):], points)
        actual_outside = sum(not adapter.drivable.buffer(1e-6).covers(
            Polygon(rectangle_corners(point, heading, 5., 2.)))
            for point, heading in zip(points, headings))
        self.assertEqual(detail['outside_samples'], actual_outside)
        self.assertEqual(actual_outside, 0)
        self.assertLessEqual(detail['maximum_steering'], .6)
        self.assertGreater(detail['from_end_trim_m'], 0.)
        self.assertGreater(detail['to_start_trim_m'], 0.)
        self.assertGreater(len(remaining), 0)


if __name__ == '__main__':
    unittest.main()
