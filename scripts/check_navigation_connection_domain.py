"""Confirmed navigation bridges: actual geometry, scoped domains; no simulator/SQP/model."""
from pathlib import Path
import copy
import sys
import unittest

import numpy as np
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nine_grid.contracts import Actor, Lane, SceneSnapshot
from nine_grid.lane_roles import PARAMETERS, trace_lane
from nine_grid.navigation_connections import VERSION, confirmed_navigation_connections
from nine_grid.potential_domain import build_lane_potential_domain


def fixture():
    own = Lane('in', np.array([[-20., 0.], [10., 0.]]),
               np.array(box(-25, -2, 15, 2).exterior.coords), 4., 12., successors=['ring', 'fork'])
    ring = Lane('ring', np.array([[16., 0.], [60., 0.]]),
                np.array(box(11, -2, 65, 2).exterior.coords), 4., 12.)
    fork = Lane('fork', np.array([[16., 0.], [16., 30.]]),
                np.array(box(14, -5, 18, 35).exterior.coords), 4., 12.)
    points = np.column_stack((np.arange(8., 18.01, .5), np.zeros(21)))
    connection = dict(version=VERSION, from_lane_id='in', to_lane_id='ring',
        points_world=points.tolist(), headings_world=[0.]*len(points), outside_samples=0,
        maximum_steering=.1, steering_limit=.6, ego_length_m=4., ego_width_m=1.8,
        from_end_trim_m=2., to_start_trim_m=2., validated=True)
    ego = Actor('ego', 'vehicle', np.array([5., 0.]), np.array([2., 0.]),
                np.zeros(2), 0., 4., 1.8, lane_id='in')
    return SceneSnapshot('bridge:1', 0., ego, [], [own, ring, fork],
        np.vstack(([[-20., 0.]], points, [[60., 0.]])), ['in', 'ring'], box(-30, -2, 70, 2),
        'regression', diagnostics={'lane_map_complete': True, 'navigation': {'connections': [connection]}})


def row(scene):
    return scene.diagnostics['navigation']['connections'][0]


class NavigationConnections(unittest.TestCase):
    def test_valid_bridge_continues_trace_and_fills_only_narrow_domain(self):
        scene = fixture()
        accepted = confirmed_navigation_connections(scene)
        self.assertEqual(set(accepted), {('in', 'ring')})
        self.assertIs(accepted, confirmed_navigation_connections(scene))
        self.assertFalse(accepted[('in', 'ring')]['points'].flags.writeable)
        trace = trace_lane(scene, 'in', front_m=40., back_m=20.)
        self.assertEqual(trace['lane_ids'], ['in', 'ring'])
        self.assertEqual(len(trace['navigation_connections']), 1)
        domain, evidence = build_lane_potential_domain(scene)
        self.assertTrue(domain.covers(Point(13., 0.)))
        self.assertFalse(domain.covers(Point(13., 3.)))
        self.assertEqual(evidence['navigation_connections'], [{'from_lane_id': 'in', 'to_lane_id': 'ring'}])
        self.assertEqual(PARAMETERS['connection_gap_m'], 2.)

    def test_without_evidence_native_gap_still_stops_chain(self):
        scene = fixture()
        scene.diagnostics['navigation']['connections'] = []
        self.assertEqual(trace_lane(scene, 'in')['lane_ids'], ['in'])
        domain, evidence = build_lane_potential_domain(scene)
        self.assertFalse(domain.covers(Point(13., 0.)))
        self.assertEqual(evidence['navigation_connections'], [])

    def test_repeated_native_sampling_endpoints_preserve_connection(self):
        scene = fixture()
        for lane in scene.lanes:
            lane.center = np.repeat(lane.center, 2, axis=0)
        self.assertEqual(set(confirmed_navigation_connections(scene)), {('in', 'ring')})
        self.assertEqual(trace_lane(scene, 'in')['lane_ids'], ['in', 'ring'])
        self.assertTrue(build_lane_potential_domain(scene)[0].covers(Point(13., 0.)))

    def test_failed_producer_evidence_is_not_authorization(self):
        for key, value in [('validated', False), ('outside_samples', 1),
                           ('maximum_steering', .7), ('version', 'unknown')]:
            with self.subTest(key=key):
                scene = fixture();row(scene)[key] = value
                self.assertFalse(confirmed_navigation_connections(scene))

    def test_branch_not_in_ordered_navigation_rejected(self):
        scene = fixture();row(scene)['to_lane_id'] = 'fork'
        self.assertFalse(confirmed_navigation_connections(scene))
        scene = fixture();scene.route_lane_ids = ['in', 'fork', 'ring']
        self.assertFalse(confirmed_navigation_connections(scene))

    def test_missing_public_successor_rejected(self):
        scene = fixture();scene.lanes[0].successors = ['fork']
        self.assertFalse(confirmed_navigation_connections(scene))

    def test_full_body_is_checked_independently_of_reported_zero(self):
        scene = fixture()
        # Centres still fit, but the vehicle body does not fit the actual road.
        scene.drivable = box(-30., -.5, 70., .5)
        self.assertFalse(confirmed_navigation_connections(scene))
        self.assertEqual(scene.assets['_confirmed_navigation_connections']['rejected'][0]['reason'],
                         'independent_full_body_road_check_failed')

    def test_reversed_heading_rejected(self):
        scene = fixture();row(scene)['headings_world'] = [np.pi]*21
        self.assertFalse(confirmed_navigation_connections(scene))

    def test_different_vehicle_or_detached_native_ends_rejected(self):
        scene = fixture();row(scene)['ego_width_m'] = 1.
        self.assertFalse(confirmed_navigation_connections(scene))
        scene = fixture();row(scene)['points_world'][0][1] = .1
        self.assertFalse(confirmed_navigation_connections(scene))

    def test_frozen_route_changed_in_place_invalidates_cached_proof(self):
        scene = fixture();self.assertTrue(confirmed_navigation_connections(scene))
        scene.route[:, 1] += .2
        self.assertFalse(confirmed_navigation_connections(scene))

    def test_mutated_or_deepcopied_metadata_cannot_reuse_cache(self):
        scene = fixture();self.assertTrue(confirmed_navigation_connections(scene))
        changed = copy.deepcopy(scene);row(changed)['validated'] = False
        self.assertFalse(confirmed_navigation_connections(changed))
        row(scene)['outside_samples'] = 1
        self.assertFalse(confirmed_navigation_connections(scene))

    def test_duplicate_connection_is_ambiguous(self):
        scene = fixture();scene.diagnostics['navigation']['connections'].append(copy.deepcopy(row(scene)))
        self.assertFalse(confirmed_navigation_connections(scene))

    def test_untraversed_valid_bridge_not_added_to_domain(self):
        scene = fixture();scene.ego.position = np.array([-19., 0.])
        self.assertTrue(confirmed_navigation_connections(scene))
        domain, evidence = build_lane_potential_domain(scene, front_m=5., rear_m=0.)
        self.assertFalse(domain.covers(Point(13., 0.)))
        self.assertEqual(evidence['navigation_connections'], [])


if __name__ == '__main__':
    unittest.main()
