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
from nine_grid.navigation_connections import (VERSION, active_navigation_connection,
                                              confirmed_navigation_connections)
from nine_grid.object_scope import lane_role_geometry, select_scope_objects
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


def active_bridge_fixture():
    scene = fixture()
    # A native nearest-lane label can identify the other branch while the body
    # is following the independently validated, frozen navigation connector.
    scene.ego.position = np.array([16., 0.])
    scene.ego.lane_id = 'fork'
    scene.diagnostics.update(reference_anchor_s=36., route_s=36.)
    return scene


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

    def test_current_connector_resolves_roles_without_changing_native_identity(self):
        scene = active_bridge_fixture()
        active = active_navigation_connection(scene)
        self.assertIsNotNone(active)
        self.assertEqual(active['evidence']['raw_lane_id'], 'fork')
        traces, roads, _ = lane_role_geometry(scene)
        self.assertEqual(traces['center']['lane_ids'], ['in', 'ring'])
        self.assertEqual(traces['center']['current_lane_association']['to_lane_id'], 'ring')
        self.assertEqual(scene.ego.lane_id, 'fork')
        for side in ('left', 'right'):
            self.assertIsNone(traces[side])
            self.assertFalse(roads[side]['known'])
            self.assertIsNone(roads[side]['crossable'])
        domain, evidence = build_lane_potential_domain(scene)
        self.assertTrue(domain.covers(Point(30., 0.)))
        self.assertFalse(domain.covers(Point(16., 12.)))
        self.assertEqual(evidence['current_lane_association']['raw_lane_id'], 'fork')
        allowed = copy.deepcopy(scene.ego)
        allowed.id, allowed.lane_id, allowed.position = 'exit_vehicle', 'ring', np.array([28., 0.])
        branch = copy.deepcopy(scene.ego)
        branch.id, branch.lane_id, branch.position = 'branch_vehicle', 'fork', np.array([16., 12.])
        scene.actors = [allowed, branch]
        selected, evidence = select_scope_objects(scene)
        self.assertEqual([actor.id for actor in selected], ['exit_vehicle'])
        self.assertEqual(evidence['objects']['exit_vehicle']['roles'], ['front'])

    def test_explicit_current_lane_is_not_overridden_by_navigation(self):
        scene = active_bridge_fixture()
        traces, _, _ = lane_role_geometry(scene, current_lane_id='fork')
        self.assertEqual(traces['center']['lane_ids'], ['fork'])
        self.assertIsNone(traces['center'].get('current_lane_association'))

    def test_connector_association_rejects_wrong_direction_band_anchor_and_evidence(self):
        for case in ('reverse', 'outside_band', 'wrong_anchor', 'missing_anchor', 'forged_evidence'):
            with self.subTest(case=case):
                scene = active_bridge_fixture()
                if case == 'reverse':
                    scene.ego.heading = np.pi
                elif case == 'outside_band':
                    scene.ego.position[1] = 2.1
                elif case == 'wrong_anchor':
                    scene.diagnostics.update(reference_anchor_s=0., route_s=0.)
                elif case == 'missing_anchor':
                    scene.diagnostics.pop('reference_anchor_s')
                    scene.diagnostics.pop('route_s')
                else:
                    row(scene)['validated'] = False
                self.assertIsNone(active_navigation_connection(scene))
                traces, _, _ = lane_role_geometry(scene)
                self.assertEqual(traces['center']['lane_ids'], ['fork'])
                self.assertIsNone(traces['center'].get('current_lane_association'))

    def test_native_endpoint_is_not_mistaken_for_connector_interior(self):
        for x in (8., 18.):
            with self.subTest(x=x):
                scene = active_bridge_fixture()
                scene.ego.position = np.array([x, 0.])
                scene.diagnostics.update(reference_anchor_s=x+20., route_s=x+20.)
                self.assertIsNone(active_navigation_connection(scene))


if __name__ == '__main__':
    unittest.main()
