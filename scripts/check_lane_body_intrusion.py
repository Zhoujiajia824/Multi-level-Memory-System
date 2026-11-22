"""纯几何回归：真实车身侵入在对象范围、检索八向和规划选择中保持一致。"""
from pathlib import Path
import sys
import unittest
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from check_lane_object_scope import actor, lane, straight
from nine_grid.geometry import actor_shape
from nine_grid.lane_roles import ASSIGNMENT_PROTOCOL, belongs_to_trace
from nine_grid.object_scope import lane_role_geometry, scope_scene
from nine_grid.planning_objects import planning_scene, select_planning_objects
from nine_grid.representations import build_grid


def current_only():
    scene = straight()
    scene.lanes = scene.lanes[:1]
    scene.lanes[0].left = scene.lanes[0].right = None
    return scene


def summaries(scene):
    scene.drivable = unary_union([Polygon(item.polygon) for item in scene.lanes])
    scene.diagnostics['visibility_wkt'] = scene.drivable.wkt
    _, scoped = scope_scene(scene)
    _, planning = select_planning_objects(scene)
    grid = build_grid(scene, object_range={'front_m': 40., 'rear_m': 20.}, include_static=True)
    return scoped, planning, grid


class BodyIntrusionTests(unittest.TestCase):
    def assert_front(self, scene, identity):
        scope, planning, grid = summaries(scene)
        self.assertIn(identity, scope['selected_ids'])
        self.assertIn('front', scope['objects'][identity]['roles'])
        self.assertEqual(planning['region_objects']['front'], identity)
        self.assertEqual(grid['slots']['front']['id'], identity)
        return scope

    def test_crossing_body_invasion_preserves_source_owner(self):
        scene = current_only()
        scene.lanes.append(lane('cross', [[10, 30], [10, -30]]))
        scene.actors = [actor('crossing', 10, 3., heading=-np.pi/2, lane='cross', speed=3)]
        self.assertAlmostEqual(actor_shape(scene.actors[0]).intersection(Polygon(scene.lanes[0].polygon)).area, 1.35)
        scope = self.assert_front(scene, 'crossing')
        self.assertEqual(scope['objects']['crossing']['owner_lane_id'], 'cross')

    def test_turning_body_invasion_is_not_rejected_by_source_tangent(self):
        scene = current_only()
        scene.lanes.append(lane('turn', [[15, 8], [12, 5], [10, 3], [8, 0]]))
        scene.actors = [actor('turning', 10, 3., heading=-3*np.pi/4, lane='turn', speed=3)]
        self.assert_front(scene, 'turning')

    def test_opposing_lane_intruder_but_not_nonintruding_opposing_vehicle(self):
        scene = current_only()
        scene.lanes.append(lane('opposing', [[100, 3.5], [-60, 3.5]]))
        scene.actors = [actor('intruder', 10, 2.2, heading=np.pi, lane='opposing', speed=3),
                        actor('outside', 20, 3.5, heading=np.pi, lane='opposing', speed=3)]
        scope = self.assert_front(scene, 'intruder')
        self.assertEqual(scope['selected_ids'], ['intruder'])

    def test_parallel_intruder_can_occupy_two_roles(self):
        scene = straight()
        scene.actors = [actor('intruder', 10, 2.2, lane='l', speed=10)]
        scope = self.assert_front(scene, 'intruder')
        self.assertEqual(scope['objects']['intruder']['roles'], ['front', 'left_front'])

    def test_unequal_lane_width_small_real_invasion(self):
        scene = current_only()
        scene.lanes = [lane('c', [[-60, 0], [100, 0]], width=5.5),
                       lane('r', [[-60, -4.5], [100, -4.5]], width=3.5)]
        scene.actors = [actor('intruder', 10, -3.6, lane='r', speed=10)]
        self.assertAlmostEqual(actor_shape(scene.actors[0]).intersection(Polygon(scene.lanes[0].polygon)).area, .2)
        self.assert_front(scene, 'intruder')

    def test_owner_none_uses_same_intrusion_in_scope_and_grid(self):
        scene = current_only()
        scene.actors = [actor('intruder', 10, 2.2, lane=None, speed=10)]
        scope = self.assert_front(scene, 'intruder')
        self.assertIsNone(scope['objects']['intruder']['owner_lane_id'])

    def test_wide_junction_polygon_does_not_create_physical_occupancy(self):
        scene = current_only()
        scene.lanes[0].polygon = np.asarray(box(-60, -10, 100, 10).exterior.coords)
        scene.actors = [actor('map_bbox_only', 10, 7., lane='c', speed=3)]
        self.assertGreater(actor_shape(scene.actors[0]).intersection(Polygon(scene.lanes[0].polygon)).area, 0.)
        scope, planning, grid = summaries(scene)
        self.assertEqual(scope['selected_ids'], [])
        self.assertEqual(planning['selected_ids'], [])
        self.assertIsNone(grid['slots']['front']['id'])

    def test_intrusion_does_not_expand_center_arc_range(self):
        scene = current_only()
        scene.actors = [actor('beyond_front', 41., 2.2, lane=None, speed=10),
                        actor('beyond_rear', -21., 2.2, lane=None, speed=10)]
        scope, planning, grid = summaries(scene)
        self.assertEqual(scope['selected_ids'], [])
        self.assertEqual(planning['selected_ids'], [])
        self.assertIsNone(grid['slots']['front']['id'])
        self.assertIsNone(grid['slots']['rear']['id'])

    def test_solid_neighbor_stays_excluded_until_body_intrudes_current_lane(self):
        scene = straight(False)
        scene.actors = [actor('intruder', 10, 2.2, lane='l', speed=10),
                        actor('within_neighbor_only', 20, 3.5, lane='l', speed=10)]
        scope = self.assert_front(scene, 'intruder')
        self.assertEqual(scope['selected_ids'], ['intruder'])
        self.assertEqual(scope['objects']['intruder']['roles'], ['front'])
        self.assertFalse(scope['roads']['left']['crossable'])

    def test_nearest_representative_does_not_truncate_all_lane_planning(self):
        scene = straight()
        scene.actors = [actor('near', 10, 2.2, lane='l', speed=10),
                        actor('far', 18, 2.2, lane='l', speed=10)]
        scope, planning, grid = summaries(scene)
        self.assertEqual(scope['selected_ids'], ['far', 'near'])
        self.assertEqual(planning['selected_ids'], ['far', 'near'])
        self.assertEqual(planning['base_selected_ids'], ['near'])
        self.assertEqual(grid['slots']['front']['id'], 'near')
        scene.actors[1].velocity[1] = -1.
        _, planning, grid = summaries(scene)
        self.assertEqual(planning['selected_ids'], ['far', 'near'])
        self.assertEqual(planning['exception_added_ids'], ['far'])
        self.assertEqual(grid['slots']['front']['id'], 'near')

    def test_invalid_width_fails_closed(self):
        scene = current_only()
        obj = actor('a', 10.)
        for bad in (None, 0., -1., np.nan, np.inf, True, '3.5'):
            with self.subTest(width=bad):
                scene.lanes[0].width = bad
                traces, _, cache = lane_role_geometry(scene)
                self.assertFalse(belongs_to_trace(obj, actor_shape(obj), 'c', traces['center'], cache, 1e-5))

    def test_old_assignment_cache_cannot_reuse_missing_intruders(self):
        scene = straight()
        scene.actors = [actor('near', 10, 2.2, lane='l', speed=10)]
        selected, _ = planning_scene(scene)
        for old_protocol in ('lane-topology-role-assignment-1.0', None):
            cached = deepcopy(selected)
            cached.diagnostics['planning_scope']['assignment_protocol'] = old_protocol
            for entrance in (select_planning_objects, planning_scene):
                with self.subTest(protocol=old_protocol, entrance=entrance.__name__):
                    with self.assertRaisesRegex(ValueError, '完整冻结帧重新筛选'):
                        entrance(cached)

    def test_current_assignment_cache_retains_conflict_supplement(self):
        scene = straight()
        scene.actors = [actor('near', 10, 2.2, lane='l', speed=10),
                        actor('far', 18, 2.2, lane='l', speed=10)]
        scene.actors[1].velocity[1] = -1.
        selected, original = planning_scene(scene)
        self.assertEqual(original['assignment_protocol'], ASSIGNMENT_PROTOCOL)
        self.assertEqual(original['exception_added_ids'], ['far'])
        actors, direct = select_planning_objects(selected)
        repeated_scene, repeated = planning_scene(selected)
        self.assertEqual([a.id for a in actors], ['far', 'near'])
        self.assertEqual([a.id for a in repeated_scene.actors], ['far', 'near'])
        self.assertEqual(direct, original)
        self.assertEqual(repeated, original)

    def test_only_existing_confirmed_connection_band_can_admit_gap_object(self):
        scene = current_only()
        scene.lanes[0] = lane('c', [[-60, 0], [0, 0]])
        obj = actor('in_connection', 5., lane=None)
        traces, _, cache = lane_role_geometry(scene)
        trace = traces['center']
        self.assertFalse(belongs_to_trace(obj, actor_shape(obj), None, trace, cache, 1e-5))
        trace['navigation_connections'] = [{'shape': box(0., -1.75, 10., 1.75)}]
        self.assertTrue(belongs_to_trace(obj, actor_shape(obj), None, trace, cache, 1e-5))
        outside = actor('outside_connection', 5., 4., lane=None)
        self.assertFalse(belongs_to_trace(outside, actor_shape(outside), None, trace, cache, 1e-5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
