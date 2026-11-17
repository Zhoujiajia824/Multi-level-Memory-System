"""Causal public priority regression suite; no CARLA or model connection."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_public_priority import PublicPriorityTracker


NAV = [[0., 0.], [100., 0.]]
LANES = {'priority_approach': {'center': [[25., 30.], [25., 2.]], 'successors': ['junction']},
         'junction': {'center': [[25., 2.], [25., -30.]], 'successors': []},
         'ego_lane': {'center': NAV, 'successors': []}}


def actor(identity, x, y, lane, vx=0., vy=0., heading=0.):
    return {'id': identity, 'kind': 'vehicle', 'position': [float(x), float(y)],
            'velocity': [float(vx), float(vy)], 'heading': heading, 'length': 4., 'width': 2.,
            'z': .8, 'lane_id': lane}


def ego(x=10.):
    return actor('ego', x, 0., 'ego_lane', 3., 0.)


def priority(y=10., lane='priority_approach', speed=5.):
    return actor('priority_car', 25., y, lane, 0., -speed, -math.pi/2)


def payload(frame=1):
    return {'frame': frame, 'rules': [{'id': 'yield', 'kind': 'yield', 'source': 'published_road_rule',
        'controlled_lane_ids': ['ego_lane'], 'priority_lane_ids': ['priority_approach'],
        'stop_route_s_m': 18., 'conflict_route_s_m': [20., 30.]}]}


class PriorityTests(unittest.TestCase):
    def update(self, tracker, actors, frame=1, vehicle=None, lanes=LANES):
        return tracker.update(payload(frame), vehicle or ego(), actors, NAV, frame*.1, lanes)

    def test_approaching_visible_priority_vehicle_produces_public_stop_position(self):
        result = self.update(PublicPriorityTracker(), [priority()])
        self.assertTrue(result['available'])
        self.assertEqual(result['stop_position'], [18., 0.])
        self.assertEqual(result['evidence']['rules'][0]['blockers'][0]['actor_id'], 'priority_car')

    def test_observed_priority_survives_lane_transition_then_releases_on_exit(self):
        t = PublicPriorityTracker()
        self.update(t, [priority()])
        at_junction = self.update(t, [priority(2., 'junction')], frame=2)
        self.assertIsNotNone(at_junction['stop_position'])
        departed = self.update(t, [priority(-10., 'junction')], frame=3)
        self.assertIsNone(departed['stop_position'])
        self.assertEqual(departed['evidence']['rules'][0]['current_observed_priority_ids'], [])

    def test_disappeared_actor_is_not_extrapolated_or_reidentified_on_unseen_lane(self):
        t = PublicPriorityTracker()
        self.update(t, [priority()])
        self.assertIsNone(self.update(t, [], frame=2)['stop_position'])
        self.assertIsNone(self.update(t, [priority(1., 'junction')], frame=3)['stop_position'])

    def test_stationary_distant_actor_does_not_hold_ego(self):
        result = self.update(PublicPriorityTracker(), [priority(12., speed=0.)])
        self.assertIsNone(result['stop_position'])
        self.assertEqual(result['evidence']['rules'][0]['ignored'][0]['reason'], 'stationary_outside_conflict')

    def test_stationary_actor_currently_in_conflict_still_requires_yield(self):
        self.assertIsNotNone(self.update(PublicPriorityTracker(), [priority(0., speed=0.)])['stop_position'])

    def test_static_mesh_is_not_given_vehicle_priority(self):
        obstacle = priority(0., speed=0.)
        obstacle['kind'] = 'static_obstacle'
        result = self.update(PublicPriorityTracker(), [obstacle])
        self.assertIsNone(result['stop_position'])
        self.assertEqual(result['evidence']['rules'][0]['current_observed_priority_ids'], [])

    def test_route_overlap_keeps_causal_lap_progress(self):
        t = PublicPriorityTracker()
        route = [[0., 0.], [10., 0.], [10., 10.], [0., 10.], [0., 0.], [10., 0.]]
        points = ([(x, 0.) for x in range(11)]+[(10., y) for y in range(1, 11)]+
                  [(x, 10.) for x in range(9, -1, -1)]+[(0., y) for y in range(9, -1, -1)]+
                  [(x, 0.) for x in range(1, 6)])
        for frame, position in enumerate(points):
            vehicle = ego()
            vehicle['position'] = list(position)
            result = t.update({'frame': frame, 'rules': []}, vehicle, [], route, frame*.1)
        self.assertAlmostEqual(result['evidence']['ego_locked_route_s_m'], 45.)

    def test_merged_same_lane_front_vehicle_is_left_to_following(self):
        t = PublicPriorityTracker()
        self.update(t, [priority(10.)])
        merged = actor('priority_car', 25., 0., 'ego_lane', 3., 0.)
        result = self.update(t, [merged], frame=2)
        self.assertIsNone(result['stop_position'])
        self.assertIn('same_direction_same_lane', result['evidence']['rules'][0]['ignored'][0]['reason'])

    def test_stop_line_behind_front_is_not_reintroduced(self):
        result = self.update(PublicPriorityTracker(), [priority(0.)], vehicle=ego(19.))
        self.assertIsNone(result['stop_position'])
        self.assertTrue(result['evidence']['rules'][0]['blockers'])
        self.assertEqual(result['evidence']['rules'][0]['status'], 'conflict_already_entered_use_current_obstacle_avoidance')

    def test_curved_public_approach_reaches_conflict_even_if_current_tangent_misses(self):
        lanes = {'curve': {'center': [[5., 15.], [25., 15.], [25., 0.]], 'successors': []}}
        car = actor('priority_car', 5., 15., 'priority_approach', 5., 0.)
        lanes['priority_approach'] = lanes.pop('curve')
        result = self.update(PublicPriorityTracker(approach_horizon_s=8.), [car], lanes=lanes)
        self.assertIsNotNone(result['stop_position'])
        self.assertEqual(result['evidence']['rules'][0]['blockers'][0]['reason'], 'public_lane_and_successor_sweep')

    def test_missing_map_geometry_is_explicit(self):
        result = self.update(PublicPriorityTracker(), [priority()], lanes={})
        self.assertFalse(result['available'])
        self.assertIsNotNone(result['stop_position'])

    def test_current_observation_can_include_more_than_eight_actors(self):
        actors = [actor(str(i), 50.+i, 0., 'ego_lane') for i in range(15)]+[priority()]
        result = self.update(PublicPriorityTracker(), actors)
        self.assertEqual(result['evidence']['full_observed_actor_count'], 16)
        self.assertIsNotNone(result['stop_position'])

    def test_old_observation_and_changed_navigation_are_rejected(self):
        t = PublicPriorityTracker()
        self.update(t, [], frame=2)
        with self.assertRaises(ValueError):
            self.update(t, [], frame=1)
        with self.assertRaises(ValueError):
            t.update(payload(3), ego(), [], [[0., 1.], [100., 1.]], .3, LANES)

    def test_opposite_motion_and_different_elevation_do_not_block(self):
        away = priority(10., speed=-5.)
        result = self.update(PublicPriorityTracker(), [away])
        self.assertIsNone(result['stop_position'])
        overhead = priority(0.)
        overhead['z'] = 10.
        self.assertIsNone(self.update(PublicPriorityTracker(), [overhead])['stop_position'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
