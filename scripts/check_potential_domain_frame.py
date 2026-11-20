"""World-domain queries under ego poses; portable topology regression, no simulator/model.

The small touching-hole polygon is valid in its authoritative frame. Rotating
its vertices can make the touching hole cross its shell through roundoff, as
with the archived roundabout lane union. No user result directory is needed.
"""
from pathlib import Path
import json
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box

from nine_grid.contracts import Actor, Lane, PredictionBundle, SceneSnapshot
from nine_grid.geometry import ego_to_world, world_to_ego
from nine_grid.lane_potential import LanePotentialGeometry
from nine_grid.planner import Planner


ORIGIN = np.array([3.1, 5.7])
ANGLE = .7
REFERENCE = np.array([[-20., 5.], [40., 5.]])


def touching_hole():
    return Polygon([(0., 0.), (10., 0.), (10., 10.), (0., 10.)],
                   holes=[[(5., 0.), (7., 2.), (3., 2.)]])


def world_signed_distance(domain, local_points):
    """Independent scalar geometry oracle; no potential query helpers."""
    result = []
    for xy in ego_to_world(np.atleast_2d(local_points), ORIGIN, ANGLE):
        point = Point(xy)
        result.append(domain.boundary.distance(point)*(1. if domain.covers(point) else -1.))
    return np.asarray(result)


class PotentialDomainFrames(unittest.TestCase):
    def test_world_and_local_queries_have_identical_membership(self):
        domain = MultiPolygon([box(0., 0., 10., 10.), box(20., 1., 24., 4.)])
        points = np.array([[2., 3.], [9., 9.], [22., 2.], [15., 3.], [-1., 3.]])
        world = LanePotentialGeometry(REFERENCE, domain)
        local = LanePotentialGeometry(world_to_ego(REFERENCE, ORIGIN, ANGLE), domain,
                                      domain_pose=(ORIGIN, ANGLE))
        expected = [True, True, True, False, False]
        np.testing.assert_array_equal(world.domain_contains(points), expected)
        np.testing.assert_array_equal(local.domain_contains(world_to_ego(points, ORIGIN, ANGLE)), expected)
        self.assertIs(local.domain, domain)
        self.assertTrue(domain.is_valid)

    def test_holes_and_gaps_are_not_authorized_by_pose(self):
        domain = MultiPolygon([touching_hole(), box(20., 1., 24., 4.)])
        points = np.array([[5., .5], [5., 3.], [15., 3.], [22., 2.]])
        geometry = LanePotentialGeometry(world_to_ego(REFERENCE, ORIGIN, ANGLE), domain,
                                         domain_pose=(ORIGIN, ANGLE))
        np.testing.assert_array_equal(geometry.domain_contains(world_to_ego(points, ORIGIN, ANGLE)),
                                      [False, True, False, True])
        margin, _ = geometry.domain_clearance(world_to_ego(points, ORIGIN, ANGLE))
        np.testing.assert_array_equal(margin > 0., [False, True, False, True])

    def test_projection_mask_uses_the_original_world_domain(self):
        domain = touching_hole()
        points = np.array([[2., 3.], [5., .5], [11., 5.]])
        world = LanePotentialGeometry(REFERENCE, domain)
        local = LanePotentialGeometry(world_to_ego(REFERENCE, ORIGIN, ANGLE), domain,
                                      domain_pose=(ORIGIN, ANGLE))
        a = world.project(points)
        b = local.project(world_to_ego(points, ORIGIN, ANGLE))
        np.testing.assert_array_equal(a['domain_valid'], [True, False, False])
        np.testing.assert_array_equal(a['domain_valid'], b['domain_valid'])
        np.testing.assert_array_equal(a['valid'], b['valid'])
        np.testing.assert_allclose(a['s'], b['s'], atol=1e-8)
        np.testing.assert_allclose(a['d'], b['d'], atol=1e-8)

    def test_signed_clearance_and_rotated_normals_match_finite_differences(self):
        domain = touching_hole()
        # Avoid equidistant corners/non-differentiable nearest-edge switches.
        points = world_to_ego(np.array([[1., 4.], [9., 5.], [11., 5.], [5.2, .5], [5., 3.]]), ORIGIN, ANGLE)
        geometry = LanePotentialGeometry(world_to_ego(REFERENCE, ORIGIN, ANGLE), domain,
                                         domain_pose=(ORIGIN, ANGLE))
        margin, normal = geometry.domain_clearance(points)
        np.testing.assert_allclose(margin, world_signed_distance(domain, points), atol=1e-12)
        gradient = np.empty_like(points)
        epsilon = 1e-5
        for axis in range(2):
            shift = np.zeros(2);shift[axis] = epsilon
            gradient[:, axis] = (world_signed_distance(domain, points+shift)-
                                 world_signed_distance(domain, points-shift))/(2*epsilon)
        np.testing.assert_allclose(normal, gradient, atol=2e-8)
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1., atol=1e-12)

    def test_invalid_authoritative_domain_still_rejected(self):
        invalid = Polygon([(0., 0.), (10., 10.), (0., 10.), (10., 0.)])
        self.assertFalse(invalid.is_valid)
        for pose in (None, (ORIGIN, ANGLE)):
            with self.subTest(pose=pose):
                with self.assertRaisesRegex(ValueError, 'valid geometry'):
                    LanePotentialGeometry(REFERENCE, invalid, domain_pose=pose)

    def test_valid_world_domain_survives_legacy_rotation_topology_failure(self):
        domain = touching_hole()
        self.assertTrue(domain.is_valid)
        rotated = shapely.transform(domain, lambda xy: world_to_ego(xy, ORIGIN, ANGLE))
        self.assertFalse(rotated.is_valid, 'Fixture must reproduce the legacy vertex-rotation topology failure')
        before = domain.wkb
        geometry = LanePotentialGeometry(world_to_ego(REFERENCE, ORIGIN, ANGLE), domain,
                                         domain_pose=(ORIGIN, ANGLE))
        np.testing.assert_array_equal(geometry.domain_contains(world_to_ego([[1., 4.], [5., .5]], ORIGIN, ANGLE)),
                                      [True, False])
        self.assertEqual(domain.wkb, before)
        self.assertEqual(geometry.diagnostics['domain_coordinates'], 'unchanged_world_frame')

    def test_boundary_is_inclusive_and_clearance_is_finite(self):
        geometry = LanePotentialGeometry(REFERENCE, box(0., 0., 10., 10.))
        points = np.array([[0., 4.], [10., 4.], [4., 0.]])
        self.assertTrue(geometry.domain_contains(points).all())
        margin, normal = geometry.domain_clearance(points)
        np.testing.assert_allclose(margin, 0., atol=1e-12)
        self.assertTrue(np.isfinite(normal).all())

    def test_planner_accepts_valid_world_domain_without_rotating_it(self):
        domain = touching_hole()
        direction = np.array([np.cos(ANGLE), np.sin(ANGLE)])
        center = ORIGIN+np.array([-30., 100.])[:, None]*direction
        lane = Lane('own', center, np.array(box(0., 0., 10., 10.).exterior.coords), 3.5, 5.)
        ego = Actor('ego', 'vehicle', ORIGIN.copy(), .5*direction, np.zeros(2), ANGLE, 4., 1.8, 'own')
        scene = SceneSnapshot('domain-frame:1', 0., ego, [], [lane], center, ['own'],
                              box(-50., -50., 80., 80.), 'integration',
                              rules={'desired_speed': .5, 'speed_limit': 5.},
                              diagnostics={'lane_map_complete': True})
        config = json.loads((ROOT/'config/parameters.json').read_text(encoding='utf-8'))
        predictions = PredictionBundle(scene.frame_id, scene.time, np.arange(31)*.1, {}, {})
        with patch('nine_grid.planner.build_lane_potential_domain', return_value=(domain, {})), \
                patch('nine_grid.planner.LanePotentialGeometry', wraps=LanePotentialGeometry) as constructor:
            plan = Planner(config).solve(scene, predictions)
        self.assertIs(constructor.call_args.args[1], domain)
        np.testing.assert_allclose(constructor.call_args.kwargs['domain_pose'][0], ORIGIN)
        self.assertAlmostEqual(constructor.call_args.kwargs['domain_pose'][1], ANGLE)
        self.assertTrue(plan.valid)
        self.assertTrue(np.isfinite(plan.states).all() and np.isfinite(plan.controls).all())
        self.assertTrue(plan.diagnostics['safety_feasible'])
        self.assertEqual(plan.diagnostics['potential_field']['invalid_samples'], 0)
        self.assertEqual(plan.diagnostics['dense_validation']['outside_potential_domain_samples'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
