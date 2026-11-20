"""Offline dense bicycle-state/Jacobian and between-node line checks.

These checks create no environment, actors, model or planner. Evaluating a
symbolic rollout must not advance a simulator or alter its input arrays.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nine_grid.geometry import rectangle_corners
from nine_grid.realtime_solver import Dynamics


class DenseDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n, cls.dt, cls.substeps = 30, .1, 5
        cls.model = Dynamics(cls.n, cls.dt, cls.substeps, 5.)
        phase = np.arange(cls.n, dtype=float)
        cls.controls = np.column_stack((.35*np.sin(.6*phase), .21+.08*np.cos(.4*phase)))
        cls.initial = np.array([1., -2., .35, 6.2])

    def test_dense_shape_and_legacy_node_interfaces_are_identical(self):
        dense = self.model.dense_rollout(self.controls, self.initial)
        dense_jac = self.model.dense_sensitivity(self.controls, self.initial)
        self.assertEqual(dense.shape, (151, 4))
        self.assertEqual(dense_jac.shape, (151, 4, 60))
        self.assertTrue(np.isfinite(dense).all() and np.isfinite(dense_jac).all())
        nodes, jac = self.model(self.controls, self.initial)
        self.assertEqual(nodes.shape, (31, 4))
        self.assertEqual(jac.shape, (31, 4, 60))
        np.testing.assert_array_equal(dense[::self.substeps], nodes)
        np.testing.assert_array_equal(dense[::self.substeps], self.model.rollout(self.controls, self.initial))
        np.testing.assert_array_equal(dense_jac[::self.substeps], jac)
        np.testing.assert_array_equal(dense_jac[::self.substeps], self.model.sensitivity(self.controls, self.initial))

    def test_all_dense_control_derivatives_match_central_differences(self):
        actual = self.model.dense_sensitivity(self.controls, self.initial)
        step = 1e-6
        numeric = np.empty_like(actual)
        for column in range(2*self.n):
            plus, minus = self.controls.copy(), self.controls.copy()
            plus.ravel()[column] += step
            minus.ravel()[column] -= step
            numeric[:, :, column] = (self.model.dense_rollout(plus, self.initial)-
                self.model.dense_rollout(minus, self.initial))/(2*step)
        np.testing.assert_allclose(actual, numeric, rtol=3e-6, atol=2e-7)

    def test_controls_are_bound_to_their_own_physical_substeps(self):
        jac = self.model.dense_sensitivity(self.controls, self.initial)
        np.testing.assert_array_equal(jac[0], np.zeros((4, 60)))
        for frame in range(151):
            # Frame 0 is the observation. Frames 1..5 use control 0, 6..10
            # use control 1, and the last five use control 29.
            prefix = (frame+self.substeps-1)//self.substeps
            np.testing.assert_array_equal(jac[frame, :, 2*prefix:],
                                          np.zeros_like(jac[frame, :, 2*prefix:]))
        baseline = self.model.dense_rollout(self.controls, self.initial)
        for segment in (0, 7, 29):
            changed = self.controls.copy()
            changed[segment, 0] += .001
            actual = self.model.dense_rollout(changed, self.initial)
            begin = segment*self.substeps
            np.testing.assert_array_equal(actual[:begin+1], baseline[:begin+1])
            self.assertAlmostEqual(actual[begin+1, 3]-baseline[begin+1, 3],
                                   .001*self.dt/self.substeps, places=12)
            changed = self.controls.copy()
            changed[segment, 1] += .001
            actual = self.model.dense_rollout(changed, self.initial)
            np.testing.assert_array_equal(actual[:begin+1], baseline[:begin+1])
            self.assertGreater(np.linalg.norm(actual[begin+1, :3]-baseline[begin+1, :3]), 0.)

    def test_queries_preserve_inputs_and_integrate_exactly_the_same_duration(self):
        controls = np.column_stack((np.full(30, .7), np.zeros(30)))
        initial = np.array([1., 2., 0., 3.])
        saved_controls, saved_initial = controls.copy(), initial.copy()
        first = self.model.dense_rollout(controls, initial)
        self.model.dense_sensitivity(controls, initial)
        second = self.model.dense_rollout(controls, initial)
        np.testing.assert_array_equal(controls, saved_controls)
        np.testing.assert_array_equal(initial, saved_initial)
        np.testing.assert_array_equal(first, second)
        step, physical_steps = .02, 150
        # Explicit Euler: position uses the speed before each acceleration
        # update. This also detects an extra initial/final physical update.
        expected_x = initial[0]+initial[3]*physical_steps*step + .7*step**2*physical_steps*(physical_steps-1)/2
        self.assertAlmostEqual(first[-1, 0], expected_x, places=11)
        self.assertAlmostEqual(first[-1, 3], initial[3]+.7*3., places=11)
        np.testing.assert_array_equal(first[:, 1], np.full(151, initial[1]))
        np.testing.assert_array_equal(first[:, 2], np.zeros(151))

    def test_platform_wheelbase_preserves_dense_node_contract(self):
        model = Dynamics(3, .1, 5, 4.6, 2.8, 1.2)
        controls = self.controls[:3]
        dense = model.dense_rollout(controls, self.initial)
        jac = model.dense_sensitivity(controls, self.initial)
        self.assertEqual(dense.shape, (16, 4))
        self.assertEqual(jac.shape, (16, 4, 6))
        np.testing.assert_array_equal(dense[::5], model.rollout(controls, self.initial))
        np.testing.assert_array_equal(jac[::5], model.sensitivity(controls, self.initial))


class DenseLineGeometry(unittest.TestCase):
    def test_clear_nodes_do_not_certify_the_intermediate_vehicle_body(self):
        # An independent counterexample, not a reconstruction of the reported
        # roundabout: both 0.5 s nodes clear a short line by 0.5 m while the
        # same 0.02 s Euler trajectory crosses it between the nodes.
        model = Dynamics(1, .5, 25, 5.)
        dense = model.dense_rollout(np.zeros((1, 2)), np.array([-3., 0., 0., 12.]))
        nodes = dense[::25]
        self.assertEqual(len(dense), 26)
        np.testing.assert_allclose(nodes[:, 0], [-3., 3.], rtol=0., atol=1e-12)
        line = shapely.LineString([(0., -1.5), (0., 1.5)])
        bodies = shapely.polygons(np.stack([rectangle_corners(s[:2], s[2], 5., 2.) for s in dense]))
        node_distance = shapely.distance(bodies[::25], line)
        dense_distance = shapely.distance(bodies, line)
        self.assertTrue((node_distance > .05).all())
        self.assertFalse(shapely.intersects(bodies[::25], line).any())
        self.assertTrue(shapely.intersects(bodies[1:-1], line).any())
        self.assertEqual(float(dense_distance.min()), 0.)


if __name__ == '__main__':
    unittest.main()
