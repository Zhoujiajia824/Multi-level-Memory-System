"""Numerical checks of the production Frenet potential kernel; no simulator/model.

Run: python scripts/check_frenet_apf.py
The checks address geometry, gradients, objective residuals and finite support.
They are not a claim about closed-loop collision avoidance in CARLA.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nine_grid.lane_potential import LanePotentialGeometry
from nine_grid.geometry import project_polyline
from nine_grid.goals import first_arrival_residual


REPULSION = {"longitudinal_m": 20., "lateral_m": 6., "peak": 8.}
ATTRACTION = {"weight": 3., "longitudinal_scale_m": 30.,
              "lateral_scale_m": 15., "shift": 6.}


def straight(domain=None):
    ref = np.column_stack((np.linspace(-80., 100., 91), np.zeros(91)))
    return LanePotentialGeometry(ref, box(-79., -8., 99., 8.) if domain is None else domain)


def bend():
    theta = np.linspace(-.7, 1.6, 181)
    ref = np.column_stack((30. * np.sin(theta), 30. * (1. - np.cos(theta))))
    return LanePotentialGeometry(ref, LineString(ref).buffer(5.25, cap_style=2))


def difference(fun, xy, epsilon=1e-5):
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([(np.asarray(fun(xy + np.eye(2)[i] * epsilon)) -
                             np.asarray(fun(xy - np.eye(2)[i] * epsilon))) / (2 * epsilon)
                            for i in range(2)])


class FrenetPotentialChecks(unittest.TestCase):
    def test_finite_ellipse_support_and_gradual_decay(self):
        geometry = straight()
        xy = np.column_stack(([0., 5., 10., 15., 20., 21.], np.zeros(6)))
        field = geometry.repulsion(xy, [0., 0.], REPULSION)
        np.testing.assert_allclose(field["energy"], [8., 7.171875, 4., .828125, 0., 0.], atol=1e-7)
        self.assertTrue(np.all(np.diff(field["energy"]) <= 0))
        side = geometry.repulsion([[0., 6.], [0., 6.5]], [0., 0.], REPULSION)
        np.testing.assert_allclose(side["energy"], 0., atol=1e-10)
        # The peak and support are smooth. Force is not monotone near a smooth peak.
        edge = geometry.repulsion([[0., 0.], [20., 0.]], [0., 0.], REPULSION)
        np.testing.assert_allclose(edge["gradient"], 0., atol=1e-8)

    def test_repulsion_gradient_and_objective_residual(self):
        for geometry in (straight(), bend()):
            xy = geometry.sample(np.array([24., 30., 35., 39.]), np.array([.4, -1.2, 2., -.8]))
            source = geometry.sample(32., .1)
            result = geometry.repulsion(xy, source, REPULSION)
            numerical = difference(lambda p: geometry.repulsion(p, source, REPULSION)["energy"], xy)
            np.testing.assert_allclose(result["gradient"], numerical, rtol=3e-4, atol=2e-5)
            np.testing.assert_allclose(result["residual"] ** 2, result["energy"], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(2 * result["residual"][:, None] * result["residual_jacobian"],
                                       result["gradient"], rtol=1e-9, atol=1e-9)

    def test_attraction_gradient_and_residual_shift(self):
        geometry = bend()
        xy = geometry.sample([20., 30., 38.], [.2, -.7, 1.])
        goal = geometry.sample(50., 0.)
        result = geometry.attraction(xy, goal, ATTRACTION)
        numerical = difference(lambda p: geometry.attraction(p, goal, ATTRACTION)["energy"], xy)
        np.testing.assert_allclose(result["gradient"], numerical, rtol=3e-4, atol=2e-5)
        np.testing.assert_allclose(np.sum(result["residual"] ** 2, axis=1) - ATTRACTION["shift"],
                                   result["energy"], rtol=1e-10, atol=1e-10)
        chain = 2 * np.einsum("ni,nij->nj", result["residual"], result["residual_jacobian"])
        np.testing.assert_allclose(chain, result["gradient"], rtol=1e-9, atol=1e-9)

    def test_projection_jacobian_through_turn(self):
        geometry = bend()
        xy = geometry.sample([10., 20., 30., 40., 50.], [.4, -.7, 1., -.5, .2])
        projected = geometry.project(xy)
        for coordinate, index in (("s", 0), ("d", 1)):
            numerical = difference(lambda p: geometry.project(p)[coordinate], xy)
            np.testing.assert_allclose(projected["jacobian"][:, index, :], numerical, rtol=3e-4, atol=2e-5)

    def test_attention_scales_height_not_support(self):
        geometry = straight()
        xy = np.array([[0., 0.], [5., 1.], [19., 0.], [20., 0.], [21., 0.]])
        base = geometry.repulsion(xy, [0., 0.], REPULSION)
        attended = geometry.repulsion(xy, [0., 0.], REPULSION, attention_gain=2.)
        np.testing.assert_allclose(attended["energy"], 2 * base["energy"], atol=1e-10)
        np.testing.assert_allclose(attended["gradient"], 2 * base["gradient"], atol=1e-10)
        np.testing.assert_allclose(attended["residual"], np.sqrt(2) * base["residual"], atol=1e-10)
        np.testing.assert_equal(attended["energy"] > 0, base["energy"] > 0)

    def test_body_size_independence_of_potential(self):
        # Production APF takes centers, never a physical footprint. Collision constraints
        # must still use the physical body; this check only covers potential energy.
        geometry = straight()
        small = {"position": [0., 0.], "length": 1., "width": .5}
        large = {"position": [0., 0.], "length": 15., "width": 4.}
        xy = [[5., .2], [10., 1.], [19., 0.]]
        np.testing.assert_equal(geometry.repulsion(xy, small["position"], REPULSION)["energy"],
                                geometry.repulsion(xy, large["position"], REPULSION)["energy"])

    def test_offroad_is_invalid_not_zero_energy(self):
        domain = box(-50., -1.75, 50., 5.25).difference(Point(10., 0.).buffer(.5))
        geometry = straight(domain)
        xy = [[0., 0.], [0., 3.5], [0., -3.5], [0., 7.], [10., 0.]]
        for result in (geometry.repulsion(xy, [0., 0.], REPULSION),
                       geometry.attraction(xy, [30., 0.], ATTRACTION)):
            np.testing.assert_equal(result["valid"], [True, True, False, False, False])
            self.assertTrue(np.all(np.isnan(result["energy"][2:])))

    def test_attraction_follows_navigation_tangent(self):
        geometry = bend()
        xy = geometry.sample([18., 24., 30., 36.], 0.)
        frame = geometry.frame([18., 24., 30., 36.])
        goal = geometry.sample(55., 0.)
        force = -geometry.attraction(xy, goal, ATTRACTION)["gradient"]
        along = np.sum(force * frame["tangent"], axis=1)
        normal = np.sum(force * frame["normal"], axis=1)
        self.assertTrue(np.all(along > 0))
        np.testing.assert_allclose(normal, 0., atol=1e-5)

    def test_nearby_wrong_route_branch_has_no_longitudinal_shortcut(self):
        # Two road legs only 5 m apart, but separated by over 50 m of route.
        arc = np.column_stack((15. + 2.5 * np.sin(np.linspace(0., np.pi, 41)),
                               2.5 - 2.5 * np.cos(np.linspace(0., np.pi, 41))))
        ref = np.vstack((np.column_stack((np.linspace(-30., 15., 61), np.zeros(61))),
                         arc[1:], np.column_stack((np.linspace(15., -30., 61), np.full(61, 5.)))[1:]))
        geometry = LanePotentialGeometry(ref, LineString(ref).buffer(1.7, cap_style=2))
        result = geometry.repulsion([[-10., 5.]], [-10., 0.], REPULSION)
        self.assertTrue(bool(result["valid"][0]))
        self.assertEqual(float(result["energy"][0]), 0.)

    def test_batch_sum_and_mean_match_individual_samples(self):
        geometry = bend()
        xy = geometry.sample(np.linspace(15., 42., 19), .5)
        source = geometry.sample(30., 0.)
        batch = geometry.repulsion(xy, source, REPULSION)["energy"]
        scalar = np.array([geometry.repulsion(p[None, :], source, REPULSION)["energy"][0] for p in xy])
        np.testing.assert_allclose(batch, scalar, atol=1e-10)
        self.assertAlmostEqual(float(np.sum(batch)), float(np.sum(scalar)), places=9)
        self.assertAlmostEqual(float(np.mean(batch)), float(np.mean(scalar)), places=9)

    def test_goal_first_arrival_chain_rule_on_curved_lane(self):
        geometry = bend()
        goal = geometry.sample(33., 1.3)
        goal_s = project_polyline(goal, geometry.reference)[0]
        root = np.diag([1. / 30., 1. / 15.])
        for samples, mode in (([23., 28., 36., 40.], "first_crossing"),
                              ([18., 23., 27., 30.], "terminal_approach")):
            positions = geometry.sample(samples, [.1, .3, .7, .8])
            residual, jac, evidence = first_arrival_residual(
                positions, geometry.reference, goal_s, goal, root, frenet_geometry=geometry)
            self.assertEqual(evidence["mode"], mode)
            self.assertEqual(evidence["potential_coordinates"], "lane_frenet")
            numerical = np.empty_like(jac)
            for index in np.ndindex(positions.shape):
                delta = np.zeros_like(positions)
                delta[index] = 1e-5
                high = first_arrival_residual(positions + delta, geometry.reference, goal_s,
                                             goal, root, frenet_geometry=geometry)[0]
                low = first_arrival_residual(positions - delta, geometry.reference, goal_s,
                                            goal, root, frenet_geometry=geometry)[0]
                numerical[:, index[0], index[1]] = (high - low) / 2e-5
            np.testing.assert_allclose(jac, numerical, rtol=5e-4, atol=2e-5)

    def test_cached_projection_matches_direct_field(self):
        geometry = bend()
        xy = geometry.sample([20., 26., 35., 42.], [0., 1., -2., 8.])
        source = geometry.sample(30., .4)
        direct = geometry.repulsion(xy, source, REPULSION, attention_gain=1.4)
        cached = geometry.repulsion_from_projection(geometry.project(xy),
                    geometry.project(source, require_domain=False), REPULSION, attention_gain=1.4)
        for key in ("energy", "gradient", "residual", "residual_jacobian", "valid"):
            np.testing.assert_allclose(cached[key], direct[key], rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_exact_projection_window_rejects_uncovered_point(self):
        geometry = straight()
        result = geometry.project([[2., 0.], [2., 0.]], anchors=[80., 80.], window_m=[1., 3.])
        np.testing.assert_equal(result["valid"], [False, True])
        self.assertLessEqual(abs(result["s"][0] - 80.), 1. + 1e-8)
        self.assertAlmostEqual(result["s"][1], 82., places=7)

    def test_arc_conversion_and_local_sequence_are_continuous(self):
        geometry = bend()
        chord_nodes = np.r_[0., np.cumsum(np.linalg.norm(np.diff(geometry.reference, axis=0), axis=1))]
        converted = geometry.chart_arc_from_polyline_s(chord_nodes)
        np.testing.assert_allclose(converted, geometry.arc_nodes, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(geometry.sample(converted), geometry.reference, atol=1e-8)
        ss = np.linspace(12., 45., 81)
        positions = geometry.sample(ss, .6)
        projected = geometry.project_sequence(positions, ss[0])
        self.assertTrue(np.all(projected["valid"]))
        np.testing.assert_allclose(projected["s"], ss, atol=1e-7)
        np.testing.assert_allclose(projected["d"], .6, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
