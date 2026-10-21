"""Independent physical geometry and active-branch derivative regressions."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import MultiPoint, Point, Polygon

from carla_bridge.planning.mpc_geometry import rectangle_signed_distance


SIGNS = np.array([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.]])


def polygon(pose, half, rear_offset=0.):
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    center = np.asarray(pose[:2]) + rear_offset * rotation[:, 0]
    return Polygon(center + (SIGNS * half) @ rotation.T)


def independent_signed_distance(ego, other):
    """Minkowski hull distance also checks penetration magnitude independently."""
    first = np.asarray(ego.exterior.coords[:-1])
    second = np.asarray(other.exterior.coords[:-1])
    differences = (first[:, None, :] - second[None, :, :]).reshape(-1, 2)
    hull = MultiPoint(differences).convex_hull
    origin = Point(0., 0.)
    if hull.contains(origin):
        return -hull.boundary.distance(origin)
    return hull.distance(origin)


def test_vectorized_1500_random_poses_match_independent_shapely_and_central_differences():
    rng = np.random.default_rng(164)
    # Keep one row per random draw so the original scalar-prototype seed is
    # reproducible. The product geometry evaluates the complete batch at once.
    samples = rng.uniform(size=(1500, 6))
    ego = np.c_[samples[:, :2] * 14. - 7., samples[:, 2] * 2.*np.pi - np.pi]
    centers = samples[:, 3:5] * 14. - 7.
    yaws = samples[:, 5] * 2.*np.pi - np.pi
    ego_half, other_half, offset = np.array([2.4, 1.]), np.array([2.1, .9]), 1.4
    values, gradients = rectangle_signed_distance(
        ego, centers, yaws, ego_half, other_half, rear_axle_to_center=offset)
    oracle = np.array([
        independent_signed_distance(polygon(state, ego_half, offset),
                                    polygon(np.r_[center, yaw], other_half))
        for state, center, yaw in zip(ego, centers, yaws)])
    np.testing.assert_allclose(values, oracle, rtol=1e-11, atol=1e-11)
    assert np.count_nonzero(values < 0.) == 265
    assert np.count_nonzero(values > 0.) == 1235
    numeric = np.empty_like(gradients)
    for axis in range(3):
        change = np.eye(3)[axis] * 1e-6
        plus = rectangle_signed_distance(ego + change, centers, yaws, ego_half,
                                         other_half, rear_axle_to_center=offset)[0]
        minus = rectangle_signed_distance(ego - change, centers, yaws, ego_half,
                                          other_half, rear_axle_to_center=offset)[0]
        numeric[:, axis] = (plus-minus) / 2e-6
    np.testing.assert_allclose(gradients, numeric, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("ego,center,yaw,ego_half,other_half,expected", [
    ([0., 0., 0.], [3., 3.], 0., [1., 1.], [1., 1.], math.sqrt(2.)),
    ([0., 0., 0.], [0., 0.], math.pi/2., [3., .3], [3., .3], -3.3),
    ([0., 0., 0.], [0., 0.], 0., [1., .5], [3., 2.], -2.5),
    ([0., 0., 0.], [5.5, 0.], 0., [3., .3], [3., .3], -.5),
    ([0., 0., 0.], [0., 2.], 0., [1., 1.], [1., 1.], 0.),
    ([0., 0., 0.], [0., 4.], 0., [2.4, 1.], [2.1, .9], 2.1),
    ([0., 0., 0.], [4.9, 0.], 0., [2.4, 1.], [2.1, .9], .4),
])
def test_full_body_crossing_containment_parallel_queue_corner_and_touching(
        ego, center, yaw, ego_half, other_half, expected):
    value, gradient = rectangle_signed_distance(ego, center, yaw, ego_half, other_half)
    assert value == pytest.approx(expected, abs=1e-12)
    assert np.isfinite(gradient).all()
    assert value == pytest.approx(independent_signed_distance(
        polygon(ego, ego_half), polygon(np.r_[center, yaw], other_half)), abs=1e-12)
    if yaw == math.pi/2.:
        # An eight-point inside-only test would miss this true edge crossing.
        first, second = polygon(ego, ego_half), polygon(np.r_[center, yaw], other_half)
        assert first.intersects(second)
        assert not any(second.contains(Point(vertex)) for vertex in first.exterior.coords[:-1])
        assert not any(first.contains(Point(vertex)) for vertex in second.exterior.coords[:-1])


def test_corner_midpoint_collisions_cannot_hide_between_ego_disc_centers():
    # Physical overlap at a front corner and at the middle of a long body.
    poses = np.array([[0., 0., .37], [0., 0., -.23]])
    rotation = np.array([[math.cos(.37), -math.sin(.37)],
                         [math.sin(.37), math.cos(.37)]])
    centers = np.array([rotation @ np.array([2.35, .95]), [0., 0.]])
    values, _ = rectangle_signed_distance(poses, centers, [.11, .19], [2.4, 1.], [.25, .25])
    assert np.all(values < 0.)


def test_feature_switch_is_continuous_and_gradient_correct_on_both_sides():
    yaws = np.array([-1e-4, -1e-6, 0., 1e-6, 1e-4])
    ego = np.c_[np.zeros(5), np.zeros(5), yaws]
    values, gradients = rectangle_signed_distance(ego, [0., 4.], 0., [2.4, 1.], [2.1, .9])
    np.testing.assert_allclose(values, values[::-1], atol=1e-12)
    assert abs(values[2]-values[1]) < 3e-6
    assert np.isfinite(gradients).all()
    for index in [0, 4]:
        change = np.array([0., 0., 1e-7])
        plus = rectangle_signed_distance(ego[index]+change, [0., 4.], 0., [2.4, 1.], [2.1, .9])[0]
        minus = rectangle_signed_distance(ego[index]-change, [0., 4.], 0., [2.4, 1.], [2.1, .9])[0]
        assert gradients[index, 2] == pytest.approx((plus-minus)/2e-7, abs=1e-7)
    # At the exact non-smooth switch a unique central derivative is not claimed.
    assert gradients[0, 2] > 0. and gradients[4, 2] < 0.


def test_horizon_by_actor_broadcast_matches_individual_calls_without_mutation():
    rng = np.random.default_rng(321)
    poses = rng.normal(size=(40, 1, 3))
    centers = rng.normal(size=(1, 30, 2)) * 5.
    yaws = rng.normal(size=30)
    half = rng.uniform(.3, 2., size=(30, 2))
    offsets = np.linspace(.2, 1.8, 40)[:, None]
    original = poses.copy()
    values, gradients = rectangle_signed_distance(
        poses, centers, yaws, [2.4, 1.], half, rear_axle_to_center=offsets)
    assert values.shape == (40, 30) and gradients.shape == (40, 30, 3)
    for i, j in [(0, 0), (9, 11), (39, 29)]:
        scalar, gradient = rectangle_signed_distance(
            poses[i, 0], centers[0, j], yaws[j], [2.4, 1.], half[j],
            rear_axle_to_center=offsets[i, 0])
        assert values[i, j] == pytest.approx(scalar, abs=1e-12)
        np.testing.assert_allclose(gradients[i, j], gradient, atol=1e-12)
    np.testing.assert_array_equal(poses, original)


def test_product_geometry_has_no_per_sample_python_loops():
    # This is a performance contract: 2N samples x all actors remains vectorized.
    source = Path(__file__).parents[1] / "planning" / "mpc_geometry.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
                   for node in ast.walk(tree))


@pytest.mark.parametrize("changes", [
    {"ego_poses": [0., 0.]}, {"ego_poses": [0., float("nan"), 0.]},
    {"obstacle_centers": [0., float("inf")]}, {"obstacle_yaws": float("nan")},
    {"ego_half_extents": [0., 1.]}, {"obstacle_half_extents": [1., -1.]},
    {"rear_axle_to_center": float("inf")},
    {"ego_poses": np.zeros((2, 3)), "obstacle_centers": np.zeros((3, 2))},
])
def test_malformed_nonfinite_degenerate_and_incompatible_inputs_rejected(changes):
    arguments = dict(ego_poses=[0., 0., 0.], obstacle_centers=[4., 3.],
                     obstacle_yaws=.2, ego_half_extents=[2.4, 1.],
                     obstacle_half_extents=[2., 1.])
    arguments.update(changes)
    with pytest.raises(ValueError):
        rectangle_signed_distance(**arguments)
