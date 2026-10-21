"""Vectorized physical OBB clearance and analytic ego-pose derivatives.

This module does not apply collision margins, uncertainty or actor filtering.
Callers retain those policies separately. All leading dimensions broadcast, so
one call can check a prediction horizon or a horizon-by-actor batch. There are
no Python loops over prediction samples or collision pairs.
"""
from __future__ import annotations

import numpy as np


_CORNER_SIGNS = np.array([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.]])


def _vector(value, dimensions, name):
    result = np.asarray(value, dtype=float)
    if result.ndim < 1 or result.shape[-1] != dimensions or not np.isfinite(result).all():
        raise ValueError("%s must have finite trailing dimension %d" % (name, dimensions))
    return result


def _dot(vector, axes):
    return np.einsum("...i,...ki->...k", vector, axes)


def _pick(values, indices):
    return np.take_along_axis(values, indices[..., None, None], axis=-2)[..., 0, :]


def rectangle_signed_distance(ego_poses, obstacle_centers, obstacle_yaws,
                              ego_half_extents, obstacle_half_extents, *,
                              rear_axle_to_center=0.):
    """Return physical signed clearance and d(clearance)/d(ego x,y,yaw).

    ``ego_poses[..., 3]`` are rear-axle x/y/yaw poses; with the default zero
    offset they are body-center poses. ``rear_axle_to_center`` is a longitudinal
    offset in the ego frame. A caller with lateral bbox offset or relative bbox
    yaw can first transform to its body-center pose and apply the chain rule.
    Half extents have trailing dimension two, in each body's local x/y frame.
    Obstacle poses are held fixed when differentiating the ego pose.

    Disjoint boxes use exact Euclidean closest-feature distance. Overlapping
    boxes use negative minimum translation depth from all four SAT axes; this
    also detects crossed edges when no vertex lies inside the other box.
    Clearance is continuous and piecewise differentiable. At an exact feature
    or axis tie a deterministic active branch supplies a subgradient,
    rather than claiming a unique classical derivative. No margin is inflated
    into either body, and no smooth maximum can turn overlap into clearance.
    """
    poses = _vector(ego_poses, 3, "ego_poses")
    centers = _vector(obstacle_centers, 2, "obstacle_centers")
    ego_half = _vector(ego_half_extents, 2, "ego_half_extents")
    other_half = _vector(obstacle_half_extents, 2, "obstacle_half_extents")
    yaw = np.asarray(obstacle_yaws, dtype=float)
    offset = np.asarray(rear_axle_to_center, dtype=float)
    if not np.isfinite(yaw).all() or not np.isfinite(offset).all():
        raise ValueError("obstacle yaws and rear axle offsets must be finite")
    if np.any(ego_half <= 0) or np.any(other_half <= 0):
        raise ValueError("rectangle half extents must be positive")
    shape = np.broadcast_shapes(poses.shape[:-1], centers.shape[:-1], yaw.shape,
                                ego_half.shape[:-1], other_half.shape[:-1], offset.shape)
    poses = np.broadcast_to(poses, shape + (3,))
    centers = np.broadcast_to(centers, shape + (2,))
    yaw, offset = np.broadcast_to(yaw, shape), np.broadcast_to(offset, shape)
    ego_half = np.broadcast_to(ego_half, shape + (2,))
    other_half = np.broadcast_to(other_half, shape + (2,))

    e = np.stack((np.cos(poses[..., 2]), np.sin(poses[..., 2])), axis=-1)
    f = np.stack((-e[..., 1], e[..., 0]), axis=-1)
    oe = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
    of = np.stack((-oe[..., 1], oe[..., 0]), axis=-1)
    ego_centers = poses[..., :2] + offset[..., None] * e
    delta = centers - ego_centers
    axes = np.stack((e, f, oe, of), axis=-2)
    zeros = np.zeros_like(e)
    axis_derivatives = np.stack((f, -e, zeros, zeros), axis=-2)
    projection = _dot(delta, axes)
    edot, fdot, oedot, ofdot = _dot(e, axes), _dot(f, axes), _dot(oe, axes), _dot(of, axes)
    support_ego = ego_half[..., :1] * np.abs(edot) + ego_half[..., 1:] * np.abs(fdot)
    support_other = other_half[..., :1] * np.abs(oedot) + other_half[..., 1:] * np.abs(ofdot)
    gaps = np.abs(projection) - support_ego - support_other
    yaw_derivative = np.sign(projection) * (
        -offset[..., None] * fdot + _dot(delta, axis_derivatives))
    yaw_derivative -= (
        ego_half[..., :1] * np.sign(edot) * (fdot + _dot(e, axis_derivatives)) +
        ego_half[..., 1:] * np.sign(fdot) * (-edot + _dot(f, axis_derivatives)))
    yaw_derivative -= (
        other_half[..., :1] * np.sign(oedot) * _dot(oe, axis_derivatives) +
        other_half[..., 1:] * np.sign(ofdot) * _dot(of, axis_derivatives))
    gap_gradients = np.concatenate((-np.sign(projection)[..., None] * axes,
                                     yaw_derivative[..., None]), axis=-1)
    active_axis = np.argmax(gaps, axis=-1)
    penetration = np.take_along_axis(gaps, active_axis[..., None], axis=-1)[..., 0]
    penetration_gradient = _pick(gap_gradients, active_axis)

    ego_vertices = (ego_centers[..., None, :] +
        (ego_half[..., :1] * _CORNER_SIGNS[:, 0])[..., None] * e[..., None, :] +
        (ego_half[..., 1:] * _CORNER_SIGNS[:, 1])[..., None] * f[..., None, :])
    other_vertices = (centers[..., None, :] +
        (other_half[..., :1] * _CORNER_SIGNS[:, 0])[..., None] * oe[..., None, :] +
        (other_half[..., 1:] * _CORNER_SIGNS[:, 1])[..., None] * of[..., None, :])
    ego_to_other = ego_vertices - centers[..., None, :]
    local_other = np.stack((np.sum(ego_to_other * oe[..., None, :], axis=-1),
                            np.sum(ego_to_other * of[..., None, :], axis=-1)), axis=-1)
    clipped_other = np.clip(local_other, -other_half[..., None, :], other_half[..., None, :])
    other_witnesses = (centers[..., None, :] + clipped_other[..., :1] * oe[..., None, :] +
                       clipped_other[..., 1:] * of[..., None, :])
    other_to_ego = other_vertices - ego_centers[..., None, :]
    local_ego = np.stack((np.sum(other_to_ego * e[..., None, :], axis=-1),
                          np.sum(other_to_ego * f[..., None, :], axis=-1)), axis=-1)
    clipped_ego = np.clip(local_ego, -ego_half[..., None, :], ego_half[..., None, :])
    ego_witnesses = (ego_centers[..., None, :] + clipped_ego[..., :1] * e[..., None, :] +
                     clipped_ego[..., 1:] * f[..., None, :])
    all_ego = np.concatenate((ego_vertices, ego_witnesses), axis=-2)
    all_other = np.concatenate((other_witnesses, other_vertices), axis=-2)
    differences = all_ego - all_other
    squared_distances = np.sum(differences * differences, axis=-1)
    closest = np.argmin(squared_distances, axis=-1)
    closest_difference = _pick(differences, closest)
    distance = np.linalg.norm(closest_difference, axis=-1)
    normal = np.divide(closest_difference, distance[..., None],
        out=np.zeros_like(closest_difference), where=distance[..., None] > 0.)
    lever = _pick(all_ego, closest) - poses[..., :2]
    distance_yaw_derivative = -normal[..., 0] * lever[..., 1] + normal[..., 1] * lever[..., 0]
    distance_gradient = np.concatenate((normal, distance_yaw_derivative[..., None]), axis=-1)
    overlap = penetration <= 0.
    return (np.where(overlap, penetration, distance),
            np.where(overlap[..., None], penetration_gradient, distance_gradient))
