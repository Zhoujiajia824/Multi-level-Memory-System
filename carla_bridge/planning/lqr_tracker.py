"""Bind independent native PDM longitudinal and lateral preview horizons.

The upstream tracker shares one horizon between velocity and steering. A longer
steering preview must not also weaken its braking response. This adapter keeps
the native velocity-profile extraction, solvers and stopping controller, routes
their preview horizons independently, and bounds finite lateral velocity
forecasts at zero so braking does not predict reverse motion after stopping.
"""
from numbers import Integral
import math

import numpy as np


def speed_scaled_lateral_samples(speed_mps, wheelbase_m, *, discretization_time,
                                 min_samples, max_samples):
    """Quantize a two-wheelbase preview to the nearest supported sample.

    At rest the configured maximum is retained. At driving speed the preview
    covers approximately two measured wheelbases, bounded by the configured
    horizons. Half-sample ties retain the longer preview.
    """
    speed, wheelbase, dt = float(speed_mps), float(wheelbase_m), float(discretization_time)
    if not math.isfinite(speed) or speed < 0:
        raise ValueError("LQR preview speed_mps must be nonnegative and finite")
    if not math.isfinite(wheelbase) or wheelbase <= 0:
        raise ValueError("LQR preview wheelbase_m must be positive and finite")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("LQR preview discretization_time must be positive and finite")
    for name, value in (("min_samples", min_samples), ("max_samples", max_samples)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
            raise ValueError("LQR preview " + name + " must be an integer of at least two samples")
    if min_samples > max_samples:
        raise ValueError("LQR preview min_samples must not exceed max_samples")
    raw_steps = 2.0 * wheelbase / max(speed, 1e-3) / dt
    bounded = min(max_samples, max(min_samples, raw_steps))
    return min(max_samples, max(min_samples, int(math.floor(bounded + .5 + 1e-9))))


def build_lqr_tracker(api, vehicle, discretization_time, *,
                      longitudinal_tracking_horizon_samples=10,
                      lateral_tracking_horizon_samples=20):
    """Create a native tracker with explicit, independently bound previews.

    Imports stay lazy through ``api`` so the optional pinned PDM dependency is
    still checked by the planner's normal dependency loader.
    """
    dt = float(discretization_time)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("LQR discretization_time must be positive and finite")
    for name, value in (("longitudinal_tracking_horizon_samples", longitudinal_tracking_horizon_samples),
                        ("lateral_tracking_horizon_samples", lateral_tracking_horizon_samples)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
            raise ValueError(name + " must be an integer of at least two samples")
    longitudinal_steps = int(longitudinal_tracking_horizon_samples)
    lateral_steps = int(lateral_tracking_horizon_samples)
    native = api.BatchLQRTracker

    class AxisSeparatedBatchLQRTracker(native):
        def __init__(self):
            super().__init__(vehicle=vehicle, discretization_time=dt, tracking_horizon=lateral_steps)
            self.longitudinal_tracking_horizon_samples = longitudinal_steps
            self.lateral_tracking_horizon_samples = lateral_steps
            # Delegate to the unchanged native longitudinal solver with its
            # original B=N*dt, rather than just changing the reference index.
            self._longitudinal_tracker = native(
                vehicle=vehicle, discretization_time=dt, tracking_horizon=longitudinal_steps)

        def _compute_reference_velocity_and_curvature_profile(self, current_iteration):
            _, curvature_profiles = super()._compute_reference_velocity_and_curvature_profile(current_iteration)
            reference_index = min(current_iteration.index + longitudinal_steps, self._velocity_profile.shape[1] - 1)
            return self._velocity_profile[:, reference_index], curvature_profiles

        def _longitudinal_lqr_controller(self, initial_velocities, reference_velocities):
            return self._longitudinal_tracker._longitudinal_lqr_controller(initial_velocities, reference_velocities)

        def _lateral_lqr_controller(self, initial_lateral_state_vector, velocity_profile, curvature_profile):
            # Extrapolating a one-second braking command over the longer
            # steering preview must stop at zero, rather than inventing reverse
            # travel and reversing the lateral solver's steering direction.
            # Preserve NaN/Inf for existing invalid-candidate handling, and do
            # not mutate arrays shared with longitudinal control or proposals.
            forward_velocity = np.where(np.isfinite(velocity_profile) & (velocity_profile < 0),
                                        0.0, velocity_profile)
            return super()._lateral_lqr_controller(initial_lateral_state_vector,
                                                   forward_velocity, curvature_profile)

    return AxisSeparatedBatchLQRTracker()
