"""A forward braking preview must not reverse the lateral LQR's turn direction."""
from importlib.util import find_spec

import numpy as np
import pytest

from carla_bridge.planning.lqr_tracker import build_lqr_tracker
from carla_bridge.tests.test_pdm_planner import make_planner


pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


@pytest.fixture
def trackers():
    planner = make_planner(speed=4, allow_change=False)
    args = dict(vehicle=planner.vehicle_parameters, discretization_time=planner.dt)
    native = planner.api.BatchLQRTracker(**args, tracking_horizon=20)
    adapted = build_lqr_tracker(planner.api, **args)
    return planner, native, adapted


@pytest.mark.parametrize("direction", [-1, 1])
def test_braking_on_eight_metre_turn_preserves_native_forward_turn_direction(trackers, monkeypatch, direction):
    planner, native, adapted = trackers
    # Use the unchanged native H10 braking command. Extrapolating it over H20
    # invents reverse travel although the forward stopping proposal cannot reverse.
    initial_speed = np.array([4.])
    acceleration = adapted._longitudinal_lqr_controller(initial_speed, np.array([0.]))
    velocity = initial_speed[:, None] + acceleration[:, None] * np.arange(20)[None, :] * planner.dt
    original = velocity.copy()
    assert velocity[0, -1] < -2.5
    lateral_state = np.zeros((1, 3))
    curvature = np.full((1, 20), direction / 8.)
    forward_velocity = np.maximum(velocity, 0)
    expected = native._lateral_lqr_controller(lateral_state, forward_velocity, curvature)
    reverse_preview_result = native._lateral_lqr_controller(lateral_state, velocity, curvature)
    # This fixture explicitly reproduces the pre-fix steering reversal.
    assert reverse_preview_result[0] * direction < 0
    assert expected[0] * direction > 0

    native_method = planner.api.BatchLQRTracker._lateral_lqr_controller
    received = []

    def native_entry(self, initial, velocities, curvatures):
        received.append(velocities.copy())
        return native_method(self, initial, velocities, curvatures)

    monkeypatch.setattr(planner.api.BatchLQRTracker, "_lateral_lqr_controller", native_entry)
    actual = adapted._lateral_lqr_controller(lateral_state, velocity, curvature)
    assert len(received) == 1  # Continue delegating to the pinned native solver.
    assert received[0].min() >= 0, "negative future speed reached the forward lateral solver"
    np.testing.assert_array_equal(received[0], forward_velocity)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] * direction > 0
    np.testing.assert_array_equal(velocity, original)  # Never mutate shared proposal/cache arrays.


@pytest.mark.parametrize("batch", [0, 1, 5])
def test_normal_accelerating_preview_keeps_native_output_exactly(trackers, batch):
    _, native, adapted = trackers
    velocity = np.broadcast_to(np.linspace(.5, 5., 20), (batch, 20)).copy()
    original = velocity.copy()
    curvature = np.broadcast_to(np.linspace(-.05, .1, 20), (batch, 20)).copy()
    initial = np.tile([.2, -.03, .05], (batch, 1))
    expected = native._lateral_lqr_controller(initial, velocity, curvature)
    actual = adapted._lateral_lqr_controller(initial, velocity, curvature)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(velocity, original)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_preview_is_preserved_for_existing_invalid_candidate_handling(trackers, monkeypatch, invalid):
    planner, _, adapted = trackers
    velocity = np.ones((1, 20))
    velocity[0, 10] = invalid
    original = velocity.copy()
    received = []
    sentinel = np.array([7.])

    def native_entry(self, initial, velocities, curvatures):
        received.append(velocities.copy())
        return sentinel

    monkeypatch.setattr(planner.api.BatchLQRTracker, "_lateral_lqr_controller", native_entry)
    actual = adapted._lateral_lqr_controller(np.zeros((1, 3)), velocity, np.zeros((1, 20)))
    assert len(received) == 1
    # In particular, maximum(-inf, 0) must not turn invalid input into valid zero.
    np.testing.assert_array_equal(received[0], original)
    np.testing.assert_array_equal(velocity, original)
    assert actual is sentinel
