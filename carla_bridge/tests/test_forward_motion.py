"""Forward-gear zero crossings, including the measured R05 braking impulse."""
from importlib.util import find_spec

import numpy as np
import pytest

from carla_bridge.planning.forward_motion import build_forward_motion_model
from carla_bridge.tests.test_pdm_planner import make_planner, semantic

pytestmark = pytest.mark.skipif(find_spec("tuplan_garage") is None or find_spec("nuplan") is None,
                                reason="optional pinned PDM dependencies are not installed")


def models():
    from nuplan.common.actor_state.state_representation import TimeDuration
    planner = make_planner(speed=2.8, allow_change=False)
    native = planner.api.BatchKinematicBicycleModel(vehicle=planner.vehicle_parameters,
                                                  max_steering_angle=planner.max_steer)
    forward = build_forward_motion_model(planner.api, planner.vehicle_parameters, planner.max_steer)
    return planner, native, forward, TimeDuration.from_s(.1)


def test_zero_crossing_integrates_to_stop_then_holds_without_mutating_input():
    planner, native, forward, dt = models()
    si = planner.api.StateIndex
    states = np.zeros((2, si.size()))
    states[:, si.VELOCITY_X] = [1., 6.]
    states[:, si.ACCELERATION_X] = -20.
    commands = np.zeros((2, 2))
    commands[:, 0] = -20.
    before = states.copy(), commands.copy()
    baseline = native.propagate_state(states, commands, dt)
    result = forward.propagate_state(states, commands, dt)
    assert baseline[0, si.VELOCITY_X] == pytest.approx(-1.)
    assert result[0, si.VELOCITY_X] == 0.
    assert result[0, si.X] == pytest.approx(.025)  # 1^2/(2*20), not an entire tick.
    np.testing.assert_array_equal(result[1], baseline[1])
    commands[0, 0] = -20.
    held = forward.propagate_state(result[:1], commands[:1], dt)
    assert held[0, si.VELOCITY_X] == held[0, si.ACCELERATION_X] == 0.
    assert held[0, si.X] == result[0, si.X]
    np.testing.assert_array_equal(states, before[0])
    np.testing.assert_array_equal(commands, before[1])


@pytest.mark.parametrize("velocity,acceleration,command", [(6., 0., 2.), (6., -2., -4.), (0., 0., 2.)])
def test_normal_forward_steps_remain_bit_exact_native(velocity, acceleration, command):
    planner, native, forward, dt = models()
    si = planner.api.StateIndex
    state = np.zeros((1, si.size()))
    state[0, si.HEADING], state[0, si.STEERING_ANGLE] = .3, .08
    state[0, si.VELOCITY_X], state[0, si.ACCELERATION_X] = velocity, acceleration
    commands = np.array([[command, -.02]])
    np.testing.assert_array_equal(forward.propagate_state(state, commands, dt),
                                  native.propagate_state(state, commands, dt))


def test_r05_measured_braking_impulse_cannot_create_reverse_rollouts():
    import carla
    from carla_bridge.control.trajectory_validation import TrajectoryValidator
    planner = make_planner(speed=2.833974, allow_change=False)
    # From the actual 8.8 s physical regression tick, rather than an invented
    # moderate braking value which never exercises the native zero crossing.
    planner.ego.get_acceleration = lambda: carla.Vector3D(x=-23.491500854492188)
    result = planner.plan(semantic("YIELD"))
    si = planner.api.StateIndex
    assert planner.scorer._states[:, :, si.VELOCITY_X].min() >= 0.
    assert np.diff(planner.scorer._states[:, :, si.X], axis=1).min() >= -1e-8
    checked = TrajectoryValidator().validate(result["trajectory"], result["target_speed"],
        2.833974, 3., use_speed_profile=True)
    assert checked.accepted, checked.to_dict()
    reference = planner.generator._state_array.copy()
    state, _ = planner._ego_state(2_000_000)
    planner.simulator._motion_model = planner.api.BatchKinematicBicycleModel(
        vehicle=planner.vehicle_parameters, max_steering_angle=planner.max_steer)
    raw = planner.simulator.simulate_proposals(reference, state)
    assert raw[:, :, si.VELOCITY_X].min() < -.1
