"""MPC runtime wiring, source identity and optimized input execution contracts."""
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from carla_bridge.inference_mode import model_inference_enabled
from carla_bridge.planning.mpc_tracking import override_mpc_tracking, TRACKING_SOURCE
from carla_bridge.control.steering_model import SteeringModel
from carla_bridge.tests.test_model_inference_mode import trajectory
from carla_bridge.tests.test_route_acceptance import fake_loop


def profile():
    return {"mpc_tracking": {"active": True, "source": TRACKING_SOURCE,
        "issued_at_s": 1., "expires_at_s": 1.3, "sample_interval_s": .2,
        "bicycle_angles_rad": [.1, -.1, 0.]}}


def test_mpc_model_off_is_legal_without_loading_pdm():
    planning = {"backend": "mpc", "model_inference": False, "semantic_influence": False}
    assert not model_inference_enabled(planning, "memory_off")
    with pytest.raises(ValueError, match="planning-memory full"):
        model_inference_enabled(planning, "memory_on")
    assert not model_inference_enabled({**planning, "planning_memory": "full"}, "memory_on")


def test_mpc_dependency_check_is_independent_in_a_fresh_process():
    import json
    import subprocess
    import sys
    code = r'''
import builtins, json, sys
sys.path[:] = json.loads(sys.argv[1])
original = builtins.__import__
def isolated(name, *args, **kwargs):
    if name.split('.')[0] in {'nuplan', 'tuplan_garage', 'carla'}:
        raise AssertionError('MPC dependency check imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = isolated
from carla_bridge.planning.mpc_planner import validate_dependencies
result = validate_dependencies()
assert result['upstream_pdm_used'] is False
assert result['geometry_operation_probe'].endswith('passed')
print(result['backend'])
'''
    process = subprocess.run([sys.executable, '-c', code, json.dumps(sys.path)],
        text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == 'mpc'


@pytest.mark.parametrize("override, message", [
    ({"replan_interval_s": .4}, "fresh optimized controls"),
    ({"mpc": {"horizon_steps": 5}}, "cover the semantic execution interval"),
    ({"mpc": {"dt": .05, "horizon_steps": 80}}, "integer multiple of the physics tick"),
    ({"mpc": {"dt": .15, "horizon_steps": 30}}, "integer multiple of the physics tick"),
    ({"replan_interval_s": .15}, "integer multiple of the physics tick"),
])
def test_mpc_invalid_runtime_timing_fails_before_connecting(override, message):
    from carla_bridge.closed_loop import ClosedLoop
    from src.vla_memory.common.config import Config
    config = Config(overrides={"carla": {"fixed_delta_seconds": .1, "replan_interval_s": 3.,
        "planning": {"backend": "mpc", "model_inference": False, "semantic_influence": False,
                     **override}}})
    with pytest.raises(ValueError, match=message):
        ClosedLoop(config, "unused.yaml", mode="memory_off")


def test_mpc_input_uses_right_positive_first_input_and_preserves_pid():
    control = NS(throttle=.3, brake=0., steer=.8)
    model = SteeringModel(.6)
    result, audit = override_mpc_tracking(control, profile(), now_s=1., speed_mps=3., steering_model=model)
    assert audit["applied"] and audit["profile_sample_index"] == 0
    assert result.steer == pytest.approx(.1/.6)
    assert result.throttle == control.throttle and result.brake == control.brake
    assert control.steer == .8
    result, audit = override_mpc_tracking(control, profile(), now_s=1.2, speed_mps=3., steering_model=model)
    assert result.steer < 0 and audit["profile_sample_index"] == 1


@pytest.mark.parametrize("time", [.9, 1.3, 2., float('nan')])
def test_mpc_stale_or_invalid_profile_cannot_fall_back_to_unoptimized_steering(time):
    result, audit = override_mpc_tracking(NS(throttle=.8, brake=0., steer=.4), profile(),
        now_s=time, speed_mps=3., steering_model=SteeringModel(.6))
    assert not audit["applied"] and audit["failsafe_brake"]
    assert result.throttle == 0 and result.brake == 1 and result.steer == 0


def test_mpc_merge_reports_actual_backend_and_speed_profile():
    loop = fake_loop()
    loop.planner_backend = "mpc"
    plan = trajectory()
    merged = loop._merge_plan({}, None, plan)
    assert merged["trajectory_source"] == "mpc"
    assert merged["planning_context"]["backend"] == "mpc"
    assert loop._validate_decision(merged, {"ego_state": {"speed": 5.}})["accepted"]


def test_mpc_numeric_refresh_runs_between_model_calls(tmp_path):
    loop = fake_loop()
    loop.planner_backend = "mpc"
    loop.model_inference = False
    loop.config.data['output_dir'] = tmp_path
    loop.pdm_planner = Mock()
    loop.pdm_planner.plan.return_value = deepcopy(trajectory())
    loop._refresh_pdm_control(loop.scenario.ego_vehicle.get_transform(), 0., 5.)
    loop.pdm_planner.plan.assert_called_once()
    assert loop.tracker.set_trajectory.call_args.kwargs['follow_speed_profile'] is True


def test_mpc_diagnostics_and_control_log_use_optimized_inputs():
    loop = fake_loop()
    loop.planner_backend = "mpc"
    loop._elapsed = 1.
    feedback = Mock()
    loop.pdm_planner = NS(steering_model=SteeringModel(.6), record_control_execution=feedback)
    loop._last_plan_diagnostics = profile()
    import carla
    loop._apply_control(loop.scenario.ego_vehicle, carla.VehicleControl(throttle=.3, steer=.8))
    assert loop.scenario.ego_vehicle.apply_control.call_args.args[0].steer == pytest.approx(.1/.6)
    assert loop._last_control_command['mpc_tracking']['applied'] is True
    assert loop._last_control_command['native_tracking'] is None
    feedback.assert_called_once_with(loop._last_plan_diagnostics, loop._last_control_command['mpc_tracking'],
                                     now_s=1., source='tracker',
                                     tracker_diagnostics=loop.tracker.last_control_diagnostics)


def test_mpc_failed_control_dispatch_does_not_record_an_executed_input():
    loop = fake_loop()
    loop.planner_backend = "mpc"
    loop._elapsed = 1.
    feedback = Mock()
    loop.pdm_planner = NS(steering_model=SteeringModel(.6), record_control_execution=feedback)
    loop._last_plan_diagnostics = profile()
    loop.scenario.ego_vehicle.apply_control.side_effect = RuntimeError('dispatch failed')
    import carla
    with pytest.raises(RuntimeError, match='dispatch failed'):
        loop._apply_control(loop.scenario.ego_vehicle, carla.VehicleControl(throttle=.3))
    feedback.assert_not_called()
