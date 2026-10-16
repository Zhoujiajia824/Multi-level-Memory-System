"""Relate the command before a physics tick to its actual measured response."""
import json
import math
from unittest.mock import Mock

import carla
import pytest

from carla_bridge.metrics.collector import MetricsCollector
from carla_bridge.tests.test_route_acceptance import fake_loop


def audited_loop(tmp_path):
    loop = fake_loop()
    loop._control_audit_path = tmp_path / "control.jsonl"
    ego = loop.scenario.ego_vehicle
    ego.get_control.return_value = carla.VehicleControl()
    loop.metrics = MetricsCollector(None, None)
    loop.metrics._check_red_light = Mock()
    loop.metrics._check_wrong_way = Mock()
    return loop, ego


def records(loop):
    return [json.loads(line) for line in loop._control_audit_path.read_text(encoding="utf-8").splitlines()]


def test_command_precedes_measured_tick_and_copies_tracker_snapshot(tmp_path):
    loop, ego = audited_loop(tmp_path)
    loop._elapsed = 2.0
    loop._pdm_plan_count = 7
    loop._semantic_frame_id = "frame_1"
    loop.tracker.last_control_diagnostics = {"target_speed_mps": 2.0, "reason": "speed_tracking"}
    command = carla.VehicleControl(brake=.2)
    loop._apply_control(ego, command)
    loop.tracker.last_control_diagnostics["target_speed_mps"] = 5.0
    ego.get_control.return_value = command
    ego.get_transform.return_value = carla.Transform(rotation=carla.Rotation(yaw=90))
    ego.get_acceleration.return_value = carla.Vector3D(x=3, y=-4)
    loop._elapsed = 2.1
    loop._record_tick(ego)
    audit = records(loop)[0]
    assert audit["sim_time_s"] == 2.1
    assert audit["command"]["sim_time_s"] == 2.0
    assert audit["command"]["planner_sequence"] == 7
    assert audit["command"]["tracking"]["target_speed_mps"] == 2.0
    assert audit["applied_control"]["brake"] == pytest.approx(.2)
    assert audit["measured"]["yaw_rad"] == pytest.approx(math.pi / 2)
    assert audit["measured"]["acceleration_longitudinal_mps2"] == pytest.approx(-4)
    assert loop.metrics.summary()["max_longitudinal_deceleration_mps2"] == 4


def test_direct_brake_cannot_be_misattributed_to_old_tracker_target(tmp_path):
    loop, ego = audited_loop(tmp_path)
    loop.tracker.last_control_diagnostics = {"target_speed_mps": 5, "reason": "speed_tracking"}
    loop._apply_control(ego, carla.VehicleControl(throttle=.5))
    loop._apply_brake()
    ego.get_control.return_value = ego.apply_control.call_args[0][0]
    loop._elapsed = .1
    loop._record_tick(ego)
    audit = records(loop)[0]
    assert audit["command"]["source"] == "direct_brake"
    assert audit["command"]["tracking"] is None
    assert audit["applied_control"]["brake"] == 1
    assert audit["applied_control"]["throttle"] == 0


def test_first_observation_does_not_invent_a_tracker_command(tmp_path):
    loop, ego = audited_loop(tmp_path)
    loop._elapsed = .1
    loop._record_tick(ego)
    assert records(loop)[0]["command"] is None


def test_control_audit_failure_brakes_before_propagating(tmp_path):
    loop, ego = audited_loop(tmp_path)
    loop._control_audit_path.mkdir()  # Cannot open a directory as a JSONL file.
    with pytest.raises(OSError):
        loop._record_tick(ego)
    assert ego.apply_control.call_args[0][0].brake == 1


def test_full_cycle_records_previous_command_and_final_measured_tick(tmp_path):
    loop, ego = audited_loop(tmp_path)

    def apply(control):
        ego.get_control.return_value = control

    ego.apply_control.side_effect = apply
    loop._replan_cycle()
    audit = records(loop)
    assert len(audit) == 10  # Duration guard at 1 s, including the first capture tick.
    assert audit[0]["command"] is None
    for row in audit[1:]:
        assert row["sim_time_s"] - row["command"]["sim_time_s"] == pytest.approx(.1)
        for key in ("throttle", "brake", "steer"):
            assert row["applied_control"][key] == row["command"][key]
    assert audit[-1]["sim_time_s"] == pytest.approx(1)
    # The final timeout brake is issued after the last measured response. It
    # must not replace that row's actual preceding command in the report.
    assert loop._last_control_command["source"] == "direct_brake"
    assert audit[-1]["command"]["source"] == "tracker"


def test_nonfinite_measurement_does_not_write_invalid_json(tmp_path):
    loop, ego = audited_loop(tmp_path)
    ego.get_acceleration.return_value = carla.Vector3D(x=float("nan"))
    with pytest.raises(ValueError):
        loop._record_tick(ego)
    assert ego.apply_control.call_args[0][0].brake == 1
    assert loop._control_audit_path.read_text(encoding="utf-8") == ""
