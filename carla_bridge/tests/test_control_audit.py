"""Relate the command before a physics tick to its actual measured response."""
import json
import math
from types import SimpleNamespace as NS
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
    loop.carla_client.get_world.return_value.get_snapshot.side_effect = lambda: NS(
        frame=1000 + round(loop._elapsed * 10),
        timestamp=NS(elapsed_seconds=500.0 + loop._elapsed, delta_seconds=.1))
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
    command = carla.VehicleControl(brake=.2, gear=-1, reverse=True, hand_brake=True)
    loop._apply_control(ego, command)
    loop.tracker.last_control_diagnostics["target_speed_mps"] = 5.0
    ego.get_control.return_value = command
    ego.get_transform.return_value = carla.Transform(rotation=carla.Rotation(yaw=90))
    ego.get_acceleration.return_value = carla.Vector3D(x=3, y=-4)
    ego.get_velocity.return_value = carla.Vector3D(x=3, y=-2)
    loop._elapsed = 2.1
    loop._record_tick(ego)
    audit = records(loop)[0]
    assert audit["sim_time_s"] == 2.1
    assert audit["command"]["sim_time_s"] == 2.0
    assert audit["command"]["planner_sequence"] == 7
    assert audit["command"]["tracking"]["target_speed_mps"] == 2.0
    assert audit["applied_control"]["brake"] == pytest.approx(.2)
    for key, value in (("gear", -1), ("reverse", True), ("hand_brake", True)):
        assert audit["command"][key] == value
        assert audit["applied_control"][key] == value
    assert audit["world_snapshot"] == {"frame": 1021, "elapsed_seconds": 502.1, "delta_seconds": .1}
    assert audit["measured"]["velocity_x"] == 3
    assert audit["measured"]["velocity_y"] == -2
    assert audit["measured"]["speed"] == pytest.approx(math.sqrt(13))
    assert audit["measured"]["speed_longitudinal_mps"] == pytest.approx(-2)
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
        for key in ("throttle", "brake", "steer", "gear", "reverse", "hand_brake"):
            assert row["applied_control"][key] == row["command"][key]
        assert row["world_snapshot"]["elapsed_seconds"] == pytest.approx(500 + row["sim_time_s"])
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


def test_server_clock_exposes_extra_tick_without_rewriting_scheduler_time(tmp_path):
    loop, ego = audited_loop(tmp_path)
    snapshot_reader = loop.carla_client.get_world.return_value.get_snapshot
    snapshot_reader.side_effect = [
        NS(frame=2001, timestamp=NS(elapsed_seconds=100.1, delta_seconds=.1)),
        NS(frame=2003, timestamp=NS(elapsed_seconds=100.3, delta_seconds=.1)),
    ]
    loop._elapsed = 1.1
    loop._record_tick(ego)
    loop._elapsed = 1.2
    loop._record_tick(ego)
    first, second = records(loop)
    assert second["sim_time_s"] - first["sim_time_s"] == pytest.approx(.1)
    assert second["world_snapshot"]["frame"] - first["world_snapshot"]["frame"] == 2
    assert second["world_snapshot"]["elapsed_seconds"] - first["world_snapshot"]["elapsed_seconds"] == pytest.approx(.2)
    assert snapshot_reader.call_count == 2
    loop.carla_client.tick.assert_not_called()


def test_snapshot_read_failure_keeps_measurements_and_records_reason(tmp_path):
    loop, ego = audited_loop(tmp_path)
    loop.carla_client.get_world.return_value.get_snapshot.side_effect = RuntimeError("snapshot unavailable")
    loop._record_tick(ego)
    row = records(loop)[0]
    assert row["world_snapshot"]["frame"] is None
    assert row["world_snapshot"]["errors"]["snapshot"] == "RuntimeError: snapshot unavailable"
    assert row["measured"]["velocity_x"] == 5


def test_control_context_records_units_and_observes_without_changing_physics(tmp_path):
    loop, ego = audited_loop(tmp_path)
    world = Mock()
    world.get_settings.return_value = NS(synchronous_mode=True, fixed_delta_seconds=.1,
        substepping=True, max_substeps=10, max_substep_delta_time=.01)
    ego.get_physics_control.return_value = NS(mass=1800.0, wheels=[
        NS(radius=35.0, max_brake_torque=1500.0), NS(radius=36.0, max_brake_torque=1200.0)])
    loop._write_control_context(world, ego)
    context = json.loads((tmp_path / "control_context.json").read_text(encoding="utf-8"))
    assert context["errors"] == {}
    assert context["world_settings"]["max_substeps"] == 10
    assert context["vehicle_physics"] == {"mass_kg": 1800.0, "wheels": [
        {"index": 0, "radius_cm": 35.0, "max_brake_torque_nm": 1500.0},
        {"index": 1, "radius_cm": 36.0, "max_brake_torque_nm": 1200.0}]}
    world.apply_settings.assert_not_called()
    world.tick.assert_not_called()
    ego.apply_physics_control.assert_not_called()
    world.get_settings.assert_called_once()
    ego.get_physics_control.assert_called_once()


def test_context_records_unavailable_fields_without_fabricated_mock_values(tmp_path):
    loop, ego = audited_loop(tmp_path)
    world = Mock()
    world.get_settings.side_effect = RuntimeError("settings unavailable")
    ego.get_physics_control.return_value = NS(mass=float("nan"), wheels=[NS(radius=35)])
    loop._write_control_context(world, ego)
    context = json.loads((tmp_path / "control_context.json").read_text(encoding="utf-8"))
    assert context["world_settings"] is None
    assert context["vehicle_physics"]["mass_kg"] is None
    assert context["vehicle_physics"]["wheels"][0]["max_brake_torque_nm"] is None
    assert "settings unavailable" in context["errors"]["world_settings"]
    assert "vehicle_physics.mass" in context["errors"]
    assert "vehicle_physics.wheels[0].max_brake_torque" in context["errors"]


def test_control_context_write_failure_brakes_and_propagates(tmp_path):
    loop, ego = audited_loop(tmp_path)
    (tmp_path / "control_context.json").mkdir()
    with pytest.raises(OSError):
        loop._write_control_context(Mock(), ego)
    assert ego.apply_control.call_args[0][0].brake == 1
