"""Route acceptance and control integration without a simulator or VLM calls."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import carla
import pytest

from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.env.route_planner import RoutePlanner
from carla_bridge.env.route_progress import RouteProgress
from carla_bridge.tests.test_run_safety import session_at
from src.vla_memory.common.config import Config


def loc(x, y=0, z=0):
    return carla.Location(x=x, y=y, z=z)


def straight():
    return RouteProgress([loc(x) for x in range(0, 101, 2)])


def test_route_start_is_zero_and_repeated_queries_do_not_advance():
    route = straight()
    for _ in range(50):
        assert route.update(loc(0)) == 0
    assert not route.completed


def test_progress_uses_arc_length_and_never_regresses():
    route = RouteProgress([loc(0), loc(10), loc(100)])
    route.update(loc(0))
    assert route.update(loc(25)) == pytest.approx(.25)
    assert route.update(loc(20)) == pytest.approx(.25)


def test_adjacent_lane_manoeuvre_does_not_freeze_navigation():
    route = straight()
    for x, y in [(0, 0), (5, 1.75), (10, 3.5), (20, 3.5), (30, 3.5), (40, 0), (100, 0)]:
        route.update(loc(x, y))
    assert route.completed
    assert route.fraction == 1


def test_returning_to_nearby_endpoint_cannot_skip_loop():
    points = [loc(0), loc(20), loc(20, 20), loc(0, 20), loc(0, 1)]
    route = RouteProgress(points)
    for _ in range(10):
        route.update(loc(0))
    route.update(loc(0, 1))
    assert not route.completed
    for p in points:
        route.update(p)
    assert route.completed


def test_swept_ticks_do_not_miss_interior_checkpoints():
    route = straight()
    for x in (0, 25, 50, 75, 99):
        route.update(loc(x))
    assert route.completed


def test_endpoint_requires_current_position_within_tolerance():
    route = straight()
    for x in (0, 25, 50, 75, 95, 105):
        route.update(loc(x))
    assert not route.completed
    assert route.fraction <= .999
    route.update(loc(100))
    assert route.completed


def test_off_route_shortcut_does_not_pass_skipped_checkpoints():
    route = RouteProgress([loc(0), loc(20), loc(20, 20)])
    route.update(loc(0))
    route.update(loc(20, 20))
    assert not route.completed


@pytest.mark.parametrize("points", [[], [loc(0)], [loc(0), loc(0)]])
def test_empty_degenerate_route_rejected(points):
    with pytest.raises(ValueError):
        RouteProgress(points)


def test_astar_retains_exact_destination_in_same_lane():
    def waypoint(x, lane=1):
        return NS(road_id=1, section_id=0, lane_id=lane, s=x,
                  transform=carla.Transform(loc(x)), next=lambda step: [])
    start, end = waypoint(0), waypoint(1.5)
    route = RoutePlanner.__new__(RoutePlanner)
    points, _ = route._astar_search(start, end, 2)
    assert points[-1] is end
    assert route._astar_search(start, waypoint(1, lane=2), 2) == ([], [])


def fake_loop(*, complete_at=None, duration=1.0, record=None):
    cfg = Config(overrides={"carla": {"fixed_delta_seconds": .1, "replan_interval_s": 3,
                                     "max_duration_s": duration}})
    loop = ClosedLoop(cfg, "unused")
    ego = Mock()
    ego.get_transform.return_value = carla.Transform(loc(0))
    ego.get_velocity.return_value = carla.Vector3D(x=5)
    ego.get_acceleration.return_value = carla.Vector3D()
    route = Mock()
    done = lambda: complete_at is not None and loop._elapsed + 1e-8 >= complete_at
    route.is_complete.side_effect = done
    route.progress_fraction.side_effect = lambda: 1.0 if done() else .4
    route.nav_instruction.return_value = "straight"
    loop.scenario = NS(ego_vehicle=ego, route=route, duration_s=lambda: duration,
                       scenario_name=lambda: "test", destroy=Mock(return_value=True))
    loop.carla_client = Mock()
    loop.carla_client.tick.return_value = 1
    loop.cameras = Mock()
    loop.cameras.capture.return_value = "unused.jpg"
    loop.history = Mock()
    loop.history.yaw_rate.return_value = 0
    loop.history.build_ego_centric.return_value = []
    loop.ego_provider = Mock()
    state = Mock(speed=5)
    state.to_ego_centric.return_value = {"speed": 5}
    loop.ego_provider.build.return_value = state
    loop.perception = Mock()
    loop.perception.get_objects.return_value = []
    loop.kf_builder = Mock()
    loop.kf_builder.build.side_effect = lambda **kwargs: {"ego_state": kwargs["ego_state"]}
    loop.tracker = Mock()
    loop.tracker.compute_control.return_value = carla.VehicleControl()
    loop.loop = Mock()
    loop.loop.step.return_value = record or {
        "decision_output": {"target_speed": 5, "trajectory": [{"x": 1, "y": 0, "t": .2}]},
        "decision_validation": {"accepted": True},
    }
    loop._route_inject_enabled = False
    loop._stuck_enabled = False
    loop._raw_capture = Mock(return_value=True)
    return loop


@pytest.mark.parametrize("complete_at,status,elapsed", [(None, "timeout", 1),
                                                       (.5, "route_completed", .5),
                                                       (1, "route_completed", 1)])
def test_only_route_completion_is_success_and_no_extra_ticks(complete_at, status, elapsed):
    loop = fake_loop(complete_at=complete_at)
    loop.run()
    assert loop.run_status == status
    assert loop.outcome()["success"] == (status == "route_completed")
    assert loop._elapsed == pytest.approx(elapsed)
    assert loop.carla_client.tick.call_count == round(elapsed / .1)
    assert loop.scenario.ego_vehicle.apply_control.call_args[0][0].brake == 1


def test_capture_to_capture_interval_matches_validated_horizon():
    loop = fake_loop(duration=30)
    loop._replan_cycle()
    loop._replan_cycle()
    samples = [call.args[0] for call in loop.loop.step.call_args_list]
    assert samples[1]["sim_time_s"] - samples[0]["sim_time_s"] == pytest.approx(3)
    assert samples[0]["planning_constraints"]["required_horizon_s"] == 3
    assert loop.carla_client.tick.call_count == 60


def test_rejected_or_unvalidated_decision_never_reaches_controller():
    loop = fake_loop(duration=30, record={"decision_output": {
        "target_speed": 6, "trajectory": [{"x": 53.5, "y": -63.9, "t": 3}]},
        "decision_validation": {"accepted": False, "reasons": ["infeasible"]}})
    loop.run()
    assert loop.run_status == "invalid_trajectory"
    assert loop._trajectory_rejections == 3
    for call in loop.tracker.set_trajectory.call_args_list:
        assert call.args[0] == [] and call.args[-1] == 0
    assert loop.scenario.ego_vehicle.apply_control.call_args[0][0].brake == 1


def test_vlm_fallback_cannot_be_executed():
    loop = fake_loop()
    result = loop._validate_decision({"fallback_used": True}, {"ego_state": {"speed": 5}})
    assert not result["accepted"]
    assert result["reasons"] == ["rule_fallback_not_executable"]


def test_planned_stop_does_not_trigger_stuck_timer():
    loop = fake_loop(duration=30, record={"decision_output": {"target_speed": 0, "trajectory": []},
                                         "decision_validation": {"accepted": True}})
    loop._stuck_enabled = True
    loop._stuck_term_s = .2
    loop.scenario.ego_vehicle.get_velocity.return_value = carla.Vector3D()
    loop._replan_cycle()
    loop._replan_cycle()
    assert not loop._early_terminated
    assert loop._stuck_since is None


def test_cleanup_failure_is_in_final_report_and_cannot_be_success(tmp_path):
    loop = fake_loop(complete_at=.5)
    loop.config.data["output_dir"] = tmp_path
    loop.run()
    loop.metrics = Mock()
    loop.metrics.summary.return_value = {"route_completion": 1., "route_completed": True}
    loop.scenario.destroy.return_value = False
    loop.close()
    loop.close()
    assert loop.run_status == "cleanup_failed"
    assert not loop.outcome()["success"]
    loop.carla_client.restore.assert_not_called()
    report = json.loads(next(tmp_path.glob("carla_run_*.json")).read_text(encoding="utf-8"))
    assert report["run_info"]["status"] == "cleanup_failed"
    assert not report["run_info"]["success"]
    loop.metrics.summary.assert_called_once()


def test_report_write_failure_cannot_be_success(monkeypatch):
    loop = fake_loop(complete_at=.5)
    loop.run()
    loop.metrics = Mock()
    loop.metrics.summary.return_value = {}
    monkeypatch.setattr("carla_bridge.closed_loop.write_report", Mock(side_effect=OSError("disk full")))
    loop.close()
    assert loop.run_status == "report_failed"
    assert not loop.outcome()["success"]


@pytest.mark.parametrize("status", ["timeout", "stuck", "invalid_trajectory", "cleanup_failed", "route_completed"])
def test_entry_exit_status_and_manifest_match_outcome(tmp_path, monkeypatch, status):
    from carla_bridge import run_carla_demo as entry
    helper = session_at(tmp_path)
    monkeypatch.setattr(entry, "load_config", Mock(return_value=helper.config))
    monkeypatch.setattr(entry, "setup_logger", Mock())
    monkeypatch.setattr(entry, "set_log_directory", Mock())
    for block in ("scene_understanding", "decision"):
        monkeypatch.setenv(helper.config.get_nested(block, "api_key_env", default="DASHSCOPE_API_KEY"), "test-only")
    loop = Mock(run_status=status)
    loop.outcome.return_value = {"success": status == "route_completed", "route_completed": status == "route_completed"}
    monkeypatch.setattr(entry, "ClosedLoop", Mock(return_value=loop))
    monkeypatch.setattr("sys.argv", ["run_carla_demo", "--scenario", str(helper.scenario_path)])
    if status == "route_completed":
        entry.main()
    else:
        with pytest.raises(RuntimeError, match=status):
            entry.main()
    manifest = json.loads((helper.config.get("output_dir") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == status
    assert manifest["outcome"]["success"] == (status == "route_completed")
    loop.close.assert_called_once()
