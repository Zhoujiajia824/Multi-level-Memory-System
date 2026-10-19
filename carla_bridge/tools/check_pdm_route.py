"""Drive a route with constant semantic intent, without VLM, DINO or memory.

This is an engineering diagnostic, not a paper VLM experiment. Run on an idle
CARLA instance: ScenarioManager loads the requested map and owns its actors.
Only ordered 100% route completion without collisions or invalid plans passes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import subprocess
from pathlib import Path
import threading
from uuid import uuid4

import carla
import yaml

from carla_bridge.control.trajectory_tracker import TrajectoryTracker
from carla_bridge.control.trajectory_validation import TrajectoryValidator
from carla_bridge.env.carla_client import CarlaClient
from carla_bridge.env.scenario_manager import ScenarioManager
from carla_bridge.planning.native_tracking import (
    MAX_PROFILE_LIFETIME_S, SAMPLE_INTERVAL_S, override_native_tracking,
)


ROOT = Path(__file__).resolve().parents[2]
CONSTANT_SEMANTICS = {
    "lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "MAINTAIN",
    "behavior_reason": "工程诊断：固定沿路线正常行驶。", "risk_level": "low", "safety_notes": [],
}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _source_fingerprint():
    paths = [path for path in (ROOT / "carla_bridge").rglob("*.py")
             if "tests" not in path.relative_to(ROOT).parts]
    paths.append(ROOT / "carla_bridge/config/carla.yaml")
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _observation(ego):
    transform = ego.get_transform()
    velocity, acceleration = ego.get_velocity(), ego.get_acceleration()
    state = {
        "x": transform.location.x, "y": transform.location.y, "z": transform.location.z,
        "yaw_rad": math.radians(transform.rotation.yaw),
        "velocity_x": velocity.x, "velocity_y": velocity.y,
        "speed": math.hypot(velocity.x, velocity.y),
        "acceleration_x": acceleration.x, "acceleration_y": acceleration.y,
    }
    if not all(math.isfinite(value) for value in state.values()):
        raise ValueError("CARLA returned non-finite ego state")
    return transform, state


def _control_values(control):
    return {key: getattr(control, key) for key in
            ("throttle", "brake", "steer", "gear", "reverse", "hand_brake")}


def run_diagnostic(args, *, planner_factory=None, collision_grace_s=.25):
    """Run one condition; factories can be replaced in offline lifecycle tests."""
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)  # Never mix or overwrite attempts.
    for name in ("planner.jsonl", "control.jsonl"):
        (output / name).touch()
    report = {
        "schema_version": 1, "run_kind": "engineering_diagnostic", "paper_vlm_experiment": False,
        "vlm_called": False, "feature_model_loaded": False, "memory_access": "off",
        "constant_semantics": deepcopy(CONSTANT_SEMANTICS),
        "semantic_influence": args.semantic_influence == "on", "empty_traffic": bool(args.empty_traffic),
        "scenario_path": str(Path(args.scenario).resolve()), "output_dir": str(output.resolve()),
        "status": "setup_failed", "success": False, "route_completed": False,
        "route_completion": 0.0, "duration_s": 0.0, "planner_count": 0,
        "trajectory_rejections": 0, "collisions": [], "cleanup_errors": [],
        "acceptance": "ordered_route_100_percent_and_endpoint_no_collision_no_invalid_plan_no_error",
        "host": args.host, "port": args.port,
        "no_rendering_mode": not args.rendering,
    }
    client = scenario = sensor = ego = None
    elapsed = 0.0
    collisions = report["collisions"]
    try:
        report["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
        report["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, timeout=10).strip())
    except (OSError, subprocess.SubprocessError):
        report["commit"] = None
        report["dirty"] = None
    try:
        report["source_sha256"] = _source_fingerprint()
        raw = Path(args.scenario).read_bytes()
        original = yaml.safe_load(raw.decode("utf-8-sig"))
        if not isinstance(original, dict):
            raise ValueError("scenario must be a YAML mapping")
        config = deepcopy(original)
        report.update(scenario_sha256=hashlib.sha256(raw).hexdigest(), seed=original.get("seed", 42),
                      original_scenario=original, original_route_sha256=_digest(original.get("route")),
                      traffic_conditions={"original": deepcopy(original.get("traffic") or {})})
        # Running an event case while omitting its scheduler would change the
        # experiment silently. This small diagnostic explicitly does not support it.
        if config.get("events"):
            report["status"] = "unsupported_events"
            raise ValueError("event scenarios are unsupported by this diagnostic; use an event-free case")
        if args.empty_traffic:
            config.setdefault("traffic", {}).update(num_vehicles=0, num_walkers=0)
        report["traffic_conditions"]["effective"] = deepcopy(config.get("traffic") or {})
        report["effective_scenario_sha256"] = _digest(config)
        _write_json(output / "effective_scenario.json", config)
        ccfg = yaml.safe_load((ROOT / "carla_bridge/config/carla.yaml").read_text(encoding="utf-8"))["carla"]
        ccfg = deepcopy(ccfg)
        ccfg["experiment_output_dir"] = str(output)
        dt = float(ccfg.get("fixed_delta_seconds", .1))
        planning = deepcopy(ccfg.get("planning") or {})
        if args.pdm_lateral_control is not None:
            planning["lateral_control"] = args.pdm_lateral_control
        report["pdm_lateral_control"] = planning.get("lateral_control", "pure_pursuit")
        report["pdm_lateral_preview"] = planning.get("lqr_lateral_preview", "fixed")
        interval = float(planning.get("replan_interval_s", .2))
        required_horizon = float(ccfg.get("replan_interval_s", 3.0))
        maximum = float(args.max_duration_s if args.max_duration_s is not None else config.get("duration_s", 900))
        if not all(math.isfinite(value) and value > 0 for value in (dt, interval, maximum, required_horizon)):
            raise ValueError("time steps, horizons and maximum duration must be finite and positive")
        if interval < dt:
            raise ValueError("PDM interval cannot be shorter than one CARLA tick")
        if planning.get("lateral_control", "pure_pursuit") == "native_lqr":
            if abs(dt - SAMPLE_INTERVAL_S) > 1e-8:
                raise ValueError("Native LQR tracking requires fixed_delta_seconds=0.1")
            if interval > MAX_PROFILE_LIFETIME_S + 1e-9:
                raise ValueError("Native LQR tracking requires PDM replan_interval_s <= 0.3")
        planning["horizon_s"] = max(float(planning.get("horizon_s", 4)), required_horizon)
        planning["semantic_influence"] = report["semantic_influence"]
        ticks_per_plan = max(1, round(interval / dt))
        report["parameters"] = {"fixed_delta_seconds": dt, "pdm_interval_s": ticks_per_plan * dt,
                                "required_horizon_s": required_horizon, "max_duration_s": maximum,
                                "planning": planning, "trajectory_validation": ccfg.get("trajectory_validation")}
        client = CarlaClient(host=args.host, port=args.port, timeout_s=args.timeout_s,
                             synchronous=True, fixed_delta_seconds=dt)
        client.connect()
        report["server_version"] = client.client.get_server_version()
        report["client_version"] = client.client.get_client_version()
        if report["server_version"] != report["client_version"]:
            raise ValueError("CARLA client/server versions differ")
        scenario = ScenarioManager(client, ccfg, config)
        scenario.setup()
        world, ego, route = scenario.world, scenario.ego_vehicle, scenario.route
        settings = world.get_settings()
        settings.no_rendering_mode = not args.rendering
        world.apply_settings(settings)  # CarlaClient.restore() restores the map's original settings.
        if route is None:
            raise ValueError("scenario did not create an ordered route")
        report["resolved_route"] = [{"x": p.x, "y": p.y, "z": p.z} for p in route._route]
        report["resolved_route_sha256"] = _digest(report["resolved_route"])
        traffic_manifest = deepcopy(scenario.traffic.spawn_manifest)
        report["traffic_conditions"].update(
            traffic_manager_port=getattr(scenario.traffic, "tm_port", None),
            spawned_vehicles=len(traffic_manifest["vehicles"]), spawned_walkers=len(traffic_manifest["walkers"]),
            initial_actors=traffic_manifest, experiment_snapshot=deepcopy(scenario.experiment_snapshot))
        sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                                   carla.Transform(), attach_to=ego)

        def on_collision(event):
            impulse = event.normal_impulse
            collisions.append({"frame": event.frame, "other_actor_id": event.other_actor.id,
                               "other_actor_type": event.other_actor.type_id,
                               "normal_impulse": [impulse.x, impulse.y, impulse.z]})

        sensor.listen(on_collision)
        ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(5):
            client.tick()  # Same bounded traffic initialization as the full loop.
        if planner_factory is None:
            from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
            planner_factory = CarlaPDMPlanner
        planner = planner_factory(world, ego, route, planning)
        tracker = TrajectoryTracker(ccfg.get("controller") or {}, control_dt_s=dt)
        tracker.bind_vehicle_model(planner.vehicle_parameters.wheel_base,
                                   steering_model=planner.steering_model,
                                   rear_axle_actor_xyz_m=planner.rear_axle_actor_xyz_m)
        report["tracking_vehicle_model"] = {
            "wheelbase_m": tracker.pp.L, "max_steer_rad": tracker.max_steer_rad,
            "physical_inner_max_steer_rad": tracker.steering_model.physical_max_steer_rad,
            "front_track_m": tracker.steering_model.front_track_m,
            "steering_curve_kmh_scale": [list(point) for point in tracker.steering_model.curve],
            "mapping": tracker.steering_model.mapping,
            "tracking_reference": "pdm_rear_axle",
            "rear_axle_actor_xyz_m": list(tracker.rear_axle_actor_xyz_m),
            "input_latency_modeled": False,
        }
        validator = TrajectoryValidator(ccfg.get("trajectory_validation"))
        sequence = tick = 0
        command = None
        report["status"] = "running"
        while elapsed < maximum - 1e-9:
            transform, state = _observation(ego)
            fraction = route.progress_fraction()  # Existing ordered RouteProgress.
            report.update(route_completion=fraction, route_completed=route.is_complete())
            if collisions:
                report["status"] = "collision"
                break
            if report["route_completed"]:
                report["status"] = "route_complete"
                break
            if tick % ticks_per_plan == 0:
                sequence += 1
                report["planner_count"] = sequence
                try:
                    plan = planner.plan(deepcopy(CONSTANT_SEMANTICS),
                                        semantic_enabled=report["semantic_influence"], sim_time_s=elapsed)
                except Exception as exc:
                    _append_json(output / "planner.jsonl", {"sim_time_s": elapsed, "sequence": sequence,
                        "semantic_decision": CONSTANT_SEMANTICS, "error": type(exc).__name__ + ": " + str(exc),
                        "planner_diagnostics": getattr(exc, "diagnostics", {})})
                    raise
                validation = validator.validate(plan.get("trajectory"), plan.get("target_speed"), state["speed"],
                                                required_horizon, use_speed_profile=True).to_dict()
                if (plan.get("planner_diagnostics") or {}).get("emergency_brake") is True:
                    validation["accepted"] = False
                    validation["reasons"].append("pdm_emergency_brake")
                _append_json(output / "planner.jsonl", {
                    "sim_time_s": elapsed, "sequence": sequence, "source": "constant_semantic_diagnostic",
                    "semantic_decision": CONSTANT_SEMANTICS, "semantic_influence": report["semantic_influence"],
                    "planner_diagnostics": plan.get("planner_diagnostics"), "trajectory": plan.get("trajectory"),
                    "validation": validation})
                if validation["accepted"] is not True:
                    report.update(status="invalid_trajectory", trajectory_rejections=1)
                    break
                tracker.set_trajectory(plan["trajectory"], state["x"], state["y"], state["yaw_rad"],
                                       plan["target_speed"], follow_speed_profile=True)
            control = tracker.compute_control(state["x"], state["y"], state["yaw_rad"], state["speed"])
            native_tracking = None
            diagnostics = plan.get("planner_diagnostics") or {}
            if diagnostics.get("native_tracking", {}).get("active") is True:
                control, native_tracking = override_native_tracking(control, diagnostics,
                    now_s=elapsed, speed_mps=state["speed"], steering_model=planner.steering_model)
            command = {"sim_time_s": elapsed, "planner_sequence": sequence,
                       **_control_values(control), "tracking": deepcopy(tracker.last_control_diagnostics),
                       "native_tracking": native_tracking}
            ego.apply_control(control)
            frame = client.tick()
            tick += 1
            elapsed = round(tick * dt, 10)
            _, measured = _observation(ego)
            fraction = route.progress_fraction()
            report.update(duration_s=elapsed, route_completion=fraction, route_completed=route.is_complete())
            snapshot = world.get_snapshot()
            _append_json(output / "control.jsonl", {
                "sim_time_s": elapsed, "frame": frame, "world_elapsed_seconds": snapshot.timestamp.elapsed_seconds,
                "measured": measured, "applied_control": _control_values(ego.get_control()),
                "command": command, "route_completion": fraction})
            if tick % max(1, round(10 / dt)) == 0:
                print(json.dumps({"run_kind": "engineering_diagnostic", "sim_time_s": elapsed,
                                  "route_completion": fraction, "speed_mps": measured["speed"],
                                  "planner_count": sequence}, ensure_ascii=False), flush=True)
        if report["status"] == "running":
            report["status"] = "route_complete" if report["route_completed"] else "timeout"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        if report["status"] != "unsupported_events":
            report["status"] = "error"
        report["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["duration_s"] = elapsed
        try:
            report["source_changed_during_run"] = _source_fingerprint() != report.get("source_sha256")
        except OSError as exc:
            # Audit failure must never bypass actor or server cleanup.
            report["source_changed_during_run"] = True
            report["fingerprint_error"] = type(exc).__name__ + ": " + str(exc)
        # ScenarioManager may raise after partially spawning traffic; preserve
        # those conditions as well as cleaning up the partially built scene.
        traffic = getattr(scenario, "traffic", None)
        if traffic is not None and "traffic_conditions" in report:
            manifest = deepcopy(traffic.spawn_manifest)
            report["traffic_conditions"].update(
                traffic_manager_port=getattr(traffic, "tm_port", None),
                spawned_vehicles=len(manifest["vehicles"]), spawned_walkers=len(manifest["walkers"]),
                initial_actors=manifest,
                experiment_snapshot=deepcopy(getattr(scenario, "experiment_snapshot", None)))
        if ego is not None:
            try:
                ego.apply_control(carla.VehicleControl(brake=1.0))
            except Exception as exc:
                report["cleanup_errors"].append("brake: " + str(exc))
        if sensor is not None:
            # Collision callbacks use a separate client transport; allow already
            # produced events to arrive before stopping the sensor and accepting.
            threading.Event().wait(collision_grace_s)
            for action in (sensor.stop, sensor.destroy):
                try:
                    if action() is False:
                        report["cleanup_errors"].append("collision sensor cleanup returned False")
                except Exception as exc:
                    report["cleanup_errors"].append("collision sensor: " + str(exc))
        if scenario is not None:
            try:
                if scenario.destroy() is False:
                    report["cleanup_errors"].append("scenario cleanup incomplete")
            except Exception as exc:
                report["cleanup_errors"].append("scenario: " + str(exc))
        if client is not None:
            try:
                if client.restore() is False:
                    report["cleanup_errors"].append("server settings restoration incomplete")
            except Exception as exc:
                report["cleanup_errors"].append("server restore: " + str(exc))
        if collisions:
            report["status"] = "collision"
        elif report["cleanup_errors"] and report["status"] == "route_complete":
            report["status"] = "cleanup_failed"
        elif report["source_changed_during_run"] and report["status"] == "route_complete":
            report["status"] = "source_changed_during_run"
        report["success"] = (report["status"] == "route_complete" and report["route_completed"]
                             and report["route_completion"] == 1.0 and not collisions
                             and not report["trajectory_rejections"] and not report["cleanup_errors"])
        _write_json(output / "report.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--semantic-influence", choices=("on", "off"), default="off")
    parser.add_argument("--pdm-lateral-control", choices=("native_lqr", "pure_pursuit"), default=None,
                        help="PDM横向执行模式；省略时使用carla.yaml，纵向仍PID")
    parser.add_argument("--empty-traffic", action="store_true")
    parser.add_argument("--rendering", action="store_true", help="enable rendering (diagnostic defaults to no rendering)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pdm_diagnostics" /
                        (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]))
    parser.add_argument("--max-duration-s", type=float)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_diagnostic(args)
    print(json.dumps({key: report[key] for key in ("run_kind", "status", "success", "route_completion",
                                                 "duration_s", "output_dir")}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
