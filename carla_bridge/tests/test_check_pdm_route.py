"""Diagnostic acceptance and cleanup use offline fake physics, never VLM APIs."""
import builtins
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import carla
import pytest
import yaml

from carla_bridge.env.route_progress import RouteProgress
from carla_bridge.control.steering_model import SteeringModel
from carla_bridge.tools import check_pdm_route as diagnostic


class Sensor:
    def __init__(self):
        self.callback = None
        self.stopped = self.destroyed = False

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True
        return True


class Ego:
    def __init__(self):
        self.x = 0.0
        self.control = carla.VehicleControl(brake=1)

    def get_transform(self):
        return carla.Transform(carla.Location(x=self.x))

    def get_velocity(self):
        return carla.Vector3D(x=1)

    def get_acceleration(self):
        return carla.Vector3D()

    def get_control(self):
        return self.control

    def apply_control(self, control):
        self.control = control

    def get_physics_control(self):
        return NS(steering_curve=[NS(x=0, y=1), NS(x=36, y=.5)])


class World:
    def __init__(self):
        self.sensor = Sensor()
        self.time = 0.0
        self.settings = NS(no_rendering_mode=False)

    def get_blueprint_library(self):
        return NS(find=lambda identity: identity)

    def spawn_actor(self, *args, **kwargs):
        return self.sensor

    def get_snapshot(self):
        return NS(timestamp=NS(elapsed_seconds=self.time))

    def get_settings(self):
        return deepcopy(self.settings)

    def apply_settings(self, settings):
        self.settings = settings


class Client:
    last = None
    collision = False

    def __init__(self, **kwargs):
        Client.last = self
        self.client = NS(get_server_version=lambda: "0.9.15", get_client_version=lambda: "0.9.15")
        self.world = World()
        self.ego = Ego()
        self.restored = False
        self.tick_count = 0

    def connect(self):
        return self

    def tick(self):
        self.tick_count += 1
        self.world.time += .1
        if self.ego.control.brake < 1:
            self.ego.x += .1
        if self.collision and self.tick_count == 6:
            self.world.sensor.callback(NS(frame=6, other_actor=NS(id=123, type_id="vehicle.test"),
                                          normal_impulse=carla.Vector3D(x=1)))
        return self.tick_count

    def restore(self):
        self.restored = True
        self.world.settings.no_rendering_mode = False
        return True


class Scenario:
    last = None
    setup_error = False
    cleanup_error = False

    def __init__(self, client, ccfg, config):
        Scenario.last = self
        self.world, self.ego_vehicle, self.cfg = client.world, client.ego, config
        self.destroyed = False
        self.experiment_snapshot = None
        self.route = None
        self.traffic = NS(tm_port=8200, spawn_manifest={"vehicles": [], "walkers": []})

    def setup(self):
        if self.setup_error:
            raise RuntimeError("partial setup failed")
        points = [carla.Location(**p) for p in self.cfg["route"]["points"]]
        progress = RouteProgress(points, tolerance_m=.01, corridor_m=.1)
        self.route = NS(_route=points,
            progress_fraction=lambda: progress.update(self.ego_vehicle.get_transform().location),
            is_complete=lambda: progress.completed)
        for key, label in (("num_vehicles", "vehicles"), ("num_walkers", "walkers")):
            self.traffic.spawn_manifest[label] = [{"id": i} for i in range(self.cfg["traffic"][key])]

    def destroy(self):
        self.destroyed = True
        return not self.cleanup_error


class Planner:
    calls = []
    invalid = False
    error = False

    def __init__(self, world, ego, route, planning):
        self.vehicle_parameters = NS(wheel_base=3.0)
        self.steering_model = SteeringModel(1.2, wheelbase_m=3, front_track_m=1.6)
        self.max_steer = self.steering_model.max_steer_rad
        self.rear_axle_actor_xyz_m = (-1.3, 0, .3)

    def plan(self, semantic, semantic_enabled):
        self.calls.append((deepcopy(semantic), semantic_enabled))
        if self.error:
            raise RuntimeError("planning failed")
        points = [{"t": i * .1, "x": i * .1, "y": 0, "optional_v": 1.0,
                   "optional_heading_rad": 0} for i in range(41)]
        if self.invalid:
            points[2]["x"] = 100
        return {"trajectory": points, "target_speed": 1.0,
                "planner_diagnostics": {"emergency_brake": False, "selected_id": "center:MAINTAIN"}}


@pytest.fixture
def diagnostic_case(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "CarlaClient", Client)
    monkeypatch.setattr(diagnostic, "ScenarioManager", Scenario)
    for cls, attrs in ((Client, {"collision": False}),
                       (Scenario, {"setup_error": False, "cleanup_error": False}),
                       (Planner, {"invalid": False, "error": False, "calls": []})):
        for name, value in attrs.items():
            monkeypatch.setattr(cls, name, value)
    scenario = tmp_path / "case.yaml"
    source = {"scenario_name": "case", "map": "Town05", "duration_s": 2, "seed": 101,
              "traffic": {"num_vehicles": 4, "num_walkers": 2}, "events": [],
              "route": {"points": [{"x": 0, "y": 0, "z": 0}, {"x": .2, "y": 0, "z": 0},
                                     {"x": .4, "y": 0, "z": 0}]}}
    scenario.write_text(yaml.safe_dump(source), encoding="utf-8")
    args = diagnostic.parse_args(["--scenario", str(scenario), "--output-dir", str(tmp_path / "out")])
    return args, source


def run(args):
    return diagnostic.run_diagnostic(args, planner_factory=Planner, collision_grace_s=0)


def test_complete_route_passes_only_with_ordered_100_percent_and_records_both_streams(diagnostic_case):
    args, _ = diagnostic_case
    report = run(args)
    assert report["success"] and report["route_completed"]
    assert report["route_completion"] == 1
    assert report["run_kind"] == "engineering_diagnostic"
    assert report["paper_vlm_experiment"] is False
    assert report["vlm_called"] is report["feature_model_loaded"] is False
    assert report["traffic_conditions"]["spawned_vehicles"] == 4
    assert report["traffic_conditions"]["spawned_walkers"] == 2
    assert report["traffic_conditions"]["traffic_manager_port"] == 8200
    assert Planner.calls and all(not enabled for semantic, enabled in Planner.calls)
    assert all(semantic["lateral_decision"] == "FOLLOW_ROUTE" and
               semantic["longitudinal_decision"] == "MAINTAIN" for semantic, enabled in Planner.calls)
    assert (args.output_dir / "planner.jsonl").read_text(encoding="utf-8")
    assert (args.output_dir / "control.jsonl").read_text(encoding="utf-8")
    assert json.loads((args.output_dir / "report.json").read_text(encoding="utf-8")) == report
    assert Scenario.last.destroyed and Client.last.restored
    assert Client.last.world.sensor.stopped and Client.last.world.sensor.destroyed
    assert Client.last.ego.get_control().brake == 1
    assert Client.last.world.settings.no_rendering_mode is False


def test_empty_traffic_preserves_route_seed_and_original_yaml(diagnostic_case):
    args, original = diagnostic_case
    before = args.scenario.read_bytes()
    args.empty_traffic = True
    args.semantic_influence = "on"
    report = run(args)
    assert report["success"]
    assert report["seed"] == 101 and Scenario.last.cfg["seed"] == 101
    assert Scenario.last.cfg["route"] == original["route"]
    assert report["traffic_conditions"]["original"] == original["traffic"]
    assert report["traffic_conditions"]["effective"] == {"num_vehicles": 0, "num_walkers": 0}
    assert report["traffic_conditions"]["spawned_vehicles"] == report["traffic_conditions"]["spawned_walkers"] == 0
    assert args.scenario.read_bytes() == before
    assert all(enabled for semantic, enabled in Planner.calls)


@pytest.mark.parametrize("failure,status", [("invalid", "invalid_trajectory"), ("error", "error")])
def test_invalid_plan_or_exception_never_passes_and_cleans_up(diagnostic_case, monkeypatch, failure, status):
    args, _ = diagnostic_case
    monkeypatch.setattr(Planner, failure, True)
    report = run(args)
    assert report["success"] is False and report["status"] == status
    assert Scenario.last.destroyed and Client.last.restored
    assert Client.last.world.sensor.stopped and Client.last.world.sensor.destroyed
    assert Client.last.ego.get_control().brake == 1


def test_collision_never_passes_even_if_otherwise_on_route(diagnostic_case, monkeypatch):
    args, _ = diagnostic_case
    monkeypatch.setattr(Client, "collision", True)
    report = run(args)
    assert not report["success"] and report["status"] == "collision"
    assert report["collisions"][0]["other_actor_id"] == 123
    assert Scenario.last.destroyed and Client.last.restored


def test_timeout_is_failure_and_is_not_a_route_completion_test(diagnostic_case):
    args, _ = diagnostic_case
    args.max_duration_s = .1
    report = run(args)
    assert not report["success"] and report["status"] == "timeout"
    assert report["duration_s"] == .1 and report["route_completion"] < 1


def test_partial_setup_exception_still_destroys_actors_and_restores_world(diagnostic_case, monkeypatch):
    args, _ = diagnostic_case
    monkeypatch.setattr(Scenario, "setup_error", True)
    report = run(args)
    assert report["status"] == "error" and not report["success"]
    assert Scenario.last.destroyed and Client.last.restored


def test_cleanup_failure_cannot_be_declared_success(diagnostic_case, monkeypatch):
    args, _ = diagnostic_case
    monkeypatch.setattr(Scenario, "cleanup_error", True)
    report = run(args)
    assert report["route_completed"] and report["route_completion"] == 1
    assert not report["success"] and report["status"] == "cleanup_failed"
    assert Client.last.restored


def test_event_case_is_explicitly_rejected_before_connecting(diagnostic_case, monkeypatch):
    args, source = diagnostic_case
    source["events"] = [{"id": "crossing", "type": "pedestrian_crossing"}]
    args.scenario.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setattr(diagnostic, "CarlaClient", lambda **kwargs: pytest.fail("must not connect"))
    report = run(args)
    assert report["status"] == "unsupported_events" and not report["success"]
    assert "unsupported" in report["error"]


def test_output_is_never_overwritten(diagnostic_case):
    args, _ = diagnostic_case
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        run(args)


def test_source_change_during_run_cannot_pass(diagnostic_case, monkeypatch):
    args, _ = diagnostic_case
    fingerprints = iter([{"a.py": "before"}, {"a.py": "after"}])
    monkeypatch.setattr(diagnostic, "_source_fingerprint", lambda: next(fingerprints))
    report = run(args)
    assert report["route_completed"] and report["source_changed_during_run"]
    assert not report["success"] and report["status"] == "source_changed_during_run"
    assert Scenario.last.destroyed and Client.last.restored


def test_fingerprint_read_error_cannot_skip_cleanup_or_pass(diagnostic_case, monkeypatch):
    args, _ = diagnostic_case
    calls = []

    def fingerprint():
        calls.append(1)
        if len(calls) == 2:
            raise FileNotFoundError("source disappeared during audit")
        return {"a.py": "before"}

    monkeypatch.setattr(diagnostic, "_source_fingerprint", fingerprint)
    report = run(args)
    assert report["route_completed"] and not report["success"]
    assert "FileNotFoundError" in report["fingerprint_error"]
    assert Scenario.last.destroyed and Client.last.restored
    assert Client.last.world.sensor.stopped and Client.last.world.sensor.destroyed


def test_no_vlm_feature_or_memory_module_is_imported_by_diagnostic(diagnostic_case, monkeypatch):
    args, _ = diagnostic_case
    original_import = builtins.__import__

    def guarded_import(name, *positional, **kwargs):
        if name.startswith(("transformers", "torch", "src.vla_memory.memory", "src.vla_memory.decision")):
            pytest.fail("diagnostic imported heavyweight/model module " + name)
        return original_import(name, *positional, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert run(args)["success"]
