"""Research scenario contracts; no server, model, or network required."""
import copy
import csv
import json
from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import Mock

import carla
import pytest
import yaml

from carla_bridge.paper_suite import read_catalog, materialize, validate_prepared, digest, summarize_runs
from carla_bridge.env.paper_events import PaperEventScheduler
from carla_bridge.env.traffic_spawner import TrafficSpawner
from carla_bridge.tests.test_route_acceptance import fake_loop
from carla_bridge.tests.test_run_safety import session_at
from carla_bridge.env.route_planner import RoutePlanner
from carla_bridge.env.scenario_manager import ScenarioManager


def prepared_profiles(catalog):
    profiles = {}
    for key, spec in catalog["profiles"].items():
        anchors = {}
        for event_id, event in spec["anchors"].items():
            distance = event["search_from_m"]
            item = {"id": event_id, "type": event["type"], "anchor_distance_m": distance}
            if event["type"] == "pedestrian_crossing":
                item.update(start={"x": distance, "y": 5., "z": .3},
                            target={"x": distance, "y": -5., "z": .3},
                            speed_mps=event["speed_mps"], trigger_distance_m=distance - event["lead_distance_m"])
            else:
                item.update(spawn={"x": distance, "y": 0., "z": .3, "yaw": 0.},
                            blueprint=event["blueprint"], trigger_distance_m=0.)
            anchors[event_id] = item
        profiles[key] = {"map": spec["map"], "spawn": {"x": 0., "y": 0., "z": .3, "yaw": 0.},
                         "points": [{"x": float(x), "y": 0., "z": 0.} for x in range(0, spec["length_m"] + 1, 2)],
                         "length_m": spec["length_m"], "anchors": anchors,
                         "map_sha256": "test-map-hash", "server_version": "0.9.15"}
    return profiles


def cases(split="train"):
    catalog = read_catalog()
    return materialize(catalog, prepared_profiles(catalog), split)


def test_catalog_uses_installed_maps_and_has_regular_single_and_compound_conditions():
    catalog = read_catalog()
    assert Counter(c["category"] for c in catalog["cases"].values()) == {"regular": 4, "longtail": 4, "compound": 2}
    assert {p["map"] for p in catalog["profiles"].values()} == {"Town03", "Town04", "Town05", "Town10HD"}
    assert not set(catalog["splits"]["train"]) & set(catalog["splits"]["eval"])


def test_materialized_cases_validate_and_preserve_paired_route_and_traffic():
    suite = cases()
    assert len(suite) == 30
    for cfg in suite.values():
        validate_prepared(cfg)
        control = cfg["experiment"].get("paired_control")
        if control:
            baseline = suite[f"{control}_train_s{cfg['seed']}"]
            assert cfg["route"] == baseline["route"]
            assert cfg["traffic"] == baseline["traffic"]
            assert cfg["reserved_event_sites"] == baseline["reserved_event_sites"]
    assert cases("eval")["R05_multilane_eval_s101"]["seed"] == 101


def test_variants_do_not_share_mutable_config():
    suite = cases()
    first, second = suite["R05_multilane_train_s11"], suite["L05_crossing_train_s11"]
    first["route"]["points"][0]["x"] = 999
    assert second["route"]["points"][0]["x"] == 0


@pytest.mark.parametrize("field", ["route", "events", "ego", "reserved_event_sites"])
def test_geometry_tampering_is_rejected(field):
    cfg = cases()["L05_crossing_train_s11"]
    if field == "route":
        cfg[field]["points"][0]["x"] += 1
    elif field == "events":
        cfg[field][0]["trigger_distance_m"] += 1
    elif field == "ego":
        cfg[field]["spawn"]["x"] += 1
    else:
        cfg[field] = []
    with pytest.raises(ValueError, match="changed"):
        validate_prepared(cfg)


def test_training_seed_cannot_use_reserved_evaluation_seed():
    catalog = read_catalog()
    with pytest.raises(ValueError, match="overlap"):
        materialize(catalog, prepared_profiles(catalog), "train", seeds=[101])


def scheduler_fixture(case_id="L05_crossing_train_s11"):
    cfg = cases()[case_id]
    world, ego, route = Mock(), Mock(), Mock()
    route.distance_completed_m.return_value = 0.
    ego.get_location.return_value = carla.Location(x=180.)
    bp = Mock(id="walker.pedestrian.0001")
    world.get_blueprint_library.return_value.filter.return_value = [bp]
    world.get_blueprint_library.return_value.find.return_value = Mock(id="vehicle.audi.etron")
    world.get_weather.return_value.sun_altitude_angle = 70.
    spawned = []
    def spawn(blueprint, tf):
        actor = Mock(id=100 + len(spawned), is_alive=True)
        actor.get_location.return_value = tf.location
        actor.destroy.return_value = True
        spawned.append(actor)
        return actor
    world.try_spawn_actor.side_effect = spawn
    scheduler = PaperEventScheduler(world, ego, route, cfg["events"], elapsed=lambda: 12.0)
    return scheduler, world, route, spawned


def test_pedestrian_exists_before_trigger_and_starts_on_route_progress_without_ticks():
    scheduler, world, route, actors = scheduler_fixture()
    scheduler.setup()
    assert actors and scheduler.records[0]["state"] == "waiting"
    scheduler.tick()
    assert actors[0].apply_control.call_args.args[0].speed == 0
    route.distance_completed_m.return_value = 160.
    scheduler.tick()
    assert actors[0].apply_control.call_args.args[0].speed == pytest.approx(1.4)
    assert scheduler.records[0]["triggered_sim_time_s"] == 12.
    assert not scheduler.summary()["valid"]  # issued control alone is not motion evidence
    actors[0].get_location.return_value = carla.Location(x=180, y=4, z=.3)
    scheduler.tick()
    assert scheduler.summary()["valid"]
    world.tick.assert_not_called()
    world.wait_for_tick.assert_not_called()


def test_crossing_arrival_stops_actor():
    scheduler, _, route, actors = scheduler_fixture()
    scheduler.setup()
    route.distance_completed_m.return_value = 180
    actors[0].get_location.return_value = carla.Location(x=180, y=-5.5, z=.3)
    scheduler.tick()
    assert scheduler.records[0]["state"] == "completed"
    assert actors[0].apply_control.call_args.args[0].speed == 0


def test_missing_event_actor_is_failure_not_successful_trigger():
    scheduler, _, _, actors = scheduler_fixture()
    scheduler.setup()
    actors[0].is_alive = False
    scheduler.tick()
    assert scheduler.has_failed()
    assert not scheduler.summary()["valid"]


def test_partial_setup_failure_retains_prior_actor_for_cleanup():
    scheduler, world, _, actors = scheduler_fixture("C05_crossing_blocked_train_s11")
    spawn = world.try_spawn_actor.side_effect
    world.try_spawn_actor.side_effect = lambda bp, tf: spawn(bp, tf) if not actors else None
    with pytest.raises(RuntimeError, match="invalid"):
        scheduler.setup()
    assert scheduler.destroy()
    actors[0].destroy.assert_called_once()
    assert scheduler.has_failed()


def test_route_completion_with_unexecuted_event_is_invalid_experiment():
    loop = fake_loop(complete_at=.5)
    scheduler, _, _, _ = scheduler_fixture()
    scheduler.setup()
    loop.event_scheduler = scheduler
    loop.run()
    assert loop.route_completed
    assert loop.run_status == "scenario_invalid"
    assert not loop.outcome()["success"]


def test_empty_events_are_valid_for_regular_and_weather_cases():
    scheduler = PaperEventScheduler(Mock(), Mock(), Mock(), [], elapsed=lambda: 0)
    scheduler.setup()
    assert scheduler.summary() == {"required": 0, "executed": 0, "valid": True, "events": []}


def test_both_carla_random_generators_are_seeded():
    world, client = Mock(), Mock()
    world.get_weather.return_value.sun_altitude_angle = 70.
    spawner = TrafficSpawner(world, client, Mock())
    spawner.configure_reproducibility(101, [])
    client.get_trafficmanager.return_value.set_random_device_seed.assert_called_once_with(101)
    world.set_pedestrians_seed.assert_called_once_with(101)


def test_summary_keeps_failed_attempts_with_no_report(tmp_path):
    cfg = cases("eval")["R05_multilane_eval_s101"]
    run = tmp_path / "runs" / "failed"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "failed", "status": "failed",
        "scenario_yaml": yaml.safe_dump(cfg), "config": {}, "mode": "memory_off"}), encoding="utf-8")
    output = tmp_path / "results.csv"
    assert summarize_runs(tmp_path / "runs", output) == 1
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig")))
    assert rows[0]["status"] == "failed"
    assert rows[0]["case_id"] == "R05_multilane"
    assert rows[0]["case_config_sha256"] == digest(cfg)
    with pytest.raises(FileExistsError):
        summarize_runs(tmp_path / "runs", output)


def test_geometry_preflight_uses_actual_sidewalks_and_lawful_adjacent_lane(monkeypatch):
    from carla_bridge.paper_suite import prepare_profile
    catalog = read_catalog()
    profile = copy.deepcopy(catalog["profiles"]["town05"])
    profile["start_candidates"] = [0]
    client, world, road = Mock(), Mock(), Mock()
    client.load_world.return_value = world
    client.get_server_version.return_value = "0.9.15"
    world.get_settings.return_value.synchronous_mode = True
    world.get_map.return_value = road
    world.get_blueprint_library.return_value.filter.return_value = [Mock()]
    road.get_spawn_points.return_value = [carla.Transform(carla.Location(z=.3))]
    road.to_opendrive.return_value = "<OpenDRIVE/>"
    def waypoint(p, project_to_road=True, lane_type=carla.LaneType.Driving):
        if lane_type == carla.LaneType.Sidewalk:
            return NS(transform=carla.Transform(carla.Location(x=p.x, y=p.y, z=.2))) if abs(p.y) >= 4 else None
        return NS(transform=carla.Transform(p), is_junction=False, lane_id=1, road_id=1,
                  lane_change=carla.LaneChange.Both,
                  get_left_lane=lambda: NS(lane_id=2, road_id=1, lane_type=carla.LaneType.Driving),
                  get_right_lane=lambda: None)
    road.get_waypoint.side_effect = waypoint
    points = [carla.Location(x=x) for x in range(0, 901, 2)]
    monkeypatch.setattr(RoutePlanner, "plan", lambda *a, **kw: points)
    result = prepare_profile(carla, client, profile, catalog["defaults"])
    assert result["length_m"] == 900
    assert result["anchors"]["crossing"]["start"]["y"] == 4
    assert result["anchors"]["crossing"]["target"]["y"] == -4
    assert result["anchors"]["blocked_lane"]["spawn"]["x"] == 500
    world.try_spawn_actor.assert_not_called()
    world.spawn_actor.assert_not_called()
    # Removing the adjacent lane invalidates the route instead of placing a
    # supposedly bypassable obstacle on a one-lane road.
    original = road.get_waypoint.side_effect
    def no_lane(*args, **kwargs):
        wp = original(*args, **kwargs)
        if wp is not None and hasattr(wp, "lane_change"):
            wp.lane_change = carla.LaneChange.NONE
        return wp
    road.get_waypoint.side_effect = no_lane
    with pytest.raises(ValueError, match="blocked_lane"):
        prepare_profile(carla, client, profile, catalog["defaults"])


def test_paper_ego_spawn_failure_never_falls_back_to_another_start():
    cfg = cases()["R05_multilane_train_s11"]
    manager = ScenarioManager(Mock(), {}, cfg)
    manager.world = Mock()
    manager.world.try_spawn_actor.return_value = None
    with pytest.raises(RuntimeError):
        manager._spawn_ego()
    manager.world.try_spawn_actor.assert_called_once()
    manager.world.get_map.assert_not_called()


def test_event_journal_keeps_failure_evidence_and_cleanup_survives_write_error(tmp_path):
    scheduler, _, _, actors = scheduler_fixture()
    scheduler.audit_path = tmp_path / "events.json"
    scheduler.setup()
    assert json.loads(scheduler.audit_path.read_text())["events"][0]["state"] == "waiting"
    actors[0].get_location.side_effect = RuntimeError("lost actor")
    scheduler.tick()
    assert json.loads(scheduler.audit_path.read_text())["events"][0]["state"] == "failed"
    scheduler._write_journal = Mock(side_effect=OSError("full disk"))
    with pytest.raises(OSError):
        scheduler.destroy()
    actors[0].destroy.assert_called_once()


@pytest.mark.parametrize("split", ["train", "eval"])
def test_paper_entry_enforces_eval_readonly_and_applies_prepared_time_budget(tmp_path, monkeypatch, split):
    from carla_bridge import run_carla_demo as entry
    helper = session_at(tmp_path)
    scenario = next(iter(cases(split).values()))
    helper.scenario_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    monkeypatch.setattr(entry, "load_config", Mock(return_value=helper.config))
    monkeypatch.setattr(entry, "set_log_directory", Mock())
    monkeypatch.setattr(entry, "setup_logger", Mock())
    for block in ("scene_understanding", "decision"):
        monkeypatch.setenv(helper.config.get_nested(block, "api_key_env", default="DASHSCOPE_API_KEY"), "test-only")
    loop = Mock()
    loop.setup.side_effect = RuntimeError("intentional test stop before server")
    factory = Mock(return_value=loop)
    monkeypatch.setattr(entry, "ClosedLoop", factory)
    monkeypatch.setattr("sys.argv", ["run", "--scenario", str(helper.scenario_path)])
    if split == "eval":
        with pytest.raises(ValueError, match="read_only"):
            entry.main()
        factory.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="intentional test stop"):
            entry.main()
        assert helper.config.data["carla"]["max_duration_s"] == 900
        loop.close.assert_called_once()
