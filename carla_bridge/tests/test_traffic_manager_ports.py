"""Independent CARLA instances must never share the default TM endpoint."""
import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, call

import carla
import pytest

from carla_bridge.env import scenario_manager as scenarios
from carla_bridge.env.traffic_spawner import TrafficSpawner, resolve_traffic_manager_port


@pytest.mark.parametrize("rpc, expected", [(2000, 8000), (2200, 8200), (2310, 8310), (59535, 65535)])
def test_default_tm_endpoint_is_deterministic_per_rpc_instance(rpc, expected):
    assert resolve_traffic_manager_port(rpc) == expected


def test_explicit_override_supports_high_rpc_ports_without_wrapping():
    assert resolve_traffic_manager_port(65535, 9000) == 9000
    assert resolve_traffic_manager_port("2200", "9001") == 9001
    with pytest.raises(ValueError, match="traffic_manager_port"):
        resolve_traffic_manager_port(59536)


@pytest.mark.parametrize("value", [0, -1, 65536, True, False, 8000.5, 8000.0, "invalid", ""])
def test_invalid_explicit_tm_ports_fail_before_any_native_calls(value):
    client = Mock()
    with pytest.raises(ValueError, match="traffic_manager_port"):
        TrafficSpawner(Mock(), client, Mock(), tm_port=value)
    client.get_trafficmanager.assert_not_called()
    with pytest.raises(ValueError, match="traffic_manager_port"):
        resolve_traffic_manager_port(2000, value)


@pytest.mark.parametrize("value", [0, 65536, True, 2200.5])
def test_rpc_port_is_validated_even_when_tm_override_is_valid(value):
    with pytest.raises(ValueError, match="RPC port"):
        resolve_traffic_manager_port(value, 9000)


def test_bind_error_identifies_selected_port_and_never_retries_another_port():
    client = Mock()
    original = RuntimeError("rpc server bind error")
    client.get_trafficmanager.side_effect = original
    with pytest.raises(RuntimeError, match="8310.*bind error") as caught:
        TrafficSpawner(Mock(), client, Mock(), tm_port=8310)
    assert caught.value.__cause__ is original
    assert "No port fallback or process takeover" in str(caught.value)
    client.get_trafficmanager.assert_called_once_with(8310)


def test_sync_initialization_failure_is_not_silently_accepted():
    client = Mock()
    client.get_trafficmanager.return_value.set_synchronous_mode.side_effect = RuntimeError("TM unavailable")
    with pytest.raises(RuntimeError, match="8200.*TM unavailable"):
        TrafficSpawner(Mock(), client, Mock(), tm_port=8200)
    client.get_trafficmanager.assert_called_once_with(8200)
    client.get_trafficmanager.return_value.set_synchronous_mode.assert_called_once_with(True)


@pytest.mark.parametrize("tm_port", [8000, 8200, 8310, 9001])
def test_spawn_unregister_and_cleanup_use_the_same_selected_tm_port(tm_port):
    world, client, ego, vehicle = Mock(), Mock(), Mock(), Mock()
    ego.get_location.return_value = carla.Location()
    vehicle.is_alive = True
    vehicle.destroy.return_value = True
    bp = Mock(id="vehicle.test")
    bp.has_attribute.return_value = False
    world.get_blueprint_library.return_value.filter.return_value = [bp]
    world.get_map.return_value.get_spawn_points.return_value = [carla.Transform(carla.Location(x=10))]
    world.spawn_actor.return_value = vehicle
    spawner = TrafficSpawner(world, client, ego, seed=101, tm_port=tm_port)
    assert spawner.spawn_vehicles(1) == 1
    assert spawner.destroy() is True
    assert vehicle.set_autopilot.call_args_list == [call(True, tm_port), call(False, tm_port)]
    assert client.get_trafficmanager.return_value.set_synchronous_mode.call_args_list == [call(True), call(False)]
    client.get_trafficmanager.assert_called_once_with(tm_port)


@pytest.mark.parametrize("paper", [False, True])
@pytest.mark.parametrize("rpc, override, expected", [(2000, None, 8000), (2200, None, 8200),
                                                    (2310, None, 8310), (2310, 9001, 9001)])
def test_scenario_passes_actual_connection_port_and_preserves_seed_counts_and_snapshot(
        monkeypatch, tmp_path, paper, rpc, override, expected):
    world, native = Mock(), Mock()
    wrapper = NS(client=native, port=rpc, load_map=Mock(return_value=world), tick=Mock())
    native.get_server_version.return_value = "0.9.15"
    world.get_map.return_value.to_opendrive.return_value = "<OpenDRIVE/>"
    world.get_map.return_value.name = "Carla/Maps/Town05"
    world.get_weather.return_value = NS(**{key: 0.0 for key in (
        "cloudiness", "precipitation", "precipitation_deposits", "sun_altitude_angle",
        "sun_azimuth_angle", "fog_density", "fog_distance", "wetness")})
    cfg = {"scenario_name": "test", "map": "Town05", "seed": 101,
           "route": {"points": []}, "traffic": {"num_vehicles": 3, "num_walkers": 2}}
    if paper:
        cfg["experiment"] = {"suite": "paper_v1", "server_version": "0.9.15",
            "map_sha256": hashlib.sha256(b"<OpenDRIVE/>").hexdigest()}
        monkeypatch.setattr("carla_bridge.paper_suite.validate_prepared", Mock())
    ccfg = {"port": 2000, "experiment_output_dir": str(tmp_path)}  # May differ from CLI connection.
    if override is not None:
        ccfg["traffic_manager_port"] = override
    route = Mock()
    route._progress = NS(started=True, lengths=[0, 100])
    monkeypatch.setattr(scenarios, "RoutePlanner", Mock(return_value=route))
    monkeypatch.setattr(scenarios, "WeatherController", Mock())
    manager = scenarios.ScenarioManager(wrapper, ccfg, cfg)
    ego = Mock(id=5)
    monkeypatch.setattr(manager, "_spawn_ego", Mock(return_value=(ego, carla.Transform())))
    spawner = Mock(spawn_manifest={"vehicles": [{"id": i} for i in range(3)],
                                  "walkers": [{"id": i + 3} for i in range(2)]})
    spawner.spawn_vehicles.return_value = 3
    spawner.spawn_walkers.return_value = 2

    def make_spawner(*args, **kwargs):
        spawner.tm_port = kwargs["tm_port"]
        return spawner

    factory = Mock(side_effect=make_spawner)
    monkeypatch.setattr(scenarios, "TrafficSpawner", factory)
    assert manager.setup() is manager
    assert manager.traffic_manager_port == expected
    factory.assert_called_once_with(world, native, ego, seed=101, tm_port=expected)
    spawner.spawn_vehicles.assert_called_once_with(3, radius=80.0)
    spawner.spawn_walkers.assert_called_once_with(2, radius=60.0)
    if paper:
        spawner.configure_reproducibility.assert_called_once_with(101, [])
        assert manager.experiment_snapshot["traffic_manager_port"] == expected
        assert manager.experiment_snapshot["seed"] == 101
        snapshot = json.loads((tmp_path / "experiment_setup.json").read_text(encoding="utf-8"))
        assert snapshot["traffic_manager_port"] == expected
        assert (snapshot["requested_vehicles"], snapshot["spawned_vehicles"]) == (3, 3)
        assert (snapshot["requested_walkers"], snapshot["spawned_walkers"]) == (2, 2)


def test_scenario_rejects_invalid_tm_config_before_loading_map_or_ticking():
    wrapper = NS(client=Mock(), port=2200, load_map=Mock(), tick=Mock())
    manager = scenarios.ScenarioManager(wrapper, {"traffic_manager_port": 0}, {"map": "Town05"})
    with pytest.raises(ValueError, match="traffic_manager_port"):
        manager.setup()
    wrapper.load_map.assert_not_called()
    wrapper.tick.assert_not_called()
    wrapper.client.get_trafficmanager.assert_not_called()
