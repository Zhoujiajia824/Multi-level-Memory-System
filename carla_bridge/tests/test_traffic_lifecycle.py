"""CARLA traffic regressions: no server or native navigation calls are made."""
from unittest.mock import Mock

import carla
import pytest

from carla_bridge.env.event_scheduler import EventScheduler
from carla_bridge.env.carla_client import CarlaClient
from carla_bridge.env.scenario_manager import ScenarioManager
from carla_bridge.env.traffic_spawner import TrafficSpawner
from carla_bridge.env.walker_controller import WalkerController


def actor(name, events):
    obj = Mock(name=name)
    obj.is_alive = True

    def destroy():
        events.append((name, "destroy"))
        obj.is_alive = False
        return True

    obj.destroy.side_effect = destroy
    obj.stop.side_effect = lambda: events.append((name, "stop"))
    obj.start.side_effect = lambda: events.append((name, "start"))
    obj.set_autopilot.side_effect = lambda enabled, port: events.append(
        (name, "autopilot", enabled, port))
    return obj


def traffic_fixture():
    events = []
    world, client, ego = Mock(), Mock(), Mock()
    ego.get_location.return_value = carla.Location()
    world.get_settings.return_value.synchronous_mode = True
    world.tick.side_effect = lambda: events.append(("world", "tick"))
    world.get_blueprint_library.return_value.filter.return_value = [Mock()]
    spawner = TrafficSpawner(world, client, ego)
    spawner.tm.set_synchronous_mode.side_effect = lambda sync: events.append(("tm", sync))
    return spawner, world, events


def test_ai_walkers_use_navigation_and_start_after_snapshot():
    spawner, world, events = traffic_fixture()
    walker, controller = actor("walker", events), actor("controller", events)
    spawn, target = carla.Location(x=10, y=5), carla.Location(x=20, y=5)
    world.get_random_location_from_navigation.side_effect = [spawn, target]
    world.spawn_actor.side_effect = [walker, controller]

    assert spawner.spawn_walkers(1) == 1
    assert events == [("world", "tick"), ("controller", "start")]
    assert world.spawn_actor.call_args_list[0].args[1].location.distance(spawn) == 0
    assert world.spawn_actor.call_args_list[1].args[1].location.distance(carla.Location()) == 0
    controller.go_to_location.assert_called_once_with(target)
    world.get_map.assert_not_called()
    world.get_random_location_from_street.assert_not_called()


def test_partial_controller_spawn_rolls_back_registered_walker():
    spawner, world, events = traffic_fixture()
    walker = actor("walker", events)
    locations = iter([carla.Location(x=10), carla.Location(x=20)])
    world.get_random_location_from_navigation.side_effect = lambda: next(locations, None)
    world.spawn_actor.side_effect = [walker, RuntimeError("controller creation failed")]

    assert spawner.spawn_walkers(1) == 0
    walker.destroy.assert_called_once()
    assert not spawner._walkers and not spawner._walker_controllers
    world.tick.assert_not_called()


def test_controller_start_failure_stops_and_destroys_before_parent():
    spawner, world, events = traffic_fixture()
    walker, controller = actor("walker", events), actor("controller", events)
    world.get_random_location_from_navigation.side_effect = [carla.Location(x=10), carla.Location(x=20)]
    world.spawn_actor.side_effect = [walker, controller]
    controller.start.side_effect = RuntimeError("start failed")

    assert spawner.spawn_walkers(1) == 0
    assert events == [("world", "tick"), ("controller", "stop"),
                      ("controller", "destroy"), ("walker", "destroy")]
    assert not spawner._walkers and not spawner._walker_controllers


def test_snapshot_failure_leaves_actors_registered_for_final_cleanup():
    spawner, world, events = traffic_fixture()
    walker, controller = actor("walker", events), actor("controller", events)
    world.get_random_location_from_navigation.side_effect = [carla.Location(x=10), carla.Location(x=20)]
    world.spawn_actor.side_effect = [walker, controller]
    world.tick.side_effect = RuntimeError("snapshot timeout")
    with pytest.raises(RuntimeError, match="snapshot timeout"):
        spawner.spawn_walkers(1)
    controller.start.assert_not_called()
    spawner.destroy()
    assert events == [("controller", "stop"), ("controller", "destroy"),
                      ("walker", "destroy"), ("tm", False)]


def test_missing_navigation_never_falls_back_to_driving_lane_or_starts_ai():
    spawner, world, events = traffic_fixture()
    world.get_random_location_from_navigation.return_value = None
    assert spawner.spawn_walkers(1) == 0
    world.spawn_actor.assert_not_called()
    world.get_map.assert_not_called()
    assert not events


def test_all_vehicles_unregistered_before_any_actor_destroy_and_tm_resume():
    spawner, _, events = traffic_fixture()
    spawner._vehicles = [actor("vehicle1", events), actor("vehicle2", events)]
    spawner._walkers = [actor("walker", events)]
    spawner._walker_controllers = [actor("controller", events)]
    assert spawner.destroy() is True
    assert spawner.destroy() is True
    assert events == [("vehicle1", "autopilot", False, 8000),
                      ("vehicle2", "autopilot", False, 8000),
                      ("controller", "stop"), ("controller", "destroy"),
                      ("walker", "destroy"), ("vehicle1", "destroy"),
                      ("vehicle2", "destroy"), ("tm", False)]


def test_actor_cleanup_failure_does_not_skip_other_actors():
    spawner, _, events = traffic_fixture()
    first, second = actor("first", events), actor("second", events)
    first.set_autopilot.side_effect = RuntimeError("unregister failed")
    first.destroy.side_effect = RuntimeError("destroy failed")
    spawner._vehicles = [first, second]
    assert spawner.destroy() is False
    second.set_autopilot.assert_called_once_with(False, 8000)
    second.destroy.assert_called_once()
    first.destroy.assert_not_called()
    assert spawner._vehicles == [first]
    assert ("tm", False) not in events
    first.set_autopilot.side_effect = None
    first.destroy.side_effect = None
    first.destroy.return_value = True
    assert spawner.destroy() is True
    assert spawner._vehicles == []
    assert events[-1] == ("tm", False)


def test_event_walker_is_scripted_moves_then_stops_without_extra_world_ticks():
    events = []
    world = Mock()
    world.get_blueprint_library.return_value.filter.return_value = [Mock()]
    walker = actor("walker", events)
    world.spawn_actor.return_value = walker
    controller = WalkerController(world)
    start, target = carla.Location(x=20, y=8), carla.Location(x=20, y=-8)
    assert controller._spawn_walker(start, target) is walker
    walker.get_location.assert_not_called()
    world.get_blueprint_library.return_value.find.assert_not_called()
    walker.get_location.return_value = start

    controller.tick()
    moving = walker.apply_control.call_args.args[0]
    assert moving.speed > 0 and moving.direction.y == -1
    walker.get_location.return_value = carla.Location(x=20, y=3)
    controller.tick()
    assert walker.apply_control.call_args.args[0].speed == moving.speed
    walker.get_location.return_value = carla.Location(x=20, y=-8.5)
    controller.tick()
    assert walker.apply_control.call_args.args[0].speed == 0
    assert not controller._motions
    controller.destroy()
    controller.destroy()
    walker.destroy.assert_called_once()
    world.tick.assert_not_called()
    world.wait_for_tick.assert_not_called()


def test_scheduler_does_not_read_new_walker_until_next_regular_frame():
    events = []
    world, ego = Mock(), Mock()
    world.get_blueprint_library.return_value.filter.return_value = [Mock()]
    walker = actor("walker", events)
    world.spawn_actor.return_value = walker
    ego.get_location.return_value = carla.Location()
    ego.get_transform.return_value = carla.Transform()
    scheduler = EventScheduler(world, ego, [{"type": "pedestrian_crossing", "at_distance": 0}], {})
    scheduler.tick()
    walker.get_location.assert_not_called()
    walker.apply_control.assert_not_called()
    walker.get_location.return_value = carla.Location(x=20, y=8)
    scheduler.tick()
    assert walker.apply_control.call_args.args[0].direction.y == -1
    world.tick.assert_not_called()
    scheduler.destroy()
    walker.destroy.assert_called_once()


def test_event_walker_failed_location_read_stops_previous_motion():
    world = Mock()
    world.get_blueprint_library.return_value.filter.return_value = [Mock()]
    walker = actor("walker", [])
    world.spawn_actor.return_value = walker
    controller = WalkerController(world)
    controller._spawn_walker(carla.Location(x=20, y=8), carla.Location(x=20, y=-8))
    walker.get_location.side_effect = RuntimeError("actor state unavailable")
    controller.tick()
    assert walker.apply_control.call_args.args[0].speed == 0
    assert not controller._motions


def test_traffic_pending_walker_defers_tm_restore_and_reports_failure():
    spawner, _, events = traffic_fixture()
    walker = actor("walker", events)
    walker.destroy.side_effect = None
    walker.destroy.return_value = False
    spawner._walkers = [walker]
    assert spawner.destroy() is False
    assert spawner._walkers == [walker]
    assert ("tm", False) not in events
    walker.destroy.return_value = True
    assert spawner.destroy() is True
    assert events == [("tm", False)]


def test_traffic_tm_restore_failure_is_reported_and_retried():
    spawner, _, _ = traffic_fixture()
    spawner.tm.set_synchronous_mode.side_effect = RuntimeError("TM timeout")
    assert spawner.destroy() is False
    spawner.tm.set_synchronous_mode.side_effect = None
    assert spawner.destroy() is True
    assert spawner.destroy() is True
    assert spawner.tm.set_synchronous_mode.call_count == 3  # initial True + two restore attempts


@pytest.mark.parametrize("throws", [False, True])
def test_scenario_propagates_traffic_failure_and_still_cleans_ego(throws):
    manager = ScenarioManager(Mock(), {}, {})
    manager.traffic = Mock()
    manager.traffic.destroy.return_value = False
    if throws:
        manager.traffic.destroy.side_effect = RuntimeError("traffic cleanup failed")
    ego = actor("ego", [])
    manager.ego_vehicle = ego
    assert manager.destroy() is False
    ego.destroy.assert_called_once()
    assert manager.ego_vehicle is None
    manager.traffic.destroy.side_effect = None
    manager.traffic.destroy.return_value = True
    assert manager.destroy() is True
    ego.destroy.assert_called_once()


@pytest.mark.parametrize("throws", [False, True])
def test_scenario_retains_failed_ego_for_retry(throws):
    manager = ScenarioManager(Mock(), {}, {})
    ego = actor("ego", [])
    manager.ego_vehicle = ego
    ego.destroy.side_effect = RuntimeError("ego timeout") if throws else None
    ego.destroy.return_value = False
    assert manager.destroy() is False
    assert manager.ego_vehicle is ego
    ego.destroy.side_effect = None
    ego.destroy.return_value = True
    assert manager.destroy() is True
    assert manager.ego_vehicle is None


def test_world_restore_failure_is_reported_and_keeps_settings_for_retry():
    client = CarlaClient()
    client.world = Mock()
    settings = object()
    client._orig_settings = settings
    client.world.apply_settings.side_effect = RuntimeError("world unavailable")
    assert client.restore() is False
    assert client._orig_settings is settings
    client.world.apply_settings.side_effect = None
    assert client.restore() is True
    assert client.restore() is True
    assert client._orig_settings is None
    assert client.world.apply_settings.call_count == 2


def test_event_cleanup_propagates_walker_and_prop_failure_with_retry():
    world, ego = Mock(), Mock()
    ego.get_location.return_value = carla.Location()
    scheduler = EventScheduler(world, ego, [], {})
    walker, prop = actor("walker", []), actor("prop", [])
    scheduler.walker_ctrl._walkers = [walker]
    scheduler._static_props = [prop]
    walker.destroy.side_effect = RuntimeError("walker timeout")
    prop.destroy.side_effect = None
    prop.destroy.return_value = False
    assert scheduler.destroy() is False
    assert scheduler.walker_ctrl._walkers == [walker]
    assert scheduler._static_props == [prop]
    walker.destroy.side_effect = None
    walker.destroy.return_value = True
    prop.destroy.return_value = True
    assert scheduler.destroy() is True
    assert scheduler.destroy() is True
