"""Spawn preflight contracts using synthetic CARLA maps; no running server."""
import copy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import carla
import pytest

from carla_bridge import paper_suite as suite
from carla_bridge.env.paper_events import event_blueprint
from carla_bridge.env.route_planner import RoutePlanner


COLLISION = "Spawn failed because of collision at spawn position"


def map_fixture():
    road = Mock()

    def waypoint(p, project_to_road=True, lane_type=carla.LaneType.Driving):
        if lane_type == carla.LaneType.Sidewalk:
            if not 4 <= abs(p.y) <= 8:
                return None
            return NS(transform=carla.Transform(carla.Location(x=p.x, y=p.y, z=.2)),
                      road_id=1, section_id=0, lane_id=3 if p.y > 0 else -3,
                      lane_type=carla.LaneType.Sidewalk)
        return NS(transform=carla.Transform(p), is_junction=False,
                  road_id=1, section_id=0, lane_id=1,
                  lane_type=carla.LaneType.Driving, lane_change=carla.LaneChange.Both,
                  get_left_lane=lambda: NS(road_id=1, lane_id=2, lane_type=carla.LaneType.Driving),
                  get_right_lane=lambda: None)

    road.get_waypoint.side_effect = waypoint
    world = Mock()
    world.get_map.return_value = road
    library = world.get_blueprint_library.return_value
    library.filter.return_value = [Mock(id="walker.pedestrian.0002"), Mock(id="walker.pedestrian.0001")]
    library.find.side_effect = lambda name: Mock(id=name)
    actors = []

    def spawn(bp, tf):
        actor = Mock()
        actor.destroy.return_value = True
        actors.append(actor)
        return actor

    world.spawn_actor.side_effect = spawn
    points = [carla.Location(x=x) for x in range(0, 101, 2)]
    arc = [float(x) for x in range(0, 101, 2)]
    spec = {"type": "pedestrian_crossing", "search_from_m": 20., "search_to_m": 30.,
            "speed_mps": 1.4, "lead_distance_m": 10.}
    return world, road, points, arc, spec, actors


def test_spawn_probe_success_initializes_snapshot_before_destroy():
    world, _, _, _, _, actors = map_fixture()
    bp, tf = Mock(id="walker.pedestrian.0001"), carla.Transform(carla.Location(z=1.2))
    result = suite._probe_spawn(world, bp, tf)
    assert result["ok"] is True
    world.spawn_actor.assert_called_once_with(bp, tf)
    actors[0].destroy.assert_called_once()
    world.tick.assert_called_once()
    world.wait_for_tick.assert_not_called()


def test_spawn_collision_is_rejected_with_native_error():
    world = Mock()
    world.spawn_actor.side_effect = RuntimeError(COLLISION)
    result = suite._probe_spawn(world, Mock(id="walker.pedestrian.0001"), carla.Transform())
    assert result["ok"] is False
    assert COLLISION in result["error"]


def test_spawn_none_is_not_success():
    world = Mock()
    world.spawn_actor.return_value = None
    assert suite._probe_spawn(world, Mock(id="walker.pedestrian.0001"), carla.Transform())["ok"] is False


def test_new_actor_destroy_does_not_use_stale_pre_spawn_snapshot():
    world = Mock()
    actor = world.spawn_actor.return_value
    actor.id, actor.is_alive = 1422, True
    world.get_settings.return_value.synchronous_mode = True
    initialized = []
    world.tick.side_effect = lambda: initialized.append(True)
    actor.destroy.side_effect = lambda: bool(initialized)
    result = suite._probe_spawn(world, Mock(), carla.Transform())
    assert result["ok"]
    assert initialized == [True]
    actor.destroy.assert_called_once()


def test_async_probe_waits_for_snapshot_without_forcing_ticks():
    world = Mock()
    world.get_settings.return_value.synchronous_mode = False
    world.spawn_actor.return_value.destroy.return_value = True
    assert suite._probe_spawn(world, Mock(), carla.Transform())["ok"]
    world.wait_for_tick.assert_called_once_with(30.0)
    world.tick.assert_not_called()


def test_actor_missing_after_initialization_is_not_a_valid_spawn():
    world = Mock()
    world.get_snapshot.return_value.find.return_value = None
    result = suite._probe_spawn(world, Mock(), carla.Transform())
    assert result["ok"] is False
    assert "initialization" in result["error"]
    world.spawn_actor.return_value.destroy.assert_not_called()


def test_destroy_false_accepts_confirmed_removal_in_new_snapshot():
    world = Mock()
    before, after = Mock(), Mock()
    after.find.return_value = None
    world.get_snapshot.side_effect = [before, after]
    world.spawn_actor.return_value.destroy.return_value = False
    assert suite._probe_spawn(world, Mock(), carla.Transform())["ok"]
    assert world.tick.call_count == 2


def test_pending_kill_actor_is_not_a_valid_spawn_even_when_cleanup_succeeds():
    world = Mock()
    actor = world.spawn_actor.return_value
    actor.is_alive = False
    actor.destroy.return_value = True
    assert not suite._probe_spawn(world, Mock(), carla.Transform())["ok"]
    actor.destroy.assert_called_once()


def test_snapshot_timeout_aborts_instead_of_trying_different_geometry():
    world = Mock()
    world.tick.side_effect = RuntimeError("snapshot timeout")
    world.spawn_actor.return_value.destroy.return_value = True
    with pytest.raises(suite.SpawnProbeError, match="snapshot timeout"):
        suite._probe_spawn(world, Mock(), carla.Transform())
    world.spawn_actor.return_value.destroy.assert_called_once()


def test_spawn_connection_error_is_not_treated_as_bad_geometry():
    world = Mock()
    world.spawn_actor.side_effect = RuntimeError("time-out of 120000ms while waiting for the simulator")
    with pytest.raises(RuntimeError, match="time-out"):
        suite._probe_spawn(world, Mock(id="walker.pedestrian.0001"), carla.Transform())


@pytest.mark.parametrize("destroy_result", [False, RuntimeError("destroy RPC failed")])
def test_spawn_cleanup_failure_is_fatal(destroy_result):
    world = Mock()
    actor = world.spawn_actor.return_value
    if isinstance(destroy_result, Exception):
        actor.destroy.side_effect = destroy_result
    else:
        actor.destroy.return_value = destroy_result
    with pytest.raises(suite.SpawnProbeCleanupError):
        suite._probe_spawn(world, Mock(id="walker.pedestrian.0001"), carla.Transform())
    world.spawn_actor.assert_called_once()
    actor.destroy.assert_called_once()


def test_sidewalk_candidates_keep_clearance_from_both_lane_edges():
    _, road, points, _, _, _ = map_fixture()
    wp = road.get_waypoint(points[10])
    for side in (1, -1):
        candidates = suite._sidewalk_candidates(carla, road, wp, side)
        assert candidates
        assert [round(abs(p.y), 2) for p in candidates] == [4.5, 5., 5.5, 6., 6.5, 7., 7.5]
        assert all(p.z == pytest.approx(.2) for p in candidates)


def test_sidewalk_margin_cannot_cross_to_different_sidewalk_lane():
    _, road, points, _, _, _ = map_fixture()
    original = road.get_waypoint.side_effect

    def split_sidewalk(p, **kwargs):
        wp = original(p, **kwargs)
        if wp is not None and kwargs.get("lane_type") == carla.LaneType.Sidewalk and abs(p.y) >= 6:
            wp.lane_id *= 2
        return wp

    road.get_waypoint.side_effect = split_sidewalk
    candidates = suite._sidewalk_candidates(carla, road, road.get_waypoint(points[10]), 1)
    assert 5.5 not in [p.y for p in candidates]
    assert 6.0 not in [p.y for p in candidates]


def test_resolver_tries_interior_candidates_and_probes_both_frozen_endpoints():
    world, _, points, arc, spec, actors = map_fixture()
    successful_spawn = world.spawn_actor.side_effect
    attempts = []

    def blocked_curb(bp, tf):
        attempts.append((bp.id, suite.point(tf.location)))
        if abs(tf.location.y) < 5:
            raise RuntimeError(COLLISION)
        return successful_spawn(bp, tf)

    world.spawn_actor.side_effect = blocked_curb
    event = suite._resolve_anchor(carla, world, points, arc, spec, "crossing")
    assert event["anchor_distance_m"] == 20.
    assert event["blueprint"] == "walker.pedestrian.0001"
    assert abs(event["start"]["y"]) >= 5 and abs(event["target"]["y"]) >= 5
    assert event["start"]["z"] == pytest.approx(1.4)
    assert event["target"]["z"] == pytest.approx(1.4)
    proof = event["spawn_validation"]
    assert proof["method"] == "actor_spawn_probe_v1"
    assert proof["blueprint"] == event["blueprint"]
    assert any(pose == event["start"] for _, pose in attempts)
    assert any(pose == event["target"] for _, pose in attempts)
    assert {identity for identity, _ in attempts} == {event["blueprint"]}
    assert all(actor.destroy.call_count == 1 for actor in actors)
    assert world.tick.call_count == len(actors)
    world.wait_for_tick.assert_not_called()


def test_crossing_with_unspawnable_target_is_rejected_instead_of_omitted():
    world, _, points, arc, spec, actors = map_fixture()
    successful_spawn = world.spawn_actor.side_effect

    def blocked_other_side(bp, tf):
        if tf.location.y < 0:
            raise RuntimeError(COLLISION)
        return successful_spawn(bp, tf)

    world.spawn_actor.side_effect = blocked_other_side
    with pytest.raises(ValueError, match="crossing") as failure:
        suite._resolve_anchor(carla, world, points, arc, spec, "crossing")
    assert "collision" in str(failure.value)
    assert all(actor.destroy.call_count == 1 for actor in actors)


def test_stationary_vehicle_skips_collision_site_and_keeps_heading():
    world, _, points, arc, _, actors = map_fixture()
    spec = {"type": "stationary_vehicle", "blueprint": "vehicle.audi.etron",
            "search_from_m": 20., "search_to_m": 30.}
    successful_spawn = world.spawn_actor.side_effect

    def blocked_first_station(bp, tf):
        if tf.location.x == 20:
            raise RuntimeError(COLLISION)
        return successful_spawn(bp, tf)

    world.spawn_actor.side_effect = blocked_first_station
    event = suite._resolve_anchor(carla, world, points, arc, spec, "blocked_lane")
    assert event["anchor_distance_m"] == 22
    assert event["spawn"]["yaw"] == 0
    assert event["spawn_validation"]["blueprint"] == "vehicle.audi.etron"
    assert all(actor.destroy.call_count == 1 for actor in actors)


def test_profile_does_not_try_another_route_after_cleanup_failure(monkeypatch):
    world, road, _, _, _, _ = map_fixture()
    catalog = suite.read_catalog()
    profile = copy.deepcopy(catalog["profiles"]["town05"])
    profile["start_candidates"] = [0, 1]
    client = Mock()
    client.load_world.return_value = world
    world.get_settings.return_value.synchronous_mode = True
    road.get_spawn_points.return_value = [carla.Transform(), carla.Transform(carla.Location(x=2))]
    points = [carla.Location(x=x) for x in range(0, 901, 2)]
    plan = Mock(return_value=points)
    monkeypatch.setattr(RoutePlanner, "plan", plan)
    actor = Mock()
    actor.destroy.return_value = False
    world.spawn_actor.side_effect = None
    world.spawn_actor.return_value = actor
    with pytest.raises(suite.SpawnProbeCleanupError):
        suite.prepare_profile(carla, client, profile, catalog["defaults"])
    plan.assert_called_once()
    world.spawn_actor.assert_called_once()


def test_frozen_pedestrian_blueprint_is_selected_exactly_and_not_invincible():
    library = Mock()
    bp = library.find.return_value
    bp.has_attribute.return_value = True
    actual = event_blueprint(library, {"type": "pedestrian_crossing", "blueprint": "walker.pedestrian.0020"})
    assert actual is bp
    library.find.assert_called_once_with("walker.pedestrian.0020")
    library.filter.assert_not_called()
    bp.set_attribute.assert_called_once_with("is_invincible", "false")


def test_legacy_crossing_keeps_deterministic_blueprint_selection():
    library = Mock()
    first, later = Mock(id="walker.pedestrian.0001"), Mock(id="walker.pedestrian.0020")
    library.filter.return_value = [later, first]
    assert event_blueprint(library, {"type": "pedestrian_crossing"}) is first
    library.find.assert_not_called()


def test_frozen_pedestrian_blueprint_cannot_be_replaced_by_vehicle():
    library = Mock()
    with pytest.raises(ValueError, match="walker"):
        event_blueprint(library, {"type": "pedestrian_crossing", "blueprint": "vehicle.audi.etron"})
    library.find.assert_not_called()
