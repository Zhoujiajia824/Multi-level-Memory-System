"""R05's last exit has an intermediate junction and a short legal return lane."""
import math
import os
from pathlib import Path
from types import SimpleNamespace

import carla
import pytest

from carla_bridge.tests.test_pdm_planner import Vehicle, add_scene_context_fixture


TOWN05 = Path(os.environ.get("CARLA_TOWN05_XODR", ""))


@pytest.mark.skipif(not TOWN05.is_file(), reason="set CARLA_TOWN05_XODR for real-map return regression")
@pytest.mark.parametrize("y,speed", [(118.94, 7.), (156., 4.), (118.94, 12.)])
def test_last_r05_exit_uses_short_legal_corridor_after_the_intermediate_junction(y, speed):
    pytest.importorskip("tuplan_garage")
    pytest.importorskip("nuplan")
    from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
    from carla_bridge.planning.route_waypoints import resolve_route_waypoints
    from carla_bridge.control.trajectory_validation import TrajectoryValidator

    road = carla.Map("Town05", TOWN05.read_text())
    # Recorded R05 final approach: 28 -> 2083 (straight junction) -> 27 ->
    # 2353 (required right turn). The 30 m return overlaps junction 2083;
    # road 27 itself has a shorter, genuinely broken-line return corridor.
    xy = [(24.56948 - .0004 * (2 * i), 106.59671 + 2 * i) for i in range(37)]
    xy += [(24.51324,180.44435),(24.19544,182.01494),(23.53488,183.47488),
           (22.56493,184.7504),(21.30518,185.80922),(19.82754,186.66284),
           (18.22882,187.25961),(16.55326,187.58301)]
    points = [carla.Location(x=x, y=y, z=.00002) for x, y in xy]
    waypoints, _ = resolve_route_waypoints(road, points)
    ego = Vehicle(x=28.064, y=y, z=.0018, yaw=90.02, speed=speed)
    ego.bounding_box = carla.BoundingBox(carla.Location(x=.0292188,z=.73586),
                                        carla.Vector3D(x=2.3958898,y=1.081725,z=.75))
    ego.wheelbase_m, ego.max_steer_deg, ego.rear_axle_local_x_m = 3.0046469567, 70., -1.3864
    ego.get_velocity = lambda: carla.Vector3D(
        x=ego.speed*math.cos(math.radians(ego.yaw)), y=ego.speed*math.sin(math.radians(ego.yaw)))
    world = SimpleNamespace(get_map=lambda:road, get_actors=lambda:[ego],
        get_snapshot=lambda:SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=2.)))
    route = SimpleNamespace(_route=points, _route_waypoints=waypoints,
        _nearest_index=lambda location:min(range(len(points)),key=lambda i:points[i].distance(location)))
    add_scene_context_fixture(world,ego,route)
    planner = CarlaPDMPlanner(world,ego,route,{"lateral_control":"native_lqr"})
    plan = planner.plan(None,semantic_enabled=False)
    diagnostics = plan["planner_diagnostics"]
    left = diagnostics["geometry"]["lateral_paths"]["left"]
    connection = left["route_connection"]
    if speed == 12.:
        # At this speed a 16 m return is insufficient. Keep a finite stop
        # corridor instead of stretching a merge across junction 2083.
        assert connection["reason"] == "route_exit_has_no_legal_return"
        assert connection["minimum_return_length_m"] == pytest.approx(24.)
        assert connection["finite_stopping_corridor_retained"]
        return
    assert connection.get("method") == "legal_return_in_spatial_proposal", connection
    assert 10.5 <= connection["return_length_m"] < 30.
    assert connection["legal_return_lane_keys"] == ["27:0:1"]
    assert connection["selection_gate_applied"] is False
    assert left["path_length_m"] > 40.  # No invented IDM leader at lane 27's end.
    validation = TrajectoryValidator().validate(plan["trajectory"],plan["target_speed"],speed,3.,use_speed_profile=True)
    assert validation.accepted,validation.to_dict()
