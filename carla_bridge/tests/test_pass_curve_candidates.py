"""Frozen Town05 close-walker state through unmodified native PDM feedback.

This numerical receding-horizon test uses the actual failed run's state, map,
vehicle and steering feedback. It does not use a simulator, future actor GT,
candidate filters, alternate curvature extraction or tuned LQR previews.
"""
import hashlib
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from carla_bridge.tests.test_recovery_planner_integration import make_snapshot_planner
from carla_bridge.tests.test_return_curve_candidates import advance_native_feedback


@pytest.fixture
def short_pass_snapshot():
    if find_spec("tuplan_garage") is None or find_spec("nuplan") is None:
        pytest.skip("optional pinned PDM dependencies are not installed")
    import carla
    import yaml
    from carla_bridge.tests.test_pdm_planner import Vehicle
    fixture = Path(__file__).parent / "fixtures" / "town05_short_pass_native.json"
    snapshot = json.loads(fixture.read_text(encoding="utf-8"))
    path = Path(os.environ.get("CARLA_TEST_TOWN05_XODR",
        "D:/software/carla/WindowsNoEditor/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr"))
    if not path.is_file():
        pytest.skip("set CARLA_TEST_TOWN05_XODR to the CARLA 0.9.15 Town05 OpenDRIVE map")
    source = path.read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode()).hexdigest() == snapshot["map_sha256"]
    root = Path(__file__).resolve().parents[2]
    snapshot["planning"] = yaml.safe_load((root / "carla_bridge/config/carla.yaml").read_text(
        encoding="utf-8"))["carla"]["planning"]
    planner, clock = make_snapshot_planner((carla.Map("Town05", source), snapshot))
    start, end = snapshot["route_xy"]
    planner.route.plan(start_loc=carla.Location(x=start[0], y=start[1]),
                       end_loc=carla.Location(x=end[0], y=end[1]))
    clock.time = snapshot["timestamp_s"]
    ego, measured = planner.ego, snapshot["measured"]
    physics = ego.get_physics_control

    def actual_physics():
        result = physics()
        result.steering_curve = [carla.Vector2D(x=x, y=y)
            for x, y in snapshot["steering_curve_kmh_scale"]]
        return result

    ego.get_physics_control = actual_physics
    ego.get_speed_limit = lambda: 30.
    ego.get_velocity = lambda: carla.Vector3D(x=measured["velocity_x"], y=measured["velocity_y"])
    ego.get_acceleration = lambda: carla.Vector3D(x=measured["acceleration_x"], y=measured["acceleration_y"])
    ego.get_control = lambda: carla.VehicleControl(**snapshot["applied_control"])
    ego.get_wheel_steer_angle = lambda wheel: snapshot["steering_state"][
        "front_left_deg" if wheel == carla.VehicleWheelLocation.FL_Wheel else "front_right_deg"]
    value = snapshot["blocker"]
    walker = Vehicle(actor_id=int(value["id"]), x=value["x"], y=-value["y"],
                     z=.6, yaw=-math.degrees(value["heading"]))
    walker.type_id = "walker.pedestrian.0037"
    walker.bounding_box = carla.BoundingBox(carla.Location(),
        carla.Vector3D(x=value["length"]/2, y=value["width"]/2, z=.9))
    walker.get_velocity = lambda: carla.Vector3D(x=value["vx"], y=-value["vy"])
    planner.world.get_actors = lambda: [ego, walker]
    # Restore only observed persistence, not a fabricated future or maneuver.
    planner.lane_maneuver._observed = dict(id=walker.id, since=2., count=8)
    return planner, clock, walker


def footprint(actor):
    from shapely.geometry import MultiPoint
    return MultiPoint([(v.x, -v.y) for v in actor.bounding_box.get_world_vertices(
        actor.get_transform())]).convex_hull


def test_actual_close_walker_prefers_smooth_reachable_curve_and_retains_all_candidates(short_pass_snapshot):
    planner, clock, _ = short_pass_snapshot
    diagnostic = planner.plan(None, False, sim_time_s=clock.time)["planner_diagnostics"]
    maneuver = diagnostic["lane_maneuver"]
    assert maneuver["transition_length_m"] == 16.
    # Coalescing redundant map knots may change the finite-difference maximum
    # slightly. The smooth curve must still retain its physically useful cap.
    assert 4.3 < maneuver["speed_cap_mps"] < 4.4
    assert maneuver["passing_checks"]["speed_search_round"] == "maximum"
    assert any(a["length_m"] == 20. for a in maneuver["passing_checks"]["attempts"])
    assert diagnostic["safety_gate_applied"] is False
    assert {"center", "left"}.issubset({c["physical_lane_side"] for c in diagnostic["candidates"]})
    assert all(c["semantic_bonus"] == 0 for c in diagnostic["candidates"])


def test_real_return_station_roundoff_cannot_invent_a_one_mps_steering_limit(short_pass_snapshot):
    from carla_bridge.planning.lane_maneuver import _curvature
    planner, clock, _ = short_pass_snapshot
    planner.plan(None, False, sim_time_s=clock.time)
    engine = planner.lane_maneuver
    reference, stations = engine._last_reference, engine._last_stations
    # Frozen first-return anchor from physical_005923 at 12.3 s. Exact map
    # knots at ~44/52 m differ from the .25 m mesh by 1e-7/2e-8 metres.
    anchor = 27.725759497754424
    for length, expected_maximum in ((16., .08), (24., .04)):
        points, samples, _ = engine._build(reference, stations, anchor, length,
            "left", min(stations[-1], anchor + 90.), returning=True)
        assert anchor in samples and anchor + length in samples
        assert np.min(np.diff(samples)) >= .001
        assert 0.02 < np.max(_curvature(points)) < expected_maximum
        # Independently reproduce the old sampling set through the same
        # unchanged Hermite map interpolation and quintic geometry.
        indices = [i for i, s in enumerate(stations) if anchor-6 <= s <= anchor+92
                   and reference[i].get_left_lane() is not None]
        subset = reference[indices[0]:indices[-1]+1]
        knots = stations[indices[0]:indices[-1]+1]
        raw = np.unique(np.r_[knots, np.arange(knots[0], min(stations[-1], anchor+90), .25),
                              anchor, anchor+length])
        raw = raw[(raw >= knots[0]) & (raw <= min(stations[-1], anchor+90))]
        from carla_bridge.planning.lane_maneuver import _quintic
        center, adjacent = engine._samples(subset, "center", knots), engine._samples(subset, "left", knots)
        blend = 1 - _quintic((raw-anchor)/length)
        old = center(raw) + blend[:, None] * (adjacent(raw)-center(raw))
        assert np.min(np.diff(raw)) < 1e-6
        assert np.max(_curvature(old)) > 1.


def test_actual_close_walker_native_execution_passes_and_returns_with_body_clearance(short_pass_snapshot,
                                                                                     record_property):
    from carla_bridge.planning.lane_quality import lane_surface_union
    from carla_bridge.planning.lane_maneuver import project_polyline
    planner, clock, walker = short_pass_snapshot
    clearances, body_gaps, phases, rows = [], [], set(), []
    completed_at, original_anchor = None, None
    for _ in range(100):
        diagnostic = planner.plan(None, False, sim_time_s=clock.time)["planner_diagnostics"]
        winner = next(c for c in diagnostic["candidates"] if c["id"] == diagnostic["selected_id"])
        metadata = planner._evaluated_lateral_metadata[winner["lateral_path_index"]]
        keys = metadata.get("lane_quality_boundary_keys", ())
        if not keys:
            # After measured return completion ordinary lane quality uses its
            # centreline-width reference, without a maneuver-union metadata
            # key. Independently keep checking the actual original lane map.
            from carla_bridge.planning.pdm_planner import _quality_boundary_lane_keys
            state, _ = planner._ego_state(round(clock.time * 1e6))
            keys = _quality_boundary_lane_keys(planner._reference_for_height,
                state.rear_axle.point.array, ("center",))
        boundary = lane_surface_union([planner._lane_quality_boundary_polygons[k] for k in keys])
        body = footprint(planner.ego)
        assert boundary is not None and boundary.covers(body), (clock.time, winner["id"], diagnostic)
        clearances.append(boundary.boundary.distance(body))
        body_gaps.append(body.distance(footprint(walker)))
        # The existing maneuver checks use separate oriented longitudinal and
        # lateral reserves. They do not promise a universal Euclidean .5 m
        # reserve. Record physical clearance and reject actual contact.
        assert body_gaps[-1] > 0., (clock.time, body_gaps[-1])
        active = planner.lane_maneuver._active
        reference = np.array([[w.transform.location.x, -w.transform.location.y]
            for w in planner.lane_maneuver._last_reference])
        pose = planner.ego.get_transform()
        station, lateral, heading = project_polyline(pose.location.x, -pose.location.y, reference)
        phases.add(diagnostic["lane_maneuver"]["phase"])
        rows.append((clock.time, station, lateral, winner["id"], clearances[-1], body_gaps[-1]))
        if active is not None:
            original_anchor = active["anchor"] if original_anchor is None else original_anchor
            assert winner["desired_speed_mps"] <= active["speed_cap"] + 1e-6
            if active["phase"] == "passing":
                assert active["anchor"] == original_anchor
                # Every ordinary abort and every offset shares the same
                # executed speed responsibility; identity stays ordinary.
                if diagnostic["lane_maneuver"]["committed"]:
                    for candidate in diagnostic["candidates"]:
                        assert candidate["maneuver_execution_limit"]
                        assert candidate["desired_speed_mps"] <= active["speed_cap"] + 1e-6
                    if diagnostic["lane_maneuver"]["phase"] == "passing":
                        for entry in planner._evaluated_lateral_metadata:
                            if entry["physical_lane_side"] == "center" and not entry.get("maneuver"):
                                assert entry["maneuver_geometry_source"] == "fixed_original_route_abort_reference"
                            if (entry["physical_lane_side"] == "left" and entry["lane_offset_m"]
                                    and not entry.get("maneuver")):
                                assert entry["maneuver_geometry_source"] == "fixed_nominal_pass_lane_offset"
        elif "returning" in phases:
            completed_at = clock.time if completed_at is None else completed_at
            assert lateral < .35
            assert math.cos(-math.radians(pose.rotation.yaw) - heading) > .985
            if clock.time - completed_at >= 1.:
                break
        states = planner.scorer._states[winner["index"]]
        # Only the scored native .2-second execution prefix becomes feedback.
        advance_native_feedback(planner, clock, states[2], .2)
        velocity = walker.get_velocity()
        walker.x += velocity.x * .2
        walker.y += velocity.y * .2
    assert {"passing", "returning", "idle"}.issubset(phases), rows
    assert completed_at is not None and completed_at < 20., rows
    assert min(clearances) > .05, rows
    assert planner.ego.get_velocity().length() > 2.
    assert rows[-1][1] > 45.
    record_property("minimum_actual_body_gap_m", min(body_gaps))
    record_property("minimum_actual_lane_clearance_m", min(clearances))
    record_property("measured_return_completion_s", completed_at)
