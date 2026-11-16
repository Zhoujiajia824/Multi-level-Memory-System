#!/usr/bin/env python3
"""Measure CARLA camera cadence without waiting for any particular image frame.

Python 3.7+, standard library plus the CARLA SDK. Run only on an idle, exclusively
owned server. This changes world settings temporarily, never loads a map, and
creates only three unattached cameras. No driving, models, or image files.

Example:
    python scripts/check_carla_camera_clock.py --port 22000 --exclusive-world
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid


SENSOR_TICKS = ("0.1", "0.09999999403953552", "0")
SETTING_NAMES = (
    "synchronous_mode", "fixed_delta_seconds", "no_rendering_mode",
    "substepping", "max_substep_delta_time", "max_substeps",
    "max_culling_distance", "deterministic_ragdolls", "tile_stream_distance",
    "actor_active_distance", "spectator_as_ego",
)


class RefusedError(RuntimeError):
    pass


def settings_dict(settings):
    return {name: getattr(settings, name) for name in SETTING_NAMES
            if hasattr(settings, name)}


def error_text(exc):
    return "{}: {}".format(type(exc).__name__, str(exc)[:1000])


def blockers(world):
    actors = [actor for actor in world.get_actors()
              if actor.type_id.startswith(("vehicle.", "walker.", "static.prop."))]
    return {"count": len(actors), "first_25": [
        {"id": actor.id, "type_id": actor.type_id} for actor in actors[:25]]}


def increment(histogram, key):
    # Keep the report bounded even if an unexpected external client is active.
    key = str(key)
    if key not in histogram and len(histogram) >= 128:
        key = "other"
    histogram[key] = histogram.get(key, 0) + 1


class Recorder:
    def __init__(self, world):
        self.world = world
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.completed_frame = None
        self.accepting = True
        self.data = {}
        self.previous = {}

    def callback(self, sensor_tick):
        row = {
            "requested_sensor_tick": sensor_tick, "count": 0,
            "first_12_samples": [], "frame_interval_counts": {},
            "timestamp_interval_seconds_counts": {},
            "snapshot_frame_lag_counts": {}, "callback_errors": 0,
        }
        with self.lock:
            self.data[sensor_tick] = row

        def receive(image):
            received = time.monotonic() - self.started
            try:
                # Keep metadata only; never access or retain image.raw_data.
                frame, timestamp = int(image.frame), float(image.timestamp)
                with self.lock:
                    completed = self.completed_frame
                snapshot_frame = int(self.world.get_snapshot().frame)
                sample = {
                    "frame": frame, "timestamp": timestamp,
                    "last_completed_world_frame": completed,
                    "world_snapshot_frame_at_receive": snapshot_frame,
                    "completed_frame_minus_sensor_frame": (
                        None if completed is None else completed - frame),
                    "snapshot_frame_minus_sensor_frame": snapshot_frame - frame,
                    "wall_received_seconds": round(received, 6),
                }
                with self.lock:
                    if not self.accepting:
                        return
                    previous = self.previous.get(sensor_tick)
                    if previous is not None:
                        increment(row["frame_interval_counts"], frame - previous[0])
                        increment(row["timestamp_interval_seconds_counts"],
                                  "{:.9f}".format(timestamp - previous[1]))
                    increment(row["snapshot_frame_lag_counts"], snapshot_frame - frame)
                    self.previous[sensor_tick] = (frame, timestamp)
                    row["count"] += 1
                    row["last_sample"] = sample
                    if len(row["first_12_samples"]) < 12:
                        row["first_12_samples"].append(sample)
            except Exception as exc:
                with self.lock:
                    row["callback_errors"] += 1
                    row["last_callback_error"] = error_text(exc)

        return receive

    def mark_completed(self, frame):
        with self.lock:
            self.completed_frame = frame

    def finish(self):
        with self.lock:
            self.accepting = False
            return json.loads(json.dumps(self.data))


def camera_pose(carla, world):
    points = world.get_map().get_spawn_points()
    source = points[0] if points else world.get_spectator().get_transform()
    return carla.Transform(
        carla.Location(x=source.location.x, y=source.location.y,
                       z=source.location.z + 2.5),
        carla.Rotation(pitch=source.rotation.pitch, yaw=source.rotation.yaw,
                       roll=source.rotation.roll)), ("first_map_spawn" if points else "spectator")


def run_probe(args, carla, report):
    world, previous_settings, recorder = None, None, None
    settings_attempted = False
    owned_sensors = []
    cleanup = {"errors": [], "destroyed_sensor_ids": [],
               "settings_restore_attempted": False, "settings_restored": False}
    report["cleanup"] = cleanup
    try:
        if not args.exclusive_world:
            raise RefusedError("--exclusive-world is required; use an idle private server")
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        report["sdk"] = {"client_version": client.get_client_version(),
                         "server_version": client.get_server_version(),
                         "module_path": str(getattr(carla, "__file__", "unknown"))}
        world = client.get_world()
        previous_settings = world.get_settings()
        report["previous_settings"] = settings_dict(previous_settings)
        if previous_settings.synchronous_mode:
            raise RefusedError("World is already synchronous; settings were not changed")
        report["initial_blockers"] = blockers(world)
        if report["initial_blockers"]["count"]:
            raise RefusedError("World contains vehicles, walkers, or props; settings were not changed")

        # Recheck immediately before mutation. The explicit flag is the caller's
        # assurance of exclusivity; CARLA does not provide a client ownership lock.
        desired = world.get_settings()
        if desired.synchronous_mode or blockers(world)["count"]:
            raise RefusedError("World became occupied or synchronous during validation")
        if settings_dict(desired) != settings_dict(previous_settings):
            raise RefusedError("World settings changed during validation")
        desired.synchronous_mode = True
        desired.fixed_delta_seconds = 0.02
        desired.no_rendering_mode = False
        desired.substepping = True
        desired.max_substep_delta_time = 0.01
        desired.max_substeps = 2
        settings_attempted = True  # Even a timed-out RPC may have changed settings.
        world.apply_settings(desired)
        actual = world.get_settings()
        report["actual_settings"] = settings_dict(actual)
        for name in ("synchronous_mode", "fixed_delta_seconds", "no_rendering_mode",
                     "substepping", "max_substep_delta_time", "max_substeps"):
            if getattr(actual, name) != getattr(desired, name):
                raise RuntimeError("World setting did not take effect: " + name)
        expected_frame = int(world.get_snapshot().frame)
        report["baseline_world_frame"] = expected_frame
        recorder = Recorder(world)
        recorder.mark_completed(expected_frame)
        pose, source = camera_pose(carla, world)
        report["camera_pose"] = {
            "source": source,
            "location": {key: getattr(pose.location, key) for key in ("x", "y", "z")},
            "rotation": {key: getattr(pose.rotation, key) for key in ("pitch", "yaw", "roll")},
        }
        for sensor_tick in SENSOR_TICKS:
            blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(args.width))
            blueprint.set_attribute("image_size_y", str(args.height))
            blueprint.set_attribute("sensor_tick", sensor_tick)
            sensor = world.spawn_actor(blueprint, pose)
            owned_sensors.append(sensor)
            callback = recorder.callback(sensor_tick)
            recorder.data[sensor_tick]["actor_id"] = sensor.id
            recorder.data[sensor_tick]["actual_attributes"] = {
                name: sensor.attributes.get(name) for name in
                ("sensor_tick", "image_size_x", "image_size_y")}
            sensor.listen(callback)

        report["completed_world_ticks"] = 0
        report["first_12_world_frames"] = []
        for _ in range(args.ticks):
            before_frame = int(world.get_snapshot().frame)
            if before_frame != expected_frame:
                raise RuntimeError("Unexpected external world tick: expected {}, observed {}".format(
                    expected_frame, before_frame))
            tick_frame = int(world.tick(args.timeout))
            snapshot = world.get_snapshot()
            if tick_frame != expected_frame + 1 or snapshot.frame != tick_frame:
                raise RuntimeError("Unexpected world frame: expected {}, tick {}, snapshot {}".format(
                    expected_frame + 1, tick_frame, snapshot.frame))
            expected_frame = tick_frame
            recorder.mark_completed(tick_frame)
            report["completed_world_ticks"] += 1
            report["last_world_frame"] = tick_frame
            report["last_world_timestamp"] = float(snapshot.timestamp.elapsed_seconds)
            if len(report["first_12_world_frames"]) < 12:
                report["first_12_world_frames"].append(tick_frame)
            # Wall-clock pacing only: no target-frame waits, queues, or barriers.
            time.sleep(args.tick_wait)
        time.sleep(args.tail_wait)
        if int(world.get_snapshot().frame) != expected_frame:
            raise RuntimeError("Unexpected external world tick during final callback drain")
        report["status"] = "completed"
    except RefusedError as exc:
        report["status"], report["error"] = "refused", error_text(exc)
    except KeyboardInterrupt:
        report["status"], report["error"] = "interrupted", "KeyboardInterrupt"
    except Exception as exc:
        report["status"], report["error"] = "failed", error_text(exc)
    finally:
        for sensor in owned_sensors:
            try:
                sensor.stop()
            except Exception as exc:
                cleanup["errors"].append("stop sensor {}: {}".format(sensor.id, error_text(exc)))
        if recorder is not None:
            report["sensors"] = recorder.finish()
            if report.get("status") == "completed" and any(
                    row["callback_errors"] for row in report["sensors"].values()):
                report["status"] = "failed"
                report["error"] = "Camera metadata callback failed; see per-sensor errors"
        for sensor in owned_sensors:
            try:
                if sensor.destroy() is False:
                    raise RuntimeError("destroy() returned False")
                cleanup["destroyed_sensor_ids"].append(sensor.id)
            except Exception as exc:
                cleanup["errors"].append("destroy sensor {}: {}".format(sensor.id, error_text(exc)))
        if settings_attempted:
            cleanup["settings_restore_attempted"] = True
            try:
                world.apply_settings(previous_settings)
                cleanup["restored_settings"] = settings_dict(world.get_settings())
                if cleanup["restored_settings"] != report["previous_settings"]:
                    raise RuntimeError("Restored settings differ from original settings")
                cleanup["settings_restored"] = True
            except Exception as exc:
                cleanup["errors"].append("restore world settings: " + error_text(exc))
        if cleanup["errors"]:
            report["status_before_cleanup_failure"] = report["status"]
            report["status"] = "failed"
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22000)
    parser.add_argument("--exclusive-world", action="store_true",
                        help="Confirm this is an idle server exclusively owned by this probe")
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--tick-wait", type=float, default=0.05,
                        help="Wall seconds after every tick; never waits for an image frame")
    parser.add_argument("--tail-wait", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", type=Path,
                        help="New JSON path; existing files are never overwritten")
    args = parser.parse_args(argv)
    if not args.exclusive_world:
        parser.error("--exclusive-world is required (idle, exclusively owned server only)")
    for name, low, high in (("port", 1, 65535), ("ticks", 1, 10000),
                            ("width", 1, 4096), ("height", 1, 4096),
                            ("tick_wait", 0, 5), ("tail_wait", 0, 30),
                            ("timeout", 0.1, 60)):
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error("{} must be between {} and {}".format(name.replace("_", "-"), low, high))
    return args


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.datetime.now(datetime.timezone.utc)
    output = args.output or Path("outputs") / "carla_camera_clock_{}_{}_{}.json".format(
        stamp.strftime("%Y%m%dT%H%M%SZ"), os.getpid(), uuid.uuid4().hex[:8])
    output = output.resolve()
    # Reserve the output before any connection or world mutation.
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        report_file = output.open("x", encoding="utf-8")
    except OSError as exc:
        print("Cannot create report: " + error_text(exc), file=sys.stderr)
        return 2
    report = {
        "schema_version": 1, "status": "failed", "started_utc": stamp.isoformat(),
        "python": sys.version, "endpoint": {"host": args.host, "port": args.port},
        "probe": {key: getattr(args, key) for key in
                  ("exclusive_world", "ticks", "tick_wait", "tail_wait", "timeout", "width", "height")},
        "sensor_ticks": list(SENSOR_TICKS),
        "note": "The second period is the float32 predecessor of 0.1; no exact-frame waits are used.",
    }
    with report_file:
        try:
            import carla
            run_probe(args, carla, report)
        except Exception as exc:
            report["error"] = error_text(exc)
        json.dump(report, report_file, indent=2, sort_keys=True, allow_nan=False)
        report_file.write("\n")
    print("Camera clock probe: {} | {} world ticks | {}".format(
        report["status"], report.get("completed_world_ticks", 0), output))
    for sensor_tick in SENSOR_TICKS:
        row = report.get("sensors", {}).get(sensor_tick)
        if row is not None:
            print("  sensor_tick={}: {} samples; frame intervals {}".format(
                sensor_tick, row["count"], row["frame_interval_counts"]))
    if report.get("error"):
        print("  " + report["error"], file=sys.stderr)
    for error in report.get("cleanup", {}).get("errors", []):
        print("  CLEANUP ERROR: " + error, file=sys.stderr)
    return {"completed": 0, "refused": 2, "interrupted": 130}.get(report["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
