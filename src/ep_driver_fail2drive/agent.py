"""Official Fail2Drive Agent: observation in, VehicleControl out; never ticks world."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

# Upstream imports the agent by file basename rather than as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from ep_driver_fail2drive.archive import RouteArchive
from ep_driver_fail2drive.execution import (ControlTimeline, SignedAccelerationController,
                                          planning_previous_control)
from ep_driver_fail2drive.protocol import SCHEMA, WorkerClient
from ep_driver_fail2drive.public_state import PublicStateExtractor
from ep_driver_fail2drive.sensor_pose import SensorPoseTap
from ep_driver_fail2drive.vehicle_geometry import measure_vehicle_geometry, json_safe, WheelGeometryError

PHYSICS_HZ = 20
CAMERA_POSES = (("front_rgb", 0), ("front_left_rgb", -60), ("front_right_rgb", 60),
                ("back_rgb", 180), ("back_left_rgb", -120), ("back_right_rgb", 120))


def get_entry_point():
    return "EPDriverFail2DriveAgent"


def _control_record(control):
    return {key: getattr(control, key) for key in
            ("throttle", "brake", "steer", "hand_brake", "reverse", "manual_gear_shift", "gear")}


class EPDriverFail2DriveAgent(AutonomousAgent):
    execution_modes = ("A_base", "B_model")

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self.archive = None
        self.worker = None
        self.extractor = None
        self.last_frame = None
        self.first_frame = None
        self.route_id = None
        self.previous_execution = None
        self._pose_tap = None

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        # Keep the supplied dense public navigation; base implementation downsamples at 200m.
        self._global_plan = list(global_plan_gps)
        self._global_plan_world_coord = list(global_plan_world_coord)
        if len(self._global_plan_world_coord) < 2:
            raise ValueError("official route requires at least two navigation points")

    def setup(self, path_to_conf_file):
        self.config = json.loads(Path(path_to_conf_file).read_text(encoding="utf-8"))
        self.track = Track.MAP
        self.planner_hz = self.config.get("planner_hz", 10)
        if type(self.planner_hz) is not int or self.planner_hz not in (1, 2, 10):
            raise ValueError("planner_hz must be 1, 2 or 10")
        model_hz = self.config.get("model_hz", 1)
        if type(model_hz) not in (int, float) or model_hz != 1:
            raise ValueError("this adapter uses 1Hz model source snapshots")
        self.execution_mode = self.config.get("execution_mode", "B_model")
        if self.execution_mode not in self.execution_modes:
            raise ValueError("Fail2Drive execution_mode must be A_base or B_model, without memory retrieval/writes")
        self.config_path = str(Path(path_to_conf_file).resolve())
        # Install before AgentWrapper creates sensors. Preserve the official
        # image parser/queue while retaining raw same-frame transform metadata.
        spec = next(sensor for sensor in self.sensors() if sensor['id'] == 'front_rgb')
        mount = carla.Transform(carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
            carla.Rotation(pitch=spec['pitch'], yaw=spec['yaw'], roll=spec['roll']))
        if self._pose_tap is not None:
            self._pose_tap.close()
        self._pose_tap = SensorPoseTap(self.sensor_interface, camera_id='front_rgb',
                                     camera_mount_matrix_m=mount.get_matrix())
        try:
            self._pose_tap.install()
        except Exception:
            self._pose_tap.close()
            raise

    def sensors(self):
        width, height = self.config.get("camera_width", 800), self.config.get("camera_height", 450)
        if any(type(v) is not int or not 1 <= v <= 4096 for v in (width, height)):
            raise ValueError("model camera width/height must be integer pixels between 1 and 4096")
        sensors = [{"type": "sensor.camera.rgb", "x": 0.0, "y": 0.0, "z": 1.8,
                 "roll": 0.0, "pitch": 0.0, "yaw": yaw, "width": width, "height": height,
                 "fov": 100, "id": name} for name, yaw in CAMERA_POSES]
        # Exact pose/settings from official team_code/viz_path_agent.py + config.py.
        # Display only: it is excluded from camera_images and every model request.
        sensors.append({"type": "sensor.camera.rgb", "x": -5.5, "y": 0.0, "z": 3.5,
                        "roll": 0.0, "pitch": -8.0, "yaw": 0.0, "width": 1920,
                        "height": 1080, "fov": 110, "id": "rgb"})
        return sensors

    def _initialize(self, snapshot):
        self.world = CarlaDataProvider.get_world()
        client = CarlaDataProvider.get_client()
        client_version, server_version = client.get_client_version(), client.get_server_version()
        if client_version != server_version:
            raise RuntimeError("CARLA client/server version mismatch: %s != %s" % (client_version, server_version))
        settings = self.world.get_settings()
        if not settings.synchronous_mode or abs(settings.fixed_delta_seconds-1/PHYSICS_HZ) > 1e-7:
            raise RuntimeError("official synchronous 20Hz settings required; Agent does not change world settings")
        self.ego = CarlaDataProvider.get_hero_actor()
        if self.ego is None:
            raise RuntimeError("official CarlaDataProvider has no hero actor")
        route_signature = [(round(point[0].location.x, 3), round(point[0].location.y, 3))
                           for point in self._global_plan_world_coord]
        route_hash = hashlib.sha256(json.dumps(route_signature).encode()).hexdigest()[:12]
        self.route_id = "route_"+route_hash
        unique = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.archive = RouteArchive(Path(self.config["output_dir"])/"agent"/(self.route_id+"_"+unique),
                                    video=self.config.get("video", True), fps=PHYSICS_HZ)
        self.archive.json("configuration.json", self.config)
        self.archive.json("carla_identity.json", {"client_version": client_version, "server_version": server_version,
            "map": self.world.get_map().name, "physics_hz": PHYSICS_HZ,
            "synchronous_mode": settings.synchronous_mode,
            "simulator_installation": self.config.get("simulator_installation"),
            "limitation": "version equality alone does not identify custom assets; install manifest and launch path must also match"})
        self.archive.json("category.json", {"track": "MAP", "input_category": "privileged_current_state",
            "source": "current_public_world_and_dense_official_navigation", "future_events": False,
            "note": "This is not a sensor-only perception submission."})
        navigation_record = {"route_id": self.route_id, "public_route_xy": route_signature,
                             "source": "official_set_global_plan_dense", "official_score": False,
                             "checkpoint_association": self.config.get("archive_association", {})}
        self.archive.json("navigation.json", navigation_record)
        import os
        viz_root = Path(os.environ.get("VIZ_PATH", str(Path(self.config["output_dir"])/"official_visualization")))
        self.viz_path = viz_root/(self.route_id+"_"+unique)
        self.viz_path.mkdir(parents=True, exist_ok=False)
        self.archive.json("official_visualization.json", {"path": str(self.viz_path.resolve()),
            "source": "official_CARLA_rendered_camera", "renderer_pose_reference":
            "Fail2Drive/team_code/viz_path_agent.py and team_code/config.py at "+self.config["official_commit"],
            "layout": "VIZ_PATH/<route_run>/<step:05>.jpg", "save_interval_ticks": 10,
            "model_input": False, "independent_official_viewer": False})
        self.archive.json("sensor_metadata_tap.json", dict(self._pose_tap.describe(),
                          official_commit=self.config["official_commit"]))
        try:
            source_time = float(snapshot.timestamp.elapsed_seconds)
            frozen_ego = snapshot.find(self.ego.id)
            if frozen_ego is None:
                raise ValueError("ego is absent from the source world snapshot")
            pose = self._pose_tap.for_frame(snapshot.frame, source_time)
            physics = self.ego.get_physics_control()
            box = self.ego.bounding_box
            geometry = measure_vehicle_geometry(
                actor_transform_matrix_m=frozen_ego.get_transform().get_matrix(), wheels=physics.wheels,
                bbox_center_m=[box.location.x, box.location.y, box.location.z],
                bbox_extent_m=[box.extent.x, box.extent.y, box.extent.z], sensor_pose=pose,
                frame=snapshot.frame, timestamp=source_time,
                bbox_heading_deg=box.rotation.yaw)
            if int(self.world.get_snapshot().frame) != int(snapshot.frame):
                raise WheelGeometryError("world advanced while measuring source-frame wheel geometry", geometry)
        except Exception as exc:
            failure = dict(getattr(exc, "evidence", {}))
            failure.update(status="invalid", source_frame=int(snapshot.frame), type=type(exc).__name__)
            failure.setdefault("error", str(exc))
            self.archive.json("ego_geometry.json", json_safe(failure))
            raise
        self.archive.json("ego_geometry.json", geometry)
        self.extractor = PublicStateExtractor(self.world, self.ego, self._global_plan_world_coord,
            config=self.config.get("public_state", {}), route_id=self.route_id, ego_geometry=geometry)
        # Preserve the exact published finish transform separately from the
        # rounded XY route signature; no official score or live route index is read.
        finish_gate = getattr(getattr(self.extractor, "route", None), "finish_gate", None)
        if finish_gate is not None:
            navigation_record["finish_gate"] = json_safe(finish_gate)
            self.archive.json("navigation.json", navigation_record)
        execution = dict(self.config.get("execution", {}))
        minimum = float(execution.get("acceleration_min", -6.0))
        maximum = float(execution.get("acceleration_max", 3.0))
        steering = float(execution.get("steering_max", .6))
        wheel_angles = [wheel.max_steer_angle for wheel in physics.wheels
                        if wheel.max_steer_angle > 0]
        if not wheel_angles:
            raise RuntimeError("ego physics has no steering wheels")
        physical_steering = math.radians(max(wheel_angles))
        self.timeline = ControlTimeline(minimum, maximum, steering, physics_hz=PHYSICS_HZ)
        self.controller = SignedAccelerationController(physical_steering, dt=1/PHYSICS_HZ,
            acceleration_min=minimum, acceleration_max=maximum,
            steering_rate=float(execution.get("steering_rate", .8)), settings=execution.get("longitudinal", {}))
        self.worker = self._create_worker()
        self.first_frame = snapshot.frame

    def _create_worker(self):
        """Formal driving always uses the paper planner; demos override only this factory."""
        return WorkerClient(self.config["worker_python"], self.config_path,
                            self.archive.directory/"worker.log", self.config.get("worker_timeout_s", 240))

    def run_step(self, input_data, timestamp):
        started = time.monotonic()
        world = CarlaDataProvider.get_world()
        snapshot = world.get_snapshot()
        frame = int(snapshot.frame)
        source_time = float(snapshot.timestamp.elapsed_seconds)
        if self.last_frame is not None and frame != self.last_frame+1:
            raise RuntimeError("official physical source frames must be consecutive")
        try:
            if self.extractor is None:
                self._initialize(snapshot)
            relative_tick = frame-self.first_frame
            plan_due = relative_tick % (PHYSICS_HZ//self.planner_hz) == 0
            model_due = relative_tick % PHYSICS_HZ == 0
            images = self.archive.capture_images(input_data, frame, source_time,
                        [name for name, _ in CAMERA_POSES], save=model_due)
            if "rgb" not in input_data or int(input_data["rgb"][0]) != frame:
                raise ValueError("official visualization RGB/state source frame mismatch")
            self.archive.capture_display(input_data["rgb"], frame, source_time)
            if relative_tick % 10 == 0:
                import cv2
                destination = self.viz_path/("%05d.jpg" % relative_tick)
                success, encoded = cv2.imencode(".jpg", input_data["rgb"][1][:, :, :3])
                if not success:
                    raise RuntimeError("failed to save official visualization camera frame")
                image_bytes = encoded.tobytes()
                temporary = destination.with_suffix(".jpg.tmp")
                temporary.write_bytes(image_bytes)
                temporary.replace(destination)
                metadata = {"schema": "ep-driver-f2d-display-frame-v1", "jpeg": destination.name,
                    "jpeg_sha256": hashlib.sha256(image_bytes).hexdigest(), "carla_frame": frame,
                    "carla_timestamp": source_time, "relative_tick": relative_tick,
                    "route_id": self.route_id, "route_run": self.viz_path.name,
                    "camera_id": "rgb", "source": "official_CARLA_rendered_camera",
                    "sampling_hz_sim": PHYSICS_HZ/10, "saved_wall_time": time.time()}
                # Publish metadata last: a viewer must never infer frame identity
                # from the relative filename or read a half-written JPEG.
                metadata_path = destination.with_suffix(".json")
                temporary = metadata_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                temporary.replace(metadata_path)
            plan_wall_s = None
            if plan_due:
                scene = self.extractor.capture(frame, camera_images=images, world_snapshot=snapshot)
                prior_control = planning_previous_control(self.previous_execution,
                    -self.ego.get_control().steer*self.controller.maximum_steering)
                prior_frame = self.last_frame if self.previous_execution is not None else None
                scene.setdefault("diagnostics", {}).update(previous_control=prior_control,
                    previous_control_source={"kind": "previous_planned_acceleration_and_returned_steering", "frame": prior_frame,
                        "initialization": prior_frame is None,
                        "note": "jerk uses the last selected planned acceleration; inner-loop feedback effort is separate",
                        "actuator_requested_acceleration_mps2": (self.previous_execution.get("requested_acceleration_mps2")
                            if self.previous_execution is not None else None)})
                packet = {"schema": SCHEMA, "route_id": self.route_id, "frame": frame,
                          "time": source_time, "scene": scene, "images": images,
                          "output_dir": str(self.archive.directory.resolve())}
                # Preserve the exact public input even when decoding or planning
                # fails before the numeric worker can archive a Scene.
                self.archive.append("agent_sources", {"frame": frame, "time": source_time, "scene": scene})
                begin_plan = time.monotonic()
                plan = self.worker.plan(packet)
                plan_wall_s = time.monotonic()-begin_plan
                if not plan.get("executable", False):
                    raise RuntimeError("numeric worker returned a non-executable control sequence")
                diagnostics = plan.get("diagnostics", {})
                emergency = (diagnostics.get("emergency") is True or
                    str(diagnostics.get("execution_mode", "")).startswith("emergency"))
                self.timeline.install(plan["controls"], source_time, float(plan["control_dt"]),
                                      source_speed_mps=math.hypot(*scene["ego"]["velocity"]), source_tick=frame,
                                      emergency_braking=emergency)
                self.archive.append("agent_plans", plan)
            selected = self.timeline.at(source_time, physical_tick=frame)
            if not selected["available"]:
                raise RuntimeError("control sequence unavailable: "+str(selected["reason"]))
            transform = self.ego.get_transform()
            heading = math.radians(transform.rotation.yaw)
            velocity, acceleration = self.ego.get_velocity(), self.ego.get_acceleration()
            measured_speed = max(0., velocity.x*math.cos(heading)+velocity.y*math.sin(heading))
            measured_acceleration = acceleration.x*math.cos(heading)+acceleration.y*math.sin(heading)
            actual = self.ego.get_control()
            command = self.controller.step(frame, selected["control"][0], selected["control"][1],
                selected["target_speed_mps"], measured_speed, measured_acceleration,
                actual_steering_rad=-actual.steer*self.controller.maximum_steering,
                emergency_braking=self.timeline.emergency_braking)
            control = carla.VehicleControl(throttle=command["throttle"], brake=command["brake"],
                        steer=command["steer_normalized_right"], hand_brake=False,
                        reverse=False, manual_gear_shift=False)
            self.archive.append("execution", {"frame": frame, "time": source_time,
                "official_game_time": timestamp, "source_frame": self.timeline.source_tick,
                "source_time": self.timeline.source_time, "segment": selected["index"],
                "controller": command, "observed_applied_control": _control_record(actual),
                "returned_control": _control_record(control), "plan_wall_s": plan_wall_s,
                "agent_wall_s": time.monotonic()-started})
            self.last_frame = frame
            self.previous_execution = command
            return control
        except Exception as exc:
            if self.archive is not None:
                self.archive.append("agent_errors", {"frame": frame, "time": source_time,
                    "type": type(exc).__name__, "error": str(exc),
                    "evidence": json_safe(getattr(exc, "evidence", None))})
            raise  # Official evaluator must record Agent failure rather than a hidden successful fallback.

    def destroy(self):
        if self._pose_tap is not None:
            self._pose_tap.close()
            self._pose_tap = None
        worker, archive = self.worker, self.archive
        self.worker, self.archive = None, None
        try:
            if worker is not None:
                worker.close()
        finally:
            if archive is not None:
                archive.close()
