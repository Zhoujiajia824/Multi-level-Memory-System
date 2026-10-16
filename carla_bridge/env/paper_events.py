"""Frozen, pre-spawned research events with explicit execution evidence.

No world ticks, Traffic Manager agents, automatic retries, or silent fallbacks.
Pedestrians wait at resolved sidewalk positions until ordered route progress
reaches the trigger. Parked vehicles are present from the beginning of the run.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import carla


class PaperEventScheduler:
    def __init__(self, world, ego, route, events, elapsed, audit_path=None):
        self.world, self.ego, self.route, self.elapsed = world, ego, route, elapsed
        self.events = copy.deepcopy(events)
        self.actors = {}
        self.audit_path = Path(audit_path) if audit_path else None
        self.records = [{"id": e["id"], "type": e["type"], "state": "pending",
                         "spawned": False, "encountered": False, "motion_updates": 0,
                         "trigger_distance_m": e["trigger_distance_m"],
                         "anchor_distance_m": e["anchor_distance_m"]} for e in self.events]

    def setup(self):
        if not self.events:
            self._write_journal()
            return
        library = self.world.get_blueprint_library()
        walker_bps = sorted(library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
        for spec, record in zip(self.events, self.records):
            if spec["type"] == "pedestrian_crossing":
                if not walker_bps:
                    raise ValueError("no pedestrian blueprint available")
                bp = walker_bps[0]
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                tf = carla.Transform(carla.Location(**spec["start"]))
            elif spec["type"] == "stationary_vehicle":
                bp = library.find(spec["blueprint"])
                p = spec["spawn"]
                tf = carla.Transform(carla.Location(x=p["x"], y=p["y"], z=p["z"]),
                                     carla.Rotation(yaw=p["yaw"]))
            else:
                raise ValueError(f"unsupported paper event: {spec['type']}")
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is None:
                record.update(state="failed", error="actor_spawn_failed")
                self._write_journal()
                raise RuntimeError(f"paper event {spec['id']} spawn failed; case is invalid")
            self.actors[spec["id"]] = actor
            record.update(state="waiting", spawned=True, actor_id=actor.id,
                          blueprint=bp.id, spawn={"x": tf.location.x, "y": tf.location.y, "z": tf.location.z})
            if spec["type"] == "stationary_vehicle":
                actor.apply_control(carla.VehicleControl(throttle=0, brake=1, hand_brake=True))
                if self.world.get_weather().sun_altitude_angle < 0:
                    actor.set_light_state(carla.VehicleLightState.Position)
                record["state"] = "active"
        self._write_journal()

    def tick(self):
        previous = [(r["state"], r["encountered"], r.get("max_displacement_m", 0) >= .5) for r in self.records]
        progress_m = self.route.distance_completed_m()
        for spec, record in zip(self.events, self.records):
            if record["state"] in ("failed", "completed"):
                continue
            try:
                actor = self.actors[spec["id"]]
                if not actor.is_alive:
                    raise RuntimeError("event actor no longer exists")
                position = actor.get_location()
                if (progress_m >= spec["anchor_distance_m"] - 30
                        and self.ego.get_location().distance(position) <= 35):
                    record["encountered"] = True
                if spec["type"] == "stationary_vehicle":
                    continue
                if record["state"] == "waiting" and progress_m >= spec["trigger_distance_m"]:
                    record.update(state="active", triggered_sim_time_s=self.elapsed(),
                                  actual_trigger_distance_m=progress_m)
                if record["state"] == "waiting":
                    actor.apply_control(carla.WalkerControl(speed=0))
                    continue
                start, target = spec["start"], spec["target"]
                dx, dy = target["x"] - start["x"], target["y"] - start["y"]
                length = math.hypot(dx, dy)
                if length < 1:
                    raise ValueError("invalid crossing geometry")
                ux, uy = dx / length, dy / length
                rx, ry = target["x"] - position.x, target["y"] - position.y
                arrived = math.hypot(rx, ry) <= .3 or rx * ux + ry * uy <= 0
                actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=ux, y=uy),
                                                        speed=0 if arrived else float(spec["speed_mps"])))
                record["motion_updates"] += 1
                moved = math.hypot(position.x - start["x"], position.y - start["y"])
                record["max_displacement_m"] = max(record.get("max_displacement_m", 0), moved)
                if arrived:
                    record.update(state="completed", completed_sim_time_s=self.elapsed())
            except Exception as exc:
                record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        current = [(r["state"], r["encountered"], r.get("max_displacement_m", 0) >= .5) for r in self.records]
        if current != previous:
            self._write_journal()

    def _write_journal(self):
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.audit_path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.audit_path)

    def has_failed(self):
        return any(r["state"] == "failed" for r in self.records)

    def summary(self):
        def valid(record):
            if not record["spawned"] or not record["encountered"] or record["state"] == "failed":
                return False
            if record["type"] == "pedestrian_crossing":
                return record.get("motion_updates", 0) > 0 and record.get("max_displacement_m", 0) >= .5
            return True
        return {"required": len(self.records), "executed": sum(valid(r) for r in self.records),
                "valid": all(valid(r) for r in self.records), "events": copy.deepcopy(self.records)}

    def destroy(self):
        remaining = {}
        for key, actor in self.actors.items():
            try:
                if actor.is_alive and actor.destroy() is False:
                    remaining[key] = actor
            except Exception:
                remaining[key] = actor
        self.actors = remaining
        self._write_journal()
        return not remaining
