"""Python 3.12 numerical process. No NAVSIM/nuPlan imports or future annotations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import shapely

from .protocol import digest, trajectory_from_centers, validate_packet, write_json
from nine_grid.contracts import Actor, Lane, SceneSnapshot, RetrievalResult, plain
from nine_grid.configuration import validate_config
from nine_grid.models import VisionClient, load_system_prompts, build_prompt, parse_answer, model_image_paths
from nine_grid.memory import MemoryStore
from nine_grid.object_scope import scope_scene
from nine_grid.planner_factory import create_planner
from nine_grid.planning_context import prepare_planning_context
from nine_grid.quality import evaluate, execution_audit
from nine_grid.representations import DINOExtractor, NineGridBuilder, build_representations, render_goal_overlay
from nine_grid.retrieval import retrieve


def planning_config(parameters, dino_model):
    config = copy.deepcopy(parameters)
    config["platform"] = "navsim"
    config["scenario"] = "navsim"
    config["perception"]["fully_occluded_in_current_planning"] = True
    config["mpc"]["horizon_steps"] = 40
    config["mpc"]["step_s"] = 0.1
    config["prediction"]["horizon_s"] = 4.0
    config["prediction"]["step_s"] = 0.1
    config["prediction"]["diagnostic_horizon_s"] = 4.9
    config["dino"]["model"] = str(Path(dino_model).resolve())
    config["memory"].update(write_enabled=False, mode="formal")
    return validate_config(config)


def snapshot_from_packet(packet, config):
    packet = validate_packet(packet)
    def actor(row):
        row = copy.deepcopy(row)
        for key in ("position", "velocity", "acceleration"):
            row[key] = np.asarray(row[key], float)
        return Actor(**row)
    lanes = []
    for row in packet["lanes"]:
        row = copy.deepcopy(row)
        row["center"] = np.asarray(row["center"], float)
        row["polygon"] = np.asarray(row["polygon"], float)
        lanes.append(Lane(**row))
    identity = packet["identity"]
    diagnostics = copy.deepcopy(packet["diagnostics"])
    diagnostics["bev_label_options"] = config["bev"]["labels"]
    diagnostics["planning_objects_config"] = config["planning_objects"]
    diagnostics["navsim_identity"] = identity
    return SceneSnapshot(identity["token"], 0.0, actor(packet["ego"]),
        [actor(row) for row in packet["actors"]], lanes, np.asarray(packet["route"], float),
        packet["route_lane_ids"], shapely.from_wkt(packet["drivable_wkt"]), "navsim",
        rules=copy.deepcopy(packet["rules"]), assets={"camera_images": packet["camera_images"]},
        diagnostics=diagnostics)


def _provenance_values(value, keys):
    """Read provenance only; arbitrary experience prose is not a token list."""
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, (str, int)):
                found.add(str(child))
            if key in ("source", "provenance", "source_metadata", "situation", "dataset", "metadata", "frame"):
                found.update(_provenance_values(child, keys))
    return found


def check_memory_disjoint(records, evaluation_logs, evaluation_tokens):
    for record in records:
        logs = _provenance_values(record, {"log_name", "source_log_name", "log_id", "source_log"})
        tokens = _provenance_values(record, {"token", "source_token", "anchor_token", "initial_token",
                                            "source_anchor_token", "scene_token", "source_scene_token"})
        if logs & set(evaluation_logs) or tokens & set(evaluation_tokens):
            raise ValueError("Frozen memory overlaps evaluation logs/tokens: " + record["id"])
        # Records from real data need auditable source IDs. Simulator experiences
        # have an explicit simulator origin and can be distinguished from NAVSIM.
        origins = _provenance_values(record, {"platform", "simulator", "record_origin"})
        simulator_source = bool(origins & {"highway", "highway-env", "carla", "online_highway", "online_carla"})
        if not logs and not tokens and not simulator_source:
            raise ValueError("Memory has no auditable source log/token: " + record["id"])


class NumericalPipeline:
    def __init__(self, settings):
        import yaml
        from .launch import source_fingerprint, ROOT
        if source_fingerprint(ROOT, "src")[1] != settings["numerical_source_sha256"]:
            raise ValueError("Numerical source changed after benchmark settings were frozen")
        for key, expected in settings["config_file_sha256"].items():
            if hashlib.sha256(Path(settings[key]).read_bytes()).hexdigest() != expected:
                raise ValueError("Configuration changed after benchmark settings were frozen: " + key)
        self.settings = settings
        self.config = planning_config(json.loads(Path(settings["parameters"]).read_text(encoding="utf-8")),
                                      settings["dino_model"])
        self.config["dino"]["device"] = settings["dino_device"]
        prompts = load_system_prompts(settings["prompts_config"])
        models = yaml.safe_load(Path(settings["models_config"]).read_text(encoding="utf-8"))
        fast = dict(models["fast"], backend=settings["fast"], system_prompt=prompts["fast"])
        self.fast = VisionClient("fast", fast)
        self.extractor = DINOExtractor(self.config["dino"]["model"],
            device=self.config["dino"].get("device"), local_files_only=True)
        self.store = MemoryStore(settings.get("memory_read_dir") if settings["retrieve"] else None, None)
        self.records = self.store.snapshot()
        if self.store.snapshot_id != settings["memory_snapshot_id"]:
            raise ValueError("Memory changed after the benchmark snapshot was frozen")
        check_memory_disjoint(self.records, settings["evaluation_logs"], settings["evaluation_tokens"])
        if settings["retrieve"] and not self.records:
            raise ValueError("Retrieval requested but no compatible audited memory was loaded")

    def compute(self, packet, output_dir):
        started = time.perf_counter()
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        scene = snapshot_from_packet(packet, self.config)
        write_json(root / "input_packet.json", packet)
        config_hash = digest(self.config)
        timing = {}
        tick = time.perf_counter()
        reps = build_representations(scene, root / "representations", self.extractor, NineGridBuilder(self.config))
        timing["representations_s"] = time.perf_counter() - tick
        retrieved = RetrievalResult(scene.frame_id, snapshot_id=self.store.snapshot_id, time=0.0)
        if self.settings["retrieve"]:
            tick = time.perf_counter()
            retrieved = retrieve(reps, self.records, self.store.snapshot_id, self.config)
            timing["retrieval_s"] = time.perf_counter() - tick
        answer, model_error, raw = None, None, None
        scoped_for_model, _ = scope_scene(scene, self.config)
        if self.fast.enabled:
            prompt = ("评测输入条件：NAVSIM特权输入基线，使用当前及历史交通对象、车道地图、导航和当前灯态真值。"
                      "不判断遮挡；只使用已给出的车道范围内对象，禁止推测任何未来真值。\n"
                      + build_prompt(scoped_for_model, reps, retrieved))
            (root / "model_prompt.txt").write_text(prompt, encoding="utf-8")
            tick = time.perf_counter()
            try:
                raw = self.fast.call(prompt, model_image_paths(scene, reps))
                answer = parse_answer(raw, scoped_for_model, config=self.config)
            except Exception as exc:
                # Follow existing EP-Driver failure policy. It remains visible in
                # evidence, and official evaluation still scores the base fallback.
                model_error = type(exc).__name__ + ": " + str(exc)
            timing["model_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        scoped, predictions = prepare_planning_context(scene, self.config, answer=answer)
        timing["prediction_s"] = time.perf_counter() - tick
        # Official tokens may be processed in any order, including nearby samples
        # and synthetic follow-ups. Do not reuse state across independent anchors.
        planner = create_planner(self.config)
        tick = time.perf_counter()
        plan = planner.solve(scoped, predictions, answer)
        timing["planner_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        base_scene, base_predictions = prepare_planning_context(scene, self.config)
        baseline = create_planner(self.config).solve(base_scene, base_predictions)
        timing["reference_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        quality = evaluate(scoped, plan, baseline, predictions, self.config)
        audit = execution_audit(scoped, plan, predictions, self.config)
        timing["internal_quality_s"] = time.perf_counter() - tick
        if audit.get("alarm") and plan.valid:
            plan = planner.emergency_brake(scoped, predictions, "independent_predicted_execution_alarm")
            quality = evaluate(scoped, plan, baseline, predictions, self.config)
            audit = execution_audit(scoped, plan, predictions, self.config)
        render_goal_overlay(scene, plan).save(root / "trajectory_bev.png")
        evidence = dict(identity=packet["identity"], input_condition=packet["input_condition"],
            parameters_sha256=config_hash, parameters=self.config, timing_s=timing,
            model_answer=plain(answer), model_error=model_error, model_raw=raw,
            retrieval=plain(retrieved), memory_snapshot_id=self.store.snapshot_id,
            dino_signature=self.extractor.signature, plan=plain(plan), reference=plain(baseline),
            predictions=plain(predictions),
            internal_quality=plain(quality), predicted_execution_audit=plain(audit),
            official_score=None, internal_quality_is_official_metric=False,
            sample_state_policy="independent_anchor_reset_no_online_memory_writes",
            input_packet_sha256=digest(packet))
        write_json(root / "planning_evidence.json", evidence)
        if not plan.valid:
            raise RuntimeError("Planner did not produce an executable trajectory: " + plan.reason)
        poses = trajectory_from_centers(plan.states, self.config["mpc"]["step_s"],
                                        scene.ego.rear_axle_to_center_m)
        timing["total_s"] = time.perf_counter() - started
        result = dict(poses=poses.tolist(), frame_id=scene.frame_id, sampling=dict(horizon_s=4.0, interval_s=0.5),
                      execution_mode=plan.diagnostics.get("execution_mode"), timing_s=timing,
                      safety_feasible=plan.diagnostics.get("safety_feasible"), evidence=str(root / "planning_evidence.json"))
        write_json(root / "trajectory.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    pipeline = NumericalPipeline(settings)
    protocol_stdout.write("EP_NAVSIM_REPLY " + json.dumps({"ready": True}) + "\n")
    protocol_stdout.flush()
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("operation") == "close":
            break
        try:
            result = pipeline.compute(request["packet"], request["output_dir"])
            reply = dict(ok=True, result=result, request_id=request["request_id"])
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = dict(ok=False, error=type(exc).__name__ + ": " + str(exc), request_id=request["request_id"])
            write_json(Path(request["output_dir"]) / "failure.json", reply)
        protocol_stdout.write("EP_NAVSIM_REPLY " + json.dumps(reply, allow_nan=False) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
