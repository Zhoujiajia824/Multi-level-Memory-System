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

from .protocol import (digest, trajectory_from_centers, validate_packet, write_json, horizon_contract,
                       relative_asset_path, HORIZON_S, PLANNING_STEP_S, PLANNING_STEPS)
from .archive import write_input_packet, serialize_retrieval
from .resume import (build_run_identity, sample_identity, load_completed_sample,
                     begin_sample, complete_sample)
from .memory import check_memory_disjoint, load_memory_snapshot
from .model import StructuredServerVisionClient, structured_response_format
from nine_grid.contracts import Actor, Lane, SceneSnapshot, RetrievalResult, plain
from nine_grid.configuration import validate_config
from nine_grid.models import (VisionClient, load_system_prompts, build_prompt, parse_answer,
                              model_image_paths, _goal_reference)
from nine_grid.perception_scope import perception_scene
from nine_grid.planner_factory import create_planner
from nine_grid.planning_context import prepare_planning_context
from nine_grid.quality import evaluate, execution_audit
from nine_grid.paths import filesystem_path
from nine_grid.representations import DINOExtractor, NineGridBuilder, build_representations, render_goal_overlay
from nine_grid.retrieval import retrieve


def planning_config(parameters, dino_model):
    config = copy.deepcopy(parameters)
    config["platform"] = "navsim"
    config["scenario"] = "navsim"
    config["perception"]["fully_occluded_in_current_planning"] = True
    config["mpc"]["horizon_steps"] = PLANNING_STEPS
    config["mpc"]["step_s"] = PLANNING_STEP_S
    config["prediction"]["horizon_s"] = HORIZON_S
    config["prediction"]["step_s"] = PLANNING_STEP_S
    # Retain the real TTC tail beyond the four-second planning/reference horizon.
    # Predictor.predict reads this diagnostic horizon when no override is supplied.
    config["prediction"]["diagnostic_horizon_s"] = HORIZON_S + max(
        config.get("quality", {}).get("ttc_offsets_s", [0, .3, .6, .9]))
    config["dino"]["model"] = str(Path(dino_model).resolve())
    config["memory"].update(write_enabled=False, mode="formal")
    config = validate_config(config)
    horizon_contract(config)
    return config


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
        structured = settings.get("structured_model_output", False)
        if type(structured) is not bool or (structured and settings["fast"] != "server"):
            raise ValueError("structured_model_output must be boolean and requires the server backend")
        self.config = planning_config(json.loads(Path(settings["parameters"]).read_text(encoding="utf-8")),
                                      settings["dino_model"])
        self.config["dino"]["device"] = settings["dino_device"]
        prompts = load_system_prompts(settings["prompts_config"])
        models = yaml.safe_load(Path(settings["models_config"]).read_text(encoding="utf-8"))
        fast = dict(models["fast"], backend=settings["fast"], system_prompt=prompts["fast"])
        if settings["retrieve"]:
            self.records, self.memory_snapshot_id, selection = load_memory_snapshot(
                settings["memory_read_dir"], settings["evaluation_logs"], settings["evaluation_tokens"])
            if selection != settings["memory_selection"]:
                raise ValueError("Memory selection changed after the benchmark snapshot was frozen")
        else:
            self.records, self.memory_snapshot_id = [], hashlib.sha256(b"[]").hexdigest()
        if self.memory_snapshot_id != settings["memory_snapshot_id"]:
            raise ValueError("Memory changed after the benchmark snapshot was frozen")
        check_memory_disjoint(self.records, settings["evaluation_logs"], settings["evaluation_tokens"])
        if settings["retrieve"] and not self.records:
            raise ValueError("Retrieval requested but no compatible audited memory was loaded")
        self.run_identity = build_run_identity(settings, self.config, fast, prompts, self.memory_snapshot_id)
        self.resume_enabled = settings.get("resume", True)
        self.fast = (StructuredServerVisionClient if structured else VisionClient)("fast", fast)
        self.extractor = DINOExtractor(self.config["dino"]["model"],
            device=self.config["dino"].get("device"), local_files_only=True)

    def compute(self, packet, output_dir):
        started = time.perf_counter()
        root = filesystem_path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        packet = validate_packet(packet)
        identity = sample_identity(packet, self.run_identity)
        if self.resume_enabled:
            cached = load_completed_sample(root, identity)
            if cached is not None:
                return cached
        begin_sample(root)
        scene = snapshot_from_packet(packet, self.config)
        packet_path = write_input_packet(root, packet)
        config_hash = digest(self.config)
        timing = {}
        tick = time.perf_counter()
        reps = build_representations(scene, root / "representations", self.extractor, NineGridBuilder(self.config))
        timing["representations_s"] = time.perf_counter() - tick
        retrieved = RetrievalResult(scene.frame_id, snapshot_id=self.memory_snapshot_id, time=0.0)
        if self.settings["retrieve"]:
            tick = time.perf_counter()
            retrieved = retrieve(reps, self.records, self.memory_snapshot_id, self.config)
            timing["retrieval_s"] = time.perf_counter() - tick
        answer, model_error, model_error_detail, raw = None, None, None, None
        response_format = None
        scoped_for_model, _ = perception_scene(scene, self.config)
        if self.fast.enabled:
            structured = self.settings.get("structured_model_output", False)
            # Cache before build_prompt: prompt, request schema and strict parser
            # must use this exact original-frame reference and configured limits.
            reference = _goal_reference(scoped_for_model, self.config) if structured else None
            if structured:
                response_format = structured_response_format(scoped_for_model, self.config, reference)
                write_json(root / "model_response_format.json", response_format)
            prompt = ("评测输入条件：NAVSIM特权输入基线，使用当前及历史交通对象、车道地图、导航和当前灯态真值。"
                      "不判断遮挡；对象候选为道路范围全部对象与当前BEV内车辆、自行车、行人的并集，"
                      "最多关注其中3个对象；八向最近代表用于检索，不是关注对象白名单。禁止推测任何未来真值。\n"
                      + build_prompt(scoped_for_model, reps, retrieved))
            (root / "model_prompt.txt").write_text(prompt, encoding="utf-8")
            tick = time.perf_counter()
            try:
                images = model_image_paths(scene, reps)
                raw = (self.fast.call_structured(prompt, images, response_format) if structured
                       else self.fast.call(prompt, images))
            except Exception as exc:
                # Follow existing EP-Driver failure policy. It remains visible in
                # evidence, and official evaluation still scores the base fallback.
                # Exception text may contain provider URLs or credentials.
                model_error = type(exc).__name__
            else:
                try:
                    answer = parse_answer(raw, scoped_for_model, config=self.config, reference=reference)
                except ValueError as exc:
                    # Local validation reasons are inspectable; remote exception
                    # text is never copied into evidence or worker replies.
                    model_error, model_error_detail = type(exc).__name__, str(exc)
                except Exception as exc:
                    model_error = type(exc).__name__
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
            plan = planner.emergency_brake(scoped, predictions, "independent_predicted_execution_alarm", answer=answer)
            quality = evaluate(scoped, plan, baseline, predictions, self.config)
            audit = execution_audit(scoped, plan, predictions, self.config)
        checked_horizon = horizon_contract(self.config,
            plans=dict(plan=plan, reference=baseline),
            predictions=dict(prediction=predictions, reference_prediction=base_predictions))
        representation_paths = dict(bev=Path(reps.bev_path), feature=Path(reps.feature_image_path),
            grid=Path(reps.grid_path), occupancy=Path(reps.bev_path).with_name(
                Path(reps.bev_path).name.replace("_bev.png", "_occupancy.npz")))
        render_goal_overlay(scene, plan).save(root / "trajectory_bev.png")
        evidence = dict(identity=packet["identity"], input_condition=packet["input_condition"],
            parameters_sha256=config_hash, parameters=self.config, timing_s=timing,
            model_answer=plain(answer), model_error=model_error, model_error_detail=model_error_detail, model_raw=raw,
            structured_model_output=self.settings.get("structured_model_output", False),
            model_response_format_sha256=digest(response_format) if response_format else None,
            retrieval=serialize_retrieval(retrieved, root), memory_snapshot_id=self.memory_snapshot_id,
            dino_signature=self.extractor.signature, plan=plain(plan), reference=plain(baseline),
            predictions=plain(predictions),
            reference_prediction_times_s=base_predictions.times.tolist(),
            horizon_contract=checked_horizon,
            representation_assets={key: relative_asset_path(value, root)
                                   for key, value in representation_paths.items()},
            internal_quality=plain(quality), predicted_execution_audit=plain(audit),
            official_score=None, internal_quality_is_official_metric=False,
            sample_state_policy="independent_anchor_reset_no_online_memory_writes",
            input_packet_sha256=digest(packet), resume_identity_sha256=digest(identity))
        write_json(root / "planning_evidence.json", evidence)
        if not plan.valid:
            raise RuntimeError("Planner did not produce an executable trajectory: " + plan.reason)
        poses = trajectory_from_centers(plan.states, self.config["mpc"]["step_s"],
                                        scene.ego.rear_axle_to_center_m)
        timing["total_s"] = time.perf_counter() - started
        result = dict(poses=poses.tolist(), frame_id=scene.frame_id, sampling=dict(horizon_s=4.0, interval_s=0.5),
                      execution_mode=plan.diagnostics.get("execution_mode"), timing_s=timing,
                      safety_feasible=plan.diagnostics.get("safety_feasible"), evidence=str(root / "planning_evidence.json"))
        result["condition_audit"] = dict(model_requested=self.fast.enabled, model_answer_valid=answer is not None,
            model_error=model_error, model_error_detail=model_error_detail,
            structured_model_output=self.settings.get("structured_model_output", False),
            retrieval_enabled=self.settings["retrieve"],
            retrieved_memory_ids=[record["id"] for record in retrieved.records])
        result["resume"] = dict(reused=False, identity_sha256=digest(identity))
        write_json(root / "trajectory.json", result)
        artifact_paths = [packet_path, root / "planning_evidence.json", root / "trajectory.json",
                          root / "trajectory_bev.png", root / "retrieval_diagnostics.json.gz",
                          *representation_paths.values(),
                          *(Path(row["path"]) for row in packet["camera_images"])]
        if self.fast.enabled:
            artifact_paths.append(root / "model_prompt.txt")
        if response_format is not None:
            artifact_paths.append(root / "model_response_format.json")
        complete_sample(root, identity, artifact_paths)
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
