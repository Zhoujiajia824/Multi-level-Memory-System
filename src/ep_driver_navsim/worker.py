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
                       relative_asset_path, HORIZON_S, PLANNING_STEP_S, PLANNING_STEPS,
                       prediction_diagnostic_horizon)
from .archive import write_input_packet, serialize_retrieval, write_json_gz
from .resume import (build_run_identity, sample_identity, load_completed_sample,
                     begin_sample, complete_sample, fail_sample)
from .memory import check_memory_disjoint, load_memory_snapshot
from .model import (NoLegalGoalReference, AuditedVisionClient, StructuredServerVisionClient,
                    structured_response_format, goal_retry_policy, rejection_stage,
                    goal_retry_prompt, MODEL_ATTEMPT_PROTOCOL, SEMANTIC_RETRY_STAGES)
from .model_transport import model_request_policy
from .model_guidance import (GuidanceMismatch, guidance_prompt, validate_answer_consistency,
                             refine_model_prompt)
from .progress import record_sample_progress
from .execution_refinement import validate_config as validate_refinement_config, refine_execution
from .execution_evidence import expected_execution_plan, audit_expected_execution
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
    # NAVSIM executes the submitted four-second trajectory. Its independent
    # audits describe that whole submission; they do not replace the optimizer's
    # finite best candidate with a second, separately selected braking plan.
    supplied_policy = config.get("navsim_execution", {})
    if not isinstance(supplied_policy, dict):
        raise ValueError("NAVSIM execution policy must be a mapping")
    policy = {"postsolve_safety_policy": "audit_only", "audit_horizon_s": HORIZON_S,
              "actor_tracking_margin_m": .75, "road_tracking_margin_m": .5,
              "steering_rate_weight": 1.0, **supplied_policy}
    if set(policy) != {"postsolve_safety_policy", "audit_horizon_s", "actor_tracking_margin_m",
                      "road_tracking_margin_m", "steering_rate_weight"}:
        raise ValueError("Unknown NAVSIM execution policy setting")
    if policy["postsolve_safety_policy"] != "audit_only":
        raise ValueError("NAVSIM postsolve_safety_policy must be audit_only")
    if (type(policy["audit_horizon_s"]) not in (int, float)
            or not np.isclose(policy["audit_horizon_s"], HORIZON_S, rtol=0, atol=1e-10)):
        raise ValueError("NAVSIM execution audit must cover the complete four-second submission")
    margin = policy["actor_tracking_margin_m"]
    if type(margin) not in (int, float) or not np.isfinite(margin) or margin < 0:
        raise ValueError("NAVSIM actor tracking margin must be finite and nonnegative")
    road_margin = policy["road_tracking_margin_m"]
    if type(road_margin) not in (int, float) or not np.isfinite(road_margin) or road_margin < 0:
        raise ValueError("NAVSIM road tracking margin must be finite and nonnegative")
    steering_weight = policy["steering_rate_weight"]
    if type(steering_weight) not in (int, float) or not np.isfinite(steering_weight) or steering_weight < 0:
        raise ValueError("NAVSIM steering-rate weight must be finite and nonnegative")
    # A planner-side tracking preference for the official controller, rather
    # than a certified tracking bound or another postsolve rejection policy.
    # Keep the shared Highway/CARLA MPC weights unchanged and permit an exact
    # explicit override (including zero) for NAVSIM experiments.
    config["mpc"]["weights"]["steering_rate"] = steering_weight
    config["navsim_execution"] = policy
    config["navsim_execution_refinement"] = validate_refinement_config(
        config.get("navsim_execution_refinement"))
    config["perception"]["fully_occluded_in_current_planning"] = True
    config["mpc"]["horizon_steps"] = PLANNING_STEPS
    config["mpc"]["step_s"] = PLANNING_STEP_S
    config["prediction"]["horizon_s"] = HORIZON_S
    config["prediction"]["step_s"] = PLANNING_STEP_S
    # Retain the real TTC tail beyond the four-second planning/reference horizon.
    # Predictor.predict reads this diagnostic horizon when no override is supplied.
    config["prediction"]["diagnostic_horizon_s"] = prediction_diagnostic_horizon(config)
    config["dino"]["model"] = str(Path(dino_model).resolve())
    config["memory"].update(write_enabled=False, mode="formal")
    config = validate_config(config)
    horizon_contract(config)
    return config


def snapshot_from_packet(packet, config):
    packet = validate_packet(packet)
    from .geometry_transfer import decode_drivable
    road, geometry_evidence = decode_drivable(packet)
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
    diagnostics["geometry_transfer"] = geometry_evidence
    if (diagnostics.get("has_occlusion") is False and "drivable_wkt" in packet
            and diagnostics.get("visibility_wkt") == packet["drivable_wkt"]):
        diagnostics.pop("visibility_wkt")
        diagnostics["visibility_source"] = "scene_drivable_no_occlusion"
    diagnostics["bev_label_options"] = config["bev"]["labels"]
    diagnostics["planning_objects_config"] = config["planning_objects"]
    diagnostics["navsim_identity"] = identity
    return SceneSnapshot(identity["token"], 0.0, actor(packet["ego"]),
        [actor(row) for row in packet["actors"]], lanes, np.asarray(packet["route"], float),
        packet["route_lane_ids"], road, "navsim",
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
        if structured:
            fast["model_request_policy"] = model_request_policy(settings)
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
        goal_retry_policy(settings)
        self.fast = (StructuredServerVisionClient if structured else AuditedVisionClient)("fast", fast)
        self.extractor = DINOExtractor(self.config["dino"]["model"],
            device=self.config["dino"].get("device"), local_files_only=True)

    def compute(self, packet, output_dir, native_preview=None):
        attempt = {}
        try:
            return self._compute(packet, output_dir, attempt, native_preview)
        except Exception as exc:
            source = packet.get("identity") if isinstance(packet, dict) else None
            token = source.get("token") if isinstance(source, dict) else None
            try:
                fail_sample(output_dir, exc, attempt.get("state"), source_token=token)
                if attempt.get("state"):
                    record_sample_progress(output_dir, attempt["state"], "failed")
            except Exception as state_error:
                logger = logging.getLogger(__name__)
                logger.error("Failed to publish sample failure state: %s", type(state_error).__name__)
            raise

    def _compute(self, packet, output_dir, attempt, native_preview=None):
        started = time.perf_counter()
        root = filesystem_path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        packet = validate_packet(packet)
        identity = sample_identity(packet, self.run_identity)
        if self.resume_enabled:
            cached = load_completed_sample(root, identity)
            if cached is not None:
                return cached
        attempt["state"] = begin_sample(root, identity)
        state = attempt["state"]
        scene = snapshot_from_packet(packet, self.config)
        packet_path = write_input_packet(root, packet)
        config_hash = digest(self.config)
        timing = {}
        tick = time.perf_counter()
        record_sample_progress(root, state, "representations")
        reps = build_representations(scene, root / "representations", self.extractor, NineGridBuilder(self.config))
        timing["representations_s"] = time.perf_counter() - tick
        retrieved = RetrievalResult(scene.frame_id, snapshot_id=self.memory_snapshot_id, time=0.0)
        if self.settings["retrieve"]:
            tick = time.perf_counter()
            record_sample_progress(root, state, "retrieval")
            retrieved = retrieve(reps, self.records, self.memory_snapshot_id, self.config)
            timing["retrieval_s"] = time.perf_counter() - tick
        answer, model_error, model_error_detail, raw = None, None, None, None
        model_preparation_fallback = None
        model_validation_stage = "not_requested"
        model_attempts = []
        retry_policy = goal_retry_policy(self.settings)
        model_call_attempts = 0
        model_request_sent = False
        model_consistency = None
        transport_events = []
        effective_request_policy = None
        response_format = None
        scoped_for_model, _ = perception_scene(scene, self.config)
        if self.fast.enabled:
            tick = time.perf_counter()
            record_sample_progress(root, state, "model_preparation")
            attempts_before = int(getattr(self.fast, "attempts", 0))
            structured = self.settings.get("structured_model_output", False)
            effective_request_policy = model_request_policy(self.settings) if structured else None
            model_deadline = (time.monotonic() + effective_request_policy["total_deadline_s"]
                              if effective_request_policy else None)
            # Cache before build_prompt: prompt, request schema and strict parser
            # must use this exact original-frame reference and configured limits.
            reference = _goal_reference(scoped_for_model, self.config)
            # Frame/source/image integrity remains a hard input error, even if a
            # valid numerical base plan could be produced without model semantics.
            images = model_image_paths(scene, reps)
            current_scene_prompt = build_prompt(scoped_for_model, reps, retrieved)
            if structured:
                current_scene_prompt = refine_model_prompt(current_scene_prompt)
            prompt = ("评测输入条件：NAVSIM特权输入基线，使用当前及历史交通对象、车道地图、导航和当前灯态真值。"
                      "本帧候选真值不判断遮挡；只从给定候选ID中选择最多3个关注对象。"
                      "禁止推测任何未提供的未来真值。\n"
                      + current_scene_prompt)
            (root / "model_prompt.txt").write_text(prompt, encoding="utf-8")
            try:
                if structured:
                    response_format = structured_response_format(scoped_for_model, self.config, reference)
                    prompt += "\n" + guidance_prompt(scoped_for_model, reference)
                    (root / "model_prompt.txt").write_text(prompt, encoding="utf-8")
            except NoLegalGoalReference as exc:
                # Only an explicitly unavailable semantic reference/goal interval
                # skips HTTP. It does not request braking or alter the base MPC.
                model_error, model_error_detail = type(exc).__name__, str(exc)
                model_validation_stage = "preparation_unavailable"
                model_preparation_fallback = dict(reason=exc.reason_code,
                    detail=exc.detail, stage="structured_request_preparation", request_sent=False,
                    policy="same_base_mpc_without_model_semantics")
                (root / "model_prompt.txt").write_text(prompt + "\n模型请求未发送：" +
                    json.dumps(model_preparation_fallback, ensure_ascii=False), encoding="utf-8")
            else:
                if response_format is not None:
                    write_json(root / "model_response_format.json", response_format)
                current_prompt = prompt
                for semantic_index in range(retry_policy["limit"]+1):
                    record = dict(attempt=semantic_index+1,
                        prompt_sha256=hashlib.sha256(current_prompt.encode("utf-8")).hexdigest(),
                        original_reference_id=reference["reference_id"],
                        response_format_sha256=digest(response_format) if response_format else None)
                    raw, answer, model_error, model_error_detail = None, None, None, None
                    model_consistency = None

                    def audit_transport(row):
                        transport_events.append(dict(semantic_attempt=semantic_index+1,
                                                     **copy.deepcopy(row)))
                        write_json_gz(root / "model_transport_events.json.gz", dict(
                            schema="navsim-model-transport-events-1", frame_id=scene.frame_id,
                            sample_attempt_id=state["attempt_id"],
                            policy=effective_request_policy, events=transport_events))
                        choices = row.get("choices", [])
                        record_sample_progress(root, state, "model_request",
                            model_semantic_attempt=semantic_index+1,
                            model_request_attempt=int(getattr(self.fast, "attempts", 0))-attempts_before,
                            last_finish_reason=choices[0].get("finish_reason") if choices else None,
                            model_elapsed_s=time.perf_counter()-tick,
                            model_deadline_remaining_s=max(0., model_deadline-time.monotonic())
                                if model_deadline is not None else None)

                    def transport_progress(row):
                        record_sample_progress(root, state, "model_request",
                            model_semantic_attempt=semantic_index+1,
                            model_request_attempt=int(getattr(self.fast, "attempts", 0))-attempts_before,
                            model_elapsed_s=time.perf_counter()-tick,
                            model_deadline_remaining_s=max(0., model_deadline-time.monotonic())
                                if model_deadline is not None else None)

                    try:
                        record_sample_progress(root, state, "model_request",
                            model_semantic_attempt=semantic_index+1,
                            model_request_attempt=int(getattr(self.fast, "attempts", 0))-attempts_before+1)
                        raw = (self.fast.call_structured(current_prompt, images, response_format,
                                    deadline=model_deadline, on_attempt=audit_transport,
                                    on_progress=transport_progress) if structured
                               else self.fast.call(current_prompt, images))
                    except Exception as exc:
                        # Record only remote exception types, never credential-bearing text.
                        model_error = type(exc).__name__
                        transport = copy.deepcopy(getattr(self.fast, "last_transport_attempts", []))
                        choices = transport[-1].get("choices", []) if transport else []
                        model_validation_stage = getattr(exc, "validation_stage", None) or (
                            "response_truncated" if choices and choices[0].get("finish_reason") == "length"
                            else "transport_error")
                        model_error_detail = getattr(exc, "safe_detail", None)
                        raw = choices[0].get("raw") if choices else None
                    else:
                        record_sample_progress(root, state, "model_validation",
                            model_semantic_attempt=semantic_index+1)
                        try:
                            candidate_answer = parse_answer(raw, scoped_for_model, config=self.config, reference=reference)
                            model_consistency = (validate_answer_consistency(candidate_answer, scoped_for_model, reference)
                                                 if structured else None)
                            answer = candidate_answer
                        except GuidanceMismatch as exc:
                            model_error, model_error_detail = type(exc).__name__, str(exc)
                            model_validation_stage = ("attention_identity" if exc.reason_code == "attention_duplicate"
                                                      else "decision_goal_mismatch")
                        except ValueError as exc:
                            model_error, model_error_detail = type(exc).__name__, str(exc)
                            model_validation_stage = rejection_stage(raw, scoped_for_model, self.config, reference)
                        except Exception as exc:
                            model_error = type(exc).__name__
                            model_validation_stage = "parser_error"
                        else:
                            model_validation_stage = "accepted"
                    record.update(raw=raw, validation_stage=model_validation_stage,
                        error=model_error, error_detail=model_error_detail,
                        consistency=copy.deepcopy(model_consistency),
                        transport=copy.deepcopy(getattr(self.fast, "last_transport_attempts", [])))
                    if semantic_index:
                        record["feedback"] = dict(rejected_attempt=semantic_index,
                            reason=model_attempts[-1]["error_detail"], stage=model_attempts[-1]["validation_stage"])
                    model_attempts.append(record)
                    write_json_gz(root / "model_attempts.json.gz", dict(schema=MODEL_ATTEMPT_PROTOCOL,
                        frame_id=scene.frame_id, policy=retry_policy, request_policy=effective_request_policy,
                        attempts=model_attempts, final_stage=model_validation_stage))
                    record_sample_progress(root, state, "model_validation",
                        validation_stage=model_validation_stage,
                        model_elapsed_s=time.perf_counter()-tick)
                    if (model_validation_stage not in SEMANTIC_RETRY_STAGES
                            or semantic_index == retry_policy["limit"]):
                        break
                    current_prompt = goal_retry_prompt(prompt, raw, model_error_detail, semantic_index+2,
                                                      stage=model_validation_stage)
            model_call_attempts = int(getattr(self.fast, "attempts", 0)) - attempts_before
            model_request_sent = model_call_attempts > 0
            timing["model_s"] = time.perf_counter() - tick
            # Publish before planning so a subsequent numerical failure retains
            # the original answer, each bounded correction and transport evidence.
            write_json_gz(root / "model_attempts.json.gz", dict(schema=MODEL_ATTEMPT_PROTOCOL,
                frame_id=scene.frame_id, policy=retry_policy, attempts=model_attempts,
                request_policy=effective_request_policy,
                final_stage=model_validation_stage))
        tick = time.perf_counter()
        record_sample_progress(root, state, "prediction")
        scoped, predictions = prepare_planning_context(scene, self.config, answer=answer)
        timing["prediction_s"] = time.perf_counter() - tick
        # Official tokens may be processed in any order, including nearby samples
        # and synthetic follow-ups. Do not reuse state across independent anchors.
        planner = create_planner(self.config)
        tick = time.perf_counter()
        record_sample_progress(root, state, "planning")
        plan = planner.solve(scoped, predictions, answer)
        timing["planner_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        record_sample_progress(root, state, "reference")
        base_scene, base_predictions = prepare_planning_context(scene, self.config)
        baseline = create_planner(self.config).solve(base_scene, base_predictions)
        timing["reference_s"] = time.perf_counter() - tick
        execution_refinement = None
        final_plan, final_baseline = plan, baseline
        if self.config["navsim_execution_refinement"]["enabled"]:
            tick = time.perf_counter()
            record_sample_progress(root, state, "execution_refinement")
            main_execution = refine_execution(scoped, plan, predictions, self.config,
                                              native_preview, answer=answer,
                                              benchmark_profile=self.settings["benchmark_profile"])
            reference_execution = refine_execution(base_scene, baseline, base_predictions,
                                                   self.config, native_preview,
                                                   benchmark_profile=self.settings["benchmark_profile"])
            execution_refinement = dict(schema="navsim-execution-refinement-evidence-1",
                source_token=scene.frame_id, source_timestamp_us=packet["identity"]["timestamp_us"],
                benchmark_profile=self.settings["benchmark_profile"],
                uses_future_ground_truth=False, uses_metric_cache=False,
                controller_provenance=getattr(native_preview, "provenance", None),
                plan=plain(main_execution), reference=plain(reference_execution))
            final_plan = expected_execution_plan(scoped, plan, main_execution)
            final_baseline = expected_execution_plan(base_scene, baseline, reference_execution)
            timing["execution_refinement_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        record_sample_progress(root, state, "quality")
        quality = evaluate(scoped, final_plan, final_baseline, predictions, self.config)
        if execution_refinement is not None:
            audit = audit_expected_execution(quality, final_plan, main_execution, self.config)
        else:
            audit = execution_audit(scoped, plan, predictions, self.config,
                execution_horizon_s=self.config["navsim_execution"]["audit_horizon_s"])
        timing["internal_quality_s"] = time.perf_counter() - tick
        audit.update(policy="diagnostic_only", veto_enabled=False,
                     submission_replaced=False,
                     source="causal_native_controller_preview" if execution_refinement is not None else "submitted_optimizer_candidate",
                     executed_policy="完整四秒提交独立诊断；不否决或替换MPC输出")
        semantic_fallback=plan.diagnostics.get("semantic_fallback")
        model_semantics_applied=answer is not None and semantic_fallback is None
        checked_horizon = horizon_contract(self.config,
            plans=dict(plan=plan, reference=baseline),
            predictions=dict(prediction=predictions, reference_prediction=base_predictions))
        representation_paths = dict(bev=Path(reps.bev_path), feature=Path(reps.feature_image_path),
            grid=Path(reps.grid_path), occupancy=Path(reps.bev_path).with_name(
                Path(reps.bev_path).name.replace("_bev.png", "_occupancy.npz")))
        render_goal_overlay(scene, final_plan).save(root / "trajectory_bev.png")
        evidence = dict(identity=packet["identity"], input_condition=packet["input_condition"],
            benchmark_profile=self.settings.get("benchmark_profile", "v2_two_stage"),
            official_metric=self.settings.get("official_metric"), official_commit=self.settings.get("official_commit"),
            parameters_sha256=config_hash, parameters=self.config, timing_s=timing,
            model_answer=plain(answer), model_error=model_error, model_error_detail=model_error_detail, model_raw=raw,
            model_preparation_fallback=model_preparation_fallback,
            model_validation_stage=model_validation_stage, model_goal_retry_policy=retry_policy,
            model_request_policy=effective_request_policy, model_consistency=model_consistency,
            model_transport_events_asset="model_transport_events.json.gz" if transport_events else None,
            model_semantic_attempts=len(model_attempts),
            model_attempts_asset="model_attempts.json.gz" if self.fast.enabled else None,
            model_call_attempts=model_call_attempts, model_request_sent=model_request_sent,
            model_semantics_applied=model_semantics_applied, model_planning_fallback=plain(semantic_fallback),
            structured_model_output=self.settings.get("structured_model_output", False),
            model_response_format_sha256=digest(response_format) if response_format else None,
            retrieval=serialize_retrieval(retrieved, root), memory_snapshot_id=self.memory_snapshot_id,
            dino_signature=self.extractor.signature, plan=plain(plan), reference=plain(baseline),
            execution_refinement=execution_refinement,
            final_execution_diagnostics=plain(final_plan.diagnostics) if execution_refinement is not None else None,
            internal_quality_trajectory_source="causal_native_controller_preview" if execution_refinement is not None else "nominal_mpc",
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
        poses = (np.asarray(main_execution["submitted_poses"], dtype=np.float32)
                 if execution_refinement is not None else trajectory_from_centers(
                     plan.states, self.config["mpc"]["step_s"], scene.ego.rear_axle_to_center_m))
        timing["total_s"] = time.perf_counter() - started
        result = dict(poses=poses.tolist(), frame_id=scene.frame_id, sampling=dict(horizon_s=4.0, interval_s=0.5),
                      execution_mode=final_plan.diagnostics.get("execution_mode"), timing_s=timing,
                      safety_feasible=final_plan.diagnostics.get("safety_feasible"), evidence=str(root / "planning_evidence.json"),
                      trajectory_source="execution_aware_reference" if execution_refinement is not None else "nominal_mpc")
        result["condition_audit"] = dict(model_requested=self.fast.enabled, model_answer_valid=model_semantics_applied,
            benchmark_profile=self.settings.get("benchmark_profile", "v2_two_stage"),
            model_answer_parsed_valid=answer is not None, model_semantics_applied=model_semantics_applied,
            model_planning_fallback=plain(semantic_fallback),
            model_preparation_fallback=model_preparation_fallback,
            model_call_attempts=model_call_attempts, model_request_sent=model_request_sent,
            model_error=model_error, model_error_detail=model_error_detail,
            model_validation_stage=model_validation_stage, model_goal_retry_policy=retry_policy,
            model_request_policy=effective_request_policy, model_consistency=model_consistency,
            model_semantic_attempts=len(model_attempts),
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
            artifact_paths.append(root / "model_attempts.json.gz")
            if transport_events:
                artifact_paths.append(root / "model_transport_events.json.gz")
        if response_format is not None:
            artifact_paths.append(root / "model_response_format.json")
        complete_sample(root, identity, artifact_paths, attempt_id=attempt["state"]["attempt_id"])
        record_sample_progress(root, state, "complete", validation_stage=model_validation_stage)
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
            from .execution_rpc import WorkerPreview
            preview = WorkerPreview(protocol_stdout, sys.stdin, request["request_id"],
                                    request["packet"]["identity"]["token"],
                                    packet_sha256=digest(request["packet"]))
            result = pipeline.compute(request["packet"], request["output_dir"], native_preview=preview)
            reply = dict(ok=True, result=result, request_id=request["request_id"])
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = dict(ok=False, error=type(exc).__name__ + ": " + str(exc), request_id=request["request_id"])
            write_json(Path(request["output_dir"]) / "failure.json", reply)
        protocol_stdout.write("EP_NAVSIM_REPLY " + json.dumps(reply, allow_nan=False) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
