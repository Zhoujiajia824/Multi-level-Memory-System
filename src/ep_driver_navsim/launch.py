"""Launch pinned, unmodified official scoring scripts with explicit provenance."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

from . import INPUT_CONDITION, OFFICIAL_COMMIT
from .protocol import digest, write_json
from .progress import ProgressMonitor, progress_interval

ROOT = Path(__file__).resolve().parents[2]


def evaluation_worker_count(value=1, platform=None):
    """Bound official process parallelism; Windows remains sequential."""
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError("evaluation_workers must be an integer between 1 and 8")
    return 1 if (platform or sys.platform) == "win32" else value


def freeze_model_request_budget(settings):
    """Record the exact policy used by the bounded NAVSIM structured client."""
    if "model_request_policy" not in settings and not settings.get("structured_model_output", False):
        return
    from .model_transport import model_request_policy
    policy = model_request_policy(settings)
    if settings.get("fast", "off") != "off":
        outer = settings.get("timeout_s", 600.)
        if (isinstance(outer, bool) or not isinstance(outer, (int, float))
                or not math.isfinite(outer) or policy["total_deadline_s"]+60 > outer):
            raise ValueError("Core timeout must reserve at least 60 seconds beyond the shared model deadline")
    settings["model_request_policy"] = policy


def source_fingerprint(root, subdir="navsim"):
    root = Path(root).resolve()
    files = sorted((p for p in (root / subdir).rglob("*") if p.is_file() and p.suffix in (".py", ".yaml")),
                   key=lambda p: p.relative_to(root).as_posix())
    entries = [p.relative_to(root).as_posix() + ":" + hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
               for p in files]
    return len(files), hashlib.sha256("\n".join(entries).encode()).hexdigest()


def verify_sdk(sdk_root):
    lock = json.loads((ROOT / "config/navsim/official_v2_2.json").read_text(encoding="utf-8"))
    count, fingerprint = source_fingerprint(sdk_root)
    if count != lock["source_files"] or fingerprint != lock["source_sha256"]:
        raise ValueError("NAVSIM SDK source/config does not match pinned official v2.2; do not patch scoring code")
    return lock


def split_contract(sdk_root, split):
    root = Path(sdk_root).resolve() / "navsim/planning/script/config/common/train_test_split"
    filter_path = root / "scene_filter" / (split + ".yaml")
    mapping_path = root / (split + ".yaml")
    scene_filter = yaml.safe_load(filter_path.read_text(encoding="utf-8"))
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))["reactive_all_mapping"]
    originals = set(scene_filter["tokens"])
    synthetic = set(scene_filter["reactive_synthetic_initial_tokens"])
    required = originals | synthetic
    mapped = set()
    for now, previous, second_stage in mapping:
        mapped.update((now, previous))
        for pair in second_stage:
            mapped.update(pair)
    if mapped != required:
        raise ValueError("Official two-stage mapping/filter coverage differs; cannot define the evaluation set")
    return dict(evaluation_tokens=sorted(required), evaluation_logs=sorted(scene_filter["log_names"]),
                original_tokens=sorted(originals), reactive_synthetic_tokens=sorted(synthetic),
                original_count=len(originals), reactive_synthetic_count=len(synthetic),
                scene_filter_sha256=hashlib.sha256(filter_path.read_bytes()).hexdigest(),
                mapping_sha256=hashlib.sha256(mapping_path.read_bytes()).hexdigest())


def check_weight_file(model):
    model = Path(model)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "dinov2" or config.get("hidden_size") != 768:
        raise ValueError("DINO retrieval requires a local DINOv2 Base model")
    weights = model / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError("Local DINO model.safetensors is required")
    with weights.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if size < 2 or size > min(16 * 1024 * 1024, weights.stat().st_size - 8):
            raise ValueError("DINO weights are invalid (possibly a Git LFS pointer): " + str(weights))
        header = json.loads(stream.read(size))
    if not isinstance(header, dict) or not any("data_offsets" in value for value in header.values() if isinstance(value, dict)):
        raise ValueError("DINO safetensors contains no tensors")


def stage_maps(data_root, destination):
    """Keep nuPlan's .maplocks and any derived files outside the source dataset."""
    source, target = Path(data_root) / "maps", Path(destination)
    if target.resolve() == source.resolve():
        raise ValueError("maps_runtime_root must differ from the read-only source dataset")
    metadata = source / "nuplan-maps-v1.0.json"
    catalog = json.loads(metadata.read_text(encoding="utf-8"))
    files = [metadata] + [source / location / catalog[location]["version"] / "map.gpkg"
                         for location in ("sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood")]
    hashes = {}
    for original in files:
        relative = original.relative_to(source)
        output = target / relative
        with original.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        if not output.is_file() or output.stat().st_size != original.stat().st_size:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, output)
        with output.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != fingerprint:
                raise ValueError("Staged map differs from source; choose an empty maps_runtime_root: " + str(output))
        hashes[relative.as_posix()] = fingerprint
    return hashes


def prepare_settings(config_path, split_override=None, output_override=None, bev_video_override=None):
    source = Path(config_path).resolve()
    settings = json.loads(source.read_text(encoding="utf-8"))
    required = {"sdk_root", "sdk_python", "core_python", "data_root", "dino_model", "output_dir"}
    if not required.issubset(settings):
        raise ValueError("Missing NAVSIM runtime paths: " + ",".join(sorted(required - set(settings))))
    def path(value):
        value = Path(value).expanduser()
        return str((value if value.is_absolute() else ROOT / value).resolve())
    defaults = dict(parameters="config/parameters.json", models_config="config/models.example.yaml",
                    prompts_config="config/prompts.yaml", split="warmup_two_stage", fast="off", retrieve=False,
                    memory_read_dir=None, desired_speed_mps=10.0, map_radius_m=160.0,
                    dino_device="cpu", timeout_s=600.0, bev_video=True, resume=True,
                    model_revision=None, model_service={}, structured_model_output=False,
                    evaluation_workers=1, progress_interval_s=15.)
    settings = dict(defaults, **settings)
    settings.setdefault("model_goal_retry_limit", 1 if settings["fast"] == "server" else 0)
    if (type(settings["model_goal_retry_limit"]) is not int or settings["model_goal_retry_limit"] not in (0, 1)
            or (settings["fast"] == "off" and settings["model_goal_retry_limit"] != 0)):
        raise ValueError("model_goal_retry_limit must be 0 or 1 (0 when fast=off)")
    settings["evaluation_workers"] = evaluation_worker_count(settings["evaluation_workers"])
    settings["progress_interval_s"] = progress_interval(settings["progress_interval_s"])
    if split_override:
        settings["split"] = split_override
    if output_override:
        settings["output_dir"] = output_override
    if bev_video_override is not None:
        settings["bev_video"] = bev_video_override
    if type(settings["bev_video"]) is not bool:
        raise ValueError("bev_video must be true or false")
    if type(settings["resume"]) is not bool:
        raise ValueError("resume must be true or false")
    if type(settings["structured_model_output"]) is not bool:
        raise ValueError("structured_model_output must be true or false")
    if not isinstance(settings["model_service"], dict):
        raise ValueError("model_service must describe the declared server preprocessing")
    if settings["split"] not in ("warmup_two_stage", "navhard_two_stage"):
        raise ValueError("Only pinned v2.2 warmup_two_stage/navhard_two_stage are supported")
    if settings["fast"] not in ("off", "api", "server") or type(settings["retrieve"]) is not bool:
        raise ValueError("Invalid model/retrieval mode")
    if settings["structured_model_output"] and settings["fast"] != "server":
        raise ValueError("structured_model_output requires fast=server")
    freeze_model_request_budget(settings)
    if settings["retrieve"] and (settings["fast"] == "off" or not settings["memory_read_dir"]):
        raise ValueError("Frozen retrieval requires an enabled fast model and memory_read_dir")
    for key in required | {"parameters", "models_config", "prompts_config"}:
        settings[key] = path(settings[key])
    if settings["memory_read_dir"]:
        settings["memory_read_dir"] = path(settings["memory_read_dir"])
    settings["metric_cache_path"] = path(settings.get("metric_cache_path",
        "outputs/navsim/metric_cache_v2_2/" + settings["split"]))
    settings["maps_runtime_root"] = path(settings.get("maps_runtime_root", "outputs/navsim/maps_runtime"))
    for key in ("sdk_python", "core_python", "parameters", "models_config", "prompts_config"):
        if not Path(settings[key]).is_file():
            raise FileNotFoundError(key + ": " + settings[key])
    settings["config_file_sha256"] = {key: hashlib.sha256(Path(settings[key]).read_bytes()).hexdigest()
                                     for key in ("parameters", "models_config", "prompts_config")}
    if settings["fast"] != "off":
        if settings["resume"] and (not isinstance(settings["model_revision"], str)
                                   or not settings["model_revision"].strip()):
            raise ValueError("Model resume requires an explicit model_revision; use resume=false for an unpinned API model")
        model = yaml.safe_load(Path(settings["models_config"]).read_text(encoding="utf-8"))["fast"]
        if not model.get("model") or not model.get("base_url"):
            raise ValueError("Enabled fast model requires model and base_url in models_config")
        if settings["fast"] == "api" and not os.environ.get(model.get("api_key_env", "")):
            raise ValueError("Set the fast model API key environment variable before starting: " +
                             str(model.get("api_key_env", "<missing api_key_env>")))
    # Validate before loading all scene indices, maps, DINO or the official SDK.
    # A declaration that says compact while the live service permits runaway
    # whitespace must never silently start another full B evaluation.
    from .service_validation import validate_model_service
    settings.pop("model_service_validation", None)
    service_validation = validate_model_service(settings)
    if service_validation is not None:
        settings["model_service_validation"] = service_validation
        print("Model service validation: " + ("verified" if service_validation["verified"] else "MANUAL_UNVERIFIED")
              + " (" + service_validation["verification_method"] + ")", flush=True)
    data = Path(settings["data_root"])
    for name in ("maps", "navsim_logs/test", "sensor_blobs/test", settings["split"] + "/sensor_blobs",
                 settings["split"] + "/synthetic_scene_pickles"):
        if not (data / name).is_dir():
            raise FileNotFoundError("Missing data directory: " + str(data / name))
    lock = verify_sdk(settings["sdk_root"])
    settings.update(split_contract(settings["sdk_root"], settings["split"]))
    settings.update(memory_snapshot_id=hashlib.sha256(b"[]").hexdigest(), memory_record_count=0)
    if settings["retrieve"]:
        from .memory import load_memory_snapshot
        started = time.perf_counter()
        print("Validating the frozen memory library and excluding evaluation sources; large libraries may take several minutes.", flush=True)
        records, snapshot_id, selection = load_memory_snapshot(
            settings["memory_read_dir"], settings["evaluation_logs"], settings["evaluation_tokens"])
        if not records:
            raise ValueError("Retrieval requested but no compatible audited memory was loaded")
        settings.update(memory_snapshot_id=snapshot_id, memory_record_count=len(records), memory_selection=selection)
        print("Frozen memory: selected=%d source=%d excluded_overlap=%d reader_rejected=%d validation_s=%.1f" %
              (len(records), selection["source_records"], len(selection["excluded_records"]),
               len(selection["reader_rejected_records"]), time.perf_counter()-started), flush=True)
    check_weight_file(settings["dino_model"])
    settings["map_source_sha256"] = stage_maps(data, settings["maps_runtime_root"])
    settings.update(input_condition=INPUT_CONDITION, standard_sensor_only_comparable=False,
                    official_metric=lock["metric"], official_commit=OFFICIAL_COMMIT,
                    official_source_sha256=lock["source_sha256"], online_memory_write=False,
                    reflection_enabled=False, root=str(ROOT),
                    numerical_source_sha256=source_fingerprint(ROOT, "src")[1])
    frozen = Path(settings["output_dir"]) / "run_settings.json"
    settings["settings_path"] = str(frozen)
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != settings:
        raise ValueError("Output directory belongs to different settings/code; choose a new output_dir")
    write_json(frozen, settings)
    if service_validation is not None:
        write_json(Path(settings["output_dir"]) / "model_service_validation.json", service_validation)
    return settings


def environment(settings):
    env = dict(os.environ)
    env.pop("EP_NAVSIM_REPLAY_SETTINGS", None)
    env.pop("EP_NAVSIM_EVALUATION_WORKERS", None)
    env.update(OPENSCENE_DATA_ROOT=settings["data_root"], NUPLAN_MAPS_ROOT=settings["maps_runtime_root"],
               NUPLAN_MAP_VERSION="nuplan-maps-v1.0", NAVSIM_DEVKIT_ROOT=settings["sdk_root"],
               NAVSIM_EXP_ROOT=str(Path(settings["output_dir"]).parent),
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + settings["sdk_root"] + os.pathsep + env.get("PYTHONPATH", "")
    return env


def scoring_environment(settings):
    env = environment(settings)
    env["EP_NAVSIM_EVALUATION_WORKERS"] = str(evaluation_worker_count(settings.get("evaluation_workers", 1)))
    # This observer only exports the official scorer's already computed output.
    # It is installed before the unchanged official script imports pdm_score.
    # Minimal numerical records are independent of optional video rendering.
    env["EP_NAVSIM_REPLAY_SETTINGS"] = settings["settings_path"]
    return env


def render_existing(config_path, output_override=None, token=None):
    """Render saved traces with official SDK plotting, without inference/scoring."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_override or config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    settings = json.loads((output / "run_settings.json").read_text(encoding="utf-8"))
    samples = output / "samples"
    paths = ([samples / digest(token)[:24] / "official_replay" / "trace.json.gz"]
             if token else sorted(samples.glob("*/official_replay/trace.json.gz")))
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("No saved official scoring trace; old planned BEV/CSV cannot reconstruct reactive execution")
    failures = []
    for path in paths:
        try:
            subprocess.run([settings["core_python"], "-m", "ep_driver_navsim.replay_render",
                            "--replay", str(path), "--output-dir", str(path.parent)],
                           env=environment(settings), check=True, cwd=ROOT)
        except subprocess.CalledProcessError as exc:
            write_json(path.parent / "render_status.json", dict(status="failed", error=str(exc),
                       failure_scope="visualization_only", official_score_unchanged=True))
            failures.append(str(path))
    from .replay_gallery import update_gallery
    gallery = update_gallery(output)
    print("Official scoring replay gallery:", gallery, flush=True)
    if failures:
        raise RuntimeError("BEV rendering failed for %d saved traces; scoring results unchanged" % len(failures))
    return 0


def run_sdk(settings, action):
    subprocess.run([settings["sdk_python"], "-m", "ep_driver_navsim.sdk_runtime", action,
                    "--settings", settings["settings_path"]], env=environment(settings), check=True, cwd=ROOT)


def cache_contract(settings):
    return {key: settings[key] for key in ("official_commit", "official_source_sha256", "split",
            "scene_filter_sha256", "mapping_sha256", "data_root", "map_source_sha256")}


def verify_cache_contract(settings, creating=False):
    path = Path(settings["metric_cache_path"])
    lock = path / "ep_navsim_cache_contract.json"
    if lock.exists():
        if json.loads(lock.read_text(encoding="utf-8")) != cache_contract(settings):
            raise ValueError("Metric cache belongs to another SDK/split/dataset")
    elif path.exists() and any(path.iterdir()):
        raise ValueError("Existing cache has no v2.2 provenance; use a new empty metric_cache_path")
    elif not creating:
        raise FileNotFoundError("Generate the official metric cache first")
    return lock


def normalize_cache_metadata(settings):
    """The official loader splits paths on '/'; normalize our own cache CSVs."""
    root = Path(settings["metric_cache_path"]).resolve()
    for path in (root / "metadata").glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows or rows[0] != ["file_name"] or any(len(row) != 1 for row in rows[1:]):
            raise ValueError("Unexpected official cache metadata schema")
        normalized = []
        for row in rows[1:]:
            target = Path(row[0]).resolve()
            if not target.is_relative_to(root) or not target.is_file() or "," in str(target):
                raise ValueError("Official cache metadata refers to a missing/unsupported path")
            normalized.append([target.as_posix()])
        if rows[1:] != normalized:
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows([rows[0]] + normalized)


def official_command(settings, cache=False):
    script = "run_metric_caching.py" if cache else "run_pdm_score.py"
    sdk = Path(settings["sdk_root"])
    # These are direct Hydra overrides of the official entry point; no local
    # scoring formula, custom aggregation or changed traffic policy is used.
    quote = lambda value: json.dumps(str(value).replace("\\", "/"))
    workers = 1 if cache else evaluation_worker_count(settings.get("evaluation_workers", 1))
    command = [settings["sdk_python"], "-m", "ep_driver_navsim.sdk_bootstrap",
        str(sdk / "navsim/planning/script" / script),
        "train_test_split=" + settings["split"],
        "worker=" + ("single_machine_thread_pool" if workers > 1 else "sequential"), "gpu=false",
        "navsim_log_path=" + quote(Path(settings["data_root"]) / "navsim_logs/test"),
        "original_sensor_path=" + quote(Path(settings["data_root"]) / "sensor_blobs/test"),
        "synthetic_sensor_path=" + quote(Path(settings["data_root"]) / settings["split"] / "sensor_blobs"),
        "synthetic_scenes_path=" + quote(Path(settings["data_root"]) / settings["split"] / "synthetic_scene_pickles"),
        "metric_cache_path=" + quote(settings["metric_cache_path"])]
    if cache:
        command += ["force_feature_computation=false"]
    else:
        if workers > 1:
            command += ["worker.use_process_pool=true", "worker.max_workers=" + str(workers)]
        command += ["agent._target_=ep_driver_navsim.agent.EPDriverPrivilegedAgent",
                    "+agent.settings_path=" + quote(settings["settings_path"]),
                    "+agent.core_python=" + quote(settings["core_python"]),
                    "+agent.timeout_s=" + str(settings["timeout_s"]),
                    "experiment_name=ep_driver_privileged_" + settings["split"],
                    "output_dir=" + quote(Path(settings["output_dir"]) / "official_scores")]
    return command


def verify_scores(settings, csv_path):
    from .evaluation_result import validate_csv, sample_state
    from nine_grid.paths import filesystem_path
    checked = validate_csv(settings, csv_path)
    expected = set(settings["evaluation_tokens"])
    result = dict(status="complete", phase="complete", official_csv=str(csv_path),
                official_two_stage_epdms=checked["scores"]["combined"],
                official_stage_scores={stage: checked["scores"][stage] for stage in ("stage_one", "stage_two")},
                official_metrics=checked["metrics"], official_stage_counts=checked["stage_counts"],
                evaluated_tokens=len(expected), failed_tokens=0, input_condition=INPUT_CONDITION,
                standard_sensor_only_comparable=False, official_commit=OFFICIAL_COMMIT,
                split=settings["split"], settings_sha256=digest(settings))
    result["trajectory_sampling"] = dict(horizon_s=4.0, interval_s=0.5, poses=8,
                                         short_plan_padding_allowed=False)
    if settings.get("model_service_validation") is not None:
        # This is the safe allowlisted summary frozen by prepare_settings, not
        # the service's raw configuration or environment.
        result["model_service_validation"] = json.loads(json.dumps(settings["model_service_validation"]))
    if settings.get("output_dir") is not None:
        audits = []
        model_requested = settings.get("fast", "off") in ("api", "server")
        for token in sorted(expected):
            path = filesystem_path(settings["output_dir"]) / "samples" / digest(token)[:24] / "trajectory.json"
            state, state_error = sample_state(path.parent, token)
            if state_error is not None:
                raise RuntimeError("Official benchmark sample attempt is not complete: " + token)
            try:
                trajectory = json.loads(path.read_text(encoding="utf-8"))
                audit = trajectory.get("condition_audit")
                poses = trajectory["poses"]
                valid_poses = (len(poses) == 8 and all(len(pose) == 3 and all(
                    math.isfinite(float(value)) for value in pose) for pose in poses))
            except (OSError, ValueError, TypeError, KeyError, OverflowError):
                raise RuntimeError("Official benchmark lacks a valid submitted trajectory: " + token) from None
            if (not valid_poses or trajectory.get("frame_id") != token
                    or trajectory.get("sampling") != dict(horizon_s=4.0, interval_s=0.5)):
                raise RuntimeError("Official benchmark trajectory token/horizon disagrees: " + token)
            if state is not None and (trajectory.get("resume") or {}).get("identity_sha256") != state["identity_sha256"]:
                raise RuntimeError("Official benchmark trajectory belongs to another sample attempt: " + token)
            if (not audit or audit.get("model_requested") is not model_requested
                    or audit.get("retrieval_enabled") is not settings.get("retrieve", False)):
                raise RuntimeError("Official benchmark lacks consistent per-sample condition evidence: " + token)
            audits.append(audit)
        valid = sum(audit.get("model_semantics_applied", audit.get("model_answer_valid")) is True for audit in audits)
        parsed = sum(audit.get("model_answer_parsed_valid", audit.get("model_answer_valid")) is True for audit in audits)
        hits = sum(bool(audit.get("retrieved_memory_ids")) for audit in audits)
        def attempts(audit, key):
            value = audit.get(key)
            return value if type(value) is int and value >= 0 else 0
        validation_stages = Counter(str(audit.get("model_validation_stage") or
            ("unarchived" if model_requested else "not_requested")) for audit in audits)
        result["condition_audit"] = dict(model_requested=len(audits) if model_requested else 0, valid_model_answers=valid,
            parsed_model_answers=parsed, model_semantics_applied=valid,
            model_semantic_attempts=sum(attempts(audit, "model_semantic_attempts") for audit in audits),
            model_call_attempts=sum(attempts(audit, "model_call_attempts") for audit in audits),
            samples_with_goal_retry=sum(attempts(audit, "model_semantic_attempts") > 1 for audit in audits),
            model_validation_stages=dict(validation_stages),
            base_fallback_samples=len(audits)-valid if model_requested else 0, samples_with_retrieved_memories=hits,
            all_samples_received_valid_model_answer=valid == len(audits) if model_requested else None,
            memory_snapshot_id=settings.get("memory_snapshot_id"), memory_record_count=settings.get("memory_record_count", 0))
    return result


def evaluate(settings):
    from .run_lock import evaluation_lock
    # Acquire before publishing any lifecycle state so a rejected second
    # launcher cannot replace the running owner's benchmark_result.json.
    with evaluation_lock(settings) as lease:
        return _evaluate_locked(settings, lease)


def _wait_owned_child(child, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return child.wait(timeout=max(0., deadline - time.monotonic()))
        except KeyboardInterrupt:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(child.args, timeout) from None


def _stop_official_child(child):
    """Stop only our explicitly created SDK child/session, with bounded waits."""
    if os.name != "nt":
        try:
            if os.getpgid(child.pid) != child.pid or os.getsid(child.pid) != child.pid:
                raise RuntimeError("Owned official child no longer has its own session; refusing group signal")
        except ProcessLookupError:
            # The group can outlive its leader; its ID remains reserved while
            # those descendants exist. It is the session created by Popen.
            pass
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif child.poll() is None:
        child.terminate()
    try:
        _wait_owned_child(child, 10.)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            child.kill()
        _wait_owned_child(child, 5.)
    if os.name != "nt":
        # A multiprocessing worker may still be finishing after its main
        # process exited. Do not let a log-sized batch continue model calls.
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_official(settings, lease):
    """Keep ownership until the official child exits, including interruption."""
    interval = progress_interval(settings.get("progress_interval_s", 15.))
    monitor = ProgressMonitor(settings) if interval else None
    child = subprocess.Popen(official_command(settings), env=scoring_environment(settings), cwd=ROOT,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                             start_new_session=os.name != "nt", **lease.child_options())
    try:
        if monitor is not None:
            monitor.emit("official_scoring_started")
        while True:
            try:
                code = child.wait(timeout=interval or None)
                break
            except subprocess.TimeoutExpired:
                if monitor is not None:
                    monitor.emit()
    except BaseException:
        # No PID scan: only the child/session explicitly created immediately
        # above is managed. External deployed model/viewer services are outside.
        _stop_official_child(child)
        if monitor is not None:
            monitor.emit("official_scoring_interrupted")
        raise
    if code:
        _stop_official_child(child)
        if monitor is not None:
            monitor.emit("official_scoring_failed")
        raise subprocess.CalledProcessError(code, child.args)
    if monitor is not None:
        monitor.emit("official_scoring_finished")


def _evaluate_locked(settings, lease):
    """Run official evaluation, preserving truthful status even when it fails."""
    from .evaluation_result import artifact_audit, run_state, utc_now
    previous = Path(settings["output_dir"]) / "benchmark_result.json"
    if previous.is_file() and json.loads(previous.read_text(encoding="utf-8")).get("status") == "complete":
        raise ValueError("This output directory already contains a complete benchmark; choose a new output_dir")
    started, phase, candidates = utc_now(), "cache_verification", []
    run_state(settings, phase, started_utc=started)
    try:
        verify_cache_contract(settings)
        normalize_cache_metadata(settings)
        phase = "data_inspection"
        run_state(settings, phase, started_utc=started)
        run_sdk(settings, "inspect")
        phase = "cache_verification"
        run_state(settings, phase, started_utc=started)
        run_sdk(settings, "check-cache")
        output = Path(settings["output_dir"]) / "official_scores"
        before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in output.glob("*.csv")}
        phase = "official_scoring"
        run_state(settings, phase, started_utc=started)
        run_official(settings, lease)
        phase = "official_result_validation"
        candidates = [path for path in output.glob("*.csv")
                      if before.get(path) != (path.stat().st_size, path.stat().st_mtime_ns)]
        run_state(settings, phase, started_utc=started)
        if len(candidates) != 1:
            raise RuntimeError("Expected one newly generated official result CSV")
        result = verify_scores(settings, candidates[0])
        audit = artifact_audit(settings)
        if audit["missing_trajectory_tokens"] or audit["planning_failure_tokens"]:
            raise RuntimeError("Official CSV and per-sample trajectory/failure evidence disagree")
        result.update(started_utc=started, finished_utc=utc_now(), artifact_audit=audit)
        write_json(Path(settings["output_dir"]) / "benchmark_result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    except BaseException as exc:
        # Never replace partial CSVs or sample artifacts with invented scores.
        # A rerun/resume can inspect this record instead of mistaking missing
        # benchmark_result.json for a still-running or successfully finished run.
        try:
            audit = artifact_audit(settings)
        except Exception as audit_error:
            audit = dict(planning_failure_tokens=[], audit_error=repr(audit_error))
        if phase == "official_scoring":
            candidates = list((Path(settings["output_dir"]) / "official_scores").glob("*.csv"))
        run_state(settings, phase, status="failed", started_utc=started, error=exc,
                  artifact_audit=audit, failed_tokens=len(audit["planning_failure_tokens"]),
                  official_csv_candidates=[str(path) for path in candidates])
        raise
    finally:
        if settings["bev_video"]:
            try:
                from .replay_gallery import update_gallery
                print("Official scoring replay gallery:", update_gallery(settings["output_dir"]), flush=True)
            except Exception as exc:
                # Optional gallery output must not mask a numerical failure or
                # invalidate an otherwise complete official benchmark result.
                print("Official replay gallery refresh failed (scores unchanged): " + str(exc), file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="EP-Driver privileged NAVSIM v2.2 official evaluation")
    parser.add_argument("action", choices=("check", "smoke", "cache", "evaluate", "render", "gallery"))
    parser.add_argument("--config", default="config/navsim.local.json")
    parser.add_argument("--split", choices=("warmup_two_stage", "navhard_two_stage"))
    parser.add_argument("--output-dir")
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--bev-video", dest="bev_video", action="store_true", default=None,
                       help="Save official execution BEV MP4 and trace for each scored sample (default)")
    video.add_argument("--no-bev-video", dest="bev_video", action="store_false")
    parser.add_argument("--token", help="Render one saved token only; does not restrict official evaluation")
    args = parser.parse_args()
    if args.action in ("render", "gallery"):
        if args.split or args.bev_video is not None:
            parser.error("render/gallery uses existing run settings; only --config/--output-dir/--token apply")
        if args.action == "gallery":
            if args.token:
                parser.error("--token only applies to rendering saved traces")
            from .replay_gallery import update_gallery
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            output = Path(args.output_dir or config["output_dir"])
            print(update_gallery(output if output.is_absolute() else ROOT / output))
            return 0
        return render_existing(args.config, args.output_dir, args.token)
    if args.token:
        parser.error("--token only applies to rendering saved traces")
    if args.action == "smoke" and args.output_dir is None:
        args.output_dir = str(ROOT / "test" / "artifacts" / "navsim" / ("smoke_" + str(time.time_ns())))
    settings = prepare_settings(args.config, args.split, args.output_dir, args.bev_video)
    print("Input condition:", INPUT_CONDITION, "(NOT standard sensor-only)", flush=True)
    print("Official split:", settings["split"], "original:", settings["original_count"],
          "reactive synthetic:", settings["reactive_synthetic_count"], flush=True)
    if args.action in ("check", "smoke"):
        run_sdk(settings, "inspect" if args.action == "check" else "smoke")
    elif args.action == "cache":
        run_sdk(settings, "inspect")
        lock = verify_cache_contract(settings, creating=True)
        # Write provenance before work begins so an interrupted cache generation
        # can resume safely; coverage is checked again before any scoring.
        write_json(lock, cache_contract(settings))
        subprocess.run(official_command(settings, cache=True), check=True, env=environment(settings), cwd=ROOT)
        normalize_cache_metadata(settings)
        run_sdk(settings, "check-cache")
    else:
        evaluate(settings)
    return 0
