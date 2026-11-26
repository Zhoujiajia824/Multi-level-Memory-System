"""Resumable current-pose/map checks; never official metric-cache entries.

Only this adapter's metadata lives below ``ep_navigation_preflight``. A cached
failure remains a failure. A report passes only after every selected token has
been checked against its current public input and the same source/map contract.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

from nine_grid.paths import filesystem_path
from .protocol import digest, write_json
from .profiles import get_profile, is_two_stage

SCHEMA = "ep-navsim-navigation-preflight-4"
MAP_VERSION = "nuplan-maps-v1.0"
SCOPE = "all_current_poses_causal_yaw_rates_motion_commands_public_routes_maps_and_initialization_corridors;no_future_trajectory_or_metric_cache"


def worker_count(settings):
    count = settings.get("navigation_preflight_workers", 1)
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError("navigation_preflight_workers must be an integer from 1 to 32")
    return 1 if os.name == "nt" else count


def verified_runtime(settings):
    """Recheck real SDK/map bytes even when called with an old settings file."""
    from .launch import ROOT, source_fingerprint
    if source_fingerprint(settings["sdk_root"])[1] != settings["official_source_sha256"]:
        raise ValueError("Navigation SDK source differs from the verified settings")
    if source_fingerprint(ROOT, "src")[1] != settings["numerical_source_sha256"]:
        raise ValueError("Navigation adapter source differs from the verified settings")
    root = filesystem_path(settings["maps_runtime_root"])
    for name, expected in settings["map_source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe map fingerprint path")
        path = filesystem_path(root / relative)
        # filesystem_path resolves both paths (including links) before this
        # relative comparison; it also normalizes Windows extended paths.
        path.relative_to(root)
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != expected:
            raise ValueError("Navigation map source differs from the verified settings: " + name)
    packages = {}
    for package in ("numpy", "shapely", "nuplan-devkit", "geopandas", "fiona", "pyogrio", "pyquaternion"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    import shapely
    geos = getattr(shapely, "geos_version_string", None)
    if geos is None:
        from shapely.geos import geos_version_string
        geos = geos_version_string
    return dict(python=sys.version, packages=packages, geos_version=geos)


def cache_contract(settings, tokens):
    # Include actual local code, rather than trusting an old frozen setting.
    source = Path(__file__).resolve().parent
    adapter = {name: hashlib.sha256((source / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
               for name in ("nav_preflight.py", "sdk_runtime.py", "input_adapter.py",
                            "protocol.py", "sdk_bootstrap.py")}
    for name in ('navigation_initialization.py','geometry.py','potential_domain.py'):
        adapter['nine_grid/'+name] = hashlib.sha256(
            (source.parent/'nine_grid'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    required = ("official_commit", "official_source_sha256", "map_source_sha256",
                "scene_filter_sha256", "numerical_source_sha256")
    if is_two_stage(settings):
        required += ("mapping_sha256",)
    if any(not settings.get(key) for key in required):
        raise ValueError("Navigation preflight requires verified SDK/map/source fingerprints")
    return dict(schema=SCHEMA, adapter_source_sha256=adapter,
        benchmark_profile=get_profile(settings)["id"],
        official_metric=settings.get("official_metric"),
        verified_sources={key: settings[key] for key in required},
        selected_tokens_sha256=digest(tokens), selected_token_count=len(tokens),
        split=settings["split"], data_root=str(Path(settings["data_root"]).resolve()),
        maps_root=str(Path(settings["maps_runtime_root"]).resolve()), map_version=MAP_VERSION,
        desired_speed_mps=float(settings["desired_speed_mps"]),
        map_radius_m=float(settings["map_radius_m"]),
        runtime=verified_runtime(settings),
        semantics="pinned_map_api_current_center_from_causal_rear_axle_motion_public_route_and_validated_initialization_corridor",
        scope=SCOPE, official_metric_cache=False)


def extract_task(loader, token):
    """Read the current raw row and preceding yaw/time; never a future row.

    Synthetic files use the official trusted pickle format. Deserialization is
    unavoidable, but no future row/trajectory/annotation is inspected or sent.
    Original yaw uses precisely the quaternion convention in Scene's builder.
    """
    synthetic_scenes = getattr(loader, "synthetic_scenes", {})
    if token in synthetic_scenes:
        with filesystem_path(synthetic_scenes[token][0]).open("rb") as stream:
            value = pickle.load(stream)
        metadata = value["scene_metadata"]
        if metadata["num_history_frames"] != 4:
            raise ValueError("Synthetic history count differs from official protocol")
        now = value["frames"][3]
        if str(metadata["initial_token"]) != str(now["token"]):
            raise ValueError("Scene metadata/current frame token mismatch")
        pose = now["ego_status"]["ego_pose"]
        previous = value['frames'][2]
        previous_yaw = float(previous['ego_status']['ego_pose'][2])
        velocity = now["ego_status"]["ego_velocity"]
        command = now["ego_status"]["driving_command"]
        map_name = metadata["map_name"]
    else:
        from pyquaternion import Quaternion
        now = loader.scene_frames_dicts[token][3]
        translation = now["ego2global_translation"]
        velocity = now["ego_dynamic_state"][:2]
        command = now["driving_command"]
        pose = [translation[0], translation[1],
                Quaternion(*now["ego2global_rotation"]).yaw_pitch_roll[0]]
        previous = loader.scene_frames_dicts[token][2]
        previous_yaw = float(Quaternion(*previous['ego2global_rotation']).yaw_pitch_roll[0])
        map_name = now["map_location"]
    if str(now["token"]) != token:
        raise ValueError("Requested navigation token differs from the current frame")
    pose = [float(value) for value in pose]
    if len(pose) != 3 or not all(math.isfinite(value) for value in pose):
        raise ValueError("Navigation current pose must contain three finite values")
    if not isinstance(map_name, str) or not map_name:
        raise ValueError("Navigation map name is missing")
    from .input_adapter import navigation_command,navigation_ego
    dt = (int(now['timestamp'])-int(previous['timestamp']))/1e6
    if not math.isfinite(previous_yaw) or dt<=0:
        raise ValueError('Navigation history yaw/time is invalid')
    yaw_rate = ((pose[2]-previous_yaw+math.pi)%(2*math.pi)-math.pi)/dt
    current_ego = navigation_ego(velocity,yaw_rate)
    current_speed = math.hypot(*current_ego['velocity'])
    return dict(token=token, map_name=map_name, map_version=MAP_VERSION, origin=pose,
                public_roadblock_ids=[str(value) for value in now["roadblock_ids"]],
                source_anchor_token=str(now["token"]), timestamp_us=int(now["timestamp"]),
                current_speed_mps=current_speed, driving_command=navigation_command(command)["one_hot"],
                rear_axle_velocity_mps=[float(x) for x in velocity],yaw_rate_rps=yaw_rate,
                history_frames=4, input_scope="current_pose_causal_motion_command_public_route_map_only")


def check_map_task(task, contract):
    """Runs in one process; map objects are never passed through the executor."""
    from .sdk_bootstrap import platform_setup
    platform_setup()
    import numpy as np
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from .input_adapter import _map_packet,navigation_ego,validate_navigation_initialization

    api = get_maps_api(contract["maps_root"], task["map_version"], task["map_name"])
    ego = navigation_ego(task['rear_axle_velocity_mps'],task['yaw_rate_rps'])
    current_speed = math.hypot(*ego['velocity'])
    if not math.isclose(current_speed,task['current_speed_mps'],rel_tol=0.,abs_tol=1e-8):
        raise ValueError('Navigation preflight current-state speed differs from packet helper')
    result = _map_packet(api, np.asarray(task["origin"], float), task["public_roadblock_ids"],
                         contract["desired_speed_mps"], contract["map_radius_m"],
                         current_speed_mps=current_speed, driving_command=task["driving_command"])
    ego['lane_id'] = result[-1]['observed_current_lane_id']
    route,recovery = validate_navigation_initialization(
        ego,result[0],result[1],result[2],result[4],result[-1],task['yaw_rate_rps'],task['token'])
    # Store a compact causal validation result, not another copy of route/map.
    recovery = {key:value for key,value in recovery.items() if key not in ('points_world','headings_world')}
    return dict(route_evidence=result[-1], usable_lane_count=len(result[0]),
                route_point_count=len(route), skipped_lane_ids=result[-2],
                initialization_recovery=dict(recovery,independently_validated=True))


def _error(exc):
    return dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())


def _record(token, task, contract_sha, evidence=None, error=None, phase="map_check"):
    value = dict(schema=SCHEMA, complete=True, token=token, contract_sha256=contract_sha,
                 task=task, task_sha256=digest(task), phase=phase,
                 status="failed" if error else "passed", evidence=evidence, error=error)
    return dict(value, record_sha256=digest(value))


def _cached(path, token, task, contract_sha):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        raw = {key: value for key, value in record.items() if key != "record_sha256"}
        if (record["schema"] != SCHEMA or record["complete"] is not True
                or record["token"] != token or record["contract_sha256"] != contract_sha
                or record["task"] != task or record["task_sha256"] != digest(task)
                or record["record_sha256"] != digest(raw) or record["phase"] != "map_check"):
            return None
        if record["status"] == "passed":
            evidence = record["evidence"]
            route = evidence["route_evidence"]
            if (record["error"] is not None or evidence["usable_lane_count"] < 1
                    or evidence["route_point_count"] < 2
                    or type(route["topology_correction_changed"]) is not bool
                    or not math.isfinite(route["forward_reference_m"])
                    or not math.isfinite(route["required_forward_reference_m"])
                    or route["required_forward_reference_m"] <= 0
                    or route["forward_reference_m"] < 0
                    or evidence['initialization_recovery']['status'] not in ('active','disabled','reference_only_unverified')
                    or evidence['initialization_recovery']['reference_feasibility_is_input_gate'] is not False
                    or evidence['initialization_recovery']['independently_validated'] is not True):
                return None
        elif record["status"] == "failed":
            if not record["error"]["type"] or not isinstance(record["error"]["message"], str):
                return None
        else:
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def inspect_navigation(settings, loader, map_checker=None):
    """Checkpoint every result and validate all current inputs before passing.

    Optional workers only parallelize this map check, with a spawn context and
    a bounded queue. Windows stays serial. Nothing invokes models, predictors,
    metric-cache generation, simulators or scorers.
    """
    tokens = sorted(settings["evaluation_tokens"])
    if not tokens or len(tokens) != len(set(tokens)) or not all(isinstance(t, str) and t for t in tokens):
        raise ValueError("Navigation preflight needs a nonempty unique token selection")
    workers = worker_count(settings)
    if map_checker is not None and workers != 1:
        raise ValueError("Injected map checkers are only supported in serial offline checks")
    checker = map_checker or check_map_task
    report_path = filesystem_path(settings["output_dir"]) / "navigation_preflight.json"
    write_json(report_path, dict(schema=SCHEMA, status="in_progress", complete=False,
        expected_tokens=len(tokens), checked_tokens=0, pending_tokens=len(tokens),
        scope=SCOPE, official_metric_cache=False, phase="verifying_sources"))
    contract = cache_contract(settings, tokens)
    contract_sha = digest(contract)
    metric_root = filesystem_path(settings["metric_cache_path"])
    cache_base = filesystem_path(settings.get("navigation_preflight_cache") or
                      metric_root.with_name(metric_root.name + "_ep_navigation_preflight"))
    if cache_base.is_relative_to(metric_root):
        raise ValueError("Navigation preflight cache must be outside the official metric cache")
    cache_dir = cache_base / contract_sha
    cache_dir.mkdir(parents=True, exist_ok=True)
    results, reused, computed = {}, 0, 0
    started = time.perf_counter()

    def publish(complete=False, interrupted=None):
        failed = [dict(token=token, task=row["task"], error=row["error"], phase=row["phase"])
                  for token, row in sorted(results.items()) if row["status"] == "failed"]
        corrected = [dict(token=token, **row["evidence"]["route_evidence"])
                     for token, row in sorted(results.items()) if row["status"] == "passed"
                     and row["evidence"]["route_evidence"]["topology_correction_changed"]]
        degraded=[dict(token=token,reasons=row['evidence']['initialization_recovery'].get('reference_quality_reasons',[]),
                       status=row['evidence']['initialization_recovery']['status'])
                  for token,row in sorted(results.items()) if row['status']=='passed'
                  and row['evidence']['initialization_recovery'].get('reference_quality_reasons')]
        reason_counts=Counter(reason for row in degraded for reason in row['reasons'])
        report = dict(schema=SCHEMA, status=("failed" if failed else "passed") if complete else "in_progress",
            complete=complete, expected_tokens=len(tokens), checked_tokens=len(results),
            pending_tokens=len(tokens) - len(results), failed_token_count=len(failed),
            failed_tokens=failed, topology_corrected_tokens=corrected,
            degraded_reference_count=len(degraded),degraded_references=degraded,
            reference_quality_reason_counts=dict(sorted(reason_counts.items())),
            reference_feasibility_is_input_gate=False,
            selected_tokens_sha256=digest(tokens), checked_tokens_sha256=digest(sorted(results)),
            task_manifest_sha256=digest([[token, row["task_sha256"]] for token, row in sorted(results.items())]),
            cache_contract=contract, cache_contract_sha256=contract_sha, cache_dir=str(cache_dir.resolve()),
            reused_tokens=reused, computed_tokens=computed, workers=workers,
            elapsed_s=time.perf_counter() - started, benchmark_result=False,
            official_metric_cache=False, scope=SCOPE, interruption=interrupted)
        write_json(report_path, report)
        return report

    def accept(record, path, reuse=False):
        nonlocal reused, computed
        if not reuse:
            # Each failure is durable before updating any aggregate progress.
            write_json(path, record)
        results[record["token"]] = record
        reused += int(reuse)
        computed += int(not reuse)
        if record["status"] == "failed" or len(results) % 50 == 0:
            publish()
            print("Navigation preflight: %d/%d; failed=%d; reused=%d" % (
                len(results), len(tokens), sum(row["status"] == "failed" for row in results.values()), reused), flush=True)

    pending = {}
    executor = None
    publish()  # Supersede an older passed report before reading any new inputs.
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))

        def drain():
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                token, task, path = pending.pop(future)
                try:
                    record = _record(token, task, contract_sha, evidence=future.result())
                except Exception as exc:
                    record = _record(token, task, contract_sha, error=_error(exc))
                accept(record, path)

        for token in tokens:
            path = cache_dir / (digest(token) + ".json")
            try:
                task = extract_task(loader, token)
            except Exception as exc:
                # Missing/unreadable inputs cannot be safely reused, even as failures.
                accept(_record(token, dict(token=token), contract_sha, error=_error(exc),
                               phase="input_extraction"), path)
                continue
            cached = _cached(path, token, task, contract_sha)
            if cached is not None:
                accept(cached, path, reuse=True)
            elif executor is not None:
                pending[executor.submit(check_map_task, task, contract)] = (token, task, path)
                if len(pending) >= workers * 2:
                    drain()
            else:
                try:
                    record = _record(token, task, contract_sha, evidence=checker(task, contract))
                except Exception as exc:
                    record = _record(token, task, contract_sha, error=_error(exc))
                accept(record, path)
        while pending:
            drain()
        if set(results) != set(tokens):
            raise RuntimeError("Navigation preflight did not cover the exact selected token set")
        if cache_contract(settings, tokens) != contract:
            raise RuntimeError("Navigation sources or runtime changed during preflight")
        report = publish(complete=True)
    except BaseException as exc:
        publish(interrupted=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    print("Navigation preflight: %d/%d; failed=%d; reused=%d; complete=true" % (
        len(results), len(tokens), report["failed_token_count"], reused), flush=True)
    if report["failed_token_count"]:
        raise RuntimeError("NAVSIM navigation preflight failed for %d/%d tokens; see navigation_preflight.json" %
                           (report["failed_token_count"], len(tokens)))
    return dict(checked_tokens=len(tokens), failed_tokens=0, complete=True,
                degraded_reference_count=report['degraded_reference_count'],
                reference_quality_reason_counts=report['reference_quality_reason_counts'],
                topology_corrected_tokens=len(report["topology_corrected_tokens"]),
                reused_tokens=reused, computed_tokens=computed,
                cache_contract_sha256=contract_sha, workers=workers)
