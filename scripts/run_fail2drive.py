"""Launch the pinned, unchanged official evaluator; no project episode clock."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = "bceb18afb7d62e06e990739e013e750d92ad7a88"
SIMULATOR_HF_COMMIT = "e8bd082943eb90d517c091d77c5f1bbc521f9391"
SIMULATOR_ARCHIVE_SIZE = 12783415657
SIMULATOR_ARCHIVE_SHA256 = "8f024e6accf70de27260c1c7e5f7c9e2ad82e63452e2a4809f84189437d7fc06"


def absolute(value, base=ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base/path).resolve()


def io_path(path):
    """Allow integrity checks of the long Windows development worktree path."""
    path = path.resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\"+str(path))
    return path


def verify_official(source):
    """Protect upstream scoring/runner files; never patch their behavior."""
    required = ("leaderboard/leaderboard/leaderboard_evaluator.py", "scenario_runner/srunner/scenariomanager/watchdog.py")
    for name in required:
        if not io_path(source/name).is_file():
            raise ValueError("incomplete official Fail2Drive checkout: "+name)
    if (source/".git").exists():
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if head != OFFICIAL_COMMIT:
            raise ValueError("official checkout SHA mismatch: "+head)
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"], text=True)
        if dirty.strip():
            raise ValueError("official checkout has changes; preserve evaluator/scoring source unchanged")
        return {"commit": head, "verification": "clean_git_checkout"}
    manifest_path = Path(str(source)+".source.json")
    if not manifest_path.is_file():
        raise ValueError("archive checkout needs adjacent .source.json provenance manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("commit") != OFFICIAL_COMMIT or not manifest.get("files"):
        raise ValueError("official source manifest SHA/files missing or mismatched")
    for name, expected in manifest["files"].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source):
            raise ValueError("source manifest path escapes official directory")
        if not io_path(path).is_file() or hashlib.sha256(io_path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("official source integrity mismatch: "+name)
    read_source = io_path(source)
    unexpected = [path.relative_to(read_source).as_posix() for path in read_source.rglob("*.py")
                  if "__pycache__" not in path.parts and path.relative_to(read_source).as_posix() not in manifest["files"]]
    if unexpected:
        raise ValueError("unrecorded Python source inside official checkout: "+unexpected[0])
    return {"commit": OFFICIAL_COMMIT, "verification": "archive_file_sha256", "manifest": str(manifest_path),
            "archive_sha256": manifest.get("archive_sha256")}


def verify_simulator(root):
    manifest_path = root/"_install/ep-driver-simulator-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("simulator installation identity manifest missing: "+str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"schema": "ep-driver-fail2drive-simulator-v1", "official_hf_commit": SIMULATOR_HF_COMMIT,
                "archive_size_bytes": SIMULATOR_ARCHIVE_SIZE, "archive_sha256": SIMULATOR_ARCHIVE_SHA256,
                "archive_actual_sha256": SIMULATOR_ARCHIVE_SHA256, "archive_verified": True, "launch_cwd": "."}
    if any(manifest.get(key) != value for key,value in expected.items()):
        raise ValueError("simulator installation manifest does not match the pinned official Fail2Drive archive")
    artifacts = manifest.get("artifacts", {})
    for name in ("launcher", "server_binary", "sdk_wheel"):
        record = artifacts.get(name, {})
        path = (root/record.get("path", "")).resolve()
        if not path.is_relative_to(root) or not io_path(path).is_file():
            raise ValueError("simulator installation artifact missing or outside root: "+name)
        digest = hashlib.sha256()
        with io_path(path).open("rb") as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record.get("sha256"):
            raise ValueError("simulator installation artifact SHA256 mismatch: "+name)
    return {"root": str(root), "manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "official_hf_commit": SIMULATOR_HF_COMMIT, "archive_sha256": SIMULATOR_ARCHIVE_SHA256,
            "launch_cwd": str(root), "launcher": str(root/artifacts["launcher"]["path"]),
            "sdk_wheel": str(root/artifacts["sdk_wheel"]["path"]),
            "verification": "installation_manifest_and_launcher_binary_sdk_hashes",
            "limitation": "not a running-process identity or exhaustive rehash of all map assets"}


def verify_sdk(evaluator_python, simulator):
    """Import the SDK without constructing a Client; bind loaded extension to the pinned wheel."""
    code = ("import importlib,json,sys,carla; m=importlib.import_module('carla.libcarla'); "
            "print(json.dumps({'python':sys.executable,'python_version':list(sys.version_info[:2]),"
            "'package':carla.__file__,'extension':m.__file__}))")
    sdk_env = dict(os.environ)
    sdk_root = Path(simulator.get("root", Path(simulator["sdk_wheel"]).parent))
    sdk_env["PYTHONPATH"] = str(sdk_root/"PythonAPI/carla")
    result = subprocess.run([str(evaluator_python), "-c", code], capture_output=True, text=True, timeout=30, env=sdk_env)
    if result.returncode:
        raise ValueError("evaluator Python cannot import the custom CARLA SDK: "+result.stderr.strip())
    try:
        identity = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("custom CARLA SDK preflight returned invalid JSON") from exc
    if identity.get("python_version") != [3,10]:
        raise ValueError("the pinned custom Fail2Drive wheel requires evaluator Python 3.10")
    extension = Path(identity["extension"]).resolve()
    if not extension.is_file():
        raise ValueError("installed CARLA extension is missing")
    installed = hashlib.sha256(extension.read_bytes()).hexdigest()
    with zipfile.ZipFile(simulator["sdk_wheel"]) as wheel:
        candidates = [name for name in wheel.namelist() if Path(name).name == extension.name]
        if len(candidates) != 1:
            raise ValueError("installed CARLA extension filename is absent or ambiguous in official wheel")
        digest = hashlib.sha256()
        with wheel.open(candidates[0]) as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024), b""):
                digest.update(chunk)
    if installed != digest.hexdigest():
        raise ValueError("installed CARLA extension differs from the pinned custom SDK wheel")
    return dict(identity, extension_sha256=installed, wheel_member=candidates[0],
                verification="installed_extension_matches_pinned_wheel", connected_to_carla=False)


def route_identity(path):
    match = re.fullmatch(r"(?:Base|Generalization)_.+_(\d{4})\.xml", path.name)
    if not match:
        raise ValueError("use an official single-route split XML named Base/Generalization_*_####.xml")
    route_id = int(match.group(1))
    rows = ET.parse(path).getroot().findall("route")
    if len(rows) != 1 or int(rows[0].attrib["id"]) != route_id:
        raise ValueError("each parser-compatible result must contain exactly its named official route")
    return route_id


def adapter_source_hash():
    code = hashlib.sha256()
    source_files = sorted([*(ROOT/"src/nine_grid").rglob("*.py"),
                           *(ROOT/"src/ep_driver_fail2drive").rglob("*.py"), Path(__file__)])
    for path in source_files:
        code.update(path.relative_to(ROOT).as_posix().encode())
        code.update(path.read_bytes())
    return code.hexdigest()


def official_result(checkpoint, route_id):
    """Read the upstream verdict; a finished process is not a completed route."""
    try:
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        records = data["_checkpoint"]["records"]
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("expected one official record")
        record = records[0]
        if record["route_id"] != "RouteScenario_%d_rep0" % route_id:
            raise ValueError("official route identity mismatch")
        scores = record["scores"]
        completion = scores["score_route"]
        if type(completion) not in (int, float) or not math.isfinite(completion):
            raise ValueError("invalid official completion score")
        checkpoint_data = data["_checkpoint"]
        global_record = checkpoint_data.get("global_record")
        valid = (data.get("entry_status") == "Finished" and data.get("eligible") is True
                 and checkpoint_data.get("progress") == [1, 1]
                 and isinstance(global_record, dict)
                 and global_record.get("status") in ("Perfect", "Completed", "Failed"))
        record_completed = record["status"] in ("Perfect", "Completed") and completion >= 100.
        completed = valid and record_completed and global_record["status"] in ("Perfect", "Completed")
        return dict(route_id=route_id, official_status=record["status"],
                    entry_status=data.get("entry_status"), route_completed=completed,
                    official_result_valid=valid, record_completed=record_completed,
                    official_scores=scores, checkpoint=str(checkpoint))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(route_id=route_id, route_completed=False,
                    official_status="result_missing_or_invalid", checkpoint=str(checkpoint))


def fingerprint(config, args, routes):
    code_hash = getattr(args,"_adapter_source_sha256",None) or adapter_source_hash()
    inputs = {"official_commit": OFFICIAL_COMMIT, "adapter_source_sha256": code_hash,
              "routes_sha256": hashlib.sha256(routes.read_bytes()).hexdigest(),
              "routes_subset": args.routes_subset, "repetitions": 1,
              "official_route_id": config["archive_association"]["official_route_id"],
              "repetition_seed": args.seed,
              "traffic_manager_seed": config["archive_association"]["traffic_manager_seed"],
              "track": "MAP", "physics_hz": 20, "config": config,
              "input_config_sha256": {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                  for key in ("parameters_path", "models_path", "prompts_path")}}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


def prepare(args, preflight=None):
    source = absolute(args.fail2drive_root)
    if preflight is None:
        provenance = verify_official(source)
        simulator = verify_simulator(absolute(args.simulator_root))
        sdk = verify_sdk(args.evaluator_python or sys.executable, simulator)
    else:
        provenance,simulator,sdk = preflight
    config = json.loads(absolute(args.config).read_text(encoding="utf-8"))
    if config.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("runtime config official_commit does not match adapter lock")
    for key in ("parameters_path", "models_path", "prompts_path"):
        config[key] = str(absolute(getattr(args, key) or config[key]))
        if not Path(config[key]).is_file():
            raise ValueError("missing "+key+": "+config[key])
    worker = args.worker_python or config["worker_python"]
    worker = shutil.which(worker) or str(absolute(worker))
    if not Path(worker).is_file():
        raise ValueError("worker Python executable not found: "+worker)
    config["worker_python"] = worker
    config["simulator_installation"] = simulator
    config["evaluator_sdk"] = sdk
    config["planner_hz"] = args.planner_hz or config.get("planner_hz", 10)
    config["video"] = args.video
    if args.output_dir:
        method_root = absolute(args.output_dir)
    else:
        method_root = absolute(config["output_dir"])/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    routes = absolute(args.routes, source)
    if not routes.is_file() or routes.suffix.lower() != ".xml":
        raise ValueError("routes must name an existing official XML file")
    # No shortened XML, rewritten task, custom timeout or private scenario metadata enters the Agent.
    if not routes.is_relative_to(source):
        raise ValueError("use routes from the pinned official checkout")
    route_id = route_identity(routes)
    traffic_manager_seed = route_id % 1000 + 10000*args.seed
    seed_root = method_root/str(args.seed)
    output = seed_root/"agent_runs"/("%04d" % route_id)
    checkpoint = seed_root/"res"/("%04d_res.json" % route_id)
    config["output_dir"] = str(output)
    config["archive_association"] = {"official_routes_path": str(routes), "routes_subset": args.routes_subset,
        "official_route_id": route_id, "repetition_seed": args.seed, "traffic_manager_seed": traffic_manager_seed,
        "repetitions": 1, "official_checkpoint": str(checkpoint), "method_results_root": str(method_root),
        "model_input": False}
    runtime_path = seed_root/"run"/("%04d_agent_runtime.json" % route_id)
    launch_path = seed_root/"run"/("%04d_launch.json" % route_id)
    command = [args.evaluator_python or sys.executable,
        str(source/"leaderboard/leaderboard/leaderboard_evaluator.py"),
        "--host", args.host, "--port", str(args.port),
        "--traffic-manager-port", str(args.traffic_manager_port),
        "--traffic-manager-seed", str(traffic_manager_seed), "--timeout", "300", "--track", "MAP",
        "--routes", str(routes), "--repetitions", "1",
        "--agent", str(ROOT/"src/ep_driver_fail2drive/agent.py"), "--agent-config", str(runtime_path),
        "--checkpoint", str(checkpoint),
        "--debug-checkpoint", str(seed_root/"out"/("%04d_live.txt" % route_id)), "--debug", str(args.debug)]
    if args.routes_subset:
        command += ["--routes-subset", args.routes_subset]
    if args.record:
        command += ["--record", str(seed_root/"recorder"/("%04d" % route_id))]
    if args.resume:
        command += ["--resume", "True"]
    env = dict(os.environ)
    # Do not inherit old CARLA egg/agents paths from a user's other experiment shell.
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT/"src"), str(Path(simulator["root"])/"PythonAPI/carla"),
                                         str(source), str(source/"leaderboard"), str(source/"scenario_runner")])
    env["SCENARIO_RUNNER_ROOT"] = str(source/"scenario_runner")
    env["LEADERBOARD_ROOT"] = str(source/"leaderboard")
    env["VIZ_PATH"] = str(seed_root/"viz"/("%04d" % route_id))
    evaluator_path = shutil.which(command[0]) or str(absolute(command[0]))
    env["PATH"] = str(Path(evaluator_path).parent)+os.pathsep+env.get("PATH", "")
    identity, inputs = fingerprint(config, args, routes)
    if args.resume:
        if not runtime_path.is_file() or not launch_path.is_file() or not checkpoint.is_file():
            raise ValueError("resume requires existing runtime, launch provenance and official checkpoint")
        previous = json.loads(launch_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != identity:
            raise ValueError("resume fingerprint mismatch: code, route, seed or configuration changed; use a new output directory")
        if json.loads(runtime_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume runtime configuration differs from archived configuration")
    return config, runtime_path, command, env, provenance, identity, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail2drive-root", required=True)
    parser.add_argument("--simulator-root", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--routes-subset", default="")
    parser.add_argument("--route-ids", nargs="+", type=int, help="Select IDs from the official route directory")
    parser.add_argument("--config", default="config/fail2drive.example.json")
    parser.add_argument("--parameters-path")
    parser.add_argument("--models-path")
    parser.add_argument("--prompts-path")
    parser.add_argument("--worker-python")
    parser.add_argument("--evaluator-python")
    parser.add_argument("--output-dir")
    parser.add_argument("--planner-hz", type=int, choices=(1,2,10))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39000)
    parser.add_argument("--traffic-manager-port", type=int, default=39003)
    parser.add_argument("--seed", "--repetition-seed", dest="seed", type=int,
                        help="Official repetition seed, not the derived traffic-manager seed")
    parser.add_argument("--seeds", nargs="+", type=int, help="Repetition seeds; directory default: 1 2 3")
    parser.add_argument("--repetitions", type=int, choices=(1,), default=1,
                        help="Exactly one evaluator record per official parser result file")
    parser.add_argument("--debug", type=int, choices=(0,1,2,3), default=0)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and print launch plan without writing or connecting CARLA")
    parser.add_argument("--prepare-only", action="store_true", help="write runtime/launch provenance without starting evaluator")
    args = parser.parse_args()
    if args.seed is not None and args.seeds is not None:
        parser.error("choose --seed for one repetition seed or --seeds for several")
    if args.port != 39000 or args.traffic_manager_port != 39003:
        parser.error("this installation uses isolated Fail2Drive RPC 39000 and traffic-manager 39003; old CARLA ports must remain untouched")
    try:
        source = absolute(args.fail2drive_root)
        route_input = absolute(args.routes, source)
        if route_input.is_dir():
            if args.routes_subset:
                raise ValueError("directory runs select --route-ids; --routes-subset applies to one XML only")
            routes = sorted(route_input.glob("*.xml"))
            seeds = args.seeds or ([args.seed] if args.seed is not None else [1,2,3])
        else:
            routes = [route_input]
            seeds = args.seeds or [args.seed if args.seed is not None else 1]
        if not routes or not seeds or len(set(seeds)) != len(seeds) or any(seed < 1 for seed in seeds):
            raise ValueError("official repetition seeds must be distinct positive integers and routes nonempty")
        identified = [(route_identity(path), path) for path in routes]
        if len({identity for identity,_ in identified}) != len(identified):
            raise ValueError("route IDs must be unique within a method result directory")
        if args.route_ids:
            requested = set(args.route_ids)
            if not requested.issubset({identity for identity,_ in identified}):
                raise ValueError("selected --route-ids are absent from the official route input")
            identified = [(identity,path) for identity,path in identified if identity in requested]
        provenance = verify_official(source)
        simulator = verify_simulator(absolute(args.simulator_root))
        sdk = verify_sdk(args.evaluator_python or sys.executable, simulator)
        preflight = provenance,simulator,sdk
        args._adapter_source_sha256 = adapter_source_hash()
        if args.output_dir is None:
            base = json.loads(absolute(args.config).read_text(encoding="utf-8"))["output_dir"]
            args.output_dir = str(absolute(base)/datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        jobs = []
        for seed in seeds:
            for route_id,path in identified:
                job_args = argparse.Namespace(**vars(args))
                job_args.seed,job_args.routes = seed,str(path)
                if args.resume and len(seeds)*len(identified)>1:
                    seed_root=absolute(args.output_dir)/str(seed)
                    existing=(seed_root/"run"/("%04d_agent_runtime.json" % route_id),
                              seed_root/"run"/("%04d_launch.json" % route_id),
                              seed_root/"res"/("%04d_res.json" % route_id),
                              seed_root/"agent_runs"/("%04d" % route_id))
                    if not any(path.exists() for path in existing):
                        job_args.resume=False  # Untouched jobs in an explicitly resumed batch are fresh jobs.
                    elif (existing[0].is_file() and existing[1].is_file() and not existing[2].exists()
                            and (not existing[3].exists() or not any(existing[3].iterdir()))):
                        job_args.resume=False  # Prepared-only identity is checked again before use below.
                jobs.append((job_args,prepare(job_args,preflight)))
    except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    preview = [{"route_id": job[1][0]["archive_association"]["official_route_id"],
                "repetition_seed": job[0].seed,
                "traffic_manager_seed": job[1][0]["archive_association"]["traffic_manager_seed"],
                "checkpoint": job[1][0]["archive_association"]["official_checkpoint"]} for job in jobs[:8]]
    print(json.dumps({"command": jobs[0][1][2] if len(jobs)==1 else None, "official": provenance,
        "method_results_root": args.output_dir, "execution_mode": "B_model", "dry_run": args.dry_run,
        "jobs": len(jobs), "repetition_seeds": seeds, "preview": preview,
        "custom_episode_limit": None, "simulator_started_by_launcher": False}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    failed_routes = 0
    for job_args,prepared in jobs:
        config,runtime_path,command,env,provenance,identity,inputs = prepared
        association = config["archive_association"]
        route_id = association["official_route_id"]
        seed_root = runtime_path.parent.parent
        output = Path(config["output_dir"])
        checkpoint = Path(association["official_checkpoint"])
        launch_path = runtime_path.with_name("%04d_launch.json" % route_id)
        if not job_args.resume:
            existing = runtime_path.exists() or launch_path.exists() or checkpoint.exists() or (output.exists() and any(output.iterdir()))
            prepared_only = (runtime_path.is_file() and launch_path.is_file() and not checkpoint.exists()
                and (not output.exists() or not any(output.iterdir()))
                and json.loads(launch_path.read_text(encoding="utf-8")).get("fingerprint")==identity
                and json.loads(runtime_path.read_text(encoding="utf-8"))==config)
            if existing and not prepared_only:
                parser.error("route output is not empty; choose a new method directory or explicit --resume")
            if not prepared_only:
                for folder in ("run", "res", "out", "err", "agent_runs"):
                    (seed_root/folder).mkdir(parents=True, exist_ok=True)
                output.mkdir(parents=True, exist_ok=True)
                Path(env["VIZ_PATH"]).mkdir(parents=True, exist_ok=True)
                (seed_root/"recorder"/("%04d" % route_id)).mkdir(parents=True, exist_ok=True)
                runtime_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
                launch_path.write_text(json.dumps({"command": command, "official": provenance,
                    "fingerprint": identity, "fingerprint_inputs": inputs,
                    "official_routes_unmodified": True, "custom_episode_limit": None}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.prepare_only:
            if adapter_source_hash() != inputs["adapter_source_sha256"]:
                parser.error("adapter source changed after the batch plan; stop and use a new consistent result identity")
            mode = "a" if job_args.resume else "w"
            print("Running official route %04d, repetition seed %d, TM seed %d" %
                  (route_id,job_args.seed,association["traffic_manager_seed"]), flush=True)
            with (seed_root/"out"/("%04d_out.log" % route_id)).open(mode, encoding="utf-8") as stdout, \
                    (seed_root/"err"/("%04d_err.log" % route_id)).open(mode, encoding="utf-8") as stderr:
                result = subprocess.call(command, env=env, cwd=ROOT, stdout=stdout, stderr=stderr)
            if result:
                return result
            verdict = official_result(checkpoint, route_id)
            print(json.dumps(dict(verdict, repetition_seed=job_args.seed), ensure_ascii=False), flush=True)
            failed_routes += int(not verdict["route_completed"])
    # Keep collecting a requested batch after route-level failures, but do not
    # report shell success merely because upstream cleaned up normally.
    return 2 if failed_routes else 0


if __name__ == "__main__":
    raise SystemExit(main())
