"""Manually serve shared read-only assets with private state and a read-only root.

The default is a preview. No public manager is imported or executed, and no
existing process is stopped. A real launch requires Linux bubblewrap; failure
never falls back to an unprotected process.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from .service_profiles import load_service_profile, SERVICE_PROFILES


def gpu_ids(value):
    try:
        result = [int(part) for part in value.split(",")]
    except ValueError:
        raise ValueError("--gpus must be an explicit comma-separated GPU list") from None
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise ValueError("GPU IDs must be nonnegative and distinct")
    return result


def _nvidia(query):
    result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15)
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def check_gpus(profile, selected):
    if len(selected) not in profile["gpu_counts"]:
        raise ValueError("Selected GPU count is unsupported by the owned 32B profile")
    rows = _nvidia("--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu")
    inventory = {int(row[0]): dict(uuid=row[1], total_mib=int(row[2]),
        used_mib=int(row[3]), utilization=int(row[4])) for row in rows}
    if not set(selected).issubset(inventory):
        raise ValueError("One or more explicitly selected GPU IDs do not exist")
    occupied = {row[0] for row in _nvidia("--query-compute-apps=gpu_uuid,pid") if len(row) == 2}
    busy = [index for index in selected if inventory[index]["used_mib"] >= 1024
        or inventory[index]["utilization"] > 0 or inventory[index]["uuid"] in occupied]
    if busy:
        raise ValueError("Selected GPUs are occupied; no process will be stopped: " + str(busy))
    if sum(inventory[index]["total_mib"] for index in selected) < profile["min_total_memory_gib"]*1024:
        raise ValueError("Selected GPUs have insufficient total VRAM for this BF16 profile")
    return {str(index): inventory[index] for index in selected}


def check_port(profile):
    with socket.socket() as sock:
        try:
            sock.bind((profile["host"], profile["port"]))
        except OSError:
            raise ValueError("Owned service port is occupied; reuse or inspect it manually, never stop it automatically") from None


def private_root(value, public):
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("--runtime-root must be an absolute directory owned by you")
    root, public = candidate.resolve(), Path(public).resolve()
    if root == public or root.is_relative_to(public) or public.is_relative_to(root):
        raise ValueError("Runtime directory must be separate from the public asset directory")
    existing = root
    while not existing.exists():
        existing = existing.parent
    if not existing.is_dir():
        raise ValueError("Runtime directory has a non-directory ancestor")
    if hasattr(os, "getuid") and existing.stat().st_uid != os.getuid():
        raise ValueError("Runtime directory/nearest existing parent must be owned by the current user")
    return root


def _child(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Private runtime subtree escapes through a symlink: " + relative)
    return target


def cache_environment(profile, root, python):
    """Only return a controlled allowlist; never print or archive the parent env."""
    folders = dict(XDG_CACHE_HOME="cache", XDG_CONFIG_HOME="config", XDG_DATA_HOME="data",
        HF_HOME="cache/huggingface", HF_HUB_CACHE="cache/huggingface/hub",
        HUGGINGFACE_HUB_CACHE="cache/huggingface/hub", HF_MODULES_CACHE="cache/huggingface/modules",
        TORCH_HOME="cache/torch", TORCH_EXTENSIONS_DIR="cache/torch_extensions",
        TORCHINDUCTOR_CACHE_DIR="cache/torchinductor", TRITON_CACHE_DIR="cache/triton",
        TRITON_DUMP_DIR="cache/triton_dump", CUDA_CACHE_PATH="cache/cuda", NUMBA_CACHE_DIR="cache/numba",
        VLLM_CACHE_ROOT="cache/vllm", VLLM_CONFIG_ROOT="config/vllm",
        VLLM_ASSETS_CACHE="cache/vllm/assets", VLLM_XLA_CACHE_PATH="cache/vllm/xla",
        VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="cache/vllm/flashinfer_autotune",
        VLLM_TORCH_PROFILER_DIR="profiles", FLASHINFER_WORKSPACE_BASE="flashinfer",
        MPLCONFIGDIR="config/matplotlib", RAY_TMPDIR="tmp/ray", TMPDIR="tmp", TMP="tmp", TEMP="tmp")
    env = {key: str(_child(root, name)) for key, name in folders.items()}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
        VLLM_NO_USAGE_STATS="1", VLLM_DO_NOT_TRACK="1", FLASHINFER_DISABLE_VERSION_CHECK="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID", PATH=str(python.parent)+os.pathsep+"/usr/local/bin:/usr/bin:/bin")
    return env, sorted(set(folders.values()) | {"work", "logs", "toolchain"})


def assets(profile):
    public = Path(profile["public_root"]).resolve()
    model = (public / profile["model_path"]).resolve()
    # Our explicitly selected existing interpreter is independent of the
    # public venv, whose uv symlink can point into another user's private home.
    # Never fall back to it, change permissions or install/update packages.
    python = Path(profile["python_executable"])
    if not model.is_relative_to(public) or not (model / "config.json").is_file():
        raise ValueError("Public model path is missing or escapes its configured root")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Explicit owned interpreter/its symlink target is not executable by this user")
    if python.resolve().is_relative_to(public):
        raise ValueError("Owned interpreter must not resolve into the public environment")
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise ValueError("Public weight index is empty")
    for shard in shards:
        path = (model / shard).resolve()
        if not path.is_relative_to(model) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Public weight shard is missing, empty or outside its model directory")
    return public, model, python


def model_command(profile, model, python, count):
    compact = lambda value: json.dumps(value, separators=(",", ":"))
    return [str(python), "-m", "vllm.entrypoints.cli.main", "serve", str(model),
        "--served-model-name", profile["served_model_name"], "--host", profile["host"],
        "--port", str(profile["port"]), "--tensor-parallel-size", str(count),
        "--dtype", profile["dtype"], "--gpu-memory-utilization", str(profile["gpu_memory_utilization"]),
        "--max-model-len", str(profile["max_model_len"]), "--max-num-seqs", str(profile["max_num_seqs"]),
        "--limit-mm-per-prompt", compact(dict(image=profile["image_limit"], video=0)),
        "--mm-processor-kwargs", compact(dict(max_pixels=profile["max_pixels_per_image"])),
        "--override-generation-config", compact(dict(max_new_tokens=profile["max_new_tokens"])),
        "--structured-outputs-config", compact(profile["structured_outputs"]),
        "--default-chat-template-kwargs", compact(dict(enable_thinking=profile["enable_thinking"]))]


def prepare_plan(profile, runtime, selected, *, bwrap=None):
    public, model, python = assets(profile)
    root = private_root(runtime, public)
    for protected in {python.parent.parent.resolve(), python.resolve().parent.parent.resolve()}:
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError("Runtime directory must not overlap the owned Python environment")
    repository = Path(__file__).resolve().parents[2]
    if repository.is_relative_to(root):
        raise ValueError("Runtime directory must not expose the project source root as writable")
    executable = bwrap or shutil.which("bwrap")
    if not executable:
        raise ValueError("bubblewrap is required for a read-only filesystem; no unprotected fallback")
    environment, folders = cache_environment(profile, root, python)
    command = [executable, "--die-with-parent", "--unshare-pid", "--ro-bind", "/", "/",
        "--bind", str(root), str(root), "--bind", str(_child(root, "tmp")), "/tmp",
        "--bind", str(_child(root, "tmp")), "/var/tmp", "--dev-bind", "/dev", "/dev",
        "--tmpfs", "/dev/shm", "--proc", "/proc", "--chdir", str(_child(root, "work")), "--"]
    command += model_command(profile, model, python, len(selected))
    return dict(schema="ep-navsim-owned-model-launch-1", profile=profile,
        runtime_root=str(root), public_root=str(public), python=str(python),
        python_target=str(python.resolve()), command=command, environment=environment,
        directories=folders, gpus=selected, filesystem="read_only_root_private_runtime_and_shm")


def _bind_gpus(plan, inventory):
    # nvidia-smi indices need not match CUDA's PCI ordering on every host.
    # UUIDs bind exactly the cards whose occupancy we checked, including when
    # another experiment uses an adjacent index.
    plan["gpu_inventory"] = inventory
    plan["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(inventory[str(i)]["uuid"] for i in plan["gpus"])


def _toolchain(plan, env):
    """Expose read-only pip CUDA with link names created only in our runtime."""
    venv = Path(plan["python"]).parent.parent
    for cuda in sorted(venv.glob("lib/python*/site-packages/nvidia/cu13")):
        if not (cuda / "bin/nvcc").is_file() or not (cuda / "include/cuda.h").is_file():
            continue
        root = Path(plan["runtime_root"])
        links = _child(root, "toolchain")
        targets = {"libcudart.so": next(iter(sorted((cuda / "lib").glob("libcudart.so.*"))), None)}
        driver = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
        if driver.is_file():
            targets["libcuda.so"] = driver
        for name, target in targets.items():
            if target is None:
                continue
            link = links / name
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise ValueError("Private toolchain link already differs; preserved: " + str(link))
            link.symlink_to(target.resolve())
        paths = [str(links), str(cuda / "lib")]
        env.update(CUDA_HOME=str(cuda), CUDA_PATH=str(cuda),
            PATH=str(cuda / "bin")+os.pathsep+env["PATH"],
            LD_LIBRARY_PATH=os.pathsep.join(paths), LIBRARY_PATH=os.pathsep.join(paths),
            FLASHINFER_EXTRA_LDFLAGS=" ".join("-L"+shlex.quote(path) for path in paths))
        break


def start(plan):
    if not sys.platform.startswith("linux"):
        raise ValueError("Owned model launch requires Linux bubblewrap")
    profile, selected = plan["profile"], plan["gpus"]
    _bind_gpus(plan, check_gpus(profile, selected))
    check_port(profile)
    root = private_root(plan["runtime_root"], plan["public_root"])
    for relative in plan["directories"]:
        _child(root, relative).mkdir(parents=True, exist_ok=True)
    # Do not inherit another conda environment's Python paths, CUDA libraries,
    # compiler overrides or code-generation destinations into the owned venv.
    keep = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TZ")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    generated = dict(plan["environment"])
    _toolchain(plan, generated)
    env.update(generated)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    record = _child(root, "logs") / ("launch_"+stamp+".json")
    # This is the allowlisted generated environment, never credentials or a
    # copy of os.environ. It binds the exact owned process command and inputs.
    actual_plan = dict(plan, environment=generated)
    record.write_text(json.dumps(actual_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    log = record.with_suffix(".log")
    print("Owned launch record:", record, "\nOwned service log:", log, flush=True)
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(plan["command"], cwd=_child(root, "work"), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            return child.wait()
        except KeyboardInterrupt:
            # The foreground terminal signals this process group. Bubblewrap's
            # die-with-parent handles only our namespace if the launcher exits.
            return 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-profile", choices=SERVICE_PROFILES, default="shared_qwen3_32b")
    parser.add_argument("--runtime-root", required=True, help="Absolute private, user-owned runtime directory")
    parser.add_argument("--gpus", type=gpu_ids, required=True, help="Explicit currently idle GPU IDs, e.g. 0,2")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--start", action="store_true", help="Start our isolated foreground process; never switch or stop others")
    action.add_argument("--dry-run", action="store_true", help="Only inspect resources and print the command (default)")
    args = parser.parse_args()
    try:
        plan = prepare_plan(load_service_profile(args.service_profile), args.runtime_root, args.gpus)
        _bind_gpus(plan, check_gpus(plan["profile"], args.gpus))
        check_port(plan["profile"])
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return start(plan) if args.start else 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, "Owned model launch refused: "+str(exc)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())
