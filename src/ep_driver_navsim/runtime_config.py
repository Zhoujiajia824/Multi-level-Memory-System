"""Prepare separate local/server evaluation groups without touching environments."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml

from .protocol import write_json
from .progress import progress_interval
from .profiles import get_profile, normalize_profile, DEFAULT_PROFILE, PROFILES
from .service_profiles import load_service_profile, SERVICE_PROFILES


def generate(base, destination, profile, revision, base_url=None, model=None, split=None,
             overwrite=False, *, groups=None, run_name=None, max_tokens=None,
             max_model_len=None, model_goal_retry_limit=1, request_timeout_s=120.,
             model_deadline_s=300., transport_retry_limit=1, model_concurrency=None,
             evaluation_workers=None, vllm_version=None, structured_backend="xgrammar",
             allow_any_whitespace=False, progress_interval_s=15., benchmark=DEFAULT_PROFILE,
             sdk_root=None, sdk_python=None, service_profile=None):
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("An explicit model weights/config revision is required")
    service = load_service_profile(service_profile) if service_profile else None
    def service_value(key, value, legacy):
        expected = service[key] if service else legacy
        if service and value is not None and value != expected:
            raise ValueError("Argument differs from the owned service profile: " + key)
        return expected if value is None else value
    endpoint_default = ("http://%s:%s/v1" % (service["host"], service["port"]) if service
                        else "http://127.0.0.1:8000/v1")
    if service and base_url is not None and base_url != endpoint_default:
        raise ValueError("base_url differs from the owned service profile")
    base_url = base_url if base_url is not None else endpoint_default
    model = service_value("served_model_name", model, "Qwen2.5-VL-3B-Instruct")
    max_model_len = service_value("max_model_len", max_model_len, 16384)
    model_concurrency = service_value("max_num_seqs", model_concurrency, 1)
    vllm_version = service_value("vllm_version", vllm_version, None)
    max_tokens = max_tokens if max_tokens is not None else (service["max_new_tokens"] if service else 4096)
    if service:
        if not run_name:
            raise ValueError("An owned model service profile requires a separate run_name")
        if (structured_backend != service["structured_outputs"]["backend"]
                or allow_any_whitespace):
            raise ValueError("Structured output policy differs from the owned service profile")
    protocol = get_profile(benchmark)
    split = split or protocol["splits"][0]
    if split not in protocol["splits"]:
        raise ValueError("Split is incompatible with benchmark " + protocol["id"])
    groups = tuple(groups) if groups is not None else ("A_base", "B_model", "C_model_memory")
    allowed = {"A_base", "B_model", "C_model_memory"}
    if not groups or len(set(groups)) != len(groups) or not set(groups).issubset(allowed):
        raise ValueError("groups must select distinct A_base/B_model/C_model_memory conditions")
    if run_name is not None and (not isinstance(run_name, str) or
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_name) is None):
        raise ValueError("run_name must be one safe directory label")
    for name, value in (("max_tokens", max_tokens), ("max_model_len", max_model_len)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    if service and max_tokens > service["max_new_tokens"]:
        raise ValueError("max_tokens exceeds the owned model service output cap")
    if max_tokens >= max_model_len:
        raise ValueError("max_tokens must leave context capacity for model inputs")
    if type(model_goal_retry_limit) is not int or model_goal_retry_limit not in (0, 1):
        raise ValueError("model_goal_retry_limit must be zero or one")
    from .model_transport import model_request_policy
    request_policy = model_request_policy(dict(model_request_policy=dict(
        request_timeout_s=request_timeout_s, total_deadline_s=model_deadline_s,
        transport_retry_limit=transport_retry_limit)))
    if request_policy["total_deadline_s"]+60 > 3600:
        raise ValueError("Model requests must fit the shared model deadline and leave time within the core timeout")
    if model_concurrency is None:
        model_concurrency = 1
    if type(model_concurrency) is not int or not 1 <= model_concurrency <= 8:
        raise ValueError("model_concurrency must be an integer between one and eight")
    if evaluation_workers is None:
        evaluation_workers = 1 if service else (4 if profile == "server" else 1)
    if type(evaluation_workers) is not int or not 1 <= evaluation_workers <= 8:
        raise ValueError("evaluation_workers must be an integer between one and eight")
    if structured_backend not in ("auto", "xgrammar") or type(allow_any_whitespace) is not bool:
        raise ValueError("Declare the deployed structured backend and whitespace policy")
    if vllm_version is not None and (not isinstance(vllm_version, str) or not vllm_version.strip()):
        raise ValueError("vllm_version must be the actual served version, or omitted")
    progress_interval_s = progress_interval(progress_interval_s)
    endpoint = urlsplit(base_url)
    if (endpoint.scheme not in ("http", "https") or not endpoint.hostname or
            endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("base_url must be an HTTP endpoint without embedded credentials")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("An explicit served model name is required")
    destination = Path(destination).resolve()
    template = json.loads(Path(base).read_text(encoding="utf-8"))
    suffix_parts = [profile]
    if benchmark != DEFAULT_PROFILE:
        suffix_parts.append(benchmark)
    if service:
        suffix_parts.append(service_profile)
    if run_name:
        suffix_parts.append(run_name)
    suffix = ".".join(suffix_parts)
    prefix = "navsim." + suffix
    model_path = destination / ("models.navsim." + suffix + ".local.yaml")
    files = [model_path] + [destination / (prefix + "." + name + ".local.json")
                            for name in groups]
    if not overwrite and any(path.exists() for path in files):
        raise FileExistsError("Runtime configs already exist; use a new directory or --overwrite")
    if profile == "server":
        root = "/data/zjj/code/EP-Driver"
        template.update(sdk_root=root + "/navsim-devkit",
            sdk_python="/home/zjj/miniconda3/envs/ep-navsim/bin/python",
            core_python="/home/zjj/miniconda3/envs/ep-driver-navsim/bin/python",
            data_root="/data/zjj/datasets/navsim/dataset",
            dino_model=root + "/EP-Driver/models/dinov2-base",
            maps_runtime_root=root + "/experiments/navsim/maps_runtime",
            metric_cache_path=root + "/experiments/navsim/cache/" + split)
        output = root + "/experiments/navsim/" + split
        if not protocol["two_stage"]:
            version = "v1" if protocol["is_v1"] else "v2"
            if protocol["is_v1"]:
                template.update(sdk_root=root + "/navsim-devkit-v1.1",
                    sdk_python="/home/zjj/miniconda3/envs/ep-navsim-v1/bin/python")
            template["metric_cache_path"] = root + "/experiments/navsim/cache/" + version + "/navtest"
            output = root + "/experiments/navsim/" + version + "/navtest"
        memory = root + "/driving_experience"
        # The generated files will be copied/published under the branch's config directory.
        model_reference = "config/" + model_path.name
    elif profile == "local":
        output = "test/artifacts/navsim/local/" + split
        if not protocol["two_stage"]:
            version = "v1" if protocol["is_v1"] else "v2"
            output = "test/artifacts/navsim/local/" + version + "/navtest"
            template["metric_cache_path"] = "test/artifacts/navsim/cache/" + version + "/navtest"
            if protocol["is_v1"] and not sdk_root:
                raise ValueError("Local v1 evaluation requires --sdk-root pointing to pinned official v1.1 source")
        memory = "E:/driving_memory_workspace/libraries/driving_experience"
        model_reference = str(model_path)
    else:
        raise ValueError("profile must be local or server")
    if run_name:
        output += "/" + run_name
    if sdk_root:
        template["sdk_root"] = str(sdk_root)
    if sdk_python:
        template["sdk_python"] = str(sdk_python)
    settings = dict(template, split=split, resume=True, model_revision=revision,
        timeout_s=3600.0, bev_video=True,
        navigation_preflight_workers=8 if profile == "server" else 1,
        evaluation_workers=evaluation_workers, progress_interval_s=progress_interval_s,
        model_request_policy=request_policy,
        model_service=dict(image_limit=service["image_limit"] if service else 9,
            max_pixels_per_image=service["max_pixels_per_image"] if service else 602112, max_model_len=max_model_len,
            max_num_seqs=model_concurrency, vllm_version=vllm_version,
            structured_outputs=dict(backend=structured_backend, disable_any_whitespace=not allow_any_whitespace)),
        models_config=model_reference, benchmark_profile=benchmark)
    settings = normalize_profile(settings)
    if service:
        # A template's manual escape hatch must never weaken our owned launch
        # contract; this profile has explicit parameters for strict attestation.
        settings.update(model_service_validation_mode="strict", model_service_profile=service_profile,
            model_service_profile_sha256=hashlib.sha256(json.dumps(service,
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())
    destination.mkdir(parents=True, exist_ok=True)
    models = dict(fast=dict(backend="server", base_url=base_url, model=model,
                           api_key_env="FAST_VLM_API_KEY", timeout_s=float(request_timeout_s), max_tokens=max_tokens),
                  reflection=dict(backend="off"))
    model_path.write_text(yaml.safe_dump(models, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for path, name in zip(files[1:], groups):
        enabled, retrieve = name != "A_base", name == "C_model_memory"
        write_json(path, dict(settings, fast="server" if enabled else "off", retrieve=retrieve,
                              structured_model_output=enabled,
                              model_goal_retry_limit=model_goal_retry_limit if enabled else 0,
                              memory_read_dir=memory if retrieve else None, output_dir=output + "/" + name))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config/navsim.example.json")
    parser.add_argument("--destination", help="Defaults to test/config for local bringup, config for server")
    parser.add_argument("--profile", choices=("local", "server"), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-url", help="Defaults to the selected owned service, otherwise localhost:8000/v1")
    parser.add_argument("--model", help="Defaults to the selected owned service, otherwise Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--service-profile", choices=SERVICE_PROFILES,
                        help="Use our private serving profile; never starts or modifies the shared deployment")
    parser.add_argument("--benchmark", choices=tuple(PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--split", choices=("navtest", "warmup_two_stage", "navhard_two_stage"))
    parser.add_argument("--sdk-root", help="Explicit pinned SDK directory; v1 and v2 must be separate")
    parser.add_argument("--sdk-python", help="Python 3.9 interpreter for the selected SDK")
    parser.add_argument("--groups", nargs="+", choices=("A_base", "B_model", "C_model_memory"),
                        help="Select A_base B_model for the paired no-retrieval comparison")
    parser.add_argument("--run-name", help="New shared run label; keeps old results and configs separate")
    parser.add_argument("--max-tokens", type=int, help="Owned profile output budget, otherwise 4096")
    parser.add_argument("--max-model-len", type=int,
                        help="Actual deployed context limit; must be declared, not inferred from a model name")
    parser.add_argument("--model-goal-retry-limit", type=int, choices=(0, 1), default=1)
    parser.add_argument("--request-timeout-s", type=float, default=120.)
    parser.add_argument("--model-deadline-s", type=float, default=300.,
                        help="Shared model-phase deadline, including transport and semantic retries")
    parser.add_argument("--transport-retry-limit", type=int, choices=(0, 1), default=1)
    parser.add_argument("--model-concurrency", type=int,
                        help="Actual deployed max-num-seqs; defaults to one")
    parser.add_argument("--evaluation-workers", type=int,
                        help="Defaults to four on server, one locally; A/B use the same setting")
    parser.add_argument("--vllm-version", help="Actual served vLLM version, recorded in model identity")
    parser.add_argument("--structured-backend", choices=("auto", "xgrammar"), default="xgrammar")
    parser.add_argument("--allow-any-whitespace", action="store_true",
                        help="Declare an existing service with unrestricted whitespace; does not fix the service")
    parser.add_argument("--progress-interval-s", type=float, default=15., help="0 disables the progress monitor")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_model_len is None and not args.service_profile:
        parser.error("--max-model-len is required without an explicit owned --service-profile")
    destination = args.destination or ("test/config" if args.profile == "local" else "config")
    for path in generate(args.base_config, destination, args.profile, args.model_revision,
                         args.base_url, args.model, args.split, args.overwrite,
                         groups=args.groups, run_name=args.run_name, max_tokens=args.max_tokens,
                         max_model_len=args.max_model_len, model_goal_retry_limit=args.model_goal_retry_limit,
                         request_timeout_s=args.request_timeout_s, model_deadline_s=args.model_deadline_s,
                         transport_retry_limit=args.transport_retry_limit, model_concurrency=args.model_concurrency,
                         evaluation_workers=args.evaluation_workers, vllm_version=args.vllm_version,
                         structured_backend=args.structured_backend, allow_any_whitespace=args.allow_any_whitespace,
                         progress_interval_s=args.progress_interval_s, benchmark=args.benchmark,
                         sdk_root=args.sdk_root, sdk_python=args.sdk_python,
                         service_profile=args.service_profile):
        print(path)


if __name__ == "__main__":
    main()
