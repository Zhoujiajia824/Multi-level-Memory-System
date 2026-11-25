"""Prepare separate local/server evaluation groups without touching environments."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml

from .protocol import write_json
from .progress import progress_interval


def generate(base, destination, profile, revision, base_url, model, split="warmup_two_stage",
             overwrite=False, *, groups=None, run_name=None, max_tokens=4096,
             max_model_len=16384, model_goal_retry_limit=1, request_timeout_s=120.,
             model_deadline_s=300., transport_retry_limit=1, model_concurrency=None,
             evaluation_workers=None, vllm_version=None, structured_backend="xgrammar",
             allow_any_whitespace=False, progress_interval_s=15.):
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("An explicit model weights/config revision is required")
    if split not in ("warmup_two_stage", "navhard_two_stage"):
        raise ValueError("Unknown pinned official split")
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
        evaluation_workers = 4 if profile == "server" else 1
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
    suffix = profile + ("." + run_name if run_name else "")
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
        memory = root + "/driving_experience"
        # The generated files will be copied/published under the branch's config directory.
        model_reference = "config/" + model_path.name
    elif profile == "local":
        output = "test/artifacts/navsim/local/" + split
        memory = "E:/driving_memory_workspace/libraries/driving_experience"
        model_reference = str(model_path)
    else:
        raise ValueError("profile must be local or server")
    if run_name:
        output += "/" + run_name
    settings = dict(template, split=split, resume=True, model_revision=revision,
        timeout_s=3600.0, bev_video=True,
        navigation_preflight_workers=8 if profile == "server" else 1,
        evaluation_workers=evaluation_workers, progress_interval_s=progress_interval_s,
        model_request_policy=request_policy,
        model_service=dict(image_limit=9, max_pixels_per_image=602112, max_model_len=max_model_len,
            max_num_seqs=model_concurrency, vllm_version=vllm_version,
            structured_outputs=dict(backend=structured_backend, disable_any_whitespace=not allow_any_whitespace)),
        models_config=model_reference)
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
    parser.add_argument("--base-config", default="config/navsim.local.json")
    parser.add_argument("--destination", help="Defaults to test/config for local bringup, config for server")
    parser.add_argument("--profile", choices=("local", "server"), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--split", choices=("warmup_two_stage", "navhard_two_stage"), default="warmup_two_stage")
    parser.add_argument("--groups", nargs="+", choices=("A_base", "B_model", "C_model_memory"),
                        help="Select A_base B_model for the paired no-retrieval comparison")
    parser.add_argument("--run-name", help="New shared run label; keeps old results and configs separate")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, required=True,
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
    destination = args.destination or ("test/config" if args.profile == "local" else "config")
    for path in generate(args.base_config, destination, args.profile, args.model_revision,
                         args.base_url, args.model, args.split, args.overwrite,
                         groups=args.groups, run_name=args.run_name, max_tokens=args.max_tokens,
                         max_model_len=args.max_model_len, model_goal_retry_limit=args.model_goal_retry_limit,
                         request_timeout_s=args.request_timeout_s, model_deadline_s=args.model_deadline_s,
                         transport_retry_limit=args.transport_retry_limit, model_concurrency=args.model_concurrency,
                         evaluation_workers=args.evaluation_workers, vllm_version=args.vllm_version,
                         structured_backend=args.structured_backend, allow_any_whitespace=args.allow_any_whitespace,
                         progress_interval_s=args.progress_interval_s):
        print(path)


if __name__ == "__main__":
    main()
