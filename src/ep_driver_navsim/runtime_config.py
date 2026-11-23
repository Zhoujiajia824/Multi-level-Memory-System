"""Prepare separate local/server evaluation groups without touching environments."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml

from .protocol import write_json


def generate(base, destination, profile, revision, base_url, model, split="warmup_two_stage",
             overwrite=False, *, groups=None, run_name=None, max_tokens=4096,
             max_model_len=16384, model_goal_retry_limit=1):
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
        evaluation_workers=4 if profile == "server" else 1,
        model_service=dict(image_limit=9, max_pixels_per_image=602112, max_model_len=max_model_len),
        models_config=model_reference)
    destination.mkdir(parents=True, exist_ok=True)
    models = dict(fast=dict(backend="server", base_url=base_url, model=model,
                           api_key_env="FAST_VLM_API_KEY", timeout_s=180, max_tokens=max_tokens),
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.destination or ("test/config" if args.profile == "local" else "config")
    for path in generate(args.base_config, destination, args.profile, args.model_revision,
                         args.base_url, args.model, args.split, args.overwrite,
                         groups=args.groups, run_name=args.run_name, max_tokens=args.max_tokens,
                         max_model_len=args.max_model_len, model_goal_retry_limit=args.model_goal_retry_limit):
        print(path)


if __name__ == "__main__":
    main()
