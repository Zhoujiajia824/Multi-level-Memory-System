"""Prepare separate local/server evaluation groups without touching environments."""
import argparse
import json
from pathlib import Path

import yaml

from .protocol import write_json


def generate(base, destination, profile, revision, base_url, model, split="warmup_two_stage",
             overwrite=False):
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("An explicit model weights/config revision is required")
    destination = Path(destination).resolve()
    template = json.loads(Path(base).read_text(encoding="utf-8"))
    prefix = "navsim." + profile
    model_path = destination / ("models.navsim." + profile + ".local.yaml")
    files = [model_path] + [destination / (prefix + "." + name + ".local.json")
                            for name in ("A_base", "B_model", "C_model_memory")]
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
        output = "outputs/navsim/bringup/" + split
        memory = "E:/driving_memory_workspace/libraries/driving_experience"
        model_reference = str(model_path)
    else:
        raise ValueError("profile must be local or server")
    settings = dict(template, split=split, resume=True, model_revision=revision,
        timeout_s=3600.0, bev_video=True,
        navigation_preflight_workers=8 if profile == "server" else 1,
        model_service=dict(image_limit=9, max_pixels_per_image=602112, max_model_len=16384),
        models_config=model_reference)
    destination.mkdir(parents=True, exist_ok=True)
    models = dict(fast=dict(backend="server", base_url=base_url, model=model,
                           api_key_env="FAST_VLM_API_KEY", timeout_s=180, max_tokens=1024),
                  reflection=dict(backend="off"))
    model_path.write_text(yaml.safe_dump(models, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for path, name, enabled, retrieve in zip(files[1:], ("A_base", "B_model", "C_model_memory"),
                                           (False, True, True), (False, False, True)):
        write_json(path, dict(settings, fast="server" if enabled else "off", retrieve=retrieve,
                              structured_model_output=enabled,
                              memory_read_dir=memory if retrieve else None, output_dir=output + "/" + name))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config/navsim.local.json")
    parser.add_argument("--destination", default="config")
    parser.add_argument("--profile", choices=("local", "server"), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--split", choices=("warmup_two_stage", "navhard_two_stage"), default="warmup_two_stage")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in generate(args.base_config, args.destination, args.profile, args.model_revision,
                         args.base_url, args.model, args.split, args.overwrite):
        print(path)


if __name__ == "__main__":
    main()
