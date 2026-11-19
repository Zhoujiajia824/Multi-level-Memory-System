"""按正式场景配置采集原始highway素材；直接用于离线工作台prepare。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="实验场景原始素材采集：无VLM、检索、DINO或参照评分")
    parser.add_argument("--scenario", choices=["highway", "merge", "roundabout", "intersection"], required=True)
    parser.add_argument("--config", default="config/parameters.json")
    parser.add_argument("--scenarios-config", default="config/scenarios.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000, help="首轮seed，之后逐轮加1")
    parser.add_argument("--duration", type=float, default=30.)
    parser.add_argument("--prefix", default=datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("轮数须为正，seed不得为负")
    if not args.prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.prefix):
        parser.error("prefix仅允许字母、数字、下划线和连字符")

    import yaml
    from nine_grid.configuration import validate_config
    from nine_grid.highway_capture import capture_episode, _json
    from nine_grid.paths import filesystem_path

    def path(value):
        item = Path(value)
        return filesystem_path(item if item.is_absolute() else ROOT / item)

    parameter_path, scenario_path = path(args.config), path(args.scenarios_config)
    config = json.loads(parameter_path.read_text(encoding="utf-8-sig"))
    scenarios = yaml.safe_load(scenario_path.read_text(encoding="utf-8-sig"))
    if not isinstance(scenarios, dict) or args.scenario not in scenarios:
        parser.error("场景YAML缺少指定场景")
    for name, settings in scenarios.items():
        config.setdefault("scenarios", {}).setdefault(name, {}).update(settings)
    config.update(platform="highway", scenario=args.scenario, duration=args.duration)
    config.setdefault("diagnostics", {}).update(enabled=False, dino=False)
    config.setdefault("memory", {}).update(read_enabled=False, write_enabled=False)
    config.setdefault("models", {}).update(fast_backend="off", reflection_backend="off")
    config = validate_config(config)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    provenance = {
        "git_revision": git.stdout.strip() if git.returncode == 0 else None,
        "input_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (parameter_path, scenario_path)},
        "source_files_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((ROOT / "src/nine_grid").glob("*.py"))},
        "capture_only_overrides": ["关闭模型、记忆、表示和质量诊断；不改交通、动力学和主规划参数"],
    }
    output = path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = output / f"{args.prefix}_batch.json"
    if report.exists():
        parser.error("该批次已存在，请使用新prefix，不覆盖已有采集")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    results = []
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            directory = output / f"{args.prefix}_{args.scenario}_seed{seed}"
            run = capture_episode(config, seed, directory, provenance=provenance, live=args.live)
            results.append({key: run[key] for key in (
                "run_path", "scenario", "seed", "complete", "frame_count", "planning_steps",
                "simulated_seconds", "collision", "offroad", "termination_reason", "wall_seconds")})
            _json(report, results)
    finally:
        if results:
            _json(report, results)
    print(json.dumps({"批量报告": str(report), "完成轮数": len(results),
                      "离线工具输入": str(output), "模型调用": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
