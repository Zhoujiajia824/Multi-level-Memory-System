"""CARLA 闭环入口
================

用法（先手动启动 CARLA 服务器 CarlaUE4.exe）::

    conda activate mulmem_carla
    python -m carla_bridge.run_carla_demo --scenario straight_traffic --mode memory_on

加载 ``carla_bridge/config/carla.yaml`` 作为 overrides 深合并进主配置（不改 src/
与现有 config/），再跑 :class:`ClosedLoop`。
"""
from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _configure_runtime_diagnostics() -> None:
    """Keep native crash traces and timely redirected logs without Python flags."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True, write_through=True)
            except (OSError, ValueError):
                pass  # Notebook/test capture streams may have no usable descriptor.
    if not faulthandler.is_enabled():
        for stream in (sys.stderr, sys.__stderr__):
            try:
                faulthandler.enable(file=stream, all_threads=True)
                break
            except (AttributeError, OSError, ValueError):
                continue


# Must precede CARLA / PyTorch / FAISS native imports, not just loop.setup().
_configure_runtime_diagnostics()

import yaml

from src.vla_memory.common.config import load_config
from src.vla_memory.common.logging_utils import setup_logger, set_log_directory
from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.run_session import RunSession
from carla_bridge.inference_mode import model_inference_enabled


def _resolve_scenario_path(scenario_arg: str, config) -> Path:
    scenario_dir = Path(
        config.get_nested("carla", "scenario_dir", default="carla_bridge/config/scenarios")
    )
    if not scenario_dir.is_absolute():
        scenario_dir = PROJECT_ROOT / scenario_dir

    # 先按"场景名"找 <dir>/<name>.yaml
    p = scenario_dir / f"{scenario_arg}.yaml"
    if p.exists():
        return p
    # 再当完整路径
    p = Path(scenario_arg)
    if p.exists():
        return p
    raise FileNotFoundError(f"场景 YAML 不存在: {scenario_arg}（搜索目录 {scenario_dir}）")


def main() -> None:
    ap = argparse.ArgumentParser(description="CARLA 闭环：多层次记忆系统 + CARLA 实时回控")
    ap.add_argument("--scenario", default=None, help="场景 YAML 名（不带扩展）或绝对路径")
    ap.add_argument(
        "--mode", choices=["memory_on", "memory_off"], default="memory_on",
        help="memory_on 启用记忆访问（模型记忆或无模型full数值记忆），memory_off禁止所有记忆读写",
    )
    ap.add_argument("--memory-bank", help="持久化记忆库路径；省略时为本轮独立库")
    ap.add_argument("--memory-access", choices=["read_write", "read_only"], default="read_write")
    ap.add_argument("--memory-seed", help="复制已有库到新目标库，原始库保持不变")
    ap.add_argument("--no-memory-retrieval", action="store_true", help="不检索持久文本/数值经验，仍按写入开关积累")
    ap.add_argument("--no-short-term", action="store_true", help="关闭本轮短期记忆")
    ap.add_argument("--planner", choices=["pdm_closed", "vlm_trajectory"], default=None,
                    help="轨迹后端，默认 PDM-Closed；vlm_trajectory 为旧模型直接输出轨迹")
    ap.add_argument("--semantic-influence", choices=["on", "off"], default=None,
                    help="PDM 候选评分是否接受模型横纵向语义偏好；与记忆开关独立")
    ap.add_argument("--semantic-policy", choices=["selective", "always", "shadow"], default=None,
                    help="语义策略：selective 条件介入；always 旧版加分；shadow 仅记录建议，执行基础PDM")
    ap.add_argument("--planning-memory", choices=["off", "gate", "full"], default=None,
                    help="数值规划经验：off 关闭；gate 仅辅助介入判断；full 还提供已执行参数候选先验，受mode/access限制")
    ap.add_argument("--model-inference", choices=["on", "off"], default=None,
                    help="模型调用开关；off 要求 pdm_closed、语义off，以及memory_off或显式planning-memory full的数值记忆模式")
    ap.add_argument("--dinov2-model-dir", help="本地 DINOv2 模型目录或缓存 snapshot 目录")
    ap.add_argument("--dinov2-cache-dir", help="Hugging Face 缓存目录（默认项目 .cache/huggingface）")
    ap.add_argument("--allow-model-download", action="store_true", help="显式允许 DINOv2 联网下载；默认仅使用本地文件")
    args = ap.parse_args()

    # 修复 Anaconda/conda 常见坑：SSL_CERT_FILE 指向不存在的 cacert.pem 会让
    # httpx(openai) 构造 SSL context 时 FileNotFoundError。指向坏路径就清除，改用 certifi 默认证书。
    import os as _os
    _ssl = _os.environ.get("SSL_CERT_FILE", "")
    if _ssl and not _os.path.exists(_ssl):
        print(f"[carla_bridge] SSL_CERT_FILE 指向不存在的路径，已清除（改用 certifi 默认证书）: {_ssl}")
        _os.environ.pop("SSL_CERT_FILE", None)

    # 加载 carla.yaml 作为 overrides 深合并
    carla_yaml = PROJECT_ROOT / "carla_bridge" / "config" / "carla.yaml"
    with open(carla_yaml, "r", encoding="utf-8") as f:
        carla_cfg = yaml.safe_load(f)
    planning = carla_cfg.setdefault("carla", {}).setdefault("planning", {})
    if args.planner is not None:
        planning["backend"] = args.planner
    if args.semantic_policy is not None:
        planning["semantic_policy"] = args.semantic_policy
    if args.planning_memory is not None:
        planning["planning_memory"] = args.planning_memory
    if args.model_inference is not None:
        planning["model_inference"] = args.model_inference == "on"
    if args.semantic_influence is not None:
        if planning.get("backend", "vlm_trajectory") != "pdm_closed":
            ap.error("--semantic-influence requires --planner pdm_closed")
        planning["semantic_influence"] = args.semantic_influence == "on"
    if planning.get("backend") == "vlm_trajectory":
        planning["semantic_influence"] = False
    feature_cfg = carla_cfg.setdefault("feature_extractor", {})
    feature_cfg.setdefault("local_files_only", True)
    if args.dinov2_model_dir:
        feature_cfg["model_dir"] = args.dinov2_model_dir
    if args.dinov2_cache_dir:
        feature_cfg["cache_dir"] = args.dinov2_cache_dir
    if args.allow_model_download:
        feature_cfg["local_files_only"] = False
    config = load_config(overrides=carla_cfg)
    try:
        model_inference = model_inference_enabled(
            config.get_nested("carla", "planning", default={}) or {}, args.mode)
    except ValueError as exc:
        ap.error(str(exc))

    # fail-fast：VLM API Key 缺失立即报错（避免跑完 CARLA+DINOv2 setup 才在第一次 VLM 调用崩）
    import os as _os
    _envs = set()
    for _blk in ("scene_understanding", "decision"):
        _e = config.get_nested(_blk, "api_key_env", default=None)
        if _e:
            _envs.add(_e)
    if not _envs:
        _envs = {"DASHSCOPE_API_KEY"}
    _missing = [e for e in _envs if not _os.environ.get(e, "")] if model_inference else []
    if _missing:
        _e0 = _missing[0]
        raise EnvironmentError(
            f"VLM API Key 未设置：环境变量 {_missing} 为空。\n"
            f"  Linux/macOS Bash 或 Git Bash：使用 export\n"
            f"    export {_e0}=你的key\n"
            f"  Windows CMD:  set {_e0}=你的key\n"
            f"  PowerShell:   $env:{_e0}='你的key'\n"
            f"设好后与运行命令在同一个终端执行。"
        )

    scenario_name = args.scenario or config.get_nested(
        "carla", "default_scenario", default="straight_traffic"
    )
    scenario_path = _resolve_scenario_path(scenario_name, config)
    with scenario_path.open(encoding="utf-8") as scenario_file:
        scenario_config = yaml.safe_load(scenario_file)
    if (scenario_config.get("experiment") or {}).get("suite") == "paper_v1":
        from carla_bridge.paper_suite import validate_prepared
        validate_prepared(scenario_config)
        if (scenario_config["experiment"]["split"] == "eval" and args.mode == "memory_on"
                and args.memory_access != "read_only"):
            raise ValueError("paper eval requires --memory-access read_only with a training bank; "
                             "use a train case for memory accumulation, or --mode memory_off for baseline")
        # The prepared suite owns its declared time budget; legacy scenes keep
        # the existing global cap. No general YAML-to-config injection.
        config.data["carla"]["max_duration_s"] = float(scenario_config["duration_s"])
        config.data["carla"]["timeout_s"] = 120.0

    session = RunSession(config, scenario_path, args.mode, memory_bank=args.memory_bank,
                         memory_access=args.memory_access, memory_seed=args.memory_seed,
                         short_term=not args.no_short_term, retrieve=not args.no_memory_retrieval)
    loop = None
    try:
        session.prepare()
        print(f"[carla_bridge] 本轮输出目录: {session.run_dir}", flush=True)
        if (scenario_config.get("experiment") or {}).get("suite") == "paper_v1":
            config.data["carla"]["experiment_output_dir"] = str(session.run_dir)
        set_log_directory(config.get_path("log_dir"))
        setup_logger(name="carla_demo", level=config.get_nested("logging", "level", default="INFO"),
                     log_dir=config.get_path("log_dir"))
        loop = ClosedLoop(config=config, scenario_yaml=str(scenario_path), mode=args.mode)
        session.set_status("initializing")
        loop.setup()
        session.set_status("running")
        loop.run()
    except BaseException as exc:
        if loop is not None:
            loop.run_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        if session.run_dir.exists():
            session.set_status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exc)
        raise
    finally:
        try:
            if loop is not None:
                loop.close()
                result = loop.outcome()
                session.set_status(loop.run_status, outcome=result if isinstance(result, dict) else None)
        finally:
            session.close()
    if loop is not None and loop.run_status != "route_completed":
        raise RuntimeError(f"运行未成功结束: {loop.run_status}，详见 {session.run_dir}")


if __name__ == "__main__":
    main()
