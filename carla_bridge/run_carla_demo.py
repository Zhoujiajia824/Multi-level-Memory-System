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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.vla_memory.common.config import load_config
from src.vla_memory.common.logging_utils import setup_logger, set_log_directory
from carla_bridge.closed_loop import ClosedLoop
from carla_bridge.run_session import RunSession


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
        help="memory_on 用三层记忆，memory_off 对照基线",
    )
    ap.add_argument("--memory-bank", help="持久化记忆库路径；省略时为本轮独立库")
    ap.add_argument("--memory-access", choices=["read_write", "read_only"], default="read_write")
    ap.add_argument("--memory-seed", help="复制已有库到新目标库，原始库保持不变")
    ap.add_argument("--no-memory-retrieval", action="store_true", help="不检索中长期记忆，仍按写入开关积累")
    ap.add_argument("--no-short-term", action="store_true", help="关闭本轮短期记忆")
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
    config = load_config(overrides=carla_cfg)

    # fail-fast：VLM API Key 缺失立即报错（避免跑完 CARLA+DINOv2 setup 才在第一次 VLM 调用崩）
    import os as _os
    _envs = set()
    for _blk in ("scene_understanding", "decision"):
        _e = config.get_nested(_blk, "api_key_env", default=None)
        if _e:
            _envs.add(_e)
    if not _envs:
        _envs = {"DASHSCOPE_API_KEY"}
    _missing = [e for e in _envs if not _os.environ.get(e, "")]
    if _missing:
        _e0 = _missing[0]
        raise EnvironmentError(
            f"VLM API Key 未设置：环境变量 {_missing} 为空。\n"
            f"  你在用 Git Bash(MINGW64)：必须用 export，不是 set！\n"
            f"    export {_e0}=你的key\n"
            f"  Windows CMD:  set {_e0}=你的key\n"
            f"  PowerShell:   $env:{_e0}='你的key'\n"
            f"设好后与运行命令在同一个终端执行。"
        )

    scenario_name = args.scenario or config.get_nested(
        "carla", "default_scenario", default="straight_traffic"
    )
    scenario_path = _resolve_scenario_path(scenario_name, config)

    session = RunSession(config, scenario_path, args.mode, memory_bank=args.memory_bank,
                         memory_access=args.memory_access, memory_seed=args.memory_seed,
                         short_term=not args.no_short_term, retrieve=not args.no_memory_retrieval)
    loop = None
    try:
        session.prepare()
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
                session.set_status(loop.run_status)
        finally:
            session.close()
    if loop is not None and loop.run_status in ("memory_save_failed", "sensor_failure"):
        raise RuntimeError(f"运行未成功结束: {loop.run_status}，详见 {session.run_dir}")


if __name__ == "__main__":
    main()
