#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整 Demo 运行脚本（在线循环版 R2）
===================================
逐帧在线循环：感知 → 检索 → 决策 → 更新记忆，单次运行只跑一种 mode。

核心链路：
  nuScenes 数据加载 -> 关键帧提取 -> 状态/轨迹/导航补充 ->
  for each keyframe:
      DINOv2 特征 + VLM 场景理解 -> 三层记忆检索 ->
      决策 VLM(图像+历史+记忆) -> parse -> 更新短期/中期记忆 -> append jsonl

前置条件：
  - nuScenes v1.0-mini 数据集已放置到 data/nuscenes/raw/
  - DINOv2 模型权重已下载（运行 00_prepare_models.py）
  - VLM API Key 已设置（DASHSCOPE_API_KEY 环境变量）
  - FAISS 已安装（pip install faiss-cpu==1.9.0）

如果不满足任何前置条件，脚本会 hard fail 并输出中文错误信息。
评测请用 scripts/06_run_evaluation.py --decisions <jsonl_path>。

用法:
    # memory_off（基线）
    python scripts/07_run_full_demo.py --mode memory_off --max-scenes 1 --max-frames 5

    # memory_on（带三层记忆）
    python scripts/07_run_full_demo.py --mode memory_on --max-scenes 1 --max-frames 5

    # 不 resume，强制重跑覆盖已有 jsonl
    python scripts/07_run_full_demo.py --mode memory_on --no-resume

    # 自定义输出路径
    python scripts/07_run_full_demo.py --mode memory_off --output outputs/my_run.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="完整 Demo：逐帧在线循环，单次运行只跑一种 mode",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["memory_on", "memory_off"],
        required=True,
        help="运行模式（必填）：memory_on 使用三层记忆，memory_off 作为对照基线",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选，默认加载 config/ 下所有 YAML）",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default=None,
        help="nuScenes 数据集根目录（覆盖配置文件中的 dataroot）",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="nuScenes 数据集版本（默认 v1.0-mini）",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="最多处理的场景数（调试用，默认全部）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="每个场景最多返回的关键帧数（调试用）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="决策 jsonl 输出路径（覆盖默认 outputs/decisions_<mode>_<run_id>.jsonl）",
    )
    # resume 默认 True，使用 --no-resume 关闭
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="启动时扫描已有 jsonl，跳过已处理 sample_token（默认开启）",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="不 resume：删掉已有 jsonl 重新跑",
    )
    return parser.parse_args()


def main() -> None:
    """主函数：CLI 入口。

    解析参数 → 加载配置 → 调用 ``run_full_demo``（OnlineDrivingLoop 在线循环）。
    API Key 缺失、DINOv2 权重不可加载、FAISS 不可用、nuScenes 数据缺失时 hard fail。
    """
    from src.vla_memory.common.config import load_config
    from src.vla_memory.common.logging_utils import setup_logger
    from src.vla_memory.pipeline.full_demo_pipeline import run_full_demo

    args = parse_args()

    # ---- 构建 CLI 配置覆盖 ----
    overrides = {}
    if args.dataroot:
        overrides["dataroot"] = args.dataroot
    if args.version:
        overrides["version"] = args.version
    if args.max_scenes is not None:
        overrides.setdefault("subset", {})["enabled"] = True
        overrides["subset"]["max_scenes"] = args.max_scenes
    if args.max_frames is not None:
        overrides.setdefault("subset", {})["enabled"] = True
        overrides["subset"]["max_samples_per_scene"] = args.max_frames

    # ---- 加载配置 ----
    config = load_config(overrides=overrides if overrides else None)
    config.ensure_output_dirs()

    logger = setup_logger(
        name="07_full_demo",
        level=config.get_nested("logging", "level", default="INFO"),
        log_dir=config.get_path("log_dir"),
    )

    logger.info("=" * 60)
    logger.info("完整 Demo 启动 (mode=%s, resume=%s)", args.mode, args.resume)
    logger.info("  数据目录: %s", config.get("dataroot"))
    logger.info("  数据集版本: %s", config.get("version", "v1.0-mini"))
    if args.max_scenes:
        logger.info("  最大场景数: %d", args.max_scenes)
    if args.max_frames:
        logger.info("  最大帧数: %d", args.max_frames)
    if args.output:
        logger.info("  输出路径（覆盖默认）: %s", args.output)
    logger.info("=" * 60)

    try:
        result = run_full_demo(
            config=config,
            mode=args.mode,
            resume=args.resume,
            output_jsonl_path=args.output,
        )
        if "error" in result:
            logger.error("Demo 运行中断: %s", result["error"])
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("完整 Demo 运行成功！")
        logger.info("  mode = %s", result["mode"])
        logger.info("  写入 %d 帧决策记录", len(result["records"]))
        logger.info("  jsonl 路径: %s", result["output_path"])
        logger.info("")
        logger.info("评测请运行：")
        logger.info(
            "  python scripts/06_run_evaluation.py --decisions %s",
            result["output_path"],
        )
        logger.info("=" * 60)
    except FileNotFoundError as e:
        logger.error("文件或数据集不存在: %s", e)
        sys.exit(1)
    except EnvironmentError as e:
        logger.error("环境错误（API Key / FAISS / 模型权重）: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Demo 运行失败: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
