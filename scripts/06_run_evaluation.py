#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测运行脚本（R3 重构）
========================
对 OnlineDrivingLoop 写出的决策结果 jsonl 跑评测。
支持两种入口：

1. 单 mode 评测：``--decisions PATH [--mode LABEL]``
2. 双 mode 对比：``--compare PATH_A PATH_B``

核心流程：
1. 加载 JSONL 格式的决策结果。
2. 从 nuScenes ego_pose 构建未来真值轨迹（RouteInfer 生成伪行为标签）。
3. 计算 ADE / FDE / L2@1s/2s/3s / 轨迹有效率 / 行为准确率。
4. 写 CSV / JSONL / Markdown 报告到 ``outputs/reports/``。

输入文件不存在 / nuScenes 数据不存在（且决策中无内置真值）时 hard fail。

用法:
    # 单 mode
    python scripts/06_run_evaluation.py --decisions outputs/decisions_memory_off_xxx.jsonl

    # 单 mode 自定义 mode label（默认从文件名推断或填 "memory_on"）
    python scripts/06_run_evaluation.py --decisions outputs/d.jsonl --mode my_run

    # 双 mode 对比
    python scripts/06_run_evaluation.py --compare \\
        outputs/decisions_memory_on_xxx.jsonl outputs/decisions_memory_off_xxx.jsonl

    # 限制评测帧数（调试）
    python scripts/06_run_evaluation.py --decisions outputs/d.jsonl --max-frames 10

    # 自定义报告输出目录
    python scripts/06_run_evaluation.py --decisions outputs/d.jsonl --output-dir outputs/my_eval
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="评测运行：对决策 jsonl 跑评测，支持单 mode / 双 mode 对比",
    )
    parser.add_argument(
        "--decisions",
        type=str,
        default=None,
        help="单 mode 评测的决策 jsonl 路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help=(
            "（仅与 --decisions 同用）单 mode 评测时的 mode 标签。"
            "未提供时从文件名 decisions_<mode>_xxx.jsonl 推断；推断失败默认 memory_on。"
        ),
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("JSONL_A", "JSONL_B"),
        default=None,
        help="双 mode 对比评测：传两条 jsonl 路径（mode 标签从文件名推断）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多评测的帧数（调试用）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="报告输出目录（覆盖配置 evaluation.output.report_dir）",
    )
    return parser.parse_args()


# decisions_<mode>_<run_id>.jsonl  →  mode
_MODE_FROM_FILENAME = re.compile(r"^decisions_([a-zA-Z_]+?)(?:_[\w\-]+)?\.jsonl$")


def _infer_mode_from_path(path: Path, fallback: str = "memory_on") -> str:
    """从 decisions_<mode>_<run_id>.jsonl 推断 mode；推断失败用 fallback。"""
    m = _MODE_FROM_FILENAME.match(path.name)
    if m:
        return m.group(1)
    return fallback


def main() -> None:
    """主函数。"""
    from src.vla_memory.common.config import load_config
    from src.vla_memory.common.logging_utils import setup_logger
    from src.vla_memory.pipeline.eval_pipeline import (
        load_decisions_jsonl,
        run_eval_compare,
        run_eval_pipeline,
    )

    args = parse_args()
    if not (args.decisions or args.compare):
        print(
            "错误：必须提供 --decisions PATH 或 --compare PATH_A PATH_B 之一。\n"
            "用法示例：\n"
            "  单 mode：python scripts/06_run_evaluation.py "
            "--decisions outputs/decisions_memory_off_xxx.jsonl\n"
            "  双 mode：python scripts/06_run_evaluation.py --compare "
            "outputs/decisions_memory_on_xxx.jsonl outputs/decisions_memory_off_xxx.jsonl",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.decisions and args.compare:
        print("错误：--decisions 与 --compare 互斥，请只用其一。", file=sys.stderr)
        sys.exit(2)

    config = load_config()
    config.ensure_output_dirs()

    logger = setup_logger(
        name="06_evaluation",
        level=config.get_nested("logging", "level", default="INFO"),
        log_dir=config.get_path("log_dir"),
    )

    nuscenes_cfg = config.get("data_nuscenes", {}) or {}
    nuscenes_dataroot = nuscenes_cfg.get("dataroot") or config.get("dataroot")
    nuscenes_version = nuscenes_cfg.get("version", "v1.0-mini")

    try:
        if args.decisions:
            # ---- 单 mode 评测 ----
            path = Path(args.decisions)
            if not path.exists():
                logger.error("决策结果文件不存在: %s", path)
                sys.exit(1)
            mode = args.mode or _infer_mode_from_path(path)
            logger.info("=" * 60)
            logger.info("单 mode 评测: mode=%s, 输入=%s", mode, path)
            logger.info("=" * 60)

            results = load_decisions_jsonl(path)
            if args.max_frames:
                results = results[: args.max_frames]
                logger.info("限制评测帧数: %d", args.max_frames)
            logger.info("加载 %d 条决策记录", len(results))

            summary = run_eval_pipeline(
                results=results, mode=mode, config=config,
                nuscenes_dataroot=nuscenes_dataroot,
                nuscenes_version=nuscenes_version,
                report_dir=args.output_dir,
            )
            _log_summary(logger, mode, summary)

        else:
            # ---- 双 mode 对比 ----
            path_a, path_b = (Path(p) for p in args.compare)
            for p in (path_a, path_b):
                if not p.exists():
                    logger.error("决策结果文件不存在: %s", p)
                    sys.exit(1)
            mode_a = _infer_mode_from_path(path_a, fallback="memory_on")
            mode_b = _infer_mode_from_path(path_b, fallback="memory_off")
            if mode_a == mode_b:
                # 同名 mode 加 _A/_B 后缀避免报告中覆盖
                mode_a, mode_b = f"{mode_a}_A", f"{mode_b}_B"

            logger.info("=" * 60)
            logger.info("双 mode 对比评测")
            logger.info("  %s ← %s", mode_a, path_a)
            logger.info("  %s ← %s", mode_b, path_b)
            logger.info("=" * 60)

            results_a = load_decisions_jsonl(path_a)
            results_b = load_decisions_jsonl(path_b)
            if args.max_frames:
                results_a = results_a[: args.max_frames]
                results_b = results_b[: args.max_frames]
                logger.info("限制评测帧数: %d", args.max_frames)
            logger.info("%s: %d 条；%s: %d 条",
                        mode_a, len(results_a), mode_b, len(results_b))

            summaries = run_eval_compare(
                results_by_mode={mode_a: results_a, mode_b: results_b},
                config=config,
                nuscenes_dataroot=nuscenes_dataroot,
                nuscenes_version=nuscenes_version,
                report_dir=args.output_dir,
            )
            for mode, s in summaries.items():
                _log_summary(logger, mode, s)

        report_dir = args.output_dir or config.get_nested(
            "evaluation", "output", "report_dir", default="outputs/reports",
        )
        logger.info("=" * 60)
        logger.info("✅ 评测完成！报告已生成到: %s", report_dir)
        logger.info("  - eval_summary.csv")
        logger.info("  - eval_detail.jsonl")
        logger.info("  - eval_report.md")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error("nuScenes 数据不存在: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("评测失败: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("评测异常: %s", e)
        raise


def _log_summary(logger, mode: str, summary) -> None:
    logger.info(
        "[%s] ADE均值=%s, FDE均值=%s, 轨迹有效率=%.4f, 行为准确率=%s, Fallback=%d",
        mode,
        f"{summary.ade_mean:.4f}" if summary.ade_mean is not None else "N/A",
        f"{summary.fde_mean:.4f}" if summary.fde_mean is not None else "N/A",
        summary.valid_trajectory_rate,
        f"{summary.behavior_accuracy:.4f}" if summary.behavior_accuracy is not None else "N/A",
        summary.fallback_count,
    )
    if summary.l2_mean_per_horizon:
        l2_parts = ", ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=N/A"
            for k, v in sorted(
                summary.l2_mean_per_horizon.items(),
                key=lambda kv: float(kv[0].replace("L2_", "").rstrip("s")),
            )
        )
        logger.info("  L2 per horizon: %s", l2_parts)


if __name__ == "__main__":
    main()
