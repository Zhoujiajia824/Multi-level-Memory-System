"""评测报告生成模块
==================
生成 eval_summary.csv、eval_detail.jsonl、eval_report.md 三种报告。

Markdown 报告包含：
- 实验标题和 run_id
- 总体指标（memory_on / memory_off）
- 差值对比表
- 分组统计（scene_id / weather_id / behavior）
- 失败案例示例
- fallback 使用统计
- 第一版评测限制说明
- 后续评测升级方向
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from src.vla_memory.schemas.evaluation import EvalSampleResult, EvalSummary
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("report_writer")


class ReportWriter:
    """评测报告生成器。

    生成三种格式的评测报告：
    - eval_summary.csv: 汇总指标 CSV
    - eval_detail.jsonl: 逐样本详细结果
    - eval_report.md: 完整 Markdown 报告

    Args:
        report_dir: 报告输出目录。
    """

    def __init__(self, report_dir: str = "outputs/reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(self, summaries: Dict[str, EvalSummary]) -> Path:
        """写入汇总 CSV 文件。

        包含各模式的总体评测指标。P6 起新增 L2_*s_mean 列。
        L2 horizon 集合按各 summary 出现过的标签的并集统一列出，
        某 mode 缺该 horizon 时填 ``N/A``。

        Args:
            summaries: {mode: EvalSummary} 字典。

        Returns:
            CSV 文件路径。
        """
        csv_path = self.report_dir / "eval_summary.csv"

        # 收集所有 L2 horizon 标签（去重 + 稳定排序：按数字大小）
        l2_labels: List[str] = []
        seen = set()
        for s in summaries.values():
            if s.l2_mean_per_horizon:
                for k in s.l2_mean_per_horizon.keys():
                    if k not in seen:
                        seen.add(k)
                        l2_labels.append(k)
        # 排序：L2_1s -> L2_2s -> L2_3s
        def _sort_key(label: str) -> float:
            try:
                return float(label.replace("L2_", "").rstrip("s"))
            except ValueError:
                return float("inf")
        l2_labels.sort(key=_sort_key)
        l2_columns = [f"{lbl}_mean" for lbl in l2_labels]

        with open(str(csv_path), "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "memory_mode",
                "num_samples",
                "valid_trajectory_rate",
                "ade_mean",
                "ade_median",
                "ade_std",
                "fde_mean",
                "fde_median",
                "fde_std",
                *l2_columns,
                "behavior_accuracy",
                "behavior_valid_count",
                "fallback_count",
            ])
            for mode, summary in summaries.items():
                l2_vals = summary.l2_mean_per_horizon or {}
                l2_cells = [
                    f"{l2_vals[lbl]:.4f}" if l2_vals.get(lbl) is not None else "N/A"
                    for lbl in l2_labels
                ]
                writer.writerow([
                    mode,
                    summary.total_samples,
                    f"{summary.valid_trajectory_rate:.4f}",
                    f"{summary.ade_mean:.4f}" if summary.ade_mean is not None else "N/A",
                    f"{summary.ade_median:.4f}" if summary.ade_median is not None else "N/A",
                    f"{summary.ade_std:.4f}" if summary.ade_std is not None else "N/A",
                    f"{summary.fde_mean:.4f}" if summary.fde_mean is not None else "N/A",
                    f"{summary.fde_median:.4f}" if summary.fde_median is not None else "N/A",
                    f"{summary.fde_std:.4f}" if summary.fde_std is not None else "N/A",
                    *l2_cells,
                    f"{summary.behavior_accuracy:.4f}" if summary.behavior_accuracy is not None else "N/A",
                    summary.behavior_valid_count,
                    summary.fallback_count,
                ])
        logger.info(f"汇总 CSV 已写入: {csv_path}")
        return csv_path

    def write_jsonl(self, results: List[EvalSampleResult]) -> Path:
        """写入详细 JSONL 文件。

        每行包含一个样本的完整评测结果。
        P6：pydantic v2 用 model_dump 代替 v1 的 dict。

        Args:
            results: 评测结果列表。

        Returns:
            JSONL 文件路径。
        """
        jsonl_path = self.report_dir / "eval_detail.jsonl"
        with open(str(jsonl_path), "w", encoding="utf-8") as f:
            for r in results:
                payload = r.model_dump() if hasattr(r, "model_dump") else r.dict()
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        logger.info(f"详细 JSONL 已写入: {jsonl_path} ({len(results)} 条)")
        return jsonl_path

    def write_markdown(
        self,
        summaries: Dict[str, EvalSummary],
        results: Optional[List[EvalSampleResult]] = None,
        dataset_version: str = "v1.0-mini",
        dataset_dir: str = "data/nuscenes/raw",
    ) -> Path:
        """写入 Markdown 评测报告。

        包含完整的评测信息：总体指标、差值对比、分组统计、失败案例、
        fallback 统计、限制说明、后续方向。

        Args:
            summaries: {mode: EvalSummary} 字典。
            results: 完整评测结果列表（用于提取失败案例）。
            dataset_version: 数据集版本。
            dataset_dir: 数据集目录。

        Returns:
            Markdown 文件路径。
        """
        md_path = self.report_dir / "eval_report.md"
        run_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")

        lines = []

        # ---- 标题 ----
        lines.append("# 评测报告：memory_on vs memory_off 对比")
        lines.append("")
        lines.append(f"**run_id**: `{run_id}`")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**数据集版本**: {dataset_version}")
        lines.append(f"**数据目录**: `{dataset_dir}`")
        lines.append("")

        # ---- 重要声明 ----
        lines.append("## ⚠️ 重要声明")
        lines.append("")
        lines.append("> **第一版评测是 demo 内部 memory_on vs memory_off 对比，不是官方 nuScenes planning benchmark。**")
        lines.append(">")
        lines.append("> - behavior_accuracy 使用伪标签（来自 RouteInfer 导航语义推断），不是人工标注。")
        lines.append("> - 真值轨迹来自 nuScenes ego_pose 派生，不是 ground truth planning 标注。")
        lines.append("> - 当前只使用 CAM_FRONT 和 ego 状态，未使用多摄像头、LiDAR、地图等信息。")
        lines.append("> - collision / offroad 尚未严格评测。")
        lines.append("")

        # ---- 评测配置 ----
        lines.append("## 评测配置")
        lines.append("")
        lines.append("| 项目 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 数据集版本 | {dataset_version} |")
        lines.append(f"| 坐标系 | ego-centric（x 前向，y 左向，单位米） |")
        lines.append(f"| 轨迹重采样 | 统一 waypoint 数 |")
        lines.append("")

        # ---- 总体指标 ----
        for mode, summary in summaries.items():
            lines.append(f"## 模式: {mode}")
            lines.append("")
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| 总样本数 | {summary.total_samples} |")
            lines.append(f"| 有效样本数 | {summary.valid_samples} |")
            lines.append(f"| 轨迹有效率 | {summary.valid_trajectory_rate:.4f} |")
            ade_mean_str = f"{summary.ade_mean:.4f}" if summary.ade_mean is not None else "N/A"
            ade_median_str = f"{summary.ade_median:.4f}" if summary.ade_median is not None else "N/A"
            fde_mean_str = f"{summary.fde_mean:.4f}" if summary.fde_mean is not None else "N/A"
            fde_median_str = f"{summary.fde_median:.4f}" if summary.fde_median is not None else "N/A"
            ade_std_str = f"{summary.ade_std:.4f}" if summary.ade_std is not None else "N/A"
            fde_std_str = f"{summary.fde_std:.4f}" if summary.fde_std is not None else "N/A"
            lines.append(f"| ADE 均值 | {ade_mean_str} |")
            lines.append(f"| ADE 中位数 | {ade_median_str} |")
            lines.append(f"| ADE 标准差 | {ade_std_str} |")
            lines.append(f"| FDE 均值 | {fde_mean_str} |")
            lines.append(f"| FDE 中位数 | {fde_median_str} |")
            lines.append(f"| FDE 标准差 | {fde_std_str} |")
            # P6 新增：L2 per horizon 均值
            if summary.l2_mean_per_horizon:
                for lbl in sorted(
                    summary.l2_mean_per_horizon.keys(),
                    key=lambda x: float(x.replace("L2_", "").rstrip("s")),
                ):
                    val = summary.l2_mean_per_horizon[lbl]
                    val_str = f"{val:.4f}" if val is not None else "N/A"
                    lines.append(f"| {lbl} 均值 | {val_str} |")
            ba_str = f"{summary.behavior_accuracy:.4f}" if summary.behavior_accuracy is not None else "N/A"
            lines.append(f"| 行为准确率 | {ba_str} |")
            lines.append(f"| 行为有效样本数 | {summary.behavior_valid_count} |")
            lines.append(f"| Fallback 使用数 | {summary.fallback_count} |")
            lines.append("")

        # ---- 差值对比 ----
        if "memory_on" in summaries and "memory_off" in summaries:
            lines.append("## memory_on vs memory_off 差值对比")
            lines.append("")
            lines.append("| 指标 | memory_on | memory_off | 差值 | 说明 |")
            lines.append("|------|-----------|------------|------|------|")

            s_on = summaries["memory_on"]
            s_off = summaries["memory_off"]

            # ADE 对比
            if s_on.ade_mean is not None and s_off.ade_mean is not None:
                delta_ade = s_on.ade_mean - s_off.ade_mean
                note = "memory_on 更优 ↓" if delta_ade < 0 else "memory_off 更优 ↑" if delta_ade > 0 else "持平"
                lines.append(
                    f"| ADE 均值 | {s_on.ade_mean:.4f} | {s_off.ade_mean:.4f} | "
                    f"{delta_ade:+.4f} | {note} |"
                )
            else:
                lines.append("| ADE 均值 | N/A | N/A | N/A | 数据不足 |")

            # FDE 对比
            if s_on.fde_mean is not None and s_off.fde_mean is not None:
                delta_fde = s_on.fde_mean - s_off.fde_mean
                note = "memory_on 更优 ↓" if delta_fde < 0 else "memory_off 更优 ↑" if delta_fde > 0 else "持平"
                lines.append(
                    f"| FDE 均值 | {s_on.fde_mean:.4f} | {s_off.fde_mean:.4f} | "
                    f"{delta_fde:+.4f} | {note} |"
                )
            else:
                lines.append("| FDE 均值 | N/A | N/A | N/A | 数据不足 |")

            # 有效率对比
            delta_rate = s_on.valid_trajectory_rate - s_off.valid_trajectory_rate
            lines.append(
                f"| 轨迹有效率 | {s_on.valid_trajectory_rate:.4f} | "
                f"{s_off.valid_trajectory_rate:.4f} | {delta_rate:+.4f} | |"
            )

            # 行为准确率对比
            if s_on.behavior_accuracy is not None and s_off.behavior_accuracy is not None:
                delta_ba = s_on.behavior_accuracy - s_off.behavior_accuracy
                lines.append(
                    f"| 行为准确率 | {s_on.behavior_accuracy:.4f} | "
                    f"{s_off.behavior_accuracy:.4f} | {delta_ba:+.4f} | |"
                )
            else:
                lines.append("| 行为准确率 | N/A | N/A | N/A | 数据不足 |")

            lines.append("")

        # ---- 分组统计 ----
        for mode, summary in summaries.items():
            # 按 scene_id 分组
            if summary.scene_grouped:
                lines.append(f"### {mode} — 按 scene_id 分组")
                lines.append("")
                lines.append("| 场景 | 数量 | ADE均值 | FDE均值 | 轨迹有效率 |")
                lines.append("|------|------|---------|---------|-----------|")
                for sid, stats in summary.scene_grouped.items():
                    ade = f"{stats['ade_mean']:.4f}" if stats.get("ade_mean") is not None else "N/A"
                    fde = f"{stats['fde_mean']:.4f}" if stats.get("fde_mean") is not None else "N/A"
                    lines.append(f"| {sid} | {stats['count']} | {ade} | {fde} | {stats['valid_rate']:.4f} |")
                lines.append("")

            # 按 weather_id 分组
            if summary.weather_grouped:
                lines.append(f"### {mode} — 按 weather_id 分组")
                lines.append("")
                lines.append("| 天气 | 数量 | ADE均值 | FDE均值 | 轨迹有效率 |")
                lines.append("|------|------|---------|---------|-----------|")
                for wid, stats in summary.weather_grouped.items():
                    ade = f"{stats['ade_mean']:.4f}" if stats.get("ade_mean") is not None else "N/A"
                    fde = f"{stats['fde_mean']:.4f}" if stats.get("fde_mean") is not None else "N/A"
                    lines.append(f"| {wid} | {stats['count']} | {ade} | {fde} | {stats['valid_rate']:.4f} |")
                lines.append("")

            # 按 behavior 分组
            if summary.behavior_grouped:
                lines.append(f"### {mode} — 按 behavior 分组")
                lines.append("")
                lines.append("| 行为 | 数量 | ADE均值 | FDE均值 | 轨迹有效率 |")
                lines.append("|------|------|---------|---------|-----------|")
                for bid, stats in summary.behavior_grouped.items():
                    ade = f"{stats['ade_mean']:.4f}" if stats.get("ade_mean") is not None else "N/A"
                    fde = f"{stats['fde_mean']:.4f}" if stats.get("fde_mean") is not None else "N/A"
                    lines.append(f"| {bid} | {stats['count']} | {ade} | {fde} | {stats['valid_rate']:.4f} |")
                lines.append("")

        # ---- Fallback 使用统计 ----
        lines.append("## Fallback 使用统计")
        lines.append("")
        lines.append("| 模式 | 总样本数 | Fallback 数 | Fallback 率 |")
        lines.append("|------|----------|------------|-------------|")
        for mode, summary in summaries.items():
            fb_rate = summary.fallback_count / summary.total_samples if summary.total_samples > 0 else 0.0
            lines.append(f"| {mode} | {summary.total_samples} | {summary.fallback_count} | {fb_rate:.4f} |")
        lines.append("")

        # ---- 失败案例示例 ----
        if results:
            failed_results = [
                r for r in results
                if not r.is_valid_trajectory or r.error_message
            ]
            if failed_results:
                lines.append("## 失败案例示例（前 10 个）")
                lines.append("")
                lines.append("| sample_token | 模式 | 错误原因 | ADE | FDE |")
                lines.append("|--------------|------|----------|-----|-----|")
                for r in failed_results[:10]:
                    token_short = r.sample_token[:16] + "..." if len(r.sample_token) > 16 else r.sample_token
                    err = r.valid_error or r.error_message or "未知"
                    if len(err) > 40:
                        err = err[:40] + "..."
                    ade_str = f"{r.ade:.4f}" if r.ade is not None else "N/A"
                    fde_str = f"{r.fde:.4f}" if r.fde is not None else "N/A"
                    lines.append(f"| {token_short} | {r.mode} | {err} | {ade_str} | {fde_str} |")
                lines.append("")
            else:
                lines.append("## 失败案例示例")
                lines.append("")
                lines.append("所有样本均通过评测，无失败案例。")
                lines.append("")

        # ---- 第一版评测限制 ----
        lines.append("## 第一版评测限制")
        lines.append("")
        lines.append("1. **不是官方 nuScenes planning benchmark**：本评测是 demo 内部对比，不使用 nuScenes 官方 planning 评测协议。")
        lines.append("2. **behavior_accuracy 使用伪标签**：行为准确率基于 RouteInfer 推断的伪导航语义映射，不是人工标注真值。映射关系可能不完美。")
        lines.append("3. **真值轨迹来自 ego_pose 派生**：未来轨迹真值通过 nuScenes ego_pose 序列构建，转换为 ego-centric 坐标系。不是 ground truth planning 标注。")
        lines.append("4. **当前只使用 CAM_FRONT 和 ego 状态**：未使用多摄像头、LiDAR、雷达、地图等信息。")
        lines.append("5. **collision / offroad 尚未严格评测**：碰撞和偏离道路检测是预留接口，第一版不实现。需要 nuScenes bounding box 和地图信息。")
        lines.append("6. **轨迹插值可能引入误差**：重采样到统一 waypoint 数时使用线性插值，可能影响位移计算精度。")
        lines.append("")

        # ---- 后续评测升级方向 ----
        lines.append("## 后续评测升级方向")
        lines.append("")
        lines.append("1. 接入 nuScenes 官方 planning benchmark 评测协议。")
        lines.append("2. 使用 nuScenes bounding box 实现碰撞率评测。")
        lines.append("3. 使用 nuScenes 地图实现偏离道路率评测。")
        lines.append("4. 引入人工标注行为标签替代伪标签。")
        lines.append("5. 支持多摄像头融合后的评测。")
        lines.append("6. 支持更长预测时间窗口（5s / 8s）的评测。")
        lines.append("7. 支持 CARLA 仿真环境中的在线评测。")
        lines.append("8. 引入轨迹平滑度、加速度舒适度等评测指标。")
        lines.append("")

        with open(str(md_path), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Markdown 报告已写入: {md_path}")
        return md_path
