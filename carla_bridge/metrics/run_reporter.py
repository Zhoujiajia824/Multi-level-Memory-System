"""闭环运行报告
==============
把 :class:`MetricsCollector.summary()` + run 元信息写成 Markdown + JSON。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple


def write_report(summary: dict, run_info: dict, output_dir: str) -> Tuple[str, str]:
    """写 Markdown + JSON 报告，返回 ``(md_path, json_path)``。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"carla_run_{run_info.get('scenario', 'unknown')}_{run_info.get('mode', 'unknown')}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    payload = {"run_info": run_info, "metrics": summary}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_md(run_info, summary))

    return str(md_path), str(json_path)


def _render_md(run_info: dict, s: dict) -> str:
    lines = [
        "# CARLA 闭环运行报告",
        "",
        "## 运行信息",
        f"- 运行编号: {run_info.get('run_id', '')}",
        f"- 状态: {run_info.get('status', 'unknown')}",
        f"- 路线验收通过: {run_info.get('success', False)}（按序完成整条路线并到达终点）",
        f"- 异常轨迹拒绝次数: {run_info.get('trajectory_rejections', 0)}",
        f"- 采集失败次数: {run_info.get('capture_failures', 0)}",
        f"- 场景: `{run_info.get('scenario')}`",
        f"- 模式: `{run_info.get('mode')}`",
        f"- 轨迹后端: `{run_info.get('planner_backend', 'vlm_trajectory')}`",
        f"- 语义影响候选选择: {run_info.get('semantic_influence', False)}",
        f"- 仿真时长: {run_info.get('duration_s')} s",
        f"- 全量认知间隔: {run_info.get('replan_interval_s')} s",
        f"- PDM 重规划间隔: {run_info.get('pdm_replan_interval_s')} s",
    ]
    if run_info.get("early_terminated"):
        lines.append(
            f"- ⚠️ **提前终止**（原因: {run_info.get('early_terminated_reason', '?')}）"
        )
    lines += [
        "",
        "## 安全指标",
        f"- 碰撞次数: **{s.get('collision_count', 0)}**"
        f"（去重后；原始事件 {run_info.get('collision_raw_events', 0)}，"
        f"涉及 {run_info.get('collision_unique_actors', 0)} 个独立 actor）",
        f"- 路线完成度: {s.get('route_completion', 0) * 100:.1f}%",
        f"- 闯红灯: {s.get('red_light_violations', 0)}",
        f"- 逆行: {s.get('wrong_way_violations', 0)}",
        f"- 超速 tick 数: {s.get('speeding_ticks', 0)} / {s.get('total_ticks', 0)}",
        "",
        "## 舒适度指标",
        f"- 最大速度: {s.get('max_speed_mps', 0):.2f} m/s",
        f"- 最大平面加速度模长: {s.get('max_accel_mps2', 0):.2f} m/s²",
        f"- 最大加速度模长变化率（旧 jerk 字段）: {s.get('max_jerk_mps3', 0):.2f} m/s³",
    ]
    if s.get("comfort_schema_version", 1) >= 2:
        sampling = s.get("comfort_sampling", {})
        lines += [
            f"- 平面加速度模长 P95 / RMS: {_number(s.get('p95_accel_mps2'))} / "
            f"{_number(s.get('rms_accel_mps2'))} m/s²；最大值时刻 {_number(s.get('max_accel_time_s'))} s",
            f"- 纵向最大加速 / 减速: {_number(s.get('max_longitudinal_acceleration_mps2'))} / "
            f"{_number(s.get('max_longitudinal_deceleration_mps2'))} m/s²；"
            f"最大横向加速度 {_number(s.get('max_lateral_acceleration_mps2'))} m/s²",
            f"- 平面向量 jerk 最大值 / P95 / RMS: {_number(s.get('max_planar_jerk_mps3'))} / "
            f"{_number(s.get('p95_planar_jerk_mps3'))} / {_number(s.get('rms_planar_jerk_mps3'))} m/s³；"
            f"最大值时刻 {_number(s.get('max_planar_jerk_time_s'))} s",
            f"- 纵向 jerk 最大值 / P95 / RMS: {_number(s.get('max_longitudinal_jerk_mps3'))} / "
            f"{_number(s.get('p95_longitudinal_jerk_mps3'))} / {_number(s.get('rms_longitudinal_jerk_mps3'))} m/s³",
            f"- 急刹段数（纵向加速度 ≤ {s.get('hard_braking_threshold_mps2', -3):.1f} m/s²）: "
            f"{s.get('hard_braking_events') if s.get('hard_braking_events') is not None else '未采集'}",
            f"- 观测静止时长（速度 ≤ {s.get('stop_speed_threshold_mps', .2):.1f} m/s）: "
            f"{_number(s.get('stopped_time_s'))} s；"
            f"连续静止 ≥ {s.get('stop_episode_min_duration_s', 1):.1f} s 的段数 {s.get('stop_episodes', 0)}；"
            f"最长连续静止 {_number(s.get('longest_stop_s'))} s",
            f"- 采样间隔（仿真时间）: 配置 {_number(sampling.get('configured_dt_s'))} s，"
            f"观测 {_number(sampling.get('min_observed_dt_s'))}–{_number(sampling.get('max_observed_dt_s'))} s；"
            f"间隔异常 {sampling.get('gapped_intervals', 0)}，无效时间差 {sampling.get('invalid_intervals', 0)}",
            "",
            "> jerk 按相邻运行采样时间差计算（同步循环按固定 tick 累加）；平面向量 jerk 保留加速转制动的方向变化，"
            "纵向 jerk 为有符号车体纵向加速度变化率，旧字段只计算加速度模长变化率。"
            "P95/RMS 按观测样本统计，未平滑或剔除启动、碰撞、紧急制动尖峰；"
            "采样缺口内的 jerk 仅为区间平均变化率。静止统计不区分合理等待与无必要停车。",
        ]
    lines += [
        "",
        "> 闭环评测指标（原 ADE/FDE 不适用：未来由自车决策产生）。",
    ]
    return "\n".join(lines) + "\n"


def _number(value) -> str:
    return "未采集" if value is None else f"{value:.2f}"
