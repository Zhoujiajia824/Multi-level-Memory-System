"""评测流水线
============
对 OnlineDrivingLoop 写出的决策结果（jsonl）跑评测，
支持单 mode（仅一条 jsonl）和双 mode 对比（两条 jsonl）。

核心流程：
1. 加载 JSONL 格式的决策结果。
2. 从 nuScenes ego_pose 构造未来真值轨迹。
3. 使用 RouteInfer 生成伪行为标签。
4. 使用 Evaluator 计算各项指标（含 L2 per horizon）。
5. 使用 ReportWriter 生成 CSV/JSONL/Markdown 报告。

R3 重构：以前 ``run_eval_pipeline(results_on, results_off, ...)`` 强制要两条
mode 同时存在。现在拆为：
* ``run_eval_pipeline(results, mode, ...)`` —— 单 mode 评测。
* ``run_eval_compare({"memory_on": [...], "memory_off": [...]}, ...)`` —— 对比评测。

不允许生成假真值轨迹。不允许用预测轨迹当真值。
nuScenes 数据不存在时 hard fail。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.vla_memory.common.config import Config, load_config
from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.evaluation.evaluator import Evaluator
from src.vla_memory.evaluation.report_writer import ReportWriter
from src.vla_memory.schemas.evaluation import EvalSampleResult, EvalSummary

logger = get_logger("eval_pipeline")


def load_decisions_jsonl(path: Path) -> List[Dict[str, Any]]:
    """加载 JSONL 格式的决策结果文件。

    每行一个 JSON 对象，支持 # 开头的注释行。

    Args:
        path: JSONL 文件路径。

    Returns:
        决策结果列表。

    Raises:
        FileNotFoundError: 文件不存在。
        RuntimeError: 文件为空。
    """
    if not path.exists():
        raise FileNotFoundError(f"决策结果文件不存在: {path}")

    results = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"跳过第 {line_num} 行（JSON 解析失败）: {e}")

    if not results:
        raise RuntimeError(f"决策结果文件为空或全部解析失败: {path}")

    return results


def build_ground_truth_from_nuscenes(
    decisions: List[Dict[str, Any]],
    nuscenes_dataroot: str,
    nuscenes_version: str = "v1.0-mini",
    future_seconds: float = 3.0,
    nav_to_behavior_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """从 nuScenes ego_pose 构建未来真值轨迹和伪行为标签。

    对每条决策记录：
    1. 使用 sample_token 查询 nuScenes ego_pose。
    2. 构建未来 N 秒的 ego-centric 真值轨迹。
    3. 使用 RouteInfer 推断伪导航语义作为行为真值。
    4. 将真值轨迹和行为标签写入决策记录。

    不允许伪造真值轨迹。
    真值轨迹不足时跳过该样本并记录 warning。
    不会补假点。

    Args:
        decisions: 决策结果列表（每条需包含 sample_token 字段）。
        nuscenes_dataroot: nuScenes 数据集根目录。
        nuscenes_version: 数据集版本。
        future_seconds: 未来轨迹时间窗（秒）。
        nav_to_behavior_map: 导航语义到行为枚举的映射。

    Returns:
        (增强后的决策列表, 成功构建真值的样本数) 元组。

    Raises:
        FileNotFoundError: nuScenes 数据不存在时 hard fail。
        ImportError: nuscenes-devkit 未安装时 hard fail。
    """
    from src.vla_memory.data.nuscenes_adapter import NuScenesAdapter
    from src.vla_memory.data.route_infer import RouteInfer

    # 检查数据集
    dataroot_path = Path(nuscenes_dataroot)
    if not dataroot_path.exists():
        raise FileNotFoundError(
            f"nuScenes 数据集目录不存在: {dataroot_path}\n"
            f"无法构建真值轨迹，评测终止。\n"
            f"请将 nuScenes 数据放置到正确位置。不允许使用假数据。"
        )

    # 初始化适配器
    adapter = NuScenesAdapter(
        dataroot=dataroot_path,
        version=nuscenes_version,
    )
    adapter.load()

    # 初始化导航语义推断器
    route_infer = RouteInfer()

    gt_success = 0
    gt_fail = 0

    for i, record in enumerate(decisions):
        sample_token = record.get("sample_token", "")
        if not sample_token:
            # 尝试从 frame_id 获取
            sample_token = record.get("frame_id", "")

        if not sample_token:
            record["ground_truth_trajectory"] = []
            record["ground_truth_behavior"] = ""
            gt_fail += 1
            logger.warning(f"第 {i} 条记录缺少 sample_token，跳过真值构建")
            continue

        try:
            # 构建未来真值轨迹
            gt_traj = adapter.get_future_ego_trajectory(
                sample_token=sample_token,
                future_seconds=future_seconds,
            )

            if not gt_traj or len(gt_traj) < 2:
                record["ground_truth_trajectory"] = []
                record["ground_truth_behavior"] = ""
                gt_fail += 1
                logger.warning(
                    f"样本 {sample_token[:16]}...: "
                    f"真值轨迹不足 ({len(gt_traj)} 点)，跳过"
                )
                continue

            record["ground_truth_trajectory"] = gt_traj

            # 获取当前自车状态（用于速度判断）
            ego_state = record.get("ego_state")
            current_speed = 0.0
            if ego_state and isinstance(ego_state, dict):
                current_speed = ego_state.get("speed", 0.0)
            elif hasattr(ego_state, "speed"):
                current_speed = ego_state.speed

            # 推断伪导航语义
            # RouteInfer 接受 ego_pose 格式的 future_poses
            # 将 gt_traj 转换为 ego_pose 格式
            future_poses = _trajectory_to_poses(gt_traj)
            nav_label = route_infer.infer(
                future_poses=future_poses,
                current_speed=current_speed,
            )
            record["ground_truth_behavior"] = nav_label

            gt_success += 1

        except Exception as e:
            logger.warning(f"样本 {sample_token[:16]}...: 真值构建失败: {e}")
            record["ground_truth_trajectory"] = []
            record["ground_truth_behavior"] = ""
            gt_fail += 1

    logger.info(
        f"真值构建完成: 成功 {gt_success}/{len(decisions)}, "
        f"失败 {gt_fail}/{len(decisions)}"
    )

    return decisions, gt_success


def _trajectory_to_poses(trajectory: List[Dict]) -> List[Dict]:
    """将 ego-centric 轨迹点转换为 RouteInfer 可用的 poses 格式。

    由于 RouteInfer 需要全局坐标的 ego_pose 来计算 yaw 变化，
    而 gt_traj 是 ego-centric 坐标，这里用简单的笛卡尔坐标来近似计算。
    第一版使用位移方向来推断行为，这是 demo 级近似。

    注意：这个转换是为了让 RouteInfer 能从 ego-centric 轨迹推断行为。
    yaw 计算使用相邻点方向，仅用于伪行为标签推断。

    Args:
        trajectory: ego-centric 轨迹点列表。

    Returns:
        模拟的 pose 列表（含 translation 字段）。
    """
    poses = []
    for pt in trajectory:
        poses.append({
            "translation": [pt.get("x", 0), pt.get("y", 0), 0],
        })
    return poses


def _prepare_ground_truth(
    decisions: List[Dict[str, Any]],
    config: Config,
    nuscenes_dataroot: Optional[str],
    nuscenes_version: str,
    skip_ground_truth: bool,
    mode_label: str,
) -> List[Dict[str, Any]]:
    """构造（或复用内置）真值轨迹 + 伪行为标签。

    R3 拆出来的辅助：单 mode / 双 mode 都用同一个流程。
    nuScenes 不可用 + 决策中无内置真值时 hard fail。

    Args:
        decisions: 决策结果列表（就地修改：会添加 ground_truth_trajectory/behavior）。
        config: 项目 Config。
        nuscenes_dataroot: 可选覆盖；None 时从 config 读。
        nuscenes_version: nuScenes 版本。
        skip_ground_truth: True 时跳过 nuScenes 构建（仅用于决策已含真值的场景）。
        mode_label: 仅用于日志识别。

    Returns:
        增强后的 decisions（与入参同一对象）。
    """
    eval_cfg = config.get("evaluation", {})
    behavior_cfg = eval_cfg.get("behavior_accuracy", {})
    future_seconds = eval_cfg.get("prediction_horizon_seconds", 3.0)
    nav_to_behavior_map = behavior_cfg.get("nav_to_behavior_map", {})

    if skip_ground_truth:
        return decisions

    if nuscenes_dataroot is None:
        nuscenes_cfg = config.get("data_nuscenes", {}) or {}
        nuscenes_dataroot = nuscenes_cfg.get("dataroot") or config.get("dataroot")

    try:
        decisions, success_n = build_ground_truth_from_nuscenes(
            decisions=decisions,
            nuscenes_dataroot=str(nuscenes_dataroot),
            nuscenes_version=nuscenes_version,
            future_seconds=future_seconds,
            nav_to_behavior_map=nav_to_behavior_map,
        )
        if success_n == 0:
            has_builtin = any(r.get("ground_truth_trajectory") for r in decisions)
            if not has_builtin:
                raise RuntimeError(
                    f"[{mode_label}] 无法从 nuScenes 构建任何真值轨迹，"
                    f"且决策结果中无内置真值。评测终止。\n"
                    f"请确保 nuScenes 数据已正确放置。不允许伪造真值。"
                )
            logger.warning(
                "[%s] nuScenes 真值构建全部失败，回退到决策结果内置真值。",
                mode_label,
            )
    except FileNotFoundError:
        has_builtin = any(r.get("ground_truth_trajectory") for r in decisions)
        if not has_builtin:
            raise RuntimeError(
                f"[{mode_label}] nuScenes 数据不存在，且决策结果中无内置真值。\n"
                f"无法进行评测。请放置 nuScenes 数据或提供含真值的决策文件。\n"
                f"不允许伪造真值。"
            )
        logger.warning(
            "[%s] nuScenes 数据不存在，使用决策结果中的内置真值。",
            mode_label,
        )

    return decisions


def _evaluate_one_mode(
    results: List[Dict[str, Any]],
    mode_label: str,
    evaluator: Evaluator,
) -> Tuple[List[EvalSampleResult], EvalSummary]:
    """对单 mode 的决策结果跑评测，返回 (per-sample 结果, 汇总)。"""
    logger.info("评测模式: %s, 样本数: %d", mode_label, len(results))

    sample_results: List[EvalSampleResult] = []
    for r in results:
        decision_output = r.get("decision_output") or {}
        predicted = decision_output.get("trajectory", [])
        gt = r.get("ground_truth_trajectory", [])
        pred_behavior = decision_output.get("behavior", "")
        gt_behavior = r.get("ground_truth_behavior", "")
        fallback_used = r.get("fallback_used", False)

        eval_result = evaluator.evaluate_sample(
            predicted_trajectory=predicted,
            ground_truth_trajectory=gt,
            predicted_behavior=pred_behavior,
            ground_truth_behavior=gt_behavior,
            sample_token=r.get("sample_token", r.get("frame_id", "")),
            scene_token=r.get("scene_token", ""),
            mode=mode_label,
            scene_id=r.get("scene_id", ""),
            weather_id=r.get("weather_id", ""),
            fallback_used=fallback_used,
        )
        sample_results.append(eval_result)

    summary = evaluator.aggregate_results(sample_results)
    logger.info(
        "%s 评测结果: ADE=%s, FDE=%s, 有效率=%.4f, 行为准确率=%s, fallback=%d",
        mode_label,
        summary.ade_mean, summary.fde_mean,
        summary.valid_trajectory_rate, summary.behavior_accuracy,
        summary.fallback_count,
    )
    return sample_results, summary


def run_eval_pipeline(
    results: List[Dict[str, Any]],
    mode: str = "memory_on",
    config: Optional[Config] = None,
    nuscenes_dataroot: Optional[str] = None,
    nuscenes_version: str = "v1.0-mini",
    skip_ground_truth: bool = False,
    report_dir: Optional[str] = None,
) -> EvalSummary:
    """对单 mode 的决策结果跑评测，生成 CSV/JSONL/Markdown 报告。

    R3 改造：现在签名是 ``(results, mode, ...)`` 单 mode。
    双 mode 对比请用 :func:`run_eval_compare`。

    Args:
        results: 单 mode 的决策结果列表。
        mode: 该结果的 mode 标签（用于报告与日志）。
        config: 项目 Config。
        nuscenes_dataroot: 可选覆盖；None 时读 config。
        nuscenes_version: nuScenes 版本。
        skip_ground_truth: True 时跳过 nuScenes 真值构建。
        report_dir: 可选覆盖；None 时读 config 中
            ``evaluation.output.report_dir``。

    Returns:
        EvalSummary。
    """
    if config is None:
        config = load_config()

    eval_cfg = config.get("evaluation", {}) or {}
    output_cfg = eval_cfg.get("output", {}) or {}

    # ---- 1. 真值构建 ----
    results = _prepare_ground_truth(
        decisions=results,
        config=config,
        nuscenes_dataroot=nuscenes_dataroot,
        nuscenes_version=nuscenes_version,
        skip_ground_truth=skip_ground_truth,
        mode_label=mode,
    )

    # ---- 2. 评测 ----
    evaluator = Evaluator.from_config(eval_cfg)
    sample_results, summary = _evaluate_one_mode(results, mode, evaluator)

    # ---- 3. 写报告（仅本 mode） ----
    report_dir_resolved = report_dir or output_cfg.get(
        "report_dir", "outputs/reports",
    )
    writer = ReportWriter(report_dir=str(report_dir_resolved))
    writer.write_csv({mode: summary})
    writer.write_jsonl(sample_results)
    writer.write_markdown({mode: summary})

    return summary


def run_eval_compare(
    results_by_mode: Dict[str, List[Dict[str, Any]]],
    config: Optional[Config] = None,
    nuscenes_dataroot: Optional[str] = None,
    nuscenes_version: str = "v1.0-mini",
    skip_ground_truth: bool = False,
    report_dir: Optional[str] = None,
) -> Dict[str, EvalSummary]:
    """多 mode 对比评测。报告（CSV/Markdown）会包含所有 mode 列。

    Args:
        results_by_mode: ``{"memory_on": [...], "memory_off": [...]}``，
            支持任意数量的 mode 标签。
        config: 项目 Config。
        nuscenes_dataroot/nuscenes_version/skip_ground_truth/report_dir: 同
            ``run_eval_pipeline``。

    Returns:
        ``{mode: EvalSummary}``。

    Raises:
        ValueError: 传入空字典。
        RuntimeError: 真值构建全部失败且无内置真值。
    """
    if not results_by_mode:
        raise ValueError("results_by_mode 不能为空")
    if config is None:
        config = load_config()

    eval_cfg = config.get("evaluation", {}) or {}
    output_cfg = eval_cfg.get("output", {}) or {}

    evaluator = Evaluator.from_config(eval_cfg)
    summaries: Dict[str, EvalSummary] = {}
    all_sample_results: List[EvalSampleResult] = []

    for mode_label, decisions in results_by_mode.items():
        prepared = _prepare_ground_truth(
            decisions=decisions,
            config=config,
            nuscenes_dataroot=nuscenes_dataroot,
            nuscenes_version=nuscenes_version,
            skip_ground_truth=skip_ground_truth,
            mode_label=mode_label,
        )
        sample_results, summary = _evaluate_one_mode(prepared, mode_label, evaluator)
        summaries[mode_label] = summary
        all_sample_results.extend(sample_results)

    # ---- 写对比报告 ----
    report_dir_resolved = report_dir or output_cfg.get(
        "report_dir", "outputs/reports",
    )
    writer = ReportWriter(report_dir=str(report_dir_resolved))
    writer.write_csv(summaries)
    writer.write_jsonl(all_sample_results)
    writer.write_markdown(summaries)

    return summaries
