"""Demo 主流水线（R2 重构：逐帧在线循环）
==========================================
单次运行只跑一种 mode（``memory_on`` 或 ``memory_off``），把所有 keyframes
按时间顺序丢给 OnlineDrivingLoop。决策结果以 jsonl append 形式落盘，
评测作为独立步骤跑 ``scripts/06_run_evaluation.py``。

旧的 7 步批处理瀑布（scene_pipeline / memory_pipeline / decision_pipeline）
已删除，原因：批处理会让中期记忆在决策前包含所有未来帧 → 严重 data leakage。

``enrich_keyframes_with_state`` 仍然保留作为 per-frame 的数据准备函数
（自车状态 / 历史轨迹 / 未来真值轨迹 / 伪导航语义），它本身不引入 batch 问题。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.vla_memory.common.config import Config, load_config
from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.data.ego_state_builder import EgoStateBuilder
from src.vla_memory.data.nuscenes_adapter import NuScenesAdapter
from src.vla_memory.data.route_infer import RouteInfer
from src.vla_memory.data.trajectory_builder import TrajectoryBuilder
from src.vla_memory.pipeline.online_loop import OnlineDrivingLoop, default_output_path
from src.vla_memory.pipeline.prepare_nuscenes import run_prepare_nuscenes

logger = get_logger("full_demo")


# ============================================================
# 每帧数据准备（保留：与 batch 无关）
# ============================================================


def enrich_keyframes_with_state(
    adapter: NuScenesAdapter,
    keyframe_index: Dict[str, list],
    config: Config,
) -> List[Dict[str, Any]]:
    """为关键帧补充自车状态、历史轨迹、未来真值轨迹、导航语义。

    这是 per-frame 的纯数据准备，可以提前一次性算好（不存在 data leakage
    问题，因为这些信息全部来自当前帧的过去或全局已知数据）。

    Args:
        adapter: 已加载的 NuScenesAdapter（含 can_bus_loader 可选）。
        keyframe_index: ``{scene_token: [FrameMeta, ...]}``，由
            ``run_prepare_nuscenes`` 返回。
        config: 项目 Config。

    Returns:
        ``List[Dict[str, Any]]``：每条 keyframe 含 sample_token / scene_token /
        timestamp / image_path / ego_state / history_trajectory /
        ground_truth_trajectory / nav_instruction 字段。
    """
    history_seconds = config.get("history_seconds", 5.0)
    future_seconds = config.get("future_seconds", 3.0)

    # 感知输入模式（single_front / surround_mosaic）+ oracle 感知开关
    perception_mode = config.get_nested("perception", "mode", default="single_front")
    perception_cameras = config.get_nested("perception", "cameras", default=None) or []
    oracle_enabled = bool(config.get_nested("perception", "oracle_objects", default=False))
    mosaic_cell_w = int(config.get_nested("perception", "mosaic", "cell_width", default=480))
    mosaic_cell_h = int(config.get_nested("perception", "mosaic", "cell_height", default=270))
    mosaic_label = bool(config.get_nested("perception", "mosaic", "label_subimages", default=True))
    oracle_cfg = config.get_nested("perception", "oracle", default={}) or {}
    if perception_mode == "surround_mosaic":
        logger.info(
            "感知输入模式: surround_mosaic（六视角 2x3 拼接替代前视图）, oracle_objects=%s",
            oracle_enabled,
        )
        from src.vla_memory.perception.surround_mosaic import build_surround_mosaic

    ego_builder = EgoStateBuilder()
    traj_builder = TrajectoryBuilder()
    route_infer = RouteInfer()

    all_keyframes: List[Dict[str, Any]] = []

    for scene_token, frames in keyframe_index.items():
        scene_samples = adapter._get_scene_samples(scene_token)
        all_poses = [
            adapter._get_sample_ego_pose(s["token"]) for s in scene_samples
        ]
        sorted_poses = sorted(all_poses, key=lambda p: p["timestamp"])

        for frame in frames:
            sample_token = frame.sample_token
            image_path = frame.image_path

            current_pose = adapter._get_sample_ego_pose(sample_token)

            # 在排序后的 pose 列表中找当前帧索引
            current_idx = None
            for idx, p in enumerate(sorted_poses):
                if p["timestamp"] == current_pose["timestamp"]:
                    current_idx = idx
                    break

            prev_pose = (
                sorted_poses[current_idx - 1]
                if current_idx and current_idx > 0 else None
            )
            prev_prev_pose = (
                sorted_poses[current_idx - 2]
                if current_idx and current_idx > 1 else None
            )

            # 自车状态：CAN bus 优先（若 adapter 注入了 loader），否则差分
            scene_name = None
            if adapter.can_bus_loader is not None:
                try:
                    sample_obj = adapter.nusc.get("sample", sample_token)
                    scene_obj = adapter.nusc.get("scene", sample_obj["scene_token"])
                    scene_name = scene_obj["name"]
                except Exception:
                    scene_name = None
            ego_state = ego_builder.build(
                current_pose,
                scene_name=scene_name,
                can_bus_loader=adapter.can_bus_loader,
                prev_pose=prev_pose,
                prev_prev_pose=prev_prev_pose,
            )

            # 历史轨迹（容忍空轨迹）
            past_poses = sorted_poses[:current_idx] if current_idx else []
            history_traj = traj_builder.build_history_trajectory(
                current_pose=current_pose, past_poses=past_poses,
                history_seconds=history_seconds,
            )
            history_list = history_traj.to_list()

            # 未来真值轨迹（容忍空）
            future_poses = (
                sorted_poses[current_idx + 1:] if current_idx is not None else []
            )
            gt_trajectory = traj_builder.build_future_trajectory(
                current_pose=current_pose, future_poses=future_poses,
                future_seconds=future_seconds,
            )
            gt_list = gt_trajectory.to_list()

            # 伪导航语义
            nav_instruction = route_infer.infer(
                future_poses=future_poses,
                current_speed=ego_state.speed,
            )

            # 感知输入：surround_mosaic 模式拼六视角图替代 image_path（下游全自适应）
            actual_image_path = image_path
            image_paths_raw = list(frame.image_paths) if frame.image_paths else []
            if perception_mode == "surround_mosaic" and image_paths_raw:
                mosaic_out = str(config.root / "outputs" / "mosaic" / f"{sample_token}.jpg")
                try:
                    actual_image_path = build_surround_mosaic(
                        image_paths_raw, perception_cameras,
                        cell_width=mosaic_cell_w, cell_height=mosaic_cell_h,
                        label_subimages=mosaic_label, out_path=mosaic_out,
                    )
                except Exception as e:
                    logger.warning("mosaic 拼接失败 sample=%s，回退前视图: %s", sample_token[:8], e)
                    actual_image_path = image_path

            # oracle 感知对象（nuScenes GT 投影，非模型预测；is_oracle=True）
            perception_objects: List[Dict[str, Any]] = []
            if oracle_enabled:
                try:
                    perception_objects = adapter.get_perception_objects(sample_token, oracle_cfg)
                except Exception as e:
                    logger.warning("oracle perception 生成失败 sample=%s: %s", sample_token[:8], e)

            all_keyframes.append({
                "sample_token": sample_token,
                "scene_token": scene_token,
                "scene_name": scene_name,  # Phase 1 新增：来源场景名（如 scene-0061），供中期记忆 source_scene_name 使用；can_bus 关闭时为 None
                "timestamp": current_pose["timestamp"],
                "image_path": actual_image_path,
                "image_paths_raw": image_paths_raw,
                "perception_mode": perception_mode,
                "perception_objects": perception_objects,
                "ego_state": ego_state.to_ego_centric(),
                "history_trajectory": history_list,
                "ground_truth_trajectory": gt_list,
                "nav_instruction": nav_instruction,
            })

    logger.info("关键帧状态补充完成: %d 个关键帧", len(all_keyframes))
    return all_keyframes


# ============================================================
# 主 demo 入口（R2 薄壳）
# ============================================================


def run_full_demo(
    config: Optional[Config] = None,
    mode: str = "memory_on",
    resume: bool = True,
    output_jsonl_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """运行完整 demo 流水线（逐帧在线循环）。

    步骤：
    1. ``run_prepare_nuscenes`` 加载 nuScenes + 采样关键帧。
    2. ``enrich_keyframes_with_state`` 给每帧补 ego_state / history / nav。
    3. ``OnlineDrivingLoop.run`` 逐帧：感知 → 检索 → 决策 → 更新记忆 → 写 jsonl。
    4. 不在这里跑评测——评测请用 ``scripts/06_run_evaluation.py --decisions <jsonl>``。

    Args:
        config: 项目 Config。None 时自动 load_config。
        mode: ``"memory_on"`` 或 ``"memory_off"``。
        resume: True 时启动扫描已写 jsonl 跳过已处理 sample_token。
        output_jsonl_path: 决策结果输出路径，None 时按
            ``outputs/decisions_<mode>_<run_id>.jsonl`` 自动命名。

    Returns:
        ``{"records": [...], "output_path": str, "mode": str}``。

    Raises:
        FileNotFoundError: nuScenes 数据集不存在。
        EnvironmentError: VLM API Key 未设置。
        RuntimeError: 场景理解 / 决策 hard fail。
    """
    if config is None:
        config = load_config()

    config.ensure_output_dirs()

    logger.info("=" * 60)
    logger.info("智能驾驶 VLA 分层记忆系统 - 在线循环 Demo (mode=%s)", mode)
    logger.info(config.summary())
    logger.info("=" * 60)

    dataroot = config.get("dataroot")
    if not Path(dataroot).exists():
        logger.error(
            "nuScenes 数据集目录不存在！\n"
            "请将 nuScenes 数据集放置到: %s\n"
            "Demo 无法在无数据集的情况下运行。", dataroot,
        )
        return {"error": "nuScenes 数据集不存在，Demo 已停止。"}

    # ---- 步骤 1：准备数据 ----
    logger.info("[步骤 1/3] 准备 nuScenes 数据...")
    prep_result = run_prepare_nuscenes(config)
    adapter = prep_result["adapter"]
    keyframe_index = prep_result["keyframe_index"]
    logger.info(
        "  数据准备完成: %d 个场景, %d 个关键帧",
        len(keyframe_index), prep_result["total_keyframes"],
    )

    # ---- 步骤 2：补充 per-frame 状态 ----
    logger.info("[步骤 2/3] 为关键帧补充自车状态、历史轨迹、导航语义...")
    keyframes = enrich_keyframes_with_state(adapter, keyframe_index, config)
    logger.info("  增强完成: %d 个关键帧", len(keyframes))

    # ---- 步骤 3：在线循环（核心） ----
    output_path = (
        Path(output_jsonl_path) if output_jsonl_path
        else default_output_path(config, mode)
    )
    logger.info(
        "[步骤 3/3] 在线循环 (mode=%s, resume=%s)，输出: %s",
        mode, resume, output_path,
    )

    loop = OnlineDrivingLoop(
        config=config,
        mode=mode,
        output_jsonl_path=output_path,
        resume=resume,
    )
    loop.setup()
    try:
        records = loop.run(keyframes)
    finally:
        loop.close()

    logger.info("=" * 60)
    logger.info(
        "Demo 运行结束！mode=%s, 写入 %d 帧决策结果 -> %s",
        mode, len(records), output_path,
    )
    logger.info(
        "评测请跑：python scripts/06_run_evaluation.py --decisions %s",
        output_path,
    )
    logger.info("=" * 60)

    return {
        "records": records,
        "output_path": str(output_path),
        "mode": mode,
    }
