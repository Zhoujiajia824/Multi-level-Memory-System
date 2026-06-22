#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键帧提取脚本
==============
从 nuScenes 数据集中提取 1Hz 关键帧，并对齐自车状态、历史轨迹和伪导航语义。
输出 keyframes.jsonl，每行包含完整的关键帧数据。

输入:
    data/nuscenes/processed/nuscenes_index.jsonl（可选，如不存在则直接从数据集读取）

输出:
    data/nuscenes/processed/keyframes.jsonl

用法:
    python scripts/02_extract_keyframes.py [选项]

选项:
    --config          配置文件路径
    --max-scenes      最多处理的场景数
    --max-frames      每个场景最多返回的关键帧数
    --keyframe-step   关键帧采样步长（默认 2，即 2Hz -> 1Hz）
    --camera-name     摄像头名称
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="关键帧提取：从 nuScenes 采样 1Hz 关键帧，补充状态和轨迹",
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--max-scenes", type=int, default=None, help="最多处理场景数")
    parser.add_argument("--max-frames", type=int, default=None, help="每场景最大关键帧数")
    parser.add_argument("--keyframe-step", type=int, default=None, help="关键帧采样步长")
    parser.add_argument("--camera-name", type=str, default=None, help="摄像头名称")
    return parser.parse_args()


def main() -> None:
    """主函数：加载 -> 采样 -> 构建 -> 输出。"""
    from src.vla_memory.common.config import load_config
    from src.vla_memory.common.logging_utils import setup_logger
    from src.vla_memory.common.path_utils import ensure_dir
    from src.vla_memory.data.nuscenes_adapter import NuScenesAdapter
    from src.vla_memory.data.ego_state_builder import EgoStateBuilder
    from src.vla_memory.data.trajectory_builder import TrajectoryBuilder
    from src.vla_memory.data.route_infer import RouteInfer
    from src.vla_memory.keyframes.nuscenes_keyframe_sampler import NuScenesKeyframeSampler

    args = parse_args()
    config = load_config()

    # CLI 覆盖
    dataroot = str(config.get("dataroot", "data/nuscenes/raw"))
    version = config.get("version", "v1.0-mini")
    camera_name = args.camera_name or config.get("camera_name", "CAM_FRONT")
    keyframe_step = args.keyframe_step or config.get_nested("keyframe", "step", default=2)
    history_seconds = config.get("history_seconds", 5.0)
    future_seconds = config.get("future_seconds", 3.0)
    max_scenes = args.max_scenes
    max_frames = args.max_frames

    # 输出目录
    processed_dir = PROJECT_ROOT / "data" / "nuscenes" / "processed"
    ensure_dir(processed_dir)

    # 日志
    log_level = config.get_nested("logging", "level", default="INFO")
    log_dir = config.get("log_dir", str(PROJECT_ROOT / "outputs" / "logs"))
    logger = setup_logger(
        name="02_extract_keyframes",
        level=log_level,
        log_dir=Path(log_dir) if log_dir else None,
    )

    logger.info("=" * 60)
    logger.info("关键帧提取流水线")
    logger.info(f"  dataroot       = {dataroot}")
    logger.info(f"  version        = {version}")
    logger.info(f"  camera_name    = {camera_name}")
    logger.info(f"  keyframe_step  = {keyframe_step}")
    logger.info(f"  history_seconds= {history_seconds}")
    logger.info(f"  future_seconds = {future_seconds}")
    logger.info(f"  max_scenes     = {max_scenes or '全部'}")
    logger.info(f"  max_frames     = {max_frames or '全部'}")
    logger.info("=" * 60)

    # ---- 1. 加载数据集 ----
    adapter = NuScenesAdapter(dataroot=dataroot, version=version, camera_name=camera_name)
    try:
        adapter.load()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "\n请将 nuScenes 数据解压到 data/nuscenes/raw，"
            "使其包含 samples、sweeps、maps、v1.0-mini。\n"
            "不允许使用假数据。"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        sys.exit(1)

    # ---- 2. 关键帧采样 ----
    sampler = NuScenesKeyframeSampler(step=keyframe_step, camera_name=camera_name)
    scene_tokens = adapter.list_scenes()
    if max_scenes is not None:
        scene_tokens = scene_tokens[:max_scenes]

    # ---- 3. 初始化构建器 ----
    ego_builder = EgoStateBuilder()
    traj_builder = TrajectoryBuilder()
    route_infer = RouteInfer()

    # ---- 4. 遍历场景 ----
    output_path = processed_dir / "keyframes.jsonl"
    total_keyframes = 0
    total_scenes = len(scene_tokens)

    with open(str(output_path), "w", encoding="utf-8") as f_out:
        for scene_idx, scene_token in enumerate(scene_tokens):
            scene_desc = adapter.get_scene_description(scene_token)
            logger.info(
                f"  场景 [{scene_idx + 1}/{total_scenes}]: "
                f"{scene_token[:8]}... ({scene_desc})"
            )

            # 采样关键帧
            keyframes = sampler.sample_scene(
                adapter, scene_token,
                max_frames=max_frames,
            )

            frame_count = 0
            for kf in keyframes:
                try:
                    # 构建 ego_state
                    ego_state = adapter.get_ego_pose(kf.sample_token)

                    # 构建历史轨迹
                    history_traj = adapter.get_history_trajectory(
                        kf.sample_token, history_seconds=history_seconds,
                    )

                    # 构建未来真值轨迹（用于评测）
                    future_traj = adapter.get_future_ego_trajectory(
                        kf.sample_token, future_seconds=future_seconds,
                    )

                    # 推断伪导航语义
                    # 获取未来 ego_pose 列表
                    future_poses = _get_future_poses(adapter, kf.sample_token, future_seconds)
                    nav_instruction = route_infer.infer(
                        future_poses=future_poses,
                        current_speed=ego_state.speed,
                    )

                except Exception as e:
                    logger.warning(
                        f"  跳过帧 {kf.sample_token[:8]}... 状态构建失败: {e}"
                    )
                    ego_state_dict = {}
                    history_traj_dicts = []
                    future_traj_dicts = []
                    nav_instruction = "unknown"

                else:
                    ego_state_dict = ego_state.to_dict()
                    history_traj_dicts = history_traj if isinstance(history_traj, list) else history_traj.to_list()
                    future_traj_dicts = future_traj if isinstance(future_traj, list) else future_traj.to_list()

                # 构建输出记录
                record = {
                    "frame_meta": kf.to_index_dict(),
                    "ego_state": ego_state_dict,
                    "history_trajectory": history_traj_dicts,
                    "nav_instruction": nav_instruction,
                    "ground_truth_trajectory": future_traj_dicts,
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                frame_count += 1

            total_keyframes += frame_count
            logger.info(f"    {frame_count} 个关键帧已写入")

    logger.info("=" * 60)
    logger.info("✅ 关键帧提取完成!")
    logger.info(f"  场景数: {total_scenes}")
    logger.info(f"  总关键帧: {total_keyframes}")
    logger.info(f"  采样率: ~{sampler.get_sample_rate():.1f} Hz")
    logger.info(f"  输出文件: {output_path}")
    logger.info("=" * 60)


def _get_future_poses(adapter, sample_token: str, future_seconds: float = 3.0) -> list:
    """获取未来 ego_pose 原始字典列表。

    Args:
        adapter: NuScenesAdapter 实例。
        sample_token: 当前样本 token。
        future_seconds: 未来时间窗口。

    Returns:
        未来 ego_pose 字典列表。
    """
    poses = []
    current_ts = adapter._get_sample_ego_pose(sample_token)["timestamp"]
    token = sample_token
    while token:
        sample = adapter.nusc.get("sample", token)
        next_token = sample.get("next", "")
        if not next_token:
            break
        next_pose = adapter._get_sample_ego_pose(next_token)
        dt = (next_pose["timestamp"] - current_ts) / 1e6
        if dt > future_seconds:
            break
        poses.append(next_pose)
        token = next_token
    return poses


if __name__ == "__main__":
    main()
