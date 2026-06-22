#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nuScenes 索引准备脚本
====================
加载 nuScenes 数据集，遍历指定数量 scene，输出 nuscenes_index.jsonl。
每行包含 frame_id、scene_token、sample_token、timestamp、image_path、ego_state。
如果数据集不存在，报错退出。

用法:
    python scripts/01_prepare_nuscenes_index.py [选项]

选项:
    --config         配置文件路径（可选，默认加载 config/ 下所有 YAML）
    --max-scenes     最多处理的场景数（默认全部）
    --version        nuScenes 数据集版本（默认 v1.0-mini）
    --dataroot       nuScenes 数据集根目录（默认 data/nuscenes/raw）
    --camera-name    摄像头名称（默认 CAM_FRONT）
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
        description="nuScenes 索引准备脚本：加载数据集并生成 nuscenes_index.jsonl",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径（可选）",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="最多处理的场景数（默认全部）",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="nuScenes 数据集版本（默认从配置读取）",
    )
    parser.add_argument(
        "--dataroot", type=str, default=None,
        help="nuScenes 数据集根目录（默认从配置读取）",
    )
    parser.add_argument(
        "--camera-name", type=str, default=None,
        help="摄像头名称（默认从配置读取）",
    )
    return parser.parse_args()


def main() -> None:
    """主函数：加载数据集、遍历场景、生成索引文件。"""
    import yaml
    from src.vla_memory.common.config import load_config
    from src.vla_memory.common.logging_utils import setup_logger, get_logger
    from src.vla_memory.common.path_utils import ensure_dir
    from src.vla_memory.data.nuscenes_adapter import NuScenesAdapter

    args = parse_args()

    # 加载配置
    config = load_config()

    # CLI 参数覆盖配置
    dataroot = args.dataroot or str(config.get("dataroot", "data/nuscenes/raw"))
    version = args.version or config.get("version", "v1.0-mini")
    camera_name = args.camera_name or config.get("camera_name", "CAM_FRONT")
    max_scenes = args.max_scenes

    # 输出目录
    output_dir = config.get_path("output_dir") if hasattr(config, 'get_path') else PROJECT_ROOT / "outputs"
    processed_dir = PROJECT_ROOT / "data" / "nuscenes" / "processed"
    ensure_dir(processed_dir)
    ensure_dir(output_dir)

    # 配置日志
    log_level = config.get_nested("logging", "level", default="INFO") if hasattr(config, 'get_nested') else "INFO"
    log_dir = config.get("log_dir", str(PROJECT_ROOT / "outputs" / "logs"))
    logger = setup_logger(
        name="01_prepare_nuscenes_index",
        level=log_level,
        log_dir=Path(log_dir) if log_dir else None,
    )

    logger.info("=" * 60)
    logger.info("nuScenes 索引准备")
    logger.info(f"  dataroot    = {dataroot}")
    logger.info(f"  version     = {version}")
    logger.info(f"  camera_name = {camera_name}")
    logger.info(f"  max_scenes  = {max_scenes or '全部'}")
    logger.info("=" * 60)

    # 初始化适配器并加载数据集
    adapter = NuScenesAdapter(
        dataroot=dataroot,
        version=version,
        camera_name=camera_name,
    )

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
    except ImportError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        sys.exit(1)

    # 遍历场景
    scene_tokens = adapter.list_scenes()
    if max_scenes is not None:
        scene_tokens = scene_tokens[:max_scenes]
    logger.info(f"待处理场景数: {len(scene_tokens)}")

    # 输出文件路径
    output_path = processed_dir / "nuscenes_index.jsonl"
    total_frames = 0

    with open(str(output_path), "w", encoding="utf-8") as f:
        for scene_idx, scene_token in enumerate(scene_tokens):
            scene_desc = adapter.get_scene_description(scene_token)
            logger.info(
                f"  场景 [{scene_idx + 1}/{len(scene_tokens)}]: "
                f"{scene_token[:8]}... ({scene_desc})"
            )

            frame_count = 0
            for frame_meta in adapter.iter_frames(scene_token):
                # 获取 ego_state
                try:
                    ego_state = adapter.get_ego_pose(frame_meta.sample_token)
                    ego_dict = ego_state.to_dict()
                except Exception as e:
                    logger.warning(
                        f"  跳过帧 {frame_meta.sample_token[:8]}... ego_state 获取失败: {e}"
                    )
                    ego_dict = {}

                # 构建索引行
                record = {
                    "frame_id": frame_meta.frame_id,
                    "scene_token": frame_meta.scene_token,
                    "sample_token": frame_meta.sample_token,
                    "timestamp": frame_meta.timestamp,
                    "image_path": frame_meta.image_path,
                    "ego_state": ego_dict,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                frame_count += 1

            total_frames += frame_count
            logger.info(f"    {frame_count} 帧已写入索引")

    logger.info("=" * 60)
    logger.info(f"✅ nuScenes 索引准备完成!")
    logger.info(f"  场景数: {len(scene_tokens)}")
    logger.info(f"  总帧数: {total_frames}")
    logger.info(f"  索引文件: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
