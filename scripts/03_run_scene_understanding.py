#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场景理解脚本
=============
对 keyframes.jsonl 中每个关键帧图像执行：
1. 真实 DINOv2 embedding 提取
2. 真实 VLM API 场景理解
3. JSON 解析和字段校验
4. 输出 scene_understanding.jsonl

不允许 --mock-vlm，不允许 --mock-feature。

用法:
    python scripts/03_run_scene_understanding.py [选项]

选项:
    --config       配置文件路径
    --max-frames   最多处理的关键帧数
    --resume       断点续传：跳过已处理的 sample_token
    --force        强制重新处理所有帧（覆盖已有结果）
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
        description="场景理解：DINOv2 特征提取 + VLM 场景理解",
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理帧数")
    parser.add_argument("--resume", action="store_true", help="断点续传：跳过已处理帧")
    parser.add_argument("--force", action="store_true", help="强制重新处理所有帧")
    return parser.parse_args()


def main() -> None:
    """主函数。"""
    from src.vla_memory.common.config import load_config
    from src.vla_memory.common.logging_utils import setup_logger
    from src.vla_memory.common.path_utils import ensure_dir
    from src.vla_memory.perception.dinov2_extractor import DINOv2Extractor
    from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient
    from src.vla_memory.perception.scene_understanding import SceneUnderstandingPipeline

    args = parse_args()
    config = load_config()
    config.ensure_output_dirs()

    # 日志
    log_level = config.get_nested("logging", "level", default="INFO")
    log_dir = config.get("log_dir", str(PROJECT_ROOT / "outputs" / "logs"))
    logger = setup_logger(
        name="03_scene_understanding",
        level=log_level,
        log_dir=Path(log_dir) if log_dir else None,
    )

    logger.info("=" * 60)
    logger.info("场景理解流水线（DINOv2 + VLM）")
    logger.info("=" * 60)

    # ---- 1. 检查 API Key ----
    scene_vlm_cfg = config.data.get("scene_understanding", {})
    api_key_env = scene_vlm_cfg.get("api_key_env", "DASHSCOPE_API_KEY")

    vlm_client = OpenAICompatibleVLMClient(
        provider=scene_vlm_cfg.get("provider", "qwen"),
        api_key_env=api_key_env,
        base_url=scene_vlm_cfg.get("base_url", ""),
        model_name=scene_vlm_cfg.get("model_name", "qwen-vl-max"),
        timeout=scene_vlm_cfg.get("timeout", 60),
        max_tokens=scene_vlm_cfg.get("max_tokens", 2048),
        temperature=scene_vlm_cfg.get("temperature", 0.1),
        retry_times=scene_vlm_cfg.get("retry_times", 3),
        retry_interval_seconds=scene_vlm_cfg.get("retry_interval_seconds", 5),
        system_prompt=scene_vlm_cfg.get("system_prompt", ""),
    )

    try:
        vlm_client.check_api_key()
    except EnvironmentError as e:
        logger.error(str(e))
        sys.exit(1)

    # ---- 2. 加载 DINOv2 ----
    feat_cfg = config.data.get("feature_extractor", {})
    extractor = DINOv2Extractor(
        model_name=feat_cfg.get("model_name", "facebook/dinov2-base"),
        cache_dir=feat_cfg.get("cache_dir", ".cache/huggingface"),
        device=config.get("device", "auto"),
        normalize=feat_cfg.get("normalize", True),
    )

    try:
        extractor.load_model()
    except RuntimeError as e:
        logger.error(str(e))
        logger.error("请先运行 python scripts/00_prepare_models.py 下载模型权重。")
        sys.exit(1)

    # ---- 3. 读取 keyframes.jsonl ----
    keyframes_path = PROJECT_ROOT / "data" / "nuscenes" / "processed" / "keyframes.jsonl"
    if not keyframes_path.exists():
        logger.error(
            f"关键帧文件不存在: {keyframes_path}\n"
            "请先运行: python scripts/02_extract_keyframes.py"
        )
        sys.exit(1)

    keyframes = []
    with open(str(keyframes_path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keyframes.append(json.loads(line))

    logger.info(f"加载关键帧: {len(keyframes)} 个")

    if args.max_frames:
        keyframes = keyframes[: args.max_frames]
        logger.info(f"限制处理帧数: {args.max_frames}")

    # ---- 4. 断点续传 ----
    processed_dir = PROJECT_ROOT / "data" / "nuscenes" / "processed"
    output_path = processed_dir / "scene_understanding.jsonl"
    done_tokens: set[str] = set()

    if args.resume and output_path.exists() and not args.force:
        with open(str(output_path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_tokens.add(rec.get("sample_token", ""))
        logger.info(f"断点续传: 已处理 {len(done_tokens)} 帧，跳过")

    if args.force and output_path.exists():
        logger.info("--force 模式: 重新处理所有帧")

    # ---- 5. 初始化流水线 ----
    feature_dir = config.get("feature_dir", str(PROJECT_ROOT / "outputs" / "features"))
    ensure_dir(Path(feature_dir))

    pipeline = SceneUnderstandingPipeline(
        feature_extractor=extractor,
        vlm_client=vlm_client,
        feature_save_dir=feature_dir,
        vlm_retry_times=scene_vlm_cfg.get("retry_times", 2),
    )

    # ---- 6. 逐帧处理 ----
    success_count = 0
    fail_count = 0
    skip_count = 0

    # 追加模式写入（resume 时追加新结果）
    mode = "a" if (args.resume and done_tokens and not args.force) else "w"

    with open(str(output_path), mode, encoding="utf-8") as f_out:
        for i, kf in enumerate(keyframes):
            frame_meta = kf.get("frame_meta", kf)
            sample_token = frame_meta.get("sample_token", f"unknown_{i}")
            image_path = frame_meta.get("image_path", "")

            # 断点续传：跳过已处理帧
            if args.resume and not args.force and sample_token in done_tokens:
                skip_count += 1
                continue

            logger.info(
                f"[{i + 1}/{len(keyframes)}] 处理: {sample_token[:16]}... "
                f"(成功={success_count}, 失败={fail_count})"
            )

            if not image_path or not Path(image_path).exists():
                logger.warning(f"图像不存在，跳过: {image_path}")
                fail_count += 1
                continue

            try:
                result = pipeline.process_frame(sample_token, image_path)
            except EnvironmentError as e:
                logger.error(f"API Key 错误: {e}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"处理异常: {e}")
                result = None

            if result is not None:
                # 写入完整记录
                record = {
                    "sample_token": sample_token,
                    "frame_meta": frame_meta,
                    "image_feature_path": result.get("image_feature_path"),
                    "scene_understanding": result.get("scene_understanding"),
                    "ego_state": kf.get("ego_state"),
                    "history_trajectory": kf.get("history_trajectory"),
                    "nav_instruction": kf.get("nav_instruction"),
                    "ground_truth_trajectory": kf.get("ground_truth_trajectory"),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()  # 实时写入，支持断点续传
                success_count += 1
            else:
                fail_count += 1

    logger.info("=" * 60)
    logger.info("✅ 场景理解流水线完成!")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  跳过(续传): {skip_count}")
    logger.info(f"  输出: {output_path}")
    logger.info(f"  特征: {feature_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
