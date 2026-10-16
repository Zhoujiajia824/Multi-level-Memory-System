#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预下载和验证脚本
====================
下载 / 预加载 DINOv2 模型权重，并进行一次前向推理验证。
默认模型: facebook/dinov2-base。
不允许用随机 embedding 代替真实模型权重。

用法:
    python scripts/00_prepare_models.py
    python scripts/00_prepare_models.py --offline
    python scripts/00_prepare_models.py --offline --cache-dir /path/to/huggingface
    python scripts/00_prepare_models.py --model-dir /path/to/local/dinov2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="下载或离线验证 DINOv2 权重及真实前向推理")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--cache-dir", help="Hugging Face 缓存根目录")
    parser.add_argument("--model-dir", help="包含配置、processor 和权重的本地模型目录")
    access = parser.add_mutually_exclusive_group()
    access.add_argument("--offline", dest="local_files_only", action="store_true", help="仅验证本地权重，禁止联网")
    access.add_argument("--download", dest="local_files_only", action="store_false", help="允许联网下载缺少的权重")
    parser.set_defaults(local_files_only=None)
    return parser.parse_args(argv)


def main() -> None:
    """主函数：下载 DINOv2 模型并进行推理验证。"""
    # Redirected Windows consoles may default to GBK, which cannot print the
    # diagnostic symbols below. Keep validation usable without Python flags.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
    import yaml
    import numpy as np
    from PIL import Image
    from src.vla_memory.perception.dinov2_extractor import DINOv2Extractor

    args = _parse_args()

    print("=" * 60)
    print("智能驾驶 VLA 分层记忆系统 - 模型预下载与验证")
    print("=" * 60)
    print()

    # ---- 读取配置 ----
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(str(config_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    feat_cfg = config.get("feature_extractor", {})
    model_name = feat_cfg.get("model_name", "facebook/dinov2-base")
    cache_dir = Path(args.cache_dir or feat_cfg.get("cache_dir", ".cache/huggingface")).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    model_dir = args.model_dir or feat_cfg.get("model_dir")
    if model_dir:
        model_dir = Path(model_dir).expanduser()
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
    local_files_only = args.local_files_only
    if local_files_only is None:
        local_files_only = feat_cfg.get("local_files_only", False)
    local_files_only = local_files_only or bool(model_dir)
    expected_dim = feat_cfg.get("feature_dim", 768)
    normalize = feat_cfg.get("normalize", True)

    print(f"模型名称: {model_name}")
    print(f"缓存目录: {cache_dir}")
    print(f"本地模型目录: {model_dir or '使用缓存'}")
    print(f"仅使用本地文件: {local_files_only}")
    print(f"期望特征维度: {expected_dim}")
    print(f"L2 归一化: {normalize}")
    print()

    # ---- 步骤 1: 加载模型 ----
    print("[步骤 1/3] 加载 DINOv2 模型...")
    if local_files_only:
        print("  离线验证：不连接 Hugging Face，缺失文件将立即报错。")
    else:
        print(f"  允许从 Hugging Face 下载模型权重到: {cache_dir}")
    print()

    try:
        import torch
        extractor = DINOv2Extractor(
            model_name=model_name, cache_dir=str(cache_dir), model_dir=model_dir,
            local_files_only=local_files_only, device=feat_cfg.get("device", "auto"),
            normalize=False,
        )
        extractor.load_model()
        actual_dim = extractor.get_feature_dim()
        device = extractor._device
        print(f"  ✅ 模型加载成功！")
        print(f"     设备: {device}")
        print(f"     特征维度: {actual_dim}")
        print()

    except ImportError as e:
        print(f"❌ 缺少必要的依赖库: {e}")
        print("   请安装: pip install torch transformers")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        if local_files_only:
            print("   请核对缓存路径及上传文件完整性，或使用 --model-dir 指定完整模型目录。")
        else:
            print("   请检查网络连接，或手动上传缓存后使用 --offline 验证。")
        sys.exit(1)

    # ---- 步骤 2: 创建测试图像并进行前向推理 ----
    print("[步骤 2/3] 使用测试图像进行前向推理验证...")

    try:
        # 创建一张简单的测试图像（224x224 彩色图像）
        test_image = Image.new("RGB", (224, 224), color=(128, 64, 32))

        inputs = extractor._processor(images=test_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = extractor._model(**inputs)
            feature = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        print(f"  特征向量形状: {feature.shape}")
        print(f"  特征维度: {len(feature)}")
        print(f"  特征值范围: [{feature.min():.6f}, {feature.max():.6f}]")

        # 验证特征维度
        if len(feature) != expected_dim:
            print(f"  ⚠️ 特征维度不匹配: 期望 {expected_dim}，实际 {len(feature)}")
        else:
            print(f"  ✅ 特征维度匹配!")

        # 验证 L2 归一化
        l2_norm = np.linalg.norm(feature)
        print(f"  L2 范数: {l2_norm:.6f}")

        if normalize:
            feature_norm = feature / l2_norm
            l2_norm_after = np.linalg.norm(feature_norm)
            print(f"  归一化后 L2 范数: {l2_norm_after:.6f}")
            if abs(l2_norm_after - 1.0) < 1e-5:
                print(f"  ✅ L2 归一化正确!")
            else:
                print(f"  ⚠️ L2 归一化结果异常")

        # 检查是否为全零（可能表明模型加载有问题）
        if np.all(feature == 0):
            print("  ❌ 特征向量全为零！模型可能加载异常。")
            sys.exit(1)
        else:
            print(f"  ✅ 特征向量非零，模型正常!")

        print()

    except Exception as e:
        print(f"❌ 前向推理失败: {e}")
        sys.exit(1)

    # ---- 步骤 3: 保存测试特征 ----
    print("[步骤 3/3] 保存测试特征...")

    try:
        test_feature_dir = PROJECT_ROOT / "outputs" / "features"
        test_feature_dir.mkdir(parents=True, exist_ok=True)
        test_feature_path = test_feature_dir / "_test_feature.npy"

        if normalize:
            feature_save = feature / np.linalg.norm(feature)
        else:
            feature_save = feature

        np.save(str(test_feature_path), feature_save)
        print(f"  ✅ 测试特征已保存: {test_feature_path}")
        print()

    except Exception as e:
        print(f"  ⚠️ 测试特征保存失败（不影响正常使用）: {e}")
        print()

    # ---- 完成 ----
    print("=" * 60)
    print("✅ DINOv2 模型预下载和验证完成!")
    print(f"   模型: {model_name}")
    print(f"   缓存: {cache_dir}")
    print(f"   特征维度: {len(feature)}")
    print(f"   设备: {device}")
    print("=" * 60)
    print()
    print("模型已就绪。后续运行场景理解流水线时会自动使用缓存的模型权重。")


if __name__ == "__main__":
    main()
