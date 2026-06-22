#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境检查脚本
============
检查当前运行环境是否满足智能驾驶 VLA 分层记忆系统 demo 的要求。
包括：conda 环境、Python 版本、依赖库、API Key、模型权重、数据集。

用法:
    python scripts/00_check_environment.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class EnvironmentChecker:
    """环境检查器。

    检查所有依赖和环境配置，输出详细的中文报告。
    """

    def __init__(self):
        self.results = []  # 检查结果列表: (名称, 状态, 详情)
        self.pass_count = 0
        self.warn_count = 0
        self.fail_count = 0

    def _record(self, name: str, status: str, detail: str) -> None:
        """记录一条检查结果。"""
        self.results.append((name, status, detail))
        if status == "PASS":
            self.pass_count += 1
        elif status == "WARN":
            self.warn_count += 1
        else:
            self.fail_count += 1

    def check_conda_environment(self) -> None:
        """检查当前是否处于 conda 环境，环境名是否为 mulmem。"""
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        conda_prefix = os.environ.get("CONDA_PREFIX", "")

        if not conda_env:
            self._record(
                "Conda 环境",
                "WARN",
                "未检测到 conda 环境。建议使用 conda 环境 'mulmem' 运行本项目。",
            )
            return

        if conda_env == "mulmem":
            self._record(
                "Conda 环境",
                "PASS",
                f"当前 conda 环境: {conda_env} (路径: {conda_prefix})",
            )
        else:
            self._record(
                "Conda 环境",
                "WARN",
                f"当前 conda 环境名为 '{conda_env}'，推荐使用 'mulmem'。"
                f"路径: {conda_prefix}",
            )

    def check_python_version(self) -> None:
        """检查 Python 版本。"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor == 12:
            self._record("Python 版本", "PASS", f"Python {version_str}")
        elif version.major == 3 and version.minor >= 10:
            self._record(
                "Python 版本",
                "WARN",
                f"Python {version_str}，推荐使用 Python 3.12",
            )
        else:
            self._record(
                "Python 版本",
                "FAIL",
                f"Python {version_str} 不兼容，需要 Python >= 3.10，推荐 3.12",
            )

    def _check_import(self, module_name: str, display_name: str) -> None:
        """检查一个 Python 包是否可以导入。"""
        try:
            __import__(module_name)
            self._record(display_name, "PASS", f"{module_name} 导入成功")
        except ImportError as e:
            self._record(
                display_name,
                "FAIL",
                f"{module_name} 导入失败: {e}\n"
                f"请安装: pip install {module_name}",
            )

    def check_torch(self) -> None:
        """检查 PyTorch 是否安装。"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                self._record(
                    "PyTorch",
                    "PASS",
                    f"PyTorch {torch.__version__}, CUDA 可用: {device_name}",
                )
            else:
                self._record(
                    "PyTorch",
                    "PASS",
                    f"PyTorch {torch.__version__}, CUDA 不可用（将使用 CPU）",
                )
        except ImportError:
            self._record(
                "PyTorch",
                "FAIL",
                "PyTorch 未安装！\n"
                "请安装: pip install torch torchvision\n"
                "或参考 PyTorch 官网选择适合的 CUDA 版本。",
            )

    def check_transformers(self) -> None:
        """检查 transformers 是否安装。"""
        try:
            import transformers
            self._record(
                "transformers",
                "PASS",
                f"transformers {transformers.__version__}",
            )
        except ImportError:
            self._record(
                "transformers",
                "FAIL",
                "transformers 未安装！\n请安装: pip install transformers",
            )

    def check_openai(self) -> None:
        """检查 openai SDK 是否安装。"""
        try:
            import openai
            self._record("openai SDK", "PASS", f"openai {openai.__version__}")
        except ImportError:
            self._record(
                "openai SDK",
                "FAIL",
                "openai SDK 未安装！\n请安装: pip install openai",
            )

    def check_nuscenes_devkit(self) -> None:
        """检查 nuscenes-devkit 是否安装。"""
        try:
            import nuscenes
            self._record(
                "nuscenes-devkit",
                "PASS",
                f"nuscenes-devkit 已安装",
            )
        except ImportError:
            self._record(
                "nuscenes-devkit",
                "FAIL",
                "nuscenes-devkit 未安装！\n"
                "请安装: pip install nuscenes-devkit\n"
                "注意: nuscenes-devkit 安装可能需要额外依赖。",
            )

    def check_faiss(self) -> None:
        """检查 FAISS 是否安装。"""
        try:
            import faiss
            self._record(
                "FAISS",
                "PASS",
                f"faiss 已安装（版本信息不可用，但导入成功）",
            )
        except ImportError:
            self._record(
                "FAISS",
                "FAIL",
                "FAISS 未安装！中期记忆向量检索必须使用 FAISS，不允许降级到 numpy。\n"
                "Windows 推荐通过 conda-forge 安装:\n"
                "  conda install -c conda-forge faiss-cpu\n"
                "或通过 pip 安装:\n"
                "  pip install faiss-cpu",
            )

    def check_api_key(self) -> None:
        """检查 VLM API Key 环境变量是否存在。"""
        # 从配置文件读取环境变量名
        import yaml

        api_config_path = PROJECT_ROOT / "config" / "api_models.yaml"
        if api_config_path.exists():
            with open(str(api_config_path), "r", encoding="utf-8") as f:
                api_config = yaml.safe_load(f)

            scene_cfg = api_config.get("scene_understanding", {})
            api_key_env = scene_cfg.get("api_key_env", "DASHSCOPE_API_KEY")
        else:
            api_key_env = "DASHSCOPE_API_KEY"

        api_key = os.environ.get(api_key_env, "")
        if api_key:
            # 不显示完整的 key，只显示前4位和后4位
            masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            self._record(
                "VLM API Key",
                "PASS",
                f"环境变量 {api_key_env} 已设置 (值: {masked})",
            )
        else:
            self._record(
                "VLM API Key",
                "FAIL",
                f"环境变量 {api_key_env} 未设置！\n"
                f"VLM 调用需要真实的 API Key，不允许 mock VLM。\n"
                f"请设置环境变量:\n"
                f"  Windows: set {api_key_env}=your-api-key\n"
                f"  Linux/Mac: export {api_key_env}='your-api-key'\n"
                f"或在 .env 文件中添加 {api_key_env}=your-api-key",
            )

    def check_dinov2(self) -> None:
        """检查 DINOv2 模型是否可以加载。"""
        try:
            import torch
            from transformers import AutoModel, AutoImageProcessor

            import yaml

            default_config_path = PROJECT_ROOT / "config" / "default.yaml"
            if default_config_path.exists():
                with open(str(default_config_path), "r", encoding="utf-8") as f:
                    default_config = yaml.safe_load(f)
                feat_cfg = default_config.get("feature_extractor", {})
                model_name = feat_cfg.get("model_name", "facebook/dinov2-base")
                cache_dir = feat_cfg.get("cache_dir", ".cache/huggingface")
            else:
                model_name = "facebook/dinov2-base"
                cache_dir = ".cache/huggingface"

            # 尝试加载模型（不一定下载，先检查缓存）
            cache_path = PROJECT_ROOT / cache_dir
            if cache_path.exists():
                self._record(
                    "DINOv2 模型",
                    "PASS",
                    f"模型缓存目录存在: {cache_path}\n"
                    f"模型名: {model_name}\n"
                    f"注意: 首次运行时会自动下载模型权重。"
                    f"如需预下载，请运行: python scripts/00_prepare_models.py",
                )
            else:
                self._record(
                    "DINOv2 模型",
                    "WARN",
                    f"模型缓存目录不存在: {cache_path}\n"
                    f"模型名: {model_name}\n"
                    f"首次运行时会自动从 HuggingFace Hub 下载模型权重。\n"
                    f"如需提前下载，请运行: python scripts/00_prepare_models.py",
                )

        except ImportError as e:
            self._record(
                "DINOv2 模型",
                "FAIL",
                f"缺少依赖: {e}\n请安装 torch 和 transformers。",
            )
        except Exception as e:
            self._record(
                "DINOv2 模型",
                "WARN",
                f"模型检查时出错: {e}\n"
                f"可能需要在首次运行时下载模型权重。"
                f"请运行: python scripts/00_prepare_models.py",
            )

    def check_nuscenes_data(self) -> None:
        """检查 nuScenes 数据集目录是否存在。"""
        import yaml

        data_config_path = PROJECT_ROOT / "config" / "data_nuscenes.yaml"
        if data_config_path.exists():
            with open(str(data_config_path), "r", encoding="utf-8") as f:
                data_config = yaml.safe_load(f)
            dataroot = PROJECT_ROOT / data_config.get("dataroot", "data/nuscenes/raw")
            version = data_config.get("version", "v1.0-mini")
        else:
            dataroot = PROJECT_ROOT / "data/nuscenes/raw"
            version = "v1.0-mini"

        # 检查数据目录
        if not dataroot.exists():
            self._record(
                "nuScenes 数据集",
                "WARN",
                f"数据集目录不存在: {dataroot}\n"
                f"请在 demo 开发完成后将 nuScenes {version} 数据集放置到该目录。\n"
                f"目录结构应为:\n"
                f"  {dataroot}/\n"
                f"    {version}/\n"
                f"    samples/\n"
                f"    maps/\n"
                f"注意: 环境检查阶段仅 warning，数据相关脚本运行时会 hard fail。",
            )
            return

        # 检查版本目录
        version_dir = dataroot / version
        if not version_dir.exists():
            self._record(
                "nuScenes 数据集",
                "WARN",
                f"数据集目录存在但版本目录不存在: {version_dir}\n"
                f"请确保已正确下载 {version} 版本数据集。",
            )
            return

        # 检查 samples 目录
        samples_dir = dataroot / "samples"
        if not samples_dir.exists():
            self._record(
                "nuScenes 数据集",
                "WARN",
                f"数据集目录存在但 samples 目录不存在: {samples_dir}\n"
                f"请确保数据集解压完整。",
            )
            return

        # 检查 maps 目录
        maps_dir = dataroot / "maps"
        if not maps_dir.exists():
            self._record(
                "nuScenes 数据集",
                "WARN",
                f"数据集目录存在但 maps 目录不存在: {maps_dir}\n"
                f"请确保数据集解压完整。",
            )
            return

        # 检查 CAM_FRONT 目录
        cam_dir = samples_dir / "CAM_FRONT"
        if cam_dir.exists():
            img_count = len(list(cam_dir.glob("*")))
            self._record(
                "nuScenes 数据集",
                "PASS",
                f"数据集完整: {dataroot}\n"
                f"  版本: {version}\n"
                f"  CAM_FRONT 图像数: {img_count}",
            )
        else:
            self._record(
                "nuScenes 数据集",
                "WARN",
                f"数据集目录存在但 CAM_FRONT 目录不存在: {cam_dir}\n"
                f"请确保数据集解压完整。",
            )

    def run_all_checks(self) -> None:
        """执行所有环境检查。"""
        print("=" * 60)
        print("智能驾驶 VLA 分层记忆系统 - 环境检查")
        print("=" * 60)
        print()

        # 执行检查
        self.check_conda_environment()
        self.check_python_version()
        self.check_torch()
        self.check_transformers()
        self.check_openai()
        self.check_nuscenes_devkit()
        self.check_faiss()
        self.check_api_key()
        self.check_dinov2()
        self.check_nuscenes_data()

        # 输出结果
        print()
        print("-" * 60)
        print("检查结果:")
        print("-" * 60)

        for name, status, detail in self.results:
            if status == "PASS":
                icon = "✅"
            elif status == "WARN":
                icon = "⚠️"
            else:
                icon = "❌"
            print(f"\n{icon} [{status}] {name}")
            for line in detail.split("\n"):
                print(f"   {line}")

        # 汇总
        print()
        print("=" * 60)
        print(f"汇总: 通过={self.pass_count}, 警告={self.warn_count}, 失败={self.fail_count}")
        print("=" * 60)

        if self.fail_count > 0:
            print()
            print("存在失败项，请根据上述提示修复后再运行 demo。")
        else:
            print()
            if self.warn_count > 0:
                print("存在警告项，demo 可能可以运行但建议修复。")
            else:
                print("所有检查通过！环境准备就绪。")


def main():
    """主函数。"""
    checker = EnvironmentChecker()
    checker.run_all_checks()


if __name__ == "__main__":
    main()
