"""
配置管理模块
============
负责读取、合并、覆盖 YAML 配置文件。支持多配置合并和 CLI 覆盖。
所有路径自动转换为 pathlib.Path 对象，保证 Windows 路径兼容。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Optional

import yaml


# 项目根目录：config/ 的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/vla_memory/common -> 项目根


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典。

    override 中的值会覆盖 base 中的同名键。对于嵌套字典，递归合并。

    Args:
        base: 基础字典。
        override: 覆盖字典。

    Returns:
        合并后的新字典（不修改原字典）。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Path) -> dict:
    """加载单个 YAML 配置文件。

    Args:
        path: YAML 文件路径。

    Returns:
        解析后的字典。

    Raises:
        FileNotFoundError: 配置文件不存在。
        yaml.YAMLError: YAML 解析错误。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def resolve_path(base_dir: Path, path_str: str) -> Path:
    """将路径字符串解析为绝对路径。

    如果路径是相对路径，则相对于 base_dir 解析。

    Args:
        base_dir: 基准目录。
        path_str: 路径字符串。

    Returns:
        解析后的绝对路径。
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


class Config:
    """项目配置管理器。

    支持多 YAML 配置合并、CLI 参数覆盖、路径自动解析。

    Attributes:
        root: 项目根目录。
        data: 合并后的配置字典。
    """

    # 默认加载的配置文件列表（按顺序合并，后面的覆盖前面的）
    DEFAULT_CONFIG_FILES = [
        "config/default.yaml",
        "config/api_models.yaml",
        "config/data_nuscenes.yaml",
        "config/memory.yaml",
        "config/decision.yaml",
        "config/evaluation.yaml",
    ]

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        overrides: Optional[dict] = None,
    ):
        """初始化配置管理器。

        Args:
            config_dir: 配置文件目录。默认为项目根目录下的 config/。
            overrides: CLI 或代码层面的配置覆盖字典。
        """
        self.root = PROJECT_ROOT
        self.config_dir = config_dir or (self.root / "config")

        # 加载并合并所有配置文件
        self.data: dict = {}
        for config_file in self.DEFAULT_CONFIG_FILES:
            config_path = self.root / config_file
            if config_path.exists():
                file_data = load_yaml(config_path)
                self.data = _deep_merge(self.data, file_data)

        # 应用覆盖
        if overrides:
            self.data = _deep_merge(self.data, overrides)

        # 解析路径
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """将配置中的路径字符串解析为 pathlib.Path 对象。"""
        # 输出目录
        path_keys = [
            ("output_dir",), ("log_dir",), ("feature_dir",), ("memory_db_dir",),
        ]
        for keys in path_keys:
            value = self.get(keys[0])
            if isinstance(value, str):
                self.data[keys[0]] = resolve_path(self.root, value)

        # 数据目录
        data_cfg = self.data.get("dataroot")
        if isinstance(data_cfg, str):
            self.data["dataroot"] = resolve_path(self.root, data_cfg)

        # 知识文件路径
        mem_cfg = self.data.get("long_term", {})
        for key in ("rules_file", "strategies_file", "knowledge_graph_dir"):
            val = mem_cfg.get(key)
            if isinstance(val, str):
                mem_cfg[key] = resolve_path(self.root, val)

        # 持久化目录
        persist_cfg = self.data.get("persistence", {})
        save_dir = persist_cfg.get("save_dir")
        if isinstance(save_dir, str):
            persist_cfg["save_dir"] = resolve_path(self.root, save_dir)

        # 评测报告目录
        eval_cfg = self.data.get("output", {})
        report_dir = eval_cfg.get("report_dir")
        if isinstance(report_dir, str):
            eval_cfg["report_dir"] = resolve_path(self.root, report_dir)

        # 特征模型缓存目录
        feat_cfg = self.data.get("feature_extractor", {})
        cache_dir = feat_cfg.get("cache_dir")
        if isinstance(cache_dir, str):
            feat_cfg["cache_dir"] = resolve_path(self.root, cache_dir)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（顶层键）。

        Args:
            key: 配置键名。
            default: 默认值。

        Returns:
            配置值或默认值。
        """
        return self.data.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """获取嵌套配置值。

        Args:
            *keys: 嵌套键名序列，如 get_nested("memory", "mid_term", "top_k")。
            default: 默认值。

        Returns:
            嵌套配置值或默认值。
        """
        current = self.data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def get_path(self, key: str) -> Path:
        """获取路径类型的配置值。

        Args:
            key: 配置键名。

        Returns:
            解析后的 Path 对象。

        Raises:
            KeyError: 配置键不存在。
        """
        value = self.data.get(key)
        if value is None:
            raise KeyError(f"配置键不存在: {key}")
        if isinstance(value, Path):
            return value
        return resolve_path(self.root, str(value))

    def ensure_output_dirs(self) -> None:
        """确保所有输出目录存在。"""
        dir_keys = ["output_dir", "log_dir", "feature_dir", "memory_db_dir"]
        for key in dir_keys:
            path = self.data.get(key)
            if path is not None:
                Path(path).mkdir(parents=True, exist_ok=True)

        # 评测报告目录
        eval_output = self.data.get("output", {})
        report_dir = eval_output.get("report_dir")
        if report_dir:
            Path(report_dir).mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """返回配置摘要字符串，用于日志记录。"""
        lines = [
            f"项目: {self.data.get('project_name', 'unknown')}",
            f"运行标识: {self.data.get('run_id', 'default')}",
            f"设备: {self.data.get('device', 'auto')}",
            f"数据集版本: {self.data.get('version', 'unknown')}",
            f"特征模型: {self.data.get('feature_extractor', {}).get('model_name', 'unknown')}",
            f"VLM 模型: {self.data.get('scene_understanding', {}).get('model_name', 'unknown')}",
            f"短期记忆容量: {self.data.get('short_term', {}).get('capacity', 10)}",
            f"中期记忆 Top-K: {self.data.get('mid_term', {}).get('top_k', 3)}",
        ]
        return "\n".join(lines)


def load_config(
    config_dir: Optional[Path] = None,
    overrides: Optional[dict] = None,
) -> Config:
    """加载项目配置的便捷函数。

    Args:
        config_dir: 配置文件目录。
        overrides: 配置覆盖字典。

    Returns:
        Config 实例。
    """
    return Config(config_dir=config_dir, overrides=overrides)
