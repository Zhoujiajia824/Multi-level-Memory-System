"""提示词模板加载器
==================
集中加载 config/prompts.yaml，提供按点号路径取值与 str.format_map 渲染。
所有 VLM 提示词通过此加载器获取，方便统一管理与人工修改。

使用方式：
    from src.vla_memory.common.prompt_loader import get_prompt_loader
    loader = get_prompt_loader()
    text = loader.render("decision.user", scene_block=..., waypoint_min_num=20, ...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import yaml

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("prompt_loader")

# 项目根目录：与 common/config.py 同样的算法（src/vla_memory/common -> 项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "config" / "prompts.yaml"


class _SafeFormatDict(dict):
    """str.format_map 的兜底字典：未定义的占位符保留原文 {key} 并 warning。

    设计目的：避免因调用方少传一个变量就直接抛 KeyError 中断整个 demo。
    渲染后调用方仍可用日志检查是否有未填充的占位符。
    """

    def __missing__(self, key: str) -> str:
        logger.warning("prompt 占位符未提供: {%s}，保留原文", key)
        return "{" + key + "}"


class PromptLoader:
    """提示词模板加载器。

    Args:
        path: prompts.yaml 路径。默认 config/prompts.yaml。
        strict: 若为 True，渲染时未定义占位符会抛 KeyError；
                若为 False（默认），保留原文 {key} 并 warning。
    """

    def __init__(
        self,
        path: Optional[Path | str] = None,
        strict: bool = False,
    ):
        self.path = Path(path) if path is not None else DEFAULT_PROMPTS_PATH
        self.strict = strict
        self._data: dict = {}
        self.reload()

    def reload(self) -> None:
        """从磁盘重新读取 prompts.yaml。"""
        if not self.path.exists():
            raise FileNotFoundError(f"提示词模板文件不存在: {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"prompts.yaml 顶层必须是字典: {self.path}")
        self._data = data
        logger.info("已加载提示词模板: %s", self.path)

    def get(self, key: str) -> str:
        """按点号路径取字符串模板，如 'decision.user'。

        Args:
            key: 点号分隔的嵌套键。

        Returns:
            模板字符串。

        Raises:
            KeyError: 路径不存在。
            TypeError: 取到的值不是字符串。
        """
        value = self._resolve(key)
        if not isinstance(value, str):
            raise TypeError(
                f"prompts.yaml 中 '{key}' 不是字符串（实际类型 {type(value).__name__}）"
            )
        return value

    def get_list(self, key: str) -> List[Any]:
        """按点号路径取列表，如 'scene_understanding.required_fields'。"""
        value = self._resolve(key)
        if not isinstance(value, list):
            raise TypeError(
                f"prompts.yaml 中 '{key}' 不是列表（实际类型 {type(value).__name__}）"
            )
        return value

    def render(self, key: str, **variables: Any) -> str:
        """取模板并用 str.format_map 渲染。

        Args:
            key: 点号路径，如 'decision.user'。
            **variables: 模板占位符变量。

        Returns:
            渲染后的字符串。

        Raises:
            KeyError: strict=True 且占位符未提供时。
        """
        template = self.get(key)
        if self.strict:
            return template.format_map(variables)
        return template.format_map(_SafeFormatDict(variables))

    # -------------------- 内部 --------------------

    def _resolve(self, key: str) -> Any:
        """点号路径解析。"""
        if not key:
            raise KeyError("prompts key 不能为空")
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"prompts.yaml 中找不到键路径: '{key}'（卡在 '{part}'）")
            node = node[part]
        return node


# -------------------- 模块级单例 --------------------

_DEFAULT_LOADER: Optional[PromptLoader] = None


def get_prompt_loader(path: Optional[Path | str] = None) -> PromptLoader:
    """返回模块级单例 PromptLoader。

    首次调用时按 path（或默认 config/prompts.yaml）初始化；
    后续调用忽略 path 参数。要换文件请用 reset_prompt_loader。
    """
    global _DEFAULT_LOADER
    if _DEFAULT_LOADER is None:
        _DEFAULT_LOADER = PromptLoader(path=path)
    return _DEFAULT_LOADER


def reset_prompt_loader(path: Optional[Path | str] = None) -> PromptLoader:
    """清除并重建模块级单例。主要供测试使用。"""
    global _DEFAULT_LOADER
    _DEFAULT_LOADER = PromptLoader(path=path)
    return _DEFAULT_LOADER
