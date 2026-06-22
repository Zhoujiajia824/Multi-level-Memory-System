"""
VLM 客户端抽象基类
====================
定义视觉语言模型（VLM）调用的统一接口。
子类必须实现 understand_scene 和 decide 两个核心方法。
不允许 mock response。

P5：``decide`` 接口扩展为支持 ``image_paths: List[str]``（一次喂多张图）。
保留旧 ``optional_image_path`` 参数作为单图兼容 shim。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class VLMClient(ABC):
    """VLM 客户端抽象基类。

    定义两个核心接口：
    - understand_scene: 带图像的驾驶场景理解
    - decide: 文本 + 1~N 张图像的驾驶决策
    """

    @abstractmethod
    def understand_scene(
        self,
        image_path: str,
        prompt: str,
        extra_context: Optional[dict] = None,
    ) -> str:
        """调用 VLM 进行驾驶场景理解（带图像）。

        Args:
            image_path: 前视角图像文件路径。
            prompt: 场景理解 prompt（要求输出 JSON）。
            extra_context: 可选的额外上下文信息。

        Returns:
            VLM 的原始文本输出（应为 JSON 字符串）。

        Raises:
            FileNotFoundError: 图像不存在。
            EnvironmentError: API Key 未设置。
            RuntimeError: API 调用失败（重试后仍失败）。
        """
        ...

    @abstractmethod
    def decide(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        optional_image_path: Optional[str] = None,
    ) -> str:
        """调用 VLM 进行驾驶决策（文本 + 0~N 张图像）。

        Args:
            prompt: 决策 prompt（包含记忆、状态等信息）。
            image_paths: 图像路径列表，按 oldest→newest 顺序排列；
                None 或空列表表示不带图。**当前帧应放在末尾。**
            optional_image_path: P5 之前的单图兼容参数。当
                ``image_paths`` 为 None 且本参数非空时，等价于传入
                ``[optional_image_path]``。新代码应使用 ``image_paths``。

        Returns:
            VLM 的原始文本输出（应为 JSON 字符串）。

        Raises:
            EnvironmentError: API Key 未设置。
            RuntimeError: API 调用失败。
        """
        ...
