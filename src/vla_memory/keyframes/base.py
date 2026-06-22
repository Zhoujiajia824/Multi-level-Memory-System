"""
关键帧采样器基类
================
定义关键帧采样策略的抽象接口 BaseKeyframeSampler。
输入 FrameMeta 列表，输出采样后的 FrameMeta 子集。
不同采样策略（周期采样、nuScenes keyframe 采样等）继承此基类。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.vla_memory.schemas.frame import FrameMeta


class BaseKeyframeSampler(ABC):
    """关键帧采样器抽象基类。

    所有采样器必须实现 sample() 和 get_sample_rate() 方法。
    sample() 接收 FrameMeta 列表，返回采样后的 FrameMeta 子集。
    """

    @abstractmethod
    def sample(self, frames: List[FrameMeta]) -> List[FrameMeta]:
        """从帧列表中采样关键帧。

        Args:
            frames: 完整的 FrameMeta 列表（按时间排序）。

        Returns:
            采样后的 FrameMeta 子列表。
        """
        ...

    @abstractmethod
    def get_sample_rate(self) -> float:
        """获取采样频率（Hz）。

        Returns:
            采样频率。
        """
        ...
