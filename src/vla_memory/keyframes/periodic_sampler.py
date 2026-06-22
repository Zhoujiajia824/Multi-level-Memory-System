"""
周期采样器
==========
按照固定步长（step）或固定时间间隔（interval_seconds）采样关键帧。
nuScenes 第一版默认 step=2，从约 2Hz keyframe 降到约 1Hz。
所有参数从 config/data_nuscenes.yaml 读取。
"""
from __future__ import annotations

from typing import List

from src.vla_memory.keyframes.base import BaseKeyframeSampler
from src.vla_memory.schemas.frame import FrameMeta
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("periodic_sampler")


class PeriodicSampler(BaseKeyframeSampler):
    """周期采样器。

    支持两种模式：
    1. 按 step 抽取：每隔 step 帧取 1 帧。
    2. 按 interval_seconds 抽取：根据帧率计算等效 step。

    nuScenes keyframe 约为 2Hz，step=2 时实现约 1Hz 采样。

    Args:
        step: 每隔 step 帧取 1 帧。默认 2。
        interval_seconds: 目标采样间隔（秒）。如果指定，优先于 step。
        source_fps: 数据源帧率（Hz），默认 2.0（nuScenes keyframe 频率）。
    """

    # nuScenes keyframe 频率常量
    NUSCENES_KEYFRAME_RATE = 2.0  # Hz

    def __init__(
        self,
        step: int = 2,
        interval_seconds: float | None = None,
        source_fps: float = 2.0,
    ):
        self.source_fps = source_fps  # 数据源帧率

        if interval_seconds is not None:
            # 按 interval_seconds 计算 step
            self.interval_seconds = interval_seconds
            self._step = max(1, round(source_fps * interval_seconds))
        else:
            # 按 step
            self._step = max(1, step)
            self.interval_seconds = self._step / source_fps

        logger.debug(
            f"PeriodicSampler 初始化: step={self._step}, "
            f"interval={self.interval_seconds:.2f}s, "
            f"source_fps={source_fps}Hz"
        )

    @property
    def step(self) -> int:
        """获取采样步长。"""
        return self._step

    def sample(self, frames: List[FrameMeta]) -> List[FrameMeta]:
        """按固定步长从帧列表中采样关键帧。

        Args:
            frames: 完整的 FrameMeta 列表（按时间排序）。

        Returns:
            采样后的 FrameMeta 子列表。
        """
        if not frames:
            logger.warning("帧列表为空，无法采样。")
            return []

        sampled = frames[::self._step]
        logger.info(
            f"周期采样完成: 原始 {len(frames)} 帧 -> "
            f"采样 {len(sampled)} 帧 "
            f"(step={self._step}, 约 {self.get_sample_rate():.2f} Hz)"
        )
        return sampled

    def get_sample_rate(self) -> float:
        """获取采样频率（Hz）。"""
        return self.source_fps / self._step
