"""
nuScenes 关键帧采样器
====================
封装 nuScenes keyframe 遍历 + 周期采样逻辑。
从 NuScenesAdapter 的 iter_frames 输出中按 step 采样，
保留 scene_token、sample_token、timestamp、image_path。
输出关键帧 FrameMeta 列表。
"""
from __future__ import annotations

from typing import List, Dict, Optional

from src.vla_memory.keyframes.base import BaseKeyframeSampler
from src.vla_memory.keyframes.periodic_sampler import PeriodicSampler
from src.vla_memory.schemas.frame import FrameMeta
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("nuscenes_keyframe_sampler")


class NuScenesKeyframeSampler(BaseKeyframeSampler):
    """nuScenes 关键帧采样器。

    封装 nuScenes 场景遍历 + PeriodicSampler 采样逻辑。
    nuScenes keyframe 约为 2Hz，默认 step=2 实现约 1Hz 采样。

    Args:
        step: 每隔 step 个 keyframe 取 1 个。默认 2。
        camera_name: 摄像头名称。默认 'CAM_FRONT'。
    """

    def __init__(self, step: int = 2, camera_name: str = "CAM_FRONT"):
        self._sampler = PeriodicSampler(step=step, source_fps=2.0)
        self.camera_name = camera_name

    def sample(self, frames: List[FrameMeta]) -> List[FrameMeta]:
        """从 FrameMeta 列表中按步长采样。

        Args:
            frames: 完整的 FrameMeta 列表。

        Returns:
            采样后的关键帧 FrameMeta 列表。
        """
        return self._sampler.sample(frames)

    def sample_scene(
        self,
        adapter,  # NuScenesAdapter 实例
        scene_token: str,
        max_frames: int | None = None,
    ) -> List[FrameMeta]:
        """对单个场景进行关键帧采样。

        Args:
            adapter: NuScenesAdapter 实例（已加载）。
            scene_token: 场景 token。
            max_frames: 每个场景最多返回的关键帧数（None 表示全部）。

        Returns:
            采样后的 FrameMeta 列表。
        """
        # 获取该场景的所有帧
        all_frames = list(adapter.iter_frames(scene_token))
        if not all_frames:
            logger.warning(f"场景 {scene_token[:8]}... 没有帧数据。")
            return []

        # 采样
        sampled = self.sample(all_frames)

        # 限制数量
        if max_frames is not None:
            sampled = sampled[:max_frames]

        logger.info(
            f"场景 {scene_token[:8]}...: "
            f"原始 {len(all_frames)} 帧 -> 采样 {len(sampled)} 帧"
        )
        return sampled

    def sample_all_scenes(
        self,
        adapter,  # NuScenesAdapter 实例
        max_scenes: int | None = None,
        max_frames_per_scene: int | None = None,
        max_samples_per_scene: int | None = None,
    ) -> Dict[str, List[FrameMeta]]:
        """对所有场景进行关键帧采样。

        Args:
            adapter: NuScenesAdapter 实例（已加载）。
            max_scenes: 最多处理的场景数（None 表示全部）。
            max_frames_per_scene: 每个场景最多返回的关键帧数。
            max_samples_per_scene: max_frames_per_scene 的兼容别名。

        Returns:
            {scene_token: [FrameMeta]} 字典。
        """
        if max_samples_per_scene is not None:
            if (
                max_frames_per_scene is not None
                and max_frames_per_scene != max_samples_per_scene
            ):
                raise ValueError(
                    "max_frames_per_scene and max_samples_per_scene must match "
                    "when both are provided."
                )
            max_frames_per_scene = max_samples_per_scene

        scene_tokens = adapter.list_scenes()
        if max_scenes is not None:
            scene_tokens = scene_tokens[:max_scenes]

        result = {}
        total_keyframes = 0
        for scene_token in scene_tokens:
            keyframes = self.sample_scene(
                adapter, scene_token,
                max_frames=max_frames_per_scene,
            )
            result[scene_token] = keyframes
            total_keyframes += len(keyframes)

        logger.info(
            f"所有场景采样完成: {len(result)} 个场景, "
            f"{total_keyframes} 个关键帧, "
            f"约 {self.get_sample_rate():.1f} Hz"
        )
        return result

    def get_sample_rate(self) -> float:
        """获取采样频率（Hz）。"""
        return self._sampler.get_sample_rate()

    @property
    def step(self) -> int:
        """获取采样步长。"""
        return self._sampler.step
