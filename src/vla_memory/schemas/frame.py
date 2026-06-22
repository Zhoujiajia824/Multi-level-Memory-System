"""
关键帧元数据数据模型
====================
定义 FrameMeta 数据结构，包含帧唯一标识、场景标识、时间戳、
摄像头名称、图像路径等基础信息。
FrameMeta 是全链路中所有后续数据结构的锚点。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FrameMeta(BaseModel):
    """关键帧元数据模型。

    表示从 nuScenes 中采样得到的一个关键帧的元信息。
    该结构贯穿数据加载、特征提取、场景理解、记忆构建、决策全链路。

    Attributes:
        frame_id: 帧唯一标识，格式为 '{scene_token}_{sample_token}' 或直接使用 sample_token。
        scene_token: 所属场景的 token（nuScenes scene token）。
        sample_token: 所属样本的 token（nuScenes sample token）。
        timestamp: 时间戳（微秒），来自 nuScenes sample timestamp。
        camera_name: 摄像头名称，默认 'CAM_FRONT'。
        image_path: 前视角图像文件的绝对路径。
    """
    frame_id: str = Field(
        ...,
        description="帧唯一标识，格式为 '{scene_token}_{sample_token}'",
    )
    scene_token: str = Field(
        ...,
        description="所属场景 token（nuScenes scene token）",
    )
    sample_token: str = Field(
        ...,
        description="所属样本 token（nuScenes sample token）",
    )
    timestamp: int = Field(
        ...,
        description="时间戳（微秒），来自 nuScenes sample timestamp",
    )
    camera_name: str = Field(
        default="CAM_FRONT",
        description="摄像头名称，默认 CAM_FRONT",
    )
    image_path: str = Field(
        ...,
        description="前视角图像文件的绝对路径",
    )

    def to_index_dict(self) -> dict:
        """转换为索引写入用的字典（用于 JSONL 序列化）。

        Returns:
            包含所有字段的字典。
        """
        return {
            "frame_id": self.frame_id,
            "scene_token": self.scene_token,
            "sample_token": self.sample_token,
            "timestamp": self.timestamp,
            "camera_name": self.camera_name,
            "image_path": self.image_path,
        }
