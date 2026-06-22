"""
驾驶数据集抽象基类
==================
定义 BaseDrivingDataset 抽象类，提供统一的数据访问接口。
后续可扩展支持 CARLA、视频输入、图像序列等数据源。
所有数据适配器（nuScenes、CARLA 等）都必须继承此基类。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator

from src.vla_memory.schemas.frame import FrameMeta
from src.vla_memory.schemas.ego_state import EgoState


class BaseDrivingDataset(ABC):
    """驾驶数据集抽象基类。

    定义统一的数据访问接口，所有数据适配器都必须实现。
    核心方法覆盖场景遍历、帧迭代、图像路径获取、位姿查询、轨迹构建。

    Subclasses:
        - NuScenesAdapter: nuScenes 数据集适配器。
        - VideoAdapter: 视频输入适配器（后续）。
        - ImageSequenceAdapter: 图像序列适配器（后续）。
    """

    # ----------------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """加载数据集。

        数据集目录不存在时必须 hard fail 并输出中文错误信息。
        不允许使用假数据。
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """检查数据集是否已加载。"""
        ...

    # ----------------------------------------------------------------
    # 场景 & 帧访问
    # ----------------------------------------------------------------

    @abstractmethod
    def list_scenes(self) -> List[str]:
        """列出所有场景标识（如 scene_token）。

        Returns:
            场景标识列表。
        """
        ...

    @abstractmethod
    def iter_frames(self, scene_token: str) -> Iterator[FrameMeta]:
        """遍历指定场景中的所有帧元数据。

        Args:
            scene_token: 场景标识。

        Yields:
            FrameMeta 实例。
        """
        ...

    # ----------------------------------------------------------------
    # 数据获取
    # ----------------------------------------------------------------

    @abstractmethod
    def get_frame_image_path(self, sample_token: str, camera_name: str = "CAM_FRONT") -> str:
        """获取指定帧的图像文件绝对路径。

        Args:
            sample_token: 样本标识。
            camera_name: 摄像头名称。

        Returns:
            图像文件的绝对路径字符串。

        Raises:
            ValueError: 样本不存在或摄像头数据缺失。
        """
        ...

    @abstractmethod
    def get_ego_pose(self, sample_token: str) -> EgoState:
        """获取指定帧的自车位姿和运动状态。

        包含位置、航向角，以及通过差分估计的速度和加速度。
        第一版通过相邻 ego_pose 差分计算速度和加速度。

        Args:
            sample_token: 样本标识。

        Returns:
            EgoState 实例。
        """
        ...

    @abstractmethod
    def get_history_trajectory(
        self,
        sample_token: str,
        history_seconds: float = 5.0,
    ) -> List[Dict[str, float]]:
        """获取指定帧最近 N 秒的历史轨迹（ego-centric 坐标系）。

        坐标系: ego-centric，x 前向，y 左向，单位米。
        每个点包含 {t, x, y}，t 为负值（过去）。

        Args:
            sample_token: 样本标识。
            history_seconds: 历史时间窗口（秒）。

        Returns:
            ego-centric 坐标系下的历史轨迹点列表。
        """
        ...

    @abstractmethod
    def get_future_ego_trajectory(
        self,
        sample_token: str,
        future_seconds: float = 3.0,
    ) -> List[Dict[str, float]]:
        """获取指定帧未来 N 秒的真值轨迹（ego-centric 坐标系）。

        用于评测模块计算 ADE / FDE。
        坐标系: ego-centric，x 前向，y 左向，单位米。

        Args:
            sample_token: 样本标识。
            future_seconds: 未来时间窗口（秒）。

        Returns:
            ego-centric 坐标系下的未来轨迹点列表。
        """
        ...
