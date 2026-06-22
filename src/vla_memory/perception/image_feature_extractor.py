"""
图像特征提取器抽象基类
======================
定义图像特征提取的统一接口。
具体实现（DINOv2 等）继承此基类。
提供 extract / batch_extract / save_feature / load_feature 四个核心方法。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import numpy as np


class ImageFeatureExtractor(ABC):
    """图像特征提取器抽象基类。

    所有特征提取器（DINOv2、CLIP 等）必须继承此类并实现 extract 方法。
    """

    @abstractmethod
    def extract(self, image_path: Union[str, Path]) -> np.ndarray:
        """从单张图像中提取特征向量。

        Args:
            image_path: 图像文件绝对路径。

        Returns:
            特征向量 (feature_dim,) 的 float32 numpy 数组。

        Raises:
            FileNotFoundError: 图像文件不存在时 hard fail。
            RuntimeError: 特征提取失败时 hard fail。
        """
        ...

    @abstractmethod
    def batch_extract(self, image_paths: List[Union[str, Path]]) -> List[np.ndarray]:
        """批量提取图像特征。

        Args:
            image_paths: 图像文件路径列表。

        Returns:
            特征向量列表。图像不存在时 hard fail，不返回随机 embedding。
        """
        ...

    @abstractmethod
    def get_feature_dim(self) -> int:
        """获取特征向量维度。

        Returns:
            特征维度整数。
        """
        ...

    @staticmethod
    def save_feature(feature: np.ndarray, path: Union[str, Path]) -> Path:
        """将特征向量保存为 .npy 文件。

        Args:
            feature: 特征向量。
            path: 保存路径（.npy）。

        Returns:
            保存后的文件 Path。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), feature.astype(np.float32))
        return path

    @staticmethod
    def load_feature(path: Union[str, Path]) -> np.ndarray:
        """从 .npy 文件加载特征向量。

        Args:
            path: .npy 文件路径。

        Returns:
            特征向量。

        Raises:
            FileNotFoundError: 文件不存在。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"特征文件不存在: {path}")
        return np.load(str(path))
