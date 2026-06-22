"""向量存储基类
==============
定义向量存储的抽象接口，具体实现（如 FAISS）继承此基类。
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseVectorStore(ABC):
    """向量存储抽象基类。"""

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """添加向量到存储中。

        Args:
            vectors: 向量矩阵，形状 (N, dim)。
            ids: 对应的 ID 列表。
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, List[str]]:
        """搜索最相似的向量。

        Args:
            query: 查询向量，形状 (dim,)。
            top_k: 返回最近的 top_k 个结果。

        Returns:
            (相似度分数数组, 对应 ID 列表)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存索引到文件。"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """从文件加载索引。"""
        pass

    @abstractmethod
    def size(self) -> int:
        """获取存储的向量数量。"""
        pass
