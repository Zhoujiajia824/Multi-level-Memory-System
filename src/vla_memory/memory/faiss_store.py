"""FAISS 向量存储
================
使用 faiss-cpu 实现 L2 归一化向量的内积检索。
必须使用真实 FAISS，不允许降级到 numpy 检索。
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from src.vla_memory.memory.vector_store import BaseVectorStore
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("faiss_store")


class FAISSVectorStore(BaseVectorStore):
    """FAISS 向量存储实现。

    使用 FAISS IndexFlatIP 进行内积检索。
    向量必须先 L2 归一化，内积等价于余弦相似度。

    Args:
        dimension: 向量维度。
        index_type: 索引类型（默认 IndexFlatIP）。
    """

    def __init__(self, dimension: int = 768, index_type: str = "IndexFlatIP"):
        self.dimension = dimension  # 向量维度
        self.index_type = index_type  # 索引类型
        self._index = None  # FAISS 索引对象
        self._ids: List[str] = []  # 向量 ID 列表（与索引中的位置对应）
        self._init_index()

    def _init_index(self) -> None:
        """初始化 FAISS 索引。"""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS 未安装！中期记忆向量检索必须使用 FAISS，不允许降级到 numpy。\n"
                "Windows 推荐通过 conda-forge 安装: conda install -c conda-forge faiss-cpu\n"
                "或通过 pip 安装: pip install faiss-cpu"
            )

        if self.index_type == "IndexFlatIP":
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"FAISS 索引初始化成功: {self.index_type}, 维度={self.dimension}")
        else:
            raise ValueError(f"不支持的 FAISS 索引类型: {self.index_type}，当前仅支持 IndexFlatIP")

    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """添加向量到索引中。

        Args:
            vectors: 向量矩阵，形状 (N, dim)，应为 float32 类型且已 L2 归一化。
            ids: 对应的 ID 列表。

        Raises:
            ValueError: 向量维度不匹配或数量与 ID 不一致。
        """
        if len(vectors) != len(ids):
            raise ValueError(f"向量数量 ({len(vectors)}) 与 ID 数量 ({len(ids)}) 不一致")

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度不匹配: 期望 {self.dimension}，实际 {vectors.shape[1]}")

        # L2 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        vectors = vectors / norms

        self._index.add(vectors)
        self._ids.extend(ids)
        logger.debug(f"添加 {len(ids)} 条向量，当前总数: {self._index.ntotal}")

    def search(self, query: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, List[str]]:
        """搜索最相似的向量。

        Args:
            query: 查询向量，形状 (dim,)。
            top_k: 返回最近的 top_k 个结果。

        Returns:
            (相似度分数数组, 对应 ID 列表)
        """
        if self._index.ntotal == 0:
            logger.warning("FAISS 索引为空，无法检索。")
            return np.array([]), []

        query = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)

        # L2 归一化查询向量
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # 限制 top_k 不超过当前向量数量
        actual_k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query, actual_k)

        # 提取结果（过滤 FAISS 在 ntotal < top_k 时返回的 -1 无效索引）
        valid_mask = (indices[0] >= 0) & (indices[0] < len(self._ids))
        result_scores = scores[0][valid_mask]
        result_ids = [self._ids[idx] for idx in indices[0][valid_mask]]

        return result_scores, result_ids

    def save(self, path: str) -> None:
        """保存 FAISS 索引和 ID 映射到文件。"""
        import faiss
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self._index, str(path))

        # 保存 ID 映射
        ids_path = path.with_suffix(".ids.json")
        with open(str(ids_path), "w", encoding="utf-8") as f:
            json.dump(self._ids, f, ensure_ascii=False)

        logger.info(f"FAISS 索引已保存: {path} ({self._index.ntotal} 条向量)")

    def load(self, path: str) -> None:
        """从文件加载 FAISS 索引和 ID 映射。"""
        import faiss
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS 索引文件不存在: {path}")

        # 加载 FAISS 索引
        self._index = faiss.read_index(str(path))

        # 从加载的索引中恢复实际维度
        self.dimension = self._index.d

        # 加载 ID 映射
        ids_path = path.with_suffix(".ids.json")
        if ids_path.exists():
            with open(str(ids_path), "r", encoding="utf-8") as f:
                self._ids = json.load(f)
        else:
            logger.warning(f"ID 映射文件不存在: {ids_path}，将使用空列表")
            self._ids = []

        logger.info(f"FAISS 索引已加载: {path} ({self._index.ntotal} 条向量)")

    def size(self) -> int:
        """获取存储的向量数量。"""
        return self._index.ntotal if self._index is not None else 0
