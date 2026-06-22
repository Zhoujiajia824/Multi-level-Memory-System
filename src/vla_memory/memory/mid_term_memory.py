"""中期记忆模块
==============
存储历史关键帧和模型决策经验。使用 FAISS 进行联合检索。
联合检索得分公式：
  final_score = visual_weight * visual_score
              + text_weight * text_score
              + scene_weight * scene_score
              + weather_weight * weather_score
              + nav_weight * nav_score
              + state_weight * state_score
必须使用 FAISS，不允许降级到 numpy 检索。
"""
from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from src.vla_memory.schemas.memory import MidTermMemoryRecord
from src.vla_memory.memory.faiss_store import FAISSVectorStore
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("mid_term_memory")


class MidTermMemory:
    """中期记忆管理器。

    使用 FAISS 进行视觉特征检索，结合文本、场景、天气、导航、状态等多维度联合评分。
    P2 新增：可选的磁盘持久化（yaml 开关控制），支持启动时自动加载历史索引。

    Args:
        faiss_store: FAISS 向量存储实例。
        weights: 联合检索权重字典。
        top_k: 返回最相似的 top_k 条结果。
        persistence_cfg: 持久化配置字典（可选）。支持字段：
            - enabled: bool — 是否启用持久化
            - save_on_close: bool — close() 时是否保存
            - auto_load_on_init: bool — 初始化时是否加载已有索引
            - strict_load: bool — 加载失败时抛错还是 warning 后空启动
        save_dir: 持久化目录路径（字符串），供 save_full/load_full 使用。
    """

    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 3,
        persistence_cfg: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
    ):
        self.faiss_store = faiss_store
        self.top_k = top_k

        # 默认权重
        default_weights = {
            "visual_weight": 0.40,
            "text_weight": 0.15,
            "scene_weight": 0.15,
            "weather_weight": 0.05,
            "nav_weight": 0.15,
            "state_weight": 0.10,
        }
        self.weights = weights or default_weights

        # 持久化配置
        self.persistence_cfg: Dict[str, Any] = persistence_cfg or {}
        self.save_dir: Optional[str] = save_dir

        # 记录存储
        self._records: Dict[str, MidTermMemoryRecord] = {}  # record_id -> record
        self._text_corpus: Dict[str, str] = {}  # record_id -> scene_text

        # 自动加载历史索引（若启用）
        if self.persistence_cfg.get("enabled") and self.persistence_cfg.get("auto_load_on_init") and save_dir:
            self._auto_load(save_dir)

    def _auto_load(self, save_dir: str) -> None:
        """初始化时自动加载已有的持久化索引 (安静模式)。"""
        meta_path = Path(save_dir) / "mid_term_meta.json"
        index_path = Path(save_dir) / "mid_term_faiss.index"
        if meta_path.exists():
            try:
                self.load(str(index_path), str(meta_path))
                logger.info("中期记忆从磁盘自动加载: %s (%d条)", save_dir, self.size())
            except Exception as e:
                if self.persistence_cfg.get("strict_load"):
                    raise
                logger.warning("中期记忆自动加载失败 (跳过): %s", e)

    def add_record(self, record: MidTermMemoryRecord, feature: Optional[np.ndarray] = None) -> None:
        """添加一条中期记忆记录。

        Args:
            record: 中期记忆记录。
            feature: 对应的图像特征向量（可选）。
        """
        self._records[record.record_id] = record
        if record.scene_text:
            self._text_corpus[record.record_id] = record.scene_text

        # 如果有特征向量，添加到 FAISS
        if feature is not None:
            self.faiss_store.add(feature.reshape(1, -1), [record.record_id])

        logger.debug(f"中期记忆添加记录: {record.record_id}")

    def search(
        self,
        query_feature: Optional[np.ndarray] = None,
        scene_text: str = "",
        scene_id: str = "",
        weather_id: str = "",
        nav_instruction: str = "",
        ego_state: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """联合检索中期记忆。

        Args:
            query_feature: 当前帧的图像特征向量。
            scene_text: 当前场景描述文本。
            scene_id: 当前场景类型。
            weather_id: 当前天气类型。
            nav_instruction: 当前导航语义。
            ego_state: 当前自车状态。

        Returns:
            检索结果列表，每个元素包含 record、final_score、各子分数。
        """
        if not self._records:
            logger.warning("中期记忆为空，无法检索。")
            return []

        # 1. 视觉相似度（FAISS 检索）
        visual_scores: Dict[str, float] = {}
        if query_feature is not None and self.faiss_store.size() > 0:
            scores, ids = self.faiss_store.search(query_feature, top_k=min(50, self.faiss_store.size()))
            for score, rid in zip(scores, ids):
                visual_scores[rid] = float(score)

        # 2. 对每条记录计算综合得分
        candidates = []
        for rid, record in self._records.items():
            sub_scores = {}

            # 视觉分数
            sub_scores["visual_score"] = visual_scores.get(rid, 0.0)

            # 文本相似度
            sub_scores["text_score"] = self._compute_text_similarity(
                scene_text, self._text_corpus.get(rid, "")
            )

            # 场景匹配
            sub_scores["scene_score"] = 1.0 if scene_id and record.scene_id == scene_id else 0.0

            # 天气匹配
            sub_scores["weather_score"] = 1.0 if weather_id and record.weather_id == weather_id else 0.0

            # 导航语义匹配
            sub_scores["nav_score"] = 1.0 if nav_instruction and record.nav_instruction == nav_instruction else 0.0

            # 状态相似度
            sub_scores["state_score"] = self._compute_state_similarity(ego_state, record.ego_state)

            # 计算加权总分
            final_score = sum(
                self.weights.get(f"{key}_weight", 0.0) * score
                for key, score in sub_scores.items()
            )

            candidates.append({
                "record": record,
                "final_score": final_score,
                "sub_scores": sub_scores,
            })

        # 按总分排序
        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        return candidates[:self.top_k]

    @staticmethod
    def _compute_text_similarity(text1: str, text2: str) -> float:
        """计算两个文本的简单相似度（基于词汇重叠率）。

        第一版使用简单的词汇重叠率。后续可替换为 TF-IDF。
        """
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _compute_state_similarity(
        state1: Optional[Dict], state2: Optional[Dict]
    ) -> float:
        """计算两个自车状态的相似度。

        基于速度、加速度、航向角的差异计算相似度。
        """
        if not state1 or not state2:
            return 0.0

        # 速度相似度
        speed1 = state1.get("speed", 0.0)
        speed2 = state2.get("speed", 0.0)
        speed_diff = abs(speed1 - speed2)
        speed_sim = max(0, 1.0 - speed_diff / 30.0)  # 30 m/s 为归一化因子

        # 加速度相似度
        acc1 = state1.get("acceleration", 0.0)
        acc2 = state2.get("acceleration", 0.0)
        acc_diff = abs(acc1 - acc2)
        acc_sim = max(0, 1.0 - acc_diff / 5.0)  # 5 m/s² 为归一化因子

        # 航向角相似度
        yaw1 = state1.get("yaw", 0.0)
        yaw2 = state2.get("yaw", 0.0)
        yaw_diff = abs(yaw1 - yaw2)
        yaw_diff = min(yaw_diff, 2 * math.pi - yaw_diff)  # 角度差归一化
        yaw_sim = max(0, 1.0 - yaw_diff / math.pi)

        # 加权平均
        return 0.4 * speed_sim + 0.3 * acc_sim + 0.3 * yaw_sim

    def get_all_records(self) -> Dict[str, MidTermMemoryRecord]:
        """获取所有中期记忆记录。"""
        return self._records

    def size(self) -> int:
        """获取记录数量。"""
        return len(self._records)

    # ================================================================
    # 持久化
    # ================================================================

    def save(self, index_path: str, meta_path: str) -> None:
        """保存中期记忆（FAISS 索引 + 元数据）到磁盘。

        Args:
            index_path: FAISS 索引文件路径（如 outputs/memory_db/mid_term_faiss.index）。
            meta_path: 元数据 JSON 文件路径（如 outputs/memory_db/mid_term_meta.json）。
        """
        from src.vla_memory.memory.memory_record_io import save_mid_term_meta

        # 保存 FAISS 索引
        if self.faiss_store.size() > 0:
            self.faiss_store.save(index_path)
        else:
            logger.warning("FAISS 索引为空，跳过索引保存。")

        # 保存元数据（MemoryRecord 列表）
        save_mid_term_meta(self._records, meta_path)
        logger.info(
            f"中期记忆已保存: {self.size()} 条记录, "
            f"索引={index_path}, 元数据={meta_path}"
        )

    def load(self, index_path: str, meta_path: str) -> None:
        """从磁盘加载中期记忆（FAISS 索引 + 元数据）。

        加载后会清空当前内存中的记录和文本语料库，替换为加载的数据。

        Args:
            index_path: FAISS 索引文件路径。
            meta_path: 元数据 JSON 文件路径。
        """
        from src.vla_memory.memory.memory_record_io import load_mid_term_meta

        idx_path = Path(index_path)

        # 加载 FAISS 索引
        if idx_path.exists():
            self.faiss_store.load(str(idx_path))
            logger.info(f"FAISS 索引已加载: {self.faiss_store.size()} 条向量")
        else:
            logger.warning(f"FAISS 索引文件不存在: {idx_path}")

        # 加载元数据并重建文本语料库
        self._records = load_mid_term_meta(str(meta_path))
        self._text_corpus = {
            rid: rec.scene_text
            for rid, rec in self._records.items()
            if hasattr(rec, "scene_text") and rec.scene_text
        }
        logger.info(f"中期记忆已加载: {self.size()} 条记录")

    # ----------------------------------------------------------------
    # P2 持久化封装 ─ 让 pipeline 可以"一键全量保存"
    # ----------------------------------------------------------------

    def save_full(self, save_dir: Optional[str] = None) -> None:
        """全量持久化中期记忆到目录。

        默认文件名：
          - mid_term_faiss.index  + mid_term_faiss.ids.json
          - mid_term_meta.json

        Args:
            save_dir: 目标目录。None 时使用 self.save_dir。
        """
        target = save_dir or self.save_dir
        if not target:
            logger.warning("save_full 调用时未提供 save_dir，跳过保存。")
            return
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)
        index_path = target_path / "mid_term_faiss.index"
        meta_path = target_path / "mid_term_meta.json"
        self.save(str(index_path), str(meta_path))

    def close(self) -> None:
        """关闭中期记忆。若 persistence.enabled 且 save_on_close，则全量落盘。"""
        if self.persistence_cfg.get("enabled") and self.persistence_cfg.get("save_on_close", True):
            self.save_full()
        else:
            logger.debug("close(): 持久化未启用或 save_on_close=False，跳过保存。")
