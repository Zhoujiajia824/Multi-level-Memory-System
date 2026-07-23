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
        retrieval_cfg: Optional[Dict[str, Any]] = None,
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

        # Phase 4 价值感知检索配置（重排/多样性/过滤/候选池；None=退化为仅相似度+过滤 inactive）
        self._retrieval_cfg: Dict[str, Any] = retrieval_cfg or {}

        # 记录存储
        self._records: Dict[str, MidTermMemoryRecord] = {}  # record_id -> record
        self._text_corpus: Dict[str, str] = {}  # record_id -> scene_text

        # Phase 3 容量管理钩子（依赖注入，None=不启用；由 online_loop.setup 注入）
        self._eviction_manager = None
        self._value_scorer = None
        self._compaction_manager = None

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

        # Phase 3：写入后按需触发容量淘汰（soft delete + 可能 rebuild）。
        # 仅在 add 末尾触发，不前移到检索前，先读后写不变。
        if self._eviction_manager is not None:
            try:
                self._eviction_manager.after_add(self, now_ts=int(getattr(record, "created_at", 0) or 0))
            except Exception as e:
                logger.warning("容量淘汰 after_add 异常 (frame=%s): %s", record.record_id, e)

    def search(
        self,
        query_feature: Optional[np.ndarray] = None,
        scene_text: str = "",
        scene_id: str = "",
        weather_id: str = "",
        nav_instruction: str = "",
        ego_state: Optional[Dict] = None,
        now_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """价值感知联合检索（Phase 4）。

        流程：FAISS 视觉相似 → 过滤 inactive/deleted/deprecated/低置信 → 6 路相似度 final_score
        → 候选池 top-N → 价值重排 value_aware_score → 多样性约束 → top-K → 更新命中统计。

        * inactive（is_active=False）**始终过滤**（soft delete 硬约束，验收要求不返回 inactive）。
        * ``enable_value_rerank=false`` 或无 retrieval_cfg → 退化为仅相似度排序（向后兼容）。
        * 命中统计在结果算完后更新，不改本次结果，不破坏先读后写。

        Args:
            query_feature: 当前帧图像特征向量。
            scene_text: 当前场景描述文本。
            scene_id: 当前场景类型。
            weather_id: 当前天气类型。
            nav_instruction: 当前导航语义。
            ego_state: 当前自车状态。
            now_ts: 当前帧时间戳（μs）；非 None 时更新命中统计，None 不更新（向后兼容）。

        Returns:
            ``{"results": [...], "stats": {...}}``。results 每项含 record / final_score /
            value_aware_score / memory_value_score / event_type / status / sub_scores；
            stats 含 candidate_count / active_candidate_count / filtered_count。
        """
        stats: Dict[str, int] = {"candidate_count": 0, "active_candidate_count": 0, "filtered_count": 0}
        if not self._records:
            logger.warning("中期记忆为空，无法检索。")
            return {"results": [], "stats": stats}

        rcfg = self._retrieval_cfg
        filters = (rcfg.get("filters", {}) or {}) if rcfg else {}
        exclude_deleted = bool(filters.get("exclude_deleted", True))
        exclude_deprecated = bool(filters.get("exclude_deprecated", False))
        min_confidence = float(filters.get("min_confidence_score", 0.0))
        pool_size = int((rcfg.get("candidate_pool_size", 20)) if rcfg else 20)
        top_k = int((rcfg.get("top_k", self.top_k)) if rcfg else self.top_k)
        enable_rerank = bool(rcfg.get("enable_value_rerank", True)) if rcfg else False
        rw = (rcfg.get("rerank_weights", {}) or {}) if rcfg else {}
        w_sim = float(rw.get("similarity_score", 0.80))
        w_val = float(rw.get("memory_value_score", 0.20))
        div_cfg = (rcfg.get("diversity", {}) or {}) if rcfg else {}
        div_enabled = bool(div_cfg.get("enabled", True)) if rcfg else False
        max_per_event = int(div_cfg.get("max_per_event_id", 1))
        max_per_scene = int(div_cfg.get("max_per_scene_token", 2))
        suppress_dup = bool(div_cfg.get("suppress_near_duplicates", True))
        dup_thr = float(div_cfg.get("duplicate_similarity_threshold", 0.95))
        # Phase 5：检索优先 event_memory（value_aware 加成 + 同分时 event 居前）
        prefer_event = bool(rcfg.get("prefer_event_memory", False)) if rcfg else False
        event_bonus = float((rcfg.get("event_memory_bonus", 0.10)) if rcfg else 0.10)

        # 1. 视觉相似度（FAISS，取足够候选覆盖候选池）
        visual_scores: Dict[str, float] = {}
        if query_feature is not None and self.faiss_store.size() > 0:
            k_faiss = min(max(pool_size, 50), self.faiss_store.size())
            scores, ids = self.faiss_store.search(query_feature, top_k=k_faiss)
            for score, rid in zip(scores, ids):
                visual_scores[rid] = float(score)

        # 2. 过滤 + 6 路相似度 final_score（inactive 始终跳过）
        candidates = []
        for rid, record in self._records.items():
            if not getattr(record, "is_active", True):
                continue  # 硬约束：不返回 inactive
            stats["candidate_count"] += 1
            if exclude_deleted and getattr(record, "status", "") == "deleted":
                stats["filtered_count"] += 1
                continue
            # Phase 6：exclude_deprecated 按 status 过滤（deprecated/superseded）；默认 false=仍返回
            if exclude_deprecated and getattr(record, "status", "") in ("deprecated", "superseded"):
                stats["filtered_count"] += 1
                continue
            conf = getattr(record, "confidence_score", None)
            if conf is not None and float(conf) < min_confidence:
                stats["filtered_count"] += 1
                continue

            sub_scores = {}
            sub_scores["visual_score"] = visual_scores.get(rid, 0.0)
            sub_scores["text_score"] = self._compute_text_similarity(scene_text, self._text_corpus.get(rid, ""))
            sub_scores["scene_score"] = 1.0 if scene_id and record.scene_id == scene_id else 0.0
            sub_scores["weather_score"] = 1.0 if weather_id and record.weather_id == weather_id else 0.0
            sub_scores["nav_score"] = 1.0 if nav_instruction and record.nav_instruction == nav_instruction else 0.0
            sub_scores["state_score"] = self._compute_state_similarity(ego_state, record.ego_state)
            final_score = sum(self.weights.get(f"{key}_weight", 0.0) * s for key, s in sub_scores.items())

            mvs = getattr(record, "memory_value_score", None)
            candidates.append({
                "record": record,
                "final_score": final_score,
                "sub_scores": sub_scores,
                "memory_value_score": float(mvs) if mvs is not None else None,
                "value_aware_score": final_score,  # 默认=相似度；重排后覆盖
                "event_type": getattr(record, "event_type", "frame_memory") or "frame_memory",
                "status": getattr(record, "status", "active"),
                "memory_type": getattr(record, "memory_type", "frame_memory") or "frame_memory",
                "source_scene_token": getattr(record, "source_scene_token", "") or "",
            })

        stats["active_candidate_count"] = len(candidates)
        if not candidates:
            return {"results": [], "stats": stats}

        # 3. 候选池：按相似度 final_score 取 top-N
        candidates.sort(key=lambda c: c["final_score"], reverse=True)
        pool = candidates[:pool_size]

        # 4. 价值感知重排：value_aware_score = w_sim·final_score + w_val·memory_value_score
        #    Phase 5：prefer_event_memory 时 event_memory 加 bonus；同分 event 居前。
        if enable_rerank:
            for c in pool:
                mvs = c["memory_value_score"] if c["memory_value_score"] is not None else 0.0
                score = w_sim * c["final_score"] + w_val * mvs
                if prefer_event and c["memory_type"] == "event_memory":
                    score += event_bonus
                c["value_aware_score"] = score
        # 排序：value_aware 降序；prefer_event 时同分让 event_memory 居前
        if prefer_event:
            pool.sort(key=lambda c: (-c["value_aware_score"], 0 if c["memory_type"] == "event_memory" else 1))
        else:
            pool.sort(key=lambda c: c["value_aware_score"], reverse=True)

        # 5. 多样性约束 → top-K（避免同一 event/scene 占据 top-k，抑制近重复）
        if div_enabled:
            selected = []
            event_cnt: Dict[str, int] = {}
            scene_cnt: Dict[str, int] = {}
            for c in pool:
                et = c["event_type"]
                st = c["source_scene_token"]
                if event_cnt.get(et, 0) >= max_per_event:
                    continue
                if st and scene_cnt.get(st, 0) >= max_per_scene:
                    continue
                if suppress_dup:
                    is_dup = any(
                        s["event_type"] == et
                        and abs(s["final_score"] - c["final_score"]) < (1.0 - dup_thr)
                        for s in selected
                    )
                    if is_dup:
                        continue
                selected.append(c)
                event_cnt[et] = event_cnt.get(et, 0) + 1
                if st:
                    scene_cnt[st] = scene_cnt.get(st, 0) + 1
                if len(selected) >= top_k:
                    break
            results = selected
        else:
            results = pool[:top_k]

        # 6. 命中统计（元数据簿记，不改本次结果，不破坏先读后写）
        if now_ts is not None:
            for r in results:
                rec = r["record"]
                try:
                    rec.hit_count = int(getattr(rec, "hit_count", 0)) + 1
                    rec.last_retrieved_at = now_ts
                except Exception as e:
                    logger.debug("命中统计更新失败 (rid=%s): %s", getattr(rec, "record_id", "?"), e)

        return {"results": results, "stats": stats}

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
        """获取所有中期记忆记录（含 inactive，保留元数据）。"""
        return self._records

    def size(self) -> int:
        """获取记录数量（含 inactive；用于 compaction 的 inactive_ratio 分母）。"""
        return len(self._records)

    # ================================================================
    # Phase 3：容量管理 / soft delete / rebuild / 依赖注入
    # ================================================================

    def active_size(self) -> int:
        """active 记录数（is_active=True）；容量上限判定用此。"""
        return sum(1 for r in self._records.values() if getattr(r, "is_active", True))

    def inactive_size(self) -> int:
        """inactive 记录数（soft delete 待 rebuild）。"""
        return sum(1 for r in self._records.values() if not getattr(r, "is_active", True))

    def get_active_records(self) -> Dict[str, MidTermMemoryRecord]:
        """获取 active 记录（is_active=True）。"""
        return {rid: r for rid, r in self._records.items() if getattr(r, "is_active", True)}

    def soft_delete(self, record_id: str, reason: str, deleted_at: int = 0) -> bool:
        """soft delete：置 is_active=False / status=deleted / deleted_reason / deleted_at。

        不物理删除元数据（``_records`` 保留），仅从 ``_text_corpus`` 移除（检索不再用其文本）。
        FAISS 向量在 ``rebuild_index`` 时才物理剔除。

        Returns:
            是否实际执行（记录存在且原本 active）。
        """
        rec = self._records.get(record_id)
        if rec is None or not getattr(rec, "is_active", True):
            return False
        rec.is_active = False
        rec.status = "deleted"
        rec.deleted_reason = reason
        rec.deleted_at = deleted_at
        self._text_corpus.pop(record_id, None)
        logger.debug("soft_delete: %s reason=%s", record_id, reason)
        return True

    def rebuild_index(self) -> None:
        """重建 FAISS 索引：仅保留 active 记录的向量（物理剔除 inactive）。

        IndexFlatIP 不支持原生删除，故用 ``reconstruct_n`` 取回全部向量 → 过滤 active →
        新建 IndexFlatIP 重新 add + 重写 _ids。inactive 元数据仍保留在 ``_records``。
        """
        store = self.faiss_store
        n = store.size()
        if n == 0:
            return
        active_ids = set(self.get_active_records().keys())
        try:
            all_vecs = store._index.reconstruct_n(0, n)  # (n, dim) float32
        except Exception as e:
            logger.warning("rebuild_index: reconstruct_n 失败，跳过重建: %s", e)
            return
        keep_idx = [i for i, rid in enumerate(store._ids) if rid in active_ids]
        if not keep_idx:
            # 没有活跃向量：重置为空索引
            store._init_index()
            store._ids = []
            return
        import numpy as _np
        keep_vecs = _np.ascontiguousarray(all_vecs[keep_idx], dtype=_np.float32)
        keep_ids = [store._ids[i] for i in keep_idx]
        store._init_index()  # 新建空 IndexFlatIP
        store._index.add(keep_vecs)
        store._ids = keep_ids
        logger.info(
            "rebuild_index: FAISS 重建完成 (active=%d, 剔除 inactive=%d)",
            len(keep_ids), n - len(keep_ids),
        )

    def set_eviction_manager(self, mgr) -> None:
        """注入容量淘汰管理器（online_loop.setup 调用）。None=禁用淘汰。"""
        self._eviction_manager = mgr

    def set_value_scorer(self, scorer) -> None:
        """注入存量价值评分器。"""
        self._value_scorer = scorer

    def set_compaction_manager(self, mgr) -> None:
        """注入 FAISS 压缩管理器。"""
        self._compaction_manager = mgr

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
