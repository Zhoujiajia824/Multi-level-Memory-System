"""中期记忆容量上限与价值淘汰模块（Phase 3）
====================================
* ``MemoryEvictionManager``：容量达阈值时按 ``memory_value_score`` 淘汰低价值记忆（soft delete），
  保护长尾 / 高风险 / 近期高价值 / 每类最少保留量。
* ``MemoryCompactionManager``：inactive 比例过高或淘汰后按需 rebuild FAISS 索引（物理压缩）。

约束
----
* soft delete：仅置 ``is_active=False`` / ``status="deleted"`` / ``deleted_reason`` / ``deleted_at``，
  ``_records`` 保留元数据；FAISS 在 rebuild 时才物理剔除。
* 淘汰在 ``add_record`` 末尾同步触发（数据量小可接受）；只读 active 记录的已算分值，不读未来。
* 不在模块顶层 import mid_term_memory（避免循环），通过传入的 mid_term 实例调用其公开方法。
* 配置驱动（capacity / eviction / compaction 从 ``memory.yaml`` 加载）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("eviction")


def _safe_ratio(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


class MemoryEvictionManager:
    """容量上限 + 价值淘汰（soft delete）。

    Args:
        capacity_cfg: ``memory.yaml -> mid_term.capacity``。
        eviction_cfg: ``memory.yaml -> mid_term.eviction``。
        value_scorer: ``MemoryValueScorer`` 实例（淘汰前重算价值）。
        compaction: ``MemoryCompactionManager`` 实例（可选，淘汰后按需 rebuild）。
    """

    def __init__(
        self,
        capacity_cfg: Optional[Dict[str, Any]] = None,
        eviction_cfg: Optional[Dict[str, Any]] = None,
        value_scorer: Any = None,
        compaction: Optional["MemoryCompactionManager"] = None,
    ):
        cap = capacity_cfg or {}
        self.enabled: bool = bool(cap.get("enabled", True))
        self.max_records: int = int(cap.get("max_records", 5000))
        self.max_disk_mb: float = float(cap.get("max_disk_mb", 2048))
        self.trigger_ratio: float = float(cap.get("eviction_trigger_ratio", 0.80))
        self.target_ratio: float = float(cap.get("eviction_target_ratio", 0.70))
        self.emergency_ratio: float = float(cap.get("emergency_trigger_ratio", 0.95))

        ev = eviction_cfg or {}
        self.strategy: str = ev.get("strategy", "value_based_soft_delete")
        self.protect_long_tail: bool = bool(ev.get("protect_long_tail", True))
        self.protect_high_risk: bool = bool(ev.get("protect_high_risk", True))
        self.protect_recent_high_value: bool = bool(ev.get("protect_recent_high_value", True))
        self.min_keep_per_event_type: Dict[str, int] = dict(ev.get("min_keep_per_event_type", {}) or {})

        self.value_scorer = value_scorer
        self.compaction = compaction

    # ------------------------------------------------------------------
    # add_record 钩子
    # ------------------------------------------------------------------
    def after_add(self, mid_term: Any, now_ts: int = 0) -> List[Tuple[str, str]]:
        """``add_record`` 末尾调用：检查容量，按需淘汰 + 压缩。

        Args:
            mid_term: ``MidTermMemory`` 实例。
            now_ts: 当前时间戳（μs，用于评分 recency 与 deleted_at）。

        Returns:
            ``[(record_id, deleted_reason), ...]`` 本轮淘汰记录（供日志/jsonl 复盘）。
        """
        if not self.enabled:
            return []
        trigger = self._should_trigger(mid_term)
        if trigger == "none":
            return []
        evicted = self.evict(mid_term, now_ts, emergency=(trigger == "emergency"))
        # 淘汰后按需压缩（rebuild_after_eviction 或 inactive 比例过高）
        if evicted and self.compaction is not None:
            self.compaction.maybe_rebuild(mid_term, forced=self.compaction.rebuild_after_eviction)
        return evicted

    # ------------------------------------------------------------------
    # 触发判定
    # ------------------------------------------------------------------
    def _should_trigger(self, mid_term: Any) -> str:
        """返回 ``"none"`` / ``"normal"`` / ``"emergency"``。"""
        active = mid_term.active_size()
        if active >= self.max_records * self.emergency_ratio:
            return "emergency"
        if active >= self.max_records * self.trigger_ratio:
            return "normal"
        if self.max_disk_mb > 0 and self._estimated_disk_mb(mid_term) >= self.max_disk_mb * self.trigger_ratio:
            return "normal"
        return "none"

    def _estimated_disk_mb(self, mid_term: Any) -> float:
        """估算磁盘占用（MB）：active 数 × (特征字节 + ~2KB 元数据)。真磁盘在 save 时才精确。"""
        dim = 768
        try:
            dim = int(getattr(mid_term.faiss_store, "dimension", 768)) or 768
        except Exception:
            pass
        bytes_per_record = dim * 4 + 2048
        return mid_term.active_size() * bytes_per_record / 1e6

    # ------------------------------------------------------------------
    # 淘汰
    # ------------------------------------------------------------------
    def evict(
        self, mid_term: Any, now_ts: int, emergency: bool = False
    ) -> List[Tuple[str, str]]:
        """重算价值 → 排序 → 保护 → soft delete 直到 active <= target。"""
        if self.value_scorer is None:
            logger.warning("eviction: 无 value_scorer，跳过淘汰。")
            return []

        # 1. 重算全部 active 记忆价值
        self.value_scorer.score_all(mid_term, now_ts)
        active = mid_term.get_active_records()
        if not active:
            return []

        target = int(math.floor(self.max_records * self.target_ratio))
        need_evict = len(active) - target
        if need_evict <= 0:
            return []

        # 2. 按 memory_value_score 升序（低价值先淘汰）；同分按 created_at 升序（旧先淘汰）
        ordered = sorted(
            active.items(),
            key=lambda kv: (
                getattr(kv[1], "memory_value_score", 0.0) or 0.0,
                getattr(kv[1], "created_at", 0) or 0,
            ),
        )

        # 3. 每类 active 计数（用于 min_keep 保护）
        active_per_type: Dict[str, int] = {}
        for rec in active.values():
            et = getattr(rec, "event_type", "frame_memory") or "frame_memory"
            active_per_type[et] = active_per_type.get(et, 0) + 1
        deleted_per_type: Dict[str, int] = {}

        evicted: List[Tuple[str, str]] = []
        for rid, rec in ordered:
            if len(evicted) >= need_evict:
                break
            et = getattr(rec, "event_type", "frame_memory") or "frame_memory"
            if self._is_protected(rec, et, active_per_type, deleted_per_type, emergency):
                continue
            reason = self._delete_reason(et, emergency)
            mid_term.soft_delete(rid, reason=reason, deleted_at=now_ts)
            deleted_per_type[et] = deleted_per_type.get(et, 0) + 1
            evicted.append((rid, reason))

        if evicted:
            logger.info(
                "eviction: %s 模式淘汰 %d 条 (active %d→%d, target=%d)；示例: %s",
                "emergency" if emergency else "normal", len(evicted),
                len(active), mid_term.active_size(), target, evicted[:3],
            )
        return evicted

    def _is_protected(
        self, record: Any, event_type: str,
        active_per_type: Dict[str, int], deleted_per_type: Dict[str, int],
        emergency: bool,
    ) -> bool:
        """判定是否受保护（不淘汰）。emergency 模式仅保留 min_keep 保护。"""
        # min_keep_per_event_type：每类至少保留 N 条（emergency 也保留）
        min_keep = self.min_keep_per_event_type.get(event_type, 0)
        if min_keep > 0:
            remaining = active_per_type.get(event_type, 0) - deleted_per_type.get(event_type, 0)
            if remaining <= min_keep:
                return True
        if emergency:
            return False  # 紧急模式：放宽其它保护，强制淘汰到 target
        # 长尾保护
        if self.protect_long_tail and event_type == "long_tail":
            return True
        # 高风险保护（risk_tags 非空）
        if self.protect_high_risk and (getattr(record, "risk_tags", None) or []):
            return True
        # 近期高价值保护（admission 高 且 recency 高）
        if self.protect_recent_high_value:
            adm = getattr(record, "admission_score", None)
            recency = getattr(record, "recency_score", None)
            if adm is not None and float(adm) >= 0.7 and recency is not None and float(recency) >= 0.5:
                return True
        return False

    @staticmethod
    def _delete_reason(event_type: str, emergency: bool) -> str:
        if emergency:
            return "emergency_eviction"
        if event_type in ("normal_cruise", "stable_stop", "redundant_frame"):
            return f"{event_type}_eviction"
        return "low_value_eviction"


class MemoryCompactionManager:
    """FAISS 索引压缩：inactive 比例过高或强制时 rebuild（物理剔除 inactive 向量）。"""

    def __init__(self, compaction_cfg: Optional[Dict[str, Any]] = None):
        cfg = compaction_cfg or {}
        self.rebuild_faiss_when_inactive_ratio: float = float(
            cfg.get("rebuild_faiss_when_inactive_ratio", 0.20)
        )
        self.rebuild_after_eviction: bool = bool(cfg.get("rebuild_after_eviction", False))

    def maybe_rebuild(self, mid_term: Any, forced: bool = False) -> bool:
        """按需 rebuild FAISS。返回是否实际重建。"""
        total = mid_term.size()
        if total <= 0:
            return False
        inactive = mid_term.inactive_size()
        inactive_ratio = _safe_ratio(inactive, total)
        if not forced and inactive_ratio < self.rebuild_faiss_when_inactive_ratio:
            return False
        before = mid_term.faiss_store.size()
        mid_term.rebuild_index()
        after = mid_term.faiss_store.size()
        logger.info(
            "compaction: rebuild FAISS (inactive_ratio=%.3f, forced=%s, faiss %d→%d)",
            inactive_ratio, forced, before, after,
        )
        return True
