"""中期记忆存量价值评分模块（Phase 3）
====================================
``MemoryValueScorer`` 对**已存记忆**计算"持续价值" ``memory_value_score``，作为容量淘汰的排序依据。

区别于阶段 2 的 ``MemoryAdmissionController``（写入时价值，决定是否入库）：本评分在淘汰前对
全部 active 记忆重算，综合写入价值 / 场景高价值 / 近期性 / 检索效用 / 冗余 / 置信 / 冲突 / 低价值惩罚。

约束
----
* 纯逻辑，无 IO；``score_all`` 仅读写传入 mid_term 实例的 record 字段（元数据更新）。
* 不依赖 faiss；冗余用 ``(scene_id, event_type)`` 分组频率近似，避免 O(n²) 两两相似度。
* 配置驱动（权重 / 阈值从 ``memory.yaml -> mid_term.eviction`` 加载）。

评分公式（权重配置驱动，默认和≈1）::

    memory_value_score = w_adm·admission + w_evt·event_highvalue + w_rec·recency
                       + w_ret·retrieval_utility + w_red·(1−redundancy) + w_conf·confidence
                       − w_confl·conflict − w_low·lowvalue_penalty   (裁剪到 [0,1])
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("value_scorer")

# 高价值 event_type（来自阶段 2 admission 的 18 类，剔除低价值/中性类）
_HIGH_VALUE_EVENT_TYPES = {
    "lane_change", "start", "hard_brake", "hard_acceleration", "obstacle_avoidance",
    "intersection", "dense_traffic", "pedestrian_interaction", "cyclist_interaction",
    "cut_in", "merge", "turn_left", "turn_right", "crosswalk", "decision_change",
    "occlusion", "ghost_probing_risk", "long_tail",
}
# 低价值 event_type（淘汰优先）
_LOW_VALUE_EVENT_TYPES = {"normal_cruise", "stable_stop", "redundant_frame"}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class MemoryValueScorer:
    """存量记忆价值评分器（配置驱动）。

    Args:
        cfg: ``memory.yaml -> mid_term.eviction`` 字典（含 weights / thresholds）。
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        w = dict(cfg.get("weights", {}) or {})
        self.weights: Dict[str, float] = {
            "admission": float(w.get("admission", 0.25)),
            "event_highvalue": float(w.get("event_highvalue", 0.20)),
            "recency": float(w.get("recency", 0.15)),
            "retrieval_utility": float(w.get("retrieval_utility", 0.15)),
            "redundancy": float(w.get("redundancy", 0.10)),
            "confidence": float(w.get("confidence", 0.10)),
            "conflict": float(w.get("conflict", 0.05)),
            "lowvalue_penalty": float(w.get("lowvalue_penalty", 0.10)),
        }
        thr = cfg.get("thresholds", {}) or {}
        self.thresholds: Dict[str, float] = {
            "recency_half_life_seconds": float(thr.get("recency_half_life_seconds", 300.0)),
            "high_value_admission": float(thr.get("high_value_admission", 0.7)),
            "retrieval_hit_cap": float(thr.get("retrieval_hit_cap", 5.0)),
            "redundancy_group_cap": float(thr.get("redundancy_group_cap", 20.0)),
            "conflict_cap": float(thr.get("conflict_cap", 3.0)),
        }

    # ------------------------------------------------------------------
    # 单条打分
    # ------------------------------------------------------------------
    def score_record(
        self,
        record: Any,
        now_ts: int,
        redundancy_ctx: Dict[Tuple[str, str], int],
    ) -> Dict[str, float]:
        """对单条记忆算各子分与 memory_value_score（不写回 record）。

        Args:
            record: ``MidTermMemoryRecord``（鸭子类型，需有 admission_score/event_type/scene_id/
                hit_count/last_retrieved_at/created_at/conflict_count/behavior/trajectory/risk_tags）。
            now_ts: 当前帧时间戳（μs），用于 recency。
            redundancy_ctx: ``(scene_id, event_type) → 该组 active 记录数``。

        Returns:
            dict：admission / event_highvalue / recency / retrieval_utility / redundancy /
            confidence / conflict / lowvalue_penalty / memory_value_score。
        """
        thr = self.thresholds
        wg = self.weights

        # admission（写入价值，0~1；legacy None→1.0）
        adm = record.admission_score
        adm = 1.0 if adm is None else _clamp(float(adm))

        # event_highvalue：高价值事件→1，frame_memory→0.5，低价值→0；risk_tags 非空再加成
        et = getattr(record, "event_type", "frame_memory") or "frame_memory"
        if et in _HIGH_VALUE_EVENT_TYPES:
            evt = 1.0
        elif et in _LOW_VALUE_EVENT_TYPES:
            evt = 0.0
        else:
            evt = 0.5
        risk_tags = getattr(record, "risk_tags", None) or []
        if risk_tags:
            evt = min(1.0, evt + 0.2)

        # recency：last_retrieved_at 优先，否则 created_at；指数衰减
        ref_ts = getattr(record, "last_retrieved_at", None)
        if ref_ts is None:
            ref_ts = getattr(record, "created_at", 0) or 0
        age_s = max(0.0, (now_ts - ref_ts) / 1e6) if now_ts and ref_ts else 0.0
        half = thr["recency_half_life_seconds"]
        recency = math.exp(-math.log(2) * age_s / half) if half > 0 else 1.0

        # retrieval_utility：hit_count 归一化
        hit = int(getattr(record, "hit_count", 0) or 0)
        retrieval_utility = _clamp(hit / thr["retrieval_hit_cap"]) if thr["retrieval_hit_cap"] > 0 else 0.0

        # redundancy：(scene_id, event_type) 组内活跃数 → 越多越冗余
        scene_id = getattr(record, "scene_id", "") or ""
        key = (scene_id, et)
        group_size = redundancy_ctx.get(key, 1)
        redundancy = _clamp((group_size - 1) / thr["redundancy_group_cap"]) if thr["redundancy_group_cap"] > 0 else 0.0

        # confidence：数据质量（behavior 非 UNKNOWN + trajectory 非空）
        behavior = getattr(record, "behavior", "") or ""
        trajectory = getattr(record, "trajectory", None) or []
        conf = 1.0
        if behavior == "UNKNOWN":
            conf = 0.2
        if not trajectory:
            conf = min(conf, 0.5)

        # conflict：冲突次数归一化
        conflict = _clamp(int(getattr(record, "conflict_count", 0) or 0) / thr["conflict_cap"]) if thr["conflict_cap"] > 0 else 0.0

        # lowvalue_penalty：低价值事件→1（扣分）
        lowvalue_penalty = 1.0 if et in _LOW_VALUE_EVENT_TYPES else 0.0

        memory_value_score = _clamp(
            wg["admission"] * adm
            + wg["event_highvalue"] * evt
            + wg["recency"] * recency
            + wg["retrieval_utility"] * retrieval_utility
            + wg["redundancy"] * (1.0 - redundancy)
            + wg["confidence"] * conf
            - wg["conflict"] * conflict
            - wg["lowvalue_penalty"] * lowvalue_penalty
        )

        return {
            "admission": round(adm, 4),
            "event_highvalue": round(evt, 4),
            "recency": round(recency, 4),
            "retrieval_utility": round(retrieval_utility, 4),
            "redundancy": round(redundancy, 4),
            "confidence": round(conf, 4),
            "conflict": round(conflict, 4),
            "lowvalue_penalty": round(lowvalue_penalty, 4),
            "memory_value_score": round(memory_value_score, 4),
        }

    # ------------------------------------------------------------------
    # 批量打分并写回
    # ------------------------------------------------------------------
    def score_all(self, mid_term: Any, now_ts: int) -> None:
        """对 mid_term 全部 active 记忆重算 memory_value_score + 子分并写回 record。

        Args:
            mid_term: ``MidTermMemory`` 实例（需有 ``get_active_records()``）。
            now_ts: 当前帧时间戳（μs）。
        """
        active = mid_term.get_active_records()
        if not active:
            return
        # 冗余上下文：(scene_id, event_type) → 组内 active 计数
        redundancy_ctx: Dict[Tuple[str, str], int] = {}
        for rec in active.values():
            scene_id = getattr(rec, "scene_id", "") or ""
            et = getattr(rec, "event_type", "frame_memory") or "frame_memory"
            k = (scene_id, et)
            redundancy_ctx[k] = redundancy_ctx.get(k, 0) + 1

        for rid, rec in active.items():
            sub = self.score_record(rec, now_ts, redundancy_ctx)
            rec.memory_value_score = sub["memory_value_score"]
            rec.recency_score = sub["recency"]
            rec.retrieval_utility = sub["retrieval_utility"]
            rec.confidence_score = sub["confidence"]
            rec.redundancy_score = sub["redundancy"]
            rec.updated_at = now_ts
        logger.debug(
            "value_scorer: 重算 %d 条 active 记忆 (now_ts=%d)", len(active), now_ts,
        )
