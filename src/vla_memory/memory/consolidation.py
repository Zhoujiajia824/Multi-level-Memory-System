"""中期记忆沉淀为长期记忆候选（Phase 7）
================================================
``MemoryConsolidationManager`` 从中期记忆库挖掘高价值、稳定、可泛化的经验，总结为长期记忆**候选**
规则（``status=pending_review``），写入单独文件，**不自动覆盖**正式长期记忆规则库（需人工审核晋升）。

流程：筛选高价值 active ``source_memory_type`` 记忆 → 按 ``(event_type, risk_tags)`` 分组 →
找多次出现（≥ ``min_evidence_count``）且平均价值≥阈值的组 → 生成候选规则（condition /
recommended_strategy / rationale / evidence memory_ids / confidence / safety_guard）→
安全过滤（剔除危险驾驶偏好）→ 按 ``min_confidence`` 过滤 → 保存 YAML。

约束
----
* 离线批处理：只读 ``mid_term_meta.json`` 元数据，**不依赖 faiss / VLM**。
* 不写正式长期记忆库（``data/knowledge/long_term_rules.yaml``）；候选写 ``output_path``，待人工审核。
* 用户风格候选不得覆盖安全规则；所有候选带 ``safety_guard.must_not_override``；危险偏好（如高风险变道）剔除。
* ``memory_value_score`` 缺失时回退 ``admission_score``（文档明示），便于未经历淘汰评分的库也能沉淀。
* 配置驱动（``mid_term.consolidation``）。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("consolidation")

# 候选 rule_id 前缀映射
_PREFIX_MAP = {
    "occlusion": "OCCLUSION", "ghost_probing_risk": "GHOST_PROBING",
    "pedestrian_interaction": "PEDESTRIAN", "cyclist_interaction": "CYCLIST",
    "cut_in": "CUT_IN", "lane_change": "LANE_CHANGE", "hard_brake": "HARD_BRAKE",
    "hard_acceleration": "HARD_ACCEL", "start": "START", "intersection": "INTERSECTION",
    "merge": "MERGE", "turn_left": "TURN_LEFT", "turn_right": "TURN_RIGHT",
    "crosswalk": "CROSSWALK", "obstacle_avoidance": "OBSTACLE_AVOIDANCE",
    "dense_traffic": "DENSE_TRAFFIC", "long_tail": "LONG_TAIL", "decision_change": "DECISION_CHANGE",
}
# 场景策略类 event_type（用于 candidate_type 分类）
_SCENE_STRATEGY_EVENTS = {
    "intersection", "lane_change", "cut_in", "merge", "crosswalk", "turn_left", "turn_right",
    "obstacle_avoidance", "dense_traffic", "pedestrian_interaction", "cyclist_interaction",
}
# 减速类 behavior（target_speed_adjustment=reduce）
_REDUCE_BEHAVIORS = {"SLOW_DOWN", "STOP", "YIELD", "hard_brake"}
# 加速类 behavior
_INCREASE_BEHAVIORS = {"hard_acceleration", "start"}
# 危险偏好：高风险 + 激进变道 → 剔除
_AGGRESSIVE_LATERAL = {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}
# 不参与沉淀的 status（已被降权/取代/删除）
_EXCLUDED_STATUS = {"deprecated", "superseded", "deleted", "inactive"}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class MemoryConsolidationManager:
    """中期记忆 → 长期记忆候选沉淀器（配置驱动）。

    Args:
        cfg: ``memory.yaml -> mid_term.consolidation`` 字典。
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.source_memory_type: str = cfg.get("source_memory_type", "event_memory")
        self.min_evidence_count: int = int(cfg.get("min_evidence_count", 3))
        self.min_average_value: float = float(cfg.get("min_average_memory_value_score", 0.70))
        self.min_confidence: float = float(cfg.get("min_confidence", 0.65))
        self.output_path: str = cfg.get("output_path", "outputs/long_term_candidates/candidate_rules.yaml")
        self.auto_promote: bool = bool(cfg.get("auto_promote_to_long_term", False))
        self._rule_counter: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def consolidate(self, records: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从中期记忆库生成候选规则列表。

        Args:
            records: ``record_id -> MidTermMemoryRecord``（来自 ``load_mid_term_meta``）。

        Returns:
            候选规则 dict 列表（每个含 rule_id / condition / recommended_strategy /
            evidence / confidence / safety_guard，status=pending_review）。
        """
        if not self.enabled:
            logger.info("consolidation: 已禁用，跳过。")
            return []
        self._rule_counter = {}

        # 1. 筛选 eligible 记录 + 分组
        groups: Dict[Tuple[str, frozenset], List[Any]] = {}
        eligible = 0
        for rec in records.values():
            if not self._is_eligible(rec):
                continue
            eligible += 1
            key = self._group_key(rec)
            groups.setdefault(key, []).append(rec)
        logger.info("consolidation: eligible=%d, groups=%d", eligible, len(groups))

        # 2. 每组生成候选
        candidates: List[Dict[str, Any]] = []
        for key, members in groups.items():
            if len(members) < self.min_evidence_count:
                continue
            avg_value = sum(self._effective_value(r) for r in members) / len(members)
            if avg_value < self.min_average_value:
                continue
            cand = self._build_candidate(key, members, avg_value)
            if cand["confidence"] < self.min_confidence:
                continue
            if self._is_dangerous(cand):
                logger.info("consolidation: 剔除危险偏好候选 %s (%s)", cand["rule_id"],
                            cand["recommended_strategy"])
                continue
            candidates.append(cand)
        logger.info("consolidation: 生成 %d 条候选规则", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # 筛选 / 分组 / 价值
    # ------------------------------------------------------------------
    def _is_eligible(self, rec: Any) -> bool:
        if getattr(rec, "memory_type", "frame_memory") != self.source_memory_type:
            return False
        if not getattr(rec, "is_active", True):
            return False
        if getattr(rec, "status", "active") in _EXCLUDED_STATUS:
            return False
        return True

    def _effective_value(self, rec: Any) -> float:
        """memory_value_score 优先；缺失回退 admission_score；再缺失 0.5。"""
        mvs = getattr(rec, "memory_value_score", None)
        if mvs is not None:
            return float(mvs)
        adm = getattr(rec, "admission_score", None)
        if adm is not None:
            return float(adm)
        return 0.5

    @staticmethod
    def _group_key(rec: Any) -> Tuple[str, frozenset]:
        et = getattr(rec, "event_type", "frame_memory") or "frame_memory"
        risk_tags = frozenset(sorted(getattr(rec, "risk_tags", []) or []))
        return (et, risk_tags)

    # ------------------------------------------------------------------
    # 候选生成
    # ------------------------------------------------------------------
    def _build_candidate(
        self, key: Tuple[str, frozenset], members: List[Any], avg_value: float
    ) -> Dict[str, Any]:
        event_type, risk_tags = key
        count = len(members)

        # 主导 behavior（众数）
        behaviors = [getattr(r, "behavior", "") or "" for r in members if getattr(r, "behavior", "")]
        dom_behavior = Counter(behaviors).most_common(1)[0][0] if behaviors else ""

        # scene_tags 并集
        scene_union = set()
        for r in members:
            scene_union.update(getattr(r, "scene_tags", []) or [])

        # risk_level：有 risk_tags → high，否则 medium
        risk_level = "high" if risk_tags else "medium"

        # confidence = 0.5·avg_value + 0.5·min(1, count/5)
        confidence = _clamp01(0.5 * avg_value + 0.5 * min(1.0, count / 5.0))

        cand_type = self._classify_type(event_type, risk_tags)
        prefix = _PREFIX_MAP.get(event_type, event_type.upper() or "UNKNOWN")
        self._rule_counter[prefix] = self._rule_counter.get(prefix, 0) + 1
        rule_id = f"CANDIDATE_RULE_{prefix}_{self._rule_counter[prefix]:03d}"

        return {
            "rule_id": rule_id,
            "candidate_type": cand_type,  # safety / strategy / style
            "source": "mid_term_memory_consolidation",
            "status": "pending_review",
            "condition": {
                "scene_tags": sorted(scene_union),
                "event_types": [event_type],
                "risk_tags": sorted(risk_tags),
            },
            "recommended_strategy": {
                "behavior": dom_behavior.lower() if dom_behavior else "",
                "risk_level": risk_level,
                "target_speed_adjustment": self._speed_adjustment(dom_behavior),
            },
            "rationale": self._rationale(event_type, risk_tags, count, dom_behavior),
            "evidence": {
                "memory_ids": [
                    getattr(r, "memory_id", "") or getattr(r, "record_id", "") for r in members
                ],
                "evidence_count": count,
            },
            "confidence": round(confidence, 4),
            "average_memory_value_score": round(avg_value, 4),
            "safety_guard": {
                "must_not_override": ["traffic_rules", "collision_avoidance"],
                "requires_human_review": True,
            },
        }

    @staticmethod
    def _classify_type(event_type: str, risk_tags: frozenset) -> str:
        if risk_tags:
            return "safety"
        if event_type in _SCENE_STRATEGY_EVENTS:
            return "strategy"
        return "style"

    @staticmethod
    def _speed_adjustment(behavior: str) -> str:
        if behavior in _REDUCE_BEHAVIORS:
            return "reduce"
        if behavior in _INCREASE_BEHAVIORS:
            return "increase"
        return "maintain"

    @staticmethod
    def _is_dangerous(cand: Dict[str, Any]) -> bool:
        """安全过滤：剔除危险驾驶偏好（高风险 + 激进变道）。"""
        strat = cand.get("recommended_strategy", {}) or {}
        return (
            strat.get("risk_level") == "high"
            and strat.get("behavior") in {b.lower() for b in _AGGRESSIVE_LATERAL}
        )

    @staticmethod
    def _rationale(event_type: str, risk_tags: frozenset, count: int, dom_behavior: str) -> str:
        cond_parts = []
        if risk_tags:
            cond_parts.append("risk=" + "/".join(sorted(risk_tags)))
        cond_parts.append(f"event={event_type}")
        cond = ", ".join(cond_parts) if cond_parts else "similar scenes"
        beh = dom_behavior.lower() if dom_behavior else "the observed strategy"
        return (
            f"{count} high-value mid-term memories show that {beh} under ({cond}) "
            f"is a recurring stable pattern; consolidate as a long-term candidate pending human review."
        )

    # ------------------------------------------------------------------
    # 保存（不覆盖正式长期记忆库）
    # ------------------------------------------------------------------
    def save(self, candidates: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """保存候选规则到 YAML（status=pending_review；不写正式长期记忆库）。返回写入路径。"""
        import yaml
        path = Path(output_path or self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# 长期记忆候选规则（由中期记忆沉淀生成）\n"
            "# status=pending_review：需人工审核后晋升为正式长期记忆，切勿直接并入 long_term_rules.yaml\n"
            f"# 候选数: {len(candidates)}\n"
        )
        data = {"candidates": candidates}
        with open(str(path), "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        logger.info("consolidation: 候选规则已保存 %s (%d 条)", path, len(candidates))
        return str(path)
