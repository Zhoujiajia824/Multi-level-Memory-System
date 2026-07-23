"""冲突感知更新模块（Phase 6）
================================
``MemoryUpdateManager``：检索到相似记忆但当前决策/安全评价冲突时，对旧记忆软更新（降权 / 标记
deprecated/superseded / 增 conflict_count / 版本化），新记忆作为新版本或替代。**不物理删除**；
unsafe 新证据不覆盖安全旧记忆。

冲突类型（按优先级）：context_mismatch / unsafe_new_evidence / unsafe_old_memory / policy_conflict /
style_conflict。

约束
----
* 软更新：只改字段（conflict_count/status/confidence/...），不动 FAISS、不物理删。
* 在写入后触发，只读已检索旧记忆 + 当前帧决策，不读未来，先读后写不变。
* 配置驱动（``mid_term.update``）。
* ``MidTermMemoryRecord`` 无 ``risk_level`` 字段（仅存 behavior）；old 的"风险"用 ``risk_tags`` 代理，
  new 的 unsafe 用 current_ctx 的 risk_level/fallback/parser_status 判定（文档明示为 v1 近似）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("update")

# behavior 战略类别（用于区分 policy_conflict 跨类 vs style_conflict 同类）
_CRUISE = {"KEEP_LANE", "FOLLOW", "SLOW_DOWN", "STOP"}
_LATERAL = {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}
_TURN = {"TURN_LEFT", "TURN_RIGHT"}
_AVOID = {"AVOID_OBSTACLE", "YIELD"}
_PARSER_FAIL = {"parse_error", "validation_error", "no_output", "failed"}


def _behavior_category(b: str) -> str:
    if b in _CRUISE:
        return "cruise"
    if b in _LATERAL:
        return "lateral"
    if b in _TURN:
        return "turn"
    if b in _AVOID:
        return "avoid"
    return "unknown"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class MemoryUpdateManager:
    """冲突感知更新器（配置驱动）。

    Args:
        cfg: ``memory.yaml -> mid_term.update`` 字典。
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.conflict_detection_enabled: bool = bool(cfg.get("conflict_detection_enabled", True))
        self.versioning_enabled: bool = bool(cfg.get("versioning_enabled", True))
        self.soft_update_only: bool = bool(cfg.get("soft_update_only", True))
        self.min_similarity: float = float(cfg.get("min_similarity_for_conflict_check", 0.75))
        self.confidence_decay: float = float(cfg.get("confidence_decay_on_conflict", 0.10))
        self.supersede_after: int = int(cfg.get("supersede_after_conflicts", 3))

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def process(
        self,
        current_ctx: Dict[str, Any],
        mid_term_results: List[Dict[str, Any]],
        new_record: Any = None,
        now_ts: int = 0,
    ) -> List[Dict[str, Any]]:
        """对检索到的相似旧记忆做冲突检测与软更新。

        Args:
            current_ctx: 当前帧决策上下文（behavior / risk_level / scene_id / nav_instruction /
                fallback_used / parser_status）。
            mid_term_results: 检索结果列表（每项含 record / final_score）。
            new_record: 本帧新入库的记忆对象（用于版本链 superseded_by / previous_versions；可为 None）。
            now_ts: 当前帧时间戳（μs）。

        Returns:
            更新动作列表 ``[{memory_id, conflict_type, action, reason}, ...]``（供日志/jsonl）。
        """
        if not self.enabled or not self.conflict_detection_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        for mr in mid_term_results:
            if not isinstance(mr, dict):
                continue
            if float(mr.get("final_score", 0.0) or 0.0) < self.min_similarity:
                continue
            old = mr.get("record")
            if old is None:
                continue
            ctype = self._classify(current_ctx, old)
            if ctype == "none":
                continue
            action = self._apply(old, ctype, new_record, now_ts, current_ctx)
            actions.append(action)
        if actions:
            logger.info(
                "update: %d 条记忆冲突更新: %s",
                len(actions), [(a["conflict_type"], a["action"]) for a in actions],
            )
        return actions

    # ------------------------------------------------------------------
    # 冲突分类
    # ------------------------------------------------------------------
    @staticmethod
    def _is_new_unsafe(ctx: Dict[str, Any]) -> bool:
        """当前决策不安全：risk=high 或 fallback/parser 失败。"""
        if ctx.get("risk_level") == "high":
            return True
        if ctx.get("fallback_used"):
            return True
        if ctx.get("parser_status") in _PARSER_FAIL:
            return True
        return False

    def _classify(self, current_ctx: Dict[str, Any], old: Any) -> str:
        cur_scene = current_ctx.get("scene_id", "unknown") or "unknown"
        cur_nav = current_ctx.get("nav_instruction", "") or ""
        cur_behavior = current_ctx.get("behavior", "") or ""
        old_scene = (getattr(old, "scene_id", "unknown") or "unknown")
        old_nav = (getattr(old, "nav_instruction", "") or "")
        old_behavior = (getattr(old, "behavior", "") or "")
        old_risk_tags = getattr(old, "risk_tags", None) or []

        # 1. context_mismatch：导航或 scene 不同 → 不同情境，不视为冲突
        context_diff = (cur_nav != old_nav) or (cur_scene != old_scene) or cur_scene in ("", "unknown")
        if context_diff:
            return "context_mismatch"

        new_unsafe = self._is_new_unsafe(current_ctx)
        old_unsafe = bool(old_risk_tags)  # v1 近似：旧记忆涉及风险场景
        beh_diff = bool(cur_behavior) and bool(old_behavior) and cur_behavior != old_behavior

        # 2. unsafe_new_evidence：新证据不安全 → 不覆盖旧（优先级高）
        if new_unsafe:
            return "unsafe_new_evidence"
        # 3. unsafe_old_memory：旧涉及风险、新安全 → 旧可被替代（v1 仅 mild 更新，不自动 deprecate）
        if old_unsafe and not new_unsafe:
            return "unsafe_old_memory"
        # 4/5. behavior 不同 → 跨类别 policy / 同类别 style
        if beh_diff:
            cur_cat = _behavior_category(cur_behavior)
            old_cat = _behavior_category(old_behavior)
            if cur_cat == "unknown" or old_cat == "unknown" or cur_cat != old_cat:
                return "policy_conflict"
            return "style_conflict"
        # 同情境同行为同安全 → 一致，无冲突
        return "none"

    # ------------------------------------------------------------------
    # 软更新
    # ------------------------------------------------------------------
    def _apply(
        self, old: Any, ctype: str, new_record: Any, now_ts: int, current_ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        cur_behavior = current_ctx.get("behavior", "") or ""
        old_behavior = getattr(old, "behavior", "") or ""
        old_mid = getattr(old, "memory_id", "") or getattr(old, "record_id", "")
        reason = f"{ctype}: cur={cur_behavior} vs old={old_behavior}"
        action = "none"

        def _append_history(rec: Any, act: str, why: str) -> None:
            try:
                rec.update_history.append({
                    "action": act, "conflict_type": ctype, "reason": why,
                    "at": now_ts, "by_new": getattr(new_record, "memory_id", None) if new_record else None,
                })
            except Exception as e:
                logger.debug("update_history 追加失败: %s", e)

        def _decay_conf(rec: Any) -> None:
            cur = getattr(rec, "confidence_score", None)
            base = 1.0 if cur is None else float(cur)
            rec.confidence_score = _clamp01(base - self.confidence_decay)

        if ctype == "context_mismatch":
            action = "skip"
            # 不更新

        elif ctype == "unsafe_new_evidence":
            # 旧保持 active；新（若存在）标 low_confidence
            action = "keep_old_lowconf_new"
            _append_history(old, "keep", "unsafe_new_evidence: new unsafe, old kept")
            if new_record is not None:
                new_record.status = "low_confidence"
                _decay_conf(new_record)
                _append_history(new_record, "low_confidence", "unsafe_new_evidence: new marked low_confidence")

        elif ctype == "unsafe_old_memory":
            # 旧 mild 更新（conflict_count++ / 衰减），不自动 deprecate（信号弱）
            action = "mild_update_old"
            old.conflict_count = int(getattr(old, "conflict_count", 0) or 0) + 1
            old.last_conflict_at = now_ts
            old.conflict_reasons = list(getattr(old, "conflict_reasons", []) or []) + [ctype]
            _decay_conf(old)
            _append_history(old, "mild_update", reason)

        elif ctype == "policy_conflict":
            old.conflict_count = int(getattr(old, "conflict_count", 0) or 0) + 1
            old.last_conflict_at = now_ts
            old.conflict_reasons = list(getattr(old, "conflict_reasons", []) or []) + [ctype]
            _decay_conf(old)
            if old.conflict_count >= self.supersede_after:
                self._supersede(old, new_record, now_ts, reason)
                action = "supersede"
                _append_history(old, "supersede", reason)
            else:
                action = "conflict_count_inc"
                _append_history(old, "conflict_count_inc", reason)

        elif ctype == "style_conflict":
            # 风格变体：两条都保留，不衰减不删除
            action = "style_variant_keep"
            old.conflict_reasons = list(getattr(old, "conflict_reasons", []) or []) + ["style_variant"]
            _append_history(old, "style_variant", reason)

        return {"memory_id": old_mid, "conflict_type": ctype, "action": action, "reason": reason}

    def _supersede(self, old: Any, new_record: Any, now_ts: int, reason: str) -> None:
        """版本化替换：old→superseded，new→active，建立版本链。"""
        old.status = "superseded"
        if self.versioning_enabled and new_record is not None:
            new_mid = getattr(new_record, "memory_id", "") or getattr(new_record, "record_id", "")
            old.superseded_by = new_mid
            try:
                old_mid = getattr(old, "memory_id", "") or getattr(old, "record_id", "")
                new_record.previous_versions = list(getattr(new_record, "previous_versions", []) or []) + [old_mid]
            except Exception as e:
                logger.debug("previous_versions 追加失败: %s", e)
