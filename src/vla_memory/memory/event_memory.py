"""事件级记忆模块（Phase 5）
========================
``EventMemoryManager`` 把连续高价值帧合并为一个 ``event_memory``：缓冲高价值帧 → 事件结束时
取 start/peak/end 关键帧 + 生成结构化摘要 → 产出一条 event_memory 记录（FAISS 用 peak 帧特征）。

事件结束条件：高价值信号消失（patience 达阈值）/ 达最大长度 / scene 切换 / run 结束 flush。
短事件（< ``min_event_length_frames``）丢弃（噪声过滤）。``enabled=false`` 退化为逐帧 frame_memory。

约束
----
* 有状态（``_buffer`` / ``_patience``），但只在 online_loop 写入路径消费 admission 结果，
  不前移到检索前，先读后写不变。
* 摘要用确定性模板生成（不调 VLM，cheap；可后续升级）。
* peak 帧特征从 ``feature_path`` 重新 ``np.load``（每事件一次 IO）。
* 不依赖 mid_term_memory（避免循环），返回 ``(record, feature)`` 由 online_loop 调 ``add_record``。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.schemas.memory import MidTermMemoryRecord

logger = get_logger("event_memory")


class EventMemoryManager:
    """事件级记忆管理器（有状态缓冲 + 关键帧 + 摘要）。

    Args:
        cfg: ``memory.yaml -> mid_term.event_memory`` 字典。
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.prefer_event_memory: bool = bool(cfg.get("prefer_event_memory", True))
        self.event_memory_bonus: float = float(cfg.get("event_memory_bonus", 0.10))
        self.max_event_length: int = int(cfg.get("max_event_length_frames", 10))
        self.min_event_length: int = int(cfg.get("min_event_length_frames", 2))
        self.event_end_patience: int = int(cfg.get("event_end_patience_frames", 2))
        self.keyframes_per_event: int = int(cfg.get("keyframes_per_event", 3))
        self.store_frame_memory_when_disabled: bool = bool(
            cfg.get("store_frame_memory_when_event_disabled", True)
        )

        self._buffer: List[Dict[str, Any]] = []
        self._patience: int = 0

    # ------------------------------------------------------------------
    # 主入口：每帧调用
    # ------------------------------------------------------------------
    def on_frame(
        self,
        admission_result: Any,
        frame_ctx: Dict[str, Any],
    ) -> Optional[Tuple[MidTermMemoryRecord, Any]]:
        """消费一帧 admission 结果，必要时 finalize 事件。

        Args:
            admission_result: ``MemoryAdmissionResult``（含 should_store / admission_score /
                event_type / scene_tags / risk_tags）。
            frame_ctx: 当前帧上下文（sample_token / scene_token / scene_name / image_path /
                feature_path / visual_input_type / source_dataset / source_version / ego_state /
                perception_objects / scene_result / parsed / scene_text / scene_id / weather_id /
                nav_instruction / timestamp / version / source_mode / feature_dim 等）。

        Returns:
            ``None``（仍缓冲中）或 ``(event_memory_record, peak_feature)``（事件 finalize）。
        """
        if not self.enabled:
            return None
        is_high_value = bool(getattr(admission_result, "should_store", False))
        if is_high_value:
            self._buffer.append({
                **frame_ctx,
                "admission_score": float(getattr(admission_result, "admission_score", 0.0) or 0.0),
                "event_type": getattr(admission_result, "event_type", "frame_memory") or "frame_memory",
                "scene_tags": list(getattr(admission_result, "scene_tags", []) or []),
                "risk_tags": list(getattr(admission_result, "risk_tags", []) or []),
            })
            self._patience = 0
            if len(self._buffer) >= self.max_event_length:
                return self._finalize()
            return None
        # 非高价值帧：若缓冲中有事件，patience 计数
        if self._buffer:
            self._patience += 1
            if self._patience >= self.event_end_patience:
                return self._finalize()
        return None

    def flush(self) -> Optional[Tuple[MidTermMemoryRecord, Any]]:
        """强制结束当前事件（scene 切换 / run 结束）。"""
        if not self._buffer:
            return None
        return self._finalize()

    # ------------------------------------------------------------------
    # 事件 finalize
    # ------------------------------------------------------------------
    def _finalize(self) -> Optional[Tuple[MidTermMemoryRecord, Any]]:
        buffer = self._buffer
        n = len(buffer)
        try:
            if n < self.min_event_length:
                logger.debug("event 丢弃：长度 %d < min %d", n, self.min_event_length)
                self._reset()
                return None

            # peak = admission_score 最高的帧（并列取最早）
            peak_idx = max(range(n), key=lambda i: buffer[i]["admission_score"])
            peak = buffer[peak_idx]
            event_type = peak["event_type"] or "frame_memory"

            key_tokens, key_paths = self._select_keyframes(buffer, peak_idx, self.keyframes_per_event)
            ego_sum, perc_sum, dec_sum, adm_sum = self._generate_summaries(buffer, peak)
            event_id = self._make_event_id(peak)

            parsed = peak.get("parsed") or {}
            record = MidTermMemoryRecord(
                record_id=event_id,
                image_feature_path=peak.get("feature_path") or "",
                scene_text=peak.get("scene_text", "") or "",
                scene_id=peak.get("scene_id", "unknown"),
                weather_id=peak.get("weather_id", "unknown"),
                nav_instruction=peak.get("nav_instruction"),
                ego_state=peak.get("ego_state"),
                history_trajectory=peak.get("history_trajectory"),
                decision_reason=parsed.get("behavior_reason", ""),
                behavior=parsed.get("behavior", ""),
                trajectory=parsed.get("trajectory"),
                # ---- 阶段 1 metadata（从 peak 帧填充）----
                memory_id=event_id,
                memory_type="event_memory",
                status="active",
                version=peak.get("version", "v0.2"),
                created_at=int(buffer[0].get("timestamp", 0) or 0),   # 事件起始
                updated_at=int(peak.get("timestamp", 0) or 0),        # peak 时间
                source_dataset=peak.get("source_dataset", ""),
                source_version=peak.get("source_version", ""),
                source_scene_token=peak.get("scene_token", "") or "",
                source_scene_name=peak.get("scene_name", "") or "",
                source_sample_token=peak.get("sample_token", ""),
                source_frame_id=peak.get("sample_token", ""),
                source_mode=peak.get("source_mode", ""),
                visual_input_type=peak.get("visual_input_type", ""),
                image_path=peak.get("image_path", "") or "",
                feature_path=peak.get("feature_path", "") or "",
                feature_dim=int(peak.get("feature_dim", 0) or 0),
                # ---- 场景标签 / 写入价值（来自 admission）----
                event_type=event_type,
                scene_tags=peak["scene_tags"],
                risk_tags=peak["risk_tags"],
                admission_score=peak["admission_score"],
                admission_reasons=list(peak.get("admission_reasons", []) or []),
                admission_policy_version=peak.get("admission_policy_version", ""),
                is_active=True,
                # ---- Phase 5 事件专属 ----
                event_id=event_id,
                event_start_sample_token=buffer[0].get("sample_token", ""),
                event_peak_sample_token=peak.get("sample_token", ""),
                event_end_sample_token=buffer[-1].get("sample_token", ""),
                anchor_sample_token=peak.get("sample_token", ""),
                key_sample_tokens=key_tokens,
                anchor_image_path=peak.get("image_path", "") or "",
                key_image_paths=key_paths,
                ego_summary=ego_sum,
                perception_summary=perc_sum,
                decision_summary=dec_sum,
                admission_summary=adm_sum,
                usage={"frame_count": n, "keyframe_count": len(key_tokens)},
            )

            # 加载 peak 帧特征（event_memory 的 FAISS 向量）
            feature = None
            fp = peak.get("feature_path")
            if fp:
                try:
                    import numpy as _np
                    feature = _np.load(fp)
                except Exception as e:
                    logger.warning("event finalize: peak 特征加载失败 (%s): %s", fp, e)
            self._reset()
            logger.info(
                "event_memory 生成: id=%s type=%s frames=%d peak=%s",
                event_id, event_type, n, peak.get("sample_token", "")[:12],
            )
            return (record, feature)
        except Exception as e:
            logger.warning("event finalize 异常: %s", e)
            self._reset()
            return None

    # ------------------------------------------------------------------
    # 关键帧 / 摘要 / id
    # ------------------------------------------------------------------
    def _select_keyframes(
        self, buffer: List[Dict[str, Any]], peak_idx: int, k: int
    ) -> Tuple[List[str], List[str]]:
        """选 start(0) / peak / end(n-1)，去重保序，截断到 k。返回 (tokens, image_paths)。"""
        idxs: List[int] = []
        for i in (0, peak_idx, len(buffer) - 1):
            if i not in idxs:
                idxs.append(i)
        idxs = idxs[: max(1, k)]
        tokens = [buffer[i].get("sample_token", "") for i in idxs]
        paths = [buffer[i].get("image_path", "") or "" for i in idxs]
        return tokens, paths

    def _generate_summaries(
        self, buffer: List[Dict[str, Any]], peak: Dict[str, Any]
    ) -> Tuple[str, str, str, str]:
        """确定性模板生成 4 摘要（不调 VLM）。"""
        # ego_summary：速度趋势
        speeds = []
        for b in buffer:
            ego = b.get("ego_state") or {}
            s = ego.get("speed")
            if s is not None:
                try:
                    speeds.append(float(s))
                except (TypeError, ValueError):
                    pass
        if speeds:
            ego_sum = f"speed {speeds[0]:.1f}->{speeds[-1]:.1f} m/s over {len(buffer)} frames"
        else:
            ego_sum = f"{len(buffer)} frames"

        # perception_summary：对象/车辆/行人峰值计数
        n_obj = 0
        for b in buffer:
            objs = b.get("perception_objects") or []
            n_obj = max(n_obj, len(objs))
        scene0 = buffer[0].get("scene_result") or {}
        n_veh = len(scene0.get("vehicles", []) or [])
        n_ped = len(scene0.get("pedestrians", []) or [])
        perc_sum = f"max {n_obj} objects, {n_veh} vehicles, {n_ped} pedestrians"

        # decision_summary：behavior 序列
        behaviors = []
        for b in buffer:
            parsed = b.get("parsed") or {}
            beh = parsed.get("behavior")
            if beh and beh not in behaviors:  # 去重连续相同
                behaviors.append(beh)
        dec_sum = " -> ".join(behaviors) if behaviors else ""

        # admission_summary
        adm_sum = (
            f"{peak['event_type']} event, peak_admission={peak['admission_score']:.2f}, "
            f"{len(buffer)} frames"
        )
        return ego_sum, perc_sum, dec_sum, adm_sum

    def _make_event_id(self, peak: Dict[str, Any]) -> str:
        """生成唯一 event_id（= record_id）：event_<sample_token>_<timestamp>。"""
        st = peak.get("sample_token", "") or "x"
        ts = int(peak.get("timestamp", 0) or 0)
        return f"event_{st[:16]}_{ts}"

    def _reset(self) -> None:
        self._buffer = []
        self._patience = 0
