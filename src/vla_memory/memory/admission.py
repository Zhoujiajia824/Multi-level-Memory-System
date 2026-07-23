"""中期记忆价值门控写入模块（Phase 2）
====================================
``MemoryAdmissionController`` 在每帧**决策完成后、写入中期记忆之前**判断该帧是否值得入库。
低价值帧（普通巡航 / 稳定停车 / 冗余帧）拒绝写入，高价值事件强制写入。``enabled=false`` 时
完全退化为阶段 1 的逐帧全存（回归基线）。

关键约束
--------
* **先读后写**：``decide()`` 只读当前帧 ``ctx`` + 上一帧 ``prev_ctx`` + 已检索的
  ``max_mid_term_score``（含 [0,i-1] 帧），不写任何状态、不读未来帧。
* **posthoc 信号不读未来**：``posthoc_outcome_value`` 只用当前帧决策质量代理
  （fallback / parser 失败 / risk=high），绝不使用未来结果。真后验留给运行后离线分析。
* **配置驱动**：``MemoryAdmissionPolicy`` 从 ``memory.yaml -> mid_term.admission`` 加载，
  权重 / 阈值 / 过滤 / 事件开关全部可配，禁硬编码。
* **纯逻辑**：``decide()`` 无 IO、无副作用，可离线单测（不依赖 faiss / VLM API）。

输出 ``MemoryAdmissionResult``：``should_store`` / ``admission_score`` / ``admission_reasons`` /
``reject_reasons`` / ``event_type`` / ``scene_tags`` / ``risk_tags`` / ``policy_version`` / ``signals``。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("admission")


# ===================== 静态策略常量（与 schemas 枚举对齐，此处为策略子集，非校验）=====================
# 高价值 scene_id（来自 VALID_SCENE_IDS：intersection/dead_end/lane_change/car_following/
# obstacle_avoidance/straight_road/turning/merge/crosswalk/unknown）
_HIGH_VALUE_SCENE_IDS = {
    "lane_change", "obstacle_avoidance", "intersection", "turning", "merge", "crosswalk",
}
_LANE_CHANGE_BEHAVIORS = {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}
_TURN_BEHAVIORS = {"TURN_LEFT", "TURN_RIGHT"}
_AVOID_BEHAVIORS = {"AVOID_OBSTACLE"}
_CRUISE_BEHAVIORS = {"KEEP_LANE", "FOLLOW"}
_FRONT_SIDE_POSITIONS = {"front_left", "front_right"}
_PARSER_FAIL_STATUS = {"parse_error", "validation_error", "no_output", "failed"}
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "unknown": 1}

# 高价值事件判定优先级（安全关键优先），用于 event_type 取最高优先命中
_EVENT_PRIORITY = [
    "cut_in", "ghost_probing_risk", "pedestrian_interaction", "cyclist_interaction",
    "hard_brake", "hard_acceleration", "start", "obstacle_avoidance", "lane_change",
    "intersection", "dense_traffic", "merge", "turn_left", "turn_right", "crosswalk",
    "decision_change", "occlusion", "long_tail",
]

# occlusion / ghost_probing 关键词（best-effort，召回有限，文档明示）
_OCCLUSION_KEYWORDS = ("遮挡", "盲区", "occlus")
_GHOST_KEYWORDS = ("鬼探头", "突发", "盲区", "突然")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """将值裁剪到 [lo, hi] 区间。"""
    return max(lo, min(hi, x))


def _angle_diff(a: float, b: float) -> float:
    """两个航向角（弧度）的最小差，处理 2π 环绕。"""
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


# ===================== 数据结构 =====================

@dataclass
class MemoryAdmissionResult:
    """单帧价值门控判定结果。

    Attributes:
        should_store: 是否写入中期记忆。
        admission_score: 综合价值分（0~1，6 信号加权求和）。
        admission_reasons: 准入原因列表（写入时填充）。
        reject_reasons: 拒绝原因列表（拒绝时填充，如 normal_cruise / score_below_threshold）。
        event_type: 主事件类型（高价值事件名 / 低价值过滤名 / frame_memory）。
        scene_tags: 场景标签列表（lane_change / intersection / ...）。
        risk_tags: 风险标签列表（cut_in / pedestrian_interaction / hard_brake / ...）。
        policy_version: 准入策略版本（写入记忆 admission_policy_version）。
        signals: 6 信号明细（供离线复盘调参，不强制写 jsonl）。
    """
    should_store: bool
    admission_score: float = 0.0
    admission_reasons: List[str] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)
    event_type: str = "frame_memory"
    scene_tags: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)
    policy_version: str = ""
    signals: Dict[str, float] = field(default_factory=dict)


class MemoryAdmissionPolicy:
    """价值门控策略（配置驱动）。

    由 ``memory.yaml -> mid_term.admission`` 字典构造。所有阈值 / 权重 / 开关均可配。
    缺失项使用与设计文档一致的默认值，保证部分配置也能运行。
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.policy_version: str = cfg.get("policy_version", "value_gated_v0.1")
        self.store_all_when_disabled: bool = bool(cfg.get("store_all_when_disabled", True))
        self.debug_memory_off: bool = bool(cfg.get("debug_memory_off", False))
        self.score_threshold: float = float(cfg.get("score_threshold", 0.55))
        self.force_store_threshold: float = float(cfg.get("force_store_threshold", 0.80))

        self.weights: Dict[str, float] = dict(cfg.get("weights", {}) or {})
        # 默认权重（和=1.0）
        _default_weights = {
            "dynamics_surprise": 0.20, "scene_salience": 0.25, "perception_change": 0.20,
            "decision_change": 0.15, "memory_novelty": 0.15, "posthoc_outcome_value": 0.05,
        }
        for k, v in _default_weights.items():
            self.weights.setdefault(k, v)

        self.low_value_filters: Dict[str, bool] = dict(cfg.get("low_value_filters", {}) or {})
        for k in ("filter_stable_stop", "filter_normal_cruise", "filter_redundant_frames"):
            self.low_value_filters.setdefault(k, True)

        self.high_value_events: Dict[str, bool] = dict(cfg.get("high_value_events", {}) or {})
        for ev in _EVENT_PRIORITY:
            self.high_value_events.setdefault(ev, True)

        _thr = cfg.get("thresholds", {}) or {}
        self.thresholds: Dict[str, float] = {
            "speed_change": float(_thr.get("speed_change", 2.0)),
            "accel_change": float(_thr.get("accel_change", 2.0)),
            "jerk": float(_thr.get("jerk", 6.0)),
            "yaw_rate": float(_thr.get("yaw_rate", 0.3)),
            "yaw_change": float(_thr.get("yaw_change", 0.2)),
            "target_speed_change": float(_thr.get("target_speed_change", 3.0)),
            "stop_speed": float(_thr.get("stop_speed", 0.5)),
            "move_speed": float(_thr.get("move_speed", 1.0)),
            "hard_brake_accel": float(_thr.get("hard_brake_accel", -3.0)),
            "hard_accel_accel": float(_thr.get("hard_accel_accel", 3.0)),
            "cut_in_distance": float(_thr.get("cut_in_distance", 15.0)),
            "pedestrian_distance": float(_thr.get("pedestrian_distance", 10.0)),
            "cyclist_distance": float(_thr.get("cyclist_distance", 10.0)),
            "object_count_change": float(_thr.get("object_count_change", 3)),
            "nearest_distance_shrink": float(_thr.get("nearest_distance_shrink", 5.0)),
            "redundant_sim": float(_thr.get("redundant_sim", 0.85)),
            "long_tail_novelty": float(_thr.get("long_tail_novelty", 0.9)),
        }


# ===================== 控制器 =====================

class MemoryAdmissionController:
    """价值门控控制器。

    用法::

        controller = MemoryAdmissionController(policy)
        result = controller.decide(ctx, prev_ctx)
        if result.should_store:
            mid_term.add_record(...)   # 用 result.event_type/scene_tags/... 填 metadata
    """

    def __init__(self, policy: MemoryAdmissionPolicy):
        self.policy = policy

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def decide(
        self,
        ctx: Dict[str, Any],
        prev_ctx: Optional[Dict[str, Any]] = None,
    ) -> MemoryAdmissionResult:
        """判定当前帧是否写入中期记忆。

        Args:
            ctx: 当前帧上下文（见模块 docstring，由 online_loop 组装）。
            prev_ctx: 上一帧上下文（None 表示首帧，无历史）。

        Returns:
            ``MemoryAdmissionResult``。
        """
        pv = self.policy.policy_version

        # ---- 关闭：退化为阶段 1 行为 ----
        if not self.policy.enabled:
            if self.policy.store_all_when_disabled:
                return MemoryAdmissionResult(
                    should_store=True, admission_score=1.0,
                    admission_reasons=["admission_disabled_store_all"],
                    event_type="frame_memory", policy_version=pv,
                )
            return MemoryAdmissionResult(
                should_store=False, admission_score=0.0,
                reject_reasons=["admission_disabled_no_store"], policy_version=pv,
            )

        # ---- 计算 6 信号 ----
        signals = self._compute_signals(ctx, prev_ctx)
        score = _clamp(sum(
            self.policy.weights.get(k, 0.0) * v for k, v in signals.items()
        ))

        # ---- 事件识别（高价值，命中即 force store）----
        events, scene_tags, risk_tags = self._detect_events(ctx, prev_ctx, signals)
        hard = bool(events)

        # ---- 决策 ----
        reasons: List[str] = []
        reject: List[str] = []
        event_type = "frame_memory"

        if hard:
            event_type = events[0]  # 已按 _EVENT_PRIORITY 排序，取最高优先
            reasons.append(f"high_value_event:{event_type}")

        if hard or score >= self.policy.force_store_threshold:
            should_store = True
            if not hard:
                reasons.append(f"force_store_score:{score:.3f}")
        else:
            filt = self._detect_low_value_filter(ctx, prev_ctx, signals)
            if filt:
                should_store = False
                reject.append(filt)
                event_type = filt
            elif score >= self.policy.score_threshold:
                should_store = True
                reasons.append(f"score_above_threshold:{score:.3f}")
            else:
                should_store = False
                reject.append(f"score_below_threshold:{score:.3f}")

        result = MemoryAdmissionResult(
            should_store=should_store,
            admission_score=round(score, 4),
            admission_reasons=reasons,
            reject_reasons=reject,
            event_type=event_type,
            scene_tags=scene_tags,
            risk_tags=risk_tags,
            policy_version=pv,
            signals={k: round(v, 4) for k, v in signals.items()},
        )
        logger.debug(
            "admission: store=%s score=%.3f event=%s reasons=%s reject=%s signals=%s",
            result.should_store, score, result.event_type, reasons, reject, result.signals,
        )
        return result

    # ------------------------------------------------------------------
    # 信号计算
    # ------------------------------------------------------------------
    def _compute_signals(
        self, ctx: Dict[str, Any], prev_ctx: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算 6 个归一化信号（各 0~1）。"""
        thr = self.policy.thresholds
        prev = prev_ctx or {}

        ego = ctx.get("ego_state") or {}
        prev_ego = prev.get("ego_state") or {}
        speed = float(ego.get("speed", 0.0) or 0.0)
        accel = float(ego.get("acceleration", 0.0) or 0.0)  # 标量幅值
        ax = ego.get("ax")  # 前向有符号加速度（用于 hard_brake/hard_accel 事件）
        yaw = float(ego.get("yaw", 0.0) or 0.0)
        yaw_rate = ego.get("yaw_rate")  # 仅 CAN bus
        prev_speed = float(prev_ego.get("speed", speed) or 0.0)
        prev_accel = float(prev_ego.get("acceleration", accel) or 0.0)
        prev_yaw = float(prev_ego.get("yaw", yaw) or 0.0)

        # 帧间 dt（秒）：优先用时间戳差，缺失则默认 0.5s（约 2Hz 关键帧）
        ts = ctx.get("timestamp")
        prev_ts = prev.get("timestamp")
        if ts and prev_ts and ts > prev_ts:
            dt = (ts - prev_ts) / 1e6
        else:
            dt = 0.5

        parsed = ctx.get("parsed") or {}
        target_speed = parsed.get("target_speed")
        prev_target_speed = prev.get("target_speed", target_speed)

        # ---- 1. dynamics_surprise ----
        speed_change = abs(speed - prev_speed) / thr["speed_change"]
        accel_change = abs(accel - prev_accel) / thr["accel_change"]
        jerk = (abs(accel - prev_accel) / dt / thr["jerk"]) if dt > 0 else 0.0
        if yaw_rate is not None:
            yr_mag = abs(float(yaw_rate)) / thr["yaw_rate"]
        else:
            yr_mag = (_angle_diff(yaw, prev_yaw) / dt / thr["yaw_rate"]) if dt > 0 else 0.0
        yaw_change = _angle_diff(yaw, prev_yaw) / thr["yaw_change"]
        if target_speed is not None and prev_target_speed is not None:
            tsc = abs(float(target_speed) - float(prev_target_speed)) / thr["target_speed_change"]
        else:
            tsc = 0.0
        dynamics_surprise = _clamp(max(
            speed_change, accel_change, jerk, yr_mag, yaw_change, tsc
        ))

        # ---- 2. scene_salience ----
        scene = ctx.get("scene_result") or {}
        scene_id = scene.get("scene_id", "unknown")
        traffic_density = scene.get("traffic_density", "unknown")
        risk_factors = scene.get("risk_factors") or []
        pedestrians = scene.get("pedestrians") or []
        vehicles = scene.get("vehicles") or []
        intersections = scene.get("intersections") or {}
        sal = 0.0
        if scene_id in _HIGH_VALUE_SCENE_IDS:
            sal = max(sal, 0.7)
        if traffic_density == "high":
            sal = max(sal, 0.5)
        if intersections.get("present"):
            sal = max(sal, 0.5)
        if pedestrians:
            sal = max(sal, 0.4)
        sal = min(1.0, sal + min(0.3, len(risk_factors) * 0.1))
        scene_salience = _clamp(sal)

        # ---- 3. perception_change ----
        objs = ctx.get("perception_objects") or []
        cur_count = len(objs)
        cur_near = self._nearest_distance(objs)
        # prev 感知：优先用标量，缺失则从 prev 对象列表推导（online_loop 存列表）
        prev_objs = prev.get("perception_objects") or []
        prev_count = prev.get("object_count")
        prev_count = int(prev_count) if prev_count is not None else len(prev_objs)
        prev_near = prev.get("nearest_distance")
        if prev_near is None and prev_objs:
            prev_near = self._nearest_distance(prev_objs)
        count_change = abs(cur_count - prev_count) / thr["object_count_change"]
        if cur_near is not None and prev_near is not None:
            shrink = max(0.0, float(prev_near) - float(cur_near)) / thr["nearest_distance_shrink"]
        else:
            shrink = 0.0
        cut_in = 1.0 if self._has_cut_in(vehicles, thr["cut_in_distance"]) else 0.0
        ped_risk = 1.0 if self._has_pedestrian_risk(
            pedestrians, objs, thr["pedestrian_distance"]
        ) else 0.0
        cyc_risk = 1.0 if self._has_cyclist_risk(objs, thr["cyclist_distance"]) else 0.0
        perception_change = _clamp(max(count_change, shrink, cut_in, ped_risk, cyc_risk))

        # ---- 4. decision_change ----
        behavior = parsed.get("behavior", "")
        prev_behavior = prev.get("behavior", behavior)
        risk_level = parsed.get("risk_level", "medium")
        prev_risk = prev.get("risk_level", risk_level)
        behavior_changed = 1.0 if behavior and prev_behavior and behavior != prev_behavior else 0.0
        risk_raised = 1.0 if _RISK_ORDER.get(risk_level, 1) > _RISK_ORDER.get(prev_risk, 1) else 0.0
        traj_change = self._trajectory_change(parsed.get("trajectory"), prev.get("trajectory"))
        fallback = 1.0 if ctx.get("fallback_used") else 0.0
        parser_fail = 1.0 if ctx.get("parser_status") in _PARSER_FAIL_STATUS else 0.0
        decision_change = _clamp(max(
            behavior_changed, risk_raised, tsc, traj_change, fallback, parser_fail
        ))

        # ---- 5. memory_novelty（复用检索 max 相似度；空库→1.0）----
        max_sim = ctx.get("max_mid_term_score")
        max_sim = 0.0 if max_sim is None else float(max_sim)
        memory_novelty = _clamp(1.0 - max_sim)

        # ---- 6. posthoc_outcome_value（只用当前帧决策质量代理，绝不读未来）----
        posthoc = 0.0
        if fallback:
            posthoc = max(posthoc, 0.6)
        if parser_fail:
            posthoc = max(posthoc, 0.8)
        if risk_level == "high":
            posthoc = max(posthoc, 0.5)
        posthoc_outcome_value = _clamp(posthoc)

        return {
            "dynamics_surprise": dynamics_surprise,
            "scene_salience": scene_salience,
            "perception_change": perception_change,
            "decision_change": decision_change,
            "memory_novelty": memory_novelty,
            "posthoc_outcome_value": posthoc_outcome_value,
        }

    # ------------------------------------------------------------------
    # 高价值事件识别
    # ------------------------------------------------------------------
    def _detect_events(
        self, ctx: Dict[str, Any], prev_ctx: Optional[Dict[str, Any]],
        signals: Dict[str, float],
    ) -> tuple:
        """识别高价值事件。返回 (按优先级排序的事件名列表, scene_tags, risk_tags)。"""
        thr = self.policy.thresholds
        prev = prev_ctx or {}
        enabled = self.policy.high_value_events

        scene = ctx.get("scene_result") or {}
        scene_id = scene.get("scene_id", "unknown")
        traffic_density = scene.get("traffic_density", "unknown")
        vehicles = scene.get("vehicles") or []
        pedestrians = scene.get("pedestrians") or []
        intersections = scene.get("intersections") or {}
        risk_factors = scene.get("risk_factors") or []

        parsed = ctx.get("parsed") or {}
        behavior = parsed.get("behavior", "")
        risk_level = parsed.get("risk_level", "medium")

        ego = ctx.get("ego_state") or {}
        prev_ego = prev.get("ego_state") or {}
        speed = float(ego.get("speed", 0.0) or 0.0)
        prev_speed = float(prev_ego.get("speed", speed) or 0.0)
        ax = ego.get("ax")
        objs = ctx.get("perception_objects") or []

        scene_tags: List[str] = []
        risk_tags: List[str] = []
        hit: Dict[str, bool] = {}

        def mark(ev: str, is_scene: bool = False, is_risk: bool = False):
            """记录命中事件并归类标签。"""
            hit[ev] = True
            if is_scene:
                scene_tags.append(ev)
            if is_risk:
                risk_tags.append(ev)

        # --- 场景语义类 ---
        if enabled.get("lane_change") and (scene_id == "lane_change" or behavior in _LANE_CHANGE_BEHAVIORS):
            mark("lane_change", is_scene=True)
        if enabled.get("obstacle_avoidance") and (scene_id == "obstacle_avoidance" or behavior in _AVOID_BEHAVIORS):
            mark("obstacle_avoidance", is_scene=True)
        if enabled.get("intersection") and (scene_id == "intersection" or intersections.get("present")):
            mark("intersection", is_scene=True)
        if enabled.get("merge") and scene_id == "merge":
            mark("merge", is_scene=True)
        if enabled.get("crosswalk") and scene_id == "crosswalk":
            mark("crosswalk", is_scene=True)
        if enabled.get("turn_left") and behavior == "TURN_LEFT":
            mark("turn_left", is_scene=True)
        if enabled.get("turn_right") and behavior == "TURN_RIGHT":
            mark("turn_right", is_scene=True)
        if enabled.get("dense_traffic") and traffic_density == "high":
            mark("dense_traffic", is_scene=True)

        # --- 自车动态突变类 ---
        if enabled.get("start") and prev_speed < thr["stop_speed"] and speed >= thr["move_speed"]:
            mark("start", is_risk=True)
        if enabled.get("hard_brake") and ax is not None and float(ax) < thr["hard_brake_accel"]:
            mark("hard_brake", is_risk=True)
        if enabled.get("hard_acceleration") and ax is not None and float(ax) > thr["hard_accel_accel"]:
            mark("hard_acceleration", is_risk=True)

        # --- 感知交互类 ---
        if enabled.get("cut_in") and self._has_cut_in(vehicles, thr["cut_in_distance"]):
            mark("cut_in", is_risk=True)
        if enabled.get("pedestrian_interaction") and self._has_pedestrian_risk(
            pedestrians, objs, thr["pedestrian_distance"]
        ):
            mark("pedestrian_interaction", is_risk=True)
        if enabled.get("cyclist_interaction") and self._has_cyclist_risk(objs, thr["cyclist_distance"]):
            mark("cyclist_interaction", is_risk=True)

        # --- 行为或决策变化类（category 2：behavior 变化/risk 升高/target_speed 大变/
        #     轨迹形态大变/fallback/parser 失败。decision_change 信号≥1.0 即任一子项超阈值）---
        if enabled.get("decision_change") and signals.get("decision_change", 0.0) >= 1.0:
            mark("decision_change")

        # --- 关键词 best-effort 类（召回有限，文档明示）---
        if enabled.get("occlusion") and self._risk_factors_match(risk_factors, _OCCLUSION_KEYWORDS):
            mark("occlusion", is_risk=True)
        if enabled.get("ghost_probing_risk") and self._risk_factors_match(risk_factors, _GHOST_KEYWORDS):
            mark("ghost_probing_risk", is_risk=True)

        # --- 长尾：mid_term 非空（有基线可比）+ 高新颖 + 非普通巡航 ---
        # 空库时 max_sim=0→novelty=1.0 是"无历史"而非"稀有"，不应判长尾。
        if enabled.get("long_tail") and not ctx.get("mid_term_empty", True) \
                and signals.get("memory_novelty", 0.0) >= thr["long_tail_novelty"] \
                and not self._is_normal_cruise(ctx, prev, signals):
            mark("long_tail", is_risk=True)

        # 按优先级排序命中事件
        ordered = [ev for ev in _EVENT_PRIORITY if hit.get(ev)]
        return ordered, scene_tags, risk_tags

    # ------------------------------------------------------------------
    # 低价值过滤
    # ------------------------------------------------------------------
    def _detect_low_value_filter(
        self, ctx: Dict[str, Any], prev_ctx: Optional[Dict[str, Any]],
        signals: Dict[str, float],
    ) -> Optional[str]:
        """识别低价值过滤（仅在无高价值事件时调用）。返回过滤名或 None。"""
        thr = self.policy.thresholds
        prev = prev_ctx or {}
        filters = self.policy.low_value_filters

        ego = ctx.get("ego_state") or {}
        prev_ego = prev.get("ego_state") or {}
        speed = float(ego.get("speed", 0.0) or 0.0)
        prev_speed = float(prev_ego.get("speed", speed) or 0.0)
        parsed = ctx.get("parsed") or {}
        behavior = parsed.get("behavior", "")
        risk_level = parsed.get("risk_level", "medium")
        scene = ctx.get("scene_result") or {}
        traffic_density = scene.get("traffic_density", "unknown")
        max_sim = ctx.get("max_mid_term_score")
        max_sim = 0.0 if max_sim is None else float(max_sim)

        # stable_stop：连续两帧静止 + 低风险 + STOP（需有历史帧，首帧不算"稳定"停车）
        if prev_ctx is not None and filters.get("filter_stable_stop") and behavior == "STOP" \
                and speed < thr["stop_speed"] and prev_speed < thr["stop_speed"] \
                and risk_level == "low":
            return "stable_stop"

        # normal_cruise：巡航 + 低风险 + 低密度 + 低动态 + 低感知变化
        if filters.get("filter_normal_cruise") and self._is_normal_cruise(ctx, prev, signals):
            return "normal_cruise"

        # redundant_frame：与已有记忆高度相似 + 无决策变化
        if filters.get("filter_redundant_frames") and max_sim > thr["redundant_sim"] \
                and signals.get("decision_change", 0.0) < 0.2:
            return "redundant_frame"

        return None

    def _is_normal_cruise(
        self, ctx: Dict[str, Any], prev: Dict[str, Any], signals: Dict[str, float]
    ) -> bool:
        """判定是否普通巡航（用于 normal_cruise 过滤与 long_tail 排除）。"""
        parsed = ctx.get("parsed") or {}
        behavior = parsed.get("behavior", "")
        risk_level = parsed.get("risk_level", "medium")
        scene = ctx.get("scene_result") or {}
        traffic_density = scene.get("traffic_density", "unknown")
        return (
            behavior in _CRUISE_BEHAVIORS
            and risk_level == "low"
            and traffic_density in ("low", "unknown")
            and signals.get("dynamics_surprise", 0.0) < 0.3
            and signals.get("perception_change", 0.0) < 0.3
        )

    # ------------------------------------------------------------------
    # 感知对象辅助（防御式兼容 PerceptionObject 实例与 dict）
    # ------------------------------------------------------------------
    @staticmethod
    def _obj_category(o: Any) -> str:
        if isinstance(o, dict):
            return str(o.get("category") or o.get("category_name") or "unknown")
        return str(getattr(o, "category", None) or getattr(o, "category_name_raw", "") or "unknown")

    @staticmethod
    def _obj_distance(o: Any) -> Optional[float]:
        """对象到 ego 距离（米）。oracle 用 distance_to_ego，VLM 用 distance_m。"""
        if isinstance(o, dict):
            d = o.get("distance_to_ego", o.get("distance_m"))
        else:
            d = getattr(o, "distance_to_ego", None)
            if d is None:
                d = getattr(o, "distance_m", None)
        try:
            return float(d) if d is not None else None
        except (TypeError, ValueError):
            return None

    def _nearest_distance(self, objs: List[Any]) -> Optional[float]:
        """oracle 对象中最小到 ego 距离（无对象返回 None）。"""
        dists = [d for d in (self._obj_distance(o) for o in objs) if d is not None]
        return min(dists) if dists else None

    def _has_cut_in(self, vehicles: List[Any], cut_in_thr: float) -> bool:
        """cut-in：VLM vehicles 中有 approaching + 前侧方位 + 近距。"""
        for v in vehicles:
            if not isinstance(v, dict):
                continue
            if v.get("motion") != "approaching":
                continue
            if v.get("relative_position") not in _FRONT_SIDE_POSITIONS:
                continue
            d = v.get("distance_m")
            if d is not None and float(d) < cut_in_thr:
                return True
        return False

    def _has_pedestrian_risk(
        self, pedestrians: List[Any], objs: List[Any], ped_thr: float
    ) -> bool:
        """行人风险：VLM pedestrians crossing+近距，或 oracle pedestrian 近距。"""
        for p in pedestrians:
            if not isinstance(p, dict):
                continue
            if p.get("intent") != "crossing":
                continue
            d = p.get("distance_m")
            if d is not None and float(d) < ped_thr:
                return True
        for o in objs:
            if self._obj_category(o) == "pedestrian":
                d = self._obj_distance(o)
                if d is not None and d < ped_thr:
                    return True
        return False

    def _has_cyclist_risk(self, objs: List[Any], cyc_thr: float) -> bool:
        """cyclist 风险：oracle category=cyclist + 近距。"""
        for o in objs:
            if self._obj_category(o) == "cyclist":
                d = self._obj_distance(o)
                if d is not None and d < cyc_thr:
                    return True
        return False

    @staticmethod
    def _risk_factors_match(risk_factors: List[Any], keywords: tuple) -> bool:
        """risk_factors 关键词匹配（best-effort）。"""
        for rf in risk_factors:
            rf = str(rf).lower()
            if any(kw.lower() in rf for kw in keywords):
                return True
        return False

    def _trajectory_change(
        self, cur: Optional[List[Any]], prev: Optional[List[Any]]
    ) -> float:
        """轨迹形态变化（归一化 0~1）：用末 waypoint 位移近似。无历史→0。"""
        if not cur or not prev:
            return 0.0
        try:
            c = cur[-1] if isinstance(cur[-1], dict) else {}
            p = prev[-1] if isinstance(prev[-1], dict) else {}
            dx = float(c.get("x", 0.0)) - float(p.get("x", 0.0))
            dy = float(c.get("y", 0.0)) - float(p.get("y", 0.0))
            return _clamp(math.sqrt(dx * dx + dy * dy) / 5.0)
        except (TypeError, ValueError):
            return 0.0
