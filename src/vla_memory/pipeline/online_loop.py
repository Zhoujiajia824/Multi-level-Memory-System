"""在线驾驶决策循环
====================
逐帧处理关键帧序列：感知 → 检索 → 决策 → 更新记忆。

这是 demo 的正确"模拟车端"运行方式（取代之前的批处理瀑布）。
关键正确性保证：
* 第 i 帧检索三层记忆时，中期记忆只包含 [0, i-1] 帧的记录
  （因为 add_record 在 step() 末尾才发生），彻底消除 data leakage。
* 短期记忆是真正的滑动窗口（push 在 step 末尾，第 i 帧拿到的是 [0, i-1]）。
* 单次运行只跑一种 mode（memory_on 或 memory_off），评测作为独立步骤。

底层模块复用现成的：DINOv2Extractor / SceneUnderstandingPipeline /
{Short,Mid,Long}TermMemory / MemoryRetriever / DecisionClient /
parse_decision_output / generate_fallback_decision。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.vla_memory.common.config import Config
from src.vla_memory.common.decision_record_io import (
    append_decision_record,
    load_processed_sample_tokens,
)
from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.decision.decision_client import DecisionClient
from src.vla_memory.decision.output_parser import parse_decision_output
from src.vla_memory.decision.rule_fallback import generate_fallback_decision
from src.vla_memory.memory.faiss_store import FAISSVectorStore
from src.vla_memory.memory.long_term_memory import LongTermMemory
from src.vla_memory.memory.mid_term_memory import MidTermMemory
from src.vla_memory.memory.retrieval import MemoryRetriever
from src.vla_memory.memory.short_term_memory import ShortTermMemory
from src.vla_memory.perception.dinov2_extractor import DINOv2Extractor
from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient
from src.vla_memory.perception.scene_understanding import SceneUnderstandingPipeline
from src.vla_memory.schemas.memory import MidTermMemoryRecord, ShortTermMemoryItem

logger = get_logger("online_loop")


# 决策 jsonl 命名：outputs/decisions_<mode>_<run_id>.jsonl
def default_output_path(config: Config, mode: str) -> Path:
    """根据 config.output_dir / config.run_id 拼出默认的 decisions jsonl 路径。"""
    output_dir = config.get("output_dir")
    if output_dir is None:
        output_dir = Path("outputs")
    output_dir = Path(output_dir)
    run_id = config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"decisions_{mode}_{run_id}.jsonl"


class OnlineDrivingLoop:
    """逐帧在线驾驶循环引擎。

    生命周期：

        loop = OnlineDrivingLoop(config, mode="memory_on",
                                 output_jsonl_path=Path(...), resume=True)
        loop.setup()                  # 一次性初始化所有模块
        records = loop.run(keyframes) # 主循环
        loop.close()                  # 中期记忆按 yaml persistence 决定是否落盘

    Args:
        config: 项目 Config。
        mode: ``"memory_on"`` 或 ``"memory_off"``。memory_off 模式：
            - 不向 VLM 传记忆段；
            - retriever 三路 use_* 全 False；
            - 短期/中期记忆不更新（不污染对比基准）。
        output_jsonl_path: 决策结果 jsonl 输出路径。None 时由
            default_output_path 自动生成。
        resume: True 时启动扫已存在 jsonl，跳过已处理 sample_token；
            False 时无脑覆盖。
    """

    VALID_MODES = ("memory_on", "memory_off")

    def __init__(
        self,
        config: Config,
        mode: str,
        output_jsonl_path: Optional[Path | str] = None,
        resume: bool = True,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode 必须是 {self.VALID_MODES}，实际: {mode!r}")
        self.config = config
        self.mode = mode
        self.use_memory = (mode == "memory_on")
        self.output_jsonl_path = (
            Path(output_jsonl_path) if output_jsonl_path
            else default_output_path(config, mode)
        )
        self.resume = resume

        # ---- setup 后填充的组件 ----
        self.feature_extractor: Optional[DINOv2Extractor] = None
        self.scene_pipeline: Optional[SceneUnderstandingPipeline] = None
        self.short_term: Optional[ShortTermMemory] = None
        self.mid_term: Optional[MidTermMemory] = None
        self.long_term: Optional[LongTermMemory] = None
        self.retriever: Optional[MemoryRetriever] = None
        self.decision_client: Optional[DecisionClient] = None

        # P5 决策图像上下文配置
        self._image_context_size = 3
        self._include_current_frame = True
        self._max_images_per_call = 4

        # resume 集合
        self._resume_set: Set[str] = set()

        self._setup_done = False

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """一次性初始化所有底层模块。可重复调用（幂等）。"""
        if self._setup_done:
            return

        cfg = self.config

        # ---- 1. DINOv2 ----
        fe_cfg = cfg.get("feature_extractor", {}) or {}
        self.feature_extractor = DINOv2Extractor(
            model_name=fe_cfg.get("model_name", "facebook/dinov2-base"),
            cache_dir=str(fe_cfg.get("cache_dir", ".cache/huggingface")),
            device=fe_cfg.get("device", "auto"),
            normalize=fe_cfg.get("normalize", True),
        )
        self.feature_extractor.load_model()

        # ---- 2. 场景理解 VLM + Pipeline ----
        scene_vlm_cfg = cfg.get("scene_understanding", {}) or {}
        scene_vlm = OpenAICompatibleVLMClient(
            provider=scene_vlm_cfg.get("provider", "qwen"),
            api_key_env=scene_vlm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            base_url=scene_vlm_cfg.get("base_url", ""),
            model_name=scene_vlm_cfg.get("model_name", "qwen-vl-max"),
            timeout=scene_vlm_cfg.get("timeout", 60),
            max_tokens=scene_vlm_cfg.get("max_tokens", 2048),
            temperature=scene_vlm_cfg.get("temperature", 0.1),
            retry_times=scene_vlm_cfg.get("retry_times", 3),
            retry_interval_seconds=scene_vlm_cfg.get("retry_interval_seconds", 5),
            system_prompt=scene_vlm_cfg.get("system_prompt", ""),
        )
        feature_dir = cfg.get("feature_dir") or Path("outputs/features")
        self.scene_pipeline = SceneUnderstandingPipeline(
            feature_extractor=self.feature_extractor,
            vlm_client=scene_vlm,
            feature_save_dir=str(feature_dir),
            vlm_retry_times=scene_vlm_cfg.get("retry_times", 2),
        )

        # ---- 3. 三层记忆 ----
        st_cap = cfg.get_nested("short_term", "capacity", default=10)
        self.short_term = ShortTermMemory(capacity=st_cap)

        mt_top_k = cfg.get_nested("mid_term", "top_k", default=3)
        mt_weights = cfg.get_nested("mid_term", "weights", default={})
        feat_dim = cfg.get_nested("feature_extractor", "feature_dim", default=768)
        faiss_type = cfg.get_nested("mid_term", "faiss_index_type", default="IndexFlatIP")
        mt_persist = cfg.get_nested("mid_term", "persistence", default={}) or {}
        save_dir = self._resolve_memory_save_dir()
        self.mid_term = MidTermMemory(
            faiss_store=FAISSVectorStore(dimension=feat_dim, index_type=faiss_type),
            weights=mt_weights,
            top_k=mt_top_k,
            persistence_cfg=mt_persist,
            save_dir=str(save_dir),
        )

        lt_rules = cfg.get_nested(
            "long_term", "rules_file",
            default="data/knowledge/long_term_rules.yaml",
        )
        lt_strats = cfg.get_nested(
            "long_term", "strategies_file",
            default="data/knowledge/driving_strategies.yaml",
        )
        lt_strict_scene = cfg.get_nested("long_term", "strict_scene_match", default=True)
        lt_strict_weather = cfg.get_nested("long_term", "strict_weather_match", default=False)
        self.long_term = LongTermMemory(
            rules_file=str(lt_rules),
            strategies_file=str(lt_strats),
            strict_scene_match=lt_strict_scene,
            strict_weather_match=lt_strict_weather,
        )
        self.long_term.load()

        self.retriever = MemoryRetriever(
            short_term=self.short_term,
            mid_term=self.mid_term,
            long_term=self.long_term,
        )

        # ---- 4. 决策 VLM + Client ----
        dec_vlm_cfg = cfg.get("decision", {}) or {}
        decision_vlm = OpenAICompatibleVLMClient(
            provider=dec_vlm_cfg.get("provider", "qwen"),
            api_key_env=dec_vlm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            base_url=dec_vlm_cfg.get("base_url", ""),
            model_name=dec_vlm_cfg.get("model_name", "qwen-vl-max"),
            timeout=dec_vlm_cfg.get("timeout", 120),
            max_tokens=dec_vlm_cfg.get("max_tokens", 4096),
            temperature=dec_vlm_cfg.get("temperature", 0.1),
            retry_times=dec_vlm_cfg.get("retry_times", 3),
            retry_interval_seconds=dec_vlm_cfg.get("retry_interval_seconds", 5),
            system_prompt=dec_vlm_cfg.get("system_prompt", ""),
        )
        self.decision_client = DecisionClient(vlm_client=decision_vlm)

        # ---- 5. P5 图像上下文配置 ----
        self._image_context_size = int(
            cfg.get_nested("vlm_inputs", "image_context_size", default=3)
        )
        self._include_current_frame = bool(
            cfg.get_nested("vlm_inputs", "include_current_frame", default=True)
        )
        self._max_images_per_call = int(
            cfg.get_nested("vlm_inputs", "max_images_per_call", default=4)
        )

        # ---- 6. resume 扫描 ----
        if self.resume:
            self._resume_set = load_processed_sample_tokens(self.output_jsonl_path)
        else:
            # 不 resume：覆盖文件
            if self.output_jsonl_path.exists():
                self.output_jsonl_path.unlink()
            self._resume_set = set()

        self._setup_done = True
        logger.info(
            "OnlineDrivingLoop setup OK: mode=%s, output=%s, resume_skip=%d, "
            "image_context_size=%d, st_capacity=%d, mid_persist=%s",
            self.mode, self.output_jsonl_path, len(self._resume_set),
            self._image_context_size, st_cap, mt_persist.get("enabled", False),
        )

    def _resolve_memory_save_dir(self) -> Path:
        """中期记忆持久化目录解析（与 P2 memory_pipeline 同样的策略）。"""
        candidate = self.config.get("memory_db_dir")
        if candidate:
            return Path(candidate)
        persist_cfg = self.config.data.get("persistence", {}) or {}
        save_dir = persist_cfg.get("save_dir")
        if save_dir:
            return Path(save_dir)
        return Path("outputs/memory_db")

    # ------------------------------------------------------------------
    # 单帧处理
    # ------------------------------------------------------------------

    def step(self, kf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理单个关键帧。

        Args:
            kf: 关键帧 dict（来自 enrich_keyframes_with_state），含
                sample_token / image_path / ego_state / history_trajectory /
                nav_instruction 等字段。

        Returns:
            决策记录 dict；若 sample_token 已在 resume 集中返回 None。
        """
        if not self._setup_done:
            raise RuntimeError("OnlineDrivingLoop 尚未 setup，先调用 .setup()")

        sample_token = str(kf.get("sample_token") or "")
        if sample_token and sample_token in self._resume_set:
            logger.info("resume 跳过已处理帧: %s", sample_token)
            return None

        image_path = kf.get("image_path", "")

        # ---- a) 感知：DINOv2 + 场景理解 ----
        # process_frame 内部会保存 .npy 特征
        perception = self.scene_pipeline.process_frame(
            sample_token=sample_token, image_path=image_path,
        )
        if perception is None:
            return self._handle_perception_failure(kf, sample_token)

        feat_path = perception.get("image_feature_path")
        scene_result = perception.get("scene_understanding") or {}
        scene_id = scene_result.get("scene_id", "unknown")
        weather_id = scene_result.get("weather_id", "unknown")
        scene_description = scene_result.get("scene_description", "")

        # 把场景理解结果回填到 kf，方便记忆 / record 用
        kf["scene_understanding_result"] = scene_result
        kf["scene_id"] = scene_id
        kf["weather_id"] = weather_id
        kf["scene_description"] = scene_description
        kf["image_feature_path"] = feat_path

        # ---- b) 检索三层记忆 ----
        feature = None
        if feat_path:
            try:
                feature = np.load(feat_path)
            except Exception as e:
                logger.warning("特征加载失败 (frame=%s): %s", sample_token, e)

        memory_result = self.retriever.retrieve(
            query_feature=feature,
            scene_text=scene_description,
            scene_id=scene_id,
            weather_id=weather_id,
            nav_instruction=kf.get("nav_instruction", ""),
            ego_state=kf.get("ego_state"),
            use_short_term=self.use_memory,
            use_mid_term=self.use_memory,
            use_long_term=self.use_memory,
        )

        mid_results = memory_result.get("mid_term_results", [])
        retrieved_memory_ids = []
        for mr in mid_results:
            rec = mr.get("record")
            if rec is not None:
                rid = (
                    rec.record_id if hasattr(rec, "record_id")
                    else rec.get("record_id", "")
                )
                retrieved_memory_ids.append(rid)

        lt_rules = memory_result.get("long_term_rules", [])
        long_term_rule_ids = [
            (r.rule_id if hasattr(r, "rule_id") else r.get("rule_id", ""))
            for r in lt_rules
        ]
        rules_text = (
            self.long_term.format_rules_text(lt_rules)
            if self.use_memory and lt_rules else ""
        )

        # ---- c) 组装 image_paths（短期窗口 + 当前帧） ----
        image_paths: List[str] = []
        if self.use_memory and self._image_context_size > 0:
            history_n = self._image_context_size - (
                1 if self._include_current_frame else 0
            )
            if history_n > 0:
                image_paths = list(self.short_term.get_recent_image_paths(history_n))
        if self._include_current_frame and image_path:
            image_paths.append(image_path)
        if len(image_paths) > self._max_images_per_call:
            image_paths = image_paths[-self._max_images_per_call:]

        # ---- d) 调决策 VLM ----
        raw_response = None
        try:
            raw_response = self.decision_client.decide(
                image_paths=image_paths,
                scene_understanding=scene_result,
                frame_id=sample_token,
                ego_state=kf.get("ego_state"),
                history_trajectory=kf.get("history_trajectory"),
                nav_instruction=kf.get("nav_instruction", ""),
                short_term_summary=memory_result.get("short_term_summary", ""),
                mid_term_memories=mid_results,
                long_term_rules_text=rules_text,
            )
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error("决策 VLM 调用异常 (frame=%s): %s", sample_token, e)

        # ---- e) 解析 / fallback ----
        parsed, errors, parser_status, fallback_used = self._parse_or_fallback(
            raw_response=raw_response,
            ego_state=kf.get("ego_state"),
            nav_instruction=kf.get("nav_instruction", ""),
        )

        # ---- f) 组装 record ----
        record = {
            "frame_id": sample_token,
            "sample_token": sample_token,
            "scene_token": kf.get("scene_token", ""),
            "memory_mode": self.mode,
            "current_scene": scene_result,
            "scene_id": scene_id,
            "weather_id": weather_id,
            "retrieved_memory_ids": retrieved_memory_ids,
            "long_term_rule_ids": long_term_rule_ids,
            "decision_output": parsed,
            "parser_status": parser_status,
            "parser_errors": errors,
            "fallback_used": fallback_used,
            "raw_response": raw_response,
            "vlm_image_paths": list(image_paths),
            "ego_state": kf.get("ego_state"),
            "nav_instruction": kf.get("nav_instruction", ""),
            "history_trajectory": kf.get("history_trajectory"),
            "ground_truth_trajectory": kf.get("ground_truth_trajectory"),
            "timestamp": int(kf.get("timestamp", 0) or 0),
        }

        # ---- g) 持久化 jsonl（必须在 push 记忆之前/之后都行，但放最前可保证中断不丢） ----
        append_decision_record(self.output_jsonl_path, record)
        if sample_token:
            self._resume_set.add(sample_token)

        # ---- g.1) 单帧完整审计日志（每帧一段，包含图片/状态/记忆/决策的所有摘要） ----
        try:
            self._log_frame_audit(
                kf=kf, record=record, scene_result=scene_result,
                memory_result=memory_result, image_paths=image_paths, parsed=parsed,
            )
        except Exception as e:
            logger.debug("审计日志生成失败 (frame=%s): %s", sample_token, e)

        # ---- h) 更新短期记忆（push 当前帧；memory_off 也不 push，保持纯净对照） ----
        if self.use_memory:
            try:
                self.short_term.add(ShortTermMemoryItem(
                    frame_id=sample_token,
                    timestamp=int(kf.get("timestamp", 0) or 0),
                    image_path=image_path or "",
                    image_feature_path=feat_path,
                    scene_description=scene_description,
                    scene_id=scene_id,
                    weather_id=weather_id,
                    nav_instruction=kf.get("nav_instruction"),
                    ego_state=kf.get("ego_state"),
                    history_trajectory=kf.get("history_trajectory"),
                    scene_understanding_result=scene_result,
                ))
            except Exception as e:
                logger.warning("短期记忆 push 失败 (frame=%s): %s", sample_token, e)

            # ---- i) 更新中期记忆（含决策后字段） ----
            try:
                mt_record = MidTermMemoryRecord(
                    record_id=sample_token or f"frame_{len(self._resume_set)}",
                    image_feature_path=feat_path,
                    scene_text=scene_description,
                    scene_id=scene_id,
                    weather_id=weather_id,
                    nav_instruction=kf.get("nav_instruction"),
                    ego_state=kf.get("ego_state"),
                    history_trajectory=kf.get("history_trajectory"),
                    decision_reason=(parsed or {}).get("behavior_reason", ""),
                    behavior=(parsed or {}).get("behavior", ""),
                    trajectory=(parsed or {}).get("trajectory"),
                )
                self.mid_term.add_record(mt_record, feature=feature)
            except Exception as e:
                logger.warning("中期记忆 add_record 失败 (frame=%s): %s", sample_token, e)

        return record

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """循环处理所有关键帧。单帧异常不会中断整体运行。"""
        if not self._setup_done:
            self.setup()
        records: List[Dict[str, Any]] = []
        for i, kf in enumerate(keyframes):
            sample_token = str(kf.get("sample_token") or f"frame_{i}")
            logger.info(
                "[%d/%d] mode=%s frame=%s",
                i + 1, len(keyframes), self.mode, sample_token[:16],
            )
            try:
                rec = self.step(kf)
                if rec is not None:
                    records.append(rec)
            except EnvironmentError:
                # API key 缺失等致命错，直接抛
                raise
            except Exception as e:
                logger.error(
                    "step 异常 (frame=%s)，本帧记 error 跳过: %s",
                    sample_token, e,
                )
                err_record = {
                    "frame_id": sample_token,
                    "sample_token": sample_token,
                    "memory_mode": self.mode,
                    "parser_status": "step_error",
                    "parser_errors": [str(e)],
                    "fallback_used": False,
                    "decision_output": None,
                }
                append_decision_record(self.output_jsonl_path, err_record)
                if sample_token:
                    self._resume_set.add(sample_token)
        logger.info(
            "OnlineDrivingLoop 完成: mode=%s, 输出帧 %d/%d，jsonl=%s",
            self.mode, len(records), len(keyframes), self.output_jsonl_path,
        )
        return records

    def close(self) -> None:
        """关闭：中期记忆按 yaml persistence 决定是否落盘。"""
        if self.mid_term is not None:
            try:
                self.mid_term.close()
            except Exception as e:
                logger.warning("mid_term.close 异常: %s", e)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _parse_or_fallback(
        self,
        raw_response: Optional[str],
        ego_state: Optional[Dict[str, Any]],
        nav_instruction: str,
    ):
        """parse + 兜底 fallback 的统一逻辑。"""
        allow_fallback = self.config.get_nested(
            "fallback", "allow_rule_fallback_for_format_error", default=True,
        )
        if raw_response is None:
            errors = ["VLM 无输出"]
            parser_status = "no_output"
            parsed = None
        else:
            parsed, errors = parse_decision_output(raw_response)
            if parsed is not None:
                parser_status = "success"
            else:
                parser_status = "parse_error" if errors else "validation_error"

        fallback_used = False
        if parsed is None and allow_fallback:
            logger.warning("决策解析失败，启用 fallback: %s", errors)
            parsed = generate_fallback_decision(
                ego_state=ego_state, nav_instruction=nav_instruction,
            )
            fallback_used = True
            parser_status = "fallback"
        elif parsed is None:
            logger.error("决策解析失败且未启用 fallback: %s", errors)
            parsed = {
                "behavior": "UNKNOWN",
                "behavior_reason": f"VLM 输出解析失败: {errors}",
                "target_speed": 0.0,
                "risk_level": "high",
                "trajectory": [],
                "safety_notes": ["决策失败"],
                "fallback_used": False,
                "parser_status": "failed",
            }
            parser_status = "failed"

        return parsed, errors, parser_status, fallback_used

    def _log_frame_audit(
        self,
        kf: Dict[str, Any],
        record: Dict[str, Any],
        scene_result: Dict[str, Any],
        memory_result: Dict[str, Any],
        image_paths: List[str],
        parsed: Optional[Dict[str, Any]],
    ) -> None:
        """把一个完整帧的处理过程日志输出为结构化 audit 块。"""
        sample_token = record["sample_token"]
        ego = kf.get("ego_state") or {}
        mid_results = memory_result.get("mid_term_results", [])
        st_summary = memory_result.get("short_term_summary", "")

        lines = [f"=========== AUDIT frame={sample_token} mode={self.mode} ==========="]
        lines.append("")

        # ① 图中用哪几张
        lines.append("📷 图片")
        for p in image_paths:
            lines.append(f"  {p}")
        lines.append("")

        # ② 自车状态
        lines.append("🚗 自车状态")
        lines.append(f"  位置: ({ego.get('x'):.2f}, {ego.get('y'):.2f})")
        lines.append(f"  航向角 yaw: {ego.get('yaw', 0):.3f} rad")
        lines.append(f"  速度: {ego.get('speed', 0):.2f} m/s")
        lines.append(f"  加速度: {ego.get('acceleration', 0):.2f} m/s²")
        for key, lbl in (("yaw_rate", "偏航角速率"), ("steering_angle", "方向盘转角"),
                         ("throttle", "油门"), ("brake", "刹车")):
            if ego.get(key) is not None:
                lines.append(f"  {lbl}: {ego[key]:.4f}")
        lines.append(f"  来源: {ego.get('source', 'pose_diff')}")
        lines.append("")

        # ③ 导航 + 历史轨迹
        nav = kf.get("nav_instruction", "") or "无"
        lines.append(f"🧭 导航指令: {nav}")
        hist = kf.get("history_trajectory") or []
        if hist:
            latest = hist[-1]
            lines.append(f"📊 历史轨迹: {len(hist)} 个点 (最近: t={latest.get('t')}, x={latest.get('x'):.1f}, y={latest.get('y'):.1f})")
        else:
            lines.append("📊 历史轨迹: 无")
        lines.append("")

        # ④ 场景理解
        lines.append("🧠 场景理解")
        lines.append(f"  场景描述: {scene_result.get('scene_description', '')[:150]}")
        lines.append(f"  场景 ID: {scene_result.get('scene_id', 'unknown')}")
        lines.append(f"  天气 ID: {scene_result.get('weather_id', 'unknown')}")
        lines.append(f"  交通密度: {scene_result.get('traffic_density', 'unknown')}")
        lanes = scene_result.get("lanes") or []
        if lanes:
            lines.append(f"  车道线: {len(lanes)} 条")
            for ln in lanes[:4]:
                lines.append(f"    side={ln.get('side')} type={ln.get('type')} color={ln.get('color')}")
        vehicles = scene_result.get("vehicles") or []
        if vehicles:
            lines.append(f"  周围车辆: {len(vehicles)} 辆")
            for v in vehicles[:3]:
                lines.append(f"    pos={v.get('relative_position')} dist={v.get('distance_m')} type={v.get('type')} motion={v.get('motion')}")
        pedestrians = scene_result.get("pedestrians") or []
        if pedestrians:
            lines.append(f"  行人: {len(pedestrians)} 人")
        traffic_lights = scene_result.get("traffic_lights") or []
        if traffic_lights:
            lines.append(f"  信号灯: {len(traffic_lights)} 个 (第一个: state={traffic_lights[0].get('state')} pos={traffic_lights[0].get('relative_position')})")
        inter = scene_result.get("intersections") or {}
        if inter and inter.get("present"):
            lines.append(f"  路口: present type={inter.get('type')} dist={inter.get('distance_m')}")
        risks = scene_result.get("risk_factors") or []
        if risks:
            lines.append(f"  风险因素: {'; '.join(risks)}")
        lines.append("")

        # ⑤ 检索到的三层记忆
        lines.append("🗂️ 记忆检索")
        # 短期摘要
        if st_summary:
            lines.append(f"  [短期] {len(self.short_term)} 帧在窗口")
        else:
            lines.append("  [短期] 无（memory_off 或窗口为空）")
        # 中期
        if mid_results:
            lines.append(f"  [中期] 检索到 {len(mid_results)} 条")
            for mr in mid_results:
                rec = mr.get("record") or {}
                if isinstance(rec, dict):
                    sid, wid, dec, beh = rec.get("scene_id","?"), rec.get("weather_id","?"), rec.get("decision_reason","")[:40], rec.get("behavior","?")
                else:
                    sid, wid, dec, beh = getattr(rec,"scene_id","?"), getattr(rec,"weather_id","?"), getattr(rec,"decision_reason","")[:40], getattr(rec,"behavior","?")
                lines.append(f"    score={mr.get('final_score', 0):.3f} scene={sid} weather={wid} behavior={beh} reason={dec}")
        else:
            lines.append("  [中期] 无（memory_off 或 FAISS 空）")
        # 长期
        lt_rules = memory_result.get("long_term_rules") or []
        lt_ids = [r.rule_id if hasattr(r,"rule_id") else r.get("rule_id","") for r in lt_rules]
        if lt_ids:
            lines.append(f"  [长期] 匹配 {len(lt_ids)} 条: {', '.join(lt_ids[:5])}")
        else:
            lines.append("  [长期] 无匹配规则")
        lines.append("")

        # ⑥ 决策 VLM 输入输出
        lines.append("📝 决策模型")
        if parsed:
            lines.append(f"  行为: {parsed.get('behavior', '?')}")
            lines.append(f"  原因: {parsed.get('behavior_reason', '')[:200]}")
            lines.append(f"  目标速度: {parsed.get('target_speed', '?')}")
            lines.append(f"  风险等级: {parsed.get('risk_level', '?')}")
            traj = parsed.get("trajectory") or []
            if traj:
                lines.append(f"  轨迹: {len(traj)} 个 waypoint (首:({traj[0].get('x',0):.2f},{traj[0].get('y',0):.2f}) 末:({traj[-1].get('x',0):.2f},{traj[-1].get('y',0):.2f}))")
            else:
                lines.append("  轨迹: 无")
            lines.append(f"  状态: {record.get('parser_status')} fallback={record.get('fallback_used')}")
        else:
            lines.append("  决策: 无（感知失败或 error 帧）")
        lines.append(f"  原始 raw_response: {str(record.get('raw_response', ''))[:300]}")
        lines.append("")

        lines.append(f"=========== AUDIT END ===========")
        logger.info("\n".join(lines))

    def _handle_perception_failure(
        self, kf: Dict[str, Any], sample_token: str,
    ) -> Dict[str, Any]:
        """感知失败（特征或场景理解失败）：写一条 error 记录，不进入决策。"""
        logger.error("感知失败，记录 error 帧: %s", sample_token)
        record = {
            "frame_id": sample_token,
            "sample_token": sample_token,
            "memory_mode": self.mode,
            "parser_status": "perception_failed",
            "fallback_used": False,
            "decision_output": None,
            "ego_state": kf.get("ego_state"),
        }
        append_decision_record(self.output_jsonl_path, record)
        if sample_token:
            self._resume_set.add(sample_token)
        return record
