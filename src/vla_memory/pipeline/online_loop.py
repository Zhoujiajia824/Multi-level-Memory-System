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

        # ---- Phase 2 价值门控 ----
        # admission_controller 在 setup() 中由 config 构建；prev_frame_ctx 是上一帧上下文快照，
        # 供下一帧 admission 的"决策变化/动力学突变"检测（只存最近一帧，不持久化，不读未来）。
        self._admission_controller = None
        self._admission_enabled = False
        self._admission_debug_memory_off = False
        self._prev_frame_ctx: Optional[Dict[str, Any]] = None

        # ---- Phase 5 事件级记忆 ----
        # event_manager 有状态缓冲连续高价值帧；_prev_scene_token 用于 scene 切换 flush 事件。
        self._event_manager = None
        self._event_memory_enabled = False
        self._prev_scene_token: str = ""

        # ---- Phase 6 冲突感知更新 ----
        # update_manager 在写入后对检索到的相似旧记忆做冲突检测与软更新（不物理删除）。
        self._update_manager = None
        self._update_enabled = False

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
        mt_retrieval = cfg.get_nested("mid_term", "retrieval", default={}) or {}
        # Phase 5：把 event_memory 的检索偏好并入 retrieval_cfg（供 search 加成）
        _ev_cfg_early = cfg.get_nested("mid_term", "event_memory", default={}) or {}
        if _ev_cfg_early.get("prefer_event_memory", True):
            mt_retrieval = {
                **mt_retrieval,
                "prefer_event_memory": True,
                "event_memory_bonus": float(_ev_cfg_early.get("event_memory_bonus", 0.10)),
            }
        save_dir = self._resolve_memory_save_dir()
        self.mid_term = MidTermMemory(
            faiss_store=FAISSVectorStore(dimension=feat_dim, index_type=faiss_type),
            weights=mt_weights,
            top_k=mt_top_k,
            persistence_cfg=mt_persist,
            save_dir=str(save_dir),
            retrieval_cfg=mt_retrieval,
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

        # ---- 5.1 感知输入模式 + 图像布局描述（注入场景理解/决策 prompt） ----
        self._perception_mode = cfg.get_nested("perception", "mode", default="single_front")
        self._image_layout_desc = self._build_image_layout_desc(cfg)

        # ---- 5.2 中期记忆 metadata 默认值（Phase 1：legacy 写入价值标记，配置驱动）----
        # Phase 1 不做价值门控/事件识别/淘汰，只扩展 metadata 结构。以下属性在 step()
        # 构造 MidTermMemoryRecord 时填充新字段：来源、视觉输入、legacy 准入标记等。
        # 价值类字段（memory_value_score 等）Phase 1 不计算 → 保持 None，绝不伪造。
        # 详见 docs/mid_term_memory_value_gating_plan.md §5、schemas/memory.py。
        mt_meta = cfg.get_nested("mid_term", default={}) or {}
        self._mt_metadata_version = mt_meta.get("metadata_schema_version", "v0.2")
        self._mt_default_memory_type = mt_meta.get("default_memory_type", "frame_memory")
        self._mt_default_status = mt_meta.get("default_status", "active")
        self._mt_legacy_admission_score = float(mt_meta.get("legacy_admission_score", 1.0))
        self._mt_legacy_admission_reason = mt_meta.get("legacy_admission_reason", "legacy_no_gating")
        self._mt_legacy_admission_policy = mt_meta.get("legacy_admission_policy_version", "legacy")
        # 来源数据集元数据（写入 source_dataset / source_version）
        self._source_dataset = cfg.get("dataset_name", "nuscenes")
        self._source_version = cfg.get("version", "")

        # ---- 5.3 Phase 2 价值门控（MemoryAdmissionController）----
        # 从 mid_term.admission 构建策略与控制器。enabled=false + store_all_when_disabled=true
        # 时退化为阶段 1 逐帧全存。控制器纯逻辑，无 IO。详见 docs/stage2_admission_design.md。
        from src.vla_memory.memory.admission import (
            MemoryAdmissionController, MemoryAdmissionPolicy,
        )
        admission_cfg = cfg.get_nested("mid_term", "admission", default={}) or {}
        admission_policy = MemoryAdmissionPolicy(admission_cfg)
        self._admission_controller = MemoryAdmissionController(admission_policy)
        self._admission_enabled = admission_policy.enabled
        self._admission_debug_memory_off = admission_policy.debug_memory_off

        # ---- 5.4 Phase 3 容量管理（价值评分 + 淘汰 + 压缩，依赖注入 mid_term）----
        # capacity.enabled=false 时不注入淘汰器（无容量上限，向后兼容）。
        # 详见 docs/stage3_eviction_design.md。
        from src.vla_memory.memory.value_scorer import MemoryValueScorer
        from src.vla_memory.memory.eviction import (
            MemoryCompactionManager, MemoryEvictionManager,
        )
        capacity_cfg = cfg.get_nested("mid_term", "capacity", default={}) or {}
        if capacity_cfg.get("enabled", True):
            eviction_cfg = cfg.get_nested("mid_term", "eviction", default={}) or {}
            compaction_cfg = cfg.get_nested("mid_term", "compaction", default={}) or {}
            value_scorer = MemoryValueScorer(eviction_cfg)
            compaction = MemoryCompactionManager(compaction_cfg)
            eviction = MemoryEvictionManager(
                capacity_cfg=capacity_cfg, eviction_cfg=eviction_cfg,
                value_scorer=value_scorer, compaction=compaction,
            )
            self.mid_term.set_value_scorer(value_scorer)
            self.mid_term.set_eviction_manager(eviction)
            self.mid_term.set_compaction_manager(compaction)

        # ---- 5.5 Phase 5 事件级记忆（EventMemoryManager）----
        # enabled=false + store_frame_memory_when_event_disabled=true → 退化为阶段 2 逐帧 frame_memory。
        # 详见 docs/stage5_event_memory_design.md。
        from src.vla_memory.memory.event_memory import EventMemoryManager
        event_cfg = cfg.get_nested("mid_term", "event_memory", default={}) or {}
        self._event_manager = EventMemoryManager(event_cfg)
        self._event_memory_enabled = bool(event_cfg.get("enabled", True))

        # ---- 5.6 Phase 6 冲突感知更新（MemoryUpdateManager）----
        # enabled=false → 无冲突更新（阶段 5 行为）。详见 docs/stage6_update_design.md。
        from src.vla_memory.memory.update import MemoryUpdateManager
        update_cfg = cfg.get_nested("mid_term", "update", default={}) or {}
        self._update_manager = MemoryUpdateManager(update_cfg)
        self._update_enabled = bool(update_cfg.get("enabled", True))

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

    def _build_image_layout_desc(self, cfg) -> str:
        """根据 perception.mode 生成图像布局描述串（注入 VLM 场景理解/决策 prompt）。"""
        mode = cfg.get_nested("perception", "mode", default="single_front")
        if mode == "surround_mosaic":
            return (
                "nuScenes 六视角摄像头拼接图（2x3 环视布局：上排 CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT，"
                "下排 CAM_BACK_LEFT | CAM_BACK | CAM_BACK_RIGHT；每个子图左上角已标注对应相机名）。"
                "注意这是六个独立相机视角的网格拼接，不是连续全景图；前方三个相机在上排，后方三个相机在下排。"
            )
        return "自动驾驶前视角摄像头（CAM_FRONT）拍摄的图片"

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
            image_layout=self._image_layout_desc,
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
            now_ts=int(kf.get("timestamp", 0) or 0),  # Phase 3：更新检索命中统计
        )

        mid_results = memory_result.get("mid_term_results", [])
        mid_stats = memory_result.get("mid_term_stats", {}) or {}
        # Phase 4：提取检索候选/过滤统计 + 逐条记忆字段（供离线复盘价值感知检索）
        retrieval_candidate_count = int(mid_stats.get("candidate_count", 0))
        retrieval_active_candidate_count = int(mid_stats.get("active_candidate_count", 0))
        retrieval_filtered_count = int(mid_stats.get("filtered_count", 0))
        retrieved_memory_ids = []
        retrieved_memory_scores = []
        retrieved_memory_value_scores = []
        retrieved_memory_event_types = []
        retrieved_memory_statuses = []
        for mr in mid_results:
            rec = mr.get("record")
            rid = (
                rec.record_id if (rec is not None and hasattr(rec, "record_id"))
                else (rec.get("record_id", "") if isinstance(rec, dict) else "")
            )
            retrieved_memory_ids.append(rid)
            retrieved_memory_scores.append(mr.get("final_score"))
            retrieved_memory_value_scores.append(mr.get("memory_value_score"))
            retrieved_memory_event_types.append(mr.get("event_type"))
            retrieved_memory_statuses.append(mr.get("status"))

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
                perception_objects=kf.get("perception_objects"),
                image_layout=self._image_layout_desc,
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

        # ---- e.1) 中期记忆价值门控（Phase 2：MemoryAdmissionController）----
        # 决策完成后、写入前判断是否入库。先读后写：只读当前帧 + 已检索 memory_result（[0,i-1]）
        # + prev_frame_ctx（i-1），不写、不读未来。memory_on 启用门控；memory_off 不入库
        # （仅当 debug_memory_off 时算 debug，不影响决策）。enabled=false 退化为逐帧全存。
        record_id = sample_token or f"frame_{len(self._resume_set)}"
        frame_ts = int(kf.get("timestamp", 0) or 0)

        # 组装 admission 上下文（max_mid_term_score 复用 step 开头的检索结果，不额外查 FAISS）
        _mid_results = memory_result.get("mid_term_results", []) if memory_result else []
        _max_sim = (
            float(_mid_results[0].get("final_score", 0.0))
            if _mid_results and isinstance(_mid_results[0], dict) else 0.0
        )
        admission_ctx = {
            "parsed": parsed,
            "scene_result": scene_result,
            "ego_state": kf.get("ego_state"),
            "perception_objects": kf.get("perception_objects") or [],
            "max_mid_term_score": _max_sim,
            "mid_term_empty": not bool(_mid_results),  # 空库时 long_tail 不触发（无基线判断稀有）
            "timestamp": frame_ts,
            "nav_instruction": kf.get("nav_instruction", ""),
            "fallback_used": fallback_used,
            "parser_status": parser_status,
        }

        # 运行门控
        admission_result = None
        if self.use_memory and self._admission_enabled:
            admission_result = self._admission_controller.decide(
                admission_ctx, self._prev_frame_ctx
            )
            mid_term_admit = admission_result.should_store
        elif self.use_memory:
            # admission 关闭 + store_all_when_disabled → 逐帧全存（阶段 1 行为）
            mid_term_admit = True
        else:
            # memory_off：不入库；可选 debug（admission 在决策后运行，不影响 memory_off 决策）
            if self._admission_debug_memory_off and self._admission_controller is not None:
                admission_result = self._admission_controller.decide(
                    admission_ctx, self._prev_frame_ctx
                )
            mid_term_admit = False

        # 派生 jsonl / mt_record 准入字段（兼容：门控命中 / legacy 全存 / memory_off）
        memory_admission_enabled = self._admission_enabled and self.use_memory
        # Phase 6：冲突更新是否生效（静态标志；具体动作在 i.1 块写入后产生，记于旧记忆 update_history + 日志）
        memory_update_enabled = self._update_enabled and self.use_memory
        if admission_result is not None:
            memory_admission_score = admission_result.admission_score
            memory_admission_should_store = admission_result.should_store
            memory_admission_reasons = admission_result.admission_reasons
            memory_admission_reject_reasons = admission_result.reject_reasons
            memory_event_type = admission_result.event_type
            memory_scene_tags = admission_result.scene_tags
            memory_risk_tags = admission_result.risk_tags
            _mt_admission_policy = admission_result.policy_version
        else:
            # legacy 路径（admission 关闭或 memory_off 无 debug）
            memory_admission_score = self._mt_legacy_admission_score if mid_term_admit else None
            memory_admission_should_store = mid_term_admit
            memory_admission_reasons = [self._mt_legacy_admission_reason] if mid_term_admit else []
            memory_admission_reject_reasons = []
            memory_event_type = "frame_memory"
            memory_scene_tags = []
            memory_risk_tags = []
            _mt_admission_policy = self._mt_legacy_admission_policy

        mid_term_memory_added = mid_term_admit
        mid_term_memory_id = record_id if mid_term_admit else ""
        memory_record_status = self._mt_default_status if mid_term_admit else ""
        # Phase 5：jsonl memory_type 反映本帧 admission（事件模式=缓冲 event_buffered；帧模式=frame_memory）。
        # 事件 finalize 在 i 块（jsonl 之后），event_id 见 mid_term_meta.json。
        memory_type_added = (
            ("event_buffered" if self._event_memory_enabled else "frame_memory")
            if mid_term_admit else ""
        )

        # ---- f) 组装 record ----
        record = {
            "frame_id": sample_token,
            "sample_token": sample_token,
            "scene_token": kf.get("scene_token", ""),
            "memory_mode": self.mode,
            "perception_mode": self._perception_mode,
            "current_scene": scene_result,
            "perception_objects": kf.get("perception_objects", []),
            "scene_id": scene_id,
            "weather_id": weather_id,
            "retrieved_memory_ids": retrieved_memory_ids,
            # ---- Phase 4 价值感知检索统计与逐条字段（供离线复盘）----
            "retrieval_candidate_count": retrieval_candidate_count,            # active 候选总数（过滤前）
            "retrieval_active_candidate_count": retrieval_active_candidate_count,  # 过滤后候选池
            "retrieval_filtered_count": retrieval_filtered_count,               # 被过滤器剔除数
            "retrieved_memory_scores": retrieved_memory_scores,                 # 各结果 final_score（相似度）
            "retrieved_memory_value_scores": retrieved_memory_value_scores,     # 各结果 memory_value_score
            "retrieved_memory_event_types": retrieved_memory_event_types,       # 各结果 event_type
            "retrieved_memory_statuses": retrieved_memory_statuses,             # 各结果 status
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
            "timestamp": frame_ts,
            # ---- Phase 1/2 中期记忆准入元数据（供离线复盘价值门控）----
            "mid_term_memory_added": mid_term_memory_added,        # 本帧是否入库中期记忆
            "mid_term_memory_id": mid_term_memory_id,              # 入库记录的 record_id（= sample_token）
            "memory_admission_enabled": memory_admission_enabled,  # 本帧门控是否生效（memory_on & enabled）
            "memory_update_enabled": memory_update_enabled,        # Phase 6 冲突更新是否生效（动作见 update_history/日志）
            "memory_admission_score": memory_admission_score,      # 价值分（0~1；legacy=1.0；memory_off 无 debug=None）
            "memory_admission_should_store": memory_admission_should_store,  # 门控判定是否应入库
            "memory_admission_reasons": memory_admission_reasons,  # 准入原因
            "memory_admission_reject_reasons": memory_admission_reject_reasons,  # 拒绝原因（normal_cruise 等）
            "memory_event_type": memory_event_type,                # 事件类型（lane_change/hard_brake/...）
            "memory_scene_tags": memory_scene_tags,                # 场景标签列表
            "memory_risk_tags": memory_risk_tags,                  # 风险标签列表
            "memory_record_status": memory_record_status,          # 记忆状态（active；未入库=""）
            # ---- Phase 5 事件级记忆：本帧入库类型（event_buffered/frame_memory/空）；event_id 见 mid_term_meta.json ----
            "memory_type": memory_type_added,
            "event_id": "",  # 事件 id 在事件 finalize 时生成（mid_term_meta.json 可查），单帧 jsonl 留空
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

            # ---- h.1) Phase 5 scene 切换 → flush 当前事件（事件结束条件之一）----
            if self._event_memory_enabled:
                cur_scene_token = kf.get("scene_token", "") or ""
                if self._prev_scene_token and cur_scene_token != self._prev_scene_token:
                    _flush_out = self._event_manager.flush()
                    if _flush_out is not None:
                        _fr, _ffeat = _flush_out
                        try:
                            self.mid_term.add_record(_fr, feature=_ffeat)
                        except Exception as e:
                            logger.warning("event_memory scene-flush add_record 失败: %s", e)
                if cur_scene_token:
                    self._prev_scene_token = cur_scene_token

            # ---- i) 更新中期记忆 ----
            # 先读后写：add_record 在 step 末尾（jsonl 之后），第 i 帧检索只看到 [0,i-1]。
            # Phase 5：event_memory 启用时，高价值帧缓冲到事件，事件结束才入库 event_memory（peak 特征）；
            #   否则退化为阶段 2 逐帧 frame_memory。短期记忆 push(h) 不受影响。
            new_record_added = None  # Phase 6：本帧入库的记忆对象（供冲突更新建版本链）
            if self._event_memory_enabled:
                _feat_dim = (
                    int(np.asarray(feature).reshape(-1).shape[0])
                    if feature is not None else 0
                )
                frame_ctx = {
                    "sample_token": sample_token,
                    "scene_token": kf.get("scene_token", "") or "",
                    "scene_name": kf.get("scene_name", "") or "",
                    "image_path": image_path or "",
                    "feature_path": feat_path or "",
                    "feature_dim": _feat_dim,
                    "visual_input_type": self._perception_mode,
                    "source_dataset": self._source_dataset,
                    "source_version": self._source_version,
                    "source_mode": self.mode,
                    "version": self._mt_metadata_version,
                    "ego_state": kf.get("ego_state"),
                    "history_trajectory": kf.get("history_trajectory"),
                    "perception_objects": kf.get("perception_objects") or [],
                    "scene_result": scene_result,
                    "parsed": parsed,
                    "scene_text": scene_description,
                    "scene_id": scene_id,
                    "weather_id": weather_id,
                    "nav_instruction": kf.get("nav_instruction"),
                    "timestamp": frame_ts,
                    "admission_reasons": memory_admission_reasons,
                    "admission_policy_version": _mt_admission_policy,
                }
                event_out = self._event_manager.on_frame(admission_result, frame_ctx)
                if event_out is not None:
                    ev_record, ev_feature = event_out
                    try:
                        self.mid_term.add_record(ev_record, feature=ev_feature)
                        new_record_added = ev_record
                    except Exception as e:
                        logger.warning("event_memory add_record 失败: %s", e)
            elif mid_term_admit:
                # event_memory 关闭 → 阶段 2 逐帧 frame_memory
                try:
                    # feature_dim 从实际特征向量推导（兼容 (768,) / (1,768)）；无特征则 0
                    feature_dim = (
                        int(np.asarray(feature).reshape(-1).shape[0])
                        if feature is not None else 0
                    )
                    mt_record = MidTermMemoryRecord(
                        record_id=record_id,
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
                        # ===== Phase 1/2 metadata（配置驱动；事件标签来自门控，不伪造价值）=====
                        memory_id=record_id,
                        memory_type=self._mt_default_memory_type,
                        status=self._mt_default_status,
                        version=self._mt_metadata_version,
                        created_at=frame_ts,
                        updated_at=frame_ts,
                        source_dataset=self._source_dataset,
                        source_version=self._source_version,
                        source_scene_token=kf.get("scene_token", "") or "",
                        source_scene_name=kf.get("scene_name", "") or "",
                        source_sample_token=sample_token,
                        source_frame_id=record_id,
                        source_mode=self.mode,
                        visual_input_type=self._perception_mode,
                        image_path=image_path or "",
                        feature_path=feat_path or "",
                        feature_dim=feature_dim,
                        event_type=memory_event_type,
                        scene_tags=memory_scene_tags,
                        risk_tags=memory_risk_tags,
                        admission_score=(
                            memory_admission_score
                            if memory_admission_score is not None
                            else self._mt_legacy_admission_score
                        ),
                        admission_reasons=memory_admission_reasons,
                        admission_policy_version=_mt_admission_policy,
                        # 6/7/8：记忆价值/使用统计/删除 保持默认（不伪造）
                    )
                    self.mid_term.add_record(mt_record, feature=feature)
                    new_record_added = mt_record
                except Exception as e:
                    logger.warning("中期记忆 add_record 失败 (frame=%s): %s", sample_token, e)

            # ---- i.1) Phase 6 冲突感知更新 ----
            # 写入后对检索到的相似旧记忆做冲突检测与软更新（不物理删除；unsafe 新证据不覆盖安全旧）。
            # 只读已检索 mid_results + 当前帧决策，不读未来，先读后写不变。
            memory_update_actions = []
            if self._update_enabled:
                try:
                    update_ctx = {
                        "behavior": (parsed or {}).get("behavior", ""),
                        "risk_level": (parsed or {}).get("risk_level", "medium"),
                        "scene_id": scene_id,
                        "nav_instruction": kf.get("nav_instruction", "") or "",
                        "fallback_used": fallback_used,
                        "parser_status": parser_status,
                    }
                    memory_update_actions = self._update_manager.process(
                        update_ctx, mid_results, new_record_added, now_ts=frame_ts,
                    )
                except Exception as e:
                    logger.warning("冲突更新异常 (frame=%s): %s", sample_token, e)
            # 动作回填到本帧 jsonl（record 已写入，此处仅记内存变量供审计日志；下一帧 jsonl 不回溯）
            if memory_update_actions:
                logger.debug("frame=%s 冲突更新: %s", sample_token, memory_update_actions)

        # ---- j) 更新上一帧上下文（供下一帧 admission 的变化检测；每帧都更新，含拒绝帧）----
        # 只存最近一帧快照，不持久化、不读未来。memory_off 也更新（供 debug_memory_off 用）。
        self._prev_frame_ctx = {
            "behavior": (parsed or {}).get("behavior", ""),
            "risk_level": (parsed or {}).get("risk_level", "medium"),
            "target_speed": (parsed or {}).get("target_speed"),
            "trajectory": (parsed or {}).get("trajectory"),
            "ego_state": kf.get("ego_state") or {},
            "scene_id": scene_id,
            "traffic_density": (scene_result or {}).get("traffic_density", "unknown"),
            "perception_objects": kf.get("perception_objects") or [],
            "timestamp": frame_ts,
        }

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
        """关闭：flush 残留事件 + 中期记忆按 yaml persistence 决定是否落盘。"""
        # Phase 5：flush 尚未结束的事件（run 结束触发 finalize）
        if self._event_memory_enabled and self._event_manager is not None:
            try:
                flush_out = self._event_manager.flush()
                if flush_out is not None and self.mid_term is not None:
                    _fr, _ffeat = flush_out
                    self.mid_term.add_record(_fr, feature=_ffeat)
            except Exception as e:
                logger.warning("event_memory close-flush 异常: %s", e)
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
