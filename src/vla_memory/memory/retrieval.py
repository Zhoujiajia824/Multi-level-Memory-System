"""记忆检索入口
==============
整合短期、中期、长期记忆的检索结果。
提供统一的记忆检索接口供决策模块使用。
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
from src.vla_memory.memory.short_term_memory import ShortTermMemory
from src.vla_memory.memory.mid_term_memory import MidTermMemory
from src.vla_memory.memory.long_term_memory import LongTermMemory
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("retrieval")


class MemoryRetriever:
    """记忆检索管理器。

    整合三层记忆的检索结果，供决策模块使用。

    Args:
        short_term: 短期记忆实例。
        mid_term: 中期记忆实例。
        long_term: 长期记忆实例。
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        mid_term: MidTermMemory,
        long_term: LongTermMemory,
    ):
        self.short_term = short_term
        self.mid_term = mid_term
        self.long_term = long_term

    def retrieve(
        self,
        query_feature: Optional[np.ndarray] = None,
        scene_text: str = "",
        scene_id: str = "",
        weather_id: str = "",
        nav_instruction: str = "",
        ego_state: Optional[Dict] = None,
        use_short_term: bool = True,
        use_mid_term: bool = True,
        use_long_term: bool = True,
        now_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行三层记忆检索。

        Args:
            query_feature: 当前帧图像特征。
            scene_text: 场景描述文本。
            scene_id: 场景类型。
            weather_id: 天气类型。
            nav_instruction: 导航语义。
            ego_state: 自车状态。
            use_short_term: 是否使用短期记忆。
            use_mid_term: 是否使用中期记忆。
            use_long_term: 是否使用长期记忆。
            now_ts: 当前帧时间戳（μs）；透传给 mid_term.search 更新命中统计（Phase 3）。
                None 时不更新（向后兼容）。

        Returns:
            包含三层记忆检索结果的字典。
        """
        result = {
            "short_term_summary": "",
            "mid_term_results": [],
            "mid_term_stats": {},
            "long_term_rules": [],
            "long_term_strategies": [],
        }

        # 短期记忆摘要
        if use_short_term:
            result["short_term_summary"] = self.short_term.generate_summary()

        # 中期记忆检索（Phase 4：search 返回 {results, stats}）
        if use_mid_term:
            mt_out = self.mid_term.search(
                query_feature=query_feature,
                scene_text=scene_text,
                scene_id=scene_id,
                weather_id=weather_id,
                nav_instruction=nav_instruction,
                ego_state=ego_state,
                now_ts=now_ts,
            )
            if isinstance(mt_out, dict):
                result["mid_term_results"] = mt_out.get("results", [])
                result["mid_term_stats"] = mt_out.get("stats", {})
            else:  # 向后兼容：旧版 search 返回 list
                result["mid_term_results"] = mt_out

        # 长期规则检索
        if use_long_term:
            result["long_term_rules"] = self.long_term.search_rules(
                scene_id=scene_id,
                weather_id=weather_id,
            )
            result["long_term_strategies"] = self.long_term.search_strategies(
                scene_id=scene_id,
            )

        logger.info(
            f"记忆检索完成: 短期={'是' if use_short_term else '否'}, "
            f"中期={len(result['mid_term_results'])}条, "
            f"长期规则={len(result['long_term_rules'])}条, "
            f"策略={len(result['long_term_strategies'])}条"
        )

        return result
