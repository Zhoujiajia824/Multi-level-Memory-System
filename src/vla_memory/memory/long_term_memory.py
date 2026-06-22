"""长期记忆模块
==============
存储文本规则和驾驶常识。使用 scene_id 和 weather_id 进行规则匹配。
第一版只存储文本规则，不强制知识图谱，但预留知识图谱目录。
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from src.vla_memory.schemas.memory import LongTermRule
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("long_term_memory")


class LongTermMemory:
    """长期记忆管理器。

    从 YAML 文件加载驾驶规则和策略知识。
    检索方式：使用 scene_id 和 weather_id 规则匹配。

    Args:
        rules_file: 长期规则 YAML 文件路径。
        strategies_file: 驾驶策略 YAML 文件路径。
        top_k: 返回最匹配的 top_k 条规则。
    """

    def __init__(
        self,
        rules_file: str = "data/knowledge/long_term_rules.yaml",
        strategies_file: str = "data/knowledge/driving_strategies.yaml",
        top_k: int = 5,
        strict_scene_match: bool = True,
        strict_weather_match: bool = False,
    ):
        """
        Args:
            rules_file: 长期规则 YAML 文件路径。
            strategies_file: 驾驶策略 YAML 文件路径。
            top_k: 返回最匹配的 top_k 条规则。
            strict_scene_match: 严格 scene_id 匹配（默认 True）。开启后，
                scene_id="all" 的通配规则不再匹配具体场景，只有显式同 scene_id
                的规则会被检索出来。可在 config/memory.yaml->long_term.strict_scene_match
                中覆盖。
            strict_weather_match: 严格 weather_id 匹配（默认 False）。开启逻辑同上。
        """
        self.rules_file = Path(rules_file)
        self.strategies_file = Path(strategies_file)
        self.top_k = top_k
        self.strict_scene_match = strict_scene_match
        self.strict_weather_match = strict_weather_match
        self._rules: List[LongTermRule] = []
        self._strategies: List[Dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        """加载长期记忆知识文件。

        如果文件不存在则记录警告，不抛出异常。
        """
        # 加载规则
        if self.rules_file.exists():
            with open(str(self.rules_file), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            raw_rules = data.get("rules", []) if data else []
            self._rules = [LongTermRule(**r) for r in raw_rules]
            logger.info(f"长期规则加载完成: {len(self._rules)} 条规则")
        else:
            logger.warning(f"长期规则文件不存在: {self.rules_file}")

        # 加载策略
        if self.strategies_file.exists():
            with open(str(self.strategies_file), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._strategies = data.get("strategies", []) if data else []
            logger.info(f"驾驶策略加载完成: {len(self._strategies)} 条策略")
        else:
            logger.warning(f"驾驶策略文件不存在: {self.strategies_file}")

        self._loaded = True

    def search_rules(
        self,
        scene_id: str = "",
        weather_id: str = "",
        top_k: Optional[int] = None,
    ) -> List[LongTermRule]:
        """检索匹配的驾驶规则。

        匹配逻辑（strict 模式由构造参数控制）：
        - strict_scene_match=True（P2 默认）：scene_id 必须精确匹配，
          rule.scene_id="all" 的通配规则会被过滤掉。
        - strict_scene_match=False：rule.scene_id="all" 视为匹配（旧行为）。
        - strict_weather_match 同理，默认 False 保持通配行为。

        Args:
            scene_id: 当前场景类型。
            weather_id: 当前天气类型。
            top_k: 返回数量（None 使用 self.top_k）。

        Returns:
            匹配的规则列表，按 (匹配分降序, 优先级升序) 排序。
        """
        if not self._loaded:
            self.load()

        matched = []
        for rule in self._rules:
            score = 0
            # ---- 场景匹配 ----
            if rule.scene_id == scene_id and scene_id:
                score += 2  # 精确匹配最优
            elif rule.scene_id == "all" and not self.strict_scene_match:
                score += 1  # 通配仅在非严格模式下可匹配
            else:
                continue  # 场景不匹配，跳过

            # ---- 天气匹配 ----
            if rule.weather_id == weather_id and weather_id:
                score += 2
            elif rule.weather_id == "all" and not self.strict_weather_match:
                score += 1
            else:
                continue

            matched.append((rule, score))

        # 按匹配分数降序，然后按优先级升序排序
        matched.sort(key=lambda x: (-x[1], x[0].priority))

        k = top_k or self.top_k
        return [rule for rule, _ in matched[:k]]

    def search_strategies(
        self,
        scene_id: str = "",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """检索匹配的驾驶策略。

        Args:
            scene_id: 场景类型。
            top_k: 返回数量。

        Returns:
            匹配的策略列表。
        """
        if not self._loaded:
            self.load()

        matched = []
        for strat in self._strategies:
            if strat.get("scene_id") == "all" or strat.get("scene_id") == scene_id:
                matched.append(strat)

        matched.sort(key=lambda x: x.get("priority", 5))
        k = top_k or self.top_k
        return matched[:k]

    def format_rules_text(self, rules: List[LongTermRule]) -> str:
        """将规则列表格式化为文本，用于拼接给决策 VLM。"""
        if not rules:
            return "无匹配的长期规则。"

        parts = []
        for rule in rules:
            parts.append(f"- [{rule.title}] (优先级: {rule.priority}): {rule.content.strip()}")

        return "\n".join(parts)

    def format_strategies_text(self, strategies: List[Dict[str, Any]]) -> str:
        """将策略列表格式化为文本，用于拼接给决策 VLM。

        Args:
            strategies: 策略字典列表。

        Returns:
            格式化后的策略文本。
        """
        if not strategies:
            return "无匹配的驾驶策略。"

        parts = []
        for strat in strategies:
            title = strat.get("title", "未知策略")
            desc = strat.get("description", "").strip()
            behavior = strat.get("recommended_behavior", "未知")
            speed_range = strat.get("target_speed_range", {})
            speed_min = speed_range.get("min", 0) if isinstance(speed_range, dict) else 0
            speed_max = speed_range.get("max", 30) if isinstance(speed_range, dict) else 30
            parts.append(
                f"- [{title}] 建议行为: {behavior}, "
                f"建议速度: {speed_min}-{speed_max} m/s\n"
                f"  {desc}"
            )

        return "\n".join(parts)
