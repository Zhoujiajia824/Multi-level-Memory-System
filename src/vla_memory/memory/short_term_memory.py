"""短期记忆模块
==============
使用 deque 滑动窗口存储最近 10 个关键帧数据。
表示当前局部时间上下文，生成短期记忆摘要供决策 VLM 使用。
"""
from __future__ import annotations
from collections import deque
from typing import List, Optional, Dict, Any
from src.vla_memory.schemas.memory import ShortTermMemoryItem
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("short_term_memory")


class ShortTermMemory:
    """短期记忆管理器。

    使用 deque 实现滑动窗口，存储最近 N 个关键帧数据。

    Args:
        capacity: 滑动窗口大小，默认 10。
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity  # 滑动窗口大小
        self._buffer: deque[ShortTermMemoryItem] = deque(maxlen=capacity)

    def add(self, item: ShortTermMemoryItem) -> None:
        """添加一个关键帧到短期记忆。

        Args:
            item: 短期记忆项。
        """
        self._buffer.append(item)
        logger.debug(f"短期记忆添加帧: {item.frame_id}, 当前容量: {len(self._buffer)}/{self.capacity}")

    def get_all(self) -> List[ShortTermMemoryItem]:
        """获取所有短期记忆项。"""
        return list(self._buffer)

    def get_latest(self, n: int = 1) -> List[ShortTermMemoryItem]:
        """获取最近 n 个短期记忆项。"""
        n = min(n, len(self._buffer))
        return list(self._buffer)[-n:]

    def get_recent_items(self, n: int) -> List[ShortTermMemoryItem]:
        """获取最近 n 个短期记忆项（别名，oldest-first 顺序）。

        与 get_latest 区别仅在命名语义：本方法明确表达"用于上下文整合"的意图，
        供 P5 决策 pipeline 把图片历史拼到 VLM 输入中。

        Args:
            n: 期望返回的条数。会自动截断到当前可用数。

        Returns:
            最近 n 条短期记忆项，顺序：oldest -> newest。
        """
        if n <= 0:
            return []
        n = min(n, len(self._buffer))
        return list(self._buffer)[-n:]

    def get_recent_image_paths(self, n: int) -> List[str]:
        """获取最近 n 帧的图像路径（oldest-first，过滤空字符串）。

        P5 决策 pipeline 会把这些路径与当前帧合并成 image_paths 列表，
        通过 OpenAI-compatible VLM 客户端按需 base64 编码喂给决策模型。

        Args:
            n: 期望返回的路径条数。

        Returns:
            最近 n 条短期记忆项的 image_path 列表（已去掉空字符串），
            顺序：oldest -> newest。
        """
        items = self.get_recent_items(n)
        return [it.image_path for it in items if it.image_path]

    def generate_summary(self, max_length: int = 2000) -> str:
        """生成短期记忆摘要文本，用于拼接给决策 VLM。

        摘要包含最近关键帧的场景描述、行为和关键状态信息。

        Args:
            max_length: 摘要最大字符长度。

        Returns:
            短期记忆摘要文本。
        """
        if not self._buffer:
            return "暂无短期记忆数据。"

        parts = []
        total_len = 0

        # 从最近的帧开始，向前遍历
        for i, item in enumerate(reversed(self._buffer)):
            frame_desc = (
                f"[t-{i}] 场景: {item.scene_id or '未知'}, "
                f"天气: {item.weather_id or '未知'}, "
                f"导航: {item.nav_instruction or '未知'}, "
                f"速度: {item.ego_state.get('speed', 'N/A') if item.ego_state else 'N/A'} m/s"
            )
            if item.scene_description:
                frame_desc += f"\n  描述: {item.scene_description[:100]}"

            if total_len + len(frame_desc) > max_length:
                break

            parts.append(frame_desc)
            total_len += len(frame_desc)

        summary = "=== 短期记忆摘要（最近关键帧） ===\n"
        summary += "\n".join(parts)
        return summary

    def clear(self) -> None:
        """清空短期记忆。"""
        self._buffer.clear()
        logger.info("短期记忆已清空。")

    def __len__(self) -> int:
        return len(self._buffer)

    def is_full(self) -> bool:
        """检查短期记忆是否已满。"""
        return len(self._buffer) >= self.capacity
