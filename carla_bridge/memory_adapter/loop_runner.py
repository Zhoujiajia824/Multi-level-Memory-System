"""封装 OnlineDrivingLoop（不改 src/）
====================================
CARLA 闭环里每 3s 重规划调一次 ``step()``：

* ``setup`` 初始化 DINOv2 / 双 VLM / 三层记忆（含 7 阶段价值门控全部生效）；
* ``step(kf)`` 跑完整 11 步（感知 -> 检索 -> 决策 -> 更新记忆），返回 record
  （含 ``decision_output.trajectory``）；
* ``close`` 落盘中期记忆（按 ``memory.yaml -> mid_term.persistence``）。

仅做薄封装，记忆系统一行不改。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.vla_memory.common.config import Config
from src.vla_memory.pipeline.online_loop import OnlineDrivingLoop, default_output_path


class LoopRunner:
    """OnlineDrivingLoop 的 CARLA 侧薄封装。"""

    def __init__(
        self,
        config: Config,
        mode: str,
        output_jsonl_path: Optional[str] = None,
        resume: bool = False,
    ):
        output_path = (
            Path(output_jsonl_path) if output_jsonl_path
            else default_output_path(config, mode)
        )
        self.loop = OnlineDrivingLoop(
            config=config,
            mode=mode,
            output_jsonl_path=output_path,
            resume=resume,
        )
        self.output_path = output_path

    def setup(self) -> None:
        self.loop.setup()

    def step(self, kf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """跑一帧完整记忆-决策管线，返回 record（含 decision_output）或 None。"""
        return self.loop.step(kf)

    def add_short_term_item(self, item) -> None:
        """直接往短期记忆 push 一个 item（供控制阶段 raw 捕获用，不经 VLM）。

        CARLA 闭环在控制阶段以 ~5Hz 做原始感知捕获，沿用上次 VLM 场景结果，
        使短期记忆队列时刻保持最新；完整 VLM 仍在 ``step`` 里 1Hz 跑。
        """
        self.loop.short_term.add(item)

    def close(self) -> None:
        self.loop.close()
