"""CARLA tick 数据 -> kf dict（喂 OnlineDrivingLoop.step）
=========================================================
组装与 nuScenes 同构的关键帧 dict。``OnlineDrivingLoop.step()`` 读取的字段全部
在这里填好：``sample_token`` / ``scene_token`` / ``scene_name`` / ``image_path``
(mosaic) / ``timestamp`` / ``ego_state`` / ``history_trajectory`` /
``nav_instruction`` / ``perception_objects``。

``ground_truth_trajectory`` 留空：CARLA 在线闭环没有 GT 未来轨迹（未来由自车
自己决策产生），原 ADE/FDE 评测不适用，改用闭环安全指标（metrics/）。
"""
from __future__ import annotations

from typing import Any, Dict, List


class KeyframeBuilder:
    """组装 OnlineDrivingLoop.step() 所需的 kf dict。"""

    def __init__(self, scenario_name: str, scene_token: str = ""):
        self.scenario_name = scenario_name
        self._scene_token = scene_token or scenario_name
        self._counter = 0

    def next_token(self) -> str:
        """生成下一个 sample_token（单调递增，跨 tick 唯一）。"""
        self._counter += 1
        return f"carla_{self.scenario_name}_{self._counter:06d}"

    def build(
        self,
        sample_token: str,
        timestamp_us: int,
        mosaic_path: str,
        ego_state: Dict[str, Any],
        history_trajectory: List[dict],
        nav_instruction: str,
        perception_objects: List[dict],
    ) -> Dict[str, Any]:
        return {
            "sample_token": sample_token,
            "scene_token": self._scene_token,
            "scene_name": self.scenario_name,
            "image_path": mosaic_path,
            "timestamp": int(timestamp_us),
            "ego_state": ego_state,
            "history_trajectory": history_trajectory,
            "nav_instruction": nav_instruction,
            "perception_objects": perception_objects,
            "ground_truth_trajectory": [],  # CARLA 在线无 GT 未来
        }
