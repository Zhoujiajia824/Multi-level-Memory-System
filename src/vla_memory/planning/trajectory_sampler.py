"""多模态轨迹采样接口（扩散 / 自回归规划器预留壳）
==================================================
P7 预留：不实现具体打分算法，仅冻结调用接口和数据约定。

设计目标
--------
当上游规划器（扩散模型、自回归 Transformer 等）输出 N 条候选轨迹时，
本模块负责：

1. 对每条候选轨迹评分（结合场景理解 / 自车状态 / 记忆上下文）。
2. 按 ``selection_policy`` 选出最终轨迹。
3. 返回选中轨迹 + 完整评分字典，供上层决策审计。

推荐内部实现（v0.2+）：
- 每条候选 score = w_safety * safety + w_comfort * comfort + w_progress * progress
- 安全分可拟合 CollisionRate（Phase 6 留的接口）；舒适分基于 jerk / max accel；
  进度分基于覆盖距离 / 与导航指令一致性。
- 也可接 learned reward model（IL / RL 训练的小型评分网络）。

接入示例
--------
::

    sampler = TrajectorySampler(selection_policy="argmax")
    chosen, scores = sampler.select(
        candidates=diffusion_model.sample(scene),  # List[List[dict]]
        scene_context=scene_understanding,
        ego_state=ego_state,
        memory_context={"short_term_summary": ..., "long_term_rules_text": ...},
    )
    # chosen 走原有 output_parser + dynamics_planner 流程
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple


SelectionPolicy = Literal["argmax", "softmax_sample", "top1_of_topk"]


class TrajectorySampler:
    """多模态轨迹候选 → 选定轨迹 + 评分。**v0.1 仅为接口壳，不做实质打分。**

    Args:
        selection_policy: 选择策略。
            * ``"argmax"`` —— 直接选总分最高的（默认）。
            * ``"softmax_sample"`` —— 按 softmax(score / temperature) 采样
              （用于 RL 探索 / 多样性）。
            * ``"top1_of_topk"`` —— 先按 score 取 top-K，再按某个次级准则（如多样性）选 1 个。
        score_weights: 各打分项权重字典，例如
            ``{"safety": 0.5, "comfort": 0.2, "progress": 0.3}``。
            未设置时由具体实现选默认。
        temperature: 仅 ``softmax_sample`` 用。越大越偏向均匀采样。
    """

    def __init__(
        self,
        selection_policy: SelectionPolicy = "argmax",
        score_weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
    ):
        self.selection_policy = selection_policy
        self.score_weights = dict(score_weights) if score_weights else {}
        self.temperature = float(temperature)

    def select(
        self,
        candidates: List[List[dict]],
        scene_context: Optional[Dict[str, Any]] = None,
        ego_state: Optional[Dict[str, Any]] = None,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[dict], Dict[int, Dict[str, float]]]:
        """从 N 条候选轨迹中选定一条。**未实现** —— 调用即抛 ``NotImplementedError``。

        Args:
            candidates: 候选轨迹列表，每条是 waypoint 列表
                ``[{"t": ..., "x": ..., "y": ..., "optional_v": ...}, ...]``。
            scene_context: VLM 场景理解结果（schemas/scene.SceneUnderstandingResult.model_dump）。
            ego_state: 当前自车状态（schemas/ego_state.EgoState.to_dict）。
            memory_context: 可选的记忆上下文字典，建议包含：
                * ``short_term_summary``: str
                * ``mid_term_memories``: List[dict]  （MemoryRetriever 结果）
                * ``long_term_rules_text``: str

        Returns:
            ``(chosen_trajectory, scores)``：
            * ``chosen_trajectory``: 选中的 waypoint 列表（直接可喂 output_parser）。
            * ``scores``: ``{candidate_idx: {"total": x, "safety": x, ...}}``，
              用于审计和可视化（写入 outputs/decisions_*.jsonl）。

        Raises:
            NotImplementedError: v0.1 仅为接口壳；要实现请参考模块 docstring。
        """
        raise NotImplementedError(
            "TrajectorySampler.select 是 v0.1 预留接口。\n"
            "本方法应对 N 条候选轨迹打分，按 selection_policy 选 1 条。\n"
            "推荐打分项：safety (collision proxy) + comfort (jerk/accel) "
            "+ progress (与导航指令一致性)。\n"
            "接入扩散 / 自回归规划器时实现：上游输出 candidates，本方法挑最优 "
            "→ output_parser → dynamics_planner → CARLA。"
        )

    def update_config(self, **kwargs: Any) -> None:
        """运行时覆盖某些参数。未知 key 抛 ``KeyError``。"""
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise KeyError(f"TrajectorySampler 没有参数 '{k}'")
            setattr(self, k, v)
