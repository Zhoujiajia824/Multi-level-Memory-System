"""决策客户端
============
调用真实 VLM API 执行驾驶决策，返回原始文本供 pipeline 端解析。
不允许 mock VLM。API Key 不存在时 hard fail。

P5：``decide`` 接收 ``image_paths: List[str]``（oldest→newest，当前帧在末尾），
把短期记忆窗口里的历史图像一起喂给决策 VLM。
保留 ``image_path`` 单图参数作为向后兼容。
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from src.vla_memory.perception.vlm_client import VLMClient
from src.vla_memory.decision.prompt_builder import DecisionPromptBuilder
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("decision_client")


class DecisionClient:
    """决策客户端。

    构建决策 prompt，调用真实 VLM API，返回原始文本输出。
    JSON 解析和字段校验由 pipeline 端的 output_parser 负责。

    Args:
        vlm_client: VLM 客户端实例（必须是 VLMClient 子类）。
    """

    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client
        self.prompt_builder = DecisionPromptBuilder()

    def decide(
        self,
        image_path: Optional[str] = None,
        scene_understanding: Optional[Dict[str, Any]] = None,
        *,
        image_paths: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[str]:
        """执行驾驶决策。

        构建决策 prompt 并调用 VLM API，返回 VLM 原始文本输出。
        不做 JSON 解析——由调用方通过 parse_decision_output() 处理。

        Args:
            image_path: P5 之前的单图参数。仍可使用：当 ``image_paths`` 未提供
                且本参数非空时等价于 ``[image_path]``；新代码请用 ``image_paths``。
            scene_understanding: 场景理解结果字典。
            image_paths: P5 新增。完整的图像路径列表（oldest→newest，当前帧在末尾）。
            **kwargs: 其他上下文信息，传给 DecisionPromptBuilder。日志专用键
                ``frame_id`` 会被 pop 出来用于打印。

        Returns:
            VLM 原始文本输出（应为 JSON 字符串）。
            VLM 调用失败时返回 None。

        Raises:
            EnvironmentError: API Key 未设置时 hard fail。
        """
        # ---- 0. 日志专用 kwargs（不进入 prompt 构建） ----
        frame_id = kwargs.pop("frame_id", "?")

        # ---- 1. 合并图像参数（image_paths 优先） ----
        if image_paths is None:
            image_paths = [image_path] if image_path else []
        elif image_path and image_path not in image_paths:
            # 两者都传时取并集，避免悄悄丢图
            warnings.warn(
                "同时传入 image_path 和 image_paths，已自动合并到 image_paths 末尾",
                DeprecationWarning, stacklevel=2,
            )
            image_paths = list(image_paths) + [image_path]

        # ---- 2. 构建 prompt ----
        prompt = self.prompt_builder.build(
            scene_understanding=scene_understanding or {},
            **kwargs,
        )

        # ---- 3. 调用真实 VLM API ----
        try:
            raw_output = self.vlm_client.decide(
                prompt=prompt,
                image_paths=image_paths if image_paths else None,
            )
            # 打印/记录原始决策响应（便于审查提示词效果和推理过程）
            logger.info(
                "[DECISION] frame=%s images=%d (%s) raw_response=\n%s",
                frame_id,
                len(image_paths),
                ", ".join(image_paths) if image_paths else "<no image>",
                raw_output,
            )
            return raw_output
        except EnvironmentError:
            # API Key 错误，hard fail 不重试
            raise
        except Exception as e:
            logger.error(f"决策 VLM 调用失败: {e}")
            return None
