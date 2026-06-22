"""
场景理解流水线
==============
对关键帧图像使用 DINOv2 提取真实 embedding，并使用真实 VLM API
生成驾驶场景结构化理解结果。
第一版不允许 mock VLM，不允许 mock feature。
必须实现 JSON 解析、字段校验、失败重试。
重试后仍失败则记录错误并停止当前样本处理，不伪造输出。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.vla_memory.common.json_utils import extract_json_from_text
from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.common.prompt_loader import get_prompt_loader
from src.vla_memory.perception.dinov2_extractor import DINOv2Extractor
from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient
from src.vla_memory.schemas.scene import (
    SceneUnderstandingResult,
    VALID_SCENE_IDS,
    VALID_WEATHER_IDS,
)

logger = get_logger("scene_understanding")


class SceneUnderstandingPipeline:
    """场景理解流水线。

    对每个关键帧：
    1. 使用 DINOv2 提取真实图像 embedding。
    2. 调用真实 VLM API 进行场景理解。
    3. 解析和校验 JSON 输出。
    4. 保存 embedding 和场景理解结果。

    Args:
        feature_extractor: DINOv2 特征提取器实例。
        vlm_client: VLM 客户端实例。
        feature_save_dir: 特征保存目录。
        vlm_retry_times: VLM 调用失败时重试次数。
    """

    def __init__(
        self,
        feature_extractor: DINOv2Extractor,
        vlm_client: OpenAICompatibleVLMClient,
        feature_save_dir: str = "outputs/features",
        vlm_retry_times: int = 2,
    ):
        self.feature_extractor = feature_extractor
        self.vlm_client = vlm_client
        self.feature_save_dir = Path(feature_save_dir)
        self.vlm_retry_times = vlm_retry_times
        self.feature_save_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(
        self,
        sample_token: str,
        image_path: str,
    ) -> Optional[Dict[str, Any]]:
        """处理单个关键帧：提取特征 + VLM 场景理解。

        Args:
            sample_token: 样本 token（用于命名特征文件）。
            image_path: 图像文件路径。

        Returns:
            包含 image_feature_path 和 scene_understanding 结果的字典。
            失败时返回 None（不伪造输出）。
        """
        result = {
            "sample_token": sample_token,
            "image_feature_path": None,
            "scene_understanding": None,
        }

        # ---- 1. DINOv2 特征提取 ----
        try:
            feature = self.feature_extractor.extract(image_path)
            feature_path = self.feature_save_dir / f"{sample_token}.npy"
            self.feature_extractor.save_feature(feature, feature_path)
            result["image_feature_path"] = str(feature_path)
            logger.info(f"特征提取成功: {sample_token} -> {feature_path}")
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"特征提取失败 ({sample_token}): {e}")
            return None

        # ---- 2. VLM 场景理解 ----
        scene_result = self._run_vlm_scene_understanding(image_path)
        if scene_result is None:
            logger.error(f"场景理解最终失败: {sample_token}，停止当前样本处理。")
            return None

        result["scene_understanding"] = scene_result
        return result

    def _run_vlm_scene_understanding(
        self,
        image_path: str,
    ) -> Optional[Dict[str, Any]]:
        """调用 VLM 进行场景理解，含重试。

        Args:
            image_path: 图像路径。

        Returns:
            场景理解字典或 None。
        """
        last_error = None
        prompt_loader = get_prompt_loader()
        scene_prompt = prompt_loader.render("scene_understanding.user")

        for attempt in range(1, self.vlm_retry_times + 1):
            try:
                # 调用 VLM
                raw_output = self.vlm_client.understand_scene(
                    image_path=image_path,
                    prompt=scene_prompt,
                )

                # 打印 / 记录 VLM 原始响应（便于人工审查提示词效果）
                logger.info(
                    "[SCENE_UNDERSTANDING] image=%s raw_response=\n%s",
                    image_path,
                    raw_output,
                )

                # 保存原始输出
                result = self._parse_and_validate(raw_output)
                if result is None:
                    raise ValueError(
                        f"VLM 输出无法解析为有效 JSON。原始输出前300字符: {raw_output[:300]}"
                    )

                # 保存 raw_response
                result["raw_response"] = raw_output

                logger.info(
                    f"场景理解成功: scene_id={result.get('scene_id')}, "
                    f"weather_id={result.get('weather_id')}"
                )
                return result

            except (FileNotFoundError, EnvironmentError):
                raise  # hard fail，不重试
            except Exception as e:
                last_error = e
                logger.warning(
                    f"场景理解失败 (第 {attempt}/{self.vlm_retry_times} 次): {e}"
                )

        logger.error(f"场景理解最终失败: {last_error}")
        logger.error("停止当前样本处理，不伪造输出。")
        return None

    def _parse_and_validate(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """解析 VLM 输出为 JSON 并校验字段。

        P4：除原有字段外，新增 lanes / vehicles / pedestrians /
        traffic_lights / intersections 五个结构化字段的解析与降级。
        缺失字段会按类型填默认值并 warning，不中断流程。

        Args:
            raw_output: VLM 原始文本输出。

        Returns:
            校验通过的场景理解字典，失败返回 None。
        """
        # 解析 JSON
        result = extract_json_from_text(raw_output)
        if result is None:
            return None

        # 校验必需字段（从 prompts.yaml 读取列表，与模板保持同步）
        required_fields = get_prompt_loader().get_list(
            "scene_understanding.required_fields"
        )
        missing = [f for f in required_fields if f not in result]
        if missing:
            logger.warning(f"场景理解缺少字段: {missing}，使用默认值填充")

        # ---- 填充默认值 ----
        defaults = {
            # 旧字段
            "scene_description": "",
            "ego_status_text": "",
            "surrounding_objects": [],
            "lane_description": "",
            "traffic_density": "unknown",
            "risk_factors": [],
            "scene_id": "unknown",
            "weather_id": "unknown",
            # P4 新增结构化字段
            "lanes": [],
            "vehicles": [],
            "pedestrians": [],
            "traffic_lights": [],
            "intersections": {"present": False, "type": None, "distance_m": None, "has_stop_sign": None},
        }
        for key, default_val in defaults.items():
            result.setdefault(key, default_val)

        # ---- 校验枚举字段 ----
        if result["scene_id"] not in VALID_SCENE_IDS:
            logger.warning(f"无效 scene_id: {result['scene_id']}，修正为 unknown")
            result["scene_id"] = "unknown"

        if result["weather_id"] not in VALID_WEATHER_IDS:
            logger.warning(f"无效 weather_id: {result['weather_id']}，修正为 unknown")
            result["weather_id"] = "unknown"

        # ---- 校验数组字段（含 P4 新增） ----
        for list_field in ("surrounding_objects", "risk_factors",
                           "lanes", "vehicles", "pedestrians", "traffic_lights"):
            if not isinstance(result.get(list_field), list):
                logger.warning("%s 不是数组，修正为空数组", list_field)
                result[list_field] = []

        # ---- intersections 必须是 dict 且 present 字段为 bool ----
        if not isinstance(result.get("intersections"), dict):
            logger.warning("intersections 不是对象，修正为默认空路口")
            result["intersections"] = {"present": False, "type": None,
                                       "distance_m": None, "has_stop_sign": None}
        else:
            # 兜底 present 字段
            present = result["intersections"].get("present")
            if not isinstance(present, bool):
                result["intersections"]["present"] = bool(present) if present is not None else False

        return result
