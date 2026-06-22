"""
OpenAI 兼容 VLM 客户端
========================
使用 openai Python SDK 调用 VLM。
从 config/api_models.yaml 读取 provider / api_key_env / base_url / model_name 等。
API Key 从环境变量读取，不存在时 hard fail。
图像以 base64 方式传入。
支持失败重试，重试后仍失败则 hard fail。
不允许使用 mock response。
"""
from __future__ import annotations

import base64
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional

from src.vla_memory.perception.vlm_client import VLMClient
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("openai_compatible_client")


class OpenAICompatibleVLMClient(VLMClient):
    """OpenAI 兼容 VLM 客户端。

    支持所有兼容 OpenAI API 格式的视觉模型服务
    （Qwen-VL、硅基流动、智谱、DeepSeek-VL 等）。

    Args:
        provider: VLM 提供商标识。
        api_key_env: API Key 对应的环境变量名。
        base_url: API base URL。
        model_name: 模型名称。
        timeout: 请求超时（秒）。
        max_tokens: 最大输出 token 数。
        temperature: 生成温度。
        retry_times: 失败重试次数。
        retry_interval_seconds: 重试间隔（秒）。
        system_prompt: 默认系统提示词。
    """

    def __init__(
        self,
        provider: str = "qwen",
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen-vl-max",
        timeout: int = 60,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        retry_times: int = 3,
        retry_interval_seconds: int = 5,
        system_prompt: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_times = retry_times
        self.retry_interval = retry_interval_seconds
        self.system_prompt = system_prompt
        self._client = None

    # ================================================================
    # API Key 管理
    # ================================================================

    def check_api_key(self) -> str:
        """检查 API Key 是否存在，不存在则 hard fail。

        Returns:
            API Key 字符串。

        Raises:
            EnvironmentError: API Key 环境变量未设置。
        """
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"VLM API Key 未设置！环境变量 {self.api_key_env} 为空或不存在。\n"
                f"请设置环境变量:\n"
                f"  Windows CMD:  set {self.api_key_env}=your-api-key\n"
                f"  PowerShell:  $env:{self.api_key_env}='your-api-key'\n"
                f"  Linux/Mac:   export {self.api_key_env}='your-api-key'\n"
                f"或在项目根目录 .env 文件中添加 {self.api_key_env}=your-api-key\n"
                f"第一版必须使用真实 VLM API，不允许 mock VLM。"
            )
        return api_key

    def _ensure_client(self) -> None:
        """确保 OpenAI 客户端已初始化。"""
        if self._client is not None:
            return

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai 库未安装！\n请运行: pip install openai"
            )

        api_key = self.check_api_key()
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        logger.info(
            f"VLM 客户端初始化: provider={self.provider}, "
            f"model={self.model_name}, base_url={self.base_url}"
        )

    # ================================================================
    # 图像编码
    # ================================================================

    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        """将图像文件编码为 base64 字符串。

        Args:
            image_path: 图像文件路径。

        Returns:
            base64 编码字符串。

        Raises:
            FileNotFoundError: 图像文件不存在。
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        with open(str(path), "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ================================================================
    # 核心调用
    # ================================================================

    def _call_with_retry(self, messages: list) -> str:
        """带重试的 VLM API 调用。

        Args:
            messages: OpenAI 格式的消息列表。

        Returns:
            VLM 原始文本输出。

        Raises:
            RuntimeError: 重试耗尽后仍失败。
        """
        self._ensure_client()

        last_error = None
        for attempt in range(1, self.retry_times + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                result = response.choices[0].message.content
                logger.debug(f"VLM 调用成功 (尝试 {attempt}/{self.retry_times})")
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"VLM 调用失败 (第 {attempt}/{self.retry_times} 次): {e}"
                )
                if attempt < self.retry_times:
                    logger.info(f"等待 {self.retry_interval}s 后重试...")
                    time.sleep(self.retry_interval)

        raise RuntimeError(
            f"VLM API 调用失败，已重试 {self.retry_times} 次。\n"
            f"最后错误: {last_error}\n"
            f"请检查:\n"
            f"  1. API Key 是否正确（环境变量 {self.api_key_env}）\n"
            f"  2. 网络是否正常\n"
            f"  3. 模型名称 '{self.model_name}' 是否正确\n"
            f"  4. base_url '{self.base_url}' 是否正确"
        )

    # ================================================================
    # 接口实现
    # ================================================================

    def understand_scene(
        self,
        image_path: str,
        prompt: str,
        extra_context: Optional[dict] = None,
    ) -> str:
        """调用 VLM 进行驾驶场景理解（带图像）。

        Args:
            image_path: 图像文件路径。
            prompt: 场景理解 prompt。
            extra_context: 额外上下文（可选）。

        Returns:
            VLM 原始输出（JSON 字符串）。
        """
        base64_image = self.encode_image_base64(image_path)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt},
        ]
        if extra_context:
            context_text = "\n".join(f"{k}: {v}" for k, v in extra_context.items())
            user_content.append({"type": "text", "text": f"\n额外上下文:\n{context_text}"})

        messages.append({"role": "user", "content": user_content})

        return self._call_with_retry(messages)

    def decide(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        optional_image_path: Optional[str] = None,
        max_images: int = 4,
    ) -> str:
        """调用 VLM 进行驾驶决策（文本 + 0~N 张图像）。

        P5：把 ``image_paths`` 中的每张图作为独立 ``image_url`` content block
        拼到 user message 中，prompt 文本附在末尾。Qwen-VL-Max 实测建议
        单次 ≤ 4 张图，超出时截断最近 N 张并发 warning。

        Args:
            prompt: 决策 prompt。
            image_paths: 图像路径列表（oldest→newest，当前帧在末尾）。
            optional_image_path: 单图兼容参数；仅当 ``image_paths`` 未提供时生效。
            max_images: 单次拼接的图片张数上限。

        Returns:
            VLM 原始输出（JSON 字符串）。
        """
        # ---- 兼容旧调用方：optional_image_path 自动包成列表 ----
        if image_paths is None:
            if optional_image_path:
                warnings.warn(
                    "decide(optional_image_path=...) 已弃用，请改用 image_paths=[...]",
                    DeprecationWarning, stacklevel=2,
                )
                image_paths = [optional_image_path]
            else:
                image_paths = []

        # ---- 图片数量上限 ----
        if len(image_paths) > max_images:
            logger.warning(
                "image_paths 共 %d 张，超过 max_images=%d，仅保留最近 %d 张",
                len(image_paths), max_images, max_images,
            )
            image_paths = image_paths[-max_images:]

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # ---- 拼接 user content ----
        if image_paths:
            user_content = []
            for p in image_paths:
                b64 = self.encode_image_base64(p)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            user_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        return self._call_with_retry(messages)
