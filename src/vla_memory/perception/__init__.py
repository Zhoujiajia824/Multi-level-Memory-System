"""感知模块
==========
包含图像特征提取（DINOv2）和 VLM 场景理解。
不允许 mock VLM，不允许 mock feature。
"""
from src.vla_memory.perception.dinov2_extractor import DINOv2Extractor
from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient
from src.vla_memory.perception.scene_understanding import SceneUnderstandingPipeline
