"""
智能驾驶 VLA 分层记忆系统
========================
核心模块包，包含以下子模块：

- common: 通用工具（配置、日志、路径、JSON、图像IO）
- schemas: 数据模型（关键帧、自车状态、轨迹、场景、记忆、决策、评测）
- data: 数据适配器（nuScenes、视频、图像序列、导航推断、状态构建、轨迹构建）
- keyframes: 关键帧采样（nuScenes keyframe、周期采样）
- perception: 感知模块（DINOv2 特征提取、VLM 客户端、场景理解）
- memory: 记忆系统（短期、中期、长期、FAISS 向量存储、检索）
- decision: 决策模块（Prompt 构建、决策客户端、输出解析、规则 Fallback）
- evaluation: 评测模块（指标、评测器、报告生成）
- pipeline: 流水线（数据准备、场景理解、记忆构建、决策、评测、完整 Demo）
"""

__version__ = "0.1.0"
__project_name__ = "vla_memory_demo"
