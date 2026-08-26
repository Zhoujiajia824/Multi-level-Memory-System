"""CARLA 闭环集成包
==================
把多层次记忆系统（``src/vla_memory``，**不改一行**）接入 CARLA 0.9.15 仿真环境：
用 CARLA 实时感知（六视角图像 / GT 障碍物 / 天气 / 导航 / 自车状态）替代离线
nuScenes 数据，决策轨迹实时回控 CARLA，实现自定义环境下的闭环驾驶。

核心思路
--------
* 复用 ``src.vla_memory.pipeline.OnlineDrivingLoop``（setup/step/close），每 3s
  用 CARLA 实时数据组装成与 nuScenes 同构的 ``kf`` dict 喂给 ``step()``，拿到
  决策轨迹。
* 同步模式 + 20Hz Pure Pursuit/PID 跟踪轨迹回控 CARLA。
* 所有新代码都在本包内，``src/vla_memory`` 与现有 ``config/`` 不动。

架构与使用见 ``carla_bridge/README.md``。
"""
from __future__ import annotations

__version__ = "0.1.0"
