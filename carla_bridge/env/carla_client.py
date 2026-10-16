"""CARLA 客户端连接 + 同步模式管理
=================================
连接用户手动启动的 CARLA 服务器（``CarlaUE4.exe``，端口 2000），设置同步模式 +
固定步长，提供 ``tick`` 接口。运行结束恢复服务器原始设置，避免把服务器卡在
同步态。

同步模式要点
------------
* 服务器只在客户端调用 ``world.tick()`` 时推进一帧。
* 传感器（相机）在 tick 后才把数据投递到各自 queue，故捕获流程必须是
  ``tick -> 从 queue 取图``。
* VLM 思考期间不 tick = 仿真冻结，天然适配 5-30s 的慢决策。

本模块需在 ``mulmem_carla``（Python 3.9，含 carla 绑定）环境运行。
"""
from __future__ import annotations

from typing import Any, Optional

import carla  # 须在 mulmem_carla(3.9) 环境运行

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("carla_client")


class CarlaClient:
    """CARLA 同步客户端封装。

    Args:
        host: 服务器地址。
        port: 服务器端口（默认 2000）。
        timeout_s: 客户端超时（秒）。
        synchronous: 是否同步模式（闭环必须 True）。
        fixed_delta_seconds: 固定步长（秒），默认 0.05 = 20Hz。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout_s: float = 30.0,
        synchronous: bool = True,
        fixed_delta_seconds: float = 0.05,
    ):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.synchronous = synchronous
        self.fixed_delta_seconds = fixed_delta_seconds
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self._orig_settings: Optional[Any] = None

    def connect(self) -> "CarlaClient":
        """连接服务器并应用同步设置。"""
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout_s)
        self.world = self.client.get_world()
        self._orig_settings = self.world.get_settings()
        self._apply_sync_settings()
        logger.info(
            "CARLA 已连接 %s:%d (同步=%s, fixed_delta=%s)",
            self.host, self.port, self.synchronous, self.fixed_delta_seconds,
        )
        return self

    def _apply_sync_settings(self) -> None:
        assert self.world is not None
        settings = self.world.get_settings()
        settings.synchronous_mode = self.synchronous
        if self.synchronous and self.fixed_delta_seconds > 0:
            settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

    def tick(self) -> int:
        """推进一帧（同步模式）。返回 frame id。"""
        assert self.world is not None
        return self.world.tick()

    def get_world(self) -> carla.World:
        """返回当前 world 对象。"""
        assert self.world is not None
        return self.world

    def load_map(self, map_name: str) -> carla.World:
        """切换地图。``load_world`` 会重置服务器设置，故切换后重新应用同步模式。"""
        assert self.client is not None
        self.world = self.client.load_world(map_name)
        self._orig_settings = self.world.get_settings()
        self._apply_sync_settings()
        logger.info("已加载地图 %s 并重应用同步设置", map_name)
        return self.world

    def set_spectator_follow(self, ego_vehicle, back_m: float = 8.0, up_m: float = 6.0) -> None:
        """把 CARLA spectator 摆到 ego 后上方第三人称跟随视角（每个 tick 调用以跟随）。"""
        if self.world is None:
            return
        tf = ego_vehicle.get_transform()
        fv = tf.get_forward_vector()  # 单位前向向量
        cam_loc = carla.Location(
            x=tf.location.x - fv.x * back_m,
            y=tf.location.y - fv.y * back_m,
            z=tf.location.z + up_m,
        )
        cam_tf = carla.Transform(cam_loc, carla.Rotation(yaw=tf.rotation.yaw, pitch=-15.0))
        self.world.get_spectator().set_transform(cam_tf)

    def restore(self) -> None:
        """恢复服务器原始（异步）设置，释放同步控制。"""
        if self.world is not None and self._orig_settings is not None:
            try:
                self.world.apply_settings(self._orig_settings)
                logger.info("CARLA 服务器设置已恢复")
                self._orig_settings = None
            except Exception as e:
                logger.warning("恢复 CARLA 设置失败: %s", e)
