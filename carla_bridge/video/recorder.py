"""CARLA 驾驶视频录制器
======================
独立小功能：录制 ego 连续驾驶视频。

核心原理
--------
* 仅在每个 **CARLA 推进 tick**（捕获 tick + 控制 tick）取一帧写入视频；
  VLM 思考期间 CARLA 冻结、不 tick，**不取帧** -> 视频自动跳过模型反应时间。
* 视频按 ``fps = 1/fixed_delta``（当前 10Hz 配置下为 10）播放，故 **视频时长 = 仿真实际驾驶时间**。
  例如一次运行中车辆真实驾驶 30s（不含 VLM），视频就是 30s 连续驾驶。
* 中断/Ctrl+C 时 :meth:`flush` 释放写句柄，按已录帧数保存（= 已驾驶时间）。

视图相机挂在 ego 上（chase 后上方第三人称，或 front 前视角），跟随自动。
需 opencv-python（已在 requirements.txt）。
"""
from __future__ import annotations

import queue
from pathlib import Path

import carla  # 须在 mulmem_carla(3.9) 运行
import cv2
import numpy as np

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("carla_video")


class VideoRecorder:
    """CARLA ego 连续驾驶视频录制器。"""

    def __init__(
        self,
        world,
        ego,
        output_path: str,
        fps: int = 20,
        width: int = 1280,
        height: int = 720,
        view: str = "chase",
    ):
        self.world = world
        self.ego = ego
        self.output_path = str(output_path)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.view = view
        self._sensor = None
        self._queue: "queue.Queue" = queue.Queue()
        self._writer = None
        self._frame_count = 0

    # ------------------------------------------------------------------

    def start(self) -> None:
        """挂载视图相机 + 打开 VideoWriter。"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov", "90")
        self._sensor = self.world.spawn_actor(
            bp, self._view_transform(), attach_to=self.ego
        )
        self._sensor.listen(self._queue.put)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        logger.info(
            "视频录制开始: %s (%dx%d @ %dfps, view=%s)",
            self.output_path, self.width, self.height, self.fps, self.view,
        )

    def _view_transform(self) -> "carla.Transform":
        """视图相机相对 ego 的 transform（attach 到 ego，自动跟随）。"""
        if self.view == "front":
            return carla.Transform(
                carla.Location(x=2.0, y=0.0, z=1.4), carla.Rotation(pitch=0, yaw=0)
            )
        # chase：后上方第三人称
        return carla.Transform(
            carla.Location(x=-7.0, y=0.0, z=4.0), carla.Rotation(pitch=-15, yaw=0)
        )

    # ------------------------------------------------------------------

    def on_tick(self) -> None:
        """每个 CARLA 推进 tick 调用：取最新一帧写入视频。

        VLM 冻结期不调用本方法 -> 自动跳过；故视频只含连续驾驶帧。
        """
        if self._writer is None:
            return
        frame = None
        try:
            while True:
                frame = self._queue.get_nowait()
        except queue.Empty:
            pass
        if frame is None:
            return
        arr = np.frombuffer(frame.raw_data, dtype=np.uint8).reshape(
            frame.height, frame.width, 4
        )
        bgr = arr[:, :, :3]  # carla BGRA -> BGR（cv2 原生）
        self._writer.write(bgr)
        self._frame_count += 1

    def flush(self) -> None:
        """释放写句柄 + 销毁相机，保存视频。中断时调用以保住已录帧。"""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception:
                pass
            try:
                self._sensor.destroy()
            except Exception:
                pass
            self._sensor = None
        dur = self._frame_count / max(1, self.fps)
        logger.info(
            "视频录制结束: %s (%d 帧, 约 %.1fs = 仿真驾驶时间)",
            self.output_path, self._frame_count, dur,
        )
