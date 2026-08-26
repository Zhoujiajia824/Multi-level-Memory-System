"""6 相机捕获与 mosaic 拼接
==========================
在 ego 上挂 6 个 RGB 相机（前左/前/前右/后左/后/后右，对应 nuScenes 2×3 布局），
每个 tick 后从各自 queue 取图落盘，并复用 ``src`` 的 ``build_surround_mosaic``
拼成 2×3 单张图喂给 VLM / DINOv2 / 记忆全流程。

同步模式要点：相机图在 ``world.tick()`` 后才到 queue，故捕获流程必须是
``tick -> 从 queue 取图 -> 拼 mosaic``。首帧可能未渲染，安装后先 warmup 一个 tick。
"""
from __future__ import annotations

import queue
from pathlib import Path
from typing import Dict, List

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.perception.surround_mosaic import build_surround_mosaic

logger = get_logger("carla_camera")


class CameraManager:
    """6 相机管理器：安装、捕获、拼 mosaic、销毁。

    Args:
        world: carla.World。
        ego_vehicle: 自车 actor。
        cameras_cfg: 6 相机配置列表（来自 ``carla.yaml -> carla.cameras``）。
        image_dir: 单相机图落盘目录。
        mosaic_dir: mosaic 落盘目录。
        mosaic_cell_w / mosaic_cell_h / label_subimages: mosaic 拼接参数
            （与 ``perception.mosaic`` 对齐）。
    """

    def __init__(
        self,
        world,
        ego_vehicle,
        cameras_cfg: List[dict],
        image_dir: str,
        mosaic_dir: str,
        mosaic_cell_w: int = 480,
        mosaic_cell_h: int = 270,
        label_subimages: bool = True,
    ):
        self.world = world
        self.ego = ego_vehicle
        self.cameras_cfg = cameras_cfg
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mosaic_dir = Path(mosaic_dir)
        self.mosaic_dir.mkdir(parents=True, exist_ok=True)
        self.mosaic_cell_w = mosaic_cell_w
        self.mosaic_cell_h = mosaic_cell_h
        self.label_subimages = label_subimages
        self._sensors: Dict[str, carla.Sensor] = {}
        self._queues: Dict[str, "queue.Queue"] = {}
        self._installed = False

    def install(self) -> None:
        """生成 6 个相机 actor 挂到 ego，每个绑定独立 queue。"""
        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.find("sensor.camera.rgb")
        for cfg in self.cameras_cfg:
            name = cfg["name"]
            bp.set_attribute("image_size_x", str(cfg.get("width", 1280)))
            bp.set_attribute("image_size_y", str(cfg.get("height", 720)))
            bp.set_attribute("fov", str(cfg.get("fov", 70)))
            tf = carla.Transform(
                carla.Location(x=float(cfg["x"]), y=float(cfg["y"]), z=float(cfg["z"])),
                carla.Rotation(
                    pitch=float(cfg.get("pitch", 0)),
                    yaw=float(cfg.get("yaw", 0)),
                    roll=float(cfg.get("roll", 0)),
                ),
            )
            sensor = self.world.spawn_actor(bp, tf, attach_to=self.ego)
            q: "queue.Queue" = queue.Queue()
            sensor.listen(q.put)
            self._sensors[name] = sensor
            self._queues[name] = q
        self._installed = True
        logger.info("6 相机已挂载: %s", list(self._sensors.keys()))

    def warmup(self, n_frames: int = 2) -> None:
        """空转几帧让传感器开始投递数据，避免首帧取图超时。"""
        for _ in range(n_frames):
            self.world.tick()

    def capture(self, sample_token: str) -> str:
        """从 6 路 queue 取图落盘 + 拼 mosaic，返回 mosaic 路径。

        须在 ``world.tick()`` 之后调用。``sample_token`` 用于命名文件。
        """
        assert self._installed, "先 install()"
        names = [c["name"] for c in self.cameras_cfg]
        image_paths: List[str] = []
        missing: List[str] = []
        for name in names:
            q = self._queues[name]
            # 排空旧帧（控制 tick 期间堆积的帧），取最新一帧；队列为空则等待
            image = None
            try:
                while True:
                    image = q.get_nowait()
            except queue.Empty:
                pass
            if image is None:
                try:
                    image = q.get(timeout=0.5)
                except Exception:
                    missing.append(name)
                    image_paths.append("")
                    continue
            path = str(self.image_dir / f"{sample_token}_{name}.jpg")
            self._save_image(image, path)
            image_paths.append(path)
        if missing:
            # 聚合为单次 warning，避免 6 相机各自刷屏
            logger.warning("相机取图超时（%d/%d 缺帧）: %s", len(missing), len(names), missing)
            path = str(self.image_dir / f"{sample_token}_{name}.jpg")
            self._save_image(image, path)
            image_paths.append(path)

        mosaic_path = str(self.mosaic_dir / f"{sample_token}.jpg")
        build_surround_mosaic(
            image_paths, names,
            cell_width=self.mosaic_cell_w, cell_height=self.mosaic_cell_h,
            label_subimages=self.label_subimages, out_path=mosaic_path,
        )
        return mosaic_path

    def drain(self) -> None:
        """排空所有相机队列（控制 tick 期间丢弃不需要的帧，防止堆积）。

        重规划间隔（默认 3s）一次捕获的架构下，相机虽随每个 tick 渲染，但控制
        tick 主动 drain 丢弃，队列永不堆积；捕获时再取最新一帧。
        """
        if not self._installed:
            return
        for q in self._queues.values():
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    @staticmethod
    def _save_image(image, path: str) -> None:
        """carla.Image(BGRA) -> JPEG 落盘。"""
        import numpy as np
        from PIL import Image
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (image.height, image.width, 4))
        arr = arr[:, :, :3]          # 丢 alpha
        arr = arr[:, :, ::-1]         # BGR -> RGB
        Image.fromarray(arr).save(path, "JPEG", quality=90)

    def destroy(self) -> None:
        for s in self._sensors.values():
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self._sensors.clear()
        self._queues.clear()
        self._installed = False
