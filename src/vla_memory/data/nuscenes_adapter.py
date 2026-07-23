"""
nuScenes 数据集适配器
=====================
实现 BaseDrivingDataset 接口，封装 nuscenes-devkit 的数据访问。
提供场景遍历、帧迭代、图像路径获取、位姿查询、轨迹构建等功能。
第一版使用 v1.0-mini，默认 CAM_FRONT。

数据集路径不存在时必须 hard fail，输出中文说明，不允许使用假数据。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator

from src.vla_memory.data.base_dataset import BaseDrivingDataset
from src.vla_memory.schemas.frame import FrameMeta
from src.vla_memory.schemas.ego_state import EgoState
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("nuscenes_adapter")


class NuScenesAdapter(BaseDrivingDataset):
    """nuScenes 数据集适配器。

    封装 nuscenes-devkit 的 NuScenes 类，实现 BaseDrivingDataset 接口。
    从 config/data_nuscenes.yaml 读取 dataroot、version、camera_name。

    Args:
        dataroot: nuScenes 数据集根目录。
        version: 数据集版本，如 'v1.0-mini'。
        camera_name: 摄像头名称，默认 'CAM_FRONT'。
    """

    def __init__(
        self,
        dataroot: str | Path = "data/nuscenes/raw",
        version: str = "v1.0-mini",
        camera_name: str = "CAM_FRONT",
        camera_names: Optional[List[str]] = None,
        can_bus_loader: Optional[object] = None,
        fallback_to_pose_diff: bool = True,
    ):
        """
        Args:
            dataroot: nuScenes 数据集根目录。
            version: 数据集版本。
            camera_name: 主摄像头名称（用于 ego_pose 关联与 single_front 模式）。
            camera_names: 多摄像头名称列表（surround_mosaic 模式用）；None 时退化为 [camera_name]。
            can_bus_loader: P3 新增。可选 CanBusLoader 实例；提供时
                ``get_ego_pose`` 优先返回 CAN bus 真值。
            fallback_to_pose_diff: CAN bus 查询失败时是否回退到差分（默认 True）。
        """
        self.dataroot = Path(dataroot)
        self.version = version
        self.camera_name = camera_name
        # 多摄像头列表：surround_mosaic 模式下为 6 相机；否则退化为 [主摄像头]
        self.camera_names: List[str] = list(camera_names) if camera_names else [camera_name]
        self._nusc = None  # 底层 NuScenes 对象
        self._loaded = False

        # 缓存：scene_token -> 该场景下所有 sample 按时间排序
        self._scene_samples_cache: Dict[str, List[dict]] = {}
        # 缓存：sample_token -> ego_pose 字典
        self._ego_pose_cache: Dict[str, dict] = {}

        # P3：CAN bus 真值
        self.can_bus_loader = can_bus_loader
        self.fallback_to_pose_diff = fallback_to_pose_diff
        # 每 scene 仅 warning 一次，避免日志刷屏
        self._can_bus_warned: set = set()

    # ================================================================
    # 生命周期
    # ================================================================

    def load(self) -> None:
        """加载 nuScenes 数据集。

        Raises:
            FileNotFoundError: 数据集目录或版本目录不存在。
            ImportError: nuscenes-devkit 未安装。
            RuntimeError: 数据集加载失败。
        """
        # 检查数据集根目录
        if not self.dataroot.exists():
            raise FileNotFoundError(
                f"nuScenes 数据集目录不存在: {self.dataroot}\n"
                f"请将 nuScenes 数据解压到 {self.dataroot}，"
                f"使其包含 samples、sweeps、maps、{self.version}。\n"
                f"不允许使用假数据。"
            )

        # 检查版本目录
        version_dir = self.dataroot / self.version
        if not version_dir.exists():
            raise FileNotFoundError(
                f"nuScenes 数据集版本目录不存在: {version_dir}\n"
                f"请确保已正确下载 {self.version} 版本数据集并解压到正确位置。\n"
                f"目录结构应为:\n"
                f"  {self.dataroot}/\n"
                f"    {self.version}/\n"
                f"    samples/\n"
                f"    maps/\n"
                f"    sweeps/"
            )

        # 检查 samples 目录
        samples_dir = self.dataroot / "samples"
        if not samples_dir.exists():
            raise FileNotFoundError(
                f"nuScenes samples 目录不存在: {samples_dir}\n"
                f"请确保数据集解压完整。"
            )

        # 加载 nuscenes-devkit
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError:
            raise ImportError(
                "nuscenes-devkit 未安装！\n"
                "请安装: pip install nuscenes-devkit"
            )

        try:
            logger.info(f"正在加载 nuScenes 数据集: version={self.version}, dataroot={self.dataroot}")
            self._nusc = NuScenes(
                version=self.version,
                dataroot=str(self.dataroot),
                verbose=True,
            )
            self._loaded = True
            logger.info(
                f"nuScenes 数据集加载成功: "
                f"{len(self._nusc.scene)} 个场景, "
                f"{len(self._nusc.sample)} 个样本"
            )
        except Exception as e:
            raise RuntimeError(f"nuScenes 数据集加载失败: {e}")

    def is_loaded(self) -> bool:
        """检查数据集是否已加载。"""
        return self._loaded

    @property
    def nusc(self):
        """获取底层 NuScenes 对象。

        Raises:
            RuntimeError: 数据集尚未加载。
        """
        if not self._loaded:
            raise RuntimeError(
                "数据集尚未加载，请先调用 load() 方法。"
            )
        return self._nusc

    # ================================================================
    # 场景 & 帧访问
    # ================================================================

    def list_scenes(self) -> List[str]:
        """列出所有场景 token。"""
        return [scene["token"] for scene in self.nusc.scene]

    def iter_frames(self, scene_token: str) -> Iterator[FrameMeta]:
        """遍历指定场景中的所有帧元数据。

        Args:
            scene_token: 场景 token。

        Yields:
            FrameMeta 实例。
        """
        samples = self._get_scene_samples(scene_token)
        for sample in samples:
            sample_token = sample["token"]
            image_path = self.get_frame_image_path(sample_token, self.camera_name)
            image_paths = self.get_frame_image_paths(sample_token, self.camera_names)
            yield FrameMeta(
                frame_id=f"{scene_token}_{sample_token}",
                scene_token=scene_token,
                sample_token=sample_token,
                timestamp=sample["timestamp"],
                camera_name=self.camera_name,
                image_path=image_path,
                image_paths=image_paths,
            )

    def get_sample_count(self) -> int:
        """获取样本总数。"""
        return len(self.nusc.sample)

    def get_scene_description(self, scene_token: str) -> str:
        """获取场景的描述信息。"""
        scene = self.nusc.get("scene", scene_token)
        return scene.get("description", scene.get("name", ""))

    # ================================================================
    # 数据获取
    # ================================================================

    def get_frame_image_path(self, sample_token: str, camera_name: str = "CAM_FRONT") -> str:
        """获取指定帧的图像文件绝对路径。"""
        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"].get(camera_name)
        if cam_token is None:
            raise ValueError(
                f"样本 {sample_token} 没有摄像头 {camera_name} 的数据"
            )
        cam_data = self.nusc.get("sample_data", cam_token)
        return str(self.dataroot / cam_data["filename"])

    def get_frame_image_paths(
        self, sample_token: str, camera_names: Optional[List[str]] = None
    ) -> List[str]:
        """获取指定帧在多个摄像头下的图像文件绝对路径列表（surround_mosaic 模式用）。

        按 ``camera_names`` 顺序返回各相机图像路径，保持顺序与 2x3 布局对齐。
        某相机通道缺失（nuScenes 正常情况下不会发生）时填空串占位并 warning，
        由 mosaic 拼接层将其作为缺图处理。

        Args:
            sample_token: 样本标识。
            camera_names: 摄像头名称列表；None 时用 self.camera_names。

        Returns:
            图像绝对路径列表，顺序与 camera_names 一致。
        """
        cameras = camera_names if camera_names is not None else self.camera_names
        sample = self.nusc.get("sample", sample_token)
        paths: List[str] = []
        for cam in cameras:
            cam_token = sample["data"].get(cam)
            if cam_token is None:
                logger.warning("样本 %s 缺少相机 %s 的数据通道，该格将以缺图处理", sample_token, cam)
                paths.append("")
                continue
            cam_data = self.nusc.get("sample_data", cam_token)
            paths.append(str(self.dataroot / cam_data["filename"]))
        return paths

    def get_perception_objects(
        self, sample_token: str, oracle_cfg: Optional[Dict[str, Any]] = None
    ) -> List[dict]:
        """生成 oracle perception_objects（nuScenes GT 投影）。

        委托 oracle_perception 模块：基于 sample_annotation 投影到 6 相机 + 因果运动学。
        所有对象的 is_oracle=True（GT 来源，非模型预测）。

        Args:
            sample_token: 样本标识。
            oracle_cfg: oracle 配置 dict（max_distance_m/box_visibility/cameras）。

        Returns:
            PerceptionObject 序列化后的 dict 列表（便于 jsonl/prompt 直接使用）。
        """
        from src.vla_memory.data import oracle_perception
        ego_pose = self._get_sample_ego_pose(sample_token)
        objs = oracle_perception.get_perception_objects(
            self.nusc, sample_token, ego_pose, oracle_cfg
        )
        return [o.to_serializable_dict() for o in objs]

    def get_ego_pose(self, sample_token: str) -> EgoState:
        """获取指定帧的自车位姿和运动状态。

        P3 起：优先从 CAN bus 真值读取（若 ``self.can_bus_loader`` 可用），
        回退到相邻帧 ego_pose 差分估计。

        Args:
            sample_token: 样本标识。

        Returns:
            EgoState 实例，其中 ``source`` 字段标识数据来源。
        """
        # ---- P3 CAN bus 优先 ----
        if self.can_bus_loader is not None:
            try:
                sample = self.nusc.get("sample", sample_token)
                scene = self.nusc.get("scene", sample["scene_token"])
                scene_name = scene["name"]  # "scene-0061"
                state = self.can_bus_loader.query_at(scene_name, int(sample["timestamp"]))
                # 转成 EgoState
                x, y, z = state.pos
                yaw = EgoState.quat_to_yaw(list(state.orientation_quat))
                vbx, vby, _ = state.vel
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                vx = cos_y * vbx - sin_y * vby
                vy = sin_y * vbx + cos_y * vby
                speed = state.speed_mps if state.speed_mps is not None else math.sqrt(vx * vx + vy * vy)
                abx, aby, _ = state.accel
                ax = cos_y * abx - sin_y * aby
                ay = sin_y * abx + cos_y * aby
                yaw_rate = state.yaw_rate if state.yaw_rate is not None else float(state.rotation_rate[2])
                source = "can_bus" if state.speed_mps is not None else "can_bus_pose_only"
                return EgoState(
                    timestamp=int(state.utime_us),
                    x=x, y=y, z=z, yaw=yaw,
                    vx=vx, vy=vy, speed=max(0.0, speed),
                    ax=ax, ay=ay, acceleration=math.sqrt(ax * ax + ay * ay),
                    yaw_rate=yaw_rate,
                    steering_angle=state.steering,
                    throttle=state.throttle,
                    brake=state.brake,
                    gear=state.gear,
                    source=source,
                )
            except (KeyError, FileNotFoundError) as e:
                # 每 scene 仅 warning 一次，避免刷屏
                if scene_name not in self._can_bus_warned:
                    logger.warning(
                        "CAN bus 查询失败 (scene=%s, token=%s)，回退到差分: %s",
                        scene_name, sample_token, e,
                    )
                    self._can_bus_warned.add(scene_name)
                if not self.fallback_to_pose_diff:
                    raise
                # fall through to pose differencing
        elif not self.fallback_to_pose_diff:
            pass  # 无 CAN bus loader 时始终走差分

        # ---- 旧路径：ego_pose 差分 ----
        current_pose = self._get_sample_ego_pose(sample_token)
        timestamp = current_pose["timestamp"]
        translation = current_pose["translation"]  # [x, y, z]
        rotation = current_pose["rotation"]  # [w, x, y, z]
        yaw = EgoState.quat_to_yaw(rotation)

        x, y, z = translation[0], translation[1], translation[2]

        # 速度、加速度差分估计
        vx, vy, speed = 0.0, 0.0, 0.0
        ax, ay, acceleration = 0.0, 0.0, 0.0

        # 找到前一帧
        sample = self.nusc.get("sample", sample_token)
        prev_token = sample.get("previous", "")
        if prev_token:
            prev_pose = self._get_sample_ego_pose(prev_token)
            dt = (timestamp - prev_pose["timestamp"]) / 1e6  # 微秒转秒
            if dt > 0:
                prev_t = prev_pose["translation"]
                vx = (x - prev_t[0]) / dt
                vy = (y - prev_t[1]) / dt
                speed = math.sqrt(vx * vx + vy * vy)

                # 找到前两帧 -> 估计加速度
                prev_sample = self.nusc.get("sample", prev_token)
                pp_token = prev_sample.get("previous", "")
                if pp_token:
                    pp_pose = self._get_sample_ego_pose(pp_token)
                    dt1 = (prev_pose["timestamp"] - pp_pose["timestamp"]) / 1e6
                    if dt1 > 0:
                        pp_t = pp_pose["translation"]
                        prev_vx = (prev_t[0] - pp_t[0]) / dt1
                        prev_vy = (prev_t[1] - pp_t[1]) / dt1
                        ax = (vx - prev_vx) / dt
                        ay = (vy - prev_vy) / dt
                        acceleration = math.sqrt(ax * ax + ay * ay)

        return EgoState(
            timestamp=timestamp,
            x=x, y=y, z=z,
            yaw=yaw,
            vx=vx, vy=vy, speed=speed,
            ax=ax, ay=ay, acceleration=acceleration,
        )

    def get_history_trajectory(
        self,
        sample_token: str,
        history_seconds: float = 5.0,
    ) -> List[Dict[str, float]]:
        """获取最近 N 秒的历史轨迹（ego-centric 坐标系）。

        坐标系: ego-centric，x 前向，y 左向，单位米。

        Args:
            sample_token: 样本标识。
            history_seconds: 历史时间窗口（秒）。

        Returns:
            ego-centric 坐标系下的历史轨迹点列表。
        """
        # 获取当前帧信息
        current_pose = self._get_sample_ego_pose(sample_token)
        current_ts = current_pose["timestamp"]
        current_t = current_pose["translation"]
        current_yaw = EgoState.quat_to_yaw(current_pose["rotation"])

        cos_yaw = math.cos(-current_yaw)
        sin_yaw = math.sin(-current_yaw)

        trajectory = []
        token = sample_token
        while token:
            sample = self.nusc.get("sample", token)
            pose = self._get_sample_ego_pose(token)
            ts = pose["timestamp"]
            dt = (current_ts - ts) / 1e6  # 微秒转秒

            if dt > history_seconds:
                break

            t = pose["translation"]
            dx = t[0] - current_t[0]
            dy = t[1] - current_t[1]
            # 转换到 ego-centric 坐标系
            ego_x = dx * cos_yaw - dy * sin_yaw
            ego_y = dx * sin_yaw + dy * cos_yaw

            trajectory.append({
                "t": round(-dt, 4),  # 负值表示过去
                "x": round(ego_x, 4),
                "y": round(ego_y, 4),
            })

            token = sample.get("previous", "")

        # 按时间倒序排列（从远到近）
        trajectory.sort(key=lambda p: p["t"])
        return trajectory

    def get_future_ego_trajectory(
        self,
        sample_token: str,
        future_seconds: float = 3.0,
    ) -> List[Dict[str, float]]:
        """获取未来 N 秒的真值轨迹（ego-centric 坐标系）。

        坐标系: ego-centric，x 前向，y 左向，单位米。
        用于评测模块计算 ADE / FDE。

        Args:
            sample_token: 样本标识。
            future_seconds: 未来时间窗口（秒）。

        Returns:
            ego-centric 坐标系下的未来轨迹点列表。
        """
        current_pose = self._get_sample_ego_pose(sample_token)
        current_ts = current_pose["timestamp"]
        current_t = current_pose["translation"]
        current_yaw = EgoState.quat_to_yaw(current_pose["rotation"])

        cos_yaw = math.cos(-current_yaw)
        sin_yaw = math.sin(-current_yaw)

        trajectory = []
        token = sample_token
        while token:
            sample = self.nusc.get("sample", token)
            next_token = sample.get("next", "")
            if not next_token:
                break

            next_pose = self._get_sample_ego_pose(next_token)
            dt = (next_pose["timestamp"] - current_ts) / 1e6

            if dt > future_seconds:
                break

            t = next_pose["translation"]
            dx = t[0] - current_t[0]
            dy = t[1] - current_t[1]
            ego_x = dx * cos_yaw - dy * sin_yaw
            ego_y = dx * sin_yaw + dy * cos_yaw

            trajectory.append({
                "t": round(dt, 4),
                "x": round(ego_x, 4),
                "y": round(ego_y, 4),
            })

            token = next_token

        return trajectory

    # ================================================================
    # 辅助方法
    # ================================================================

    def _get_scene_samples(self, scene_token: str) -> List[dict]:
        """获取指定场景的所有 sample（按时间排序）。

        带缓存，同一场景只遍历一次。
        """
        if scene_token in self._scene_samples_cache:
            return self._scene_samples_cache[scene_token]

        scene = self.nusc.get("scene", scene_token)
        samples = []
        current_token = scene["first_sample_token"]
        while current_token:
            sample = self.nusc.get("sample", current_token)
            samples.append(sample)
            current_token = sample.get("next", "")

        self._scene_samples_cache[scene_token] = samples
        return samples

    def _get_sample_ego_pose(self, sample_token: str) -> dict:
        """获取指定 sample 的 ego_pose（通过 CAM_FRONT 关联）。

        带缓存。
        """
        if sample_token in self._ego_pose_cache:
            return self._ego_pose_cache[sample_token]

        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"].get(self.camera_name)
        if cam_token is None:
            raise ValueError(
                f"样本 {sample_token} 没有摄像头 {self.camera_name} 的数据"
            )
        cam_data = self.nusc.get("sample_data", cam_token)
        pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

        self._ego_pose_cache[sample_token] = pose
        return pose
