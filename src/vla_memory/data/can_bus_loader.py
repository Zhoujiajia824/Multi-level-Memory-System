"""CAN bus 真值加载器
===================
P3 新增：从 `data/nuscenes/raw/can_bus/scene-XXXX_*.json` 加载车辆 CAN bus 真值，
替代旧的 ego_pose 有限差分估算。

涉及两个核心通道：
  - scene-XXXX_pose.json (~50 Hz)
        utime, pos[3], orientation[4 wxyz], vel[3 body], accel[3 body],
        rotation_rate[3 body]
  - scene-XXXX_vehicle_monitor.json (~2 Hz)
        utime, vehicle_speed (km/h), steering (deg), yaw_rate (deg/s),
        throttle (0-100), brake (0-100), gear_position, left/right_signal,
        rear_left/right_rpm

时间戳同 nuScenes `sample.timestamp`：微秒 Unix epoch。
查询逻辑：按 `utime` 排序后二分查找最近邻；超出容差返回 None。
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("can_bus_loader")


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------


@dataclass
class CanBusSample:
    """单帧 CAN bus 真值，已做单位转换（统一为 SI: m, m/s, m/s², rad, rad/s）。"""
    utime_us: int
    pos: Tuple[float, float, float]                        # 全局 m
    orientation_quat: Tuple[float, float, float, float]    # [w, x, y, z]
    vel: Tuple[float, float, float]                        # body-frame m/s
    accel: Tuple[float, float, float]                      # body-frame m/s²
    rotation_rate: Tuple[float, float, float]              # body-frame rad/s
    # ---- vehicle_monitor 字段（已换算） ----
    speed_mps: Optional[float] = None      # m/s
    steering: Optional[float] = None       # rad
    throttle: Optional[float] = None       # 0-1
    brake: Optional[float] = None          # 0-1
    yaw_rate: Optional[float] = None       # rad/s
    gear: Optional[str] = None
    source: str = "can_bus_pose+vehicle_monitor"


@dataclass
class CanBusScene:
    """单 scene 的 CAN bus 缓存（已按 utime 排序）。"""
    scene_name: str
    pose_utimes: List[int] = field(default_factory=list)
    pose_records: List[dict] = field(default_factory=list)
    vm_utimes: List[int] = field(default_factory=list)
    vm_records: List[dict] = field(default_factory=list)


# ----------------------------------------------------------------------
# 加载器
# ----------------------------------------------------------------------


class CanBusLoader:
    """nuScenes CAN bus 真值加载器。

    用法：
        loader = CanBusLoader(can_bus_root="data/nuscenes/raw/can_bus")
        sample = loader.query_at("scene-0061", sample_timestamp_us)

    Args:
        can_bus_root: CAN bus 文件所在目录。
        tolerance_us: 最近邻时间容差（微秒）。pose 流约 50 Hz（20ms 一帧），
            vehicle_monitor 流约 2 Hz（500ms 一帧）。默认 60_000us = 60ms
            足以覆盖 pose；超出容差则对应字段返回 None。
    """

    def __init__(
        self,
        can_bus_root: str | Path,
        tolerance_us: int = 60_000,
    ):
        self.can_bus_root = Path(can_bus_root)
        if not self.can_bus_root.exists():
            raise FileNotFoundError(
                f"CAN bus 根目录不存在: {self.can_bus_root}\n"
                f"请将 nuScenes CAN bus expansion 解压到该目录。"
            )
        self.tolerance_us = tolerance_us
        self._scene_cache: Dict[str, CanBusScene] = {}

    # ------------------------------------------------------------------
    # 加载与缓存
    # ------------------------------------------------------------------

    def load_scene(self, scene_name: str) -> CanBusScene:
        """加载并缓存指定 scene 的 pose + vehicle_monitor 文件。

        Args:
            scene_name: 形如 "scene-0061" 的 nuScenes scene 名称。

        Returns:
            CanBusScene 缓存对象。

        Raises:
            FileNotFoundError: 对应 scene 的 pose.json 缺失。
        """
        if scene_name in self._scene_cache:
            return self._scene_cache[scene_name]

        pose_path = self.can_bus_root / f"{scene_name}_pose.json"
        vm_path = self.can_bus_root / f"{scene_name}_vehicle_monitor.json"

        if not pose_path.exists():
            raise FileNotFoundError(
                f"CAN bus pose 文件不存在: {pose_path}"
            )

        with open(pose_path, "r", encoding="utf-8") as f:
            pose_records = json.load(f)

        vm_records: List[dict] = []
        if vm_path.exists():
            with open(vm_path, "r", encoding="utf-8") as f:
                vm_records = json.load(f)
        else:
            logger.warning(
                "scene=%s 没有 vehicle_monitor.json，"
                "speed/steering/throttle/brake/yaw_rate 字段将为 None",
                scene_name,
            )

        # 按 utime 排序（防御性；通常已经有序）
        pose_records.sort(key=lambda r: r["utime"])
        vm_records.sort(key=lambda r: r["utime"])

        scene = CanBusScene(
            scene_name=scene_name,
            pose_utimes=[r["utime"] for r in pose_records],
            pose_records=pose_records,
            vm_utimes=[r["utime"] for r in vm_records],
            vm_records=vm_records,
        )
        self._scene_cache[scene_name] = scene
        logger.info(
            "CAN bus 加载: %s pose=%d 条, vehicle_monitor=%d 条",
            scene_name, len(pose_records), len(vm_records),
        )
        return scene

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def query_at(
        self,
        scene_name: str,
        utime_us: int,
        *,
        tolerance_us: Optional[int] = None,
    ) -> CanBusSample:
        """在指定 scene 内按 utime 最近邻查询 CAN bus 真值。

        Args:
            scene_name: scene 名称。
            utime_us: 目标时间戳（微秒，Unix epoch，与 nuScenes
                ``sample.timestamp`` 同源）。
            tolerance_us: 临时覆盖默认 ``self.tolerance_us``。

        Returns:
            CanBusSample。pose 必有；vehicle_monitor 字段超出容差时为 None。

        Raises:
            KeyError: 时间戳超出该 scene 的 pose 流时间范围 +/- tolerance。
        """
        scene = self.load_scene(scene_name)
        tol = tolerance_us if tolerance_us is not None else self.tolerance_us

        # ---- pose 最近邻 ----
        pose = self._nearest_neighbor(
            scene.pose_utimes, scene.pose_records, utime_us, tol,
        )
        if pose is None:
            raise KeyError(
                f"CAN bus pose 在 utime={utime_us} 附近 ±{tol}us "
                f"找不到匹配记录 (scene={scene_name})"
            )

        # ---- vehicle_monitor 最近邻（可缺失） ----
        vm_tol = max(tol, 600_000)  # vm ~2Hz，容差放宽到 600ms
        vm = self._nearest_neighbor(
            scene.vm_utimes, scene.vm_records, utime_us, vm_tol,
        )

        return _to_can_bus_sample(pose, vm)

    @staticmethod
    def _nearest_neighbor(
        sorted_utimes: List[int],
        records: List[dict],
        target_us: int,
        tolerance_us: int,
    ) -> Optional[dict]:
        """在 sorted_utimes 中二分查找最近邻；超出容差返回 None。"""
        if not sorted_utimes:
            return None
        idx = bisect.bisect_left(sorted_utimes, target_us)
        candidates: List[int] = []
        if idx < len(sorted_utimes):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best_idx, best_diff = None, None
        for c in candidates:
            diff = abs(sorted_utimes[c] - target_us)
            if best_diff is None or diff < best_diff:
                best_idx, best_diff = c, diff
        if best_idx is None or best_diff is None:
            return None
        if best_diff > tolerance_us:
            return None
        return records[best_idx]


# ----------------------------------------------------------------------
# 单位转换辅助
# ----------------------------------------------------------------------

_KMH_TO_MPS = 1.0 / 3.6
_DEG_TO_RAD = math.pi / 180.0


def _to_can_bus_sample(pose: dict, vm: Optional[dict]) -> CanBusSample:
    """把原始 pose + vehicle_monitor 记录转成统一 SI 单位的 CanBusSample。"""
    sample = CanBusSample(
        utime_us=int(pose["utime"]),
        pos=tuple(pose["pos"][:3]),
        orientation_quat=tuple(pose["orientation"][:4]),
        vel=tuple(pose["vel"][:3]),
        accel=tuple(pose["accel"][:3]),
        rotation_rate=tuple(pose["rotation_rate"][:3]),
    )
    if vm is None:
        return sample

    # 单位转换：vehicle_monitor 文件统一为度/百分比/km/h
    vs = vm.get("vehicle_speed")
    if vs is not None:
        sample.speed_mps = float(vs) * _KMH_TO_MPS
    st = vm.get("steering")
    if st is not None:
        sample.steering = float(st) * _DEG_TO_RAD
    yr = vm.get("yaw_rate")
    if yr is not None:
        sample.yaw_rate = float(yr) * _DEG_TO_RAD
    th = vm.get("throttle")
    if th is not None:
        sample.throttle = max(0.0, min(1.0, float(th) / 100.0))
    br = vm.get("brake")
    if br is not None:
        sample.brake = max(0.0, min(1.0, float(br) / 100.0))
    gear = vm.get("gear_position")
    if gear is not None:
        sample.gear = str(gear)
    return sample
